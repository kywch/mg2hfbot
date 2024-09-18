import json
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

import gymnasium as gym
from gymnasium import spaces

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

import numpy as np
import torch
from datasets import Dataset
from safetensors.torch import load_file, safe_open

from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import hf_transform_to_torch
from lerobot.common.datasets.transforms import get_image_transforms


ENV_META_DIR = Path("./env_meta")

# This is for Stack. See if this works for other mimicgen envs
IMAGE_KEYS = ["agentview", "robot0_eye_in_hand"]
STATE_KEYS = [
    "object",
    "robot0_joint_pos",
    "robot0_joint_vel",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_eef_vel_lin",
    "robot0_eef_vel_ang",
    "robot0_gripper_qpos",
    "robot0_gripper_qvel",
]

# Mimicgen requires this to be set globally
ObsUtils.initialize_obs_modality_mapping_from_dict(
    modality_mapping={"rgb": [f"{k}_image" for k in IMAGE_KEYS]}
)


def load_stats_from_safetensors(filename):
    result = {}
    with safe_open(filename, framework="pt", device="cpu") as f:
        for k in f.keys():
            # NOTE: Assume k is of the form "key/stat"
            key = k.split("/")[0]
            if key not in result:
                result[key] = {}

            stat = k.split("/")[1]
            result[key][stat] = f.get_tensor(k)
    return result


def make_dataset_from_local(cfg, root_dir=".", split: str = "train") -> LeRobotDataset:
    resolve_delta_timestamps(cfg)

    image_transforms = None
    if cfg.training.image_transforms.enable:
        cfg_tf = cfg.training.image_transforms
        image_transforms = get_image_transforms(
            brightness_weight=cfg_tf.brightness.weight,
            brightness_min_max=cfg_tf.brightness.min_max,
            contrast_weight=cfg_tf.contrast.weight,
            contrast_min_max=cfg_tf.contrast.min_max,
            saturation_weight=cfg_tf.saturation.weight,
            saturation_min_max=cfg_tf.saturation.min_max,
            hue_weight=cfg_tf.hue.weight,
            hue_min_max=cfg_tf.hue.min_max,
            sharpness_weight=cfg_tf.sharpness.weight,
            sharpness_min_max=cfg_tf.sharpness.min_max,
            max_num_transforms=cfg_tf.max_num_transforms,
            random_order=cfg_tf.random_order,
        )

    hf_dataset = Dataset.load_from_disk(Path(f"{cfg.dataset_repo_id}/hf_data"))
    hf_dataset.set_transform(hf_transform_to_torch)

    info = json.load(open(Path(f"{cfg.dataset_repo_id}/meta_data/info.json")))
    stats = load_stats_from_safetensors(Path(f"{cfg.dataset_repo_id}/meta_data/stats.safetensors"))
    episode_data_index = load_file(
        Path(f"{cfg.dataset_repo_id}/meta_data/episode_data_index.safetensors")
    )

    dataset = LeRobotDataset.from_preloaded(
        transform=image_transforms,
        delta_timestamps=cfg.training.get("delta_timestamps"),
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        stats=stats,
        info=info,
        videos_dir=Path(f"{cfg.dataset_repo_id}/videos/"),
        video_backend=cfg.video_backend,
    )

    if cfg.get("override_dataset_stats"):
        for key, stats_dict in cfg.override_dataset_stats.items():
            for stats_type, listconfig in stats_dict.items():
                # example of stats_type: min, max, mean, std
                stats = OmegaConf.to_container(listconfig, resolve=True)
                dataset.stats[key][stats_type] = torch.tensor(stats, dtype=torch.float32)

    return dataset


# CHECK ME: Is this the right place? Revisit later
class MimicgenWrapper:
    def __init__(self, env, hydra_cfg, env_meta, episode_length=200):
        self.env = env
        self.cfg = hydra_cfg
        self.env_meta = env_meta

        self._max_episode_steps = episode_length
        if "episode_length" in hydra_cfg:
            self._max_episode_steps = hydra_cfg.episode_length
        self.tick = 0

        self.metadata = {
            "render_modes": ["rgb_array"],
            "render_fps": self.env_meta["env_kwargs"]["control_freq"],
        }

        # Process obs into the format that lerobot expects, and hold it
        self.obs = None

        # TODO: consider making this from the config or env_meta
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(
                    {
                        k: spaces.Box(
                            low=0,
                            high=255,
                            shape=(
                                self.env_meta["env_kwargs"]["camera_heights"],
                                self.env_meta["env_kwargs"]["camera_widths"],
                                3,
                            ),
                            dtype=np.uint8,
                        )
                        for k in IMAGE_KEYS
                    }
                ),
                "agent_pos": spaces.Box(
                    low=-1000.0, high=1000.0, shape=(self.cfg.state_dim,), dtype=np.float64
                ),
            }
        )

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.cfg.action_dim,), dtype=np.float32
        )

    def _process_obs(self, obs):
        # TODO: check if IMAGE_KEYS and STATE_KEYS are correct for other cases
        obs_dict = {
            "pixels": {
                k: (np.transpose(obs[f"{k}_image"], (1, 2, 0)) * 255).astype(np.uint8)
                for k in IMAGE_KEYS
            }
        }

        state_obs_list = []
        for k in STATE_KEYS:
            state_obs_list.append(np.array(obs[k]))
        obs_dict.update({"agent_pos": np.concatenate(state_obs_list)})

        self.obs = obs_dict
        return self.obs

    def reset(self, seed=None, options=None):
        self.tick = 0
        # NOTE: EnvRobosuite reset() does NOT take seed
        obs = self.env.reset()
        info = {"is_success": False}
        return self._process_obs(obs), info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        check_success = self.env.is_success()
        done = is_success = check_success["task"]

        info = {"is_success": is_success}

        self.tick += 1
        truncated = False
        if done is False and self.tick >= self._max_episode_steps:
            truncated = True

        return self._process_obs(obs), reward, done, truncated, info

    def render(self):
        return self.obs["pixels"]["agentview"]

    def close(self):
        self.env.env.close()


def make_mimicgen_env(cfg: DictConfig, n_envs: int | None = None) -> gym.vector.VectorEnv | None:
    """Makes a gym vector environment according to the evaluation config.

    n_envs can be used to override eval.batch_size in the configuration. Must be at least 1.
    """
    if n_envs is not None and n_envs < 1:
        raise ValueError("`n_envs must be at least 1")

    if cfg.env.name == "real_world":
        return

    # load the env meta file
    env_meta_file = ENV_META_DIR / cfg.env.meta
    with open(env_meta_file, "r") as f:
        env_meta = json.load(f)

    def env_creator():
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,  # no on-screen rendering
            render_offscreen=True,  # off-screen rendering to support rendering video frames
            use_image_obs=True,
        )
        # return env
        return MimicgenWrapper(env, cfg.env, env_meta)

    # NOTE: mimicgen disables max horizon (see ignore_done). Change in the mimicgen repo if needed.
    # if cfg.env.get("episode_length"):
    #     gym_kwgs["max_episode_steps"] = cfg.env.episode_length

    # NOTE: CANNOT use vecenv for some reason, ignore cfg.eval.batch_size
    return gym.vector.SyncVectorEnv([env_creator])

    # # batched version of the env that returns an observation of shape (b, c)
    # env_cls = gym.vector.AsyncVectorEnv if cfg.eval.use_async_envs else gym.vector.SyncVectorEnv
    # vec_env = env_cls(
    #     [
    #         lambda: env_creator()
    #         for _ in range(n_envs if n_envs is not None else cfg.eval.batch_size)
    #     ]
    # )
    # return vec_env
