from collections import deque
from pathlib import Path
import json

from omegaconf import DictConfig
import gymnasium as gym
import numpy as np

import mimicgen  # noqa
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


ENV_META_DIR = Path("./env_meta")

IMAGE_OBS_SIZE = (96, 96)
HIGHRES_IMAGE_OBS_SIZE = (256, 256)


def make_mimicgen_env(
    cfg: DictConfig, eval_init_states: list[dict] | None = None, n_envs: int | None = None
) -> gym.vector.VectorEnv | None:
    """Makes a gym vector environment according to the evaluation config.

    n_envs can be used to override eval.batch_size in the configuration. Must be at least 1.
    """

    # NOTE: ignoring n_envs for now
    if n_envs is not None and n_envs < 1:
        raise ValueError("`n_envs must be at least 1")

    if cfg.env.name == "real_world":
        return

    # Mimicgen requires this to be set globally
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        modality_mapping={"rgb": [f"{k}_image" for k in cfg.env.image_keys]}
    )

    # load the env meta file
    env_meta_file = ENV_META_DIR / cfg.env.meta
    with open(env_meta_file, "r") as f:
        env_meta = json.load(f)

    # Adjust action space and image obs space
    env_meta["env_kwargs"]["controller_configs"]["control_delta"] = cfg.env.use_delta_action

    # NOTE: mimicgen envs must have agentview image obs
    image_obs_shape = cfg.policy.input_shapes["observation.images.agentview"]
    env_meta["env_kwargs"]["camera_heights"] = image_obs_shape[1]
    env_meta["env_kwargs"]["camera_widths"] = image_obs_shape[2]

    def env_creator():
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,  # no on-screen rendering
            render_offscreen=True,  # off-screen rendering to support rendering video frames
            use_image_obs=True,
        )

        # return env
        return MimicgenWrapper(
            env,
            cfg.env,
            env_meta,
            eval_init_states=eval_init_states,
        )

    # NOTE: mimicgen disables max horizon (see ignore_done). Change in the mimicgen repo if needed.
    # if cfg.env.get("episode_length"):
    #     gym_kwgs["max_episode_steps"] = cfg.env.episode_length

    # NOTE: CANNOT use vecenv for some reason, ignore cfg.eval.batch_size
    # I guess this is OK since I am not using the vecenv for training
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


class MimicgenWrapper:
    def __init__(
        self,
        env,
        env_cfg,
        env_meta,
        episode_length=200,
        # the number of frames to be in the success state to count as success
        success_criteria=1,
        eval_init_states=None,
    ):
        self.env = env
        self.cfg = env_cfg
        self.env_meta = env_meta
        self.image_keys = env_cfg.image_keys
        self.state_keys = env_cfg.state_keys
        self.success_history = deque(maxlen=success_criteria)

        self.eval_init_states = eval_init_states
        self.eval_idx = None if eval_init_states is None else 0

        self._max_episode_steps = episode_length
        if "episode_length" in env_cfg:
            self._max_episode_steps = env_cfg.episode_length
        self.tick = 0

        self.metadata = {
            "render_modes": ["rgb_array"],
            "render_fps": self.env_meta["env_kwargs"]["control_freq"],
        }

        # Process obs into the format that lerobot expects, and hold it
        self.obs = None

        self.eef_pos = None
        self.eef_quat = None

        # TODO: consider making this from the config or env_meta
        pixel_obs_space = {}
        for k in self.image_keys:
            pixel_obs_space[k] = gym.spaces.Box(
                low=0,
                high=255,
                shape=(
                    self.env_meta["env_kwargs"]["camera_heights"],
                    self.env_meta["env_kwargs"]["camera_widths"],
                    3,
                ),
                dtype=np.uint8,
            )

        self.observation_space = gym.spaces.Dict(
            {
                "pixels": gym.spaces.Dict(pixel_obs_space),
                "agent_pos": gym.spaces.Box(
                    low=-1000.0, high=1000.0, shape=(self.cfg.state_dim,), dtype=np.float64
                ),
            }
        )

        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.cfg.action_dim,), dtype=np.float32
        )

    def _process_obs(self, obs):
        obs_dict = {"pixels": {}}
        for k in self.image_keys:
            img_array = (np.transpose(obs[f"{k}_image"], (1, 2, 0)) * 255).astype(np.uint8)
            obs_dict["pixels"][k] = img_array

        state_obs_list = []
        for k in self.state_keys:
            obs_array = np.array(obs[k])
            if obs_array.ndim == 0:
                obs_array = np.expand_dims(obs_array, axis=0)

            assert obs_array.ndim == 1, "State observations must be 1-dimensional"
            state_obs_list.append(obs_array)

        obs_dict.update({"agent_pos": np.concatenate(state_obs_list)})

        # Keep eef pos and quat
        self.eef_pos = obs["robot0_eef_pos"]
        self.eef_quat = obs["robot0_eef_quat"]

        self.obs = obs_dict
        return self.obs

    def _reset_and_get_info(self):
        self.tick = 0
        self.success_history.clear()
        self.success_history.append(0)  # add a dummy element to the history
        return {"is_success": False}  # dummy info

    def reset(self, seed=None, options=None):
        # NOTE: EnvRobosuite reset() does NOT take seed
        info = self._reset_and_get_info()
        if self.eval_init_states is not None:
            obs = self.env.reset_to(self.eval_init_states[self.eval_idx])
            self.eval_idx += 1
            self.eval_idx = self.eval_idx % len(self.eval_init_states)
        else:
            obs = self.env.reset()

        return self._process_obs(obs), info

    def reset_to(self, initial_state_dict):
        info = self._reset_and_get_info()
        obs = self.env.reset_to(initial_state_dict)
        return self._process_obs(obs), info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # NOTE: This checks if the current tick state is a success
        check_success = self.env.is_success()
        self.success_history.append(check_success["task"])

        # The task is done when the success history is full of 1s
        # NOTE: It's possible that the robot succeeded at the very end, so the history may not be full of 1s
        # Then, it will be considered as a failure.
        done = episode_success = self.is_success()
        info = {"is_success": episode_success}

        self.tick += 1
        truncated = False
        if not done and self.tick > self._max_episode_steps:
            truncated = True

        return self._process_obs(obs), reward, done, truncated, info

    def is_success(self):
        return sum(self.success_history) == len(self.success_history)

    def render(self):
        return self.obs["pixels"]["agentview"]

    def close(self):
        self.env.env.close()
