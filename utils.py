import json
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

from concurrent.futures import ThreadPoolExecutor
import PIL

import gymnasium as gym
import numpy as np
import torch
from datasets import Dataset
from safetensors.torch import load_file, safe_open

import mimicgen  # noqa
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import hf_transform_to_torch
from lerobot.common.datasets.transforms import get_image_transforms
from lerobot.scripts.push_dataset_to_hub import (
    push_meta_data_to_hub,
    push_videos_to_hub,
)

from env import MimicgenWrapper

ENV_META_DIR = Path("./env_meta")


def load_stats_from_safetensors(filename):
    result = {}
    with safe_open(filename, framework="pt", device="cpu") as f:
        for k in f.keys():
            # NOTE: Assume k is of the form "key/stat"
            # Chech if this is so with the dataset downloaded from the hub
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


def make_mimicgen_env(cfg: DictConfig, n_envs: int | None = None) -> gym.vector.VectorEnv | None:
    """Makes a gym vector environment according to the evaluation config.

    n_envs can be used to override eval.batch_size in the configuration. Must be at least 1.
    """
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

    # use lowres image for eval?
    lowres_image_obs = cfg.eval.get("lowres_image_obs", False)

    def env_creator():
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,  # no on-screen rendering
            render_offscreen=True,  # off-screen rendering to support rendering video frames
            use_image_obs=True,
        )

        # return env
        return MimicgenWrapper(env, cfg.env, env_meta, lowres_image_obs=lowres_image_obs)

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


def save_images_concurrently(
    imgs_array: np.array,
    out_dir: Path,
    lowres_out_dir: Path | None = None,
    lowres_size: tuple[int, int] | None = None,
    max_workers: int = 4,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if lowres_out_dir is not None:
        lowres_out_dir = Path(lowres_out_dir)
        lowres_out_dir.mkdir(parents=True, exist_ok=True)

    def save_image(img_array, i, out_dir):
        img = PIL.Image.fromarray(img_array)
        img.save(str(out_dir / f"frame_{i:06d}.png"), quality=100)

        if lowres_out_dir is not None:
            lowres_img = img.resize(lowres_size)
            lowres_img.save(str(lowres_out_dir / f"frame_{i:06d}.png"), quality=100)

    num_images = len(imgs_array)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        [executor.submit(save_image, imgs_array[i], i, out_dir) for i in range(num_images)]


def push_to_hub(data_dir, repo_id, revision="main"):
    hf_dataset = Dataset.load_from_disk(Path(f"{data_dir}/hf_data"))
    hf_dataset.push_to_hub(repo_id, token=True, revision=revision)

    push_meta_data_to_hub(repo_id, f"{data_dir}/meta_data", revision=revision)
    push_videos_to_hub(repo_id, f"{data_dir}/videos", revision=revision)
