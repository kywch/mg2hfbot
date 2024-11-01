import os
import json
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from omegaconf import OmegaConf
import PIL
import h5py

import numpy as np
import torch
from datasets import Dataset
from safetensors.torch import load_file, safe_open

from mimicgen import DATASET_REGISTRY, HF_REPO_ID
import mimicgen.utils.file_utils as FileUtils

from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import hf_transform_to_torch
from lerobot.common.datasets.transforms import get_image_transforms
from lerobot.common.datasets.video_utils import encode_video_frames
from lerobot.scripts.push_dataset_to_hub import (
    push_meta_data_to_hub,
    push_videos_to_hub,
)


def download_mimicgen_dataset(base_dir, task, dataset_type="source", overwrite=False):
    download_dir = os.path.abspath(os.path.join(base_dir, dataset_type))
    download_path = os.path.join(download_dir, f"{task}.hdf5")
    print(
        f"\nDownloading dataset:\n    dataset type: {dataset_type}\n    task: {task}\n    download path: {download_path}"
    )

    url = DATASET_REGISTRY[dataset_type][task]["url"]
    # Make sure path exists and create if it doesn't
    os.makedirs(download_dir, exist_ok=True)

    # If the file already exists, skip download
    if overwrite is False and os.path.exists(download_path):
        # TODO: check the checksum?
        print("File already exists, skipping download")
        return download_path

    print("")
    FileUtils.download_file_from_hf(
        repo_id=HF_REPO_ID,
        filename=url,
        download_dir=download_dir,
        check_overwrite=True,
    )
    print("")
    return download_path


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


def make_videos(output_dir, fps):
    # convert images into videos
    image_dirs = list(output_dir.glob("images/*"))
    for image_dir in image_dirs:
        video_file = Path(f"{output_dir}/videos/{image_dir.name}.mp4")
        if os.path.exists(video_file):
            os.remove(video_file)

        # This fn needs ffmpeg to be installed. $ sudo apt install ffmpeg
        encode_video_frames(
            vcodec="libx265",
            imgs_dir=Path(image_dir),
            video_path=Path(video_file),
            fps=fps,
        )


def push_to_hub(data_dir, repo_id, revision="main"):
    hf_dataset = Dataset.load_from_disk(Path(f"{data_dir}/hf_data"))
    hf_dataset.push_to_hub(repo_id, token=True, revision=revision)

    push_meta_data_to_hub(repo_id, f"{data_dir}/meta_data", revision=revision)
    push_videos_to_hub(repo_id, f"{data_dir}/videos", revision=revision)


def save_states_to_hdf5(file_path, initial_states):
    with h5py.File(file_path, "w") as hf:
        for i, init_state in enumerate(initial_states):
            group = hf.create_group(f"data_{i}")
            group.create_dataset("states", data=init_state["states"])
            group.create_dataset(
                "model", data=init_state["model"], dtype=h5py.string_dtype(encoding="utf-8")
            )


def load_states_from_hdf5(file_path):
    if not Path(file_path).exists():
        return None

    with h5py.File(file_path, "r") as hf:
        initial_states = []
        for key in hf.keys():
            initial_states.append(
                {"states": hf[key]["states"][:], "model": hf[key]["model"][()].decode("utf-8")}
            )
        return initial_states


def copy_work_dir(src_dir, dst_dir):
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)

    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory {src_dir} does not exist.")

    # Delete the destination directory if it exists
    if dst_dir.exists():
        shutil.rmtree(dst_dir)

    # Copy hf_data, meta_data, and videos directories
    dst_dir.mkdir(parents=True, exist_ok=True)
    for dir_name in ["hf_data", "meta_data", "videos"]:
        shutil.copytree(src_dir / dir_name, dst_dir / dir_name)

    shutil.copy2(src_dir / "repro_data.pt", dst_dir / "repro_data.pt")
