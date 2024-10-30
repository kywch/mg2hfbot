import json
import shutil
from pathlib import Path
from math import ceil
from concurrent.futures import ThreadPoolExecutor

from omegaconf import OmegaConf
import PIL
import h5py
import tqdm

import numpy as np
import torch
import einops
from datasets import Dataset
from safetensors.torch import load_file, safe_open

from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.compute_stats import get_stats_einops_patterns
from lerobot.common.datasets.utils import hf_transform_to_torch
from lerobot.common.datasets.transforms import get_image_transforms
from lerobot.scripts.push_dataset_to_hub import (
    push_meta_data_to_hub,
    push_videos_to_hub,
)


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

    if not dst_dir.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy hf_data, meta_data, and videos directories
    for dir_name in ["hf_data", "meta_data", "videos"]:
        shutil.copytree(src_dir / dir_name, dst_dir / dir_name)

    shutil.copy2(src_dir / "repro_data.pt", dst_dir / "repro_data.pt")


def compute_stats(dataset, batch_size=8, num_workers=8, max_num_samples=None, device="cuda"):
    """Compute mean/std and min/max statistics of all data keys in a Dataset using Welford's algorithm."""
    if max_num_samples is None:
        max_num_samples = len(dataset)

    stats_patterns = get_stats_einops_patterns(dataset, num_workers)

    # Check if CUDA is available if device is set to "cuda"
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"

    # First batch to determine shapes
    temp_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    first_batch = next(iter(temp_loader))

    # Initialize statistics dictionaries
    count = {}  # Track count for each key
    mean = {}  # Running mean
    M2 = {}  # Running sum of squares of differences
    max_vals = {}
    min_vals = {}

    # Initialize trackers for each key
    for key, pattern in stats_patterns.items():
        # Get the shape after reduction
        sample_data = first_batch[key].float()
        reduced_shape = einops.reduce(sample_data, pattern, "sum").shape

        count[key] = torch.zeros(1, device=device).float()
        mean[key] = torch.zeros(reduced_shape, device=device).float()
        M2[key] = torch.zeros(reduced_shape, device=device).float()
        max_vals[key] = torch.full(reduced_shape, -float("inf"), device=device).float()
        min_vals[key] = torch.full(reduced_shape, float("inf"), device=device).float()

    # Create optimized dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,  # No need to shuffle for statistics
        drop_last=False,
        pin_memory=True,  # Faster data transfer to GPU
        persistent_workers=True,  # Keep workers alive between iterations
        prefetch_factor=2,  # Prefetch next batches
    )

    for i, batch in enumerate(
        tqdm.tqdm(dataloader, total=ceil(max_num_samples / batch_size), desc="Computing statistics")
    ):
        this_batch_size = len(batch["index"])

        # Load data to GPU
        for key in stats_patterns:
            batch[key] = batch[key].to(device, non_blocking=True).float()

        for key, pattern in stats_patterns.items():
            # Compute batch stats
            batch_mean = einops.reduce(batch[key], pattern, "mean")
            batch_M2 = einops.reduce((batch[key] - batch_mean) ** 2, pattern, "sum")
            delta = batch_mean - mean[key]

            # Get sample count
            data_shape = batch[key].shape
            if batch[key].ndim == 4:  # Image observations
                sample_count = this_batch_size * data_shape[2] * data_shape[3]
            elif batch[key].ndim in [1, 2]:
                sample_count = this_batch_size
            else:
                raise ValueError(f"Invalid data shape: {data_shape}")

            # Update combined statistics using parallel algorithm
            # Reference: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
            mean[key] = (mean[key] * count[key] + batch_mean * sample_count) / (
                count[key] + sample_count
            )
            M2[key] = (
                M2[key]
                + batch_M2
                + (delta**2) * sample_count * count[key] / (count[key] + sample_count)
            )

            # Update min/max, count
            max_vals[key] = torch.maximum(max_vals[key], einops.reduce(batch[key], pattern, "max"))
            min_vals[key] = torch.minimum(min_vals[key], einops.reduce(batch[key], pattern, "min"))
            count[key] += sample_count

        if i == ceil(max_num_samples / batch_size) - 1:
            break

    # Compute final statistics
    stats = {}
    for key in stats_patterns:
        # Calculate standard deviation from M2
        std = (
            torch.sqrt(M2[key] / count[key])
            if count[key] > 1
            else torch.tensor(0.0, device=device).float()
        )

        stats[key] = {
            "mean": mean[key].cpu(),
            "std": std.cpu(),
            "max": max_vals[key].cpu(),
            "min": min_vals[key].cpu(),
        }

    return stats
