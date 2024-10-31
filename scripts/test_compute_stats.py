import time
from math import ceil
import argparse

import numpy as np
import torch
import einops
import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.compute_stats import get_stats_einops_patterns


def worker_init_fn(worker_id):
    # Set unique numpy/torch seeds per worker
    np.random.seed(worker_id)
    torch.manual_seed(worker_id)


def compute_stats_new(dataset, batch_size=8, num_workers=8, max_num_samples=None, device="cuda"):
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
        pin_memory=device == "cuda",
        persistent_workers=True,
        # To handle worker segfaults
        worker_init_fn=worker_init_fn,
        multiprocessing_context="forkserver",
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


def compare_stats(ref_stats, new_stats):
    try:
        for key in ref_stats:
            assert torch.allclose(ref_stats[key]["mean"], new_stats[key]["mean"])
            assert torch.allclose(ref_stats[key]["std"], new_stats[key]["std"])
            assert torch.allclose(ref_stats[key]["max"], new_stats[key]["max"])
            assert torch.allclose(ref_stats[key]["min"], new_stats[key]["min"])
    except:  # noqa
        print(f"Error: stats do not match for {key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-r",
        "--repo_id",
        type=str,
        default="lerobot/pusht",  # 212 files
        help="Repo ID to compute stats for.",
    )

    parser.add_argument(
        "-n",
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers to use for computing stats.",
    )
    args = parser.parse_args()

    repo_id = args.repo_id
    num_workers = args.num_workers
    dataset = LeRobotDataset(repo_id)

    # Default batch size is 8 (dataset shuffle=true)
    time_start = time.time()
    ref_stats = compute_stats(dataset, num_workers=num_workers)
    print(f"Reference implementation. Time taken: {time.time() - time_start:.1f} s")

    # Single pass implementation, with batch size of 8 (dataset shuffle=false), on cpu
    time_start = time.time()
    stats_cpu = compute_stats_new(dataset, num_workers=num_workers, batch_size=8, device="cpu")
    print(
        f"Single-pass implementation with batch size of 8, cpu. Time taken: {time.time() - time_start:.1f} s"
    )
    compare_stats(ref_stats, stats_cpu)

    # Single pass implementation, with large batch size, on cpu
    time_start = time.time()
    stats_cpu = compute_stats_new(dataset, num_workers=num_workers, batch_size=256, device="cpu")
    print(
        f"Single-pass implementation with large batch size, cpu. Time taken: {time.time() - time_start:.1f} s"
    )
    compare_stats(ref_stats, stats_cpu)

    # Single pass implementation, with large batch size, on gpu
    time_start = time.time()
    stats_gpu = compute_stats_new(dataset, num_workers=num_workers, batch_size=256, device="cuda")
    print(
        f"Single-pass implementation with large batch size, gpu. Time taken: {time.time() - time_start:.1f} s"
    )
    compare_stats(ref_stats, stats_gpu)

    # repo: lerobot/pusht, 212 files
    # * default: 45.9 s
    # * single pass, large bs, cpu: 21.3 s
    # * single pass, large bs, gpu: 20.4 s

    # repo: kywch/mimicgen_stack_d0, 4010 files
    # * default: 962 s
    # * new implementation (gpu): 458 s
    # * new implementation (cpu): 502 s
