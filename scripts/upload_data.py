import argparse
from pathlib import Path
from datasets import Dataset
from lerobot.scripts.push_dataset_to_hub import push_meta_data_to_hub, push_videos_to_hub


def upload_data(data_dir, repo_id):
    hf_dataset = Dataset.load_from_disk(Path(f"{data_dir}/hf_data"))
    hf_dataset.push_to_hub(repo_id, token=True, revision="main")

    push_meta_data_to_hub(repo_id, f"{data_dir}/meta_data", revision="main")
    push_videos_to_hub(repo_id, f"{data_dir}/videos", revision="main")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t",
        "--task",
        type=str,
        help="Task to upload data for.",
    )

    parser.add_argument(
        "-s",
        "--success_only",
        action="store_true",
        help="Whether to upload data for success-only demos.",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="converted",
        help="Directory to upload data from.",
    )
    parser.add_argument(
        "--repo_prefix",
        type=str,
        default="kywch/mimicgen",
        help="Repo ID to upload data to.",
    )

    args = parser.parse_args()

    if args.task is None:
        raise ValueError("Task must be provided.")

    post_fix = "_so" if args.success_only else ""

    data_dir = f"{args.data_dir}/{args.task}{post_fix}"
    repo_id = f"{args.repo_prefix}_{args.task}{post_fix}"

    upload_data(data_dir, repo_id)
