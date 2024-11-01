import os
import argparse
from pathlib import Path
from mimicgen import DATASET_REGISTRY
from mg2hfbot import PREVIOUS_ARTIFACT_FILE
from mg2hfbot.converter import make_lerobot_dataset
from mg2hfbot.utils import download_mimicgen_dataset, copy_work_dir, push_to_hub


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # directory to download datasets to
    parser.add_argument(
        "--download_dir",
        type=str,
        default="mg_download",
        help="Base download directory. Created if it doesn't exist.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="converted",
        help="Base download directory. Created if it doesn't exist.",
    )

    # dataset type to download datasets for
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="source",
        choices=list(DATASET_REGISTRY.keys()),
        help="Dataset type to download datasets for (e.g. source, core, object, robot, large_interpolation). Defaults to source.",
    )

    # single task to download and convert dataset for
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        default="stack",
        help="Task to download datasets for. Defaults to stack task.",
    )

    # ignore the previously-reproduced artifacts
    parser.add_argument(
        "--ignore_previous_artifact",
        action="store_true",
        help="Ignore the previously-reproduced artifacts.",
    )

    parser.add_argument(
        "-s",
        "--success_only",
        action="store_true",
        help="Whether to filter the data to only include successful demos.",
    )

    # limit the number of demos to convert, useful for debugging
    parser.add_argument(
        "-n",
        "--num_demos",
        type=int,
        default=None,
        help="Limit the number of demos to convert. Defaults to None (all demos).",
    )

    parser.add_argument(
        "-p",
        "--push_to_hub",
        action="store_true",
        help="Whether to push the converted dataset to the hub.",
    )

    parser.add_argument(
        "--dataset_repo_prefix",
        type=str,
        default="kywch/mimicgen",
        help="Dataset repo id prefix to push to.",
    )

    args = parser.parse_args()

    # load args
    download_dir = args.download_dir
    download_dataset_type = args.dataset_type
    download_task = args.task
    assert (
        download_task in DATASET_REGISTRY[download_dataset_type]
    ), "got unknown task {} for dataset type {}. Choose one of {}".format(
        download_task, download_dataset_type, list(DATASET_REGISTRY[download_dataset_type].keys())
    )

    # download requested datasets
    dataset_path = download_mimicgen_dataset(download_dir, download_task, download_dataset_type)

    # convert to lerobot
    output_dir = f"{args.output_dir}/{download_task}"

    # for success_only, it should be copied to new directory
    if args.success_only:
        if os.path.exists(output_dir + "/" + PREVIOUS_ARTIFACT_FILE):
            copy_work_dir(output_dir, dst_dir=output_dir + "_so")
        output_dir = output_dir + "_so"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # if there is existing repro data, use it
    previous_artifact = None
    if not args.ignore_previous_artifact and os.path.exists(output_dir / PREVIOUS_ARTIFACT_FILE):
        previous_artifact = output_dir / PREVIOUS_ARTIFACT_FILE

    make_lerobot_dataset(
        download_task,
        dataset_path,
        output_dir,
        num_demos=args.num_demos,
        previous_artifact=previous_artifact,
        success_only=args.success_only,
    )

    if args.push_to_hub:
        repo_id = f"{args.dataset_repo_prefix}_{download_task}"
        if args.success_only:
            repo_id = f"{repo_id}_so"
        push_to_hub(output_dir, repo_id=repo_id)
