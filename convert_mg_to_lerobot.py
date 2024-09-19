# Refer to https://github.com/samzapo/lerobot/blob/record_gym_script/examples/8_record_dataset_from_gym.py
import os
import json
import yaml
import argparse
from argparse import Namespace

from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import h5py

from mimicgen import DATASET_REGISTRY
import mimicgen.utils.file_utils as FileUtils

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


# datasets come with lerobot
from datasets import Dataset, Features, Sequence, Value
from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.utils import (
    concatenate_episodes,
    save_images_concurrently,
)
from lerobot.common.datasets.utils import (
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames
from lerobot.scripts.push_dataset_to_hub import (
    # push_meta_data_to_hub,
    # push_videos_to_hub,
    save_meta_data,
)

from utils import ENV_META_DIR, MimicgenWrapper


NUM_WORKERS = 8
ENV_CONFIG_DIR = Path("./configs/env")


def download_mimicgen_dataset(base_dir, task, dataset_type="source", overwrite=False):
    download_dir = os.path.abspath(os.path.join(base_dir, dataset_type))
    download_path = os.path.join(download_dir, "{}.hdf5".format(task))
    print(
        "\nDownloading dataset:\n    dataset type: {}\n    task: {}\n    download path: {}".format(
            dataset_type, task, download_path
        )
    )
    url = DATASET_REGISTRY[download_dataset_type][task]["url"]
    # Make sure path exists and create if it doesn't
    os.makedirs(download_dir, exist_ok=True)

    # If the file already exists, skip download
    if overwrite is False and os.path.exists(download_path):
        # TODO: check the checksum?
        print("File already exists, skipping download")
        return download_path

    print("")
    FileUtils.download_url_from_gdrive(
        url=url,
        download_dir=download_dir,
        check_overwrite=True,
    )
    print("")
    return download_path


# NOTE: Using only single env due to technical difficuties
def make_lerobot_dataset(task, dataset_path, output_dir, img_height=256, img_width=256):
    mg_file = h5py.File(dataset_path, "r")

    # each demonstration is a group under "data"
    demos = list(mg_file["data"].keys())
    print("hdf5 file {} has {} demonstrations".format(dataset_path, len(demos)))

    ### initialize the env to reproduce the demonstrations

    # load the lerobot env config file, which must be created manually in configs/env
    with open(ENV_CONFIG_DIR / f"{task}.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        env_config = Namespace(**cfg["env"])
        fps = cfg["fps"]

    # Save env_meta to lerobot meta_data, so that train/eval
    env_meta = json.loads(mg_file["data"].attrs["env_args"])
    env_meta["env_kwargs"]["camera_heights"] = img_height
    env_meta["env_kwargs"]["camera_widths"] = img_width
    with open(ENV_META_DIR / f"{task}_env.json", "w") as f:
        json.dump(env_meta, f, indent=2)

    # Sanity checks between env_config and env_meta
    assert (
        fps == env_meta["env_kwargs"]["control_freq"]
    ), "fps in env_config must match control_freq in env_meta"
    assert (
        env_config.image_keys == env_meta["env_kwargs"]["camera_names"]
    ), "image_keys in env_config must match camera_names in env_meta"

    # Mimicgen requires this to be set globally
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        modality_mapping={"rgb": [f"{k}_image" for k in env_config.image_keys]}
    )

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,  # no on-screen rendering
        render_offscreen=True,  # off-screen rendering to support rendering video frames
        use_image_obs=True,
    )
    env = MimicgenWrapper(env, env_config, env_meta)

    ### produce the lerobot dataset from the env and saved actions
    ep_dicts = []
    episode_data_index = {"from": [], "to": []}
    id_from = 0

    print("Replaying actions...")
    for ep_idx, demo_key in tqdm(enumerate(demos)):
        # robosuite datasets store the ground-truth simulator states under the "states" key.
        # We will use the first one, alone with the model xml, to reset the environment to
        # the initial configuration before playing back actions.
        init_state = mg_file["data/{}/states".format(demo_key)][0]
        model_xml = mg_file["data/{}".format(demo_key)].attrs["model_file"]
        initial_state_dict = dict(states=init_state, model=model_xml)

        # reset to initial state
        obs, _ = env.reset_to(initial_state_dict)

        # init buffers
        # NOTE: mimicgen wrapper only returns pixels and agent_pos
        obs_replay = {k: [] for k in env_config.image_keys + ["agent_pos"]}
        action_replay = mg_file["data/{}/actions".format(demo_key)][:]
        timestamps = []

        # playback actions one by one, and render frames
        for tick, action in enumerate(action_replay.tolist()):
            obs, reward, done, trunc, info = env.step(action)

            # NOTE: assume that obs to keep are only the agent_pos or the images
            for key in obs_replay.keys():
                if key in obs:
                    # obs_replay[key].append(deepcopy(obs[key]))
                    obs_replay[key].append(obs[key])
                else:
                    # TODO: error handling?
                    # obs_replay[key].append(deepcopy(obs["pixels"][key]))
                    obs_replay[key].append(obs["pixels"][key])

            timestamps.append(tick * 1 / fps)

        ep_dict = {}
        ep_len = action_replay.shape[0]

        for img_key in env_config.image_keys:
            save_images_concurrently(
                obs_replay[img_key],
                Path(f"{output_dir}/images/{img_key}_episode_{ep_idx:06d}"),
                NUM_WORKERS,
            )
            fname = f"{img_key}_episode_{ep_idx:06d}.mp4"

            # store the reference to the video frame
            ep_dict[f"observation.images.{img_key}"] = [
                {"path": f"videos/{fname}", "timestamp": tstp} for tstp in timestamps
            ]

        ep_dict["observation.state"] = torch.tensor(np.vstack(obs_replay["agent_pos"]))
        ep_dict["action"] = torch.tensor(action_replay)
        ep_dict["next.done"] = torch.zeros(ep_len, dtype=torch.bool)
        ep_dict["next.done"][-1] = True

        # NOTE: assuming every demo is reproduced. Add error handling if not true?
        ep_dict["episode_index"] = torch.tensor([ep_idx] * ep_len, dtype=torch.int64)
        ep_dict["frame_index"] = torch.arange(ep_len, dtype=torch.int64)  # start from 0
        ep_dict["timestamp"] = torch.tensor(timestamps)

        ep_dicts.append(ep_dict)
        episode_data_index["from"].append(id_from)
        episode_data_index["to"].append(id_from + ep_len)  # CHECK ME: keep_last?
        id_from += ep_len

    # finished collecting data
    env.close()

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

    # Prepare the dataset
    data_dict = concatenate_episodes(ep_dicts)

    features = {key: VideoFrame() for key in data_dict if key.startswith("observation.image")}
    features["observation.state"] = Sequence(
        length=data_dict["observation.state"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["action"] = Sequence(
        length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)

    info = {
        "fps": fps,
        "video": 1,
    }

    lerobot_dataset = LeRobotDataset.from_preloaded(
        hf_dataset=hf_dataset,
        episode_data_index=episode_data_index,
        info=info,
        videos_dir=Path(f"{output_dir}/videos"),
    )

    stats = compute_stats(lerobot_dataset, num_workers=NUM_WORKERS)

    hf_dataset = hf_dataset.with_format(None)  # to remove transforms that cant be saved
    hf_dataset.save_to_disk(f"{output_dir}/hf_data/")

    save_meta_data(info, stats, episode_data_index, Path(f"{output_dir}/meta_data/"))

    # Also save the env_meta file to the meta_data dir, for future reference
    with open(Path(f"{output_dir}/meta_data/env_meta.json"), "w") as f:
        json.dump(env_meta, f, indent=2)

    print(f"Finished converting the mimicgen {task} dataset to lerobot dataset.")


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
        "--task",
        type=str,
        default="stack",
        help="Task to download datasets for. Defaults to stack task.",
    )

    # TODO: add push to hub args

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
    output_dir = Path(f"{args.output_dir}/{download_task}")
    output_dir.mkdir(parents=True, exist_ok=True)
    make_lerobot_dataset(download_task, dataset_path, output_dir)
