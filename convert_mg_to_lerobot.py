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

from mimicgen import DATASET_REGISTRY, HF_REPO_ID
import mimicgen.utils.file_utils as FileUtils

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import robosuite.utils.transform_utils as T

# datasets come with lerobot
from datasets import Dataset, Features, Sequence, Value
from lerobot.common.datasets.compute_stats import compute_stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.push_dataset_to_hub.utils import concatenate_episodes
from lerobot.common.datasets.utils import hf_transform_to_torch
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames
from lerobot.scripts.push_dataset_to_hub import save_meta_data

from utils import ENV_META_DIR, save_images_concurrently, push_to_hub
from env import MimicgenWrapper


NUM_WORKERS = 8
ENV_CONFIG_DIR = Path("./configs/env")

IMAGE_OBS_SIZE = (256, 256)
SMALL_IMAGE_OBS_POSTFIX = "_lowres"
SMALL_IMAGE_OBS_SIZE = (96, 96)


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


def rollout_episode(trans_env, repro_env, initial_state_dict, action_mat, follow_through=30, use_tqdm=False):
    # reset to initial state
    trans_obs = trans_env.reset_to(initial_state_dict)
    repro_obs, _ = repro_env.reset_to(initial_state_dict)
    assert np.array_equal(repro_env.eef_pos, trans_obs["robot0_eef_pos"]), "Repro env and trans env have different eef pos"
    assert np.array_equal(repro_env.eef_quat, trans_obs["robot0_eef_quat"]), "Repro env and trans env have different eef quat"

    # init buffers
    # NOTE: mimicgen wrapper only returns pixels and agent_pos
    image_obs = {k: [] for k in repro_env.image_keys}
    state_obs = []
    new_actions = []
    rewards = []
    successes = []

    # playback actions one by one, and follow through for a few more steps

    # The EE stays in the same place, with open gripper
    final_action = np.zeros((follow_through, repro_env.cfg.action_dim))

    # append the final action to the action_mat
    mod_action = np.vstack([action_mat, final_action])

    def step(action):
        trans_obs, _, _, _ = trans_env.step(action)

        goal_pos = trans_env.env.robots[0].controller.goal_pos
        goal_ori = T.quat2axisangle(T.mat2quat(trans_env.env.robots[0].controller.goal_ori))
        new_action = np.concatenate((goal_pos, goal_ori, (action[-1],)))

        repro_obs, reward, done, trunc, info = repro_env.step(new_action)

        # Check if the new_action produces the same eef pos and quat in the repro env
        # assert np.allclose(repro_env.eef_pos, trans_obs["robot0_eef_pos"], atol=1e-1), "Repro env and trans env have different eef pos"
        # assert np.allclose(repro_env.eef_quat, trans_obs["robot0_eef_quat"], atol=1e-1), "Repro env and trans env have different eef quat"

        new_actions.append(new_action)

        # NOTE: assume that obs to keep are only the agent_pos or the images
        for key in image_obs.keys():
            image_obs[key].append(repro_obs["pixels"][key])
        state_obs.append(repro_obs["agent_pos"])

        rewards.append(reward)

        # CHECK ME: this is the "frame-level" success, which is different from the episode-level success
        # The episode-level success is defined as all of the last 30 frames being successes,
        # meaning a stable success state, like the stacked block staying in place.
        successes.append(repro_env.success_history[-1])

    if use_tqdm:
        for action in tqdm(mod_action):
            step(action)
    else:
        for action in mod_action:
            step(action)

    ep_dict = {}
    ep_len = len(new_actions)
    timestamps = [i / repro_env.env_meta["env_kwargs"]["control_freq"] for i in range(ep_len)]

    ep_dict["observation.state"] = torch.tensor(np.vstack(state_obs))
    ep_dict["action"] = torch.tensor(np.vstack(new_actions))
    ep_dict["next.done"] = torch.zeros(ep_len, dtype=torch.bool)
    ep_dict["next.done"][-1] = True
    ep_dict["next.reward"] = torch.tensor(np.array(rewards))
    ep_dict["next.success"] = torch.tensor(np.array(successes))
    
    # NOTE: assuming every demo is reproduced. Add error handling if not true?
    ep_dict["frame_index"] = torch.arange(ep_len, dtype=torch.int64)  # start from 0
    ep_dict["timestamp"] = torch.tensor(timestamps)

    return ep_dict, image_obs, repro_env.is_success()


# NOTE: Using only single env due to technical difficuties
def make_lerobot_dataset(task, dataset_path, output_dir, img_height=256, img_width=256):
    mg_file = h5py.File(dataset_path, "r")

    # each demonstration is a group under "data"
    demos = list(mg_file["data"].keys())
    print(f"hdf5 file {dataset_path} has {len(demos)} demonstrations")

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
    # Set controller to take absolute actions
    env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
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

    # LeRobot expects actions to be in absolute coordinates
    # This is the env that we'll use to translate delta actions to absolute actions
    trans_env_meta = json.loads(mg_file["data"].attrs["env_args"])
    trans_env = EnvUtils.create_env_from_metadata(
        env_meta=trans_env_meta,
        render=False,  # no on-screen rendering
        render_offscreen=False,  # No need to render anything
        use_image_obs=False,
    )

    repro_env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,  # no on-screen rendering
        render_offscreen=True,  # off-screen rendering to support rendering video frames
        use_image_obs=True,
    )
    repro_env = MimicgenWrapper(repro_env, env_config, env_meta)

    ### produce the lerobot dataset from the env and saved actions
    ep_dicts = []
    episode_data_index = {"from": [], "to": []}
    id_from = 0
    ep_success = {}

    print("Replaying actions...")
    for ep_idx, demo_key in tqdm(enumerate(demos), total=len(demos)):
        # robosuite datasets store the ground-truth simulator states under the "states" key.
        # We will use the first one, alone with the model xml, to reset the environment to
        # the initial configuration before playing back actions.
        init_state = mg_file[f"data/{demo_key}/states"][0]
        model_xml = mg_file[f"data/{demo_key}"].attrs["model_file"]
        initial_state_dict = dict(states=init_state, model=model_xml)
        action_mat = mg_file[f"data/{demo_key}/actions"][:]

        ep_dict, image_obs, is_success = rollout_episode(trans_env, repro_env, initial_state_dict, action_mat)
        ep_len = ep_dict["action"].shape[0]

        # ep_idx = int(demo_key.split("_")[-1])
        ep_dict["episode_index"] = torch.tensor([ep_idx] * ep_len, dtype=torch.int64)
        ep_success[ep_idx] = {
            "demo_key": demo_key,
            "is_success": bool(is_success),
        }

        # Convert the images into full-size and low-res videos
        for img_key in repro_env.image_keys:
            save_images_concurrently(
                image_obs[img_key],
                Path(f"{output_dir}/images/{img_key}_episode_{ep_idx:06d}"),
                lowres_out_dir=Path(f"{output_dir}/images/{img_key}_episode_{ep_idx:06d}_lowres"),
                lowres_size=SMALL_IMAGE_OBS_SIZE,
                max_workers=NUM_WORKERS,
            )
            fname = f"{img_key}_episode_{ep_idx:06d}.mp4"
            fname_lowres = f"{img_key}_episode_{ep_idx:06d}_lowres.mp4"

            # store the reference to the video frame
            ep_dict[f"observation.images.{img_key}"] = []
            ep_dict[f"observation.images.{img_key}_lowres"] = []
            for tstp in ep_dict["timestamp"].tolist():
                ep_dict[f"observation.images.{img_key}"].append(
                    {"path": f"videos/{fname}", "timestamp": tstp}
                )
                ep_dict[f"observation.images.{img_key}_lowres"].append(
                    {"path": f"videos/{fname_lowres}", "timestamp": tstp}
                )

        ep_dicts.append(ep_dict)
        episode_data_index["from"].append(id_from)
        episode_data_index["to"].append(id_from + ep_len)  # CHECK ME: keep_last?
        id_from += ep_len

    # finished collecting data
    trans_env.env.close()
    repro_env.close()

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
    features["next.reward"] = Value(dtype="float32", id=None)
    features["next.success"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)

    # CHECK ME: Add image resolution info here?
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

    # Also save the used config and env_meta files to the meta_data dir, for future reference
    with open(Path(f"{output_dir}/meta_data/config.yaml"), "w") as f:
        yaml.dump(cfg, f)

    with open(Path(f"{output_dir}/meta_data/env_meta.json"), "w") as f:
        json.dump(env_meta, f, indent=2)

    with open(Path(f"{output_dir}/meta_data/ep_success.json"), "w") as f:
        json.dump(ep_success, f, indent=2)

    num_success = sum(1 for ep_success in ep_success.values() if ep_success["is_success"])
    print(f"Finished converting the mimicgen {task} dataset to lerobot dataset. Success rate: {num_success}/{len(ep_success)}")


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
        default="coffee",
        help="Task to download datasets for. Defaults to stack task.",
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
    output_dir = Path(f"{args.output_dir}/{download_task}")
    output_dir.mkdir(parents=True, exist_ok=True)
    make_lerobot_dataset(download_task, dataset_path, output_dir)

    if args.push_to_hub:
        push_to_hub(output_dir, repo_id=f"{args.dataset_repo_prefix}_{download_task}")
