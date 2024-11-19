import json
import h5py
import yaml
import argparse

import numpy as np

from mimicgen import DATASET_REGISTRY
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from mg2hfbot import ENV_CONFIG_DIR, ENV_META_DIR
from mg2hfbot.utils import download_mimicgen_dataset


def check_env_and_create_config(task, dataset_path):
    mg_file = h5py.File(dataset_path, "r")

    demos = list(mg_file["data"].keys())
    env_meta = json.loads(mg_file["data"].attrs["env_args"])

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,  # no on-screen rendering
        render_offscreen=True,  # off-screen rendering to support rendering video frames
        use_image_obs=True,
    )

    ObsUtils.initialize_obs_modality_mapping_from_dict(
        modality_mapping={"rgb": [f"{k}_image" for k in env_meta["env_kwargs"]["camera_names"]]}
    )

    obs = env.reset()

    state_keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]

    state_obs_list = []
    for k in state_keys:
        obs_array = np.array(obs[k])
        if obs_array.ndim == 0:
            obs_array = np.expand_dims(obs_array, axis=0)

        state_obs_list.append(obs_array)

    state_dim = np.concatenate(state_obs_list).shape[0]
    print("State dim:", state_dim)

    demos = list(mg_file["data"].keys())

    num_actions = []
    for demo_key in demos:
        action_mat = mg_file[f"data/{demo_key}/actions"][:]
        num_actions.append(action_mat.shape[0])

    print("Max num actions:", max(num_actions))

    length = round(max(num_actions) / 50 + 2) * 50
    print("Set episode length to:", length)

    assert env_meta["env_kwargs"]["camera_names"] == ["agentview", "robot0_eye_in_hand"]

    env_name = env_meta["env_name"].lower()
    env_config = {
        "fps": 20,
        "env": {
            "name": "mimicgen",
            "task": env_meta["env_name"],
            "state_dim": state_dim,
            "action_dim": 7,
            "episode_length": length,
            "meta": f"{env_name}_env.json",
            "image_keys": ["agentview", "robot0_eye_in_hand"],
            "state_keys": state_keys,
            # These should match the policy requirements
            "use_delta_action": True,
        },
    }

    demo_key = demos[0]

    init_state = mg_file[f"data/{demo_key}/states"][0]
    model_xml = mg_file[f"data/{demo_key}"].attrs["model_file"]
    initial_state_dict = dict(states=init_state, model=model_xml)
    action_mat = mg_file[f"data/{demo_key}/actions"][:]

    env.reset_to(initial_state_dict)

    env.env.close()

    # save env config to configs/env
    with open(ENV_CONFIG_DIR / f"{task}.yaml", "w") as f:
        f.write("# @package _global_\n")  # necessary for hydra to parse the config
        yaml.dump(env_config, f, sort_keys=False)

    # save env meta to env_meta
    with open(ENV_META_DIR / f"{env_name}_env.json", "w") as f:
        json.dump(env_meta, f, indent=2)


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

    args = parser.parse_args()

    # load args
    download_dir = args.download_dir
    dataset_type = args.dataset_type
    task = args.task
    assert (
        task in DATASET_REGISTRY[dataset_type]
    ), "got unknown task {} for dataset type {}. Choose one of {}".format(
        task, dataset_type, list(DATASET_REGISTRY[dataset_type].keys())
    )

    # download requested datasets
    dataset_path = download_mimicgen_dataset(download_dir, task, dataset_type)

    check_env_and_create_config(task, dataset_path)
