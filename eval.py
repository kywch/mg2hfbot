# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NOTICE: This file has been modified from the original version.
# Modifications copyright 2024 Kyoung Whan Choe
# Original file: https://github.com/huggingface/lerobot/blob/main/lerobot/scripts/eval.py

import argparse
import logging
import json

from datetime import datetime as dt
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn

from lerobot.common.logger import log_output_dir
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_hydra_config,
    init_logging,
    set_global_seed,
)

from lerobot.scripts.eval import (
    get_pretrained_policy_path,
    eval_policy,
)

from utils import make_dataset_from_local, make_mimicgen_env


def main(
    pretrained_policy_path: Path | None = None,
    hydra_cfg_path: str | None = None,
    out_dir: str | None = None,
    eval_n_episodes: int = 50,
    config_overrides: list[str] | None = None,
):
    assert (pretrained_policy_path is None) ^ (hydra_cfg_path is None)
    if pretrained_policy_path is not None:
        hydra_cfg = init_hydra_config(str(pretrained_policy_path / "config.yaml"), config_overrides)
    else:
        hydra_cfg = init_hydra_config(hydra_cfg_path, config_overrides)

    # override the number of episodes to evaluate the policy on
    hydra_cfg.eval.n_episodes = eval_n_episodes

    if out_dir is None:
        out_dir = f"outputs/eval/{dt.now().strftime('%Y-%m-%d/%H-%M-%S')}_{hydra_cfg.env.name}_{hydra_cfg.policy.name}"

    # Check device is available
    device = get_safe_torch_device(hydra_cfg.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_global_seed(hydra_cfg.seed)

    log_output_dir(out_dir)

    logging.info("Making environment.")
    vec_env = make_mimicgen_env(hydra_cfg)

    logging.info("Making policy.")
    if hydra_cfg_path is None:
        policy = make_policy(
            hydra_cfg=hydra_cfg, pretrained_policy_name_or_path=str(pretrained_policy_path)
        )
    else:
        # Note: We need the dataset stats to pass to the policy's normalization modules.
        policy = make_policy(
            hydra_cfg=hydra_cfg, dataset_stats=make_dataset_from_local(hydra_cfg).stats
        )

    assert isinstance(policy, nn.Module)
    policy.eval()

    with (
        torch.no_grad(),
        torch.autocast(device_type=device.type) if hydra_cfg.use_amp else nullcontext(),
    ):
        info = eval_policy(
            vec_env,
            policy,
            hydra_cfg.eval.n_episodes,
            max_episodes_rendered=10,
            videos_dir=Path(out_dir) / "videos",
            start_seed=hydra_cfg.seed,
        )
    print(info["aggregated"])

    # Save info
    with open(Path(out_dir) / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)

    vec_env.close()

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()

    parser = argparse.ArgumentParser(
        description="Evaluating pretrained lerobot policies", add_help=False
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "-p",
        "--pretrained-policy-name-or-path",
        help=(
            "Either the repo ID of a model hosted on the Hub or a path to a directory containing weights "
            "saved using `Policy.save_pretrained`. If not provided, the policy is initialized from scratch "
            "(useful for debugging). This argument is mutually exclusive with `--config`."
        ),
    )
    group.add_argument(
        "--config",
        help=(
            "Path to a yaml config you want to use for initializing a policy from scratch (useful for "
            "debugging). This argument is mutually exclusive with `--pretrained-policy-name-or-path` (`-p`)."
        ),
    )
    parser.add_argument(
        "-n",
        "--eval-n-episodes",
        type=int,
        default=50,
        help="Number of episodes to evaluate the policy on.",
    )
    parser.add_argument("--revision", help="Optionally provide the Hugging Face Hub revision ID.")
    parser.add_argument(
        "--out-dir",
        help=(
            "Where to save the evaluation outputs. If not provided, outputs are saved in "
            "outputs/eval/{timestamp}_{env_name}_{policy_name}"
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Any key=value arguments to override config values (use dots for.nested=overrides)",
    )
    args = parser.parse_args()

    if args.pretrained_policy_name_or_path is None:
        main(
            hydra_cfg_path=args.config,
            out_dir=args.out_dir,
            eval_n_episodes=args.eval_n_episodes,
            config_overrides=args.overrides,
        )
    else:
        pretrained_policy_path = get_pretrained_policy_path(
            args.pretrained_policy_name_or_path, revision=args.revision
        )

        main(
            pretrained_policy_path=pretrained_policy_path,
            out_dir=args.out_dir,
            eval_n_episodes=args.eval_n_episodes,
            config_overrides=args.overrides,
        )
