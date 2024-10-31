#!/usr/bin/env python

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
# Original file: https://github.com/huggingface/lerobot/blob/main/lerobot/common/policies/factory.py

from omegaconf import DictConfig
import torch

from lerobot.common.policies.policy_protocol import Policy
from lerobot.common.utils.utils import get_safe_torch_device
import lerobot.common.policies.factory as lerobot_policy_factory
import lerobot.scripts.train as lerobot_train


def get_policy_and_config_classes(name: str) -> tuple[Policy, object]:
    """Get the policy's class and config class given a name (matching the policy class' `name` attribute)."""

    # Try the policies here first, then the policies in the lerobot repo
    if name == "bcrnn":
        from policies.robomimic_bc.configuration_bc import BCConfig
        from policies.robomimic_bc.modeling_bc import BCPolicy

        return BCPolicy, BCConfig

    else:
        return lerobot_policy_factory.get_policy_and_config_classes(name)


def make_optimizer_and_scheduler(cfg, policy):
    if cfg.policy.name == "bcrnn":
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.training.epoch_schedule,
            gamma=cfg.training.decay_factor,
        )

    else:
        optimizer, lr_scheduler = lerobot_train.make_optimizer_and_scheduler(cfg, policy)

    return optimizer, lr_scheduler


def make_policy(
    hydra_cfg: DictConfig, pretrained_policy_name_or_path: str | None = None, dataset_stats=None
) -> Policy:
    """Make an instance of a policy class.

    Args:
        hydra_cfg: A parsed Hydra configuration (see scripts). If `pretrained_policy_name_or_path` is
            provided, only `hydra_cfg.policy.name` is used while everything else is ignored.
        pretrained_policy_name_or_path: Either the repo ID of a model hosted on the Hub or a path to a
            directory containing weights saved using `Policy.save_pretrained`. Note that providing this
            argument overrides everything in `hydra_cfg.policy` apart from `hydra_cfg.policy.name`.
        dataset_stats: Dataset statistics to use for (un)normalization of inputs/outputs in the policy. Must
            be provided when initializing a new policy, and must not be provided when loading a pretrained
            policy. Therefore, this argument is mutually exclusive with `pretrained_policy_name_or_path`.
    """
    if not (pretrained_policy_name_or_path is None) ^ (dataset_stats is None):
        raise ValueError(
            "Exactly one of `pretrained_policy_name_or_path` and `dataset_stats` must be provided."
        )

    policy_cls, policy_cfg_class = get_policy_and_config_classes(hydra_cfg.policy.name)

    policy_cfg = lerobot_policy_factory._policy_cfg_from_hydra_cfg(policy_cfg_class, hydra_cfg)
    if pretrained_policy_name_or_path is None:
        # Make a fresh policy.
        policy = policy_cls(policy_cfg, dataset_stats)
    else:
        # Load a pretrained policy and override the config if needed (for example, if there are inference-time
        # hyperparameters that we want to vary).
        # TODO(alexander-soare): This hack makes use of huggingface_hub's tooling to load the policy with,
        # pretrained weights which are then loaded into a fresh policy with the desired config. This PR in
        # huggingface_hub should make it possible to avoid the hack:
        # https://github.com/huggingface/huggingface_hub/pull/2274.
        policy = policy_cls(policy_cfg)
        policy.load_state_dict(
            policy_cls.from_pretrained(pretrained_policy_name_or_path).state_dict()
        )

    policy.to(get_safe_torch_device(hydra_cfg.device))

    return policy
