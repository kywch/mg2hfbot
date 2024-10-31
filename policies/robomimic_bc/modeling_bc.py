from collections import OrderedDict
from pathlib import Path
import os
from typing import (
    Dict,
    Optional,
    Union,
)

import torch
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from huggingface_hub.constants import PYTORCH_WEIGHTS_NAME
from torch import Tensor, nn

import robomimic.utils.obs_utils as rm_obs_utils
import robomimic.utils.tensor_utils as rm_tensor_utils
import robomimic.models.policy_nets as rm_policy_nets

from policies.robomimic_bc.configuration_bc import BCConfig


class BCPolicy(
    nn.Module,
    PyTorchModelHubMixin,
    tags=["robotics", "bc"],
):
    name = "bcrnn"

    def __init__(
        self,
        config: BCConfig | None = None,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__()
        if config is None:
            config = BCConfig()
        self.config: BCConfig = config
        self.device = self.config.device

        self.obs_key_map = {}
        self._process_input_shape_config()

        # No obs/action normalization for BC

        # BC-RNN
        self.model = rm_policy_nets.RNNGMMActorNetwork(
            obs_shapes=self.config.input_shapes,
            # goal_shapes=self.goal_shapes,
            ac_dim=self.config.output_shapes["action"][0],
            mlp_layer_dims=[],
            # GMM args
            num_modes=self.config.gmm_num_modes,
            min_std=self.config.gmm_min_std,
            std_activation=self.config.gmm_std_activation,
            low_noise_eval=self.config.gmm_low_noise_eval,
            # RNN args
            rnn_hidden_dim=self.config.rnn_hidden_dim,
            rnn_num_layers=self.config.rnn_num_layers,
            rnn_type=self.config.rnn_type,
            rnn_kwargs=self.config.rnn_kwargs,
            # encoder args
            encoder_kwargs=self.config.encoder_config,
        )
        self.model.float().to(self.device)

        self._rnn_horizon = self.config.n_obs_steps
        self._rnn_is_open_loop = self.config.rnn_open_loop
        self._rnn_hidden_state = None
        self._rnn_counter = 0

        self.reset()

    def _process_input_shape_config(self):
        # NOTE: lerobot has . in obs names, but robomimic has issues with that
        # So, internally handle this key mapping
        robomimic_input_shapes = OrderedDict()
        image_keys = []
        for key in self.config.input_shapes:
            lerobot_key = BCConfig.lerobot_key(key)
            robomimic_key = BCConfig.robomimic_key(key)
            self.obs_key_map[robomimic_key] = lerobot_key
            robomimic_input_shapes[robomimic_key] = self.config.input_shapes[key]
            if "image" in key:
                image_keys.append(robomimic_key)

        # Replace input_shapes with new input_shapes, which has new names
        self.config.input_shapes = robomimic_input_shapes

        # robomimic needs the global obs modality mapping
        if rm_obs_utils.OBS_KEYS_TO_MODALITIES is None:
            rm_obs_utils.initialize_obs_modality_mapping_from_dict(
                modality_mapping={"rgb": image_keys}
            )
        else:
            # NOTE: there is also OBS_MODALITIES_TO_KEYS but it's not used, so ignoring it for now.
            for k in image_keys:
                rm_obs_utils.OBS_KEYS_TO_MODALITIES[k] = "rgb"

    def reset(self):
        """To be called whenever the environment is reset."""
        self._rnn_hidden_state = None
        self._rnn_counter = 0

    @torch.no_grad
    def select_action(self, obs_dict: dict[str, Tensor]) -> Tensor:
        """Return one action to run in the environment (potentially in batch mode)."""
        self.eval()

        new_obs_dict = {
            robomimic_key: obs_dict[lerobot_key]
            for robomimic_key, lerobot_key in self.obs_key_map.items()
        }

        if self._rnn_hidden_state is None or self._rnn_counter % self._rnn_horizon == 0:
            batch_size = list(new_obs_dict.values())[0].shape[0]
            self._rnn_hidden_state = self.model.get_rnn_init_state(
                batch_size=batch_size, device=self.device
            )

            if self._rnn_is_open_loop:
                # remember the initial observation, and use it instead of the current observation
                # for open-loop action sequence prediction
                self._open_loop_obs = rm_tensor_utils.clone(rm_tensor_utils.detach(new_obs_dict))

        obs_to_use = new_obs_dict
        if self._rnn_is_open_loop:
            # replace current obs with last recorded obs
            obs_to_use = self._open_loop_obs

        self._rnn_counter += 1
        action, self._rnn_hidden_state = self.model.forward_step(
            obs_to_use, rnn_state=self._rnn_hidden_state
        )  # NOTE: not supporting goal_dict=goal_dict
        return action

    def _process_batch_for_training(self, batch):
        input_batch = {
            "obs": {},
            "goal": batch.get("goal", None),
            "actions": batch["action"],
        }

        for robomimic_key, lerobot_key in self.obs_key_map.items():
            input_batch["obs"][robomimic_key] = batch[lerobot_key]

        return rm_tensor_utils.to_float(rm_tensor_utils.to_device(input_batch, self.device))

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run the batch through the model and compute the loss for training or validation."""

        input_batch = self._process_batch_for_training(batch)
        dists = self.model.forward_train(input_batch["obs"], goal_dict=input_batch["goal"])

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 2  # [B, T]

        log_probs = dists.log_prob(input_batch["actions"])

        return {
            "log_probs": rm_tensor_utils.detach(log_probs),
            # loss is just negative log-likelihood of action targets
            "loss": -log_probs.mean(),
        }

    # NOTE: safetensors causes issues with saving the rnn model (shared tensors)
    # So, replacing it with torch.save, but keeping the name
    def _save_pretrained(self, save_directory: Path) -> None:
        params = {
            "config": self.config,
            "state_dict": self.model.state_dict(),
        }
        torch.save(params, str(save_directory / PYTORCH_WEIGHTS_NAME))

    @classmethod
    def _load_as_pickle(cls, model_file: str, map_location: str, strict: bool):
        params = torch.load(model_file, map_location=torch.device(map_location))
        config = params["config"]
        config.device = map_location

        modality_mapping = {"rgb": [key for key in config.input_shapes if "image" in key]}
        # NOTE: lerobot has . in obs names, but robomimic has issues with that
        modality_mapping["rgb"] += [key.replace(".", "_") for key in modality_mapping["rgb"]]
        rm_obs_utils.initialize_obs_modality_mapping_from_dict(modality_mapping=modality_mapping)

        policy = cls(config)
        policy.model.load_state_dict(params["state_dict"])
        policy.to(map_location)
        policy.eval()
        return policy

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: Optional[bool],
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs,
    ):
        """Load Pytorch pretrained weights and return the loaded model."""
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, PYTORCH_WEIGHTS_NAME)
            return cls._load_as_pickle(model_file, map_location, strict)
        else:
            model_file = hf_hub_download(
                repo_id=model_id,
                filename=PYTORCH_WEIGHTS_NAME,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
            )
            return cls._load_as_pickle(model_file, map_location, strict)
