from dataclasses import dataclass, field


@dataclass
class BCConfig:
    """Configuration class for the Behavior Cloning - RNN policy."""

    device = "cuda"

    # Input / output structure.

    # BC-RNN context length
    n_obs_steps = 10

    input_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "observation.images.agentview": [3, 96, 96],
            "observation.images.robot0_eye_in_hand": [3, 96, 96],
            "observation.state": [9],  # robot0 eef_pos, eef_quat, gripper_qpos
        }
    )

    output_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "action": [7],
        }
    )

    # No obs normalization for BC
    # Normalization / Unnormalization
    input_normalization_modes: dict[str, str] = field(default_factory=lambda: {})
    output_normalization_modes: dict[str, str] = field(default_factory=lambda: {})

    # Architecture
    # GMM
    gmm_num_modes = 5
    gmm_min_std = 0.0001
    gmm_std_activation = "softplus"
    gmm_low_noise_eval = True

    # RNN
    rnn_open_loop = False
    rnn_hidden_dim = 1000
    rnn_num_layers = 2
    rnn_type = "LSTM"
    rnn_kwargs = {"bidirectional": False}

    # Encoder
    encoder_config = {
        "low_dim": {
            "core_class": None,
            "core_kwargs": {},
            "obs_randomizer_class": None,
            "obs_randomizer_kwargs": {},
        },
        "rgb": {
            "core_class": "VisualCore",
            "core_kwargs": {
                "feature_dimension": 64,
                "flatten": True,
                "backbone_class": "ResNet18Conv",
                "backbone_kwargs": {"pretrained": False, "input_coord_conv": False},
                "pool_class": "SpatialSoftmax",
                "pool_kwargs": {
                    "num_kp": 32,
                    "learnable_temperature": False,
                    "temperature": 1.0,
                    "noise_std": 0.0,
                    "output_variance": False,
                },
            },
            "obs_randomizer_class": "CropRandomizer",
            "obs_randomizer_kwargs": {
                "crop_height": 84,
                "crop_width": 84,
                "num_crops": 1,
                "pos_enc": False,
            },
        },
        "depth": {
            "core_class": "VisualCore",
            "core_kwargs": {},
            "obs_randomizer_class": None,
            "obs_randomizer_kwargs": {},
        },
        "scan": {
            "core_class": "ScanCore",
            "core_kwargs": {},
            "obs_randomizer_class": None,
            "obs_randomizer_kwargs": {},
        },
    }

    def __post_init__(self):
        """Input validation (not exhaustive)."""
        lerobot_keys = [self.lerobot_key(k) for k in self.input_shapes]
        if (
            not any(k.startswith("observation.image") for k in lerobot_keys)
            and "observation.environment_state" not in lerobot_keys
        ):
            raise ValueError(
                "You must provide at least one image or the environment state among the inputs."
            )

    @staticmethod
    def lerobot_key(key):
        key = key.replace("observation_", "observation.")
        key = key.replace("image_", "image.")
        key = key.replace("images_", "images.")
        return key

    @staticmethod
    def robomimic_key(key):
        key = key.replace("observation.", "observation_")
        key = key.replace("image.", "image_")
        key = key.replace("images.", "images_")
        return key
