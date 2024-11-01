from pathlib import Path

ENV_CONFIG_DIR = Path(__file__).parent.parent / "configs/env"
ENV_META_DIR = Path(__file__).parent.parent / "env_meta"

# Save the reproduced trajectory to reuse
PREVIOUS_ARTIFACT_FILE = "repro_data.pt"

# 96x96 is preferred in lerobot diffusion policy
IMAGE_OBS_SIZE = (96, 96)

# For saving images, computing stats, etc.
NUM_WORKERS = 4
