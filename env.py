from collections import deque
from gymnasium import spaces
import numpy as np
import PIL


class MimicgenWrapper:
    def __init__(self,
                 env,
                 env_cfg,
                 env_meta,
                 episode_length=200,
                 success_criteria=30,
                 lowres_image_obs=False,
                 lowres_image_size=(96, 96)):
        self.env = env
        self.cfg = env_cfg
        self.env_meta = env_meta
        self.image_keys = env_cfg.image_keys
        self.state_keys = env_cfg.state_keys
        self.success_history = deque(maxlen=success_criteria)

        # NOTE: diffusion model uses lowres image for training, so the policy needs the lowres for eval
        self.lowres_image_obs = lowres_image_obs
        self.lowres_size = lowres_image_size

        self._max_episode_steps = episode_length
        if "episode_length" in env_cfg:
            self._max_episode_steps = env_cfg.episode_length
        self.tick = 0

        self.metadata = {
            "render_modes": ["rgb_array"],
            "render_fps": self.env_meta["env_kwargs"]["control_freq"],
        }

        # Process obs into the format that lerobot expects, and hold it
        self.obs = None

        self.eef_pos = None
        self.eef_quat = None

        # TODO: consider making this from the config or env_meta
        pixel_obs_space = {}
        for k in self.image_keys:
            pixel_obs_space[k] = spaces.Box(
                low=0,
                high=255,
                shape=(
                    self.env_meta["env_kwargs"]["camera_heights"],
                    self.env_meta["env_kwargs"]["camera_widths"],
                    3,
                ),
                dtype=np.uint8,
            )

            if self.lowres_image_obs:
                pixel_obs_space[f"{k}_lowres"] = spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.lowres_size[0], self.lowres_size[1], 3),
                    dtype=np.uint8,
                )

        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(pixel_obs_space),
                "agent_pos": spaces.Box(
                    low=-1000.0, high=1000.0, shape=(self.cfg.state_dim,), dtype=np.float64
                ),
            }
        )

        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.cfg.action_dim,), dtype=np.float32
        )

    def _process_obs(self, obs):
        obs_dict = {"pixels": {}}
        for k in self.image_keys:
            img_array = (np.transpose(obs[f"{k}_image"], (1, 2, 0)) * 255).astype(np.uint8)
            obs_dict["pixels"][k] = img_array
            
            if self.lowres_image_obs:
                pil_img = PIL.Image.fromarray(img_array)
                obs_dict["pixels"][f"{k}_lowres"] = np.array(pil_img.resize(self.lowres_size))

        state_obs_list = []
        for k in self.state_keys:
            obs_array = np.array(obs[k])
            if obs_array.ndim == 0:
                obs_array = np.expand_dims(obs_array, axis=0)

            assert obs_array.ndim == 1, "State observations must be 1-dimensional"
            state_obs_list.append(obs_array)

        obs_dict.update({"agent_pos": np.concatenate(state_obs_list)})

        # Keep eef pos and quat
        self.eef_pos = obs["robot0_eef_pos"]
        self.eef_quat = obs["robot0_eef_quat"]

        self.obs = obs_dict
        return self.obs

    def _reset_and_get_info(self):
        self.tick = 0
        self.success_history.clear()
        self.success_history.append(0)  # add a dummy element to the history
        return {"is_success": False}  # dummy info

    def reset(self, seed=None, options=None):
        info = self._reset_and_get_info()
        # NOTE: EnvRobosuite reset() does NOT take seed
        obs = self.env.reset()
        return self._process_obs(obs), info

    def reset_to(self, initial_state_dict):
        info = self._reset_and_get_info()
        obs = self.env.reset_to(initial_state_dict)
        return self._process_obs(obs), info

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # NOTE: This checks if the current tick state is a success
        check_success = self.env.is_success()
        self.success_history.append(check_success["task"])

        # The task is done when the success history is full of 1s
        # NOTE: It's possible that the robot succeeded at the very end, so the history may not be full of 1s
        # Then, it will be considered as a failure.
        done = episode_success = self.is_success()
        info = {"is_success": episode_success}

        self.tick += 1
        truncated = False
        if not done and self.tick > self._max_episode_steps:
            truncated = True

        return self._process_obs(obs), reward, done, truncated, info

    def is_success(self):
        return sum(self.success_history) == len(self.success_history)

    def render(self):
        return self.obs["pixels"]["agentview"]

    def close(self):
        self.env.env.close()
