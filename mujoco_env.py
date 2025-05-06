import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import os


class RT1LikeMuJoCoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, model_path="./agent_target_scene_env/agent_target_scene.xml", render_mode=None):
        super().__init__()

        # Load MuJoCo model
        fullpath = os.path.abspath(model_path)
        self.model = mujoco.MjModel.from_xml_path(fullpath)
        self.data = mujoco.MjData(self.model)

        # Renderer
        self.render_mode = render_mode
        self.viewer = None
        if render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # Define action and observation spaces
        # Assuming 3 DoF (xyz velocity)
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(3,), dtype=np.float32)

        # Observation: RGB image (224x224) + optionally sensor data
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, shape=(224, 224, 3), dtype=np.uint8),
            # Optional: sensor data or proprioception
        })

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = -np.linalg.norm(self._get_agent_pos() - self._get_target_pos())
        terminated = False  # or add success condition
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        elif self.viewer is not None:
            self.viewer.sync()

    def _render_rgb_array(self):
        # Render RGB image at 224x224 resolution
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        mujoco.mjr_render(
            mujoco.MjvScene(self.model, maxgeom=1000),
            mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150),
            mujoco.MjvCamera(),
            img,
        )
        return img

    def _get_obs(self):
        img = self._render_rgb_array()
        return {"image": img}

    def _get_agent_pos(self):
        return self.data.qpos[:3]  # assuming first 3 qpos are xyz

    def _get_target_pos(self):
        target_body_id = self.model.body("target")
        return self.data.xpos[target_body_id]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
