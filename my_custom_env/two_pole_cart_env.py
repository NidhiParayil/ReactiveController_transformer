import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv

class TwoPoleCartEnv(MujocoEnv):
    def __init__(self, render_mode=None):
        xml_path = os.path.join(os.path.dirname(__file__), "cartpole_two_poles.xml")

        # Call MujocoEnv constructor
        super().__init__(
            model_path=xml_path,
            frame_skip=5,
            observation_space=None,  # placeholder, will fix after super().__init__()
            render_mode=render_mode
        )

        # Now we can access model.nq and model.nv
        observation_size = self.model.nq + self.model.nv
        obs_low = -np.inf * np.ones(observation_size, dtype=np.float64)
        obs_high = np.inf * np.ones(observation_size, dtype=np.float64)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float64)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        obs = self._get_obs()

        # Simple alive reward
        reward = 1.0

        # Optional termination condition
        done = np.abs(self.data.qpos[0]) > 2.4
        return obs, reward, done, False, {}

    def reset_model(self):
        qpos = self.init_qpos + np.random.uniform(low=-0.01, high=0.01, size=self.model.nq)
        qvel = self.init_qvel + np.random.uniform(low=-0.01, high=0.01, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        # print("Observation shape:", self.data.qpos.shape, self.data.qvel.shape)

        return np.concatenate([self.data.qpos, self.data.qvel])
