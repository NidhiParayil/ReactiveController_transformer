import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.mujoco import MujocoEnv


class AgentTargetEnv(MujocoEnv):
    def __init__(self, render_mode=None):
        xml_path = os.path.join(os.path.dirname(__file__), "agent_target_scene.xml")

        # Initialize Mujoco
        super().__init__(
            model_path=xml_path,
            frame_skip=5,
            observation_space=None,
            render_mode=render_mode,
        )

        # Now model is loaded, compute observation space dynamically
        obs_dim = 3 + 3 + 12
        low = -np.inf * np.ones(obs_dim, dtype=np.float64)
        high = np.inf * np.ones(obs_dim, dtype=np.float64)
        self.observation_space = spaces.Box(low, high, dtype=np.float64)
        self.force_sensor_ids =  [0,1,2,3]

        self.init_qpos = self.data.qpos.copy()
        self.init_qvel = self.data.qvel.copy()
        self.target_pos = np.array([0.1, 0.0, 0.])

    def get_force_sensor_array(self):
        m = 0.001

        force_array = []
        for sensor_id in self.force_sensor_ids:
            force =-self.data.sensordata[sensor_id :3+sensor_id ] 
            # print(self.get_ee_acc()*m, -self.data.sensordata[33+i:36+i])
            force_array.append(np.round(force,3))
        force_array = np.asarray(force_array)
        
        return force_array


    def _get_obs(self):
        # print(self.get_force_sensor_array())
        self.interaction_force = self.get_force_sensor_array()
        return np.concatenate([self.data.qpos[0:3], self.data.qvel[0:3],self.interaction_force.flatten()] )

    def reset_model(self):
        qpos = self.init_qpos.copy()
        qvel = self.init_qvel.copy()

        # Only reset the agent's joints (first 3) — leave others unchanged or zero
        qpos[:3] = np.zeros(3)
        qvel[:3] = np.zeros(3)

        self.set_state(qpos, qvel)
        self.interaction_force = self.get_force_sensor_array()
        return self._get_obs()

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        self.render()
        self.data.ctrl[:] = action
        obs = self._get_obs()
        agent_pos = self.data.qpos[:3]
        # print(agent_pos)
        dist = np.linalg.norm(agent_pos - self.target_pos)
        # print(dist)
        force =  np.linalg.norm(self.interaction_force, axis = 0)
        reward = -dist 
        terminated = dist < 0.0005  # success if very close
        truncated = False
        return obs, reward, terminated, truncated, {}
