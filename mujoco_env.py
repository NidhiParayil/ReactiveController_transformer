import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import os
import cv2
from dm_control.mujoco.wrapper import mjbindings

class RT1LikeMuJoCoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, model_path="./assests/arm_env.xml", render_mode=None):
        super().__init__()

        # Load MuJoCo model
        fullpath = os.path.abspath(model_path)
        self.model = mujoco.MjModel.from_xml_path(fullpath)
        self.data = mujoco.MjData(self.model)
        self.eef_site_id = 8
        self.joint_names = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
        self.joint_ids = self.get_joint_ids()
        # Renderer
        self.render_mode = render_mode
        self.viewer = None
        if render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, shape=(224, 224, 3), dtype=np.uint8),
        })

    def get_joint_ids(self):
        return [self.model.jnt(joint).id for joint in self.joint_names]
    
    def get_jacobian(self):
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mjbindings.mjlib.mj_jacSite(self.model, self.data, jacp, jacr, self.eef_site_id)
        jac = np.vstack([jacp, jacr])
        return jac[:, self.joint_ids]

    def resolve_rate_control(self, vel):
        v = np.zeros(6).T
        v[0:3] = vel
        J = self.get_jacobian()
        pnv_j = np.linalg.pinv(J)
        q_curr = self.data.qpos[0:7]
        # print(q_curr,pnv_j, v) 
        q = q_curr + np.matmul(pnv_j, v) * .1
        return q, J


    def step(self, action):
        dq, _ = self.resolve_rate_control(action)
        self.data.ctrl = dq
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = -np.linalg.norm(self._get_agent_pos() - self._get_target_pos())
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Directly set joint positions and velocities
        self.data.qpos[:7] = [0, -.4, 0., .13, 0, -1.1, 0]  # or whatever initial pose you want
        self.data.qvel[:] = 0.0

        mujoco.mj_step(self.model, self.data)

        if self.viewer is not None:
            self.viewer.sync()

        obs = self._get_obs()
        return obs, {}

    def render(self):
        if self.render_mode == "human" and self.viewer is not None:
            self.viewer.sync()

    def _render_rgb_array(self):
        with mujoco.Renderer(self.model, height=224, width=224) as renderer:
            sim = renderer.update_scene(self.data)
            img = renderer.render()
            cam =  renderer.update_scene(self.data, camera = "front_cam")
            img_cam  = renderer.render() 
            cv2.imshow("sim View", cv2.resize(img, (320, 256)))
            cv2.waitKey(1)
        return img_cam

    def _get_obs(self):
        if self.render_mode == "rgb_array":
            return {"image": self._render_rgb_array()}
        else:
            return {"image": np.zeros((224, 224, 3), dtype=np.uint8)}

    def _get_agent_pos(self):
        return self.data.xpos[:3]

    def _get_target_pos(self):
        try:
            target_body_id = self.model.body("target")
            return self.data.xpos[target_body_id]
        except Exception:
            return np.zeros(3)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    import time

    env = RT1LikeMuJoCoEnv(render_mode="human")
    print("Environment initialized. Starting test run...")
    env.reset()
    for _ in range(1000):
        env.render()
        time.sleep(0.05)
    env.close()
    print("Test complete. Environment closed.")
