import agent_target_scene_env 

from stable_baselines3 import PPO
import gymnasium as gym
import my_custom_env
import time

env = gym.make("AgentTarget-v0", render_mode="human")
model = PPO.load("ppo_agent_target")

obs, _ = env.reset()
for _ in range(50000):
    action, _ = model.predict(obs, deterministic=True)
    # print(action)
    obs, reward, done, truncated, _ = env.step(action)

    time.sleep(0.001)
    if done or truncated:
        obs, _ = env.reset()
env.close()
