import gymnasium as gym
import numpy as np

env = gym.make("HalfCheetah-v4", render_mode="human")  # or "Ant-v4", "Hopper-v4", etc.
obs, info = env.reset(seed=42)

for _ in range(100):
    action = env.action_space.sample()  # Replace with RT-1 actions later
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()

env.close()
