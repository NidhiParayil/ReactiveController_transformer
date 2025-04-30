import agent_target_scene_env 


import gymnasium as gym
import my_custom_env  # must be imported to register the env
from stable_baselines3 import PPO

env = gym.make("AgentTarget-v0")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save("ppo_agent_target")
