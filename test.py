import gymnasium as gym
import my_custom_env  # makes sure registration happens
from stable_baselines3 import PPO
import time
# Load the environment
env = gym.make("TwoPoleCart-v0", render_mode="human")

# Load trained model
model = PPO.load("ppo_two_pole_cart")

obs, _ = env.reset()

for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    # For human viewing
    time.sleep(0.01)

    if terminated or truncated:
        obs, _ = env.reset()

env.close()