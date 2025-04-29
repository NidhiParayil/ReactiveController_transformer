from gymnasium.envs.registration import register

register(
    id="TwoPoleCart-v0",
    entry_point="my_custom_env.two_pole_cart_env:TwoPoleCartEnv",
    max_episode_steps=500,
)