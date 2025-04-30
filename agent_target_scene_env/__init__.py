from gymnasium.envs.registration import register

register(
    id="AgentTarget-v0",
    entry_point="agent_target_scene_env.agent_target_env:AgentTargetEnv",
    max_episode_steps=2000,
)
