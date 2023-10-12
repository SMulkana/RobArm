from gymnasium.envs.registration import register

register(
    id="Rob1/Rob-v0",
    entry_point="Rob1.envs:SimpleEnv",
    max_episode_steps=300,
)