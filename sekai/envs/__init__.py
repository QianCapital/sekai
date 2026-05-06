"""Built-in environment registrations.

This module is imported by ``sekai/__init__.py`` to register all
built-in environments with the global registry.
"""

from sekai.registry import register

register(
    "CartPole-v0",
    "sekai.envs.cartpole:CartPoleEnv",
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    "GridWorld-v0",
    "sekai.envs.gridworld:GridWorldEnv",
    max_episode_steps=100,
    reward_threshold=0.9,
)
