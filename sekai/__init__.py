"""sekai — 世界 — A modern, flexible deep RL environment framework.

Quick start
-----------
>>> import sekai
>>> env = sekai.make("CartPole-v0")
>>> obs, info = env.reset(seed=42)
>>> action = env.action_space.sample()
>>> obs, reward, terminated, truncated, info = env.step(action)
>>> env.close()

Creating custom environments
----------------------------
>>> from sekai import Env
>>> from sekai.spaces import Box, Discrete
>>> import numpy as np
>>>
>>> class MyEnv(Env):
...     def __init__(self):
...         self.observation_space = Box(-1, 1, shape=(4,))
...         self.action_space = Discrete(2)
...
...     def reset(self, *, seed=None, options=None):
...         if seed is not None:
...             self._seed_np_random(seed)
...         obs = self.observation_space.sample()
...         return obs, {}
...
...     def step(self, action):
...         obs = self.observation_space.sample()
...         return obs, 0.0, False, False, {}
"""

from __future__ import annotations

# Import and register built-in environments
import sekai.envs  # noqa: F401

from sekai.core.agent import Agent, RandomAgent
from sekai.core.env import Env
from sekai.registry import EnvSpec, Registry, make, register, registry, spec
from sekai.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple
from sekai.wrappers import (
    ActionWrapper,
    ClipReward,
    ObservationWrapper,
    RecordEpisode,
    RewardWrapper,
    TimeLimit,
    Wrapper,
)

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Core
    "Env",
    "Agent",
    "RandomAgent",
    # Spaces
    "Space",
    "Box",
    "Dict",
    "Discrete",
    "MultiBinary",
    "MultiDiscrete",
    "Tuple",
    # Registry
    "registry",
    "Registry",
    "EnvSpec",
    "register",
    "make",
    "spec",
    # Wrappers
    "Wrapper",
    "ObservationWrapper",
    "ActionWrapper",
    "RewardWrapper",
    "TimeLimit",
    "RecordEpisode",
    "ClipReward",
]
