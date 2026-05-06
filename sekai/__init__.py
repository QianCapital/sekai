"""sekai — A modern, flexible RL environment framework.

世界 (sekai) is built for complex, high-dimensional environments with a focus
on financial markets. It mirrors gym/gymnasium's API surface while providing:

  - Structured StepResult/ResetResult dataclasses instead of error-prone tuples
  - Async-native envs (async_step / async_reset on every environment)
  - Multi-agent support via MultiAgentEnv
  - Render as a separate Renderer object, not bolted onto Env
  - Typed EpisodeStats rather than untyped info dicts
  - Pluggable namespace loaders in the registry
  - Three VecEnv backends: Sync, Async (asyncio), Proc (multiprocessing)

Quick start:
    import sekai

    class MyEnv(sekai.Env[np.ndarray, np.ndarray]):
        def __init__(self):
            self.observation_space = sekai.Box(-1, 1, shape=(4,))
            self.action_space = sekai.Box(-1, 1, shape=(1,))

        def reset(self, *, seed=None, options=None):
            self._set_rng(seed)
            obs = self.rng.uniform(-1, 1, size=(4,)).astype(np.float32)
            return sekai.ResetResult(observation=obs, info={})

        def step(self, action):
            obs = self.rng.uniform(-1, 1, size=(4,)).astype(np.float32)
            reward = float(-np.sum(np.square(action)))
            return sekai.StepResult(obs, reward, False, False, {})
"""

from sekai._version import __version__

# Core API
from sekai.core.env import Env
from sekai.core.wrapper import ActionWrapper, ObservationWrapper, RewardWrapper, Wrapper
from sekai.core.multi_agent import MultiAgentEnv
from sekai.core.result import MAResetResult, MAStepResult, ResetResult, StepResult

# Spaces
from sekai.spaces.space import Space
from sekai.spaces.box import Box
from sekai.spaces.discrete import Discrete
from sekai.spaces.multi_discrete import MultiDiscrete
from sekai.spaces.multi_binary import MultiBinary
from sekai.spaces.dict import Dict
from sekai.spaces.tuple import Tuple
from sekai.spaces import utils as space_utils

# Registry
from sekai.registry.registry import all_specs, make, register, register_namespace, spec
from sekai.registry.spec import EnvSpec, WrapperSpec

# Vector environments
from sekai.vector.vec_env import VecEnv
from sekai.vector.sync_vec_env import SyncVecEnv
from sekai.vector.async_vec_env import AsyncVecEnv
from sekai.vector.proc_vec_env import ProcVecEnv
from sekai.vector.result import VecResetResult, VecStepResult, stack_infos
from sekai.vector.vec_wrapper import (
    VecActionWrapper,
    VecObservationWrapper,
    VecRewardWrapper,
    VecWrapper,
)

# Statistics
from sekai.stats.tracker import EpisodeStats, StatsTracker
from sekai.stats.collectors import InMemoryTracker, LoggingTracker

# Wrappers (common ones exported at top level for convenience)
from sekai.wrappers.time_limit import TimeLimit
from sekai.wrappers.autoreset import AutoReset
from sekai.wrappers.stats import RecordEpisodeStatistics
from sekai.wrappers.clip_action import ClipAction
from sekai.wrappers.rescale_action import RescaleAction
from sekai.wrappers.flatten_obs import FlattenObservation
from sekai.wrappers.normalize_obs import NormalizeObservation
from sekai.wrappers.normalize_reward import NormalizeReward
from sekai.wrappers.frame_stack import FrameStackObservation
from sekai.wrappers.transform_obs import TransformObservation
from sekai.wrappers.transform_reward import TransformReward
from sekai.wrappers.order_enforcing import OrderEnforcing

# Sub-namespaces
from sekai import wrappers, spaces, vector, rendering, stats, utils, error

# Errors
from sekai.error import (
    AsyncCallPending,
    BackendMismatch,
    EnvAlreadyRegistered,
    EnvNotFound,
    InvalidAction,
    InvalidObservation,
    ResetRequired,
    SekaiError,
    SpaceMismatch,
)

# Utilities
from sekai.utils.seeding import np_random, make_seed_sequence
from sekai.utils.checker import check_env


__all__ = [
    "__version__",
    # Core
    "Env",
    "Wrapper",
    "ObservationWrapper",
    "RewardWrapper",
    "ActionWrapper",
    "MultiAgentEnv",
    "StepResult",
    "ResetResult",
    "MAStepResult",
    "MAResetResult",
    # Spaces
    "Space",
    "Box",
    "Discrete",
    "MultiDiscrete",
    "MultiBinary",
    "Dict",
    "Tuple",
    "space_utils",
    # Registry
    "register",
    "make",
    "spec",
    "all_specs",
    "register_namespace",
    "EnvSpec",
    "WrapperSpec",
    # Vector
    "VecEnv",
    "SyncVecEnv",
    "AsyncVecEnv",
    "ProcVecEnv",
    "VecStepResult",
    "VecResetResult",
    "stack_infos",
    "VecWrapper",
    "VecObservationWrapper",
    "VecRewardWrapper",
    "VecActionWrapper",
    # Stats
    "EpisodeStats",
    "StatsTracker",
    "InMemoryTracker",
    "LoggingTracker",
    # Wrappers
    "TimeLimit",
    "AutoReset",
    "RecordEpisodeStatistics",
    "ClipAction",
    "RescaleAction",
    "FlattenObservation",
    "NormalizeObservation",
    "NormalizeReward",
    "FrameStackObservation",
    "TransformObservation",
    "TransformReward",
    "OrderEnforcing",
    # Sub-namespaces
    "wrappers",
    "spaces",
    "vector",
    "rendering",
    "stats",
    "utils",
    "error",
    # Errors
    "SekaiError",
    "EnvNotFound",
    "EnvAlreadyRegistered",
    "InvalidAction",
    "InvalidObservation",
    "ResetRequired",
    "AsyncCallPending",
    "BackendMismatch",
    "SpaceMismatch",
    # Utils
    "np_random",
    "make_seed_sequence",
    "check_env",
]
