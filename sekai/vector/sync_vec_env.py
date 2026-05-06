from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from sekai._types import ActType, ObsType, SeedType
from sekai.core.env import Env
from sekai.utils.seeding import make_seed_sequence
from sekai.vector.result import VecResetResult, VecStepResult
from sekai.vector.vec_env import VecEnv


class SyncVecEnv(VecEnv[ObsType, ActType]):
    """Vectorised env that runs N sub-environments sequentially in a single process.

    Best for fast environments where multiprocessing overhead would dominate.
    Use ProcVecEnv for CPU-bound simulators.
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], Env[ObsType, ActType]]],
        *,
        copy: bool = True,
    ) -> None:
        self.envs: list[Env[ObsType, ActType]] = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self._copy = copy

        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def step(self, actions: Any) -> VecStepResult[ObsType]:
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        obs_list = [r.observation for r in results]
        rewards = np.array([r.reward for r in results], dtype=np.float64)
        terminated = np.array([r.terminated for r in results], dtype=np.bool_)
        truncated = np.array([r.truncated for r in results], dtype=np.bool_)
        infos = [r.info for r in results]
        return VecStepResult(
            observations=self._stack_obs(obs_list),
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
        )

    def reset(
        self,
        *,
        seed: list[SeedType] | SeedType = None,
        options: list[dict[str, Any]] | dict[str, Any] | None = None,
    ) -> VecResetResult[ObsType]:
        seeds = self._expand_seeds(seed)
        opts = options if isinstance(options, list) else [options] * self.num_envs

        results = [
            env.reset(seed=s, options=o)
            for env, s, o in zip(self.envs, seeds, opts)
        ]
        obs_list = [r.observation for r in results]
        infos = [r.info for r in results]
        return VecResetResult(observations=self._stack_obs(obs_list), infos=infos)

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def call(self, method: str, *args: Any, **kwargs: Any) -> list[Any]:
        return [getattr(env, method)(*args, **kwargs) for env in self.envs]

    def get_attr(self, name: str) -> list[Any]:
        return [getattr(env, name) for env in self.envs]

    def set_attr(self, name: str, values: list[Any] | Any) -> None:
        if not isinstance(values, list):
            values = [values] * self.num_envs
        for env, v in zip(self.envs, values):
            setattr(env, name, v)

    def _stack_obs(self, obs_list: list[ObsType]) -> Any:
        first = obs_list[0]
        if isinstance(first, np.ndarray):
            stacked = np.stack(obs_list)
            return stacked.copy() if self._copy else stacked
        if isinstance(first, dict):
            return {key: np.stack([o[key] for o in obs_list]) for key in first}  # type: ignore[index]
        if isinstance(first, tuple):
            return tuple(np.stack([o[i] for o in obs_list]) for i in range(len(first)))  # type: ignore[index]
        return obs_list

    def _expand_seeds(self, seed: list[SeedType] | SeedType) -> list[SeedType]:
        if isinstance(seed, list):
            return seed
        if seed is None:
            return [None] * self.num_envs
        seeds = make_seed_sequence(int(seed), self.num_envs)
        return seeds  # type: ignore[return-value]
