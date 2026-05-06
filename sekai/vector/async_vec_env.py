from __future__ import annotations

import asyncio
from typing import Any, Callable, Sequence

import numpy as np

from sekai._types import ActType, ObsType, SeedType
from sekai.core.env import Env
from sekai.utils.seeding import make_seed_sequence
from sekai.vector.result import VecResetResult, VecStepResult
from sekai.vector.vec_env import VecEnv


class AsyncVecEnv(VecEnv[ObsType, ActType]):
    """Vectorised env using asyncio to run N environments concurrently.

    Unlike ProcVecEnv (multiprocessing), this uses asyncio.gather — ideal for
    I/O-bound environments such as live market data feeds, REST-based simulators,
    or any env with async_step/async_reset overrides.

    CPU-bound envs with blocking step() calls will NOT benefit from this;
    use ProcVecEnv instead.
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], Env[ObsType, ActType]]],
    ) -> None:
        self.envs: list[Env[ObsType, ActType]] = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def async_step(self, actions: Any) -> VecStepResult[ObsType]:
        results = await asyncio.gather(
            *(env.async_step(a) for env, a in zip(self.envs, actions))
        )
        return VecStepResult(
            observations=self._stack_obs([r.observation for r in results]),
            rewards=np.array([r.reward for r in results], dtype=np.float64),
            terminated=np.array([r.terminated for r in results], dtype=np.bool_),
            truncated=np.array([r.truncated for r in results], dtype=np.bool_),
            infos=[r.info for r in results],
        )

    async def async_reset(
        self,
        *,
        seed: list[SeedType] | SeedType = None,
        options: list[dict[str, Any]] | dict[str, Any] | None = None,
    ) -> VecResetResult[ObsType]:
        seeds = self._expand_seeds(seed)
        opts = options if isinstance(options, list) else [options] * self.num_envs
        results = await asyncio.gather(
            *(env.async_reset(seed=s, options=o) for env, s, o in zip(self.envs, seeds, opts))
        )
        return VecResetResult(
            observations=self._stack_obs([r.observation for r in results]),
            infos=[r.info for r in results],
        )

    # ------------------------------------------------------------------
    # Sync shims — run the event loop for callers that can't use await
    # ------------------------------------------------------------------

    def step(self, actions: Any) -> VecStepResult[ObsType]:
        return asyncio.get_event_loop().run_until_complete(self.async_step(actions))

    def reset(
        self,
        *,
        seed: list[SeedType] | SeedType = None,
        options: list[dict[str, Any]] | dict[str, Any] | None = None,
    ) -> VecResetResult[ObsType]:
        return asyncio.get_event_loop().run_until_complete(
            self.async_reset(seed=seed, options=options)
        )

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def _stack_obs(self, obs_list: list[ObsType]) -> Any:
        first = obs_list[0]
        if isinstance(first, np.ndarray):
            return np.stack(obs_list)
        if isinstance(first, dict):
            return {key: np.stack([o[key] for o in obs_list]) for key in first}  # type: ignore[index]
        return obs_list

    def _expand_seeds(self, seed: list[SeedType] | SeedType) -> list[SeedType]:
        if isinstance(seed, list):
            return seed
        if seed is None:
            return [None] * self.num_envs
        return make_seed_sequence(int(seed), self.num_envs)  # type: ignore[return-value]
