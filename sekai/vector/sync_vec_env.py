from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from sekai._types import ActType, ObsType, SeedType
from sekai.core.env import Env
from sekai.core.result import StepResult
from sekai.vector.result import VecResetResult, VecStepResult
from sekai.vector.vec_env import (
    VecEnv,
    close_envs,
    env_call,
    env_get_attr,
    env_set_attr,
    merge_autoreset,
    needs_autoreset,
)


class SyncVecEnv(VecEnv[ObsType, ActType]):
    """Vectorised env that runs N sub-environments sequentially in a single process.

    Best for fast environments where multiprocessing overhead would dominate.
    Use ProcVecEnv for CPU-bound simulators.

    Args:
        env_fns: Callables that each build one sub-environment.
        autoreset: Reset sub-envs automatically when their episode ends (see VecEnv).
        copy: Kept for backwards compatibility. Batched observations are always
            freshly allocated arrays, so this has no effect.
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], Env[ObsType, ActType]]],
        *,
        autoreset: bool = True,
        copy: bool = True,
    ) -> None:
        if len(env_fns) == 0:
            raise ValueError("SyncVecEnv needs at least one env_fn.")
        self.envs: list[Env[ObsType, ActType]] = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.autoreset = autoreset
        self._copy = copy

        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def step(self, actions: Any) -> VecStepResult[Any]:
        self._check_open()
        results: list[StepResult[Any]] = []
        for env, action in zip(self.envs, self._split_actions(actions)):
            result = env.step(action)
            if self.autoreset and needs_autoreset(result):
                result = merge_autoreset(result, env.reset())
            results.append(result)
        return self._batch_step(results)

    def reset(
        self,
        *,
        seed: list[SeedType] | SeedType = None,
        options: list[dict[str, Any] | None] | dict[str, Any] | None = None,
    ) -> VecResetResult[Any]:
        self._check_open()
        seeds = self._expand_seeds(seed)
        opts = self._expand_options(options)
        results = [env.reset(seed=s, options=o) for env, s, o in zip(self.envs, seeds, opts)]
        return self._batch_reset(results)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        close_envs(self.envs)

    def call(self, method: str, *args: Any, **kwargs: Any) -> list[Any]:
        return [env_call(env, method, args, kwargs) for env in self.envs]

    def get_attr(self, name: str) -> list[Any]:
        return [env_get_attr(env, name) for env in self.envs]

    def set_attr(self, name: str, values: list[Any] | Any) -> None:
        if not isinstance(values, list):
            values = [values] * self.num_envs
        if len(values) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} values, got {len(values)}.")
        for env, v in zip(self.envs, values):
            env_set_attr(env, name, v)
