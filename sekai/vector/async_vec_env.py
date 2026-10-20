from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Sequence
from typing import Any, TypeVar

from sekai._types import ActType, ObsType, SeedType
from sekai.core.env import Env
from sekai.core.result import StepResult
from sekai.error import VecEnvClosed
from sekai.vector.result import VecResetResult, VecStepResult
from sekai.vector.vec_env import (
    VecEnv,
    build_envs,
    close_envs,
    env_call,
    env_get_attr,
    env_set_attr,
    frozen_step,
    merge_autoreset,
    needs_autoreset,
)

_R = TypeVar("_R")


class AsyncVecEnv(VecEnv[ObsType, ActType]):
    """Vectorised env using asyncio to run N environments concurrently.

    Unlike ProcVecEnv (multiprocessing), this uses asyncio.gather — ideal for
    I/O-bound environments such as live market data feeds, REST-based simulators,
    or any env with async_step/async_reset overrides.

    CPU-bound envs with blocking step() calls will NOT benefit from this;
    use ProcVecEnv instead.

    From async code, use `await vec.async_step(...)` / `await vec.async_reset(...)`.
    The sync step()/reset() run those coroutines on a private event loop owned
    by this VecEnv, so they must not be called from inside a running loop.

    If an env_fn raises, the sub-envs already built are closed before the error
    propagates.
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], Env[ObsType, ActType]]],
        *,
        autoreset: bool = True,
    ) -> None:
        if len(env_fns) == 0:
            raise ValueError("AsyncVecEnv needs at least one env_fn.")
        self.envs: list[Env[ObsType, ActType]] = build_envs(env_fns)
        self.num_envs = len(self.envs)
        self.autoreset = autoreset
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self._loop: asyncio.AbstractEventLoop | None = None
        # The step that ended each frozen sub-env's episode (autoreset off; see VecEnv).
        self._finals: list[StepResult[Any] | None] = [None] * self.num_envs

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------

    async def _step_one(self, index: int, action: Any) -> StepResult[Any]:
        final = self._finals[index]
        if final is not None:
            return frozen_step(final)
        env = self.envs[index]
        result = await env.async_step(action)
        if needs_autoreset(result):
            if self.autoreset:
                result = merge_autoreset(result, await env.async_reset())
            else:
                self._finals[index] = result
        return result

    @staticmethod
    async def _gather_all(coros: Iterable[Awaitable[_R]]) -> list[_R]:
        """Await every coroutine to completion, then raise the first error, if any.

        Unlike a bare asyncio.gather, no sub-env is left mid-call when another
        one fails, so a retry after a transient error starts from a clean state.
        """
        results = await asyncio.gather(*coros, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result
        return results  # type: ignore[return-value]

    async def async_step(self, actions: Any) -> VecStepResult[Any]:
        self._check_open()
        results = await self._gather_all(
            self._step_one(i, a) for i, a in enumerate(self._split_actions(actions))
        )
        return self._batch_step(results)

    async def async_reset(
        self,
        *,
        seed: list[SeedType] | SeedType = None,
        options: list[dict[str, Any] | None] | dict[str, Any] | None = None,
    ) -> VecResetResult[Any]:
        self._check_open()
        seeds = self._expand_seeds(seed)
        opts = self._expand_options(options)
        self._finals = [None] * self.num_envs
        results = await self._gather_all(
            env.async_reset(seed=s, options=o) for env, s, o in zip(self.envs, seeds, opts)
        )
        return self._batch_reset(results)

    # ------------------------------------------------------------------
    # Sync shims — run on a private event loop for callers that can't await
    # ------------------------------------------------------------------

    def _run(self, coro: Coroutine[Any, Any, _R]) -> _R:
        if self.closed:
            coro.close()
            raise VecEnvClosed()
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            coro.close()
            raise RuntimeError(
                "AsyncVecEnv.step()/reset() were called from inside a running event loop. "
                "Use `await vec.async_step(...)` / `await vec.async_reset(...)` instead."
            )
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        try:
            return self._loop.run_until_complete(coro)
        except BaseException:
            # E.g. KeyboardInterrupt: don't leave sub-env calls to resume on the next call.
            self._cancel_pending()
            raise

    def _cancel_pending(self) -> None:
        assert self._loop is not None
        pending = asyncio.all_tasks(self._loop)
        for task in pending:
            task.cancel()
        if pending:
            self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

    def step(self, actions: Any) -> VecStepResult[Any]:
        return self._run(self.async_step(actions))

    def reset(
        self,
        *,
        seed: list[SeedType] | SeedType = None,
        options: list[dict[str, Any] | None] | dict[str, Any] | None = None,
    ) -> VecResetResult[Any]:
        return self._run(self.async_reset(seed=seed, options=options))

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        try:
            close_envs(self.envs)
        finally:
            if self._loop is not None:
                self._cancel_pending()
                self._loop.close()
            self._loop = None

    # ------------------------------------------------------------------
    # Sub-env access
    # ------------------------------------------------------------------

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
