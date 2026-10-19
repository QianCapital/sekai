from __future__ import annotations

import abc
from collections.abc import Iterator
from typing import Any, Generic

import numpy as np

from sekai._types import ActType, ObsType, SeedType, WActType, WObsType
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.spaces.space import Space


class Wrapper(Env[WObsType, WActType], Generic[WObsType, WActType, ObsType, ActType]):
    """Transparent wrapper around an Env.

    All method calls are delegated to the wrapped env by default.
    Override only the methods you need to change.

    Improvements over gymnasium:
      - `__iter__` / `__len__` let you inspect the full wrapper stack
      - Async methods are transparently delegated
      - _observation_space / _action_space overrides are lazy

    If you override step() but not async_step(), async_step() calls your
    step(), so your logic is never bypassed. Override both to keep the inner
    env's async path end-to-end.
    """

    def __init__(self, env: Env[ObsType, ActType]) -> None:
        if not isinstance(env, Env):
            raise TypeError(
                f"{type(self).__name__} expected a sekai.Env, got {type(env).__name__}. "
                "To wrap a gymnasium env, convert it first with "
                "sekai.compat.from_gymnasium(env)."
            )
        self.env = env
        self._observation_space: Space[WObsType] | None = None
        self._action_space: Space[WActType] | None = None

    # ------------------------------------------------------------------
    # Space delegation with lazy override support
    # ------------------------------------------------------------------

    @property
    def observation_space(self) -> Space[WObsType]:
        if self._observation_space is not None:
            return self._observation_space
        return self.env.observation_space  # type: ignore[return-value]

    @observation_space.setter
    def observation_space(self, space: Space[WObsType]) -> None:
        self._observation_space = space

    @property
    def action_space(self) -> Space[WActType]:
        if self._action_space is not None:
            return self._action_space
        return self.env.action_space  # type: ignore[return-value]

    @action_space.setter
    def action_space(self, space: Space[WActType]) -> None:
        self._action_space = space

    # ------------------------------------------------------------------
    # Sync delegation
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[WObsType]:
        return self.env.reset(seed=seed, options=options)  # type: ignore[return-value]

    def step(self, action: WActType) -> StepResult[WObsType]:
        return self.env.step(action)  # type: ignore[arg-type, return-value]

    # ------------------------------------------------------------------
    # Async delegation
    # ------------------------------------------------------------------

    async def async_reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[WObsType]:
        return await self.env.async_reset(seed=seed, options=options)  # type: ignore[return-value]

    async def async_step(self, action: WActType) -> StepResult[WObsType]:
        return await self.env.async_step(action)  # type: ignore[arg-type, return-value]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.env.close()

    # ------------------------------------------------------------------
    # RNG — wrappers share the wrapped env's generator
    # ------------------------------------------------------------------

    @property
    def rng(self) -> np.random.Generator:
        return self.env.rng

    @rng.setter
    def rng(self, generator: np.random.Generator) -> None:
        self.env.rng = generator

    @property
    def _rng_seed(self) -> int | None:
        return self.env._rng_seed

    @_rng_seed.setter
    def _rng_seed(self, value: int | None) -> None:
        self.env._rng_seed = value

    def _set_rng(self, seed: SeedType) -> int:
        return self.env._set_rng(seed)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def unwrapped(self) -> Env[Any, Any]:
        return self.env.unwrapped

    def has_attr(self, name: str) -> bool:
        return hasattr(self, name) or self.env.has_attr(name)

    def get_attr(self, name: str) -> Any:
        if hasattr(self, name):
            return getattr(self, name)
        return self.env.get_attr(name)

    def set_attr(self, name: str, value: Any) -> None:
        """Set `name` on the outermost layer that has it, or on this wrapper if none does.

        So `env.set_attr("x_threshold", 3.0)` reaches the base env through any
        number of wrappers instead of creating a new attribute on the wrapper.
        """
        for layer in self:
            if hasattr(layer, name):
                setattr(layer, name, value)
                return
        setattr(self, name, value)

    @property
    def spec(self) -> Any:
        return self.env.spec

    @spec.setter
    def spec(self, value: Any) -> None:
        self.unwrapped.spec = value

    @property
    def metadata(self) -> dict[str, Any]:  # type: ignore[override]
        return self.env.metadata

    # ------------------------------------------------------------------
    # Stack introspection
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Env[Any, Any]]:
        """Iterate from outermost wrapper to base env."""
        env: Env[Any, Any] = self
        while isinstance(env, Wrapper):
            yield env
            env = env.env
        yield env

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __str__(self) -> str:
        return f"<{type(self).__name__}({self.env})>"

    def __repr__(self) -> str:
        return str(self)


# ---------------------------------------------------------------------------
# Semantic wrapper specialisations
# ---------------------------------------------------------------------------


class ObservationWrapper(
    Wrapper[WObsType, ActType, ObsType, ActType],
    Generic[WObsType, ActType, ObsType],
):
    """Applies a transformation to every observation."""

    @abc.abstractmethod
    def observation(self, obs: ObsType) -> WObsType: ...

    def _wrap_reset(self, result: ResetResult[ObsType]) -> ResetResult[WObsType]:
        return ResetResult(observation=self.observation(result.observation), info=result.info)

    def _wrap_step(self, result: StepResult[ObsType]) -> StepResult[WObsType]:
        return StepResult(
            observation=self.observation(result.observation),
            reward=result.reward,
            terminated=result.terminated,
            truncated=result.truncated,
            info=result.info,
        )

    def reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[WObsType]:
        return self._wrap_reset(self.env.reset(seed=seed, options=options))

    def step(self, action: ActType) -> StepResult[WObsType]:
        return self._wrap_step(self.env.step(action))

    async def async_reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[WObsType]:
        return self._wrap_reset(await self.env.async_reset(seed=seed, options=options))

    async def async_step(self, action: ActType) -> StepResult[WObsType]:
        return self._wrap_step(await self.env.async_step(action))


class RewardWrapper(
    Wrapper[ObsType, ActType, ObsType, ActType],
    Generic[ObsType, ActType],
):
    """Applies a transformation to every reward."""

    @abc.abstractmethod
    def reward(self, reward: float) -> float: ...

    def _wrap_step(self, result: StepResult[ObsType]) -> StepResult[ObsType]:
        return StepResult(
            observation=result.observation,
            reward=self.reward(result.reward),
            terminated=result.terminated,
            truncated=result.truncated,
            info=result.info,
        )

    def step(self, action: ActType) -> StepResult[ObsType]:
        return self._wrap_step(self.env.step(action))

    async def async_step(self, action: ActType) -> StepResult[ObsType]:
        return self._wrap_step(await self.env.async_step(action))


class ActionWrapper(
    Wrapper[ObsType, WActType, ObsType, ActType],
    Generic[ObsType, WActType, ActType],
):
    """Applies a transformation to every action before passing to the env."""

    @abc.abstractmethod
    def action(self, action: WActType) -> ActType: ...

    def reverse_action(self, action: ActType) -> WActType:
        raise NotImplementedError(
            f"{type(self).__name__} does not implement reverse_action(). "
            "Implement it if you need to map env actions back to wrapper actions."
        )

    def step(self, action: WActType) -> StepResult[ObsType]:
        return self.env.step(self.action(action))

    async def async_step(self, action: WActType) -> StepResult[ObsType]:
        return await self.env.async_step(self.action(action))
