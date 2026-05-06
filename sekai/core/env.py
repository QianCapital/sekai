from __future__ import annotations

import abc
from typing import Any, ClassVar, Generic

import numpy as np

from sekai._types import ActType, InfoDict, ObsType, SeedType
from sekai.core.result import ResetResult, StepResult
from sekai.spaces.space import Space


class Env(abc.ABC, Generic[ObsType, ActType]):
    """Abstract base class for all sekai environments.

    Subclasses must:
      1. Set self.observation_space and self.action_space in __init__
      2. Implement reset() and step()

    Unlike gym/gymnasium:
      - step() returns a StepResult dataclass, not a 5-tuple
      - reset() returns a ResetResult dataclass, not a 2-tuple
      - async_step() and async_reset() are available on all envs by default
      - render() is NOT part of this interface; use a Renderer object
    """

    observation_space: Space[ObsType]
    action_space: Space[ActType]

    metadata: ClassVar[dict[str, Any]] = {}
    spec: Any = None  # EnvSpec | None, typed loosely to avoid circular import

    _rng: np.random.Generator | None = None
    _rng_seed: int | None = None

    # ------------------------------------------------------------------
    # Core API — must be implemented by subclasses
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[ObsType]:
        """Reset the environment and return the initial observation.

        Always call super()._reset_seed(seed) or self._set_rng(seed) to
        handle seeding consistently.
        """
        ...

    @abc.abstractmethod
    def step(self, action: ActType) -> StepResult[ObsType]:
        """Take a step in the environment."""
        ...

    # ------------------------------------------------------------------
    # Async API — default implementations wrap the sync methods
    # ------------------------------------------------------------------

    async def async_reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[ObsType]:
        return self.reset(seed=seed, options=options)

    async def async_step(self, action: ActType) -> StepResult[ObsType]:
        return self.step(action)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release any resources held by the environment."""

    def __enter__(self) -> Env[ObsType, ActType]:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # RNG management
    # ------------------------------------------------------------------

    @property
    def rng(self) -> np.random.Generator:
        """The environment's RNG, lazily initialised from OS entropy."""
        if self._rng is None:
            from sekai.utils.seeding import np_random
            self._rng, _ = np_random()
        return self._rng

    def _set_rng(self, seed: SeedType) -> int:
        """Seed the environment's RNG. Returns the actual seed used."""
        from sekai.utils.seeding import np_random
        self._rng, actual_seed = np_random(seed)
        self._rng_seed = actual_seed
        return actual_seed

    # ------------------------------------------------------------------
    # Wrapper introspection
    # ------------------------------------------------------------------

    @property
    def unwrapped(self) -> Env[ObsType, ActType]:
        """The base environment underneath any wrappers."""
        return self

    def has_attr(self, name: str) -> bool:
        return hasattr(self, name)

    def get_attr(self, name: str) -> Any:
        return getattr(self, name)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        spec_id = getattr(self.spec, "id", None)
        name = spec_id or type(self).__name__
        return f"<{name}>"

    def __repr__(self) -> str:
        return str(self)
