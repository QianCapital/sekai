from __future__ import annotations

import abc
from typing import Any, ClassVar, Generic

from sekai._types import ActType, ObsType, SeedType
from sekai.core._async_fallback import install_async_fallbacks
from sekai.core._rng import SeededRNG
from sekai.core.result import ResetResult, StepResult
from sekai.spaces.space import Space


class Env(SeededRNG, abc.ABC, Generic[ObsType, ActType]):
    """Abstract base class for all sekai environments.

    Subclasses must:
      1. Set self.observation_space and self.action_space in `__init__`
      2. Implement reset() and step()

    Unlike gym/gymnasium:
      - step() returns a StepResult dataclass, not a 5-tuple
      - reset() returns a ResetResult dataclass, not a 2-tuple
      - async_step() and async_reset() are available on all envs by default
      - render() is NOT part of this interface; use a Renderer object

    If a subclass overrides step() (or reset()) without also overriding
    async_step() (or async_reset()), the async method is routed through the
    sync one. This keeps `await env.async_step(a)` from silently skipping
    logic that only lives in step(), e.g. in a custom wrapper. (A wrapper that
    only overrides step() therefore makes async calls through it synchronous;
    override async_step() too to keep an async-native env async end to end.)
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        install_async_fallbacks(cls)

    observation_space: Space[ObsType]
    action_space: Space[ActType]

    metadata: ClassVar[dict[str, Any]] = {}
    spec: Any = None  # EnvSpec | None, typed loosely to avoid circular import

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

        Call self._set_rng(seed) first thing so seeding behaves consistently:
        an int seed re-seeds the env's RNG, None continues the current stream.
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
    # Wrapper introspection
    # ------------------------------------------------------------------

    @property
    def unwrapped(self) -> Env[ObsType, ActType]:
        """The base environment underneath any wrappers."""
        return self

    def has_attr(self, name: str) -> bool:
        """Whether this env (or, for a wrapper, any layer of the stack) has `name`."""
        return hasattr(self, name)

    def get_attr(self, name: str) -> Any:
        """Read `name` from the outermost layer of the wrapper stack that has it."""
        return getattr(self, name)

    def set_attr(self, name: str, value: Any) -> None:
        """Set `name` on the outermost layer of the wrapper stack that has it."""
        setattr(self, name, value)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        spec_id = getattr(self.spec, "id", None)
        name = spec_id or type(self).__name__
        return f"<{name}>"

    def __repr__(self) -> str:
        return str(self)
