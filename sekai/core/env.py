"""Abstract base class for all sekai environments."""

from __future__ import annotations

import abc
from typing import Any, ClassVar, Generic, TypeVar

import numpy as np

from sekai.spaces.space import Space
from sekai.utils.typing import ActType, ObsType, RenderFrame, ResetResult, StepResult

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")


class Env(abc.ABC, Generic[ObsT, ActT]):
    """Abstract base class for all sekai environments.

    Environments are the central abstraction in sekai.  An environment
    models the world that an agent interacts with.  Its API is intentionally
    similar to OpenAI Gym / Gymnasium so that existing code can be ported
    with minimal effort while gaining sekai's additional flexibility.

    Subclasses **must** implement:
    - :meth:`reset`
    - :meth:`step`

    Subclasses **may** implement:
    - :meth:`render`
    - :meth:`close`

    Class attributes
    ----------------
    metadata:
        A dict of environment metadata.  Common keys include
        ``"render_modes"`` and ``"render_fps"``.
    reward_range:
        The minimum and maximum rewards as a 2-tuple.  Defaults to
        ``(-inf, +inf)``.

    Instance attributes
    -------------------
    observation_space:
        The space from which observations are drawn.
    action_space:
        The space from which actions are drawn.
    np_random:
        The environment's NumPy random number generator.
    """

    # ------------------------------------------------------------------
    # Class-level configuration
    # ------------------------------------------------------------------

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}
    reward_range: ClassVar[tuple[float, float]] = (-float("inf"), float("inf"))

    # ------------------------------------------------------------------
    # Subclass must set these
    # ------------------------------------------------------------------

    observation_space: Space[ObsT]
    action_space: Space[ActT]

    # ------------------------------------------------------------------
    # Internal state
    # ------------------------------------------------------------------

    _np_random: np.random.Generator | None = None
    _np_random_seed: int | None = None

    # ------------------------------------------------------------------
    # Core API — must be implemented by subclasses
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult:
        """Reset the environment and return an initial observation.

        Parameters
        ----------
        seed:
            Optional seed for the environment's RNG.  When provided the
            RNG is re-seeded.
        options:
            Optional environment-specific reset options.

        Returns
        -------
        observation:
            The initial observation.
        info:
            Auxiliary information.
        """

    @abc.abstractmethod
    def step(self, action: ActT) -> StepResult:
        """Run one timestep of the environment's dynamics.

        Parameters
        ----------
        action:
            An action provided by the agent.

        Returns
        -------
        observation:
            Observation of the environment after the action.
        reward:
            Scalar reward for the transition.
        terminated:
            ``True`` if the episode ended because a terminal state was
            reached (e.g. the goal was achieved or a failure occurred).
        truncated:
            ``True`` if the episode was cut short (e.g. time limit hit).
        info:
            Auxiliary diagnostic information.
        """

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def render(self) -> RenderFrame:
        """Render the environment.

        Returns
        -------
        frame:
            A rendered frame (e.g. RGB array) or ``None``.
        """
        return None

    def close(self) -> None:
        """Perform any cleanup (release resources, close windows, etc.)."""

    # ------------------------------------------------------------------
    # Seed / RNG helpers
    # ------------------------------------------------------------------

    @property
    def np_random(self) -> np.random.Generator:
        """The environment's NumPy random number generator.

        Lazily initialised on first access with a random seed.
        """
        if self._np_random is None:
            self._np_random, self._np_random_seed = _make_rng(None)
        return self._np_random

    @np_random.setter
    def np_random(self, rng: np.random.Generator) -> None:
        self._np_random = rng

    def _seed_np_random(self, seed: int | None) -> list[int]:
        """Re-seed *self.np_random* and propagate to spaces."""
        self._np_random, actual = _make_rng(seed)
        self._np_random_seed = actual
        self.observation_space.seed(actual)
        self.action_space.seed(actual + 1)
        return [actual]

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def action_space_sample(self) -> ActType:
        """Sample a random action from :attr:`action_space`."""
        return self.action_space.sample()

    def observation_space_sample(self) -> ObsType:
        """Sample a random observation from :attr:`observation_space`."""
        return self.observation_space.sample()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "Env[ObsT, ActT]":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} observation_space={self.observation_space} action_space={self.action_space}>"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _make_rng(seed: int | None) -> tuple[np.random.Generator, int]:
    ss = np.random.SeedSequence(seed)
    rng = np.random.default_rng(ss)
    actual_seed = int(ss.entropy) if seed is None else seed  # type: ignore[arg-type]
    return rng, actual_seed
