"""Base wrapper class for sekai environments."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

import numpy as np

from sekai.core.env import Env
from sekai.spaces.space import Space
from sekai.utils.typing import ActType, ObsType, RenderFrame, ResetResult, StepResult

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")
NewObsT = TypeVar("NewObsT")
NewActT = TypeVar("NewActT")


class Wrapper(Env[ObsT, ActT], Generic[ObsT, ActT]):
    """Wraps an :class:`~sekai.core.Env` to allow modular transformations.

    A :class:`Wrapper` forwards all calls to the underlying environment by
    default, so subclasses only need to override the methods they want to
    change.

    Parameters
    ----------
    env:
        The environment to wrap.
    """

    def __init__(self, env: Env[ObsT, ActT]) -> None:
        self.env: Env[ObsT, ActT] = env

    # ------------------------------------------------------------------
    # Transparent property forwarding
    # ------------------------------------------------------------------

    @property  # type: ignore[override]
    def observation_space(self) -> Space[ObsT]:  # type: ignore[override]
        return self.env.observation_space

    @observation_space.setter
    def observation_space(self, space: Space[ObsT]) -> None:
        self.env.observation_space = space  # type: ignore[assignment]

    @property  # type: ignore[override]
    def action_space(self) -> Space[ActT]:  # type: ignore[override]
        return self.env.action_space

    @action_space.setter
    def action_space(self, space: Space[ActT]) -> None:
        self.env.action_space = space  # type: ignore[assignment]

    @property
    def metadata(self) -> dict[str, Any]:  # type: ignore[override]
        return self.env.metadata

    @property
    def reward_range(self) -> tuple[float, float]:  # type: ignore[override]
        return self.env.reward_range  # type: ignore[return-value]

    @property
    def np_random(self) -> np.random.Generator:  # type: ignore[override]
        return self.env.np_random

    @np_random.setter
    def np_random(self, rng: np.random.Generator) -> None:
        self.env.np_random = rng

    # ------------------------------------------------------------------
    # Core API — delegate to wrapped env
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult:
        return self.env.reset(seed=seed, options=options)

    def step(self, action: ActT) -> StepResult:
        return self.env.step(action)

    def render(self) -> RenderFrame:
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    # ------------------------------------------------------------------
    # Unwrapping helpers
    # ------------------------------------------------------------------

    def unwrapped(self) -> Env[ObsT, ActT]:
        """Return the innermost (unwrapped) environment."""
        inner = self.env
        while isinstance(inner, Wrapper):
            inner = inner.env
        return inner

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}({self.env!r})>"


class ObservationWrapper(Wrapper[NewObsT, ActT], Generic[NewObsT, ActT]):  # type: ignore[misc]
    """Wrapper that transforms observations.

    Subclasses must implement :meth:`observation`.
    """

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NewObsT, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info

    def step(self, action: ActT) -> tuple[NewObsT, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.observation(obs), reward, terminated, truncated, info

    def observation(self, observation: ObsType) -> NewObsT:
        """Transform an observation from the wrapped environment.

        Parameters
        ----------
        observation:
            The raw observation.

        Returns
        -------
        NewObsT
            The transformed observation.
        """
        raise NotImplementedError


class ActionWrapper(Wrapper[ObsT, NewActT], Generic[ObsT, NewActT]):  # type: ignore[misc]
    """Wrapper that transforms actions before passing them to the env.

    Subclasses must implement :meth:`action` and may implement
    :meth:`reverse_action`.
    """

    def step(self, action: NewActT) -> StepResult:
        return self.env.step(self.action(action))  # type: ignore[arg-type]

    def action(self, action: NewActT) -> ActType:
        """Transform an action before passing it to the wrapped env.

        Parameters
        ----------
        action:
            The action from the agent.

        Returns
        -------
        ActType
            The transformed action.
        """
        raise NotImplementedError

    def reverse_action(self, action: ActType) -> NewActT:
        """Reverse-transform an action (optional).

        Not all wrappers support this.
        """
        raise NotImplementedError


class RewardWrapper(Wrapper[ObsT, ActT]):
    """Wrapper that transforms the scalar reward.

    Subclasses must implement :meth:`reward`.
    """

    def step(self, action: ActT) -> StepResult:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, self.reward(reward), terminated, truncated, info

    def reward(self, reward: float) -> float:
        """Transform a reward from the wrapped environment.

        Parameters
        ----------
        reward:
            The raw reward.

        Returns
        -------
        float
            The transformed reward.
        """
        raise NotImplementedError
