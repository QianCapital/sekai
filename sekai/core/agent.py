"""Abstract base class for reinforcement learning agents."""

from __future__ import annotations

import abc
from typing import Any, Generic, TypeVar

from sekai.utils.typing import ActType, ObsType

ObsT = TypeVar("ObsT")
ActT = TypeVar("ActT")


class Agent(abc.ABC, Generic[ObsT, ActT]):
    """Abstract interface for reinforcement learning agents.

    An agent encapsulates a *policy* — a mapping from observations to
    actions — and optionally a *learning algorithm* that updates the
    policy from experience.

    Subclasses **must** implement :meth:`act`.
    Subclasses **may** implement :meth:`learn` and :meth:`reset`.
    """

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def act(self, observation: ObsT, *, explore: bool = True) -> ActT:
        """Choose an action given an *observation*.

        Parameters
        ----------
        observation:
            An observation from the environment.
        explore:
            When ``True`` the agent may apply exploratory noise or
            stochastic sampling.  When ``False`` the agent should act
            greedily.

        Returns
        -------
        action:
            The chosen action.
        """

    def learn(
        self,
        observation: ObsT,
        action: ActT,
        reward: float,
        next_observation: ObsT,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update the agent's policy from a single transition.

        Parameters
        ----------
        observation:
            The observation before the action was taken.
        action:
            The action taken.
        reward:
            The reward received.
        next_observation:
            The observation after the action was taken.
        terminated:
            Whether the episode terminated.
        truncated:
            Whether the episode was truncated.
        info:
            Optional auxiliary environment info.

        Returns
        -------
        dict[str, Any]
            Logging/diagnostic information about the update.
        """
        return {}

    def reset(self) -> None:
        """Reset any internal state at the beginning of a new episode.

        Override to clear hidden states, eligibility traces, etc.
        """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"


class RandomAgent(Agent[ObsType, ActType]):
    """A simple agent that samples random actions from the action space."""

    def __init__(self, action_space: Any) -> None:
        self.action_space = action_space

    def act(self, observation: ObsType, *, explore: bool = True) -> ActType:  # noqa: ARG002
        return self.action_space.sample()
