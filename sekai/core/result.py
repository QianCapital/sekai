from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic

from sekai._types import ActType, AgentID, InfoDict, ObsType


@dataclass(frozen=True, slots=True)
class StepResult(Generic[ObsType]):
    """Structured result from Env.step().

    Replaces the error-prone 5-tuple (obs, reward, terminated, truncated, info)
    pattern from gym/gymnasium.
    """

    observation: ObsType
    reward: float
    terminated: bool
    truncated: bool
    info: InfoDict

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@dataclass(frozen=True, slots=True)
class ResetResult(Generic[ObsType]):
    """Structured result from Env.reset()."""

    observation: ObsType
    info: InfoDict


@dataclass(frozen=True, slots=True)
class MAStepResult(Generic[ObsType, AgentID]):
    """Structured result from MultiAgentEnv.step()."""

    observations: dict[Any, ObsType]
    rewards: dict[Any, float]
    terminated: dict[Any, bool]
    truncated: dict[Any, bool]
    info: dict[Any, InfoDict]

    @property
    def all_done(self) -> bool:
        return all(t or u for t, u in zip(self.terminated.values(), self.truncated.values()))

    @property
    def any_done(self) -> bool:
        return any(t or u for t, u in zip(self.terminated.values(), self.truncated.values()))


@dataclass(frozen=True, slots=True)
class MAResetResult(Generic[ObsType, AgentID]):
    """Structured result from MultiAgentEnv.reset()."""

    observations: dict[Any, ObsType]
    info: dict[Any, InfoDict]
