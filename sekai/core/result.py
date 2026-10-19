from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic

from sekai._types import AgentID, InfoDict, ObsType


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
    """Structured result from Env.reset().

    `done` is always False, so one loop handles both result types:

    ```python
    result = env.reset(seed=0)
    while not result.done:
        result = env.step(policy(result.observation))
    ```
    """

    observation: ObsType
    info: InfoDict

    @property
    def done(self) -> bool:
        return False


@dataclass(frozen=True, slots=True)
class MAStepResult(Generic[ObsType, AgentID]):
    """Structured result from MultiAgentEnv.step().

    `all_done` / `any_done` look at every agent that appears in `terminated` or
    `truncated`, matched by agent ID; an agent missing from one of the dicts counts as
    not terminated (or not truncated) there. With no agents in either dict, both are False.
    """

    observations: dict[Any, ObsType]
    rewards: dict[Any, float]
    terminated: dict[Any, bool]
    truncated: dict[Any, bool]
    info: dict[Any, InfoDict]

    def _done_flags(self) -> list[bool]:
        agents = self.terminated.keys() | self.truncated.keys()
        return [bool(self.terminated.get(a, False) or self.truncated.get(a, False)) for a in agents]

    @property
    def all_done(self) -> bool:
        flags = self._done_flags()
        return bool(flags) and all(flags)

    @property
    def any_done(self) -> bool:
        return any(self._done_flags())


@dataclass(frozen=True, slots=True)
class MAResetResult(Generic[ObsType, AgentID]):
    """Structured result from MultiAgentEnv.reset(). all_done/any_done are always False."""

    observations: dict[Any, ObsType]
    info: dict[Any, InfoDict]

    @property
    def all_done(self) -> bool:
        return False

    @property
    def any_done(self) -> bool:
        return False
