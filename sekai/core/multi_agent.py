from __future__ import annotations

import abc
from typing import Any, Generic

from sekai._types import ActType, AgentID, ObsType, SeedType
from sekai.core.result import MAResetResult, MAStepResult
from sekai.spaces.space import Space


class MultiAgentEnv(abc.ABC, Generic[ObsType, ActType, AgentID]):
    """Abstract base class for multi-agent environments.

    Each agent gets its own typed observation and action space,
    supporting heterogeneous multi-agent setups (e.g. a portfolio manager
    agent + a risk manager agent with different obs/action spaces).

    agents: currently active agents (may shrink as agents terminate)
    possible_agents: universe of all possible agent IDs
    """

    agents: list[Any]
    possible_agents: list[Any]
    observation_spaces: dict[Any, Space[ObsType]]
    action_spaces: dict[Any, Space[ActType]]

    @abc.abstractmethod
    def reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> MAResetResult[ObsType, AgentID]:
        ...

    @abc.abstractmethod
    def step(self, actions: dict[Any, ActType]) -> MAStepResult[ObsType, AgentID]:
        ...

    async def async_reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> MAResetResult[ObsType, AgentID]:
        return self.reset(seed=seed, options=options)

    async def async_step(
        self, actions: dict[Any, ActType]
    ) -> MAStepResult[ObsType, AgentID]:
        return self.step(actions)

    def close(self) -> None:
        """Release any resources held by the environment."""

    def __enter__(self) -> MultiAgentEnv[ObsType, ActType, AgentID]:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    @property
    def max_num_agents(self) -> int:
        return len(self.possible_agents)

    def observation_space(self, agent: Any) -> Space[ObsType]:
        return self.observation_spaces[agent]

    def action_space(self, agent: Any) -> Space[ActType]:
        return self.action_spaces[agent]

    def __str__(self) -> str:
        return f"<{type(self).__name__} agents={self.agents}>"

    def __repr__(self) -> str:
        return str(self)
