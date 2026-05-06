from __future__ import annotations

import abc
from typing import Any, Generic

from sekai._types import ActType, ObsType, SeedType
from sekai.spaces.space import Space
from sekai.vector.result import VecResetResult, VecStepResult


class VecEnv(abc.ABC, Generic[ObsType, ActType]):
    """Abstract base for vectorised environments.

    Runs N environments in parallel and returns batched results.

    Key differences from gymnasium's VectorEnv:
      - observation_space is always the *single-env* space
      - batch_observation_space is always the *batched* space
      - step/reset return structured dataclasses, not tuples
      - infos is list[dict], not dict[str, ndarray]
    """

    num_envs: int
    observation_space: Space[ObsType]
    action_space: Space[ActType]

    @property
    def batch_observation_space(self) -> Space[Any]:
        from sekai.spaces.utils import batch_space
        return batch_space(self.observation_space, self.num_envs)

    @abc.abstractmethod
    def step(self, actions: Any) -> VecStepResult[ObsType]:
        """Step all environments with a batch of actions."""
        ...

    @abc.abstractmethod
    def reset(
        self,
        *,
        seed: list[SeedType] | SeedType = None,
        options: list[dict[str, Any]] | dict[str, Any] | None = None,
    ) -> VecResetResult[ObsType]:
        """Reset all environments (or a subset if indices are given)."""
        ...

    @abc.abstractmethod
    def close(self) -> None:
        """Close all sub-environments and release resources."""
        ...

    def call(self, method: str, *args: Any, **kwargs: Any) -> list[Any]:
        """Call a method on each sub-environment and return the results."""
        raise NotImplementedError

    def get_attr(self, name: str) -> list[Any]:
        """Get an attribute from each sub-environment."""
        raise NotImplementedError

    def set_attr(self, name: str, values: list[Any] | Any) -> None:
        """Set an attribute on each sub-environment."""
        raise NotImplementedError

    def __len__(self) -> int:
        return self.num_envs

    def __enter__(self) -> VecEnv[ObsType, ActType]:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(num_envs={self.num_envs})"
