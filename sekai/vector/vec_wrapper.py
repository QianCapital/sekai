from __future__ import annotations

import abc
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray

from sekai._types import ActType, ObsType, SeedType, WActType, WObsType
from sekai.vector.result import VecResetResult, VecStepResult
from sekai.vector.vec_env import VecEnv


class VecWrapper(VecEnv[WObsType, WActType], Generic[WObsType, WActType, ObsType, ActType]):
    """Transparent wrapper around a VecEnv."""

    def __init__(self, venv: VecEnv[ObsType, ActType]) -> None:
        self.venv = venv
        self.num_envs = venv.num_envs

    @property  # type: ignore[override]
    def observation_space(self) -> Any:  # type: ignore[override]
        return self.venv.observation_space

    @property  # type: ignore[override]
    def action_space(self) -> Any:  # type: ignore[override]
        return self.venv.action_space

    def step(self, actions: Any) -> VecStepResult[WObsType]:
        return self.venv.step(actions)  # type: ignore[return-value]

    def reset(
        self,
        *,
        seed: list[SeedType] | SeedType = None,
        options: list[dict[str, Any]] | dict[str, Any] | None = None,
    ) -> VecResetResult[WObsType]:
        return self.venv.reset(seed=seed, options=options)  # type: ignore[return-value]

    def close(self) -> None:
        self.venv.close()

    def call(self, method: str, *args: Any, **kwargs: Any) -> list[Any]:
        return self.venv.call(method, *args, **kwargs)

    def get_attr(self, name: str) -> list[Any]:
        return self.venv.get_attr(name)

    def set_attr(self, name: str, values: list[Any] | Any) -> None:
        self.venv.set_attr(name, values)

    @property
    def unwrapped(self) -> VecEnv[Any, Any]:
        if isinstance(self.venv, VecWrapper):
            return self.venv.unwrapped
        return self.venv

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.venv})"


class VecObservationWrapper(
    VecWrapper[WObsType, ActType, ObsType, ActType],
    Generic[WObsType, ActType, ObsType],
):
    """Applies a transformation to every batch of observations."""

    @abc.abstractmethod
    def observation(self, obs: ObsType) -> WObsType:
        ...

    def step(self, actions: Any) -> VecStepResult[WObsType]:
        result = self.venv.step(actions)
        return VecStepResult(
            observations=self.observation(result.observations),  # type: ignore[arg-type]
            rewards=result.rewards,
            terminated=result.terminated,
            truncated=result.truncated,
            infos=result.infos,
        )

    def reset(
        self,
        *,
        seed: list[SeedType] | SeedType = None,
        options: list[dict[str, Any]] | dict[str, Any] | None = None,
    ) -> VecResetResult[WObsType]:
        result = self.venv.reset(seed=seed, options=options)
        return VecResetResult(
            observations=self.observation(result.observations),  # type: ignore[arg-type]
            infos=result.infos,
        )


class VecRewardWrapper(
    VecWrapper[ObsType, ActType, ObsType, ActType],
    Generic[ObsType, ActType],
):
    """Applies a transformation to every batch of rewards."""

    @abc.abstractmethod
    def reward(self, rewards: NDArray[np.float64]) -> NDArray[np.float64]:
        ...

    def step(self, actions: Any) -> VecStepResult[ObsType]:
        result = self.venv.step(actions)
        return VecStepResult(
            observations=result.observations,
            rewards=self.reward(result.rewards),
            terminated=result.terminated,
            truncated=result.truncated,
            infos=result.infos,
        )


class VecActionWrapper(
    VecWrapper[ObsType, WActType, ObsType, ActType],
    Generic[ObsType, WActType, ActType],
):
    """Applies a transformation to every batch of actions."""

    @abc.abstractmethod
    def action(self, actions: WActType) -> ActType:
        ...

    def step(self, actions: WActType) -> VecStepResult[ObsType]:
        return self.venv.step(self.action(actions))
