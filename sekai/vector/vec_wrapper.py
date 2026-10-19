from __future__ import annotations

import abc
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray

from sekai._types import ActType, InfoDict, ObsType, SeedType, WActType, WObsType
from sekai.spaces.space import Space
from sekai.vector.result import VecResetResult, VecStepResult
from sekai.vector.vec_env import VecEnv


class VecWrapper(VecEnv[WObsType, WActType], Generic[WObsType, WActType, ObsType, ActType]):
    """Transparent wrapper around a VecEnv.

    Like Wrapper, observation_space / action_space delegate to the wrapped
    VecEnv unless assigned (always the single-env spaces). Both the sync and
    async APIs are delegated, so wrapping an AsyncVecEnv keeps
    `await vec.async_step(...)` concurrent.
    """

    def __init__(self, venv: VecEnv[ObsType, ActType]) -> None:
        if not isinstance(venv, VecEnv):
            raise TypeError(f"{type(self).__name__} expected a VecEnv, got {type(venv).__name__}.")
        self.venv = venv
        self.num_envs = venv.num_envs
        self.autoreset = venv.autoreset
        self._observation_space: Space[Any] | None = None
        self._action_space: Space[Any] | None = None

    @property
    def observation_space(self) -> Any:
        if self._observation_space is not None:
            return self._observation_space
        return self.venv.observation_space

    @observation_space.setter
    def observation_space(self, space: Space[Any]) -> None:
        self._observation_space = space

    @property
    def action_space(self) -> Any:
        if self._action_space is not None:
            return self._action_space
        return self.venv.action_space

    @action_space.setter
    def action_space(self, space: Space[Any]) -> None:
        self._action_space = space

    @property
    def spec(self) -> Any:
        return self.venv.spec

    @spec.setter
    def spec(self, value: Any) -> None:
        self.venv.spec = value

    @property
    def closed(self) -> bool:
        return self.venv.closed

    @closed.setter
    def closed(self, value: bool) -> None:
        self.venv.closed = value

    def step(self, actions: Any) -> VecStepResult[WObsType]:
        return self.venv.step(actions)

    async def async_step(self, actions: Any) -> VecStepResult[WObsType]:
        return await self.venv.async_step(actions)

    def reset(
        self,
        *,
        seed: list[SeedType] | SeedType = None,
        options: list[dict[str, Any] | None] | dict[str, Any] | None = None,
    ) -> VecResetResult[WObsType]:
        return self.venv.reset(seed=seed, options=options)

    async def async_reset(
        self,
        *,
        seed: list[SeedType] | SeedType = None,
        options: list[dict[str, Any] | None] | dict[str, Any] | None = None,
    ) -> VecResetResult[WObsType]:
        return await self.venv.async_reset(seed=seed, options=options)

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
    """Applies a transformation to every batch of observations.

    Terminal observations that autoreset stores in `infos[i]["final_observation"]`
    are transformed too, via final_observation(), so value bootstrapping sees
    the same representation as the policy.
    """

    @abc.abstractmethod
    def observation(self, obs: ObsType) -> WObsType: ...

    def final_observation(self, obs: Any) -> Any:
        """Transform one sub-env's terminal observation (from autoreset infos).

        The default runs observation() on a batch of one. Override it if your
        transform is not valid for a single-row batch or has side effects (e.g.
        updating running statistics) that terminal observations should not cause.
        """
        from sekai.spaces.utils import concatenate, iterate

        batch = concatenate(self.venv.observation_space, [obs])
        return next(iter(iterate(self.observation_space, self.observation(batch))))

    def _wrap_step(self, result: VecStepResult[Any]) -> VecStepResult[WObsType]:
        observations = self.observation(result.observations)
        infos: list[InfoDict] = [
            {**info, "final_observation": self.final_observation(info["final_observation"])}
            if "final_observation" in info
            else info
            for info in result.infos
        ]
        return VecStepResult(
            observations=observations,
            rewards=result.rewards,
            terminated=result.terminated,
            truncated=result.truncated,
            infos=infos,
        )

    def _wrap_reset(self, result: VecResetResult[Any]) -> VecResetResult[WObsType]:
        return VecResetResult(
            observations=self.observation(result.observations), infos=result.infos
        )

    def step(self, actions: Any) -> VecStepResult[WObsType]:
        return self._wrap_step(self.venv.step(actions))

    async def async_step(self, actions: Any) -> VecStepResult[WObsType]:
        return self._wrap_step(await self.venv.async_step(actions))

    def reset(
        self,
        *,
        seed: list[SeedType] | SeedType = None,
        options: list[dict[str, Any] | None] | dict[str, Any] | None = None,
    ) -> VecResetResult[WObsType]:
        return self._wrap_reset(self.venv.reset(seed=seed, options=options))

    async def async_reset(
        self,
        *,
        seed: list[SeedType] | SeedType = None,
        options: list[dict[str, Any] | None] | dict[str, Any] | None = None,
    ) -> VecResetResult[WObsType]:
        return self._wrap_reset(await self.venv.async_reset(seed=seed, options=options))


class VecRewardWrapper(
    VecWrapper[ObsType, ActType, ObsType, ActType],
    Generic[ObsType, ActType],
):
    """Applies a transformation to every batch of rewards."""

    @abc.abstractmethod
    def reward(self, rewards: NDArray[np.float64]) -> NDArray[np.float64]: ...

    def _wrap_step(self, result: VecStepResult[ObsType]) -> VecStepResult[ObsType]:
        return VecStepResult(
            observations=result.observations,
            rewards=self.reward(result.rewards),
            terminated=result.terminated,
            truncated=result.truncated,
            infos=result.infos,
        )

    def step(self, actions: Any) -> VecStepResult[ObsType]:
        return self._wrap_step(self.venv.step(actions))

    async def async_step(self, actions: Any) -> VecStepResult[ObsType]:
        return self._wrap_step(await self.venv.async_step(actions))


class VecActionWrapper(
    VecWrapper[ObsType, WActType, ObsType, ActType],
    Generic[ObsType, WActType, ActType],
):
    """Applies a transformation to every batch of actions."""

    @abc.abstractmethod
    def action(self, actions: WActType) -> ActType: ...

    def step(self, actions: WActType) -> VecStepResult[ObsType]:
        return self.venv.step(self.action(actions))

    async def async_step(self, actions: WActType) -> VecStepResult[ObsType]:
        return await self.venv.async_step(self.action(actions))
