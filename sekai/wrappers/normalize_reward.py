from __future__ import annotations

from typing import Any

import numpy as np

from sekai._types import ActType, ObsType, SeedType
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.core.wrapper import Wrapper
from sekai.utils.running_stats import RunningMeanStd


class NormalizeReward(Wrapper[ObsType, ActType, ObsType, ActType]):
    """Scales rewards by a running estimate of the discounted return's std.

    As in OpenAI baselines' VecNormalize, the mean is NOT subtracted, only the
    scale is adjusted. The discounted return is reset at the end of every
    episode. Set update_running_mean = False to freeze the statistics.
    """

    def __init__(
        self,
        env: Env[ObsType, ActType],
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon
        self.return_rms = RunningMeanStd(shape=())
        self.update_running_mean: bool = True
        self._returns: float = 0.0

    def _on_step(self, result: StepResult[ObsType]) -> StepResult[ObsType]:
        self._returns = self._returns * self.gamma + float(result.reward)
        if self.update_running_mean:
            self.return_rms.update(np.array([self._returns]))
        if result.done:
            self._returns = 0.0
        scale = float(np.sqrt(self.return_rms.var + self.epsilon))
        return StepResult(
            observation=result.observation,
            reward=result.reward / scale,
            terminated=result.terminated,
            truncated=result.truncated,
            info=result.info,
        )

    def reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[ObsType]:
        self._returns = 0.0
        return self.env.reset(seed=seed, options=options)

    async def async_reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[ObsType]:
        self._returns = 0.0
        return await self.env.async_reset(seed=seed, options=options)

    def step(self, action: ActType) -> StepResult[ObsType]:
        return self._on_step(self.env.step(action))

    async def async_step(self, action: ActType) -> StepResult[ObsType]:
        return self._on_step(await self.env.async_step(action))
