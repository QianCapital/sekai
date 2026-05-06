from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from sekai._types import ActType, ObsType, SeedType
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.core.wrapper import Wrapper


class NormalizeReward(Wrapper[ObsType, ActType, ObsType, ActType]):
    """Normalises rewards using a running variance estimate.

    Tracks a discounted return variance (as in OpenAI baselines) to scale
    rewards to unit variance. The mean is NOT subtracted — only the scale
    is adjusted.
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
        self._returns: float = 0.0
        self._var: float = 1.0
        self._count: float = 0.0
        self._mean: float = 0.0

    def _update(self, reward: float) -> None:
        self._returns = self._returns * self.gamma + reward
        self._count += 1
        delta = self._returns - self._mean
        self._mean += delta / self._count
        delta2 = self._returns - self._mean
        self._var += delta * delta2

    def reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[ObsType]:
        self._returns = 0.0
        return self.env.reset(seed=seed, options=options)

    def step(self, action: ActType) -> StepResult[ObsType]:
        result = self.env.step(action)
        self._update(result.reward)
        std = float(np.sqrt(self._var / max(self._count, 1) + self.epsilon))
        return StepResult(
            observation=result.observation,
            reward=result.reward / std,
            terminated=result.terminated,
            truncated=result.truncated,
            info=result.info,
        )
