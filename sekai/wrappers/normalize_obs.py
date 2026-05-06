from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from sekai._types import ActType
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.core.wrapper import Wrapper
from sekai.spaces.box import Box
from sekai._types import SeedType


class NormalizeObservation(Wrapper[NDArray[np.float64], ActType, NDArray[np.float64], ActType]):
    """Normalises observations to have approximately zero mean and unit variance.

    Uses Welford's online algorithm to track running mean and variance.
    """

    def __init__(self, env: Env[NDArray[np.float64], ActType], epsilon: float = 1e-8) -> None:
        assert isinstance(env.observation_space, Box), (
            "NormalizeObservation requires a Box observation space."
        )
        super().__init__(env)
        self.epsilon = epsilon
        obs_space: Box = env.observation_space  # type: ignore[assignment]
        self._shape = obs_space.shape or ()
        self._mean = np.zeros(self._shape, dtype=np.float64)
        self._var = np.ones(self._shape, dtype=np.float64)
        self._count: float = 0.0

        self.observation_space = Box(  # type: ignore[assignment]
            low=-np.inf,
            high=np.inf,
            shape=self._shape,
            dtype=np.float64,
        )

    def _update_stats(self, obs: NDArray[np.float64]) -> None:
        self._count += 1
        delta = obs - self._mean
        self._mean += delta / self._count
        delta2 = obs - self._mean
        self._var += delta * delta2

    def _normalize(self, obs: NDArray[np.float64]) -> NDArray[np.float64]:
        std = np.sqrt(self._var / max(self._count, 1) + self.epsilon)
        return (obs - self._mean) / std

    def reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[NDArray[np.float64]]:
        result = self.env.reset(seed=seed, options=options)
        obs = result.observation.astype(np.float64)
        self._update_stats(obs)
        return ResetResult(observation=self._normalize(obs), info=result.info)

    def step(self, action: ActType) -> StepResult[NDArray[np.float64]]:
        result = self.env.step(action)
        obs = result.observation.astype(np.float64)
        self._update_stats(obs)
        return StepResult(
            observation=self._normalize(obs),
            reward=result.reward,
            terminated=result.terminated,
            truncated=result.truncated,
            info=result.info,
        )
