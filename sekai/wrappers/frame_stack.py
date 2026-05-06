from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sekai._types import ActType, SeedType
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.core.wrapper import Wrapper
from sekai.spaces.box import Box


class FrameStackObservation(Wrapper[NDArray[Any], ActType, NDArray[Any], ActType]):
    """Stacks the last N observations along a new leading axis.

    Commonly used in time-series and pixel-based environments where the
    agent needs temporal context.

    Input obs shape: (d,) or (h, w, c)
    Output obs shape: (N, d) or (N, h, w, c)
    """

    def __init__(self, env: Env[NDArray[Any], ActType], num_stack: int) -> None:
        assert isinstance(env.observation_space, Box), (
            "FrameStackObservation requires a Box observation space."
        )
        assert num_stack > 0, "num_stack must be positive."
        super().__init__(env)
        self.num_stack = num_stack
        self._frames: deque[NDArray[Any]] = deque(maxlen=num_stack)

        obs_space: Box = env.observation_space  # type: ignore[assignment]
        low = np.stack([obs_space.low] * num_stack, axis=0)
        high = np.stack([obs_space.high] * num_stack, axis=0)
        self.observation_space = Box(low=low, high=high, dtype=obs_space.dtype)  # type: ignore[assignment]

    def _get_obs(self) -> NDArray[Any]:
        assert len(self._frames) == self.num_stack
        return np.stack(list(self._frames), axis=0)

    def reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[NDArray[Any]]:
        result = self.env.reset(seed=seed, options=options)
        for _ in range(self.num_stack):
            self._frames.append(result.observation)
        return ResetResult(observation=self._get_obs(), info=result.info)

    def step(self, action: ActType) -> StepResult[NDArray[Any]]:
        result = self.env.step(action)
        self._frames.append(result.observation)
        return StepResult(
            observation=self._get_obs(),
            reward=result.reward,
            terminated=result.terminated,
            truncated=result.truncated,
            info=result.info,
        )
