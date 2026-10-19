from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sekai._types import ActType
from sekai.core.env import Env
from sekai.core.wrapper import ObservationWrapper
from sekai.spaces.box import Box, float_dtype


class RescaleObservation(ObservationWrapper[NDArray[Any], ActType, NDArray[Any]]):
    """Affinely maps a bounded Box observation onto [min_obs, max_obs].

    Integer observations (e.g. uint8 pixels) are rescaled to float32; floating
    observations keep their dtype.

    Example:
        ```python
        env = RescaleObservation(env, -1.0, 1.0)
        ```
    """

    def __init__(
        self,
        env: Env[NDArray[Any], ActType],
        min_obs: ArrayLike,
        max_obs: ArrayLike,
    ) -> None:
        space = env.observation_space
        if not isinstance(space, Box):
            raise TypeError(
                f"RescaleObservation requires a Box observation space, got {type(space).__name__}."
            )
        if not all(space.is_bounded):
            raise ValueError("RescaleObservation requires an observation space with finite bounds.")
        super().__init__(env)
        shape = space.shape or ()
        dtype = float_dtype(space.dtype)
        self.min_obs = np.broadcast_to(np.asarray(min_obs, dtype=dtype), shape)
        self.max_obs = np.broadcast_to(np.asarray(max_obs, dtype=dtype), shape)
        if np.any(self.min_obs >= self.max_obs):
            raise ValueError("min_obs must be strictly less than max_obs.")
        self._src_low = space.low.astype(np.float64)
        span = space.high.astype(np.float64) - self._src_low
        # Degenerate dimensions (low == high) map to min_obs.
        self._scale = np.divide(
            self.max_obs - self.min_obs, span, out=np.zeros_like(span), where=span > 0
        )
        self.observation_space = Box(low=self.min_obs, high=self.max_obs, dtype=dtype)

    def observation(self, obs: NDArray[Any]) -> NDArray[Any]:
        out = self.min_obs + self._scale * (np.asarray(obs, dtype=np.float64) - self._src_low)
        clipped: NDArray[Any] = np.clip(out, self.min_obs, self.max_obs)
        return clipped.astype(self.observation_space.dtype)
