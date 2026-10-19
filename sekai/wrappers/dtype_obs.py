from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import DTypeLike, NDArray

from sekai._types import ActType
from sekai.core.env import Env
from sekai.core.wrapper import ObservationWrapper
from sekai.spaces.box import Box


class DtypeObservation(ObservationWrapper[NDArray[Any], ActType, NDArray[Any]]):
    """Casts Box observations to dtype, e.g. float64 simulators feeding a float32 policy."""

    def __init__(self, env: Env[NDArray[Any], ActType], dtype: DTypeLike) -> None:
        space = env.observation_space
        if not isinstance(space, Box):
            raise TypeError(
                f"DtypeObservation requires a Box observation space, got {type(space).__name__}."
            )
        super().__init__(env)
        self.dtype = np.dtype(dtype)
        self.observation_space = Box(low=space.low, high=space.high, dtype=self.dtype)

    def observation(self, obs: NDArray[Any]) -> NDArray[Any]:
        return np.asarray(obs).astype(self.dtype)
