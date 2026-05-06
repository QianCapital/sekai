from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from sekai._types import ActType
from sekai.core.env import Env
from sekai.core.wrapper import ObservationWrapper
from sekai.spaces import flatten, flatten_space
from sekai.spaces.space import Space


class FlattenObservation(ObservationWrapper[NDArray[np.float32], ActType, Any]):
    """Flattens the observation from a nested space into a 1D float32 array."""

    def __init__(self, env: Env[Any, ActType]) -> None:
        super().__init__(env)
        self.observation_space = flatten_space(env.observation_space)  # type: ignore[assignment]
        self._inner_space: Space[Any] = env.observation_space

    def observation(self, obs: Any) -> NDArray[np.float32]:
        return flatten(self._inner_space, obs)
