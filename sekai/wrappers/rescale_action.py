from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sekai._types import ObsType
from sekai.core.env import Env
from sekai.core.wrapper import ActionWrapper
from sekai.spaces.box import Box


class RescaleAction(ActionWrapper[ObsType, NDArray[np.float32], NDArray[np.float32]]):
    """Rescales a continuous action from [min_action, max_action] to the env's range.

    Useful when your policy outputs actions in a fixed range (e.g. [-1, 1])
    but the environment expects a different range.
    """

    def __init__(
        self,
        env: Env[ObsType, NDArray[np.float32]],
        min_action: float | NDArray[np.float32],
        max_action: float | NDArray[np.float32],
    ) -> None:
        assert isinstance(env.action_space, Box), "RescaleAction requires a Box action space."
        super().__init__(env)
        space: Box = env.action_space  # type: ignore[assignment]
        self.min_action = np.asarray(min_action, dtype=space.dtype)
        self.max_action = np.asarray(max_action, dtype=space.dtype)
        self.action_space = Box(  # type: ignore[assignment]
            low=self.min_action,
            high=self.max_action,
            dtype=space.dtype,
        )

    def action(self, action: NDArray[np.float32]) -> NDArray[np.float32]:
        space: Box = self.env.action_space  # type: ignore[assignment]
        low, high = space.low, space.high
        scale = (high - low) / (self.max_action - self.min_action)
        return low + scale * (action - self.min_action)

    def reverse_action(self, action: NDArray[np.float32]) -> NDArray[np.float32]:
        space: Box = self.env.action_space  # type: ignore[assignment]
        low, high = space.low, space.high
        scale = (self.max_action - self.min_action) / (high - low)
        return self.min_action + scale * (action - low)
