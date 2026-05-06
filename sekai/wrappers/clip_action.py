from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sekai._types import ObsType
from sekai.core.env import Env
from sekai.core.wrapper import ActionWrapper
from sekai.spaces.box import Box


class ClipAction(ActionWrapper[ObsType, NDArray[np.float32], NDArray[np.float32]]):
    """Clips continuous actions to the env's action space bounds.

    Useful when a policy can output values outside the valid range.
    """

    def __init__(self, env: Env[ObsType, NDArray[np.float32]]) -> None:
        assert isinstance(env.action_space, Box), "ClipAction requires a Box action space."
        super().__init__(env)

    def action(self, action: NDArray[np.float32]) -> NDArray[np.float32]:
        space: Box = self.env.action_space  # type: ignore[assignment]
        return np.clip(action, space.low, space.high)

    def reverse_action(self, action: NDArray[np.float32]) -> NDArray[np.float32]:
        return action
