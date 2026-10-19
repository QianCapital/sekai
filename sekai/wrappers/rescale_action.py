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
    but the environment expects a different range. Rescaled actions are clipped to
    the env's bounds (and reverse_action() results to [min_action, max_action]).
    """

    def __init__(
        self,
        env: Env[ObsType, NDArray[np.float32]],
        min_action: float | NDArray[np.float32],
        max_action: float | NDArray[np.float32],
    ) -> None:
        if not isinstance(env.action_space, Box):
            raise TypeError(
                f"RescaleAction requires a Box action space, got {type(env.action_space).__name__}."
            )
        space: Box = env.action_space
        if not all(space.is_bounded):
            raise ValueError("RescaleAction requires a Box action space with finite bounds.")
        super().__init__(env)
        shape = space.shape or ()
        self.min_action = np.broadcast_to(np.asarray(min_action, dtype=space.dtype), shape)
        self.max_action = np.broadcast_to(np.asarray(max_action, dtype=space.dtype), shape)
        if np.any(self.min_action >= self.max_action):
            raise ValueError("min_action must be strictly less than max_action.")
        self.action_space = Box(
            low=self.min_action,
            high=self.max_action,
            dtype=space.dtype,
        )

    def action(self, action: NDArray[np.float32]) -> NDArray[np.float32]:
        space: Box = self.env.action_space  # type: ignore[assignment]
        low, high = space.low, space.high
        scale = (high - low) / (self.max_action - self.min_action)
        out = (low + scale * (action - self.min_action)).astype(space.dtype)
        # Clip after the cast: rounding can land the endpoints one ulp outside [low, high].
        clipped: NDArray[np.float32] = np.clip(out, low, high)
        return clipped

    def reverse_action(self, action: NDArray[np.float32]) -> NDArray[np.float32]:
        space: Box = self.env.action_space  # type: ignore[assignment]
        low, high = space.low, space.high
        scale = (self.max_action - self.min_action) / (high - low)
        out = (self.min_action + scale * (action - low)).astype(space.dtype)
        clipped: NDArray[np.float32] = np.clip(out, self.min_action, self.max_action)
        return clipped
