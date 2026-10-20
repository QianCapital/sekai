from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sekai._types import ObsType
from sekai.core.env import Env
from sekai.core.wrapper import ActionWrapper
from sekai.spaces.box import Box
from sekai.utils.masking import get_action_mask, supports_action_mask


class RescaleAction(ActionWrapper[ObsType, NDArray[np.float32], NDArray[np.float32]]):
    """Rescales a continuous action from [min_action, max_action] to the env's range.

    Useful when your policy outputs actions in a fixed range (e.g. [-1, 1])
    but the environment expects a different range. Rescaled actions are clipped to
    the env's bounds (and reverse_action() results to [min_action, max_action]).

    If the env has an action mask (see `sekai.utils.masking`), the components it
    masks reach the env as the env's zero (`action_space.zero()` of the env), as the
    mask convention requires, whatever the policy put there: rescaled, the wrapper's
    zero (what `action_space.sample(mask=...)` puts in a masked component) would be
    the middle of the env's range instead. reverse_action() does not apply the mask.
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
        self._masked = supports_action_mask(env)

    def action(self, action: NDArray[np.float32]) -> NDArray[np.float32]:
        space: Box = self.env.action_space  # type: ignore[assignment]
        low, high = space.low, space.high
        scale = (high - low) / (self.max_action - self.min_action)
        out = (low + scale * (action - self.min_action)).astype(space.dtype)
        # Clip after the cast: rounding can land the endpoints one ulp outside [low, high].
        clipped: NDArray[np.float32] = np.clip(out, low, high)
        mask = get_action_mask(self.env) if self._masked else None
        if mask is not None:
            try:
                keep = np.broadcast_to(np.asarray(mask).astype(bool), space.shape)
            except ValueError:  # a mask for another action space (an inner wrapper's)
                return clipped
            clipped = np.where(keep, clipped, space.zero()).astype(space.dtype)
        return clipped

    def reverse_action(self, action: NDArray[np.float32]) -> NDArray[np.float32]:
        space: Box = self.env.action_space  # type: ignore[assignment]
        low, high = space.low, space.high
        scale = (self.max_action - self.min_action) / (high - low)
        out = (self.min_action + scale * (action - low)).astype(space.dtype)
        clipped: NDArray[np.float32] = np.clip(out, self.min_action, self.max_action)
        return clipped
