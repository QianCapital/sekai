from __future__ import annotations

import numpy as np

from sekai._types import ActType, ObsType
from sekai.core.env import Env
from sekai.core.wrapper import RewardWrapper


class ClipReward(RewardWrapper[ObsType, ActType]):
    """Clips every reward to [min_reward, max_reward].

    A bound of None (as in gymnasium) or infinity leaves that side unbounded.

    Example:
        ```python
        env = ClipReward(env, -1.0, 1.0)
        env = ClipReward(env, None, 0.5)  # cap gains only
        ```
    """

    def __init__(
        self,
        env: Env[ObsType, ActType],
        min_reward: float | None = -np.inf,
        max_reward: float | None = np.inf,
    ) -> None:
        low = -np.inf if min_reward is None else float(min_reward)
        high = np.inf if max_reward is None else float(max_reward)
        if not low < high:
            raise ValueError(
                f"min_reward ({min_reward}) must be less than max_reward ({max_reward})."
            )
        super().__init__(env)
        self.min_reward = low
        self.max_reward = high

    def reward(self, reward: float) -> float:
        return float(min(max(reward, self.min_reward), self.max_reward))
