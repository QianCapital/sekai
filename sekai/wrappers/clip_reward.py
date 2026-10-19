from __future__ import annotations

import numpy as np

from sekai._types import ActType, ObsType
from sekai.core.env import Env
from sekai.core.wrapper import RewardWrapper


class ClipReward(RewardWrapper[ObsType, ActType]):
    """Clips every reward to [min_reward, max_reward].

    Example:
        ```python
        env = ClipReward(env, -1.0, 1.0)
        ```
    """

    def __init__(
        self,
        env: Env[ObsType, ActType],
        min_reward: float = -np.inf,
        max_reward: float = np.inf,
    ) -> None:
        if not min_reward < max_reward:
            raise ValueError(
                f"min_reward ({min_reward}) must be less than max_reward ({max_reward})."
            )
        super().__init__(env)
        self.min_reward = float(min_reward)
        self.max_reward = float(max_reward)

    def reward(self, reward: float) -> float:
        return float(min(max(reward, self.min_reward), self.max_reward))
