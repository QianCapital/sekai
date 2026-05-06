"""ClipReward wrapper — clips rewards to a specified range."""

from __future__ import annotations

from sekai.core.env import Env
from sekai.utils.typing import ObsType, ActType
from .base import RewardWrapper


class ClipReward(RewardWrapper[ObsType, ActType]):
    """Clips the reward to ``[min_reward, max_reward]``.

    Parameters
    ----------
    env:
        The environment to wrap.
    min_reward:
        Lower bound for clipping (default ``-1``).
    max_reward:
        Upper bound for clipping (default ``+1``).
    """

    def __init__(
        self,
        env: Env[ObsType, ActType],
        min_reward: float = -1.0,
        max_reward: float = 1.0,
    ) -> None:
        super().__init__(env)
        if min_reward > max_reward:
            raise ValueError(
                f"min_reward ({min_reward}) must be <= max_reward ({max_reward})"
            )
        self.min_reward = min_reward
        self.max_reward = max_reward

    def reward(self, reward: float) -> float:
        return max(self.min_reward, min(self.max_reward, reward))
