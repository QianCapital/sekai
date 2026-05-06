from __future__ import annotations

from typing import Callable

from sekai._types import ActType, ObsType
from sekai.core.env import Env
from sekai.core.wrapper import RewardWrapper


class TransformReward(RewardWrapper[ObsType, ActType]):
    """Applies an arbitrary callable transformation to every reward.

    Example:
        env = TransformReward(env, lambda r: np.sign(r))  # reward clipping
        env = TransformReward(env, lambda r: r * 0.01)    # reward scaling
    """

    def __init__(
        self,
        env: Env[ObsType, ActType],
        fn: Callable[[float], float],
    ) -> None:
        super().__init__(env)
        self._fn = fn

    def reward(self, reward: float) -> float:
        return self._fn(reward)
