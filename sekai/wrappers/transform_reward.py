from __future__ import annotations

from collections.abc import Callable

from sekai._types import ActType, ObsType
from sekai.core.env import Env
from sekai.core.wrapper import RewardWrapper
from sekai.wrappers.transform_obs import _func_argument


class TransformReward(RewardWrapper[ObsType, ActType]):
    """Applies an arbitrary callable transformation to every reward.

    `func` was called `fn` before sekai 0.2; `fn=` still works but is deprecated.

    Example:
        ```python
        env = TransformReward(env, lambda r: np.sign(r))  # reward clipping
        env = TransformReward(env, lambda r: r * 0.01)    # reward scaling
        ```
    """

    def __init__(
        self,
        env: Env[ObsType, ActType],
        func: Callable[[float], float] | None = None,
        *,
        fn: Callable[[float], float] | None = None,
    ) -> None:
        self.func = _func_argument(func, fn, "TransformReward")
        super().__init__(env)

    def reward(self, reward: float) -> float:
        return self.func(reward)
