from __future__ import annotations

from sekai._types import ActType, ObsType
from sekai.core.env import Env
from sekai.core.result import StepResult
from sekai.core.wrapper import Wrapper


class ActionRepeat(Wrapper[ObsType, ActType, ObsType, ActType]):
    """Repeats each action `repeat` times, summing the rewards.

    Stops early if the episode ends. The returned observation and info are
    from the last inner step; `info["action_repeats"]` says how many ran.
    Useful to decouple the agent's decision interval from the simulator's
    tick (e.g. decide every 5 minutes on a 1-minute bar simulator).
    """

    def __init__(self, env: Env[ObsType, ActType], repeat: int) -> None:
        if repeat < 1:
            raise ValueError(f"repeat must be >= 1, got {repeat}.")
        super().__init__(env)
        self.repeat = repeat

    @staticmethod
    def _combine(last: StepResult[ObsType], total: float, n: int) -> StepResult[ObsType]:
        return StepResult(
            observation=last.observation,
            reward=total,
            terminated=last.terminated,
            truncated=last.truncated,
            info={**last.info, "action_repeats": n},
        )

    def step(self, action: ActType) -> StepResult[ObsType]:
        total, n = 0.0, 0
        while True:
            result = self.env.step(action)
            total += float(result.reward)  # a Python float, even for np.float32 rewards
            n += 1
            if result.done or n == self.repeat:
                return self._combine(result, total, n)

    async def async_step(self, action: ActType) -> StepResult[ObsType]:
        total, n = 0.0, 0
        while True:
            result = await self.env.async_step(action)
            total += float(result.reward)
            n += 1
            if result.done or n == self.repeat:
                return self._combine(result, total, n)
