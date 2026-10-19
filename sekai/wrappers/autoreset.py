from __future__ import annotations

from sekai._types import ActType, ObsType
from sekai.core.result import ResetResult, StepResult
from sekai.core.wrapper import Wrapper


def _merge(step: StepResult[ObsType], reset: ResetResult[ObsType]) -> StepResult[ObsType]:
    return StepResult(
        observation=reset.observation,
        reward=step.reward,
        terminated=step.terminated,
        truncated=step.truncated,
        info={
            **step.info,
            "final_observation": step.observation,
            "final_info": step.info,
            "reset_info": reset.info,
        },
    )


class AutoReset(Wrapper[ObsType, ActType, ObsType, ActType]):
    """Automatically resets the environment when an episode ends.

    On terminal steps, the observation in StepResult is the *first observation
    of the new episode*, not the terminal observation. The terminal observation
    and info are stored in `info["final_observation"]` and `info["final_info"]`,
    and the reset's info in `info["reset_info"]`.
    """

    def step(self, action: ActType) -> StepResult[ObsType]:
        result = self.env.step(action)
        if result.done:
            return _merge(result, self.env.reset())
        return result

    async def async_step(self, action: ActType) -> StepResult[ObsType]:
        result = await self.env.async_step(action)
        if result.done:
            return _merge(result, await self.env.async_reset())
        return result
