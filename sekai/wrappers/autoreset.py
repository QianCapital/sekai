from __future__ import annotations

from typing import Any

from sekai._types import ActType, ObsType, SeedType
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.core.wrapper import Wrapper


class AutoReset(Wrapper[ObsType, ActType, ObsType, ActType]):
    """Automatically resets the environment when an episode ends.

    On terminal steps, the observation in StepResult is the *first observation
    of the new episode*, not the terminal observation. The terminal observation
    and reset info are stored in info["final_observation"] and info["final_info"].
    """

    def step(self, action: ActType) -> StepResult[ObsType]:
        result = self.env.step(action)
        if result.done:
            reset_result = self.env.reset()
            info = {
                **result.info,
                "final_observation": result.observation,
                "final_info": result.info,
            }
            return StepResult(
                observation=reset_result.observation,
                reward=result.reward,
                terminated=result.terminated,
                truncated=result.truncated,
                info=info,
            )
        return result
