from __future__ import annotations

import warnings
from typing import Any

from sekai._types import ActType, ObsType
from sekai.core.env import Env
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

    Make AutoReset the outermost wrapper (`AutoReset(TimeLimit(env, 100))`, not
    `TimeLimit(AutoReset(env), 100)`). Its reset() calls go to the layers inside it
    only, so a wrapper outside it that keeps per-episode state (TimeLimit,
    TimeAwareObservation, FrameStackObservation, DelayObservation, DelayAction,
    StickyAction) is never reset and carries one episode's state into the next:
    TimeLimit would truncate every step after the first limit. Those wrappers warn
    when they are built around an AutoReset. RecordEpisodeStatistics works on either
    side of it.
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


def _warn_if_outside_autoreset(env: Env[Any, Any], wrapper: str) -> None:
    """Warn when a wrapper with per-episode state is built around an AutoReset.

    AutoReset starts new episodes without a reset() call reaching the layers outside
    it, so such a wrapper would keep the previous episode's state.
    """
    layer = env
    while isinstance(layer, Wrapper):
        if isinstance(layer, AutoReset):
            warnings.warn(
                f"{wrapper} is wrapped around an AutoReset, so it does not see the resets "
                "that start new episodes and carries one episode's state into the next. "
                f"Make AutoReset the outermost wrapper: AutoReset({wrapper}(env, ...)).",
                UserWarning,
                stacklevel=3,
            )
            return
        layer = layer.env
