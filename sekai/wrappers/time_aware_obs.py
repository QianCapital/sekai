from __future__ import annotations

from typing import Any

import numpy as np

from sekai._types import ActType
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.core.wrapper import ObservationWrapper
from sekai.spaces.box import Box, float_dtype
from sekai.spaces.dict import Dict
from sekai.spaces.masked_box import MaskedBox


def _find_time_limit(env: Env[Any, Any]) -> int | None:
    from sekai.core.wrapper import Wrapper
    from sekai.wrappers.time_limit import TimeLimit

    layer: Env[Any, Any] = env
    while isinstance(layer, Wrapper):
        if isinstance(layer, TimeLimit):
            return layer.max_episode_steps
        layer = layer.env
    return getattr(env.spec, "max_episode_steps", None)


class TimeAwareObservation(ObservationWrapper[Any, ActType, Any]):
    """Adds the episode's elapsed time to the observation.

    Without it, a time-limited episode is not Markov: the agent cannot tell
    that it is about to be truncated (e.g. the end of a trading session).

    For a 1-D Box observation the time feature is appended as one extra
    element; for a Dict observation it is added under the key "time".
    With normalize_time=True (the default) the feature is t / max_episode_steps
    in [0, 1]; max_episode_steps is read from an inner TimeLimit or the env's
    spec when not given.
    """

    def __init__(
        self,
        env: Env[Any, ActType],
        max_episode_steps: int | None = None,
        *,
        normalize_time: bool = True,
        dict_key: str = "time",
    ) -> None:
        super().__init__(env)
        self.normalize_time = normalize_time
        self.dict_key = dict_key
        self.max_episode_steps = max_episode_steps or _find_time_limit(env)
        if normalize_time and not self.max_episode_steps:
            raise ValueError(
                "TimeAwareObservation(normalize_time=True) needs max_episode_steps; pass it "
                "explicitly or wrap the env in TimeLimit first."
            )
        self._t = 0

        high = 1.0 if normalize_time else np.inf
        space = env.observation_space
        if isinstance(space, Box) and len(space.shape or ()) == 1:
            # Integer observations become float32 so the fractional time fits.
            self.observation_space = Box(
                low=np.append(space.low, 0.0),
                high=np.append(space.high, high),
                dtype=float_dtype(space.dtype),
            )
        elif isinstance(space, Dict) and not isinstance(space, MaskedBox):
            if dict_key in space.spaces:
                raise KeyError(f"Observation space already has a {dict_key!r} key.")
            self.observation_space = Dict(
                {**space.spaces, dict_key: Box(0.0, high, shape=(1,), dtype=np.float32)}
            )
        else:
            raise TypeError(
                f"TimeAwareObservation supports 1-D Box and Dict observation spaces, got {space!r}."
            )

    def _time_feature(self) -> float:
        if self.normalize_time:
            assert self.max_episode_steps
            return min(self._t / self.max_episode_steps, 1.0)
        return float(self._t)

    def observation(self, obs: Any) -> Any:
        t = self._time_feature()
        if isinstance(self.observation_space, Dict):
            return {**obs, self.dict_key: np.array([t], dtype=np.float32)}
        return np.append(obs, t).astype(self.observation_space.dtype)

    def _wrap_reset(self, result: ResetResult[Any]) -> ResetResult[Any]:
        self._t = 0
        return super()._wrap_reset(result)

    def _wrap_step(self, result: StepResult[Any]) -> StepResult[Any]:
        self._t += 1
        return super()._wrap_step(result)
