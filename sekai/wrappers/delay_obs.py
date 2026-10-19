from __future__ import annotations

from collections import deque
from typing import Any, Literal

from sekai._types import ActType
from sekai.core.env import Env
from sekai.core.result import ResetResult
from sekai.core.wrapper import ObservationWrapper
from sekai.spaces.utils import zero_sample


class DelayObservation(ObservationWrapper[Any, ActType, Any]):
    """Delays observations by `delay` steps, modelling feed or decision latency.

    Until `delay` observations have accumulated, the wrapper returns padding:
    "zeros" (the space's zero point, clipped into bounds) or "reset" (the
    episode's first observation, often more natural for price series).
    """

    def __init__(
        self,
        env: Env[Any, ActType],
        delay: int,
        *,
        padding: Literal["zeros", "reset"] = "zeros",
    ) -> None:
        if delay < 0:
            raise ValueError(f"delay must be >= 0, got {delay}.")
        if padding not in ("zeros", "reset"):
            raise ValueError(f"padding must be 'zeros' or 'reset', got {padding!r}.")
        super().__init__(env)
        self.delay = delay
        self.padding = padding
        self._queue: deque[Any] = deque()
        self._pad: Any = None

    def observation(self, obs: Any) -> Any:
        self._queue.append(obs)
        if len(self._queue) > self.delay:
            return self._queue.popleft()
        return self._pad

    def _wrap_reset(self, result: ResetResult[Any]) -> ResetResult[Any]:
        self._queue.clear()
        self._pad = (
            result.observation if self.padding == "reset" else zero_sample(self.observation_space)
        )
        return super()._wrap_reset(result)
