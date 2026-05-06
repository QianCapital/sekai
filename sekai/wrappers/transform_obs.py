from __future__ import annotations

from typing import Any, Callable

from sekai._types import ActType
from sekai.core.env import Env
from sekai.core.wrapper import ObservationWrapper
from sekai.spaces.space import Space


class TransformObservation(ObservationWrapper[Any, ActType, Any]):
    """Applies an arbitrary callable transformation to every observation.

    Example:
        env = TransformObservation(env, lambda obs: obs / 255.0)
    """

    def __init__(
        self,
        env: Env[Any, ActType],
        fn: Callable[[Any], Any],
        new_observation_space: Space[Any] | None = None,
    ) -> None:
        super().__init__(env)
        self._fn = fn
        if new_observation_space is not None:
            self.observation_space = new_observation_space  # type: ignore[assignment]

    def observation(self, obs: Any) -> Any:
        return self._fn(obs)
