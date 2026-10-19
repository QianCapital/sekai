from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sekai._types import ActType
from sekai.core.env import Env
from sekai.core.wrapper import ObservationWrapper
from sekai.spaces.dict import Dict


class FilterObservation(ObservationWrapper[dict[str, Any], ActType, dict[str, Any]]):
    """Keeps only the given keys of a Dict observation.

    Example:
        ```python
        env = FilterObservation(env, ["prices", "holdings"])
        ```
    """

    def __init__(self, env: Env[dict[str, Any], ActType], keys: Sequence[str]) -> None:
        space = env.observation_space
        if not isinstance(space, Dict):
            raise TypeError(
                f"FilterObservation requires a Dict observation space, got {type(space).__name__}."
            )
        missing = [k for k in keys if k not in space.spaces]
        if missing:
            raise KeyError(f"Keys {missing} are not in the observation space {list(space.keys())}.")
        if not keys:
            raise ValueError("FilterObservation needs at least one key.")
        super().__init__(env)
        self.keys = tuple(keys)
        self.observation_space = Dict({k: space[k] for k in self.keys})

    def observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        return {k: obs[k] for k in self.keys}
