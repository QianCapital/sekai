from __future__ import annotations

from typing import Any

from sekai._types import ActType, ObsType, SeedType
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.core.wrapper import Wrapper
from sekai.error import ResetRequired


class OrderEnforcing(Wrapper[ObsType, ActType, ObsType, ActType]):
    """Raises ResetRequired if step() is called before the first reset().

    Enforces proper API usage — essential for catching subtle bugs during
    development, especially in distributed RL setups.
    """

    def __init__(self, env: Env[ObsType, ActType]) -> None:
        super().__init__(env)
        self._has_reset: bool = False

    def reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[ObsType]:
        self._has_reset = True
        return self.env.reset(seed=seed, options=options)

    def step(self, action: ActType) -> StepResult[ObsType]:
        if not self._has_reset:
            raise ResetRequired()
        return self.env.step(action)
