from __future__ import annotations

from typing import Any

from sekai._types import ActType, ObsType, SeedType
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.core.wrapper import Wrapper


class TimeLimit(Wrapper[ObsType, ActType, ObsType, ActType]):
    """Truncates episodes after max_episode_steps steps."""

    def __init__(self, env: Env[ObsType, ActType], max_episode_steps: int) -> None:
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self._elapsed: int = 0

    def reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[ObsType]:
        self._elapsed = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action: ActType) -> StepResult[ObsType]:
        result = self.env.step(action)
        self._elapsed += 1
        truncated = result.truncated or (self._elapsed >= self.max_episode_steps)
        return StepResult(
            observation=result.observation,
            reward=result.reward,
            terminated=result.terminated,
            truncated=truncated,
            info={**result.info, "elapsed_steps": self._elapsed},
        )
