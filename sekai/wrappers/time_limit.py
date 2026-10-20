from __future__ import annotations

from typing import Any

from sekai._types import ActType, ObsType, SeedType
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.core.wrapper import Wrapper
from sekai.wrappers.autoreset import _warn_if_outside_autoreset


class TimeLimit(Wrapper[ObsType, ActType, ObsType, ActType]):
    """Truncates episodes after max_episode_steps steps.

    Adds `info["elapsed_steps"]` to every step result.
    """

    def __init__(self, env: Env[ObsType, ActType], max_episode_steps: int) -> None:
        if max_episode_steps < 1:
            raise ValueError(f"max_episode_steps must be >= 1, got {max_episode_steps}.")
        super().__init__(env)
        _warn_if_outside_autoreset(env, "TimeLimit")
        self.max_episode_steps = max_episode_steps
        self._elapsed: int = 0

    @property
    def elapsed_steps(self) -> int:
        """Steps taken in the current episode."""
        return self._elapsed

    def _on_step(self, result: StepResult[ObsType]) -> StepResult[ObsType]:
        self._elapsed += 1
        truncated = result.truncated or (self._elapsed >= self.max_episode_steps)
        return StepResult(
            observation=result.observation,
            reward=result.reward,
            terminated=result.terminated,
            truncated=truncated,
            info={**result.info, "elapsed_steps": self._elapsed},
        )

    def reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[ObsType]:
        self._elapsed = 0
        return self.env.reset(seed=seed, options=options)

    async def async_reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[ObsType]:
        self._elapsed = 0
        return await self.env.async_reset(seed=seed, options=options)

    def step(self, action: ActType) -> StepResult[ObsType]:
        return self._on_step(self.env.step(action))

    async def async_step(self, action: ActType) -> StepResult[ObsType]:
        return self._on_step(await self.env.async_step(action))
