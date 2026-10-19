from __future__ import annotations

import time
from typing import Any

from sekai._types import ActType, ObsType, SeedType
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.core.wrapper import Wrapper
from sekai.stats.collectors import InMemoryTracker
from sekai.stats.tracker import EpisodeStats, StatsTracker


class RecordEpisodeStatistics(Wrapper[ObsType, ActType, ObsType, ActType]):
    """Records per-episode statistics via a pluggable StatsTracker.

    On episode termination, `info["episode"]` is set to an EpisodeStats dataclass
    (not an untyped dict), so type checkers and IDEs can verify access.
    """

    def __init__(
        self,
        env: Env[ObsType, ActType],
        tracker: StatsTracker | None = None,
        deque_size: int = 100,
    ) -> None:
        super().__init__(env)
        self.tracker: StatsTracker = tracker or InMemoryTracker(max_episodes=deque_size)
        self._episode_return: float = 0.0
        self._episode_length: int = 0
        self._start_time: float = time.monotonic()
        self._episode_stats: EpisodeStats | None = None

    @property
    def episode_stats(self) -> EpisodeStats | None:
        """The most recently completed episode's stats."""
        return self._episode_stats

    def _on_reset(self, result: ResetResult[ObsType]) -> ResetResult[ObsType]:
        self._episode_return = 0.0
        self._episode_length = 0
        self._start_time = time.monotonic()
        self.tracker.on_reset(env_id=getattr(self.spec, "id", None), seed=self._rng_seed)
        return result

    def _on_step(self, result: StepResult[ObsType]) -> StepResult[ObsType]:
        # float(): NumPy 2 would keep a sum of np.float32 rewards in float32.
        self._episode_return += float(result.reward)
        self._episode_length += 1
        self.tracker.on_step(result.reward, result.terminated, result.truncated, result.info)

        if not result.done:
            return result

        end_time = time.monotonic()
        stats = EpisodeStats(
            episode_return=self._episode_return,
            episode_length=self._episode_length,
            elapsed_time=end_time - self._start_time,
            start_time=self._start_time,
            end_time=end_time,
            terminated=result.terminated,
            truncated=result.truncated,
        )
        self._episode_stats = stats
        self.tracker.on_episode_end(stats)
        # Start counting the next episode now, in case an inner AutoReset continues
        # without a reset() call reaching this wrapper.
        self._episode_return = 0.0
        self._episode_length = 0
        self._start_time = end_time
        return StepResult(
            observation=result.observation,
            reward=result.reward,
            terminated=result.terminated,
            truncated=result.truncated,
            info={**result.info, "episode": stats},
        )

    def reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[ObsType]:
        return self._on_reset(self.env.reset(seed=seed, options=options))

    async def async_reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[ObsType]:
        return self._on_reset(await self.env.async_reset(seed=seed, options=options))

    def step(self, action: ActType) -> StepResult[ObsType]:
        return self._on_step(self.env.step(action))

    async def async_step(self, action: ActType) -> StepResult[ObsType]:
        return self._on_step(await self.env.async_step(action))
