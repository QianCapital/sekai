from __future__ import annotations

import time
from typing import Any

from sekai._types import ActType, ObsType, SeedType
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.core.wrapper import Wrapper
from sekai.stats.tracker import EpisodeStats, StatsTracker
from sekai.stats.collectors import InMemoryTracker


class RecordEpisodeStatistics(Wrapper[ObsType, ActType, ObsType, ActType]):
    """Records per-episode statistics via a pluggable StatsTracker.

    On episode termination, info["episode"] is set to an EpisodeStats dataclass
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
        self._start_time: float = 0.0
        self._episode_stats: EpisodeStats | None = None

    @property
    def episode_stats(self) -> EpisodeStats | None:
        """The most recently completed episode's stats."""
        return self._episode_stats

    def reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[ObsType]:
        result = self.env.reset(seed=seed, options=options)
        self._episode_return = 0.0
        self._episode_length = 0
        self._start_time = time.monotonic()
        spec_id = getattr(self.spec, "id", None)
        self.tracker.on_reset(env_id=spec_id, seed=self.env._rng_seed)
        return result

    def step(self, action: ActType) -> StepResult[ObsType]:
        result = self.env.step(action)
        self._episode_return += result.reward
        self._episode_length += 1
        self.tracker.on_step(result.reward, result.terminated, result.truncated, result.info)

        info = result.info
        if result.done:
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
            info = {**info, "episode": stats}

        return StepResult(
            observation=result.observation,
            reward=result.reward,
            terminated=result.terminated,
            truncated=result.truncated,
            info=info,
        )
