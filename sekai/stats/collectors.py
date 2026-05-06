from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

from sekai._types import InfoDict
from sekai.stats.tracker import EpisodeStats


class InMemoryTracker:
    """Keeps a rolling window of EpisodeStats in memory."""

    def __init__(self, max_episodes: int | None = 100) -> None:
        self.max_episodes = max_episodes
        self.episodes: deque[EpisodeStats] = deque(maxlen=max_episodes)
        self._current_return: float = 0.0
        self._current_length: int = 0
        self._start_time: float = 0.0
        self._env_id: str | None = None
        self._seed: int | None = None

    def on_reset(self, env_id: str | None, seed: int | None) -> None:
        self._current_return = 0.0
        self._current_length = 0
        self._start_time = time.monotonic()
        self._env_id = env_id
        self._seed = seed

    def on_step(self, reward: float, terminated: bool, truncated: bool, info: InfoDict) -> None:
        self._current_return += reward
        self._current_length += 1

    def on_episode_end(self, stats: EpisodeStats) -> None:
        self.episodes.append(stats)

    def summary(self) -> dict[str, Any]:
        if not self.episodes:
            return {}
        returns = [e.episode_return for e in self.episodes]
        lengths = [e.episode_length for e in self.episodes]
        return {
            "num_episodes": len(self.episodes),
            "mean_return": sum(returns) / len(returns),
            "min_return": min(returns),
            "max_return": max(returns),
            "mean_length": sum(lengths) / len(lengths),
        }

    def mean_return(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.episode_return for e in self.episodes) / len(self.episodes)

    def mean_length(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.episode_length for e in self.episodes) / len(self.episodes)


class LoggingTracker:
    """Emits structured log records for each completed episode."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("sekai.stats")
        self._current_return: float = 0.0
        self._current_length: int = 0

    def on_reset(self, env_id: str | None, seed: int | None) -> None:
        self._current_return = 0.0
        self._current_length = 0

    def on_step(self, reward: float, terminated: bool, truncated: bool, info: InfoDict) -> None:
        self._current_return += reward
        self._current_length += 1

    def on_episode_end(self, stats: EpisodeStats) -> None:
        self.logger.info(
            "episode_end",
            extra={
                "episode_return": stats.episode_return,
                "episode_length": stats.episode_length,
                "elapsed_time": stats.elapsed_time,
                "terminated": stats.terminated,
                "truncated": stats.truncated,
            },
        )

    def summary(self) -> dict[str, Any]:
        return {}
