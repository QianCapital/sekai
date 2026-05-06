from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from sekai._types import InfoDict


@dataclass(slots=True)
class EpisodeStats:
    """Typed episode statistics — not an untyped dict."""

    episode_return: float
    episode_length: int
    elapsed_time: float
    start_time: float
    end_time: float
    terminated: bool
    truncated: bool
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


@runtime_checkable
class StatsTracker(Protocol):
    """Protocol for pluggable statistics collection backends."""

    def on_reset(self, env_id: str | None, seed: int | None) -> None: ...
    def on_step(self, reward: float, terminated: bool, truncated: bool, info: InfoDict) -> None: ...
    def on_episode_end(self, stats: EpisodeStats) -> None: ...
    def summary(self) -> dict[str, Any]: ...
