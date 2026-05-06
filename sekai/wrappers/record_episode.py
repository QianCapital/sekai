"""RecordEpisode wrapper — records episode statistics."""

from __future__ import annotations

from typing import Any

from sekai.core.env import Env
from sekai.utils.typing import ObsType, ActType, ResetResult, StepResult
from .base import Wrapper


class RecordEpisode(Wrapper[ObsType, ActType]):
    """Records episode-level statistics (cumulative reward, length, etc.).

    After each episode ends the episode statistics are stored in
    :attr:`episode_records` and the ``info`` dict returned by :meth:`step`
    is augmented with an ``"episode"`` key containing the same data.

    Parameters
    ----------
    env:
        The environment to wrap.
    deque_size:
        Maximum number of episode records to keep.  Older records are
        discarded.  Use ``None`` (default) to keep all records.

    Attributes
    ----------
    episode_records : list[dict]
        A list of episode summary dicts, each with keys:
        ``"r"`` (total reward), ``"l"`` (episode length),
        ``"t"`` (total elapsed steps across all episodes).
    """

    def __init__(
        self,
        env: Env[ObsType, ActType],
        deque_size: int | None = None,
    ) -> None:
        super().__init__(env)
        self._deque_size = deque_size
        self.episode_records: list[dict[str, Any]] = []
        self._episode_reward: float = 0.0
        self._episode_length: int = 0
        self._total_steps: int = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult:
        self._episode_reward = 0.0
        self._episode_length = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action: ActType) -> StepResult:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_reward += reward
        self._episode_length += 1
        self._total_steps += 1

        if terminated or truncated:
            episode_info = {
                "r": self._episode_reward,
                "l": self._episode_length,
                "t": self._total_steps,
            }
            self.episode_records.append(episode_info)
            if self._deque_size is not None and len(self.episode_records) > self._deque_size:
                self.episode_records.pop(0)
            info = dict(info)
            info["episode"] = episode_info

        return obs, reward, terminated, truncated, info
