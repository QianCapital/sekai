"""TimeLimit wrapper — truncates episodes after a maximum number of steps."""

from __future__ import annotations

from typing import Any

from sekai.core.env import Env
from sekai.utils.typing import ObsType, ActType, ResetResult, StepResult
from .base import Wrapper


class TimeLimit(Wrapper[ObsType, ActType]):
    """Truncates an episode after *max_episode_steps* steps.

    When the step limit is reached ``truncated`` is set to ``True`` and
    the episode ends regardless of the wrapped environment's signal.

    Parameters
    ----------
    env:
        The environment to wrap.
    max_episode_steps:
        Maximum number of steps per episode (must be positive).

    Examples
    --------
    >>> import sekai
    >>> from sekai.wrappers import TimeLimit
    >>> env = TimeLimit(sekai.make("GridWorld-v0"), max_episode_steps=50)
    >>> obs, info = env.reset()
    """

    def __init__(self, env: Env[ObsType, ActType], max_episode_steps: int) -> None:
        if max_episode_steps <= 0:
            raise ValueError(
                f"max_episode_steps must be a positive integer, got {max_episode_steps}"
            )
        super().__init__(env)
        self._max_episode_steps: int = max_episode_steps
        self._elapsed_steps: int = 0

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult:
        self._elapsed_steps = 0
        return self.env.reset(seed=seed, options=options)

    def step(self, action: ActType) -> StepResult:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True
        info["elapsed_steps"] = self._elapsed_steps
        return obs, reward, terminated, truncated, info
