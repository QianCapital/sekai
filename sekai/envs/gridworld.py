"""Simple discrete grid-world environment.

The agent navigates a 2-D grid from a start cell to a goal cell while
avoiding walls.  This environment is useful for testing tabular RL
algorithms.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sekai.core.env import Env
from sekai.spaces.box import Box
from sekai.spaces.discrete import Discrete
from sekai.utils.typing import ResetResult, StepResult

# Actions
_UP = 0
_RIGHT = 1
_DOWN = 2
_LEFT = 3

_DELTA: dict[int, tuple[int, int]] = {
    _UP: (-1, 0),
    _RIGHT: (0, 1),
    _DOWN: (1, 0),
    _LEFT: (0, -1),
}


class GridWorldEnv(Env[np.ndarray, np.intp]):
    """A simple navigable grid world.

    Observation
    -----------
    A flat ``float32`` vector of length 2 with the normalised (row, col)
    of the agent's current position.

    Actions
    -------
    0 — move up
    1 — move right
    2 — move down
    3 — move left

    Rewards
    -------
    +1.0 when the goal is reached, -0.01 at every other step.

    Episode end
    -----------
    Terminated when the agent reaches the goal.

    Parameters
    ----------
    size:
        Number of rows (and columns) in the square grid (default 5).
    render_mode:
        Currently unused.
    """

    metadata = {"render_modes": [], "render_fps": 4}
    reward_range = (-1.0, 1.0)

    def __init__(self, size: int = 5, render_mode: str | None = None) -> None:
        if size < 2:
            raise ValueError(f"size must be >= 2, got {size}")
        self.size = size
        self.render_mode = render_mode

        self.observation_space = Box(
            low=0.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.action_space = Discrete(4)

        self._agent: tuple[int, int] = (0, 0)
        self._goal: tuple[int, int] = (size - 1, size - 1)

    def _to_obs(self) -> np.ndarray:
        row, col = self._agent
        return np.array([row / (self.size - 1), col / (self.size - 1)], dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult:
        if seed is not None:
            self._seed_np_random(seed)
        self._agent = (0, 0)
        return self._to_obs(), {"agent": self._agent, "goal": self._goal}

    def step(self, action: np.intp) -> StepResult:
        if action not in self.action_space:
            raise ValueError(f"Invalid action {action!r}")

        dr, dc = _DELTA[int(action)]
        new_row = int(np.clip(self._agent[0] + dr, 0, self.size - 1))
        new_col = int(np.clip(self._agent[1] + dc, 0, self.size - 1))
        self._agent = (new_row, new_col)

        terminated = self._agent == self._goal
        reward = 1.0 if terminated else -0.01
        obs = self._to_obs()
        info: dict[str, Any] = {"agent": self._agent, "goal": self._goal}
        return obs, reward, terminated, False, info
