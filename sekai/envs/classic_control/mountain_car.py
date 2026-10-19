from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from sekai._types import SeedType
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.envs._utils import check_action, parse_reset_bounds
from sekai.error import ResetRequired
from sekai.spaces.box import Box
from sekai.spaces.discrete import Discrete


class MountainCarEnv(Env[NDArray[np.float32], np.intp]):
    """Drive an under-powered car up a hill by building momentum.

    Moore (1990); dynamics identical to gymnasium's MountainCar-v0.

    Observation (Box, shape (2,)): position in [-1.2, 0.6], velocity in [-0.07, 0.07].

    Action (Discrete(3)): 0 accelerate left, 1 no acceleration, 2 accelerate right.

    Reward: -1 per step.

    Start: position uniform in [-0.6, -0.4], velocity 0; override the
    position range with reset(options={"low": ..., "high": ...}).

    Termination: position >= 0.5 (and velocity >= goal_velocity).
    Truncation: registered as MountainCar-v0 with max_episode_steps=200.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_fps": 30}

    def __init__(self, goal_velocity: float = 0.0) -> None:
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.5
        self.goal_velocity = goal_velocity
        self.force = 0.001
        self.gravity = 0.0025

        self.low = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high = np.array([self.max_position, self.max_speed], dtype=np.float32)
        self.observation_space = Box(self.low, self.high, dtype=np.float32)
        self.action_space = Discrete(3)
        self.state: NDArray[np.float64] | None = None

    def reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[NDArray[np.float32]]:
        self._set_rng(seed)
        low, high = parse_reset_bounds(options, -0.6, -0.4)
        self.state = np.array([self.rng.uniform(low=low, high=high), 0.0])
        return ResetResult(observation=self.state.astype(np.float32), info={})

    def step(self, action: int | np.integer[Any]) -> StepResult[NDArray[np.float32]]:
        check_action(self.action_space, action)
        if self.state is None:
            raise ResetRequired()

        position, velocity = self.state
        velocity += (int(action) - 1) * self.force + math.cos(3 * position) * (-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0.0

        terminated = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        self.state = np.array([position, velocity], dtype=np.float64)
        return StepResult(
            observation=self.state.astype(np.float32),
            reward=-1.0,
            terminated=terminated,
            truncated=False,
            info={},
        )

    @staticmethod
    def height(xs: Any) -> Any:
        """Height of the track at position xs (used for rendering)."""
        return np.sin(3 * xs) * 0.45 + 0.55
