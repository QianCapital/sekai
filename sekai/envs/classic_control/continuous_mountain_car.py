from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from sekai._types import SeedType
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.envs._utils import flat_action, parse_reset_bounds
from sekai.error import ResetRequired
from sekai.spaces.box import Box


class ContinuousMountainCarEnv(Env[NDArray[np.float32], NDArray[np.float32]]):
    """Mountain car with a continuous force action.

    Dynamics identical to gymnasium's MountainCarContinuous-v0 (including its
    float32 state after the first step).

    Observation (Box, shape (2,)): position in [-1.2, 0.6], velocity in [-0.07, 0.07].

    Action (Box(-1, 1, (1,))): force, clipped to [-1, 1] and scaled by power=0.0015.

    Reward: `-0.1 * action**2` per step (unclipped action), +100 on reaching the goal.

    Start: position uniform in [-0.6, -0.4], velocity 0.

    Termination: position >= 0.45 (and velocity >= goal_velocity).
    Truncation: registered as MountainCarContinuous-v0 with max_episode_steps=999.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_fps": 30}

    def __init__(self, goal_velocity: float = 0.0) -> None:
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = 0.45
        self.goal_velocity = goal_velocity
        self.power = 0.0015

        self.low_state = np.array([self.min_position, -self.max_speed], dtype=np.float32)
        self.high_state = np.array([self.max_position, self.max_speed], dtype=np.float32)
        self.action_space = Box(self.min_action, self.max_action, shape=(1,), dtype=np.float32)
        self.observation_space = Box(self.low_state, self.high_state, dtype=np.float32)
        self.state: NDArray[Any] | None = None

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

    def step(self, action: NDArray[np.float32]) -> StepResult[NDArray[np.float32]]:
        if self.state is None:
            raise ResetRequired()
        a = flat_action(action)[0]

        position = self.state[0]
        velocity = self.state[1]
        force = min(max(a, self.min_action), self.max_action)

        velocity += force * self.power - 0.0025 * math.cos(3 * position)
        velocity = min(max(velocity, -self.max_speed), self.max_speed)
        position += velocity
        position = min(max(position, self.min_position), self.max_position)
        if position == self.min_position and velocity < 0:
            velocity = 0

        terminated = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        reward = (100.0 if terminated else 0.0) - math.pow(float(a), 2) * 0.1

        self.state = np.array([position, velocity], dtype=np.float32)
        return StepResult(
            observation=self.state.copy(),
            reward=reward,
            terminated=terminated,
            truncated=False,
            info={},
        )

    @staticmethod
    def height(xs: Any) -> Any:
        """Height of the track at position xs (used for rendering)."""
        return np.sin(3 * xs) * 0.45 + 0.55
