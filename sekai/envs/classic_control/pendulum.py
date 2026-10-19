from __future__ import annotations

import numbers
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from sekai._types import SeedType
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.envs._utils import flat_action
from sekai.error import ResetRequired
from sekai.spaces.box import Box

DEFAULT_X = np.pi
DEFAULT_Y = 1.0


def angle_normalize(x: Any) -> Any:
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class PendulumEnv(Env[NDArray[np.float32], NDArray[np.float32]]):
    """Swing a pendulum up and keep it upright with limited torque.

    Dynamics identical to gymnasium's Pendulum-v1.

    Observation (Box, shape (3,)): cos(theta), sin(theta), angular velocity in [-8, 8].

    Action (Box(-2, 2, (1,))): torque.

    Reward: `-(theta**2 + 0.1 * theta_dot**2 + 0.001 * torque**2)`, theta
    normalised to [-pi, pi] with 0 upright. Range [-16.27, 0].

    Start: theta uniform in [-pi, pi], theta_dot uniform in [-1, 1]; override
    with reset(options={"x_init": ..., "y_init": ...}) (symmetric bounds).

    No termination. Truncation: registered as Pendulum-v1 with max_episode_steps=200.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_fps": 30}

    def __init__(self, g: float = 10.0) -> None:
        self.max_speed = 8.0
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.l = 1.0

        high = np.array([1.0, 1.0, self.max_speed], dtype=np.float32)
        self.action_space = Box(-self.max_torque, self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = Box(-high, high, dtype=np.float32)
        self.state: NDArray[np.float64] | None = None
        self.last_u: float | None = None

    def reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[NDArray[np.float32]]:
        self._set_rng(seed)
        x = DEFAULT_X if options is None else options.get("x_init", DEFAULT_X)
        y = DEFAULT_Y if options is None else options.get("y_init", DEFAULT_Y)
        for name, value in (("x_init", x), ("y_init", y)):
            if not isinstance(value, numbers.Real):
                raise TypeError(f"reset option {name!r} must be a real number, got {value!r}.")
        high = np.array([x, y], dtype=np.float64)
        self.state = self.rng.uniform(low=-high, high=high)
        self.last_u = None
        return ResetResult(observation=self._get_obs(), info={})

    def step(self, action: NDArray[np.float32]) -> StepResult[NDArray[np.float32]]:
        if self.state is None:
            raise ResetRequired()
        th, thdot = self.state
        g, m, l, dt = self.g, self.m, self.l, self.dt  # noqa: E741

        u = np.clip(flat_action(action), -self.max_torque, self.max_torque)[0]
        self.last_u = float(u)
        costs = angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * (u**2)

        newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 / (m * l**2) * u) * dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)
        newth = th + newthdot * dt

        self.state = np.array([newth, newthdot])
        return StepResult(
            observation=self._get_obs(),
            reward=float(-costs),
            terminated=False,
            truncated=False,
            info={},
        )

    def _get_obs(self) -> NDArray[np.float32]:
        assert self.state is not None
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)
