"""Classic CartPole environment implemented with the sekai API.

A pole is attached by an un-actuated joint to a cart that moves along a
frictionless track.  The pendulum starts upright and the goal is to prevent
it from falling over by applying forces of +1 or -1 to the cart.

This implementation mirrors the classic control benchmark for comparison
purposes.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from sekai.core.env import Env
from sekai.spaces.box import Box
from sekai.spaces.discrete import Discrete
from sekai.utils.typing import ResetResult, StepResult


class CartPoleEnv(Env[np.ndarray, np.intp]):
    """CartPole environment.

    Observation
    -----------
    +-------+------+----------------------+
    | Index | Name | Range                |
    +-------+------+----------------------+
    | 0     | x    | [-4.8, 4.8]          |
    | 1     | ẋ    | (-∞, ∞)              |
    | 2     | θ    | [-0.418, 0.418] rad  |
    | 3     | θ̇   | (-∞, ∞)              |
    +-------+------+----------------------+

    Actions
    -------
    0 — push cart to the left
    1 — push cart to the right

    Rewards
    -------
    +1 for every step the pole remains upright.

    Episode end
    -----------
    An episode ends when:
    * the cart position is outside ±2.4
    * the pole angle is outside ±12°

    Parameters
    ----------
    render_mode:
        Currently only ``None`` is supported.
    """

    metadata = {"render_modes": [], "render_fps": 50}
    reward_range = (0.0, 1.0)

    # Physics constants
    GRAVITY = 9.8
    MASS_CART = 1.0
    MASS_POLE = 0.1
    HALF_POLE = 0.5  # half the pole's length
    FORCE_MAG = 10.0
    TAU = 0.02  # seconds between state updates

    X_THRESHOLD = 2.4
    THETA_THRESHOLD_RAD = 12 * 2 * math.pi / 360

    def __init__(self, render_mode: str | None = None) -> None:
        self.render_mode = render_mode
        high = np.array(
            [
                self.X_THRESHOLD * 2,
                np.finfo(np.float32).max,
                self.THETA_THRESHOLD_RAD * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.observation_space = Box(-high, high, dtype=np.float32)
        self.action_space = Discrete(2)
        self._state: np.ndarray | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult:
        if seed is not None:
            self._seed_np_random(seed)
        self._state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)
        return self._state.copy(), {}

    def step(self, action: np.intp) -> StepResult:
        if self._state is None:
            raise RuntimeError("Call reset() before step()")
        if action not in self.action_space:
            raise ValueError(f"Invalid action {action!r}")

        x, x_dot, theta, theta_dot = self._state
        force = self.FORCE_MAG if action == 1 else -self.FORCE_MAG

        total_mass = self.MASS_CART + self.MASS_POLE
        pole_mass_length = self.MASS_POLE * self.HALF_POLE
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        tmp = (force + pole_mass_length * theta_dot**2 * sin_theta) / total_mass
        theta_acc = (self.GRAVITY * sin_theta - cos_theta * tmp) / (
            self.HALF_POLE * (4.0 / 3.0 - self.MASS_POLE * cos_theta**2 / total_mass)
        )
        x_acc = tmp - pole_mass_length * theta_acc * cos_theta / total_mass

        # Euler integration
        x += self.TAU * x_dot
        x_dot += self.TAU * x_acc
        theta += self.TAU * theta_dot
        theta_dot += self.TAU * theta_acc

        self._state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        terminated = bool(
            x < -self.X_THRESHOLD
            or x > self.X_THRESHOLD
            or theta < -self.THETA_THRESHOLD_RAD
            or theta > self.THETA_THRESHOLD_RAD
        )
        reward = 1.0 if not terminated else 0.0
        return self._state.copy(), reward, terminated, False, {}
