from __future__ import annotations

import math
import warnings
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


class CartPoleEnv(Env[NDArray[np.float32], np.intp]):
    """Balance a pole on a cart by pushing the cart left or right.

    The cart-pole problem of Barto, Sutton & Anderson (1983), with dynamics
    identical to gymnasium's CartPole-v1.

    Observation (Box, shape (4,)): cart position, cart velocity, pole angle
    (rad), pole angular velocity.

    Action (Discrete(2)): 0 pushes the cart left, 1 pushes it right.

    Reward: +1 for every step, including the terminating one. With
    sutton_barto_reward=True: 0 per step and -1 on termination.

    Start: every state variable uniform in [-0.05, 0.05]; override with
    reset(options={"low": ..., "high": ...}).

    Termination: |pole angle| > 12 degrees or |cart position| > 2.4.
    Truncation: registered as CartPole-v1 with max_episode_steps=500.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_fps": 50}

    def __init__(self, sutton_barto_reward: bool = False) -> None:
        self.sutton_barto_reward = sutton_barto_reward
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        high = np.array(
            [self.x_threshold * 2, np.inf, self.theta_threshold_radians * 2, np.inf],
            dtype=np.float32,
        )
        self.observation_space = Box(-high, high, dtype=np.float32)
        self.action_space = Discrete(2)

        self.state: NDArray[np.float64] | None = None
        self._steps_beyond_terminated: int | None = None

    def reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[NDArray[np.float32]]:
        self._set_rng(seed)
        low, high = parse_reset_bounds(options, -0.05, 0.05)
        self.state = self.rng.uniform(low=low, high=high, size=(4,))
        self._steps_beyond_terminated = None
        return ResetResult(observation=self.state.astype(np.float32), info={})

    def step(self, action: int | np.integer[Any]) -> StepResult[NDArray[np.float32]]:
        check_action(self.action_space, action)
        if self.state is None:
            raise ResetRequired()

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        temp = (force + self.polemass_length * np.square(theta_dot) * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * np.square(costheta) / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.array((x, x_dot, theta, theta_dot), dtype=np.float64)

        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 0.0 if self.sutton_barto_reward else 1.0
        elif self._steps_beyond_terminated is None:
            self._steps_beyond_terminated = 0
            reward = -1.0 if self.sutton_barto_reward else 1.0
        else:
            if self._steps_beyond_terminated == 0:
                warnings.warn(
                    "CartPoleEnv.step() was called after the episode terminated. "
                    "Call reset() first; further steps are undefined behaviour.",
                    stacklevel=2,
                )
            self._steps_beyond_terminated += 1
            reward = -1.0 if self.sutton_barto_reward else 0.0

        return StepResult(
            observation=self.state.astype(np.float32),
            reward=reward,
            terminated=terminated,
            truncated=False,
            info={},
        )
