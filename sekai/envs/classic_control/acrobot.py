from __future__ import annotations

from math import cos, pi, sin
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


def wrap(x: float, m: float, M: float) -> float:
    """Wrap x into [m, M] (e.g. an angle into [-pi, pi])."""
    diff = M - m
    while x > M:
        x = x - diff
    while x < m:
        x = x + diff
    return x


def bound(x: float, m: float, M: float) -> float:
    return min(max(x, m), M)


class AcrobotEnv(Env[NDArray[np.float32], np.intp]):
    """Swing the tip of a two-link pendulum above a line by torquing the middle joint.

    Sutton (1996); dynamics identical to gymnasium's Acrobot-v1 ("book"
    equations, RK4 integration with dt=0.2).

    Observation (Box, shape (6,)): cos/sin of both joint angles and both
    angular velocities (in [-4pi, 4pi] and [-9pi, 9pi]).

    Action (Discrete(3)): torque -1, 0 or +1 on the actuated joint.

    Reward: -1 per step, 0 on the terminating step.

    Start: every state variable uniform in [-0.1, 0.1].

    Termination: -cos(theta1) - cos(theta1 + theta2) > 1.
    Truncation: registered as Acrobot-v1 with max_episode_steps=500.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_fps": 15}

    dt = 0.2
    LINK_LENGTH_1 = 1.0
    LINK_LENGTH_2 = 1.0
    LINK_MASS_1 = 1.0
    LINK_MASS_2 = 1.0
    LINK_COM_POS_1 = 0.5
    LINK_COM_POS_2 = 0.5
    LINK_MOI = 1.0
    MAX_VEL_1 = 4 * pi
    MAX_VEL_2 = 9 * pi
    AVAIL_TORQUE = (-1.0, 0.0, +1.0)
    torque_noise_max = 0.0
    book_or_nips = "book"

    def __init__(self) -> None:
        high = np.array([1.0, 1.0, 1.0, 1.0, self.MAX_VEL_1, self.MAX_VEL_2], dtype=np.float32)
        self.observation_space = Box(-high, high, dtype=np.float32)
        self.action_space = Discrete(3)
        self.state: NDArray[Any] | None = None

    def reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[NDArray[np.float32]]:
        self._set_rng(seed)
        low, high = parse_reset_bounds(options, -0.1, 0.1)
        self.state = self.rng.uniform(low=low, high=high, size=(4,)).astype(np.float32)
        return ResetResult(observation=self._get_obs(), info={})

    def step(self, action: int | np.integer[Any]) -> StepResult[NDArray[np.float32]]:
        check_action(self.action_space, action)
        if self.state is None:
            raise ResetRequired()

        torque = self.AVAIL_TORQUE[int(action)]
        if self.torque_noise_max > 0:
            torque += self.rng.uniform(-self.torque_noise_max, self.torque_noise_max)

        ns = self._rk4(np.append(self.state, torque))
        ns[0] = wrap(ns[0], -pi, pi)
        ns[1] = wrap(ns[1], -pi, pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns

        terminated = self._terminal()
        return StepResult(
            observation=self._get_obs(),
            reward=0.0 if terminated else -1.0,
            terminated=terminated,
            truncated=False,
            info={},
        )

    def _get_obs(self) -> NDArray[np.float32]:
        s = self.state
        assert s is not None
        return np.array([cos(s[0]), sin(s[0]), cos(s[1]), sin(s[1]), s[2], s[3]], dtype=np.float32)

    def _terminal(self) -> bool:
        s = self.state
        assert s is not None
        return bool(-cos(s[0]) - cos(s[1] + s[0]) > 1.0)

    def _rk4(self, y0: NDArray[Any]) -> NDArray[np.float64]:
        """One 4th-order Runge-Kutta step of length dt; returns the 4 state variables."""
        y = np.asarray(y0, dtype=np.float64)
        dt = self.dt
        dt2 = dt / 2.0
        k1 = np.asarray(self._dsdt(y))
        k2 = np.asarray(self._dsdt(y + dt2 * k1))
        k3 = np.asarray(self._dsdt(y + dt2 * k2))
        k4 = np.asarray(self._dsdt(y + dt * k3))
        out: NDArray[np.float64] = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        return out[:4]

    def _dsdt(self, s_augmented: NDArray[Any]) -> tuple[float, float, float, float, float]:
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        lc1 = self.LINK_COM_POS_1
        lc2 = self.LINK_COM_POS_2
        I1 = self.LINK_MOI
        I2 = self.LINK_MOI
        g = 9.8
        a = s_augmented[-1]
        theta1, theta2, dtheta1, dtheta2 = s_augmented[:-1]

        d1 = m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * cos(theta2)) + I1 + I2
        d2 = m2 * (lc2**2 + l1 * lc2 * cos(theta2)) + I2
        phi2 = m2 * lc2 * g * cos(theta1 + theta2 - pi / 2.0)
        phi1 = (
            -m2 * l1 * lc2 * dtheta2**2 * sin(theta2)
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)
            + (m1 * lc1 + m2 * l1) * g * cos(theta1 - pi / 2)
            + phi2
        )
        if self.book_or_nips == "nips":
            ddtheta2 = (a + d2 / d1 * phi1 - phi2) / (m2 * lc2**2 + I2 - d2**2 / d1)
        else:
            ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1**2 * sin(theta2) - phi2) / (
                m2 * lc2**2 + I2 - d2**2 / d1
            )
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0
