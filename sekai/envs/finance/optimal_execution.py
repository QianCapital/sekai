from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from sekai._types import SeedType
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.envs._utils import flat_action
from sekai.envs.finance._common import check_number, episode_over
from sekai.error import InvalidAction, ResetRequired
from sekai.spaces.box import Box


class OptimalExecutionEnv(Env[NDArray[np.float32], NDArray[np.float32]]):
    """Liquidate a large position over a fixed horizon, trading off impact against risk.

    The Almgren & Chriss (2000) execution model, with defaults from the
    paper's example (1M shares of a $50 stock over 5 days). Each step the
    agent sells `n` shares, executed at `S - epsilon - eta * n / tau` (temporary
    impact); the mid then moves by `sigma * sqrt(tau) * Z - gamma * n`
    (volatility plus permanent impact).

    - **Observation**: `Box` of shape `(3,)`: remaining inventory / initial
      inventory, fraction of the horizon remaining, and the mid's move since
      arrival in units of `sigma * sqrt(total_time)`.
    - **Action**: `Box(0, 1, (1,))`, the fraction of the *remaining* inventory to
      sell this step. Whatever is left is sold at the final step regardless.
    - **Reward**, in basis points of the arrival notional `X0 * S0`:
      `1e4 * (n * (execution_price - S0) - penalty) / (X0 * S0)`, where
      `penalty = risk_aversion * sigma**2 * tau * inventory_after**2`. Summed over an
      episode this is minus (implementation shortfall + `risk_aversion` times the
      shortfall's variance proxy), the Almgren-Chriss mean-variance objective.
    - **Termination**: inventory fully sold, or the horizon reached.
    - **info**: `shares_sold`, `execution_price`, `inventory`, `mid_price`,
      `implementation_shortfall` (dollars, cumulative), `shortfall_bps`, `timestamp`
      (time since arrival, in units of `total_time`) and `dt` (the period `tau`).

    `almgren_chriss_actions()` returns the closed-form optimal schedule as a
    sequence of actions; with `risk_aversion=0` it is TWAP. `step()` after the
    episode has ended raises `ResetRequired`.

    Raises:
        ValueError: For a parameter out of range: `initial_inventory`, `s0`,
            `total_time` and `temporary_impact` must be positive (the latter above
            `permanent_impact * tau / 2`), `sigma`, `permanent_impact`, `fixed_cost`
            and `risk_aversion` non-negative, and `n_steps` >= 1.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_fps": 5}

    def __init__(
        self,
        *,
        initial_inventory: float = 1e6,
        s0: float = 50.0,
        sigma: float = 0.95,
        total_time: float = 5.0,
        n_steps: int = 25,
        permanent_impact: float = 2.5e-7,
        temporary_impact: float = 2.5e-6,
        fixed_cost: float = 0.0625,
        risk_aversion: float = 1e-6,
    ) -> None:
        if n_steps < 1:
            raise ValueError("n_steps must be >= 1.")
        self.X0 = check_number("initial_inventory", initial_inventory, ">0")
        self.s0 = check_number("s0", s0, ">0")
        self.sigma = check_number("sigma", sigma, ">=0")
        self.total_time = check_number("total_time", total_time, ">0")
        self.n_steps = n_steps
        self.tau = self.total_time / n_steps
        self.gamma = check_number("permanent_impact", permanent_impact, ">=0")
        self.eta = check_number("temporary_impact", temporary_impact, ">0")
        self.epsilon = check_number("fixed_cost", fixed_cost, ">=0")
        self.risk_aversion = check_number("risk_aversion", risk_aversion, ">=0")
        if self.eta - 0.5 * self.gamma * self.tau <= 0:
            raise ValueError("temporary_impact must exceed permanent_impact * tau / 2.")

        self.observation_space = Box(
            np.array([0.0, 0.0, -np.inf], dtype=np.float32),
            np.array([1.0, 1.0, np.inf], dtype=np.float32),
        )
        self.action_space = Box(0.0, 1.0, shape=(1,), dtype=np.float32)

        self._started = False
        self._over = False
        self._k = 0
        self.inventory = self.X0
        self.mid = s0
        self.shortfall = 0.0

    @property
    def _notional(self) -> float:
        return self.X0 * self.s0

    def almgren_chriss_trajectory(self) -> NDArray[np.float64]:
        """Optimal holdings x_0..x_N (x_0 = X0, x_N = 0) for this env's parameters."""
        t = np.arange(self.n_steps + 1, dtype=np.float64) * self.tau
        eta_tilde = self.eta - 0.5 * self.gamma * self.tau
        kappa_tilde_sq = self.risk_aversion * self.sigma**2 / eta_tilde
        # kappa solves cosh(kappa tau) = 1 + kappa_tilde^2 tau^2 / 2. Written with asinh, it
        # does not round to 0 for a tiny kappa_tilde as acosh(1 + x) does.
        kappa = (
            2.0 * math.asinh(0.5 * self.tau * math.sqrt(kappa_tilde_sq)) / self.tau
            if kappa_tilde_sq > 0
            else 0.0
        )
        if kappa * self.total_time < 1e-8:  # risk neutral, or close enough that TWAP is exact
            linear: NDArray[np.float64] = self.X0 * (1 - t / self.total_time)
            return linear
        # sinh(a) / sinh(b) as exp(a - b) * expm1(-2a) / expm1(-2b): the same value, but
        # finite however large kappa * T is (sinh itself overflows beyond about 710).
        a = kappa * (self.total_time - t)
        b = np.float64(kappa * self.total_time)
        ratio = np.exp(a - b) * (np.expm1(-2.0 * a) / np.expm1(-2.0 * b))  # exactly 1 at t = 0
        holdings: NDArray[np.float64] = self.X0 * ratio
        return holdings

    def almgren_chriss_actions(self) -> NDArray[np.float32]:
        """The optimal schedule as actions (fraction of remaining inventory per step)."""
        x = self.almgren_chriss_trajectory()
        trades = x[:-1] - x[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            fractions = np.where(x[:-1] > 0, trades / x[:-1], 1.0)
        return np.clip(fractions, 0.0, 1.0).astype(np.float32).reshape(-1, 1)

    def expected_shortfall(self, trades: NDArray[Any]) -> tuple[float, float]:
        """Closed-form mean and variance of the implementation shortfall for a static schedule."""
        n = np.asarray(trades, dtype=np.float64)
        x_after = self.X0 - np.cumsum(n)
        eta_tilde = self.eta - 0.5 * self.gamma * self.tau
        mean = (
            0.5 * self.gamma * self.X0**2
            + self.epsilon * np.abs(n).sum()
            + eta_tilde / self.tau * np.square(n).sum()
        )
        var = self.sigma**2 * self.tau * np.square(x_after).sum()
        return float(mean), float(var)

    @property
    def elapsed_steps(self) -> int:
        """Steps taken in the current episode."""
        return self._k

    def reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[NDArray[np.float32]]:
        self._set_rng(seed)
        self._started = True
        self._over = False
        self._k = 0
        self.inventory = self.X0
        self.mid = self.s0
        self.shortfall = 0.0
        return ResetResult(observation=self._get_obs(), info=self._info(0.0, math.nan))

    def step(self, action: NDArray[np.float32]) -> StepResult[NDArray[np.float32]]:
        if not self._started:
            raise ResetRequired()
        if self._over:
            raise episode_over()
        a = flat_action(action).astype(np.float64)
        if a.shape != (1,) or not np.isfinite(a[0]):
            raise InvalidAction(action, self.action_space)
        fraction = float(np.clip(a[0], 0.0, 1.0))

        last_step = self._k == self.n_steps - 1
        shares = self.inventory if last_step else fraction * self.inventory
        exec_price = self.mid
        if shares > 0:
            exec_price = self.mid - self.epsilon - self.eta * shares / self.tau
        self.inventory -= shares
        if self.inventory < 1e-9 * self.X0:
            self.inventory = 0.0
        self.shortfall += shares * (self.s0 - exec_price)

        self.mid += self.sigma * math.sqrt(self.tau) * float(self.rng.standard_normal())
        self.mid -= self.gamma * shares
        self._k += 1

        risk = self.risk_aversion * self.sigma**2 * self.tau * self.inventory**2
        reward = 1e4 * (shares * (exec_price - self.s0) - risk) / self._notional
        self._over = self.inventory == 0.0 or self._k >= self.n_steps
        return StepResult(
            observation=self._get_obs(),
            reward=float(reward),
            terminated=self._over,
            truncated=False,
            info=self._info(shares, exec_price),
        )

    def _get_obs(self) -> NDArray[np.float32]:
        scale = self.sigma * math.sqrt(self.total_time)
        return np.array(
            [
                self.inventory / self.X0,
                (self.n_steps - self._k) / self.n_steps,
                (self.mid - self.s0) / scale if scale > 0 else 0.0,
            ],
            dtype=np.float32,
        )

    def _info(self, shares: float, exec_price: float) -> dict[str, Any]:
        return {
            "shares_sold": shares,
            "execution_price": exec_price,
            "inventory": self.inventory,
            "mid_price": self.mid,
            "implementation_shortfall": self.shortfall,
            "shortfall_bps": 1e4 * self.shortfall / self._notional,
            "timestamp": self._k * self.tau,
            "dt": self.tau,
        }
