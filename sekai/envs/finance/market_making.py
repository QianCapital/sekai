from __future__ import annotations

import math
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from sekai._types import SeedType
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.envs._utils import flat_action
from sekai.error import InvalidAction, ResetRequired
from sekai.spaces.box import Box


class MarketMakingEnv(Env[NDArray[np.float32], NDArray[np.float32]]):
    """Quote a bid and an ask around a moving mid price and manage inventory risk.

    The Avellaneda & Stoikov (2008) market-making model. The mid price follows
    arithmetic Brownian motion, `S <- S + sigma * sqrt(dt) * Z`. A quote at
    distance `delta` from the mid is filled within a step with probability
    `1 - exp(-A * exp(-k * delta) * dt)`, independently on each side; each fill
    is `order_size` units at the quoted price.

    - **Observation**: `Box` of shape `(2,)`: inventory / `max_inventory` in
      [-1, 1] and the fraction of the session remaining in [0, 1]. This is the
      Markov state of the model (the price level does not affect optimal quotes).
    - **Action**: `Box(0, max_offset, (2,))`, the bid and ask distances from the
      mid. A side is not quoted while inventory is at its limit on that side.
    - **Reward**: change in mark-to-market wealth (`cash + inventory * mid`) minus
      a running inventory penalty
      `0.5 * risk_aversion * sigma**2 * dt * inventory**2`.
    - **Termination**: at the end of the session (`n_steps` steps). The time
      remaining is observed, so this is a true terminal state, not a truncation.
    - **info**: `mid_price`, `inventory`, `cash`, `wealth`, `bid`, `ask`,
      `bid_filled`, `ask_filled`, `timestamp` (session time, in units of `total_time`)
      and `dt`.

    `avellaneda_stoikov_action()` returns the model's closed-form optimal quotes,
    a strong baseline to compare learned policies against.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_fps": 20}

    def __init__(
        self,
        *,
        s0: float = 100.0,
        sigma: float = 2.0,
        total_time: float = 1.0,
        n_steps: int = 200,
        fill_intensity: float = 140.0,
        fill_decay: float = 1.5,
        risk_aversion: float = 0.1,
        max_inventory: int = 50,
        order_size: int = 1,
        max_offset: float = 5.0,
    ) -> None:
        if n_steps < 1 or max_inventory < 1 or order_size < 1:
            raise ValueError("n_steps, max_inventory and order_size must be >= 1.")
        if sigma < 0 or fill_intensity <= 0 or fill_decay <= 0 or risk_aversion < 0:
            raise ValueError("sigma, risk_aversion must be >= 0 and fill parameters > 0.")
        self.s0 = s0
        self.sigma = sigma
        self.total_time = total_time
        self.n_steps = n_steps
        self.dt = total_time / n_steps
        self.A = fill_intensity
        self.k = fill_decay
        self.risk_aversion = risk_aversion
        self.max_inventory = max_inventory
        self.order_size = order_size
        self.max_offset = max_offset

        self.observation_space = Box(
            np.array([-1.0, 0.0], dtype=np.float32), np.array([1.0, 1.0], dtype=np.float32)
        )
        self.action_space = Box(0.0, max_offset, shape=(2,), dtype=np.float32)

        self._started = False
        self._t = 0
        self.mid = s0
        self.inventory = 0
        self.cash = 0.0

    @property
    def wealth(self) -> float:
        """Mark-to-market wealth: cash plus inventory valued at the mid."""
        return self.cash + self.inventory * self.mid

    @property
    def time_remaining(self) -> float:
        return (self.n_steps - self._t) * self.dt

    def fill_probability(self, offset: float) -> float:
        """Probability that a quote `offset` away from the mid fills within one step."""
        return 1.0 - math.exp(-self.A * math.exp(-self.k * offset) * self.dt)

    def avellaneda_stoikov_action(self) -> NDArray[np.float32]:
        """The model's closed-form optimal (bid, ask) distances in the current state."""
        gamma, tau = self.risk_aversion, self.time_remaining
        skew = self.inventory * gamma * self.sigma**2 * tau
        if gamma > 0:
            spread = gamma * self.sigma**2 * tau + (2 / gamma) * math.log(1 + gamma / self.k)
        else:
            spread = 2 / self.k
        bid, ask = spread / 2 + skew, spread / 2 - skew
        quotes: NDArray[np.float32] = np.clip([bid, ask], 0.0, self.max_offset).astype(np.float32)
        return quotes

    @property
    def elapsed_steps(self) -> int:
        """Steps taken in the current episode."""
        return self._t

    def reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[NDArray[np.float32]]:
        self._set_rng(seed)
        self._started = True
        self._t = 0
        self.mid = self.s0
        self.inventory = 0
        self.cash = 0.0
        return ResetResult(
            observation=self._get_obs(), info=self._info(math.nan, math.nan, False, False)
        )

    def step(self, action: NDArray[np.float32]) -> StepResult[NDArray[np.float32]]:
        if not self._started:
            raise ResetRequired()
        a = flat_action(action).astype(np.float64)
        if a.shape != (2,) or not np.all(np.isfinite(a)):
            raise InvalidAction(action, self.action_space)
        bid_offset, ask_offset = np.clip(a, 0.0, self.max_offset)
        bid, ask = self.mid - bid_offset, self.mid + ask_offset
        wealth_before = self.wealth

        can_buy = self.inventory + self.order_size <= self.max_inventory
        can_sell = self.inventory - self.order_size >= -self.max_inventory
        u_bid, u_ask = self.rng.random(2)
        bid_filled = bool(can_buy and u_bid < self.fill_probability(bid_offset))
        ask_filled = bool(can_sell and u_ask < self.fill_probability(ask_offset))
        if bid_filled:
            self.inventory += self.order_size
            self.cash -= self.order_size * bid
        if ask_filled:
            self.inventory -= self.order_size
            self.cash += self.order_size * ask

        self.mid += self.sigma * math.sqrt(self.dt) * float(self.rng.standard_normal())
        self._t += 1

        penalty = 0.5 * self.risk_aversion * self.sigma**2 * self.dt * self.inventory**2
        reward = self.wealth - wealth_before - penalty
        return StepResult(
            observation=self._get_obs(),
            reward=float(reward),
            terminated=self._t >= self.n_steps,
            truncated=False,
            info=self._info(bid, ask, bid_filled, ask_filled),
        )

    def _get_obs(self) -> NDArray[np.float32]:
        return np.array(
            [self.inventory / self.max_inventory, (self.n_steps - self._t) / self.n_steps],
            dtype=np.float32,
        )

    def _info(self, bid: float, ask: float, bid_filled: bool, ask_filled: bool) -> dict[str, Any]:
        return {
            "mid_price": self.mid,
            "inventory": self.inventory,
            "cash": self.cash,
            "wealth": self.wealth,
            "bid": float(bid),
            "ask": float(ask),
            "bid_filled": bid_filled,
            "ask_filled": ask_filled,
            "timestamp": self._t * self.dt,
            "dt": self.dt,
        }
