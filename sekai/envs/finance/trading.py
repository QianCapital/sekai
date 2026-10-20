from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sekai._types import SeedType
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.envs._utils import check_action
from sekai.envs.finance._common import History, check_cost_inputs, field_or_none
from sekai.envs.finance._common import step_length as _step_length
from sekai.envs.finance.market import simulate_gbm
from sekai.error import ResetRequired
from sekai.finance.costs import CostModel, as_cost_model
from sekai.finance.data import MarketData
from sekai.finance.rewards import LogReturn, RewardFunction, StepOutcome
from sekai.finance.sampling import EpisodeSampler
from sekai.finance.simulators import MarketModel
from sekai.spaces.box import Box
from sekai.spaces.discrete import Discrete


class TradingEnv(Env[NDArray[np.float32], np.intp]):
    """Trade a single asset by choosing a target position each step.

    Prices are simulated (geometric Brownian motion by default, or any
    single-asset `model` from `sekai.finance.simulators`), or drawn as windows
    from a historical `prices` array or `MarketData` you provide.

    - **Observation**: `Box` of shape `(lookback + 1,)`: the last `lookback` log
      returns followed by the current position (drifted with the price, see below).
    - **Action**: `Discrete(len(positions))`, an index into `positions`, the target
      position as a fraction of equity. The default `(-1, 0, 1)` is short / flat /
      long; use `(0, 1)` for long-only.
    - **Dynamics**: trading from the current position `p` to the target `w` costs
      `cost` (by default `transaction_cost * |w - p|`), then the position earns the
      asset's return `r` and drifts with the price, as a portfolio of the asset and
      cash does: `p <- w (1 + r) / (1 + w r)`. Holding a long (`w = 1`) or flat
      position is free; holding a short or levered one trades the drift back.
    - **Reward**: by default the log growth of equity over the step,
      `log((1 - cost) * (1 + w * asset_return))`, summing to the log of
      final over initial equity. Pass `reward=` (a `sekai.finance.RewardFunction`)
      to optimise something else, e.g. `DifferentialSharpe()`.
    - **Termination**: equity falls to zero or below (only possible with leverage
      or extreme moves). **Truncation**: after `episode_length` steps (an episode
      is a window of an ongoing market, so the observation holds no clock).
    - **info**: `equity`, `position` (drifted), `price`, `asset_return`, `cost`,
      `turnover` (`|w - p|`), `timestamp` (the decision time: a date from
      MarketData, else the row index) and `dt` (step length in years).

    With a history, `reset(options={"start": t})` begins the episode with its
    first decision at timestamp (or row) `t`; otherwise `sampler` picks the
    window: "random" (default), "sequential" (consecutive windows, for
    evaluation), an int (a fixed window) or a `sekai.finance.EpisodeSampler`.

    Args:
        prices: Historical prices: a 1-D array or a one-asset MarketData.
        episode_length: Steps per episode.
        lookback: Log returns in the observation.
        positions: Allowed target positions.
        transaction_cost: A proportional cost rate, or any `sekai.finance.CostModel`.
        mu: Annualised GBM drift for the default simulation.
        sigma: Annualised GBM volatility for the default simulation.
        dt: Step length in years (default: implied by dated MarketData timestamps,
            else the `model`'s dt, else 1/252).
        s0: Initial simulated price.
        model: A one-asset market model to simulate instead of GBM.
        reward: Reward function (default `LogReturn()`).
        sampler: How episodes are placed in the history.
        capital: Account size in currency, for cost models that need it.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_fps": 10}

    def __init__(
        self,
        prices: ArrayLike | MarketData | None = None,
        *,
        episode_length: int = 252,
        lookback: int = 30,
        positions: Sequence[float] = (-1.0, 0.0, 1.0),
        transaction_cost: float | CostModel = 1e-3,
        mu: float = 0.05,
        sigma: float = 0.2,
        dt: float | None = None,
        s0: float = 100.0,
        model: MarketModel | None = None,
        reward: RewardFunction | None = None,
        sampler: str | int | EpisodeSampler = "random",
        capital: float = 1_000_000.0,
    ) -> None:
        if episode_length < 1 or lookback < 1:
            raise ValueError("episode_length and lookback must be >= 1.")
        if len(positions) < 2:
            raise ValueError("positions needs at least two choices.")
        if model is not None and model.n_assets != 1:
            raise ValueError(f"TradingEnv needs a one-asset model, got {model.n_assets} assets.")
        self._episode_length = episode_length
        self._lookback = lookback
        self.positions = np.asarray(positions, dtype=np.float64)
        self.cost_model = as_cost_model(transaction_cost)
        self.transaction_cost = transaction_cost
        self.mu, self.sigma, self.s0 = mu, sigma, s0
        self.model = model
        # A copy: stateful rewards must not be shared between envs built from one kwargs dict.
        self.reward_fn: RewardFunction = (
            copy.deepcopy(reward) if reward is not None else LogReturn()
        )
        self.capital = float(capital)
        self._window = lookback + episode_length + 1
        self._history = (
            None
            if prices is None
            else History(prices, ndim=1, window=self._window, lookback=lookback, sampler=sampler)
        )
        self.prices = None if self._history is None else self._history.prices
        self.dt = (
            float(model.dt)
            if dt is None and self._history is None and model is not None
            else _step_length(dt, self._history)
        )
        self.data: MarketData | None = None if self._history is None else self._history.data
        check_cost_inputs(
            self.cost_model, self.data, simulated=self._history is None, lookback=lookback
        )

        # The drifted position stays in [0, 1] for targets in [0, 1]; a short drifts
        # without bound below 0, and a levered long without bound above 1.
        pos_low = 0.0 if np.min(self.positions) >= 0 else -np.inf
        pos_high = 1.0 if np.max(self.positions) <= 1 else np.inf
        low = np.concatenate([np.full(lookback, -np.inf), [pos_low]])
        high = np.concatenate([np.full(lookback, np.inf), [pos_high]])
        self.observation_space = Box(
            low.astype(np.float32), high.astype(np.float32), dtype=np.float32
        )
        self.action_space = Discrete(len(self.positions))

        self._path: NDArray[np.float64] | None = None
        self._log_returns: NDArray[np.float64] = np.zeros(0)
        self._start = 0
        self._t = 0
        self.position = 0.0
        self.equity = 1.0

    @property
    def episode_length(self) -> int:
        """Steps per episode (read-only: it sizes the price windows)."""
        return self._episode_length

    @episode_length.setter
    def episode_length(self, value: int) -> None:
        raise AttributeError("episode_length is fixed at construction; make a new env.")

    @property
    def lookback(self) -> int:
        """Log returns in the observation (read-only: it sizes the observation)."""
        return self._lookback

    @lookback.setter
    def lookback(self, value: int) -> None:
        raise AttributeError("lookback is fixed at construction; make a new env.")

    @property
    def price(self) -> float:
        """The current asset price."""
        if self._path is None:
            raise ResetRequired()
        return float(self._path[self.lookback + self._t])

    @property
    def elapsed_steps(self) -> int:
        """Steps taken in the current episode."""
        return self._t

    def _simulate(self) -> NDArray[np.float64]:
        if self.model is None:
            return simulate_gbm(
                self.rng, self._window - 1, s0=self.s0, mu=self.mu, sigma=self.sigma, dt=self.dt
            )
        sim = self.model.simulate(self.rng, self._window - 1)
        if sim.mask is not None and not sim.mask.all():
            raise ValueError("TradingEnv needs a model whose asset is always listed.")
        return np.asarray(sim.prices[:, 0], dtype=np.float64)

    def reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[NDArray[np.float32]]:
        self._set_rng(seed)
        self.reward_fn.reset()
        if self._history is None:
            if options and options.get("start") is not None:
                raise ValueError("options['start'] needs a price history (prices=...).")
            self._start = 0
            self._path = self._simulate()
        else:
            self._start = self._history.start(self.rng, options, self.episode_length)
            rows = self._history.prices[self._start : self._start + self._window]
            self._path = rows.astype(np.float64)
        self._log_returns = np.diff(np.log(self._path))
        self._t = 0
        self.position = 0.0
        self.equity = 1.0
        return ResetResult(observation=self._get_obs(), info=self._info(0.0, 0.0, 0.0))

    def _cost_context(self, row: int) -> dict[str, Any]:
        needs = self.cost_model.requires
        ctx: dict[str, Any] = {}
        if "equity" in needs:
            ctx["equity"] = self.capital * self.equity
        if "volatility" in needs:
            window = self._log_returns[self._t : self._t + self.lookback]
            ctx["volatility"] = np.array([window.std()])
        for name, key in (("volume", "volume"), ("spread", "spread")):
            if name in needs:
                values = field_or_none(self.data, key)
                if values is not None:
                    value = values[self._start + row, 0]
                    if name == "volume":
                        value = value * self._path[row] if self._path is not None else value
                    ctx[name] = np.array([value])
        return ctx

    def step(self, action: int | np.integer[Any]) -> StepResult[NDArray[np.float32]]:
        check_action(self.action_space, action)
        if self._path is None:
            raise ResetRequired()

        i = self.lookback + self._t
        target = float(self.positions[int(action)])
        trade = target - self.position
        cost = float(self.cost_model(np.array([trade]), **self._cost_context(i)))

        asset_return = float(self._path[i + 1] / self._path[i] - 1.0)
        growth = (1.0 - cost) * (1.0 + target * asset_return)
        if cost >= 1.0:  # the trade cost the whole account
            growth = 0.0
        # The position drifts with the price until the next trade: the asset's value over
        # the asset's plus the cash's (computed as PortfolioEnv does, so the two agree).
        held = target * (1.0 + asset_return)
        value = held + (1.0 - target)
        self.position = held / value if growth > 0.0 and value > 0.0 else 0.0
        self._t += 1

        terminated = growth <= 0.0
        previous_equity, self.equity = self.equity, max(self.equity * growth, 0.0)
        outcome = StepOutcome.from_growth(
            growth,
            equity=self.equity,
            turnover=abs(trade),
            cost=cost,
            previous_equity=previous_equity,
        )
        reward = float(self.reward_fn(outcome))
        truncated = not terminated and self._t >= self.episode_length
        return StepResult(
            observation=self._get_obs(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=self._info(asset_return, cost, abs(trade)),
        )

    def _get_obs(self) -> NDArray[np.float32]:
        window = self._log_returns[self._t : self._t + self.lookback]
        return np.append(window, self.position).astype(np.float32)

    def _info(self, asset_return: float, cost: float, turnover: float) -> dict[str, Any]:
        row = self.lookback + self._t
        timestamp = self._t if self._history is None else self._history.timestamp(self._start + row)
        return {
            "equity": self.equity,
            "position": self.position,
            "price": self.price,
            "asset_return": asset_return,
            "cost": cost,
            "turnover": turnover,
            "timestamp": timestamp,
            "dt": self.dt,
        }
