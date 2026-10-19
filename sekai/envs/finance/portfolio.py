from __future__ import annotations

import copy
from typing import Any, ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sekai._types import SeedType
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.envs._utils import flat_action
from sekai.envs.finance._common import History, check_cost_inputs, field_or_none
from sekai.envs.finance._common import step_length as _step_length
from sekai.envs.finance.market import random_correlation, simulate_gbm
from sekai.error import InvalidAction, ResetRequired
from sekai.finance.costs import CostModel, as_cost_model
from sekai.finance.data import MarketData
from sekai.finance.rewards import LogReturn, RewardFunction, StepOutcome
from sekai.finance.sampling import EpisodeSampler
from sekai.finance.simulators import MarketModel
from sekai.spaces.box import Box
from sekai.spaces.dict import Dict


class PortfolioEnv(Env[dict[str, NDArray[np.float32]], NDArray[np.float32]]):
    """Allocate wealth across several assets and cash, rebalancing every step.

    Prices are simulated (correlated geometric Brownian motions by default, or
    any `model` from `sekai.finance.simulators`, e.g. regime switching or
    stochastic volatility), or drawn as windows from historical `prices` of
    shape `(T, n_assets)` or a `MarketData` with a complete panel. For
    universes whose members change over time, use `Market-v0` instead.

    - **Observation**: `Dict` with `"returns"`, a `Box` of shape
      `(lookback, n_assets)` holding recent log returns per asset, and
      `"weights"`, a `Box` of shape `(n_assets + 1,)` holding the current weights
      after price drift, cash last.
    - **Action**, long-only (default): `Box(0, 1, (n_assets + 1,))` of non-negative
      scores, normalised to target weights summing to 1 (cash last); all zeros
      means all cash.
    - **Action**, with `allow_short=True`: `Box(-1, 1, (n_assets,))` of target asset
      weights, scaled down if gross exposure `sum(|w|)` exceeds 1; the rest is cash.
    - **Reward**: by default the log growth of portfolio value net of transaction
      costs, where `cost = transaction_cost * turnover` (or any `CostModel`) and
      `turnover = sum(|target - drifted weights|)`. Pass `reward=` for another
      objective.
    - **Truncation**: after `episode_length` steps (an episode is a window of an
      ongoing market, so the observation holds no clock).
    - **info**: `portfolio_value` (also as `equity`), `weights`, `turnover`, `cost`,
      `asset_returns`, `timestamp` and `dt` (step length in years: from dated
      MarketData, else the `model`'s dt, else 1/252, unless `dt=` is given).

    With a history, `reset(options={"start": t})` places the first decision at
    timestamp (or row) `t`, and `sampler` chooses windows otherwise (see
    TradingEnv).
    """

    metadata: ClassVar[dict[str, Any]] = {"render_fps": 10}

    def __init__(
        self,
        prices: ArrayLike | MarketData | None = None,
        *,
        n_assets: int = 5,
        episode_length: int = 252,
        lookback: int = 30,
        transaction_cost: float | CostModel = 1e-3,
        allow_short: bool = False,
        mu: ArrayLike | None = None,
        sigma: ArrayLike | None = None,
        corr: ArrayLike | None = None,
        dt: float | None = None,
        model: MarketModel | None = None,
        reward: RewardFunction | None = None,
        sampler: str | int | EpisodeSampler = "random",
        capital: float = 1_000_000.0,
    ) -> None:
        if episode_length < 1 or lookback < 1:
            raise ValueError("episode_length and lookback must be >= 1.")
        self.cost_model = as_cost_model(transaction_cost)
        self._window = lookback + episode_length + 1
        self._history = (
            None
            if prices is None
            else History(prices, ndim=2, window=self._window, lookback=lookback, sampler=sampler)
        )
        self.prices: NDArray[np.float64] | None = (
            None if self._history is None else self._history.prices
        )
        self.data: MarketData | None = None if self._history is None else self._history.data
        self.dt = (
            float(model.dt)
            if dt is None and self._history is None and model is not None
            else _step_length(dt, self._history)
        )
        check_cost_inputs(
            self.cost_model, self.data, simulated=self._history is None, lookback=lookback
        )
        if self.prices is not None:
            n_assets = self.prices.shape[1]
        elif model is not None:
            n_assets = model.n_assets
        if n_assets < 1:
            raise ValueError("n_assets must be >= 1.")
        self.model = model
        # A copy: stateful rewards must not be shared between envs built from one kwargs dict.
        self.reward_fn: RewardFunction = (
            copy.deepcopy(reward) if reward is not None else LogReturn()
        )
        self.capital = float(capital)

        self.n_assets = n_assets
        self._episode_length = episode_length
        self._lookback = lookback
        self.transaction_cost = transaction_cost
        self.allow_short = allow_short
        # Simulation parameters. A fixed default correlation keeps the market the
        # same across resets; only the path is random.
        self.mu = np.full(n_assets, 0.06) if mu is None else np.asarray(mu, dtype=np.float64)
        self.sigma = (
            np.linspace(0.15, 0.35, n_assets)
            if sigma is None
            else np.asarray(sigma, dtype=np.float64)
        )
        self.corr = (
            random_correlation(np.random.default_rng(0), n_assets)
            if corr is None
            else np.asarray(corr, dtype=np.float64)
        )

        # Long-only weights stay in [0, 1]. With shorts, price moves can drift the
        # weights arbitrarily far from the [-1, 1] targets, so they are unbounded.
        w_low, w_high = (-np.inf, np.inf) if allow_short else (0.0, 1.0)
        self.observation_space = Dict(
            {
                "returns": Box(-np.inf, np.inf, shape=(lookback, n_assets), dtype=np.float32),
                "weights": Box(w_low, w_high, shape=(n_assets + 1,), dtype=np.float32),
            }
        )
        if allow_short:
            self.action_space = Box(-1.0, 1.0, shape=(n_assets,), dtype=np.float32)
        else:
            self.action_space = Box(0.0, 1.0, shape=(n_assets + 1,), dtype=np.float32)

        self._path: NDArray[np.float64] | None = None
        self._log_returns: NDArray[np.float64] = np.zeros((0, n_assets))
        self._start = 0
        self._t = 0
        self.weights = self._cash_only()
        self.portfolio_value = 1.0

    def _cash_only(self) -> NDArray[np.float64]:
        w = np.zeros(self.n_assets + 1)
        w[-1] = 1.0
        return w

    def action_to_weights(self, action: Any) -> NDArray[np.float64]:
        """Map an action to target weights over (assets..., cash)."""
        a = flat_action(action).astype(np.float64)
        if a.shape != self.action_space.shape or not np.all(np.isfinite(a)):
            raise InvalidAction(action, self.action_space)
        if self.allow_short:
            a = np.clip(a, -1.0, 1.0)
            gross = np.abs(a).sum()
            if gross > 1.0:
                a = a / gross
            return np.append(a, 1.0 - a.sum())
        a = np.clip(a, 0.0, None)
        total = a.sum()
        return self._cash_only() if total < 1e-12 else a / total

    @property
    def episode_length(self) -> int:
        """Steps per episode (read-only: it sizes the price windows)."""
        return self._episode_length

    @episode_length.setter
    def episode_length(self, value: int) -> None:
        raise AttributeError("episode_length is fixed at construction; make a new env.")

    @property
    def lookback(self) -> int:
        """Rows of returns in the observation (read-only: it sizes the observation)."""
        return self._lookback

    @lookback.setter
    def lookback(self, value: int) -> None:
        raise AttributeError("lookback is fixed at construction; make a new env.")

    @property
    def elapsed_steps(self) -> int:
        """Steps taken in the current episode."""
        return self._t

    def reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[dict[str, NDArray[np.float32]]]:
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
        self._log_returns = np.diff(np.log(self._path), axis=0)
        self._t = 0
        self.weights = self._cash_only()
        self.portfolio_value = 1.0
        info = self._info(np.zeros(self.n_assets), 0.0, 0.0)
        return ResetResult(observation=self._get_obs(), info=info)

    def _simulate(self) -> NDArray[np.float64]:
        if self.model is None:
            return simulate_gbm(
                self.rng,
                self._window - 1,
                s0=np.full(self.n_assets, 100.0),
                mu=self.mu,
                sigma=self.sigma,
                dt=self.dt,
                corr=self.corr,
            )
        sim = self.model.simulate(self.rng, self._window - 1)
        if sim.mask is not None and not sim.mask.all():
            raise ValueError(
                "PortfolioEnv needs a model whose assets are always listed; use Market-v0 "
                "for changing universes."
            )
        return np.asarray(sim.prices, dtype=np.float64)

    def _cost_context(self, row: int) -> dict[str, Any]:
        needs = self.cost_model.requires
        ctx: dict[str, Any] = {}
        if "equity" in needs:
            ctx["equity"] = self.capital * self.portfolio_value
        if "volatility" in needs:
            ctx["volatility"] = self._log_returns[self._t : self._t + self.lookback].std(axis=0)
        for name in ("volume", "spread"):
            values = field_or_none(self.data, name) if name in needs else None
            if values is not None and self._path is not None:
                value = values[self._start + row]
                ctx[name] = value * self._path[row] if name == "volume" else value
        return ctx

    def step(self, action: NDArray[np.float32]) -> StepResult[dict[str, NDArray[np.float32]]]:
        if self._path is None:
            raise ResetRequired()
        target = self.action_to_weights(action)
        trades = target[:-1] - self.weights[:-1]
        turnover = float(np.abs(trades).sum())
        i = self.lookback + self._t
        cost = float(self.cost_model(trades, **self._cost_context(i)))

        asset_returns = self._path[i + 1] / self._path[i] - 1.0
        gross_return = float(target[:-1] @ asset_returns)
        growth = (1.0 - cost) * (1.0 + gross_return)
        if cost >= 1.0:  # the rebalance cost the whole account
            growth = 0.0

        # Weights drift with prices until the next rebalance.
        drifted = target * np.append(1.0 + asset_returns, 1.0)
        self.weights = drifted / drifted.sum() if growth > 0 else self._cash_only()
        previous_value, self.portfolio_value = (
            self.portfolio_value,
            max(self.portfolio_value * growth, 0.0),
        )
        self._t += 1

        terminated = growth <= 0.0
        truncated = not terminated and self._t >= self.episode_length
        outcome = StepOutcome.from_growth(
            growth,
            equity=self.portfolio_value,
            turnover=turnover,
            cost=cost,
            previous_equity=previous_value,
        )
        return StepResult(
            observation=self._get_obs(),
            reward=float(self.reward_fn(outcome)),
            terminated=terminated,
            truncated=truncated,
            info=self._info(asset_returns, turnover, cost),
        )

    def _get_obs(self) -> dict[str, NDArray[np.float32]]:
        returns = self._log_returns[self._t : self._t + self.lookback]
        return {
            "returns": returns.astype(np.float32),
            "weights": self.weights.astype(np.float32),
        }

    def _info(
        self, asset_returns: NDArray[np.float64], turnover: float, cost: float
    ) -> dict[str, Any]:
        row = self.lookback + self._t
        timestamp = self._t if self._history is None else self._history.timestamp(self._start + row)
        return {
            "portfolio_value": self.portfolio_value,
            "equity": self.portfolio_value,
            "weights": self.weights.copy(),
            "turnover": turnover,
            "cost": cost,
            "asset_returns": np.asarray(asset_returns).copy(),
            "timestamp": timestamp,
            "dt": self.dt,
        }
