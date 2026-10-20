"""Natively vectorised versions of the finance environments.

Each class simulates `num_envs` copies of its single-env counterpart with
array operations: the same parameters, spaces, dynamics, rewards and info
keys, but one step advances every sub-env at once. `sekai.make_vec()` uses
them by default:

```python
vec = sekai.make_vec("Portfolio-v0", num_envs=1024)  # a PortfolioVecEnv
```

They share one random generator across sub-envs, so their simulated paths
differ draw-for-draw from the single envs' (identical in distribution). Each
sub-env samples its windows with `sampler.spawn()` (its own sequential position;
other samplers are shared), so on historical data with a deterministic sampler (a
fixed start, or "sequential") they reproduce the single envs exactly.
`max_episode_steps` adds the time limit `make()` would add.

The parameters (`sigma`, `cost_model`, ...) are those of `template`, a single
env built from the same keyword arguments (so a history is passed as `prices=`
or, as a MarketData, `data=`, and the arguments are validated as the single env
validates them), and are shared by every sub-env: `get_attr` / `set_attr` read and
change them, which takes effect as it does for each sub-env of a SyncVecEnv
(`set_attr("reward_fn", reward)` stores a copy of `reward`, as `reward=` does).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sekai._types import ActType, ObsType
from sekai.envs.finance.market_making import MarketMakingEnv
from sekai.envs.finance.optimal_execution import OptimalExecutionEnv
from sekai.envs.finance.portfolio import PortfolioEnv
from sekai.envs.finance.trading import TradingEnv
from sekai.error import InvalidAction
from sekai.finance.rewards import StepOutcome
from sekai.finance.simulators import GBM
from sekai.vector.native import InfoArrays, NativeVecEnv, shared

Array = NDArray[np.float64]


def _batch_action(actions: Any, num_envs: int, shape: tuple[int, ...], space: Any) -> Array:
    a = np.asarray(actions, dtype=np.float64).reshape(num_envs, *shape)
    if not np.all(np.isfinite(a)):
        raise InvalidAction(actions, space)
    return a


class _Param:
    """A parameter of the single env, read from and written to the native env's `template`.

    The batched dynamics read their parameters from the template, so setting one (directly
    or with set_attr) changes every sub-env's simulation, as it changes each sub-env of a
    SyncVecEnv. `readonly` gives the reason a parameter cannot change (e.g. it fixes the
    shapes of the env's arrays); an empty string means it can.
    """

    def __init__(self, *, readonly: str = "") -> None:
        self.readonly = readonly
        self.name = ""

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    def __get__(self, obj: Any, objtype: type | None = None) -> Any:
        if obj is None:
            return self
        return getattr(obj.template, self.name)

    def __set__(self, obj: Any, value: Any) -> None:
        if self.readonly:
            raise AttributeError(f"{type(obj).__name__}.{self.name} is read-only: {self.readonly}.")
        setattr(obj.template, self.name, value)


# Why the read-only parameters cannot change.
_SHAPE = "it fixes the shapes of the env's arrays; build a new env to change it"
_HISTORY = "the price history is loaded when the env is built; build a new env to change it"
_COST = "it became `cost_model` when the env was built; set `cost_model` instead"
_CLASS = "it describes the env class"


def _same(a: Any, b: Any) -> bool:
    if a is b:
        return True
    try:
        return bool(np.array_equal(a, b))
    except (TypeError, ValueError):
        return False


class _TemplateVecEnv(NativeVecEnv[ObsType, ActType]):
    """A native env whose parameters are `_Param`s of `template`, a single env.

    All sub-envs share those parameters, so get_attr() repeats them and set_attr() takes
    one value (or a list repeating one value).
    """

    template: Any

    @classmethod
    def _is_param(cls, name: str) -> bool:
        return isinstance(getattr(cls, name, None), _Param)

    def get_attr(self, name: str) -> list[Any]:
        if self._is_param(name):
            return [getattr(self, name)] * self.num_envs
        return super().get_attr(name)

    def call(self, method: str, *args: Any, **kwargs: Any) -> list[Any]:
        if self._is_param(method) and not callable(getattr(self, method)):
            return self.get_attr(method)
        return super().call(method, *args, **kwargs)

    def set_attr(self, name: str, values: list[Any] | Any) -> None:
        if self._is_param(name) and isinstance(values, list):
            if len(values) != self.num_envs:
                raise ValueError(f"Expected {self.num_envs} values, got {len(values)}.")
            if not all(_same(value, values[0]) for value in values[1:]):
                raise ValueError(
                    f"{type(self).__name__}.{name} is one parameter shared by every sub-env, "
                    "so it takes one value; for a different value per sub-env use "
                    "vectorization_mode='sync'."
                )
            values = values[0]
        super().set_attr(name, values)


class _HistoryWindows:
    """Per-sub-env window starts into a single env's History (or simulation)."""

    def __init__(self, template: TradingEnv | PortfolioEnv, num_envs: int) -> None:
        self.template = template
        self.history = template._history
        # Each sub-env samples with `sampler.spawn()`, like the sub-envs of a SyncVecEnv
        # built by make_vec(): under "sequential" every sub-env replays the same windows,
        # while a sampler without per-env state is shared (not copied).
        self.lanes = [] if self.history is None else [self.history.spawn() for _ in range(num_envs)]
        self.starts = np.zeros(num_envs, dtype=np.intp)

    def load(
        self,
        rng: np.random.Generator,
        indices: NDArray[np.intp],
        options: Sequence[dict[str, Any] | None],
        episode_length: int,
        simulate: Any,
    ) -> Array:
        """Price windows `(len(indices), window, n)` for the sub-envs being reset."""
        window = self.template._window
        if self.history is None:
            if any(opts and opts.get("start") is not None for opts in options):
                raise ValueError("options['start'] needs a price history (prices= or data=).")
            self.starts[indices] = 0
            paths: Array = simulate(len(indices))
            return paths
        starts = [
            self.lanes[i].start(rng, opts, episode_length) for i, opts in zip(indices, options)
        ]
        self.starts[indices] = starts
        rows = np.asarray(starts)[:, None] + np.arange(window)
        prices = self.history.prices
        return np.asarray(prices[rows], dtype=np.float64).reshape(len(indices), window, -1)

    def timestamps(self, t: NDArray[np.intp]) -> NDArray[Any]:
        """Decision timestamps after `t` steps (the step index itself for simulated markets)."""
        if self.history is None:
            steps: NDArray[Any] = t.copy()
            return steps
        rows = self.starts + self.template.lookback + t
        return np.asarray(self.history.data.timestamps[rows])


def _price_histories(paths: Array, current_rows: NDArray[np.intp]) -> NDArray[np.object_]:
    """Each sub-env's path up to and including its current row (lengths differ per sub-env)."""
    out = np.empty(len(paths), dtype=object)
    for i, row in enumerate(current_rows):
        out[i] = paths[i, : int(row) + 1].copy()
    return out


class TradingVecEnv(_TemplateVecEnv[NDArray[np.float32], Any]):
    """`Trading-v0` for `num_envs` sub-envs at once (see TradingEnv for the parameters).

    `max_episode_steps` truncates every sub-env after that many steps, as the `TimeLimit`
    that `make()` adds would.
    """

    template: TradingEnv
    metadata = _Param(readonly=_CLASS)
    prices = _Param(readonly=_HISTORY)
    data = _Param(readonly=_HISTORY)
    lookback = _Param(readonly=_SHAPE)
    episode_length = _Param(readonly=_SHAPE)
    transaction_cost = _Param(readonly=_COST)
    positions = _Param()
    cost_model = _Param()
    reward_fn = _Param()
    dt = _Param()
    mu = _Param()
    sigma = _Param()
    s0 = _Param()
    model = _Param()
    capital = _Param()

    def __init__(
        self,
        num_envs: int,
        *,
        autoreset: bool = True,
        max_episode_steps: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.template = TradingEnv(**kwargs)
        t = self.template
        super().__init__(
            num_envs,
            t.observation_space,
            t.action_space,
            autoreset=autoreset,
            max_episode_steps=max_episode_steps,
        )
        self._windows = _HistoryWindows(t, num_envs)
        self._paths = np.ones((num_envs, t._window))
        self._log_returns = np.zeros((num_envs, t._window - 1))
        self._t = np.zeros(num_envs, dtype=np.intp)
        self.position = np.zeros(num_envs)
        self.equity = np.ones(num_envs)

    @property
    def price(self) -> Array:
        """The current asset price of every sub-env."""
        out: Array = self._paths[np.arange(self.num_envs), self.lookback + self._t]
        return out

    @property
    def timestamp(self) -> NDArray[Any]:
        """The time of every sub-env's current decision, as in `infos[i]["timestamp"]`."""
        return self._windows.timestamps(self._t)

    @property
    def price_history(self) -> NDArray[np.object_]:
        """Every sub-env's prices observed so far, lookback included (see
        `TradingEnv.price_history`): an object array with one price array per sub-env."""
        return _price_histories(self._paths, self.lookback + self._t)

    def _simulate(self, n: int) -> Array:
        t = self.template
        if t.model is not None:
            sim = t.model.simulate(self.rng, t._window - 1, n_paths=n)
            if sim.mask is not None and not sim.mask.all():
                raise ValueError("TradingEnv needs a model whose asset is always listed.")
            return np.asarray(sim.prices[..., 0])
        model = GBM(mu=t.mu, sigma=t.sigma, s0=t.s0, dt=t.dt)
        return np.asarray(model.simulate(self.rng, t._window - 1, n_paths=n).prices[..., 0])

    def _reset_envs(
        self, indices: NDArray[np.intp], options: Sequence[dict[str, Any] | None]
    ) -> None:
        paths = self._windows.load(self.rng, indices, options, self.episode_length, self._simulate)
        self._paths[indices] = paths.reshape(len(indices), -1)
        self._log_returns[indices] = np.diff(np.log(self._paths[indices]), axis=1)
        self._t[indices] = 0
        self.position[indices] = 0.0
        self.equity[indices] = 1.0
        mask = np.zeros(self.num_envs, dtype=bool)
        mask[indices] = True
        self.reward_fn.reset(None if mask.all() else mask)

    def _observe(self) -> NDArray[np.float32]:
        rows = self._t[:, None] + np.arange(self.lookback)
        window = np.take_along_axis(self._log_returns, rows, axis=1)
        obs: NDArray[np.float32] = np.concatenate([window, self.position[:, None]], axis=1).astype(
            np.float32
        )
        return obs

    def _reset_infos(self, indices: NDArray[np.intp]) -> InfoArrays:
        k = len(indices)
        zeros = np.zeros(k)
        return {
            "equity": self.equity[indices],
            "position": self.position[indices],
            "price": self._paths[indices, self.lookback],
            "asset_return": zeros,
            "cost": zeros,
            "turnover": zeros,
            "timestamp": self._windows.timestamps(self._t)[indices],
            "dt": np.full(k, self.dt),
        }

    def _cost_context(self, i: NDArray[np.intp]) -> dict[str, Any]:
        needs = self.cost_model.requires
        ctx: dict[str, Any] = {}
        if "equity" in needs:
            ctx["equity"] = self.template.capital * self.equity
        if "volatility" in needs:
            rows = self._t[:, None] + np.arange(self.lookback)
            ctx["volatility"] = np.take_along_axis(self._log_returns, rows, axis=1).std(axis=1)[
                :, None
            ]
        data = self.template.data
        for name in ("volume", "spread"):
            if name in needs and data is not None and name in data.fields:
                values = data.field(name)[self._windows.starts + i, 0]
                if name == "volume":
                    values = values * self._paths[np.arange(self.num_envs), i]
                ctx[name] = values[:, None]
        return ctx

    def _step(self, actions: Any) -> tuple[Array, Any, Any, InfoArrays]:
        a = np.asarray(actions).reshape(self.num_envs)
        valid = np.issubdtype(a.dtype, np.integer) or bool(np.all(a == np.floor(a)))
        if not valid or not np.all((a >= 0) & (a < len(self.positions))):
            raise InvalidAction(actions, self.action_space)
        target = self.positions[a.astype(np.intp)]
        trade = target - self.position
        ar = np.arange(self.num_envs)
        active = ~self.frozen  # frozen sub-envs (autoreset off) keep their final state
        i = self.lookback + np.minimum(self._t, self.episode_length - 1)
        cost = np.asarray(self.cost_model(trade[:, None], **self._cost_context(i)), np.float64)
        asset_return = self._paths[ar, i + 1] / self._paths[ar, i] - 1.0
        growth = np.where(cost >= 1.0, 0.0, (1.0 - cost) * (1.0 + target * asset_return))
        # The position drifts with the price until the next trade (as in TradingEnv).
        held = target * (1.0 + asset_return)
        value = held + (1.0 - target)
        with np.errstate(divide="ignore", invalid="ignore"):
            drifted = held / value
        position = np.where((growth > 0.0) & (value > 0.0), drifted, 0.0)
        self.position = np.where(active, position, self.position)
        self._t = self._t + active
        terminated = growth <= 0.0
        previous_equity = self.equity
        self.equity = np.where(active, np.maximum(self.equity * growth, 0.0), self.equity)
        outcome = StepOutcome.from_growth(
            growth,
            equity=self.equity,
            turnover=np.abs(trade),
            cost=cost,
            previous_equity=previous_equity,
        )
        rewards = np.asarray(self.reward_fn(outcome), dtype=np.float64)
        truncated = ~terminated & (self._t >= self.episode_length)
        price = self._paths[ar, self.lookback + self._t]
        info = {
            "equity": self.equity.copy(),
            "position": self.position.copy(),
            "price": price,
            "asset_return": asset_return,
            "cost": cost,
            "turnover": np.abs(trade),
            "timestamp": self._windows.timestamps(self._t),
            "dt": np.full(self.num_envs, self.dt),
        }
        return rewards, terminated, truncated, info


class PortfolioVecEnv(_TemplateVecEnv[dict[str, NDArray[np.float32]], NDArray[np.float32]]):
    """`Portfolio-v0` for `num_envs` sub-envs at once (see PortfolioEnv for the parameters).

    `max_episode_steps` truncates every sub-env after that many steps, as the `TimeLimit`
    that `make()` adds would.
    """

    template: PortfolioEnv
    metadata = _Param(readonly=_CLASS)
    prices = _Param(readonly=_HISTORY)
    data = _Param(readonly=_HISTORY)
    n_assets = _Param(readonly=_SHAPE)
    lookback = _Param(readonly=_SHAPE)
    episode_length = _Param(readonly=_SHAPE)
    allow_short = _Param(readonly=_SHAPE)
    transaction_cost = _Param(readonly=_COST)
    cost_model = _Param()
    reward_fn = _Param()
    dt = _Param()
    mu = _Param()
    sigma = _Param()
    corr = _Param()
    model = _Param()
    capital = _Param()

    def __init__(
        self,
        num_envs: int,
        *,
        autoreset: bool = True,
        max_episode_steps: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.template = PortfolioEnv(**kwargs)
        t = self.template
        super().__init__(
            num_envs,
            t.observation_space,
            t.action_space,
            autoreset=autoreset,
            max_episode_steps=max_episode_steps,
        )
        self._windows = _HistoryWindows(t, num_envs)
        n = t.n_assets
        self._paths = np.ones((num_envs, t._window, n))
        self._log_returns = np.zeros((num_envs, t._window - 1, n))
        self._t = np.zeros(num_envs, dtype=np.intp)
        self.weights = np.zeros((num_envs, n + 1))
        self.weights[:, -1] = 1.0
        self.portfolio_value = np.ones(num_envs)

    @property
    def timestamp(self) -> NDArray[Any]:
        """The time of every sub-env's current decision, as in `infos[i]["timestamp"]`."""
        return self._windows.timestamps(self._t)

    @property
    def price_history(self) -> NDArray[np.object_]:
        """Every sub-env's prices observed so far, lookback included (see
        `PortfolioEnv.price_history`): an object array with one (k, n_assets) array per sub-env."""
        return _price_histories(self._paths, self.lookback + self._t)

    def _simulate(self, n_paths: int) -> Array:
        t = self.template
        if t.model is not None:
            sim = t.model.simulate(self.rng, t._window - 1, n_paths=n_paths)
            if sim.mask is not None and not sim.mask.all():
                raise ValueError(
                    "PortfolioEnv needs a model whose assets are always listed; use Market-v0 "
                    "for changing universes."
                )
            return np.asarray(sim.prices)
        model = GBM(mu=t.mu, sigma=t.sigma, s0=100.0, corr=t.corr, dt=t.dt)
        return np.asarray(model.simulate(self.rng, t._window - 1, n_paths=n_paths).prices)

    def _reset_envs(
        self, indices: NDArray[np.intp], options: Sequence[dict[str, Any] | None]
    ) -> None:
        paths = self._windows.load(self.rng, indices, options, self.episode_length, self._simulate)
        self._paths[indices] = paths
        self._log_returns[indices] = np.diff(np.log(paths), axis=1)
        self._t[indices] = 0
        self.weights[indices] = 0.0
        self.weights[indices, -1] = 1.0
        self.portfolio_value[indices] = 1.0
        mask = np.zeros(self.num_envs, dtype=bool)
        mask[indices] = True
        self.reward_fn.reset(None if mask.all() else mask)

    def _observe(self) -> dict[str, NDArray[np.float32]]:
        rows = self._t[:, None] + np.arange(self.lookback)
        returns = self._log_returns[np.arange(self.num_envs)[:, None], rows]
        return {"returns": returns.astype(np.float32), "weights": self.weights.astype(np.float32)}

    def _reset_infos(self, indices: NDArray[np.intp]) -> InfoArrays:
        k = len(indices)
        return {
            "portfolio_value": self.portfolio_value[indices],
            "equity": self.portfolio_value[indices],
            "weights": self.weights[indices],
            "turnover": np.zeros(k),
            "cost": np.zeros(k),
            "asset_returns": np.zeros((k, self.n_assets)),
            "timestamp": self._windows.timestamps(self._t)[indices],
            "dt": np.full(k, self.dt),
        }

    def _cost_context(self, i: NDArray[np.intp]) -> dict[str, Any]:
        needs = self.cost_model.requires
        ctx: dict[str, Any] = {}
        if "equity" in needs:
            ctx["equity"] = self.template.capital * self.portfolio_value
        if "volatility" in needs:
            rows = self._t[:, None] + np.arange(self.lookback)
            window = self._log_returns[np.arange(self.num_envs)[:, None], rows]
            ctx["volatility"] = window.std(axis=1)
        data = self.template.data
        for name in ("volume", "spread"):
            if name in needs and data is not None and name in data.fields:
                values = data.field(name)[self._windows.starts + i]
                if name == "volume":
                    values = values * self._paths[np.arange(self.num_envs), i]
                ctx[name] = values
        return ctx

    def action_to_weights(self, actions: Any) -> Array:
        """Batched PortfolioEnv.action_to_weights: `(num_envs, n_assets + 1)` targets."""
        shape = self.action_space.shape
        assert shape is not None
        a = _batch_action(actions, self.num_envs, shape, self.action_space)
        if self.allow_short:
            a = np.clip(a, -1.0, 1.0)
            gross = np.abs(a).sum(axis=1, keepdims=True)
            a = np.where(gross > 1.0, a / np.where(gross > 0, gross, 1.0), a)
            return np.concatenate([a, 1.0 - a.sum(axis=1, keepdims=True)], axis=1)
        a = np.clip(a, 0.0, None)
        total = a.sum(axis=1, keepdims=True)
        cash = np.zeros_like(a)
        cash[:, -1] = 1.0
        return np.where(total < 1e-12, cash, a / np.where(total < 1e-12, 1.0, total))

    def _step(self, actions: Any) -> tuple[Array, Any, Any, InfoArrays]:
        target = self.action_to_weights(actions)
        trades = target[:, :-1] - self.weights[:, :-1]
        turnover = np.abs(trades).sum(axis=1)
        ar = np.arange(self.num_envs)
        active = ~self.frozen  # frozen sub-envs (autoreset off) keep their final state
        i = self.lookback + np.minimum(self._t, self.episode_length - 1)
        cost = np.asarray(self.cost_model(trades, **self._cost_context(i)), dtype=np.float64)
        asset_returns = self._paths[ar, i + 1] / self._paths[ar, i] - 1.0
        # A batched matmul reproduces the single env's `target @ returns` bit for bit.
        gross_return = np.matmul(target[:, None, :-1], asset_returns[..., None])[:, 0, 0]
        growth = np.where(cost >= 1.0, 0.0, (1.0 - cost) * (1.0 + gross_return))
        drifted = target * np.concatenate(
            [1.0 + asset_returns, np.ones((self.num_envs, 1))], axis=1
        )
        cash_only = np.zeros_like(drifted)
        cash_only[:, -1] = 1.0
        ok = (growth > 0)[:, None]
        weights = np.where(
            ok, drifted / np.where(ok, drifted.sum(axis=1, keepdims=True), 1.0), cash_only
        )
        self.weights = np.where(active[:, None], weights, self.weights)
        previous_value = self.portfolio_value
        self.portfolio_value = np.where(
            active, np.maximum(self.portfolio_value * growth, 0.0), self.portfolio_value
        )
        self._t = self._t + active
        terminated = growth <= 0.0
        truncated = ~terminated & (self._t >= self.episode_length)
        outcome = StepOutcome.from_growth(
            growth,
            equity=self.portfolio_value,
            turnover=turnover,
            cost=cost,
            previous_equity=previous_value,
        )
        rewards = np.asarray(self.reward_fn(outcome), dtype=np.float64)
        info = {
            "portfolio_value": self.portfolio_value.copy(),
            "equity": self.portfolio_value.copy(),
            "weights": self.weights,
            "turnover": turnover,
            "cost": cost,
            "asset_returns": asset_returns,
            "timestamp": self._windows.timestamps(self._t),
            "dt": np.full(self.num_envs, self.dt),
        }
        return rewards, terminated, truncated, info


class MarketMakingVecEnv(_TemplateVecEnv[NDArray[np.float32], NDArray[np.float32]]):
    """`MarketMaking-v0` for `num_envs` sub-envs at once (see MarketMakingEnv).

    `max_episode_steps` truncates every sub-env after that many steps, as the `TimeLimit`
    that `make()` adds would.
    """

    template: MarketMakingEnv
    metadata = _Param(readonly=_CLASS)
    s0 = _Param()
    sigma = _Param()
    total_time = _Param()
    n_steps = _Param()
    dt = _Param()
    A = _Param()
    k = _Param()
    risk_aversion = _Param()
    max_inventory = _Param()
    order_size = _Param()
    max_offset = _Param()

    def __init__(
        self,
        num_envs: int,
        *,
        autoreset: bool = True,
        max_episode_steps: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.template = MarketMakingEnv(**kwargs)
        t = self.template
        super().__init__(
            num_envs,
            t.observation_space,
            t.action_space,
            autoreset=autoreset,
            max_episode_steps=max_episode_steps,
        )
        self.mid = np.full(num_envs, t.s0)
        self.inventory = np.zeros(num_envs, dtype=np.int64)
        self.cash = np.zeros(num_envs)
        self._t = np.zeros(num_envs, dtype=np.int64)
        # The last step's quotes and fills (NaN / False until a sub-env's first step).
        self.bid = np.full(num_envs, np.nan)
        self.ask = np.full(num_envs, np.nan)
        self.bid_filled = np.zeros(num_envs, dtype=bool)
        self.ask_filled = np.zeros(num_envs, dtype=bool)

    @property
    def wealth(self) -> Array:
        """Mark-to-market wealth of every sub-env."""
        out: Array = self.cash + self.inventory * self.mid
        return out

    @property
    def time_remaining(self) -> Array:
        """Session time left in every sub-env."""
        out: Array = (self.n_steps - self._t) * self.dt
        return out

    def avellaneda_stoikov_action(self) -> NDArray[np.float32]:
        """Closed-form optimal (bid, ask) distances for every sub-env, shape `(num_envs, 2)`."""
        t = self.template
        gamma, tau = t.risk_aversion, (t.n_steps - self._t) * t.dt
        skew = self.inventory * gamma * t.sigma**2 * tau
        if gamma > 0:
            spread = gamma * t.sigma**2 * tau + (2 / gamma) * math.log(1 + gamma / t.k)
        else:
            spread = np.full(self.num_envs, 2 / t.k)
        quotes = np.stack([spread / 2 + skew, spread / 2 - skew], axis=1)
        out: NDArray[np.float32] = np.clip(quotes, 0.0, t.max_offset).astype(np.float32)
        return out

    def _reset_envs(
        self, indices: NDArray[np.intp], options: Sequence[dict[str, Any] | None]
    ) -> None:
        self.mid[indices] = self.template.s0
        self.inventory[indices] = 0
        self.cash[indices] = 0.0
        self._t[indices] = 0
        self.bid[indices] = self.ask[indices] = np.nan
        self.bid_filled[indices] = self.ask_filled[indices] = False

    def _observe(self) -> NDArray[np.float32]:
        t = self.template
        obs: NDArray[np.float32] = np.stack(
            [self.inventory / t.max_inventory, (t.n_steps - self._t) / t.n_steps], axis=1
        ).astype(np.float32)
        return obs

    def _reset_infos(self, indices: NDArray[np.intp]) -> InfoArrays:
        k = len(indices)
        return {
            "mid_price": self.mid[indices],
            "inventory": self.inventory[indices],
            "cash": self.cash[indices],
            "wealth": self.wealth[indices],
            "bid": np.full(k, np.nan),
            "ask": np.full(k, np.nan),
            "bid_filled": np.zeros(k, dtype=bool),
            "ask_filled": np.zeros(k, dtype=bool),
            "timestamp": self._t[indices] * self.template.dt,
            "dt": np.full(k, self.template.dt),
        }

    def _step(self, actions: Any) -> tuple[Array, Any, Any, InfoArrays]:
        t = self.template
        a = np.clip(
            _batch_action(actions, self.num_envs, (2,), self.action_space), 0.0, t.max_offset
        )
        bid, ask = self.mid - a[:, 0], self.mid + a[:, 1]
        wealth_before = self.wealth
        can_buy = self.inventory + t.order_size <= t.max_inventory
        can_sell = self.inventory - t.order_size >= -t.max_inventory
        p_fill = 1.0 - np.exp(-t.A * np.exp(-t.k * a) * t.dt)
        u = self.rng.random((self.num_envs, 2))
        active = ~self.frozen  # frozen sub-envs (autoreset off) keep their final state
        bid_filled = active & can_buy & (u[:, 0] < p_fill[:, 0])
        ask_filled = active & can_sell & (u[:, 1] < p_fill[:, 1])
        self.inventory = self.inventory + t.order_size * (bid_filled.astype(np.int64) - ask_filled)
        self.cash = self.cash - t.order_size * bid * bid_filled + t.order_size * ask * ask_filled
        moved = self.mid + t.sigma * math.sqrt(t.dt) * self.rng.standard_normal(self.num_envs)
        self.mid = np.where(active, moved, self.mid)
        self._t = self._t + active
        self.bid, self.ask = np.where(active, bid, self.bid), np.where(active, ask, self.ask)
        self.bid_filled = np.where(active, bid_filled, self.bid_filled)
        self.ask_filled = np.where(active, ask_filled, self.ask_filled)
        penalty = 0.5 * t.risk_aversion * t.sigma**2 * t.dt * self.inventory**2
        rewards = self.wealth - wealth_before - penalty
        info = {
            "mid_price": self.mid.copy(),
            "inventory": self.inventory.copy(),
            "cash": self.cash.copy(),
            "wealth": self.wealth,
            "bid": bid,
            "ask": ask,
            "bid_filled": bid_filled,
            "ask_filled": ask_filled,
            "timestamp": self._t * t.dt,
            "dt": np.full(self.num_envs, t.dt),
        }
        terminated = self._t >= t.n_steps
        return rewards, terminated, np.zeros(self.num_envs, dtype=bool), info


class OptimalExecutionVecEnv(_TemplateVecEnv[NDArray[np.float32], NDArray[np.float32]]):
    """`OptimalExecution-v0` for `num_envs` sub-envs at once (see OptimalExecutionEnv).

    `max_episode_steps` truncates every sub-env after that many steps, as the `TimeLimit`
    that `make()` adds would.
    """

    template: OptimalExecutionEnv
    metadata = _Param(readonly=_CLASS)
    X0 = _Param()
    s0 = _Param()
    sigma = _Param()
    total_time = _Param()
    n_steps = _Param()
    tau = _Param()
    gamma = _Param()
    eta = _Param()
    epsilon = _Param()
    risk_aversion = _Param()

    def __init__(
        self,
        num_envs: int,
        *,
        autoreset: bool = True,
        max_episode_steps: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.template = OptimalExecutionEnv(**kwargs)
        t = self.template
        super().__init__(
            num_envs,
            t.observation_space,
            t.action_space,
            autoreset=autoreset,
            max_episode_steps=max_episode_steps,
        )
        self.inventory = np.full(num_envs, t.X0)
        self.mid = np.full(num_envs, t.s0)
        self.shortfall = np.zeros(num_envs)
        self._k = np.zeros(num_envs, dtype=np.int64)

    @shared  # one schedule for every sub-env: call() must not split its n_steps rows
    def almgren_chriss_actions(self) -> NDArray[np.float32]:
        """The optimal static schedule (identical for every sub-env), shape `(n_steps, 1)`."""
        return self.template.almgren_chriss_actions()

    def _reset_envs(
        self, indices: NDArray[np.intp], options: Sequence[dict[str, Any] | None]
    ) -> None:
        t = self.template
        self.inventory[indices] = t.X0
        self.mid[indices] = t.s0
        self.shortfall[indices] = 0.0
        self._k[indices] = 0

    def _observe(self) -> NDArray[np.float32]:
        t = self.template
        scale = t.sigma * math.sqrt(t.total_time)
        drift = (self.mid - t.s0) / scale if scale > 0 else np.zeros(self.num_envs)
        obs: NDArray[np.float32] = np.stack(
            [self.inventory / t.X0, (t.n_steps - self._k) / t.n_steps, drift], axis=1
        ).astype(np.float32)
        return obs

    def _reset_infos(self, indices: NDArray[np.intp]) -> InfoArrays:
        k = len(indices)
        return self._info_arrays(indices, np.zeros(k), np.full(k, np.nan))

    def _info_arrays(self, rows: Any, shares: Array, exec_price: Array) -> InfoArrays:
        notional = self.template.X0 * self.template.s0
        return {
            "shares_sold": shares,
            "execution_price": exec_price,
            "inventory": self.inventory[rows].copy(),
            "mid_price": self.mid[rows].copy(),
            "implementation_shortfall": self.shortfall[rows].copy(),
            "shortfall_bps": 1e4 * self.shortfall[rows] / notional,
            "timestamp": self._k[rows] * self.template.tau,
            "dt": np.full(len(shares), self.template.tau),
        }

    def _step(self, actions: Any) -> tuple[Array, Any, Any, InfoArrays]:
        t = self.template
        a = _batch_action(actions, self.num_envs, (1,), self.action_space)[:, 0]
        fraction = np.clip(a, 0.0, 1.0)
        last = self._k == t.n_steps - 1
        active = ~self.frozen  # frozen sub-envs (autoreset off) keep their final state
        shares = np.where(active, np.where(last, self.inventory, fraction * self.inventory), 0.0)
        exec_price = np.where(shares > 0, self.mid - t.epsilon - t.eta * shares / t.tau, self.mid)
        self.inventory = self.inventory - shares
        self.inventory[self.inventory < 1e-9 * t.X0] = 0.0
        self.shortfall = self.shortfall + shares * (t.s0 - exec_price)
        moved = self.mid + t.sigma * math.sqrt(t.tau) * self.rng.standard_normal(self.num_envs)
        self.mid = np.where(active, moved - t.gamma * shares, self.mid)
        self._k = self._k + active
        risk = t.risk_aversion * t.sigma**2 * t.tau * self.inventory**2
        rewards = 1e4 * (shares * (exec_price - t.s0) - risk) / (t.X0 * t.s0)
        terminated = (self.inventory == 0.0) | (self._k >= t.n_steps)
        info = self._info_arrays(slice(None), shares, exec_price)
        return rewards, terminated, np.zeros(self.num_envs, dtype=bool), info


__all__ = ["TradingVecEnv", "PortfolioVecEnv", "MarketMakingVecEnv", "OptimalExecutionVecEnv"]
