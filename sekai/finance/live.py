"""Paper and live trading: data feeds, brokers and risk limits.

A policy trained on `Market-v1` can trade a live account through
`sekai.envs.finance.LiveMarketEnv`, which builds the same observations and accepts the
same actions from a stream of bars. sekai defines the interfaces and never connects to a
market itself: you supply

- a `DataFeed`: anything with `assets` and `next_bar() -> Bar | None` (a market data
  service, a message queue, files that arrive each day). `ReplayFeed` replays a
  `MarketData` you own, bar by bar, for paper trading and for testing the loop.
- a `Broker`: anything that executes target weights at a bar (`execute`) and reports the
  account when the next bar arrives (`mark`). `PaperBroker` simulates the account with
  sekai's own accounting, cost and holding-cost models, so paper results are exactly
  those of the backtest; an adapter to a real broker turns target weights into orders
  (`orders_for` computes the share quantities) and reads fills and positions back.

`GuardedBroker(broker, RiskLimits(...))` puts hard limits between any policy and any
broker: caps on each position, gross and net exposure and turnover, and a drawdown kill
switch that liquidates and refuses new positions. They hold whatever the policy asks for,
and every intervention is recorded in `events`.

Example:
    ```python
    feed = ReplayFeed(test_data, start=0)
    limits = RiskLimits(max_gross=1.0, max_drawdown=0.15)
    broker = GuardedBroker(PaperBroker(transaction_cost=1e-3), limits)
    env = LiveMarketEnv(feed, broker, lookback=20)
    result = env.reset()
    while not result.done:
        result = env.step(policy(result.observation))
    ```
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from sekai.finance.account import AccountConfig, settle
from sekai.finance.costs import CostModel, HoldingCost, as_cost_model
from sekai.finance.data import MarketData

Array = NDArray[np.float64]


# ---------------------------------------------------------------------------
# Bars and feeds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bar:
    """One period of market data for a fixed list of assets, as it arrives.

    Attributes:
        timestamp: When the period ended.
        prices: `(N,)` closing prices; NaN for an asset that cannot be traded.
        fields: Per-asset data, `(N, ...)` each: volume, spread, borrow rates, and with
            `corporate_actions` the period's `split` and `dividend`.
        exogenous: Market-wide data of the period.
        corporate_actions: Whether the fields `split` (new shares per old share) and
            `dividend` (cash per share) are corporate actions effective in this period,
            as in `MarketData.simple_returns`. Otherwise fields with those names are
            ordinary fields.
    """

    timestamp: Any
    prices: Array
    fields: Mapping[str, Array] = field(default_factory=dict)
    exogenous: Mapping[str, Any] = field(default_factory=dict)
    corporate_actions: bool = False


@runtime_checkable
class DataFeed(Protocol):
    """A stream of bars for a fixed list of assets."""

    assets: tuple[Any, ...]

    def next_bar(self) -> Bar | None:
        """The next bar (blocking until it is available), or None when the feed has ended."""
        ...


class ReplayFeed:
    """A `DataFeed` that replays the rows of a `MarketData`, one bar per row.

    Args:
        data: The history.
        start: First row to replay.
        stop: Row to stop before (default: the end).
    """

    def __init__(self, data: MarketData, start: int = 0, stop: int | None = None) -> None:
        self.data = data
        self.assets = data.assets
        self.start = int(start)
        self.stop = data.n_steps if stop is None else int(stop)
        if not 0 <= self.start < self.stop <= data.n_steps:
            raise ValueError(f"Need 0 <= start < stop <= {data.n_steps}, got {start}, {stop}.")
        self.position = self.start

    def next_bar(self) -> Bar | None:
        t = self.position
        if t >= self.stop:
            return None
        self.position += 1
        d = self.data
        return Bar(
            timestamp=d.timestamps[t],
            prices=np.asarray(d.prices[t], dtype=np.float64),
            fields={k: np.asarray(v[t]) for k, v in d.fields.items() if k != d.price_field},
            exogenous={k: np.asarray(v[t]) for k, v in d.exogenous.items()},
            corporate_actions=d.has_corporate_actions,
        )

    def __repr__(self) -> str:
        return f"ReplayFeed(rows {self.position}..{self.stop} of {self.data!r})"


# ---------------------------------------------------------------------------
# Brokers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AccountState:
    """An account as a broker reports it.

    Attributes:
        equity: Equity relative to the start (1.0 at `reset`).
        weights: `(N,)` positions as fractions of equity.
        value: Equity in currency.
        financing: Interest, borrow fees and holding costs of the last period, as a
            fraction of equity (signed; 0 if the broker does not report it).
    """

    equity: float
    weights: Array
    value: float
    financing: float = 0.0


@dataclass(frozen=True)
class ExecutionReport:
    """What executing target weights cost.

    Attributes:
        weights: `(N,)` the weights actually held after the trades.
        turnover: Sum of absolute weight changes.
        cost: Trading cost as a fraction of equity.
    """

    weights: Array
    turnover: float
    cost: float


@runtime_checkable
class Broker(Protocol):
    """Executes target weights and reports the account (see `PaperBroker`)."""

    def reset(self, assets: tuple[Any, ...]) -> AccountState:
        """Start a new account over `assets` (or attach to an existing one)."""
        ...

    def execute(self, weights: Array, bar: Bar, context: Mapping[str, Any]) -> ExecutionReport:
        """Trade to `weights` (fractions of equity per asset) at `bar`'s close.

        `context` holds the inputs of cost models (`equity`, `volatility`, `volume`,
        `spread`, `borrow_rate`, `assets`) for brokers that estimate costs.
        """
        ...

    def mark(self, bar: Bar) -> AccountState:
        """The account when the next bar has arrived (positions marked to its prices)."""
        ...


class PaperBroker:
    """A simulated account with sekai's accounting: the backtest's numbers, bar by bar.

    Trades fill at the bar's close at a cost from `transaction_cost`; positions then earn
    the assets' total returns (the next bar's `split` and `dividend` fields count when it
    has `corporate_actions`), cash earns `cash_rate` (or pays `margin_rate`), shorts pay
    `borrow_rate`, and `holding_cost` charges the rest; an asset that stops trading
    returns `delisting_return` and its position becomes cash.

    Args:
        transaction_cost: Proportional rate or `CostModel`.
        holding_cost: A `HoldingCost` charged every period.
        cash_rate: Annual interest earned on positive cash.
        margin_rate: Annual interest paid on borrowed cash (default: cash_rate).
        borrow_rate: Annual fee on the value of short positions.
        capital: Starting equity in currency.
        dt: Years per bar.
        delisting_return: Return of an asset over the period in which it stops trading.
    """

    def __init__(
        self,
        *,
        transaction_cost: float | CostModel = 1e-3,
        holding_cost: HoldingCost | None = None,
        cash_rate: float = 0.0,
        margin_rate: float | None = None,
        borrow_rate: float = 0.0,
        capital: float = 1_000_000.0,
        dt: float = 1 / 252,
        delisting_return: float = 0.0,
    ) -> None:
        self.cost_model = as_cost_model(transaction_cost)
        self.holding_cost = holding_cost
        self.config = AccountConfig(
            allow_short=True,
            max_leverage=1.0,
            cash_rate=cash_rate,
            margin_rate=margin_rate,
            borrow_rate=borrow_rate,
        )
        if not (math.isfinite(capital) and capital > 0):
            raise ValueError(f"capital must be finite and positive, got {capital!r}.")
        self.capital = float(capital)
        self.dt = float(dt)
        self.delisting_return = float(delisting_return)
        self._equity = 1.0
        self._weights: Array = np.zeros(0)
        self._held: Array | None = None
        self._cost = 0.0
        self._holding: float | None = None
        self._prices: Array | None = None

    def reset(self, assets: tuple[Any, ...]) -> AccountState:
        self._equity = 1.0
        self._weights = np.zeros(len(assets))
        self._held = None
        self._prices = None
        return self._state(0.0)

    def execute(self, weights: Array, bar: Bar, context: Mapping[str, Any]) -> ExecutionReport:
        held = np.asarray(weights, dtype=np.float64)
        trades = held - self._weights
        ctx = {k: v for k, v in context.items() if k in self.cost_model.requires or k == "assets"}
        self._cost = float(
            np.asarray(self.cost_model(trades[None], **_batched(ctx))).reshape(-1)[0]
        )
        self._holding = None
        if self.holding_cost is not None:
            self._holding = float(
                np.asarray(
                    self.holding_cost(
                        held[None],
                        dt=self.dt,
                        borrow_rate=_batched(context).get("borrow_rate"),
                        assets=_batched(context).get("assets"),
                    )
                ).reshape(-1)[0]
            )
        self._held = held
        self._prices = np.asarray(bar.prices, dtype=np.float64)
        return ExecutionReport(weights=held, turnover=float(np.abs(trades).sum()), cost=self._cost)

    def mark(self, bar: Bar) -> AccountState:
        if self._held is None or self._prices is None:
            raise RuntimeError("PaperBroker.mark() needs an execute() first.")
        prev, now = self._prices, np.asarray(bar.prices, dtype=np.float64)
        value = now
        if bar.corporate_actions and "dividend" in bar.fields:
            value = value + np.nan_to_num(
                np.asarray(bar.fields["dividend"], dtype=np.float64), nan=0.0
            )
        if bar.corporate_actions and "split" in bar.fields:
            ratio = np.asarray(bar.fields["split"], dtype=np.float64)
            value = value * np.where(np.isfinite(ratio), ratio, 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = value / prev - 1.0
        valid_now, valid_next = np.isfinite(prev), np.isfinite(now)
        delisting = valid_now & ~valid_next
        ret = np.where(valid_now & valid_next, r, 0.0)
        ret = np.where(delisting, self.delisting_return, ret)
        s = settle(
            self._held[None],
            ret[None],
            cost=np.array([self._cost]),
            dt=self.dt,
            config=self.config,
            holding=None if self._holding is None else np.array([self._holding]),
        )
        growth = float(s.growth[0])
        self._weights = np.where(delisting, 0.0, s.drifted[0])
        self._equity = max(self._equity * growth, 0.0)
        self._held = None
        return self._state(float(s.financing[0]))

    def _state(self, financing: float) -> AccountState:
        return AccountState(
            equity=self._equity,
            weights=self._weights.copy(),
            value=self._equity * self.capital,
            financing=financing,
        )

    def __repr__(self) -> str:
        return f"PaperBroker(transaction_cost={self.cost_model!r}, capital={self.capital:g})"


def _batched(context: Mapping[str, Any]) -> dict[str, Any]:
    """Per-asset context values with a leading batch axis of 1 (cost models are batched)."""
    out: dict[str, Any] = {}
    for k, v in context.items():
        out[k] = np.asarray(v)[None] if np.ndim(v) >= 1 else np.asarray([v])
    return out


def orders_for(
    weights: Array, account_value: float, prices: Array, positions: Array, *, lot_size: float = 1.0
) -> Array:
    """Share quantities to buy (positive) or sell to move from `positions` to `weights`.

    Args:
        weights: `(N,)` target weights (fractions of `account_value`).
        account_value: Equity in currency.
        prices: `(N,)` prices to size the orders at (NaN: no order).
        positions: `(N,)` current holdings in shares.
        lot_size: Orders are rounded to whole lots (towards zero, so the trade never
            overshoots the target).

    Returns:
        `(N,)` order quantities in shares.
    """
    p = np.asarray(prices, dtype=np.float64)
    ok = np.isfinite(p) & (p > 0)
    target = np.where(
        ok, np.asarray(weights, dtype=np.float64) * account_value / np.where(ok, p, 1.0), 0.0
    )
    delta = np.where(ok, target - np.asarray(positions, dtype=np.float64), 0.0)
    return np.trunc(delta / lot_size) * lot_size


# ---------------------------------------------------------------------------
# Risk limits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskLimits:
    """Hard limits on what an account may hold, enforced by `GuardedBroker`.

    Args:
        max_position: Cap on each `|w_i|`.
        max_gross: Cap on gross exposure `sum(|w|)`.
        max_net: Cap on net exposure `|sum(w)|`.
        max_turnover: Cap on one rebalance's turnover `sum(|w_new - w|)`: larger moves go
            only that far towards the target.
        max_drawdown: Kill switch: once the drawdown from the account's peak reaches it,
            everything is sold and no new position is allowed until `reset`.
        allow_short: Allow negative weights.
    """

    max_position: float | None = None
    max_gross: float | None = None
    max_net: float | None = None
    max_turnover: float | None = None
    max_drawdown: float | None = None
    allow_short: bool = True

    def __post_init__(self) -> None:
        for name in ("max_position", "max_gross", "max_net", "max_turnover"):
            value = getattr(self, name)
            if value is not None and not (math.isfinite(value) and value >= 0):
                raise ValueError(f"{name} must be finite and >= 0, got {value!r}.")
        if self.max_drawdown is not None and not 0.0 < self.max_drawdown <= 1.0:
            raise ValueError(f"max_drawdown must be in (0, 1], got {self.max_drawdown!r}.")

    def apply(self, target: Array, current: Array) -> tuple[Array, list[str]]:
        """`target` within the limits (given the `current` weights), and the limits that bound."""
        w = np.asarray(target, dtype=np.float64).copy()
        hit: list[str] = []
        if not self.allow_short and np.any(w < 0):
            w = np.clip(w, 0.0, None)
            hit.append("allow_short")
        if self.max_position is not None and np.any(np.abs(w) > self.max_position):
            w = np.clip(w, -self.max_position, self.max_position)
            hit.append("max_position")
        if self.max_net is not None and abs(w.sum()) > self.max_net:
            side = w > 0 if w.sum() > 0 else w < 0  # scale down the side that is too large
            other = w[~side].sum()
            sign = 1.0 if w.sum() > 0 else -1.0
            needed = sign * self.max_net - other
            total = w[side].sum()
            w[side] = w[side] * (needed / total if total != 0 else 0.0)
            hit.append("max_net")
        gross = np.abs(w).sum()
        if self.max_gross is not None and gross > self.max_gross:
            w = w * (self.max_gross / gross)
            hit.append("max_gross")
        move = w - np.asarray(current, dtype=np.float64)
        turnover = np.abs(move).sum()
        if self.max_turnover is not None and turnover > self.max_turnover:
            w = np.asarray(current, dtype=np.float64) + move * (self.max_turnover / turnover)
            hit.append("max_turnover")
        return w, hit


@dataclass(frozen=True)
class RiskEvent:
    """A limit that changed what the policy asked for."""

    timestamp: Any
    kind: str  # the limit's name, or "kill_switch"
    detail: str


class GuardedBroker:
    """A `Broker` that enforces `RiskLimits` before every order reaches `broker`.

    The limits act outside the policy, whatever it asks for: targets are clipped to them,
    and when the drawdown reaches `max_drawdown` the kill switch sells everything and
    keeps the account flat until `reset`. `events` lists every intervention.

    Args:
        broker: The broker to protect (paper or live).
        limits: The limits.
    """

    def __init__(self, broker: Broker, limits: RiskLimits) -> None:
        self.broker = broker
        self.limits = limits
        self.events: list[RiskEvent] = []
        self.killed = False
        self._peak = 1.0
        self._weights: Array = np.zeros(0)

    def __getattr__(self, name: str) -> Any:  # cost_model, capital, ... of the inner broker
        if name.startswith("_") or name in ("broker", "limits"):
            raise AttributeError(name)
        return getattr(self.broker, name)

    def reset(self, assets: tuple[Any, ...]) -> AccountState:
        self.killed = False
        self._peak = 1.0
        state = self.broker.reset(assets)
        self._weights = state.weights
        return state

    def execute(self, weights: Array, bar: Bar, context: Mapping[str, Any]) -> ExecutionReport:
        target = np.asarray(weights, dtype=np.float64)
        if self.killed:
            if np.any(target != 0):
                self.events.append(RiskEvent(bar.timestamp, "kill_switch", "account kept flat"))
            target = np.zeros_like(target)
        else:
            target, hit = self.limits.apply(target, self._weights)
            for kind in hit:
                self.events.append(RiskEvent(bar.timestamp, kind, f"target clipped by {kind}"))
        return self.broker.execute(target, bar, context)

    def mark(self, bar: Bar) -> AccountState:
        state = self.broker.mark(bar)
        self._weights = state.weights
        self._peak = max(self._peak, state.equity)
        drawdown = 1.0 - state.equity / self._peak if self._peak > 0 else 1.0
        limit = self.limits.max_drawdown
        if limit is not None and not self.killed and drawdown >= limit:
            self.killed = True
            self.events.append(
                RiskEvent(bar.timestamp, "kill_switch", f"drawdown {drawdown:.1%} >= {limit:.1%}")
            )
        return state

    def __repr__(self) -> str:
        return f"GuardedBroker({self.broker!r}, {self.limits!r})"


__all__ = [
    "Bar",
    "DataFeed",
    "ReplayFeed",
    "AccountState",
    "ExecutionReport",
    "Broker",
    "PaperBroker",
    "orders_for",
    "RiskLimits",
    "RiskEvent",
    "GuardedBroker",
]
