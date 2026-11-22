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
switch that liquidates and refuses new positions until an operator calls `rearm()`. They
hold whatever the policy asks for, and every intervention is recorded in `events`.

Bars are checked as they arrive (`Bar.validate`): a price is NaN (the asset cannot be
traded) or finite and positive, as in `MarketData`. A bar flagged `warmup` is history for
the lookback window only, observed but never traded on: `ReplayFeed` flags the history
rows that a split prepends to a test period (`MarketData.warmup`).

Example:
    ```python
    train, test = data.split(0.8, embargo=5, history=20)
    feed = ReplayFeed(test)  # the 20 history rows are warm-up bars: observed, not traded on
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

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
        prices: `(N,)` closing prices: NaN for an asset that cannot be traded, else finite
            and positive (see `validate`).
        fields: Per-asset data, `(N, ...)` each: volume, spread, borrow rates, and with
            `corporate_actions` the period's `split` and `dividend`.
        exogenous: Market-wide data of the period.
        corporate_actions: Whether the fields `split` (new shares per old share) and
            `dividend` (cash per share) are corporate actions effective in this period,
            as in `MarketData.simple_returns`. Otherwise fields with those names are
            ordinary fields.
        warmup: History for the lookback window only, like the rows of
            `MarketData.warmup`: it may be observed, but no decision is made on it.
            Warm-up bars come before the first bar to trade on.
    """

    timestamp: Any
    prices: Array
    fields: Mapping[str, Array] = field(default_factory=dict)
    exogenous: Mapping[str, Any] = field(default_factory=dict)
    corporate_actions: bool = False
    warmup: bool = False

    def validate(self, assets: Sequence[Any]) -> None:
        """Check the bar's prices (and corporate actions) for `assets`, as `MarketData` would.

        A price must be NaN (the asset cannot be traded) or finite and positive; with
        `corporate_actions`, a `split` NaN (none) or finite and positive, and a
        `dividend` NaN (none) or finite and >= 0. `LiveMarketEnv` and `PaperBroker` check
        every bar this way, so a bad price never reaches the accounts.

        Raises:
            ValueError: For a wrong number of prices or an invalid value, naming the asset
                and the bar's timestamp.
        """
        n = len(assets)
        prices = np.asarray(self.prices, dtype=np.float64)
        if prices.shape != (n,):
            raise ValueError(
                f"The bar at {self.timestamp!r} has prices of shape {prices.shape} for {n} assets."
            )
        valid = np.isfinite(prices) & (prices > 0)
        self._check(prices, valid, assets, "price", "NaN (not tradable) or finite and > 0")
        if not self.corporate_actions:
            return
        for name, rule in (("split", "finite and > 0"), ("dividend", "finite and >= 0")):
            if name not in self.fields:
                continue
            values = np.asarray(self.fields[name], dtype=np.float64)
            if values.shape != (n,):
                raise ValueError(
                    f"The bar at {self.timestamp!r} has a {name} field of shape {values.shape} "
                    f"for {n} assets."
                )
            valid = np.isfinite(values) & (values > 0 if name == "split" else values >= 0)
            self._check(values, valid, assets, name, f"NaN (none) or {rule}")

    def _check(
        self, values: Array, valid: NDArray[np.bool_], assets: Sequence[Any], name: str, rule: str
    ) -> None:
        """Raise for the first value that is neither NaN nor `valid`."""
        bad = ~(np.isnan(values) | valid)
        if bad.any():
            i = int(np.argmax(bad))
            raise ValueError(
                f"The bar at {self.timestamp!r} gives asset {assets[i]!r} a {name} of "
                f"{float(values[i])!r}; a {name} must be {rule}."
            )


@runtime_checkable
class DataFeed(Protocol):
    """A stream of bars for a fixed list of assets.

    A feed may also declare the shapes of its data, which `LiveMarketEnv` needs for its
    observation space before any bar has arrived: `field_shapes`, the trailing shape of
    each per-asset field (`()` for one value per asset, `(3,)` for three), and
    `exogenous_shapes`, the shape of each market-wide series (`ReplayFeed` has both).
    """

    assets: tuple[Any, ...]

    def next_bar(self) -> Bar | None:
        """The next bar (blocking until it is available), or None when the feed has ended
        (for good: an ended feed stays ended)."""
        ...


class ReplayFeed:
    """A `DataFeed` that replays the rows of a `MarketData`, one bar per row.

    The rows before `data.warmup` (the history a split prepends to a test period) come as
    warm-up bars, which `LiveMarketEnv` observes but never trades on, so paper trading a
    test split makes its first decision where `Market-v1` does. `field_shapes` and
    `exogenous_shapes` are those of the data.

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
        self.field_shapes: dict[str, tuple[int, ...]] = {
            k: tuple(v.shape[2:]) for k, v in data.fields.items() if k != data.price_field
        }
        self.exogenous_shapes: dict[str, tuple[int, ...]] = {
            k: tuple(v.shape[1:]) for k, v in data.exogenous.items()
        }

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
            warmup=t < d.warmup,
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
        weights: `(N,)` the weights actually held after the trades (they may differ from
            the targets: limits, untradable assets, unfilled orders).
        turnover: Sum of absolute weight changes.
        cost: Trading cost as a fraction of equity.
    """

    weights: Array
    turnover: float
    cost: float


@runtime_checkable
class Broker(Protocol):
    """Executes target weights and reports the account (see `PaperBroker`).

    `LiveMarketEnv` calls `reset` to start an episode and then `execute` and `mark` in
    turn: an order at a bar, then the account once the next bar has arrived, so every
    bar over which the account holds positions is marked.

    A broker may also set `fresh_account`: True if `reset()` opens a new account every
    time (`PaperBroker`, a simulator), False if it attaches to the same persistent
    account (a live one). `GuardedBroker` measures the drawdown of a fresh account from
    its opening, and that of a persistent account across resets. A broker without the
    attribute is taken to be persistent, the conservative choice.
    """

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
    returns `delisting_return` and its position becomes cash. An asset without a price
    (NaN) at the bar of an order cannot be traded: its weight stays as it is, at no cost,
    as in `Market-v1`. Every bar is checked (`Bar.validate`), and every `reset()` opens a
    fresh account (`fresh_account` is True).

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

    fresh_account: ClassVar[bool] = True

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
        self._assets: tuple[Any, ...] = ()
        self._equity = 1.0
        self._weights: Array = np.zeros(0)
        self._held: Array | None = None
        self._cost = 0.0
        self._holding: float | None = None
        self._prices: Array | None = None

    def reset(self, assets: tuple[Any, ...]) -> AccountState:
        self._assets = tuple(assets)
        self._equity = 1.0
        self._weights = np.zeros(len(self._assets))
        self._held = None
        self._prices = None
        return self._state(0.0)

    def execute(self, weights: Array, bar: Bar, context: Mapping[str, Any]) -> ExecutionReport:
        bar.validate(self._assets)
        requested = np.asarray(weights, dtype=np.float64)
        if requested.shape != self._weights.shape:
            raise ValueError(
                f"Got weights of shape {requested.shape} for {len(self._assets)} assets "
                "(reset(assets) opens the account)."
            )
        prices = np.asarray(bar.prices, dtype=np.float64)
        tradable = np.isfinite(prices)
        bad = tradable & ~np.isfinite(requested)
        if bad.any():
            i = int(np.argmax(bad))
            raise ValueError(
                f"The target weight of asset {self._assets[i]!r} at {bar.timestamp!r} is "
                f"{float(requested[i])!r}; weights must be finite."
            )
        held = np.where(tradable, requested, self._weights)  # no price, no trade
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
        self._prices = prices
        return ExecutionReport(
            weights=held.copy(), turnover=float(np.abs(trades).sum()), cost=self._cost
        )

    def mark(self, bar: Bar) -> AccountState:
        if self._held is None or self._prices is None:
            raise RuntimeError("PaperBroker.mark() needs an execute() first.")
        bar.validate(self._assets)
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


# A quantity of lots within this of a whole number is that number: a target worth exactly
# k lots must not come out k - 1 because `w * value / price` rounded down by an ulp.
_LOT_ABS_TOL = 1e-9
_LOT_REL_TOL = 1e-12


def orders_for(
    weights: Array, account_value: float, prices: Array, positions: Array, *, lot_size: float = 1.0
) -> Array:
    """Share quantities to buy (positive) or sell to move from `positions` to `weights`.

    Args:
        weights: `(N,)` target weights (fractions of `account_value`), finite.
        account_value: Equity in currency, finite.
        prices: `(N,)` prices to size the orders at: NaN for an asset that cannot be traded
            (no order), else finite and positive. Only assets with a target weight or a
            position are checked.
        positions: `(N,)` current holdings in shares, finite.
        lot_size: Orders are whole numbers of lots (finite, > 0), rounded towards zero so
            the trade never overshoots the target. A quantity within float rounding of a
            whole number of lots (1e-9 lots, or 1e-12 of it) counts as that number.

    Returns:
        `(N,)` order quantities in shares, finite.

    Raises:
        ValueError: If the arrays differ in shape, `lot_size` or `account_value` is
            invalid (not finite and > 0), a weight or position is not finite, or an
            asset that needs an order has a price that is neither NaN nor finite and
            positive.
    """
    w = np.asarray(weights, dtype=np.float64)
    p = np.asarray(prices, dtype=np.float64)
    held = np.asarray(positions, dtype=np.float64)
    if w.ndim != 1 or w.shape != p.shape or w.shape != held.shape:
        raise ValueError(
            f"weights, prices and positions must be (N,) arrays of one shape, got "
            f"{w.shape}, {p.shape} and {held.shape}."
        )
    lot = _finite_float("lot_size", lot_size)
    if not lot > 0:
        raise ValueError(f"lot_size must be finite and > 0, got {lot_size!r}.")
    value = _finite_float("account_value", account_value)
    if not value > 0:  # a negative value would turn buys into sells
        raise ValueError(f"account_value must be > 0, got {account_value!r}.")
    for name, array in (("weight", w), ("position", held)):
        if not np.all(np.isfinite(array)):
            i = int(np.argmax(~np.isfinite(array)))
            raise ValueError(f"The {name} of asset {i} is {float(array[i])!r}; it must be finite.")
    tradable = ~np.isnan(p)
    priced = tradable & np.isfinite(p) & (p > 0)
    bad = tradable & ~priced & ((w != 0) | (held != 0))
    if bad.any():
        i = int(np.argmax(bad))
        raise ValueError(
            f"The price of asset {i} is {float(p[i])!r}; a price must be NaN (no order) or "
            "finite and > 0."
        )
    target = np.where(priced, w * value / np.where(priced, p, 1.0), 0.0)
    lots = np.where(priced, (target - held) / lot, 0.0)
    nearest = np.round(lots)
    close = np.abs(lots - nearest) <= _LOT_ABS_TOL + _LOT_REL_TOL * np.abs(nearest)
    lots = np.where(close, nearest, lots)
    orders: Array = np.trunc(lots) * lot + 0.0  # + 0.0: no negative zeros
    if not np.all(np.isfinite(orders)):  # e.g. an account value too large for float64
        raise ValueError("The orders overflow float64; check account_value and the prices.")
    return orders


def _finite_float(name: str, value: Any) -> float:
    if isinstance(value, bool | np.bool_):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number, got {value!r}.") from None
    if not math.isfinite(out):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")
    return out


# ---------------------------------------------------------------------------
# Risk limits
# ---------------------------------------------------------------------------


# A scale applied when a net or gross cap binds is shrunk by this much, so the book stays
# within the cap after rounding (the sum of a few thousand weights is off by less).
_MARGIN = 1.0 - 1e-12


@dataclass(frozen=True)
class RiskLimits:
    """Hard limits on what an account may hold, enforced by `GuardedBroker`.

    `apply` brings a target book within them. A non-finite target weight means "do not
    trade that asset": it keeps its current weight, and the book still goes through
    every limit. The book that results always satisfies `max_position`, `max_gross`,
    `max_net` and `allow_short`, even when the current one does not (prices drift a
    book over a cap): the trades that only bring the current book back within those
    limits are made beyond the turnover budget, and `max_turnover` limits the rest of
    the move. Assets that cannot be traded now (`tradable`) keep their current weights
    and the limits bind on the others, so the book holds once executed; only positions
    that cannot be traded can still break a limit, on their own.

    Args:
        max_position: Cap on each `|w_i|`.
        max_gross: Cap on gross exposure `sum(|w|)`.
        max_net: Cap on net exposure `|sum(w)|`.
        max_turnover: Cap on one rebalance's turnover `sum(|w_new - w|)`: larger moves go
            only that far towards the target (beyond the risk-reducing trades above).
        max_drawdown: Kill switch: once the account's drawdown from its peak reaches it,
            everything is sold and no new position is allowed until
            `GuardedBroker.rearm()`.
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

    def apply(
        self, target: Array, current: Array, tradable: NDArray[np.bool_] | None = None
    ) -> tuple[Array, list[str]]:
        """`target` within the limits (given the `current` weights), and the limits that bound.

        Args:
            target: `(N,)` target weights.
            current: `(N,)` current weights, finite.
            tradable: `(N,)` the assets that can be traded now (default: all); the others
                keep their current weights. `GuardedBroker` passes the assets with a price.

        Returns:
            The weights to trade to, and the names of the limits that changed them, in the
            order applied: `"non_finite_target"` if a target weight was not finite, the
            caps the target was clipped by, the caps the current book exceeded (reduced
            beyond the turnover budget), and `"max_turnover"`.

        Raises:
            ValueError: If the arrays differ in shape or `current` is not finite.
        """
        w, hits = self._limit(target, current, tradable)
        return w, list(dict.fromkeys(kind for kind, _ in hits))

    def _limit(
        self,
        target: Array,
        current: Array,
        tradable: NDArray[np.bool_] | None = None,
        assets: Sequence[Any] | None = None,
    ) -> tuple[Array, list[tuple[str, str]]]:
        """`apply`, with a description of each intervention (naming `assets`, if given)."""
        t = np.asarray(target, dtype=np.float64)
        now = np.asarray(current, dtype=np.float64)
        if t.shape != now.shape:
            raise ValueError(f"target has shape {t.shape} but current has {now.shape}.")
        if not np.all(np.isfinite(now)):
            raise ValueError("current weights must be finite.")
        fixed = np.zeros(t.shape, dtype=np.bool_)
        if tradable is not None:
            fixed = ~np.asarray(tradable, dtype=np.bool_)
            if fixed.shape != t.shape:
                raise ValueError(f"tradable has shape {fixed.shape} but target has {t.shape}.")
        hits: list[tuple[str, str]] = []
        unknown = ~np.isfinite(t) & ~fixed
        if unknown.any():
            names = [assets[i] if assets is not None else i for i in np.flatnonzero(unknown)]
            hits.append(
                ("non_finite_target", f"target not finite for {names}: kept the current weight")
            )
        t = np.where(unknown | fixed, now, t)  # not traded: as it is
        capped, clipped = self._caps(t, fixed)
        hits += [(kind, f"target clipped by {kind}") for kind in clipped]
        w = capped
        if self.max_turnover is not None and np.abs(capped - now).sum() > self.max_turnover:
            # The current book within the caps: trades that only reduce risk, whatever the
            # budget. The budget then moves the book from there towards the target; both
            # ends satisfy the caps (a convex set), so every point in between does too.
            base, exceeded = self._caps(now, fixed)
            hits += [
                (kind, f"current book exceeds {kind}: reduced beyond the turnover budget")
                for kind in exceeded
            ]
            move = capped - base
            room = float(np.abs(move).sum())
            if room > self.max_turnover:
                w = base + move * (self.max_turnover / room)
                detail = f"rebalance limited to turnover {self.max_turnover:g}"
                hits.append(("max_turnover", detail))
            w, _ = self._caps(w, fixed)  # rounding only: the book is within the caps already
        return w, hits

    def _caps(self, weights: Array, fixed: NDArray[np.bool_]) -> tuple[Array, list[str]]:
        """`weights` within the short, position, net and gross limits, and the limits applied.

        Only the weights not `fixed` change: the net and gross limits bind on them, with
        the fixed positions counted (if those alone break a limit, the others go to 0).
        """
        w = np.array(weights, dtype=np.float64)
        free = ~fixed
        hit: list[str] = []
        if not self.allow_short and np.any(free & (w < 0)):
            w = np.where(free & (w < 0), 0.0, w)
            hit.append("allow_short")
        cap = self.max_position
        if cap is not None and np.any(free & (np.abs(w) > cap)):
            w = np.where(free, np.clip(w, -cap, cap), w)
            hit.append("max_position")
        if self.max_net is not None and abs(w.sum()) > self.max_net:
            sign = 1.0 if w.sum() > 0 else -1.0
            side = free & (sign * w > 0)  # scale down the side that is too large
            other = w[~side].sum()
            needed = sign * self.max_net - other
            total = w[side].sum()
            ratio = min(max(needed / total, 0.0), 1.0) if total != 0 else 0.0
            w[side] = w[side] * (ratio * _MARGIN)  # within the cap after rounding, too
            hit.append("max_net")
        gross = np.abs(w).sum()
        if self.max_gross is not None and gross > self.max_gross:
            held = np.abs(w[fixed]).sum()
            room = gross - held
            scale = max(self.max_gross - held, 0.0) / room if room > 0 else 0.0
            w = np.where(free, w * (min(scale, 1.0) * _MARGIN), w)  # within it after rounding
            hit.append("max_gross")
        return w, hit


@dataclass(frozen=True)
class RiskEvent:
    """An intervention: a limit that changed what the policy asked for, the kill switch,
    or an operator's `rearm`."""

    timestamp: Any
    kind: str  # a limit's name, "non_finite_target", "kill_switch" or "rearm"
    detail: str


class GuardedBroker:
    """A `Broker` that enforces `RiskLimits` before every order reaches `broker`.

    The limits act outside the policy, whatever it asks for: targets are brought within
    them (`RiskLimits.apply`; a non-finite target leaves that asset as it is, and so does
    a bar without a price for it, at which it cannot be traded). The kill
    switch trips when the account's drawdown reaches `max_drawdown`, or when the broker
    reports a non-finite account (it fails closed): the next order sells everything,
    and the account is kept flat. It stays tripped, across `reset()`s too, until an
    operator calls `rearm()`. `events` lists every intervention.

    The drawdown is the account's: its value (`AccountState.value`) against the highest
    value it has reported. For a broker whose `reset()` opens a fresh account
    (`fresh_account`, as `PaperBroker`) the peak starts again with each account; for any
    other broker, a persistent live account, it carries over from one reset to the next,
    so a run of episodes cannot escape the limit by starting afresh.

    Args:
        broker: The broker to protect (paper or live).
        limits: The limits.
    """

    def __init__(self, broker: Broker, limits: RiskLimits) -> None:
        self.broker = broker
        self.limits = limits
        self.events: list[RiskEvent] = []
        self._killed = False
        self._peak: float | None = None  # highest account value since opened or re-armed
        self._value: float | None = None  # last finite account value reported
        self._weights: Array = np.zeros(0)
        self._assets: tuple[Any, ...] = ()
        self._timestamp: Any = None  # of the last bar seen

    def __getattr__(self, name: str) -> Any:  # cost_model, capital, ... of the inner broker
        if name.startswith("_") or name in ("broker", "limits"):
            raise AttributeError(name)
        return getattr(self.broker, name)

    @property
    def killed(self) -> bool:
        """Whether the kill switch has tripped (only `rearm()` clears it)."""
        return self._killed

    @property
    def peak(self) -> float | None:
        """The account's highest value since it was opened (or the kill switch re-armed)."""
        return self._peak

    @property
    def drawdown(self) -> float:
        """The account's drawdown from `peak` (0 before any report; 1 if the peak is <= 0)."""
        if self._peak is None or self._value is None:
            return 0.0
        return 1.0 - self._value / self._peak if self._peak > 0 else 1.0

    def reset(self, assets: tuple[Any, ...]) -> AccountState:
        """Reset the inner broker. A tripped kill switch stays tripped (see `rearm`)."""
        state = self.broker.reset(assets)
        self._assets = tuple(assets)
        if getattr(self.broker, "fresh_account", False) is True:
            self._peak = None  # a new account: its drawdown runs from its opening value
        self._check(state, self._timestamp)
        return state

    def execute(self, weights: Array, bar: Bar, context: Mapping[str, Any]) -> ExecutionReport:
        self._timestamp = bar.timestamp
        target = np.asarray(weights, dtype=np.float64)
        if not self._killed and not np.all(np.isfinite(self._weights)):
            self._trip(bar.timestamp, "the account's weights are not known (not finite)")
        if self._killed:
            if np.any(target != 0):
                self.events.append(RiskEvent(bar.timestamp, "kill_switch", "account kept flat"))
            target = np.zeros_like(target)
        else:
            # An asset without a price cannot be traded: the limits bind on the others.
            tradable = np.isfinite(np.asarray(bar.prices, dtype=np.float64))
            target, hits = self.limits._limit(target, self._weights, tradable, self._assets or None)
            for kind, detail in hits:
                self.events.append(RiskEvent(bar.timestamp, kind, detail))
        return self.broker.execute(target, bar, context)

    def mark(self, bar: Bar) -> AccountState:
        self._timestamp = bar.timestamp
        try:
            state = self.broker.mark(bar)
        except Exception:
            # The account is unknown until a mark or reset succeeds: an order before then
            # trips the kill switch (fails closed) instead of trading on stale weights.
            self._weights = np.full(np.shape(self._weights), np.nan)
            raise
        self._check(state, bar.timestamp)
        return state

    def rearm(self) -> None:
        """Clear a tripped kill switch, so that the account may trade again.

        An operator's decision: from now on the drawdown is measured from the account's
        value now (its last report), not from the peak before the trip, which would trip
        the switch again at once. Recorded in `events` as `"rearm"`. Does nothing if the
        kill switch has not tripped.
        """
        if not self._killed:
            return
        self._killed = False
        self._peak = self._value
        self.events.append(
            RiskEvent(self._timestamp, "rearm", "kill switch re-armed: drawdown measured from here")
        )

    def _check(self, state: AccountState, timestamp: Any) -> None:
        """Track the account; trip the kill switch on a non-finite report or the drawdown limit."""
        equity, value = _float_or_nan(state.equity), _float_or_nan(state.value)
        try:
            weights = np.asarray(state.weights, dtype=np.float64)
        except (TypeError, ValueError):
            weights = np.full(np.shape(self._weights), np.nan)
        self._weights = weights
        if not (math.isfinite(equity) and math.isfinite(value) and np.all(np.isfinite(weights))):
            self._trip(
                timestamp,
                f"the broker reported a non-finite account (equity {equity!r}, value {value!r})",
            )
            return
        if value <= 0:
            self._trip(timestamp, f"the account has no value left ({value!r})")
            return
        self._value = value
        self._peak = value if self._peak is None else max(self._peak, value)
        limit = self.limits.max_drawdown
        if limit is not None and not self._killed and self.drawdown >= limit:
            self._trip(timestamp, f"drawdown {self.drawdown:.1%} >= {limit:.1%}")

    def _trip(self, timestamp: Any, detail: str) -> None:
        if not self._killed:
            self._killed = True
            self.events.append(RiskEvent(timestamp, "kill_switch", detail))

    def __repr__(self) -> str:
        return f"GuardedBroker({self.broker!r}, {self.limits!r})"


def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


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
