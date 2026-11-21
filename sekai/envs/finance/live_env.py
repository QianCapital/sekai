"""`LiveMarketEnv`: the Market-v1 interface over a data feed and a broker.

A policy trained on `Market-v1` (`BatchMarketEnv`) trades a paper or live account with
the same observations and actions: this environment keeps the last `lookback + 1` bars
of a `sekai.finance.live.DataFeed`, builds the observation with the same
`ObservationBuilder`, turns actions into target weights with the same `ActionMap`, and
sends them to a `sekai.finance.live.Broker`. Each `step()` executes an order and then
waits for the next bar, at which the broker marks the account. With a `ReplayFeed` of a
history and a `PaperBroker` with the same costs, it reproduces `Market-v1` on that history
step for step, which is how the bridge is tested.
"""

from __future__ import annotations

import copy
import operator
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from sekai.backend import get_backend
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.envs.finance.batch_market import cost_context
from sekai.error import InvalidAction, ResetRequired
from sekai.finance.actions import ActionContext, ActionMap, LongOnlyScores, TargetWeights
from sekai.finance.live import AccountState, Bar, Broker, DataFeed
from sekai.finance.observations import MarketObservation, MarketView, ObservationBuilder
from sekai.finance.panel import MarketPanel
from sekai.finance.rewards import LogReturn, RewardFunction, StepOutcome

_NUMPY = get_backend("numpy")

# The bar fields that cost and holding models read, when a bar has them (as Market-v1
# reads them from its data).
_COST_FIELDS = {
    "volume": ("dollar_volume", "volume"),
    "spread": ("spread",),
    "borrow_rate": ("borrow_rate",),
    "volatility": ("volatility",),
}


class LiveMarketEnv(Env[dict[str, Any], Any]):
    """Trade a paper or live account with the Market-v1 interface.

    **reset()** asks the broker for the account first (`Broker.reset`: a new paper
    account, or the live one), then reads bars until it holds `lookback + 1` of them and
    the newest is a bar to trade on (not a `warmup` bar), and observes it. The bars carry
    over from one episode to the next: a later reset continues from the newest bar,
    without waiting for new ones, so a live account never goes unwatched between
    episodes. While reset() reads bars to trade on, an account that already holds
    positions holds them bar by bar (an order for its current weights, then a mark), so
    the broker, and a `GuardedBroker`, mark and check every bar. Once the feed has ended,
    reset() raises RuntimeError.

    **step(action)** maps the action to target weights (the action map sees which
    assets are tradable now and the current weights), executes them at the current
    bar's close (`"close"`), or the previous decision's targets (`"next_close"`, the
    default: the order placed now fills at the next close, and the observation shows it
    as pending), reads the next bar, and lets the broker mark the account. The reward is
    the reward function of the account's growth, over the weights the broker reports it
    executed (`ExecutionReport.weights`, which a `GuardedBroker` may have limited).

    **Bars** are checked as they arrive: their prices (`Bar.validate`: NaN, or finite
    and positive), and the fields and market series the environment reads, which must
    be present and of the shapes its observation space was built for: `field_shapes`
    and `exogenous_shapes`, by default the feed's attributes of those names (a
    `ReplayFeed` has them), else one value per asset (per field) and one value (per
    series). A bar that fails raises ValueError; after one in `step()`, call `reset()`
    before stepping on.

    The universe is the feed's assets, one slot each (an asset with a NaN price cannot be
    traded). **Termination**: equity wiped out, or a drawdown of `max_drawdown`.
    **Truncation**: after `episode_length` steps, or when the feed ends (then the step's
    reward is 0 and `info["feed_ended"]` is True).

    Args:
        feed: The bars.
        broker: The account (`PaperBroker`, a `GuardedBroker`, or an adapter to a real one).
        lookback: Bars in each observation window.
        episode_length: Steps per episode (None: until the feed ends).
        features: Per-asset bar fields to observe (default observation builder).
        exogenous: Market-wide bar series to observe (default observation builder).
        execution: "next_close" (default) or "close".
        action: An `ActionMap` (default: `LongOnlyScores`, or `TargetWeights` with
            `allow_short`).
        observation: An `ObservationBuilder` (replaces `features` / `exogenous`).
        reward: Reward function (default `LogReturn()`).
        allow_short: Long/short default action.
        max_leverage: Gross exposure cap of the default action.
        shortable: Name of a per-asset field, nonzero where shorting is allowed.
        max_drawdown: Terminate at this drawdown (in (0, 1]).
        dt: Years per bar (reported in info).
        field_shapes: The trailing shape of each per-asset bar field, `(k,)` for a field
            with k values per asset (default: `feed.field_shapes`, if the feed has it).
            A field not listed has one value per asset.
        exogenous_shapes: The shape of each market-wide bar series, `(k,)` or `(k, m)`
            for a vector or matrix (default: `feed.exogenous_shapes`, if the feed has
            it). A series not listed is one value.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_fps": 10}

    def __init__(
        self,
        feed: DataFeed,
        broker: Broker,
        *,
        lookback: int = 20,
        episode_length: int | None = None,
        features: Sequence[str] = (),
        exogenous: Sequence[str] = (),
        execution: str = "next_close",
        action: ActionMap | None = None,
        observation: ObservationBuilder | None = None,
        reward: RewardFunction | None = None,
        allow_short: bool = False,
        max_leverage: float = 1.0,
        shortable: str | None = None,
        max_drawdown: float | None = None,
        dt: float = 1 / 252,
        field_shapes: Mapping[str, Sequence[int]] | None = None,
        exogenous_shapes: Mapping[str, Sequence[int]] | None = None,
    ) -> None:
        if execution not in ("close", "next_close"):
            raise ValueError(f"execution must be 'close' or 'next_close', got {execution!r}.")
        if lookback < 1:
            raise ValueError("lookback must be >= 1.")
        if episode_length is not None and episode_length < 1:
            raise ValueError("episode_length must be >= 1.")
        if observation is not None and (features or exogenous):
            raise ValueError("features= and exogenous= configure the default observation.")
        if max_drawdown is not None and not 0.0 < max_drawdown <= 1.0:
            raise ValueError(f"max_drawdown must be in (0, 1], got {max_drawdown!r}.")
        self.feed = feed
        self.broker = broker
        self.assets = tuple(feed.assets)
        self._lookback = int(lookback)
        self._episode_length = episode_length
        self.execution = execution
        self.builder = observation or MarketObservation(tuple(features), tuple(exogenous))
        self.action_map = action or (
            TargetWeights(max_leverage) if allow_short else LongOnlyScores(max_leverage)
        )
        self.reward_fn = copy.deepcopy(reward) if reward is not None else LogReturn()
        self.shortable = shortable
        self.max_drawdown = max_drawdown
        self.dt = float(dt)
        self.max_assets = self.n_total = len(self.assets)
        if field_shapes is None:
            field_shapes = getattr(feed, "field_shapes", None)
        if exogenous_shapes is None:
            exogenous_shapes = getattr(feed, "exogenous_shapes", None)
        self._field_shapes = _shapes("field_shapes", field_shapes)
        self._market_shapes = _shapes("exogenous_shapes", exogenous_shapes)
        # What the environment reads from every bar: per-asset fields (observation, action
        # map, shortable flag) and market series (observation).
        shortable_field = (shortable,) if shortable is not None else ()
        self._asset_fields = tuple(
            dict.fromkeys(
                (*self.builder.fields, *getattr(self.action_map, "fields", ()), *shortable_field)
            )
        )
        self._market_fields = tuple(self.builder.market_fields)
        self.observation_space = self.builder.space(_SpaceInfo(self))
        self.action_space = self.action_map.space(self.n_slots)
        self._bars: deque[Bar] = deque(maxlen=self._lookback + 1)
        self._exhausted = False  # the feed has returned None
        self._trading = False  # a bar to trade on has arrived (warm-up bars come first)
        self._started = False
        self._over = False
        self._t = 0
        self._equity = 1.0
        self._peak = 1.0
        self._weights: NDArray[np.float64] = np.zeros(self.n_slots)
        self._pending: NDArray[np.float64] = np.zeros(self.n_slots)

    # ------------------------------------------------------------------
    # Introspection (as BatchMarketEnv)
    # ------------------------------------------------------------------

    @property
    def lookback(self) -> int:
        return self._lookback

    @property
    def n_slots(self) -> int:
        return len(self.assets)

    def field_shape(self, name: str) -> tuple[int, ...]:
        """Trailing shape of a per-asset bar field: as declared (`field_shapes`), else ()."""
        return self._field_shapes.get(name, ())

    def market_shape(self, name: str) -> tuple[int, ...]:
        """Shape of a market-wide bar series: as declared (`exogenous_shapes`), else ()."""
        return self._market_shapes.get(name, ())

    @property
    def weights(self) -> NDArray[np.float64]:
        out: NDArray[np.float64] = self._weights.copy()
        return out

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def timestamp(self) -> Any:
        return self._bars[-1].timestamp if self._bars else None

    @property
    def slot_assets(self) -> tuple[Any, ...]:
        return self.assets

    def action_mask(self) -> NDArray[np.int8]:
        """1 for the assets that can be traded now (and any extra action entries)."""
        mask = np.isfinite(self._bars[-1].prices).astype(np.int8)
        extra = int(self.action_space.shape[0]) - self.n_slots  # type: ignore[index]
        return np.append(mask, np.ones(extra, dtype=np.int8)) if extra > 0 else mask

    # ------------------------------------------------------------------
    # The rolling window as a one-path panel
    # ------------------------------------------------------------------

    def _panel(self) -> MarketPanel:
        bars = list(self._bars)
        prices = np.stack([b.prices for b in bars]).astype(np.float64)
        names = set(self._asset_fields)
        for keys in _COST_FIELDS.values():
            names |= {k for k in keys if k in bars[-1].fields}

        def stacked(name: str, rows: list[Bar]) -> NDArray[np.float64]:
            # A cost field or a corporate action that an older bar lacks is missing there.
            missing = np.full((self.n_slots, *self.field_shape(name)), np.nan)
            return np.stack([np.asarray(b.fields.get(name, missing), np.float64) for b in rows])

        fields = {name: stacked(name, bars) for name in names}
        fields["close"] = prices
        value = prices[1:]
        actions = bars[-1].corporate_actions
        if actions and "dividend" in bars[-1].fields:
            value = value + np.nan_to_num(stacked("dividend", bars[1:]), nan=0.0)
        if actions and "split" in bars[-1].fields:
            ratio = stacked("split", bars[1:])
            value = value * np.where(np.isfinite(ratio), ratio, 1.0)
        returns = np.full(prices.shape, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            returns[1:] = value / prices[:-1] - 1.0
        market = {
            name: np.stack([np.asarray(b.exogenous[name], dtype=np.float64) for b in bars])
            for name in self._market_fields
        }
        return MarketPanel(
            returns=returns[None],
            valid=np.isfinite(prices)[None],
            fields={k: v[None] for k, v in fields.items()},
            exogenous={k: v[None] for k, v in market.items()},
        )

    def _view(self, panel: MarketPanel) -> MarketView:
        row = len(self._bars) - 1
        return MarketView(
            backend=_NUMPY,
            panel=panel,
            paths=np.zeros(1, dtype=np.int64),
            rows=np.array([row]),
            slots=np.arange(self.n_slots)[None],
            lookback=self._lookback,
            episode_length=self._episode_length or 0,
            elapsed=np.array([self._t]),
            weights=self._weights[None],
            equity=np.array([self._equity]),
            peak=np.array([self._peak]),
            pending=self._pending[None] if self.execution == "next_close" else None,
        )

    def _observe(self, panel: MarketPanel) -> dict[str, Any]:
        obs: dict[str, Any] = _lane(self.builder(self._view(panel)))
        return obs

    def _context(self, panel: MarketPanel) -> dict[str, Any]:
        """The inputs of the broker's cost and holding models at the newest bar."""
        row = np.array([len(self._bars) - 1])
        needs = frozenset(
            getattr(getattr(self.broker, "cost_model", None), "requires", frozenset())
        )
        holding = getattr(self.broker, "holding_cost", None)
        if holding is not None:
            needs |= holding.requires
        context = cost_context(
            backend=_NUMPY,
            panel=panel,
            paths=np.zeros(1, dtype=np.int64),
            slots=np.arange(self.n_slots)[None],
            rows=row,
            decision=row,
            # The window is full in step(); while reset() fills it, use what there is.
            lookback=max(1, min(self._lookback, len(self._bars) - 1)),
            needs=needs,
            equity=np.array([getattr(self.broker, "capital", 1.0) * self._equity]),
        )
        return {k: np.asarray(v)[0] for k, v in context.items()}

    # ------------------------------------------------------------------
    # Env API
    # ------------------------------------------------------------------

    def _read(self) -> Bar | None:
        """The feed's next bar, checked (see the class docstring); None once it has ended."""
        bar = self.feed.next_bar()
        if bar is None:
            self._exhausted = True
            return None
        bar.validate(self.assets)
        where = f"The bar at {bar.timestamp!r}"
        if bar.warmup and self._trading:
            raise ValueError(
                f"{where} is a warm-up bar after bars to trade on: warm-up bars (history for "
                "the lookback window) must come before the first bar to trade on."
            )
        n = self.n_slots
        present = [k for keys in _COST_FIELDS.values() for k in keys if k in bar.fields]
        for name in dict.fromkeys((*self._asset_fields, *present)):
            if name not in bar.fields:
                raise ValueError(
                    f"{where} has no field {name!r}, which the environment reads (it has "
                    f"{sorted(map(str, bar.fields))})."
                )
            value = np.asarray(bar.fields[name], dtype=np.float64)
            expected = (n, *self.field_shape(name))
            if value.shape != expected:
                raise ValueError(
                    f"{where} has a field {name!r} of shape {value.shape}; the environment "
                    f"expects {expected}, the shape its observation space was built for "
                    "(declare a field's shape per asset in field_shapes, or give the feed a "
                    "field_shapes attribute)."
                )
            if np.isinf(value).any():
                raise ValueError(f"{where} has infinite values in the field {name!r}; use NaN.")
        for name in self._market_fields:
            if name not in bar.exogenous:
                raise ValueError(
                    f"{where} has no market series {name!r}, which the environment reads (it "
                    f"has {sorted(map(str, bar.exogenous))})."
                )
            value = np.asarray(bar.exogenous[name], dtype=np.float64)
            if value.shape != self.market_shape(name):
                raise ValueError(
                    f"{where} has a market series {name!r} of shape {value.shape}; the "
                    f"environment expects {self.market_shape(name)}, the shape its observation "
                    "space was built for (declare it in exogenous_shapes, or give the feed an "
                    "exogenous_shapes attribute)."
                )
            if np.isinf(value).any():
                raise ValueError(
                    f"{where} has infinite values in the market series {name!r}; use NaN."
                )
        if not bar.warmup:
            self._trading = True
        return bar

    def reset(
        self, *, seed: Any = None, options: dict[str, Any] | None = None
    ) -> ResetResult[dict[str, Any]]:
        self._set_rng(seed)
        if self._exhausted:
            raise RuntimeError("The feed has ended (next_bar() returned None): no bars are left.")
        self._started = False  # until this reset completes
        # The account first, so that the bars read below are marked if it holds positions.
        self._sync(self.broker.reset(self.assets))
        self._pending = np.zeros(self.n_slots)
        need = self._lookback + 1
        while len(self._bars) < need or self._bars[-1].warmup:
            bar = self._read()
            if bar is None:
                raise RuntimeError(
                    f"The feed ended before the {need} bars a first observation needs (the "
                    "newest of them a bar to trade on, not a warm-up bar)."
                )
            if self._bars and not self._bars[-1].warmup and np.any(self._weights != 0):
                self._hold(bar)
            self._bars.append(bar)
        self._peak = self._equity
        self.reward_fn.reset()
        self._t, self._started, self._over = 0, True, False
        panel = self._panel()
        return ResetResult(
            self._observe(panel), self._info(panel, turnover=0.0, cost=0.0, financing=0.0)
        )

    def _hold(self, nxt: Bar) -> None:
        """Hold the account's positions over the period that ends at `nxt` (in reset()).

        An order for the current weights at the newest bar, then a mark at `nxt`: the
        broker marks the positions and a `GuardedBroker` checks them, as on any step.
        """
        panel = self._panel()
        self.broker.execute(self._weights.copy(), self._bars[-1], self._context(panel))
        self._sync(self.broker.mark(nxt))

    def _sync(self, state: AccountState) -> None:
        self._equity = float(state.equity)
        self._weights = np.asarray(state.weights, dtype=np.float64)

    def step(self, action: Any) -> StepResult[dict[str, Any]]:
        if not self._started:
            raise ResetRequired()
        if self._over:
            raise ResetRequired("The episode has ended; call reset() before step().")
        a = np.asarray(action, dtype=np.float64)
        if a.shape != self.action_space.shape or not np.all(np.isfinite(a)):
            raise InvalidAction(action, self.action_space)
        panel = self._panel()
        bar = self._bars[-1]
        tradable = np.isfinite(bar.prices)
        shortable = None
        if self.shortable is not None:
            flag = np.asarray(bar.fields[self.shortable], dtype=np.float64)
            shortable = (np.isfinite(flag) & (flag != 0))[None]
        names = getattr(self.action_map, "fields", ())
        ctx = ActionContext(
            backend=_NUMPY,
            tradable=tradable[None],
            weights=self._weights[None],
            shortable=shortable,
            exposures={
                n: np.nan_to_num(np.asarray(bar.fields[n], dtype=np.float64))[None] for n in names
            }
            or None,
        )
        target = np.asarray(self.action_map(a[None], ctx))[0]
        held = target if self.execution == "close" else np.where(tradable, self._pending, 0.0)
        report = self.broker.execute(held, bar, self._context(panel))
        if self.execution == "next_close":
            self._pending = target
        previous = self._equity
        try:
            nxt = self._read()
        except Exception:
            self._over = True  # an order is out and no mark has come: reset() first
            raise
        if nxt is None:  # the feed ended: no period to settle
            self._over = True
            info = self._info(panel, turnover=report.turnover, cost=report.cost, financing=0.0)
            info["feed_ended"] = True
            return StepResult(self._observe(panel), 0.0, False, True, info)
        state = self.broker.mark(nxt)
        self._sync(state)
        self._bars.append(nxt)
        self._peak = max(self._peak, self._equity)
        growth = self._equity / previous if previous > 0 else 0.0
        outcome = StepOutcome.from_growth(
            growth,
            equity=self._equity,
            turnover=report.turnover,
            cost=report.cost,
            previous_equity=previous,
            weights=np.asarray(report.weights, dtype=np.float64),  # what was executed
        )
        reward = float(self.reward_fn(outcome))
        self._t += 1
        drawdown = 1.0 - self._equity / self._peak if self._peak > 0 else 1.0
        terminated = self._equity <= 0.0 or (
            self.max_drawdown is not None and drawdown >= self.max_drawdown
        )
        truncated = (
            not terminated and self._episode_length is not None and self._t >= self._episode_length
        )
        self._over = terminated or truncated
        panel = self._panel()
        info = self._info(
            panel, turnover=report.turnover, cost=report.cost, financing=state.financing
        )
        return StepResult(self._observe(panel), reward, terminated, truncated, info)

    def _info(
        self, panel: MarketPanel, *, turnover: float, cost: float, financing: float
    ) -> dict[str, Any]:
        w = self._weights
        return {
            "equity": self._equity,
            "turnover": turnover,
            "cost": cost,
            "financing": financing,
            "gross_exposure": float(np.abs(w).sum()),
            "net_exposure": float(w.sum()),
            "drawdown": 1.0 - self._equity / self._peak if self._peak > 0 else 1.0,
            "n_assets": int(np.isfinite(self._bars[-1].prices).sum()),
            "weights": w.copy(),
            "timestamp": self.timestamp,
            "dt": self.dt,
        }

    def __repr__(self) -> str:
        return f"<LiveMarketEnv({self.feed!r}, {self.broker!r})>"


class _SpaceInfo:
    """What an observation builder's `space()` reads, before any bar has arrived."""

    def __init__(self, env: LiveMarketEnv) -> None:
        self.n_slots = env.n_slots
        self.lookback = env.lookback
        self.execution = env.execution
        self._env = env

    def field_shape(self, name: str) -> tuple[int, ...]:
        return self._env.field_shape(name)

    def market_shape(self, name: str) -> tuple[int, ...]:
        return self._env.market_shape(name)


def _shapes(name: str, shapes: Mapping[str, Sequence[int]] | None) -> dict[str, tuple[int, ...]]:
    """Validated `{field: shape}` from a `field_shapes` / `exogenous_shapes` argument."""
    out: dict[str, tuple[int, ...]] = {}
    for key, shape in (shapes or {}).items():
        try:
            dims = tuple(operator.index(d) for d in shape)
        except TypeError:
            raise TypeError(f"{name}[{key!r}] must be a sequence of ints, got {shape!r}.") from None
        if any(d < 0 for d in dims):
            raise ValueError(f"{name}[{key!r}] must not have negative sizes, got {dims}.")
        out[key] = dims
    return out


def _lane(tree: Any) -> Any:
    if isinstance(tree, dict):
        return {k: _lane(v) for k, v in tree.items()}
    return np.asarray(tree)[0].copy()


__all__ = ["LiveMarketEnv"]
