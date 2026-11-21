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
from collections import deque
from collections.abc import Sequence
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

_COST_FIELDS = {
    "volume": ("dollar_volume", "volume"),
    "spread": ("spread",),
    "borrow_rate": ("borrow_rate",),
}


class LiveMarketEnv(Env[dict[str, Any], Any]):
    """Trade a paper or live account with the Market-v1 interface.

    **reset()** asks the broker for a new account (`Broker.reset`) and reads the first
    `lookback + 1` bars, then observes the last one. **step(action)** maps the action to
    target weights (the action map sees which assets are tradable now and the current
    weights), executes them at the current bar's close (`"close"`), or the previous
    decision's targets (`"next_close"`, the default: the order placed now fills at the next
    close, and the observation shows it as pending), reads the next bar, and lets the
    broker mark the account. The reward is the reward function of the account's growth.

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
        self._field_shapes: dict[str, tuple[int, ...]] = {}
        self._market_shapes: dict[str, tuple[int, ...]] = {}
        self.observation_space = self.builder.space(_SpaceInfo(self))
        self.action_space = self.action_map.space(self.n_slots)
        self._bars: deque[Bar] = deque(maxlen=self._lookback + 1)
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
        """Trailing shape of a per-asset bar field (read from the first bar; () until then)."""
        return self._field_shapes.get(name, ())

    def market_shape(self, name: str) -> tuple[int, ...]:
        """Trailing shape of a market-wide bar series (() until a bar is read)."""
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
        names = set(self.builder.fields) | set(getattr(self.action_map, "fields", ()))
        for keys in _COST_FIELDS.values():
            names |= {k for k in keys if k in bars[-1].fields}
        if self.shortable is not None:
            names.add(self.shortable)
        fields = {
            name: np.stack([np.asarray(b.fields[name], dtype=np.float64) for b in bars])
            for name in names
        }
        fields["close"] = prices
        value = prices[1:]
        actions = bars[-1].corporate_actions
        if actions and "dividend" in bars[-1].fields:
            div = np.stack([np.asarray(b.fields["dividend"], dtype=np.float64) for b in bars[1:]])
            value = value + np.nan_to_num(div, nan=0.0)
        if actions and "split" in bars[-1].fields:
            ratio = np.stack([np.asarray(b.fields["split"], dtype=np.float64) for b in bars[1:]])
            value = value * np.where(np.isfinite(ratio), ratio, 1.0)
        returns = np.full(prices.shape, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            returns[1:] = value / prices[:-1] - 1.0
        market = {
            name: np.stack([np.asarray(b.exogenous[name], dtype=np.float64) for b in bars])
            for name in self.builder.market_fields
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

    # ------------------------------------------------------------------
    # Env API
    # ------------------------------------------------------------------

    def _read(self) -> Bar | None:
        bar = self.feed.next_bar()
        if bar is not None:
            prices = np.asarray(bar.prices, dtype=np.float64)
            if prices.shape != (self.n_slots,):
                raise ValueError(f"A bar has {prices.shape} prices for {self.n_slots} assets.")
            for name, value in bar.fields.items():
                self._field_shapes.setdefault(name, tuple(np.shape(value)[1:]))
            for name, value in bar.exogenous.items():
                self._market_shapes.setdefault(name, tuple(np.shape(value)))
        return bar

    def reset(
        self, *, seed: Any = None, options: dict[str, Any] | None = None
    ) -> ResetResult[dict[str, Any]]:
        self._set_rng(seed)
        self._bars.clear()
        while len(self._bars) < self._lookback + 1:
            bar = self._read()
            if bar is None:
                need = self._lookback + 1
                raise RuntimeError(
                    f"The feed ended before the {need} bars a first observation needs."
                )
            self._bars.append(bar)
        state = self.broker.reset(self.assets)
        self._sync(state)
        self._peak = self._equity
        self._pending = np.zeros(self.n_slots)
        self.reward_fn.reset()
        self._t, self._started, self._over = 0, True, False
        panel = self._panel()
        return ResetResult(
            self._observe(panel), self._info(panel, turnover=0.0, cost=0.0, financing=0.0)
        )

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
        row = np.array([len(self._bars) - 1])
        needs = frozenset(
            getattr(getattr(self.broker, "cost_model", None), "requires", frozenset())
        )
        holding = getattr(self.broker, "holding_cost", None)
        if holding is not None:
            needs |= holding.requires
        context = {
            k: np.asarray(v)[0]
            for k, v in cost_context(
                backend=_NUMPY,
                panel=panel,
                paths=np.zeros(1, dtype=np.int64),
                slots=np.arange(self.n_slots)[None],
                rows=row,
                decision=row,
                lookback=self._lookback,
                needs=needs,
                equity=np.array([getattr(self.broker, "capital", 1.0) * self._equity]),
            ).items()
        }
        report = self.broker.execute(held, bar, context)
        if self.execution == "next_close":
            self._pending = target
        previous = self._equity
        nxt = self._read()
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
            weights=held,
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


def _lane(tree: Any) -> Any:
    if isinstance(tree, dict):
        return {k: _lane(v) for k, v in tree.items()}
    return np.asarray(tree)[0].copy()


__all__ = ["LiveMarketEnv"]
