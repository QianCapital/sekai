"""Limit-order-book environments: one agent, or several agents trading in one book.

Both run on `sekai.finance.lob`: a price-time priority matching engine with
background order flow, either simulated (limit orders, market orders and
cancellations arriving as Poisson events) or replayed from a historical message
stream (`MessageReplay`). Time is continuous and event-driven: between two
decisions the book evolves through however many events occur, the agent's
resting orders keep their queue position until they fill or are moved, and the
observation reports what happened in between (recent trades as a padded,
variable-size set).

A decision is carried out in a fixed order: a market order first cancels the
trader's own resting quote on the opposite side (self-trade prevention, as the
cancel-resting mode of most exchanges: a trader never trades with itself), then
executes; then the bid and the ask are kept, moved, placed or cancelled as the
action asks.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence
from fractions import Fraction
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from sekai._types import SeedType
from sekai.core.env import Env
from sekai.core.multi_agent import MultiAgentEnv
from sekai.core.result import MAResetResult, MAStepResult, ResetResult, StepResult
from sekai.envs.finance._common import check_number, check_numbers, episode_over
from sekai.error import InvalidAction, ResetRequired
from sekai.finance.data import Time
from sekai.finance.lob import Fill, MessageReplay, Order, OrderBook, Side, ZeroIntelligenceFlow
from sekai.finance.sampling import EpisodeSampler, RandomStart, as_sampler, sample_start
from sekai.spaces.box import Box
from sekai.spaces.dict import Dict
from sekai.spaces.masked_box import MaskedBox
from sekai.spaces.multi_discrete import MultiDiscrete

SELL, HOLD, BUY = 0, 1, 2
# Decision times closer than this count as simultaneous (the traders then act together).
_COINCIDENT = Fraction(1, 10**9)
# A planned quote that keeps the resting one (planned prices are >= 1 tick).
_KEEP = 0
# Tolerance, in time units, for rounding in clock arithmetic: the episode windows cut
# from a replayed message stream, and an order arriving at the next decision time.
_TIME_TOLERANCE = 1e-9

# An action priced when it is decided, for orders that reach the book later: the market
# order's side (or None), then for the bid and the ask the price to quote (None: no
# quote, _KEEP: keep the resting quote).
_Plan = tuple[Side | None, int | None, int | None]


class _Trader:
    """One agent's account and resting quotes in the book."""

    def __init__(self, owner: int) -> None:
        self.owner = owner
        self.inventory = 0
        self.cash = 0.0
        self.bid_id: int | None = None
        self.ask_id: int | None = None

    def wealth(self, mid: float) -> float:
        return self.cash + self.inventory * mid

    def record(self, fill: Fill) -> None:
        """Book a fill in which this trader took part (as maker or taker)."""
        bought_as_taker = fill.taker_owner == self.owner and fill.taker_side is Side.BID
        sold_as_taker = fill.taker_owner == self.owner and fill.taker_side is Side.ASK
        bought_as_maker = fill.maker_owner == self.owner and fill.taker_side is Side.ASK
        sold_as_maker = fill.maker_owner == self.owner and fill.taker_side is Side.BID
        bought = int(bought_as_taker) + int(bought_as_maker)
        sold = int(sold_as_taker) + int(sold_as_maker)
        self.inventory += (bought - sold) * fill.quantity
        self.cash += (sold - bought) * fill.quantity * fill.price


class _LOBCore:
    """State and mechanics shared by the single- and multi-agent environments."""

    # Defaults for attributes added after 0.2, so that a core pickled by sekai 0.2.0
    # (always with simulated flow, which never reads them) loads and runs as it did.
    sampler: EpisodeSampler = RandomStart()
    slot: float = 1.0
    episode_slots: int = 0
    n_starts: int = 0

    def __init__(
        self,
        *,
        flow: ZeroIntelligenceFlow | MessageReplay | None,
        mid_price: int,
        session_length: float,
        levels: int,
        max_quote_offset: int,
        order_size: int,
        max_inventory: int,
        inventory_penalty: float,
        max_trades: int,
        warmup: float,
        sampler: str | int | EpisodeSampler = "random",
        slot: float = 1.0,
    ) -> None:
        if mid_price < 2 * (max_quote_offset + levels):
            raise ValueError("mid_price must leave room for the quotes.")
        if levels < 1 or max_quote_offset < 1 or order_size < 1 or max_inventory < 1:
            raise ValueError("levels, max_quote_offset, order_size and max_inventory must be >= 1.")
        if max_trades < 1:
            raise ValueError("max_trades must be >= 1.")
        # A replay keeps the position of the book it drives, so each env replays its own.
        self.flow: ZeroIntelligenceFlow | MessageReplay
        if isinstance(flow, MessageReplay):
            self.flow = flow.spawn()
        else:
            self.flow = flow if flow is not None else ZeroIntelligenceFlow()
        self.mid_price = mid_price
        self.session_length = check_number("session_length", session_length, ">0")
        self.levels = levels
        self.max_quote_offset = max_quote_offset
        self.order_size = order_size
        self.max_inventory = max_inventory
        self.inventory_penalty = check_number("inventory_penalty", inventory_penalty, ">=0")
        self.max_trades = max_trades
        self.warmup = check_number("warmup", warmup, ">=0")
        self.book = OrderBook()
        self.trades: deque[tuple[float, int, int]] = deque(maxlen=max_trades)
        self.t0 = 0.0
        # The episodes of a replay are windows of its message stream on a grid of `slot`
        # (the decision interval): window k seeds the book at start_time + k * slot and
        # makes its first decision `warmup` later.
        self.sampler = as_sampler(sampler)
        self.slot = float(slot)
        self.episode_slots = math.ceil(self.session_length / self.slot - 1e-9)
        self.n_starts = 0
        if isinstance(self.flow, MessageReplay):
            replay = self.flow
            span = replay.end_time - replay.start_time
            room = span - self.warmup - self.session_length
            if room < -_TIME_TOLERANCE:
                raise ValueError(
                    f"The messages span {span:g} time units ({replay.start_time:.9g} to "
                    f"{replay.end_time:.9g}) but an episode needs warmup + session_length = "
                    f"{self.warmup + self.session_length:g}."
                )
            self.n_starts = math.floor(max(room, 0.0) / self.slot + _TIME_TOLERANCE) + 1

    # Spaces -----------------------------------------------------------------

    def action_space(self) -> MultiDiscrete:
        k = self.max_quote_offset + 1
        return MultiDiscrete([k, k, 3])

    def observation_space(self) -> Dict:
        return Dict(
            book=Box(-np.inf, np.inf, shape=(self.levels, 4), dtype=np.float32),
            trades=MaskedBox(-np.inf, np.inf, shape=(3,), max_size=self.max_trades),
            agent=Box(-np.inf, np.inf, shape=(6,), dtype=np.float32),
        )

    # Episode ----------------------------------------------------------------

    def start(self, rng: np.random.Generator, options: dict[str, Any] | None = None) -> None:
        """Set up the book for a new episode.

        With a replay, `options["start"]` places the episode (see `_seed_time`); simulated
        flow has no history to start in, so it ignores the option, as sekai 0.2 did.
        """
        flow = self.flow
        if isinstance(flow, MessageReplay):
            seed_time = self._seed_time(flow, rng, options)
            flow.initialise(self.book, rng, start=seed_time)
            if self.warmup > 0:
                flow.advance(self.book, rng, seed_time + self.warmup)
        else:
            flow.initialise(self.book, rng, mid=self.mid_price)
            if self.warmup > 0:
                flow.advance(self.book, rng, self.warmup)
        self.t0 = self.book.time
        self.trades.clear()

    def _seed_time(
        self, replay: MessageReplay, rng: np.random.Generator, options: dict[str, Any] | None
    ) -> float:
        """When the next episode's book is seeded: `warmup` before its first decision.

        `options["start"]` places the first decision: an int is a position on the grid
        of windows, a float or `Time(t)` a time on the messages' clock.
        """
        begin = replay.start_time
        option = options.get("start") if options else None
        if option is None:
            k = sample_start(
                self.sampler,
                rng,
                n_starts=self.n_starts,
                episode_length=self.episode_slots,
                lookback=0,
            )
            return begin + k * self.slot
        first = begin + self.warmup  # the earliest first decision
        last = replay.end_time - self.session_length  # the latest
        if isinstance(option, bool | np.bool_):
            raise TypeError(f"options['start'] must be a position or a time, got {option!r}.")
        if isinstance(option, int | np.integer):
            if not 0 <= option < self.n_starts:
                raise ValueError(
                    f"start {option} is outside the valid positions 0..{self.n_starts - 1} "
                    f"(first decisions at {first:.9g} + k * {self.slot:g}). An int is a "
                    "position on that grid; for a time, pass a float or Time(t)."
                )
            return begin + int(option) * self.slot
        value = option.value if isinstance(option, Time) else option
        decision = check_number("options['start']", value)
        if not first - _TIME_TOLERANCE <= decision <= last + _TIME_TOLERANCE:
            raise ValueError(
                f"start {option!r} is outside the valid first-decision times "
                f"{first:.9g}..{last:.9g}: an episode needs warmup={self.warmup:g} of "
                f"history before its first decision and session_length="
                f"{self.session_length:g} after it."
            )
        return max(decision - self.warmup, begin)

    @property
    def halted(self) -> bool:
        """Whether the replayed market is halted (traders' orders are then not sent)."""
        return isinstance(self.flow, MessageReplay) and self.flow.halted

    @property
    def elapsed(self) -> float:
        return self.book.time - self.t0

    def mid(self) -> float:
        mid = self.book.mid()
        if mid is None:  # the flow keeps both sides alive, but be safe
            return float(self.book.last_price or self.mid_price)
        return mid

    # Actions ----------------------------------------------------------------

    def action_mask(self, trader: _Trader) -> tuple[NDArray[np.int8], ...]:
        k = self.max_quote_offset + 1
        can_buy = trader.inventory + self.order_size <= self.max_inventory
        can_sell = trader.inventory - self.order_size >= -self.max_inventory
        bid = np.ones(k, dtype=np.int8)
        ask = np.ones(k, dtype=np.int8)
        bid[1:] = int(can_buy)
        ask[1:] = int(can_sell)
        market = np.array([int(can_sell), 1, int(can_buy)], dtype=np.int8)
        return bid, ask, market

    @staticmethod
    def parse(action: Any, space: MultiDiscrete) -> tuple[int, int, int]:
        """A decision as (bid level, ask level, market order); InvalidAction if out of `space`."""
        a = np.asarray(action)
        if not space.contains(a):
            raise InvalidAction(action, space)
        bid_level, ask_level, market = (int(x) for x in a)
        return bid_level, ask_level, market

    def apply(self, trader: _Trader, decision: tuple[int, int, int]) -> list[Fill]:
        """Execute a parsed decision and book the trader's side of the fills; returns the fills.

        A market order first cancels the trader's resting quote on the opposite side, so
        that it cannot fill against it (self-trade prevention), then executes; the quotes
        are then updated as the decision asks (a cancelled quote whose level is not 0 is
        placed again, at the back of its queue).
        """
        bid_level, ask_level, market = decision
        fills: list[Fill] = []
        if market == BUY and trader.inventory + self.order_size <= self.max_inventory:
            trader.ask_id = self._cancel(trader.ask_id)  # self-trade prevention
            _, fills = self.book.market(Side.BID, self.order_size, trader.owner)
        elif market == SELL and trader.inventory - self.order_size >= -self.max_inventory:
            trader.bid_id = self._cancel(trader.bid_id)  # self-trade prevention
            _, fills = self.book.market(Side.ASK, self.order_size, trader.owner)
        for fill in fills:
            trader.record(fill)
        self.note_trades(fills)
        can_buy = trader.inventory + self.order_size <= self.max_inventory
        can_sell = trader.inventory - self.order_size >= -self.max_inventory
        bid_level = bid_level if can_buy else 0
        ask_level = ask_level if can_sell else 0
        trader.bid_id, bid_fills = self._quote(trader, Side.BID, bid_level, trader.bid_id)
        trader.ask_id, ask_fills = self._quote(trader, Side.ASK, ask_level, trader.ask_id)
        return fills + bid_fills + ask_fills

    def _cancel(self, order_id: int | None) -> int | None:
        """Cancel a quote if it is still resting; returns the trader's new quote id (None)."""
        if order_id is not None:
            self.book.cancel(order_id)
        return None

    def _quote(
        self, trader: _Trader, side: Side, level: int, current: int | None
    ) -> tuple[int | None, list[Fill]]:
        """Keep, move or cancel a resting quote; level 0 means no quote, 1 joins the best.

        A resting quote already at the requested level is kept, with its queue position:
        at the price the level gives, or, for level 1, at the best price of its side even
        when it is alone there (it is then the best quote, which level 1 asks for).
        """
        resting = self.book.get(current) if current is not None else None
        if level == 0:
            if resting is not None:
                self.book.cancel(resting.order_id)
            return None, []
        price = self._level_price(side, level, resting)
        if resting is not None and self._keeps(side, level, resting, price):
            return resting.order_id, []  # at the requested level: keep the queue position
        if resting is not None:
            self.book.cancel(resting.order_id)
        return self._place(trader, side, max(price, 1))

    def _level_price(self, side: Side, level: int, resting: Order | None) -> int:
        """The price of a quote at `level` (>= 1), measured from the other traders' best.

        `resting` is the trader's own quote on `side`, or None; it does not count.
        """
        best = self._best_excluding(side, resting)
        if best is None:  # no other quotes on this side: one tick inside the opposite best
            opposite = self.book.best_ask() if side is Side.BID else self.book.best_bid()
            if opposite is not None and (resting is None or resting.price != opposite):
                best = opposite - 1 if side is Side.BID else opposite + 1
            else:
                ref = float(self.book.last_price or self.mid_price)
                best = int(np.floor(ref)) if side is Side.BID else int(np.ceil(ref))
        return best - (level - 1) if side is Side.BID else best + (level - 1)

    def _keeps(self, side: Side, level: int, resting: Order | None, price: int) -> bool:
        """Whether a resting quote is already at the requested level (and so stays put)."""
        return resting is not None and (
            resting.price == price or (level == 1 and resting.price == self._best(side))
        )

    def _place(self, trader: _Trader, side: Side, price: int) -> tuple[int | None, list[Fill]]:
        """Send a limit order of `order_size` for the trader and book its fills."""
        order_id, fills = self.book.limit(side, price, self.order_size, trader.owner)
        for fill in fills:  # a quote at the best price can meet another trader's order
            trader.record(fill)
        self.note_trades(fills)
        return order_id, fills

    # Latency: an action priced when it is decided, carried out when it reaches the book.

    def plan(self, trader: _Trader, decision: tuple[int, int, int]) -> _Plan:
        """Price a parsed decision from the book the trader observes now.

        The orders reach the book later (`execute`), so their prices are fixed now, as an
        order message's are: a bid placed at the best price may arrive behind a new best
        bid, or cross a falling ask and trade at once. The quotes are planned as if the
        market order fills; `execute` checks the inventory limits again on arrival.
        """
        bid_level, ask_level, market = decision
        size, limit = self.order_size, self.max_inventory
        buy = market == BUY and trader.inventory + size <= limit
        sell = market == SELL and trader.inventory - size >= -limit
        inventory = trader.inventory + size * (int(buy) - int(sell))
        bid_level = bid_level if inventory + size <= limit else 0
        ask_level = ask_level if inventory - size >= -limit else 0
        bid = self._plan_quote(Side.BID, bid_level, trader.bid_id, cancelled=sell)
        ask = self._plan_quote(Side.ASK, ask_level, trader.ask_id, cancelled=buy)
        return (Side.BID if buy else Side.ASK if sell else None), bid, ask

    def _plan_quote(
        self, side: Side, level: int, current: int | None, *, cancelled: bool
    ) -> int | None:
        """The price to quote on `side`: None for no quote, _KEEP to keep the resting one.

        `cancelled` marks a quote that the action's market order cancels first
        (self-trade prevention): it is placed again, at the back of its queue.
        """
        if level == 0:
            return None
        resting = self.book.get(current) if current is not None else None
        price = self._level_price(side, level, resting)
        if not cancelled and self._keeps(side, level, resting, price):
            return _KEEP
        return max(price, 1)

    def execute(self, trader: _Trader, plan: _Plan) -> list[Fill]:
        """Carry out a planned action as it reaches the book; returns the trader's fills.

        In the order of `apply`: the market order (after cancelling the trader's own
        quote on the opposite side), then the bid and the ask. A quote planned to stay
        is left alone, even if it filled meanwhile (no order was sent for it); a quote
        that moves or goes is cancelled if it still rests, which it may not (a fill can
        beat the cancel). The inventory limits are checked against the inventory on
        arrival.
        """
        market, bid, ask = plan
        size, limit = self.order_size, self.max_inventory
        fills: list[Fill] = []
        if market is Side.BID and trader.inventory + size <= limit:
            trader.ask_id = self._cancel(trader.ask_id)  # self-trade prevention
            _, fills = self.book.market(Side.BID, size, trader.owner)
        elif market is Side.ASK and trader.inventory - size >= -limit:
            trader.bid_id = self._cancel(trader.bid_id)  # self-trade prevention
            _, fills = self.book.market(Side.ASK, size, trader.owner)
        for fill in fills:
            trader.record(fill)
        self.note_trades(fills)
        can_buy = trader.inventory + size <= limit
        can_sell = trader.inventory - size >= -limit
        trader.bid_id, bid_fills = self._send(
            trader, Side.BID, bid if can_buy else None, trader.bid_id
        )
        trader.ask_id, ask_fills = self._send(
            trader, Side.ASK, ask if can_sell else None, trader.ask_id
        )
        return fills + bid_fills + ask_fills

    def _send(
        self, trader: _Trader, side: Side, target: int | None, current: int | None
    ) -> tuple[int | None, list[Fill]]:
        """Bring the trader's quote on `side` to its planned target, as the orders arrive."""
        if target == _KEEP:
            return current, []
        resting = self.book.get(current) if current is not None else None
        if target is None:
            if resting is not None:
                self.book.cancel(resting.order_id)
            return None, []
        if resting is not None:
            if resting.price == target:
                return resting.order_id, []
            self.book.cancel(resting.order_id)
        return self._place(trader, side, target)

    def _best(self, side: Side) -> int | None:
        """The best price on `side`, the trader's own orders included."""
        return self.book.best_bid() if side is Side.BID else self.book.best_ask()

    def _best_excluding(self, side: Side, own: Any) -> int | None:
        """The best price on `side` among other traders' orders (not the trader's own quote).

        Measuring quote levels from this reference keeps a quote that has become the best
        price from chasing itself: from its own price, level k would sit k - 1 ticks behind
        itself and walk away one step at a time.
        """
        best = self._best(side)
        if best is None or own is None or own.price != best:
            return best
        if self.book.volume_at(side, best) > own.quantity:
            return best
        bid_prices, bid_sizes, ask_prices, ask_sizes = self.book.depth(2)
        prices, sizes = (bid_prices, bid_sizes) if side is Side.BID else (ask_prices, ask_sizes)
        others = [int(p) for p, q in zip(prices, sizes) if q > 0 and p != best]
        return others[0] if others else None

    def note_trades(self, fills: Sequence[Fill]) -> None:
        for fill in fills:
            self.trades.append((float(fill.price), fill.quantity, int(fill.taker_side)))

    # Observations -----------------------------------------------------------

    def observe(self, trader: _Trader, space: Dict) -> dict[str, Any]:
        mid = self.mid()
        bp, bs, ap, asz = self.book.depth(self.levels)
        book = np.stack(
            [
                np.where(bs > 0, bp - mid, 0.0),
                bs,
                np.where(asz > 0, ap - mid, 0.0),
                asz,
            ],
            axis=1,
        ).astype(np.float32)
        rows = np.array([(p - mid, q, s) for p, q, s in self.trades], dtype=np.float32)
        trades_space = space["trades"]
        assert isinstance(trades_space, MaskedBox)
        trades = trades_space.pack(rows.reshape(-1, 3))
        bid = self.book.get(trader.bid_id) if trader.bid_id is not None else None
        ask = self.book.get(trader.ask_id) if trader.ask_id is not None else None
        agent = np.array(
            [
                trader.inventory / self.max_inventory,
                max(1.0 - self.elapsed / self.session_length, 0.0),
                float(bid is not None),
                float(ask is not None),
                self.book.queue_ahead(bid.order_id) if bid is not None else 0.0,
                self.book.queue_ahead(ask.order_id) if ask is not None else 0.0,
            ],
            dtype=np.float32,
        )
        return {"book": book, "trades": trades, "agent": agent}

    def info(self, trader: _Trader, fills: int) -> dict[str, Any]:
        mid = self.mid()
        info: dict[str, Any] = {
            "timestamp": self.elapsed,
            "mid_price": mid,
            "spread": self.book.spread(),
            "inventory": trader.inventory,
            "cash": trader.cash,
            "wealth": trader.wealth(mid),
            "fills": fills,
        }
        if isinstance(self.flow, MessageReplay):
            info["time"] = self.book.time  # on the messages' clock
        return info


class LimitOrderBookEnv(Env[dict[str, Any], NDArray[np.int64]]):
    """Make markets or trade in a limit order book, against simulated or historical order flow.

    Every `decision_interval` units of time the agent chooses where to rest a
    bid and an ask (or not to quote), and whether to send a market order. In
    between, background traders submit limit orders, market orders and
    cancellations as Poisson events (`sekai.finance.ZeroIntelligenceFlow`), or
    a historical message stream is replayed (`sekai.finance.MessageReplay`);
    either may fill the agent's resting quotes. A quote left at the same price
    keeps its place in the queue; moving it sends it to the back.

    A decision runs in this order: the market order, if any, after cancelling the
    agent's own quote on the opposite side (self-trade prevention: a market buy
    cancels the agent's resting ask first, a market sell its bid, so the agent never
    trades with itself); then the bid and the ask are updated per the action (a
    quote cancelled for the market order is placed again, at the back of its queue,
    unless its level is 0).

    - **Action**: `MultiDiscrete([K + 1, K + 1, 3])` with `K = max_quote_offset`:
      the bid level (0 = no bid, 1 = join the best bid, k = k - 1 ticks behind
      it), the ask level (likewise), and a market order (0 sell, 1 none, 2 buy)
      of `order_size` lots. Levels are measured from the best price of the other
      traders' orders (one tick inside the opposite best if there are none). A
      resting quote already at the requested level is left alone, with its queue
      position, and a level-1 quote stays put while it is the best price itself,
      even alone there; so an unchanged action moves a quote only to follow the
      other traders' best price, and never pulls a level-1 quote off the best.
      `action_mask()` disables buying at the long inventory limit and selling at
      the short one.
    - **Observation** (`Dict`): `"book"`, `Box (levels, 4)`: bid price offset from
      the mid (ticks), bid size, ask offset, ask size for the best `levels` levels;
      `"trades"`: `MaskedBox` of the last `max_trades` trades (price offset from
      the current mid, size, aggressor side +1 buy / -1 sell), a variable-size set;
      `"agent"`: inventory / `max_inventory`, fraction of the session remaining,
      whether a bid and an ask are resting, and the volume queued ahead of each.
    - **Reward**: change in mark-to-market wealth (cash + inventory * mid, in
      ticks) minus the running penalty `inventory_penalty * inventory^2 * dt` for the
      time `dt` the step covers.
    - **Termination**: at the end of the session (the clock is observed); `step()`
      after it raises `ResetRequired`.
    - **info**: `timestamp` (session time), `dt` (the time the step covered:
      `decision_interval`, or less on a final, shorter step), `mid_price`, `spread`,
      `inventory`, `cash`, `wealth`, `fills` (agent fills this step); with a replay,
      also `time`, the book's time on the messages' clock.

    **Historical replay.** With `flow=MessageReplay(...)` the background flow is a
    historical message stream (`sekai.finance.read_lobster` loads LOBSTER files) and
    each episode is a window of it: the book is seeded with its historical state
    `warmup` before the first decision, and the session's messages are replayed
    around the agent's orders (the replay model and its biases are in
    `MessageReplay`). `sampler` places the windows on a grid of `decision_interval`
    starting at the first message (position k seeds the book at
    `start_time + k * decision_interval`); `reset(options={"start": t})` makes the
    first decision at time `t` (a float or `Time(t)`), or at grid position `t` (an
    int); simulated flow, which has no history, ignores it. While the replayed market
    is halted, the agent's orders are not sent.

    **Latency.** An action decided at time t reaches the book at `t + latency`: the
    background flow runs up to that time first (messages stamped at or before it come
    first), then the action's cancels and orders are applied. Its prices are fixed at
    decision time, from the book the agent observed, so a quote can arrive behind a
    new best price or cross the spread, and a cancel can arrive after the fill it
    meant to avoid; a quote the action keeps is left alone. Inventory limits are
    checked on arrival. Observations are taken at decision time; to delay them too,
    wrap the env in `sekai.DelayObservation`. With `latency=0` (the default) an action
    is carried out at once, its quotes priced after its market order.

    Args:
        flow: Background order flow: a `ZeroIntelligenceFlow` (default
            `ZeroIntelligenceFlow()`) or a `MessageReplay` of historical messages (the
            env replays its own copy).
        mid_price: Initial mid price in ticks (simulated flow only).
        session_length: Length of the trading session (time units of the flow rates;
            seconds for a replay).
        decision_interval: Time between decisions.
        levels: Book levels observed.
        max_quote_offset: Farthest quote level K.
        order_size: Lots per quote or market order.
        max_inventory: Absolute inventory limit.
        inventory_penalty: Running penalty per squared lot of inventory per unit of time.
        max_trades: Recent trades kept in the observation.
        warmup: Time the book evolves before the session starts.
        latency: Time an action takes to reach the book, in `[0, decision_interval]`:
            a number, or a function of the env's `numpy.random.Generator` drawn for
            every action (random latency; a draw outside that range raises ValueError).
        sampler: How a replay's episode windows are chosen: "random" (default),
            "sequential", a fixed int position, or a `sekai.finance.EpisodeSampler`
            (see `sekai.finance.sampling`). Unused with simulated flow.

    Raises:
        ValueError: For an invalid parameter, a constant latency longer than
            `decision_interval`, or a replay too short for `warmup + session_length`.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_fps": 10}
    AGENT: ClassVar[int] = 0

    # Defaults for attributes added after 0.2, so that an env pickled by sekai 0.2.0 (which
    # had no latency) loads and continues exactly as it would have.
    latency: float | Callable[[np.random.Generator], float] = 0.0
    _latency_fn: Callable[[np.random.Generator], float] | None = None

    def __init__(
        self,
        *,
        flow: ZeroIntelligenceFlow | MessageReplay | None = None,
        mid_price: int = 10_000,
        session_length: float = 300.0,
        decision_interval: float = 1.0,
        levels: int = 5,
        max_quote_offset: int = 5,
        order_size: int = 1,
        max_inventory: int = 20,
        inventory_penalty: float = 0.01,
        max_trades: int = 16,
        warmup: float = 20.0,
        latency: float | Callable[[np.random.Generator], float] = 0.0,
        sampler: str | int | EpisodeSampler = "random",
    ) -> None:
        decision_interval = check_number("decision_interval", decision_interval, ">0")
        self._latency_fn = None
        if callable(latency):
            self._latency_fn = latency
            self.latency = latency
        else:
            delay = check_number("latency", latency, ">=0")
            if delay > decision_interval:
                raise ValueError(
                    f"latency ({delay:g}) must not exceed decision_interval "
                    f"({decision_interval:g}): an action reaches the book before the next one."
                )
            self.latency = delay
        self.core = _LOBCore(
            flow=flow,
            mid_price=mid_price,
            session_length=session_length,
            levels=levels,
            max_quote_offset=max_quote_offset,
            order_size=order_size,
            max_inventory=max_inventory,
            inventory_penalty=inventory_penalty,
            max_trades=max_trades,
            warmup=warmup,
            sampler=sampler,
            slot=decision_interval,
        )
        self.decision_interval = float(decision_interval)
        self.n_steps = int(np.ceil(session_length / decision_interval - 1e-9))
        # A step is full when what is left of the session is a whole interval, within the
        # same tolerance as n_steps.
        self._full_step = Fraction(self.decision_interval) * (1 - Fraction(1, 10**9))
        self._obs_space = self.core.observation_space()
        self._act_space = self.core.action_space()
        self.observation_space = self._obs_space
        self.action_space = self._act_space
        self.trader = _Trader(self.AGENT)
        self._started = False
        self._k = 0

    @property
    def book(self) -> OrderBook:
        """The live order book (read-only use, please)."""
        return self.core.book

    @property
    def elapsed_steps(self) -> int:
        return self._k

    def action_mask(self) -> tuple[NDArray[np.int8], ...]:
        """Per-dimension masks: no buying at the long limit, no selling at the short one."""
        return self.core.action_mask(self.trader)

    def reset(
        self, *, seed: SeedType = None, options: dict[str, Any] | None = None
    ) -> ResetResult[dict[str, Any]]:
        self._set_rng(seed)
        self.core.start(self.rng, options)
        self.trader = _Trader(self.AGENT)
        self._started = True
        self._k = 0
        info = {**self.core.info(self.trader, 0), "dt": self.decision_interval}
        return ResetResult(self.core.observe(self.trader, self._obs_space), info)

    def step(self, action: NDArray[np.int64]) -> StepResult[dict[str, Any]]:
        if not self._started:
            raise ResetRequired()
        if self._k >= self.n_steps:
            raise episode_over()
        core = self.core
        decision = core.parse(action, self._act_space)
        before = self.trader.wealth(core.mid())
        delay = self._latency()
        if delay == 0.0:  # carried out at once
            n_fills = 0 if core.halted else len(core.apply(self.trader, decision))
            self._k += 1
            until = core.t0 + min(self._k * self.decision_interval, core.session_length)
        else:  # priced now, carried out when it reaches the book (if before the close)
            plan = core.plan(self.trader, decision)
            self._k += 1
            until = core.t0 + min(self._k * self.decision_interval, core.session_length)
            arrival = core.book.time + delay
            n_fills = 0
            if arrival <= until + _TIME_TOLERANCE:  # a latency <= decision_interval, rounded
                n_fills += self._run_flow(min(arrival, until))
                if not core.halted:
                    n_fills += len(core.execute(self.trader, plan))
        n_fills += self._run_flow(until)
        # The time the step covered, from the schedule (exact, not a difference of clock
        # readings): decision_interval, or what is left of the session on a shorter last step.
        left = Fraction(core.session_length) - (self._k - 1) * Fraction(self.decision_interval)
        dt = self.decision_interval if left >= self._full_step else float(left)
        penalty = core.inventory_penalty * self.trader.inventory**2 * dt
        reward = self.trader.wealth(core.mid()) - before - penalty
        info = {**core.info(self.trader, n_fills), "dt": dt}
        return StepResult(
            observation=core.observe(self.trader, self._obs_space),
            reward=float(reward),
            terminated=self._k >= self.n_steps,
            truncated=False,
            info=info,
        )

    def _latency(self) -> float:
        """The latency of the next action: the constant, or a validated draw."""
        if self._latency_fn is None:
            assert isinstance(self.latency, float)
            return self.latency
        value = self._latency_fn(self.rng)
        try:
            delay = float(value)
        except (TypeError, ValueError):
            delay = math.nan
        if not (math.isfinite(delay) and 0.0 <= delay <= self.decision_interval):
            raise ValueError(
                f"latency() returned {value!r}; a latency must be a finite number in "
                f"[0, decision_interval] = [0, {self.decision_interval:g}] (cap long draws, "
                f"e.g. min(draw, {self.decision_interval:g}))."
            )
        return delay

    def _run_flow(self, until: float) -> int:
        """Advance the background flow to `until` and book the agent's fills (returns how many)."""
        core = self.core
        fills = core.flow.advance(core.book, self.rng, until)
        core.note_trades(fills)
        n_fills = 0
        for fill in fills:
            if self.AGENT in (fill.maker_owner, fill.taker_owner):
                self.trader.record(fill)
                n_fills += 1
        return n_fills


class LimitOrderBookMarket(MultiAgentEnv[dict[str, Any], NDArray[np.int64], str]):
    """Several RL traders in one limit order book, each acting on its own clock.

    An event-driven multi-agent market: trader i decides every
    `decision_intervals[i]` units of time (optionally jittered), so at any
    moment only some traders act. Each `step()` takes actions from exactly the
    traders whose turn it is, advances the book (background flow and every
    trader's resting orders) to the next decision time of any trader, and
    returns:

    - `observations` only for the traders that act next (the event-driven
      convention: the keys tell you who moves);
    - `rewards`, `terminated`, `truncated` and `info` for every trader, so
      rewards earned between a trader's decisions are not lost (sum them until
      its next observation).

    Traders' orders interact: one trader's resting quote can be filled by
    another's market order (but never by its own: self-trade prevention, as in
    LimitOrderBookEnv). The inventory penalty accrues per unit of time, so a
    trader's penalty does not depend on how often the others act; `info[agent]["dt"]`
    is the time the step covered. Traders whose decision times coincide act in name
    order. Every action is validated before any is carried out, so a step that
    raises `InvalidAction` changes nothing. Actions, observations, masks
    (`action_mask(agent)`) and rewards per trader are as in LimitOrderBookEnv. At the
    end of the session every trader is terminated (and `agents` is empty); `step()`
    then raises `ResetRequired`.

    With `flow=MessageReplay(...)` the traders trade against historical order flow, in
    windows of the message stream chosen as in LimitOrderBookEnv (`sampler`, and
    `reset(options={"start": ...})`), on a grid of the shortest decision interval; while
    the replayed market is halted, no trader's orders are sent. Orders reach the book
    at once: this environment has no `latency`.

    Args:
        n_agents: Number of traders, named "trader_0", "trader_1", ...
        decision_intervals: Each trader's time between decisions (one value for all,
            or one per trader).
        jitter: Each wait is multiplied by a random factor in [1 - jitter, 1 + jitter].
        flow: Background order flow: a `ZeroIntelligenceFlow` (default
            `ZeroIntelligenceFlow()`) or a `MessageReplay` of historical messages.
        mid_price: Initial mid price in ticks (simulated flow only).
        session_length: Length of the trading session.
        levels: Book levels observed.
        max_quote_offset: Farthest quote level K.
        order_size: Lots per quote or market order.
        max_inventory: Absolute inventory limit per trader.
        inventory_penalty: Running penalty per squared lot of inventory per unit of time.
        max_trades: Recent trades kept in each observation.
        warmup: Time the book evolves before the session starts.
        sampler: How a replay's episode windows are chosen (as in LimitOrderBookEnv).
    """

    def __init__(
        self,
        n_agents: int = 2,
        *,
        decision_intervals: float | Sequence[float] = 1.0,
        jitter: float = 0.0,
        flow: ZeroIntelligenceFlow | MessageReplay | None = None,
        mid_price: int = 10_000,
        session_length: float = 300.0,
        levels: int = 5,
        max_quote_offset: int = 5,
        order_size: int = 1,
        max_inventory: int = 20,
        inventory_penalty: float = 0.01,
        max_trades: int = 16,
        warmup: float = 20.0,
        sampler: str | int | EpisodeSampler = "random",
    ) -> None:
        if n_agents < 1:
            raise ValueError("n_agents must be >= 1.")
        intervals = np.broadcast_to(
            check_numbers("decision_intervals", decision_intervals, ">0"), (n_agents,)
        )
        if not 0.0 <= check_number("jitter", jitter) < 1.0:
            raise ValueError(f"jitter must be in [0, 1), got {jitter!r}.")
        self.core = _LOBCore(
            flow=flow,
            mid_price=mid_price,
            session_length=session_length,
            levels=levels,
            max_quote_offset=max_quote_offset,
            order_size=order_size,
            max_inventory=max_inventory,
            inventory_penalty=inventory_penalty,
            max_trades=max_trades,
            warmup=warmup,
            sampler=sampler,
            slot=float(intervals.min()),
        )
        self.intervals = intervals.copy()
        self.jitter = jitter
        self.possible_agents = [f"trader_{i}" for i in range(n_agents)]
        self.agents = list(self.possible_agents)
        self._obs_space, self._act_space = self.core.observation_space(), self.core.action_space()
        self.observation_spaces = dict.fromkeys(self.possible_agents, self._obs_space)
        self.action_spaces = dict.fromkeys(self.possible_agents, self._act_space)
        self.traders = {a: _Trader(i) for i, a in enumerate(self.possible_agents)}
        self.next_decision = np.zeros(n_agents)
        # The session clock and decision times, kept exact so that a step's dt is an
        # exact difference of decision times (no drift from summing waits in floats).
        self._clock = Fraction(0)
        self._next = [Fraction(0)] * n_agents
        self.acting: list[str] = []
        self._started = False

    @property
    def book(self) -> OrderBook:
        return self.core.book

    def action_mask(self, agent: str) -> tuple[NDArray[np.int8], ...]:
        return self.core.action_mask(self.traders[agent])

    def _wait(self, i: int) -> float:
        if self.jitter == 0.0:
            return float(self.intervals[i])
        return float(self.intervals[i] * self.rng.uniform(1 - self.jitter, 1 + self.jitter))

    def _observe(self, agent: str) -> dict[str, Any]:
        return self.core.observe(self.traders[agent], self._obs_space)

    def reset(
        self, *, seed: SeedType = None, options: dict[str, Any] | None = None
    ) -> MAResetResult[dict[str, Any], str]:
        self._set_rng(seed)
        self.core.start(self.rng, options)
        self.agents = list(self.possible_agents)
        self.traders = {a: _Trader(i) for i, a in enumerate(self.possible_agents)}
        self.next_decision = np.zeros(len(self.possible_agents))
        self._clock = Fraction(0)
        self._next = [Fraction(0)] * len(self.possible_agents)
        self.acting = list(self.agents)
        self._started = True
        observations = {a: self._observe(a) for a in self.acting}
        info = {a: {**self.core.info(self.traders[a], 0), "dt": 0.0} for a in self.agents}
        return MAResetResult(observations, info)

    def step(self, actions: dict[str, Any]) -> MAStepResult[dict[str, Any], str]:
        if not self._started:
            raise ResetRequired()
        if not self.agents:
            raise episode_over()
        if set(actions) != set(self.acting):
            raise ValueError(
                f"Expected actions for the acting traders {sorted(self.acting)}, "
                f"got {sorted(actions)}."
            )
        core = self.core
        # Every action is checked before any is carried out: an invalid one changes nothing.
        decisions = {agent: core.parse(actions[agent], self._act_space) for agent in self.acting}
        mid = core.mid()
        before = {a: self.traders[a].wealth(mid) for a in self.agents}
        n_fills = dict.fromkeys(self.agents, 0)
        owners = {t.owner: a for a, t in self.traders.items()}
        for agent in self.acting:
            trader = self.traders[agent]
            own_fills = [] if core.halted else core.apply(trader, decisions[agent])
            n_fills[agent] += len(own_fills)
            for fill in own_fills:  # book the other side for a trader whose quote was hit
                maker = owners.get(fill.maker_owner)
                if maker is not None and maker != agent:
                    self.traders[maker].record(fill)
                    n_fills[maker] += 1
            i = self.possible_agents.index(agent)
            self._next[i] = self._clock + Fraction(self._wait(i))
            self.next_decision[i] = float(self._next[i])

        exact_until = min(min(self._next), Fraction(core.session_length))
        dt = float(exact_until - self._clock)
        self._clock = exact_until
        until = float(exact_until)
        fills = core.flow.advance(core.book, self.rng, core.t0 + until)
        core.note_trades(fills)
        for fill in fills:
            for owner in {fill.maker_owner, fill.taker_owner}:
                if owner in owners:
                    self.traders[owners[owner]].record(fill)
                    n_fills[owners[owner]] += 1

        mid = core.mid()
        done = until >= core.session_length - 1e-9
        rewards = {
            a: self.traders[a].wealth(mid)
            - before[a]
            - core.inventory_penalty * self.traders[a].inventory ** 2 * dt
            for a in self.agents
        }
        self.acting = (
            list(self.agents)
            if done
            else [
                a
                for i, a in enumerate(self.possible_agents)
                if self._next[i] <= exact_until + _COINCIDENT
            ]
        )
        observations = {a: self._observe(a) for a in self.acting}
        info = {
            a: {**core.info(self.traders[a], n_fills[a]), "dt": dt, "acting": a in self.acting}
            for a in self.agents
        }
        result: MAStepResult[dict[str, Any], str] = MAStepResult(
            observations=observations,
            rewards={a: float(r) for a, r in rewards.items()},
            terminated=dict.fromkeys(self.agents, done),
            truncated=dict.fromkeys(self.agents, False),
            info=info,
        )
        if done:
            self.agents = []
        return result


__all__ = ["LimitOrderBookEnv", "LimitOrderBookMarket"]
