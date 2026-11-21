"""Historical order-book replay: message streams, book snapshots and the LOBSTER file format.

sekai never fetches or sells market data: you bring your own files, and this module
parses and replays them.

- `OrderBookMessages` is a historical message stream (submissions, cancellations,
  deletions, executions, trading halts) as NumPy arrays, one per field.
- `BookSnapshot` is the visible book at one moment (price levels and sizes per side);
  `BookSnapshots` is the visible book after every message, as in a LOBSTER orderbook file.
- `MessageReplay` replays a stream into an `OrderBook` as background order flow, with the
  same `advance(book, rng, until)` interface as `ZeroIntelligenceFlow`, so agents trade
  against real historical order flow with realistic queue positions.
- `read_lobster` loads LOBSTER message and orderbook files (CSV) with NumPy alone.

Everything here is re-exported by `sekai.finance.lob` and `sekai.finance`.
"""

from __future__ import annotations

import copy
import enum
import math
import os
import warnings
from bisect import bisect_left
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sekai.finance.lob import BACKGROUND, Fill, OrderBook, Side

__all__ = [
    "MessageType",
    "OrderBookMessages",
    "BookSnapshot",
    "BookSnapshots",
    "MessageReplay",
    "read_lobster",
]


class MessageType(enum.IntEnum):
    """Kind of a historical order-book message. The values are LOBSTER's event types."""

    SUBMIT = 1
    """A new limit order rests in the book (`size` is its quantity)."""
    CANCEL = 2
    """Part of a resting order is cancelled (`size` is the cancelled quantity)."""
    DELETE = 3
    """A resting order is removed entirely (`size` is the quantity it still had)."""
    EXECUTE = 4
    """A visible resting order trades (`size` is the executed quantity)."""
    EXECUTE_HIDDEN = 5
    """A hidden (non-displayed) resting order trades; it was never in the visible book."""
    CROSS = 6
    """An auction (cross) trade; it does not touch the continuous book."""
    HALT = 7
    """A trading halt indicator: `price` -1 halts trading, 0 resumes quoting, 1 resumes trading."""


# The halt codes LOBSTER stores in the price field of a HALT message.
_HALT_CODES = (-1, 0, 1)
_RESUME_TRADING = 1


@dataclass(frozen=True, eq=False)
class OrderBookMessages:
    """A historical order-book message stream, one NumPy array per field.

    Each message refers to one order of the historical book, by `order_id`, and `side` is
    the side of that resting order: the execution of a bid (`side` = 1) is a sale,
    initiated by a seller. The arrays are converted to the dtypes below, copied and made
    read-only. `read_lobster` builds one from a LOBSTER message file.

    Attributes:
        time: Message times in seconds (float64), finite and non-decreasing.
        event: The `MessageType` of each message (int8).
        order_id: Historical order id (int64). Submissions must have distinct ids; other
            messages may refer to ids the stream never submitted (orders that were resting
            before it began).
        size: Quantity (int64, >= 1): submitted, cancelled, deleted or executed. For
            `HALT` messages it is ignored and may be 0.
        price: Price in integer ticks (int64, >= 1); for `HALT` messages the halt code
            (-1 trading halts, 0 quoting resumes, 1 trading resumes).
        side: Side of the order the message refers to, `Side.BID` (1) or `Side.ASK` (-1)
            (int8).

    Raises:
        ValueError: If the arrays are not 1-D or differ in length, a time is not finite or
            times decrease, an event is unknown, a size, price or side is out of range, or
            two submissions share an order id. The message names the first offending
            message by its index.
    """

    time: NDArray[np.float64]
    event: NDArray[np.int8]
    order_id: NDArray[np.int64]
    size: NDArray[np.int64]
    price: NDArray[np.int64]
    side: NDArray[np.int8]

    def __post_init__(self) -> None:
        time = _float_array("time", self.time)
        event = _int_array("event", self.event)
        order_id = _int_array("order_id", self.order_id)
        size = _int_array("size", self.size)
        price = _int_array("price", self.price)
        side = _int_array("side", self.side)
        fields = {
            "time": time,
            "event": event,
            "order_id": order_id,
            "size": size,
            "price": price,
            "side": side,
        }
        for name, array in fields.items():
            if array.ndim != 1:
                raise ValueError(f"{name} must be 1-D, got shape {array.shape}.")
        lengths = {name: len(array) for name, array in fields.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"The message fields differ in length: {lengths}.")
        _first_bad(~np.isfinite(time), "time {} is not finite", time)
        _first_bad(
            np.r_[False, np.diff(time) < 0], "time {} is earlier than the message before", time
        )
        known = np.isin(event, [member.value for member in MessageType])
        _first_bad(~known, "event {} is not a MessageType (1-7)", event)
        _first_bad((side != 1) & (side != -1), "side {} is not 1 (bid) or -1 (ask)", side)
        halt = event == MessageType.HALT
        _first_bad(~halt & (size < 1), "size {} is not >= 1", size)
        _first_bad(halt & (size < 0), "size {} is negative", size)
        _first_bad(~halt & (price < 1), "price {} is not >= 1 tick", price)
        _first_bad(
            halt & ~np.isin(price, _HALT_CODES),
            "halt code {} (the price of a HALT message) is not -1, 0 or 1",
            price,
        )
        submitted = np.flatnonzero(event == MessageType.SUBMIT)
        if len(submitted) > 1:
            ids = order_id[submitted]
            order = np.argsort(ids, kind="stable")
            repeated = np.flatnonzero(ids[order][1:] == ids[order][:-1])
            if len(repeated):
                index = int(submitted[order[repeated[0] + 1]])
                raise ValueError(
                    f"message {index}: order id {int(order_id[index])} is submitted twice."
                )
        for name, array, dtype in (
            ("time", time, np.float64),
            ("event", event, np.int8),
            ("order_id", order_id, np.int64),
            ("size", size, np.int64),
            ("price", price, np.int64),
            ("side", side, np.int8),
        ):
            out = array.astype(dtype)
            out.setflags(write=False)
            object.__setattr__(self, name, out)

    def __len__(self) -> int:
        return len(self.time)

    def __repr__(self) -> str:
        if not len(self):
            return "OrderBookMessages(0 messages)"
        return (
            f"OrderBookMessages({len(self)} messages, time={self.time[0]:.9g}..{self.time[-1]:.9g})"
        )


@dataclass(frozen=True, eq=False)
class BookSnapshot:
    """The visible book at one moment: price levels and their sizes per side, best first.

    The layout of `OrderBook.depth`, so `BookSnapshot(*book.depth(10))` captures a book.
    A level of size 0 is empty; empty levels must come after the others and their price
    is stored as 0. The arrays are converted to int64, copied and made read-only.

    Attributes:
        bid_prices: Bid prices in ticks, best (highest) first, strictly decreasing.
        bid_sizes: Quantity at each bid price, >= 0.
        ask_prices: Ask prices in ticks, best (lowest) first, strictly increasing.
        ask_sizes: Quantity at each ask price, >= 0.

    Raises:
        ValueError: If the arrays are not 1-D integers, the prices and sizes of a side
            differ in length, a size is negative, an empty level precedes a non-empty one,
            a price is below 1 tick or out of order, or the best bid is not below the best
            ask.
    """

    bid_prices: NDArray[np.int64]
    bid_sizes: NDArray[np.int64]
    ask_prices: NDArray[np.int64]
    ask_sizes: NDArray[np.int64]

    def __post_init__(self) -> None:
        arrays = _check_depth(
            self.bid_prices, self.bid_sizes, self.ask_prices, self.ask_sizes, ndim=1
        )
        for name, array in zip(("bid_prices", "bid_sizes", "ask_prices", "ask_sizes"), arrays):
            object.__setattr__(self, name, array)

    def price_levels(self, side: Side) -> list[tuple[int, int]]:
        """The non-empty levels of `side` as `(price, size)` pairs, best first."""
        side = Side(side)
        prices, sizes = (
            (self.bid_prices, self.bid_sizes)
            if side is Side.BID
            else (self.ask_prices, self.ask_sizes)
        )
        filled = sizes > 0
        return list(zip(prices[filled].tolist(), sizes[filled].tolist()))

    def __repr__(self) -> str:
        bids, asks = self.price_levels(Side.BID), self.price_levels(Side.ASK)
        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None
        return (
            f"BookSnapshot(best_bid={best_bid}, best_ask={best_ask}, "
            f"bid_levels={len(bids)}, ask_levels={len(asks)})"
        )


@dataclass(frozen=True, eq=False)
class BookSnapshots:
    """The visible book after each message of a stream: one `(n, levels)` array per field.

    Row `i` is the book after message `i`, as in a LOBSTER orderbook file, with the layout
    of `BookSnapshot` (best level first, empty levels of size 0 and price 0 last).
    `snapshots[i]` is row `i` as a `BookSnapshot`. A `MessageReplay` given snapshots
    starts at any time in O(levels), from the row before that time, instead of replaying
    every earlier message.

    Attributes:
        bid_prices: Bid prices in ticks, `(n, levels)`.
        bid_sizes: Bid sizes, `(n, levels)`.
        ask_prices: Ask prices in ticks, `(n, levels)`.
        ask_sizes: Ask sizes, `(n, levels)`.

    Raises:
        ValueError: If the arrays are not 2-D integers of matching shapes, or a row breaks
            the rules of a `BookSnapshot` (the message names the first such row).
    """

    bid_prices: NDArray[np.int64]
    bid_sizes: NDArray[np.int64]
    ask_prices: NDArray[np.int64]
    ask_sizes: NDArray[np.int64]

    def __post_init__(self) -> None:
        arrays = _check_depth(
            self.bid_prices, self.bid_sizes, self.ask_prices, self.ask_sizes, ndim=2
        )
        for name, array in zip(("bid_prices", "bid_sizes", "ask_prices", "ask_sizes"), arrays):
            object.__setattr__(self, name, array)

    @property
    def levels(self) -> int:
        """Price levels per side in each row."""
        return int(self.bid_prices.shape[1])

    def __len__(self) -> int:
        return len(self.bid_prices)

    def __getitem__(self, index: int) -> BookSnapshot:
        """Row `index` (negative indices count from the end) as a `BookSnapshot`."""
        return BookSnapshot(
            self.bid_prices[index],
            self.bid_sizes[index],
            self.ask_prices[index],
            self.ask_sizes[index],
        )

    def __repr__(self) -> str:
        return f"BookSnapshots({len(self)} rows, levels={self.levels})"


class MessageReplay:
    """Background order flow that replays a historical message stream into an `OrderBook`.

    It has the interface of `ZeroIntelligenceFlow` (`advance(book, rng, until)` returns
    the fills), so an environment's agents trade against real order flow; the replay
    itself is deterministic and never draws from `rng`. `initialise(book, start=t)` sets
    the book to its historical state at time `t`: after every message stamped at or
    before `t`.

    **Replay model.** Historical orders enter the book as resting limit orders of owner
    `BACKGROUND` and keep their price-time priority, so an agent's order competes with
    them for queue position. Every message is applied to the *current* book, which the
    agent's trades may have changed:

    - A **submission** rests a new background order. If an agent's order rests at a
      crossing price, the submission trades with it first, as it would have.
    - A **partial cancel** reduces the order by the cancelled quantity and a **delete**
      cancels what is left of it, if it is still resting; the agent may already have
      filled it, and then nothing happens.
    - A **visible execution** is replayed as a marketable order of the executed size,
      limited at the execution price, sent against the executed order's side. It fills
      whatever has priority in the current book, so an agent order at a better price, or
      ahead in the queue at that price, is filled first. Any unfilled remainder is
      dropped: it never trades through the historical price.
    - A **hidden execution** traded with an order the visible book never held. It is
      replayed the same way, so it can fill only orders at a price at least as good as
      the execution price (visible orders have priority over hidden ones at a price): in
      a book that follows history there are none, but an agent quote there is filled.
    - A **halt** sets `halted` until trading resumes; the environments do not send agent
      orders while it is set. Cross (auction) trades are ignored.

    Known biases of this model:

    - **The flow is exogenous.** Historical messages do not react to the agent: nobody
      cancels, re-prices or trades differently because of its orders or trades, so the
      agent's market impact on later order flow is not modelled.
    - **Consumed liquidity is gone.** An execution of an order the agent's trades already
      consumed (or moved down the queue) walks further into the queue at that price,
      filling orders behind it that history did not fill, or goes unfilled; a historical
      cancel of a consumed order does nothing.
    - **Displaced quantity stays.** When an agent order ahead in the queue absorbs part of
      a historical execution, the historical order keeps the quantity it would have
      traded, until its own later messages remove it. That is the counterfactual book
      the fixed historical order sizes imply; pass `cap_to_history=True` to cut every
      background order to its historical remaining size after each message about it, so
      the background book never holds more than history did.
    - **Unknown orders.** Orders resting before the replay starts are known only through
      the snapshot that seeds the book, as one background order per price level (ahead
      of every later order there). Their later messages reduce that level's order; orders
      deeper than the snapshot's levels are unknown, so the replayed book can lack depth
      far from the touch until those levels refill. An execution of such an unknown
      order can fill only orders at strictly better prices.
    - **Halts.** The book cannot be crossed: orders submitted during a halt that cross it
      trade at once, instead of waiting for the reopening cross.

    Without agents, the replay reproduces the historical book exactly (given a snapshot
    with every level the messages touch).

    A replay keeps the position of the book it drives, so use one per book: `spawn()`
    makes another (the environments take their own).

    Args:
        messages: The historical messages, at least one.
        snapshot: The book before the first message (default: an empty book). It seeds a
            replay that starts at or before the first message, or that has no
            `snapshots`.
        snapshots: The visible book after each message (a LOBSTER orderbook file, from
            `read_lobster`). With them, a replay starting at time `t` is seeded from the
            row of the last message at or before `t`; without them, it replays every
            earlier message from `snapshot` (exact, but O(messages before `t`)).
        cap_to_history: Cut each background order to at most its historical remaining
            size after every message about it (see the biases above).

    Raises:
        TypeError: If `messages`, `snapshot` or `snapshots` is of the wrong type.
        ValueError: If there are no messages, or `snapshots` does not have one row per
            message.

    Example:
        ```python
        messages, snapshots = read_lobster("AAPL_message_10.csv", "AAPL_orderbook_10.csv")
        replay = MessageReplay(messages, snapshots=snapshots)
        book = OrderBook()
        replay.initialise(book, start=36_000.0)  # the historical book at 10:00
        fills = replay.advance(book, None, 36_060.0)  # one minute of order flow
        ```
    """

    def __init__(
        self,
        messages: OrderBookMessages,
        *,
        snapshot: BookSnapshot | None = None,
        snapshots: BookSnapshots | None = None,
        cap_to_history: bool = False,
    ) -> None:
        if not isinstance(messages, OrderBookMessages):
            raise TypeError(
                f"messages must be an OrderBookMessages, got {type(messages).__name__}."
            )
        if snapshot is not None and not isinstance(snapshot, BookSnapshot):
            raise TypeError(f"snapshot must be a BookSnapshot, got {type(snapshot).__name__}.")
        if snapshots is not None and not isinstance(snapshots, BookSnapshots):
            raise TypeError(f"snapshots must be a BookSnapshots, got {type(snapshots).__name__}.")
        if not len(messages):
            raise ValueError("messages is empty; a replay needs at least one message.")
        if snapshots is not None and len(snapshots) != len(messages):
            raise ValueError(
                f"snapshots has {len(snapshots)} rows but there are {len(messages)} messages; "
                "it needs the book after each message."
            )
        self.messages = messages
        self.snapshot = snapshot
        self.snapshots = snapshots
        self.cap_to_history = bool(cap_to_history)
        event = messages.event
        self._halts: list[int] = np.flatnonzero(event == MessageType.HALT).tolist()
        self._executions: list[int] = np.flatnonzero(
            (event == MessageType.EXECUTE) | (event == MessageType.EXECUTE_HIDDEN)
        ).tolist()
        self._reset_state()

    def _reset_state(self) -> None:
        self._book: OrderBook | None = None
        self._position = 0
        self._halted = False
        # Historical order id -> [book order id (0 once it has none), historical remaining].
        self._orders: dict[int, list[int]] = {}
        # (side, price) of a seeded level -> [book order id, remaining of the unknown orders].
        self._seeded: dict[tuple[int, int], list[int]] = {}

    # -- state -------------------------------------------------------------------------

    @property
    def start_time(self) -> float:
        """Time of the first message."""
        return float(self.messages.time[0])

    @property
    def end_time(self) -> float:
        """Time of the last message."""
        return float(self.messages.time[-1])

    @property
    def position(self) -> int:
        """Index of the next message to replay (`len(messages)` once all are replayed)."""
        return self._position

    @property
    def done(self) -> bool:
        """Whether every message has been replayed."""
        return self._position >= len(self.messages)

    @property
    def halted(self) -> bool:
        """Whether trading is halted (from a `HALT` message until trading resumes)."""
        return self._halted

    def spawn(self) -> MessageReplay:
        """A fresh replay of the same messages and snapshots (shared, not copied), unbound."""
        twin = copy.copy(self)
        twin._reset_state()
        return twin

    # -- the flow interface ------------------------------------------------------------

    def initialise(
        self,
        book: OrderBook,
        rng: np.random.Generator | None = None,
        *,
        start: float | None = None,
    ) -> None:
        """Clear `book` and set it to its historical state; bind this replay to it.

        The book is seeded with one `BACKGROUND` order per price level of a snapshot:
        `snapshot`, for a start before the first message (or without `snapshots`), else
        the row of `snapshots` of the last message at or before `start`. Without
        `snapshots`, the messages before `start` are then replayed onto it. `book.time`
        is set to `start` (to the first message's time if `start` is `None`), and
        `book.last_price` to the last execution price before it (to the seeded book's mid,
        rounded down, if there was none).

        Args:
            book: The book to (re)initialise.
            rng: Unused (the replay is deterministic); accepted for the flow interface.
            start: Time in seconds: the book takes its state after every message stamped
                at or before it. `None` starts before the first message.

        Raises:
            ValueError: If `start` is not a finite number.
        """
        del rng  # deterministic
        times = self.messages.time
        if start is None:
            index, now = 0, float(times[0])
        else:
            now = _finite("start", start)
            index = int(np.searchsorted(times, now, side="right"))
        book.clear()
        self._reset_state()
        self._book = book
        if index > 0 and self.snapshots is not None:
            book.time = now
            self._seed(book, self.snapshots[index - 1])
        else:  # the book before the first message, then the history before `start`
            book.time = min(now, float(times[0]))
            if self.snapshot is not None:
                self._seed(book, self.snapshot)
            self._run(book, 0, index, [])
            book.time = now
        self._position = index
        halts = bisect_left(self._halts, index)
        self._halted = (
            halts > 0 and int(self.messages.price[self._halts[halts - 1]]) != _RESUME_TRADING
        )
        executions = bisect_left(self._executions, index)
        if executions > 0:
            book.last_price = int(self.messages.price[self._executions[executions - 1]])
        else:
            book.last_price = self._reference_price(book, index)

    def advance(self, book: OrderBook, rng: np.random.Generator | None, until: float) -> list[Fill]:
        """Replay every message stamped after the current position and at or before `until`.

        `book.time` is set to each message's time as it is applied and to `until` at the
        end. Past the last message the book is left as it is.

        Args:
            book: The book this replay was initialised with; it may also hold other
                owners' (agents') orders.
            rng: Unused (the replay is deterministic); accepted for the flow interface.
            until: End time, finite and >= `book.time`.

        Returns:
            Every fill that occurred, in order, including fills against agents' orders.

        Raises:
            ValueError: If `until` is not finite or is earlier than `book.time`, or if this
                replay was not initialised with `book`.
        """
        del rng  # deterministic
        self._check_book(book)
        until = _finite("until", until)
        if until < book.time:
            raise ValueError(f"until ({until}) is earlier than the book time ({book.time}).")
        end = int(np.searchsorted(self.messages.time, until, side="right"))
        fills: list[Fill] = []
        if end > self._position:
            self._run(book, self._position, end, fills)
        book.time = until
        return fills

    def step(self, book: OrderBook) -> list[Fill]:
        """Replay the next message, whatever its time, and return its fills.

        `book.time` advances to the message's time (it never goes back).

        Raises:
            ValueError: If every message has been replayed, or if this replay was not
                initialised with `book`.
        """
        self._check_book(book)
        if self.done:
            raise ValueError("every message has been replayed.")
        fills: list[Fill] = []
        self._run(book, self._position, self._position + 1, fills)
        return fills

    def __repr__(self) -> str:
        return (
            f"MessageReplay({len(self.messages)} messages, "
            f"time={self.start_time:.9g}..{self.end_time:.9g}, position={self._position})"
        )

    # -- internals ---------------------------------------------------------------------

    def _check_book(self, book: OrderBook) -> None:
        if self._book is None:
            raise ValueError("call initialise(book) before replaying messages into it.")
        if book is not self._book:
            raise ValueError(
                "this replay drives another book; use one MessageReplay per book "
                "(spawn() makes another) and initialise() it with that book."
            )

    def _seed(self, book: OrderBook, snapshot: BookSnapshot) -> None:
        """Rest one background order per non-empty level of `snapshot`."""
        for side in (Side.BID, Side.ASK):
            for price, size in snapshot.price_levels(side):
                order_id, _ = book.limit(side, price, size, BACKGROUND)
                self._seeded[int(side), price] = [order_id or 0, size]

    def _reference_price(self, book: OrderBook, index: int) -> int | None:
        """A reference price for `book.last_price` when no execution precedes `index`."""
        bid, ask = book.best_bid(), book.best_ask()
        if bid is not None and ask is not None:
            return (bid + ask) // 2
        if bid is not None or ask is not None:
            return bid if bid is not None else ask
        event = self.messages.event[index:]
        later = np.flatnonzero(event == MessageType.SUBMIT)
        return int(self.messages.price[index + later[0]]) if len(later) else None

    def _run(self, book: OrderBook, lo: int, hi: int, fills: list[Fill]) -> None:
        """Apply messages `lo..hi-1`, stamping the book with their times."""
        m = self.messages
        rows = zip(
            m.time[lo:hi].tolist(),
            m.event[lo:hi].tolist(),
            m.order_id[lo:hi].tolist(),
            m.size[lo:hi].tolist(),
            m.price[lo:hi].tolist(),
            m.side[lo:hi].tolist(),
        )
        for time, event, order_id, size, price, side in rows:
            if time > book.time:
                book.time = time
            self._apply(book, event, order_id, size, price, side, fills)
        self._position = hi

    def _apply(
        self,
        book: OrderBook,
        event: int,
        order_id: int,
        size: int,
        price: int,
        side: int,
        fills: list[Fill],
    ) -> None:
        """Apply one message to the current book (the model in the class docstring)."""
        if event == MessageType.SUBMIT:
            new_id, new_fills = book.limit(Side(side), price, size, BACKGROUND)
            fills.extend(new_fills)
            self._orders[order_id] = [new_id or 0, size]
            return
        if event == MessageType.EXECUTE_HIDDEN:
            self._execute(book, side, price, size, fills)
            return
        if event == MessageType.HALT:
            self._halted = price != _RESUME_TRADING
            return
        if event == MessageType.CROSS:
            return
        # CANCEL, DELETE or EXECUTE of a resting order. Finished orders stay in _orders, so
        # a stray later message about one is not taken for a seeded level's.
        entry = self._orders.get(order_id)
        known = entry is not None
        if entry is None:  # resting before the replay began: part of a seeded level
            entry = self._seeded.get((side, price))
        if event == MessageType.EXECUTE:
            if entry is not None:
                self._execute(book, side, price, size, fills)
            else:  # an order the book never held, ahead of all it holds at that price
                better = price + 1 if side == Side.BID else price - 1
                self._execute(book, side, better, size, fills)
        elif entry is not None and entry[0]:
            order = book.get(entry[0])
            if order is not None:
                if event == MessageType.DELETE and known:
                    book.cancel(entry[0])
                elif order.quantity > size:
                    book.reduce(entry[0], order.quantity - size)
                else:
                    book.cancel(entry[0])
        if entry is None:
            return
        entry[1] = 0 if event == MessageType.DELETE and known else max(entry[1] - size, 0)
        if self.cap_to_history and entry[0]:
            order = book.get(entry[0])
            if order is not None and order.quantity > entry[1]:
                book.reduce(entry[0], entry[1])

    @staticmethod
    def _execute(book: OrderBook, side: int, limit: int, size: int, fills: list[Fill]) -> None:
        """Trade `size` against `side`'s orders priced at `limit` or better; drop the rest."""
        if side == Side.BID:
            best = book.best_bid()
            if best is None or best < limit:
                return
            order_id, new_fills = book.limit(Side.ASK, limit, size, BACKGROUND)
        else:
            best = book.best_ask()
            if best is None or best > limit:
                return
            order_id, new_fills = book.limit(Side.BID, limit, size, BACKGROUND)
        fills.extend(new_fills)
        if order_id is not None:  # immediate or cancel
            book.cancel(order_id)


# -- LOBSTER ---------------------------------------------------------------------------

# LOBSTER stores prices as dollars times 10,000. (Its empty orderbook levels hold dummy
# prices, +-9999999999, with size 0: any level of size 0 is read as empty.)
_LOBSTER_SCALE = 10_000


def read_lobster(
    message_path: str | os.PathLike[str],
    orderbook_path: str | os.PathLike[str] | None = None,
    *,
    tick_size: float = 0.01,
    levels: int | None = None,
) -> tuple[OrderBookMessages, BookSnapshots | None]:
    """Read a LOBSTER message file, and optionally its orderbook file, that you own.

    LOBSTER (lobsterdata.com) reconstructs NASDAQ order books from exchange messages;
    sekai only parses the files, with NumPy (no pandas needed). A message file has one
    row per message and six columns: time (seconds after midnight), event type (1
    submission, 2 partial cancellation, 3 deletion, 4 visible execution, 5 hidden
    execution, 6 cross trade, 7 trading halt: see `MessageType`), order id, size, price
    (dollars times 10,000) and direction (1 buy order, -1 sell order; for an execution,
    the side of the resting order). The orderbook file has the visible book after each
    message: ask price, ask size, bid price and bid size for each level, best first;
    empty levels hold the dummy prices 9999999999 (ask) and -9999999999 (bid) with
    size 0, and are read as price 0, size 0.

    Prices become integer ticks of `tick_size` dollars. Visible orders must lie on that
    grid. Hidden executions may not (e.g. at the midpoint), so their prices are rounded
    away from the book's interior: up on the bid side, down on the ask side, which lets a
    replayed hidden execution fill only orders at least as good as its true price. Cross
    trades are rounded to the nearest tick; halt messages keep LOBSTER's code (-1, 0, 1)
    as their price.

    Args:
        message_path: Path of the message file (CSV).
        orderbook_path: Path of the matching orderbook file (CSV), or `None`.
        tick_size: Price tick in dollars; a multiple of 0.0001 (LOBSTER's resolution).
        levels: Read only the best `levels` levels of the orderbook file (default: all).

    Returns:
        `(messages, snapshots)`: the `OrderBookMessages`, and the `BookSnapshots` of the
        orderbook file (`None` without one).

    Raises:
        ValueError: If a file is empty or has the wrong number of columns, the two files
            differ in length, `tick_size` or `levels` is invalid, or a value is invalid
            (the message names the line).

    Example:
        ```python
        messages, snapshots = read_lobster(
            "AAPL_2012-06-21_34200000_57600000_message_10.csv",
            "AAPL_2012-06-21_34200000_57600000_orderbook_10.csv",
            tick_size=0.01,
        )
        replay = MessageReplay(messages, snapshots=snapshots)
        ```
    """
    tick_units = _tick_units(tick_size)
    if levels is not None:
        levels = _as_count("levels", levels)
    rows = _load_csv(message_path)
    if rows.shape[1] != 6:
        raise ValueError(
            f"{os.fspath(message_path)}: a LOBSTER message file has 6 columns (time, type, "
            f"order id, size, price, direction), got {rows.shape[1]}."
        )
    time = rows[:, 0]
    event = _integer_column(rows[:, 1], "event type", message_path)
    order_id = _integer_column(rows[:, 2], "order id", message_path)
    size = _integer_column(rows[:, 3], "size", message_path)
    raw_price = _integer_column(rows[:, 4], "price", message_path)
    side = _integer_column(rows[:, 5], "direction", message_path)
    unknown = ~np.isin(event, [member.value for member in MessageType])
    if unknown.any():
        line = int(np.argmax(unknown))
        raise ValueError(
            f"{os.fspath(message_path)}, line {line + 1}: event type {event[line]} is not 1-7."
        )
    sideless = (side != 1) & (side != -1)
    if sideless.any():
        line = int(np.argmax(sideless))
        raise ValueError(
            f"{os.fspath(message_path)}, line {line + 1}: direction {side[line]} is not 1 or -1."
        )
    off_grid = np.isin(event, (1, 2, 3, 4)) & (raw_price % tick_units != 0)
    if off_grid.any():
        line = int(np.argmax(off_grid))
        raise ValueError(
            f"{os.fspath(message_path)}, line {line + 1}: price "
            f"{raw_price[line] / _LOBSTER_SCALE} is not a multiple of tick_size={tick_size}."
        )
    price = np.where(
        event == MessageType.HALT,
        raw_price,
        np.where(
            event == MessageType.EXECUTE_HIDDEN,
            np.where(side == 1, -(-raw_price // tick_units), raw_price // tick_units),
            (raw_price + tick_units // 2) // tick_units,
        ),
    )
    try:
        messages = OrderBookMessages(
            time, event.astype(np.int8), order_id, size, price, side.astype(np.int8)
        )
    except ValueError as error:
        raise ValueError(f"{os.fspath(message_path)}: {error}") from error
    if orderbook_path is None:
        return messages, None
    book = _load_csv(orderbook_path)
    width = book.shape[1]
    if width % 4:
        raise ValueError(
            f"{os.fspath(orderbook_path)}: a LOBSTER orderbook file has 4 columns per level "
            f"(ask price, ask size, bid price, bid size), got {width} columns."
        )
    if levels is not None:
        if levels > width // 4:
            raise ValueError(
                f"{os.fspath(orderbook_path)} has {width // 4} levels; cannot read {levels}."
            )
        book = book[:, : 4 * levels]
    if len(book) != len(messages):
        raise ValueError(
            f"{os.fspath(orderbook_path)} has {len(book)} rows but "
            f"{os.fspath(message_path)} has {len(messages)} messages; they must match."
        )
    columns = {
        name: _integer_column(book[:, k::4], name, orderbook_path)
        for k, name in enumerate(("ask price", "ask size", "bid price", "bid size"))
    }
    depth = []
    for side_name in ("bid", "ask"):
        raw, sizes = columns[f"{side_name} price"], columns[f"{side_name} size"]
        filled = sizes > 0
        bad = filled & (raw % tick_units != 0)
        if bad.any():
            row, level = np.argwhere(bad)[0]
            raise ValueError(
                f"{os.fspath(orderbook_path)}, line {row + 1}: {side_name} price "
                f"{raw[row, level] / _LOBSTER_SCALE} at level {level + 1} is not a multiple "
                f"of tick_size={tick_size}."
            )
        depth += [np.where(filled, raw // tick_units, 0), sizes]
    try:
        snapshots = BookSnapshots(depth[0], depth[1], depth[2], depth[3])
    except ValueError as error:
        raise ValueError(f"{os.fspath(orderbook_path)}: {error}") from error
    return messages, snapshots


def _load_csv(path: str | os.PathLike[str]) -> NDArray[np.float64]:
    """A numeric CSV file as a 2-D float64 array (the LOBSTER integers are exact in float64)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # "input contained no data"
        rows = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    if rows.size == 0:
        raise ValueError(f"{os.fspath(path)} holds no rows.")
    return rows


def _integer_column(
    values: NDArray[np.float64], name: str, path: str | os.PathLike[str]
) -> NDArray[np.int64]:
    """A column (or `(rows, k)` block) of whole numbers as int64; else name its line."""
    bad = ~np.isfinite(values) | (values != np.round(values))
    if bad.any():
        row = int(np.argwhere(bad)[0][0])
        raise ValueError(
            f"{os.fspath(path)}, line {row + 1}: {name} {values[bad].flat[0]!r} is not an integer."
        )
    return values.astype(np.int64)


def _tick_units(tick_size: Any) -> int:
    """`tick_size` dollars in LOBSTER price units (1/10,000 dollar), as an int >= 1."""
    tick = _finite("tick_size", tick_size)
    units = round(tick * _LOBSTER_SCALE)
    if tick <= 0 or units < 1 or abs(tick * _LOBSTER_SCALE - units) > 1e-6:
        raise ValueError(
            f"tick_size must be a positive multiple of 0.0001 dollars, got {tick_size!r}."
        )
    return int(units)


# -- validation helpers ----------------------------------------------------------------


def _finite(name: str, value: Any) -> float:
    if isinstance(value, bool | np.bool_):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")
    try:
        out = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite number, got {value!r}.") from None
    if not math.isfinite(out):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")
    return out


def _as_count(name: str, value: Any) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, int | np.integer):
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")
    return int(value)


def _float_array(name: str, values: Any) -> NDArray[np.float64]:
    array = np.asarray(values)
    if array.dtype == np.bool_ or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must hold numbers, got dtype {array.dtype}.")
    return array.astype(np.float64)


def _int_array(name: str, values: Any) -> NDArray[np.int64]:
    """`values` as int64; floats must be whole, bools and out-of-range values are refused."""
    array = np.asarray(values)
    if array.dtype == np.bool_ or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must hold integers, got dtype {array.dtype}.")
    if array.dtype.kind == "f":
        whole = np.isfinite(array) & (array == np.round(array)) & (np.abs(array) < 2.0**63)
        if not whole.all():
            raise ValueError(f"{name} must hold integers, got {array[~whole].flat[0]!r}.")
    elif array.dtype.kind == "u" and array.size and array.max() > np.iinfo(np.int64).max:
        raise ValueError(f"{name} holds a value too large for int64.")
    return array.astype(np.int64)


def _first_bad(bad: NDArray[np.bool_], message: str, values: NDArray[Any]) -> None:
    """Raise for the first message flagged in `bad`, formatting its value into `message`."""
    if bad.any():
        index = int(np.argmax(bad))
        raise ValueError(f"message {index}: {message.format(values[index])}.")


def _check_depth(
    bid_prices: Any, bid_sizes: Any, ask_prices: Any, ask_sizes: Any, *, ndim: int
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Validated, read-only int64 copies of a snapshot's (ndim 1) or snapshots' (2) arrays."""
    arrays = [
        _int_array(name, values)
        for name, values in (
            ("bid_prices", bid_prices),
            ("bid_sizes", bid_sizes),
            ("ask_prices", ask_prices),
            ("ask_sizes", ask_sizes),
        )
    ]
    shape = "(levels,)" if ndim == 1 else "(n, levels)"
    for name, array in zip(("bid_prices", "bid_sizes", "ask_prices", "ask_sizes"), arrays):
        if array.ndim != ndim:
            raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    if arrays[0].shape != arrays[1].shape or arrays[2].shape != arrays[3].shape:
        raise ValueError("The prices and sizes of a side must have the same shape.")
    if ndim == 2 and arrays[0].shape[0] != arrays[2].shape[0]:
        raise ValueError("bid and ask arrays must have the same number of rows.")

    def where(bad: NDArray[np.bool_], offset: int = 1) -> str:
        """Where the first flagged level is (levels count from 1, rows from 0)."""
        if ndim == 1:
            return f"level {int(np.argmax(bad)) + offset}"
        row, level = np.argwhere(bad)[0]
        return f"row {row}, level {level + offset}"

    out: list[NDArray[np.int64]] = []
    tops: list[tuple[NDArray[np.int64], NDArray[np.bool_]]] = []
    for side_name, prices, sizes, sign in (
        ("bid", arrays[0], arrays[1], 1),
        ("ask", arrays[2], arrays[3], -1),
    ):
        if (sizes < 0).any():
            raise ValueError(f"{side_name} sizes must be >= 0 ({where(sizes < 0)}).")
        filled = sizes > 0
        gap = ~filled[..., :-1] & filled[..., 1:]
        if gap.any():
            raise ValueError(
                f"an empty {side_name} level comes before a non-empty one ({where(gap)}); "
                "empty levels must come last."
            )
        low = filled & (prices < 1)
        if low.any():
            raise ValueError(f"{side_name} prices must be >= 1 tick ({where(low)}).")
        # Empty levels come last, so a filled level k + 1 has a filled level k before it.
        unordered = filled[..., 1:] & (sign * (prices[..., :-1] - prices[..., 1:]) <= 0)
        if unordered.any():
            direction = "decreasing" if side_name == "bid" else "increasing"
            raise ValueError(
                f"{side_name} prices must be strictly {direction} ({where(unordered, 2)})."
            )
        clean = np.where(filled, prices, 0)
        out += [clean, sizes]
        # The best level: its price and whether it is filled (none without levels).
        if prices.shape[-1]:
            tops.append((clean[..., 0], filled[..., 0]))
        else:
            empty = prices.shape[:-1]
            tops.append((np.zeros(empty, np.int64), np.zeros(empty, np.bool_)))
    (bid, has_bid), (ask, has_ask) = tops
    crossed = np.atleast_1d(has_bid & has_ask & (bid >= ask))
    if crossed.any():
        place = "" if ndim == 1 else f" (row {int(np.argmax(crossed))})"
        raise ValueError(f"the best bid must be below the best ask{place}.")
    for array in out:
        array.setflags(write=False)
    return out[0], out[1], out[2], out[3]
