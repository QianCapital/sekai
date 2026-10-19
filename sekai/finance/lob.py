"""Limit order book: a price-time priority matching engine and stochastic background order flow.

`OrderBook` is a single-instrument continuous double auction in integer price ticks. An
incoming order matches against the opposite side while the prices cross, best price first
and first-in-first-out within a price level, always executing at the resting (maker)
order's price; any unfilled limit quantity then rests in the book. Price levels are kept in
sorted lists (`bisect`) with a FIFO queue per level, and cancellation is O(1) through lazy
deletion: a cancelled order is zeroed in place and purged when it reaches the front of its
queue, while per-level volumes are always exact.

`ZeroIntelligenceFlow` drives a book with Poisson limit, market and cancel orders. It
provides the background liquidity that event-driven market environments, single- or
multi-agent, trade against.
"""

from __future__ import annotations

import enum
import itertools
import math
import operator
from bisect import bisect_left, insort
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["Side", "BACKGROUND", "Order", "Fill", "OrderBook", "ZeroIntelligenceFlow"]


class Side(enum.IntEnum):
    """Side of an order. The value is the sign of the owner's position change on a fill."""

    BID = 1
    ASK = -1

    @property
    def opposite(self) -> Side:
        """The other side of the book."""
        return Side.ASK if self is Side.BID else Side.BID


BACKGROUND: int = -1
"""Owner id of orders submitted by background flow (e.g. `ZeroIntelligenceFlow`)."""


@dataclass(slots=True)
class Order:
    """A resting limit order.

    The objects returned by `OrderBook.get` and `OrderBook.orders_of` are the book's live
    internal records: read them, but never mutate them (use `OrderBook.cancel` and
    `OrderBook.reduce`). Once an order leaves the book, by a complete fill, a cancellation
    or `OrderBook.clear`, its `quantity` is 0.

    Attributes:
        order_id: Unique identifier assigned by the book, increasing in arrival order.
        side: `Side.BID` (buy) or `Side.ASK` (sell).
        price: Limit price in integer ticks, >= 1.
        quantity: Remaining quantity; > 0 while the order rests.
        owner: Owner id: `BACKGROUND` for background flow, an agent id otherwise.
        time: Book time at which the order was accepted.
    """

    order_id: int
    side: Side
    price: int
    quantity: int
    owner: int
    time: float


@dataclass(frozen=True, slots=True)
class Fill:
    """One execution between a resting (maker) order and an incoming (taker) order.

    Attributes:
        price: Execution price in ticks: the maker order's limit price.
        quantity: Executed quantity, >= 1.
        maker_order_id: Id of the resting order.
        taker_order_id: Id of the incoming order.
        maker_owner: Owner of the resting order.
        taker_owner: Owner of the incoming order.
        taker_side: Side of the incoming order (`Side.BID` for a buy that lifted an ask).
        time: Book time of the execution.
    """

    price: int
    quantity: int
    maker_order_id: int
    taker_order_id: int
    maker_owner: int
    taker_owner: int
    taker_side: Side
    time: float


# Rebuild a level's queue once it holds more than this many cancelled orders and they
# outnumber the live ones, so lazily deleted orders cannot accumulate without bound.
_COMPACT_MIN_DEAD = 16


class _Level:
    """The orders resting at one price: a FIFO queue plus exact aggregate volume."""

    __slots__ = ("dead", "orders", "volume")

    def __init__(self) -> None:
        self.orders: deque[Order] = deque()
        self.volume = 0
        self.dead = 0


class OrderBook:
    """A price-time priority limit order book for one instrument, in integer price ticks.

    Limit orders match against the opposite side while the prices cross and rest any
    remainder; market orders walk the opposite side and drop any unfilled remainder.
    Executions always happen at the resting order's price. Self-trades are allowed. Every
    accepted order, including one that fills completely on arrival, is assigned a new
    order id, increasing in arrival order (ids restart at 1 after `clear`).

    Complexity: inserting or removing a price level is O(log L) search on a sorted list
    of L prices; cancelling or reducing an order is O(1); matching is O(1) per fill.

    Attributes:
        time: Current book time. A plain attribute: the caller (or the background flow)
            advances it, and it stamps accepted orders and fills.
        last_price: Price of the most recent fill, or a reference price set by the
            caller (`ZeroIntelligenceFlow.initialise` sets it to the initial mid);
            `None` before either.
    """

    def __init__(self) -> None:
        self.time: float = 0.0
        self.last_price: int | None = None
        self._next_id = 1
        # Both price lists are ascending: the best bid is last, the best ask first.
        self._bid_prices: list[int] = []
        self._ask_prices: list[int] = []
        self._bid_levels: dict[int, _Level] = {}
        self._ask_levels: dict[int, _Level] = {}
        self._orders: dict[int, Order] = {}
        self._by_owner: dict[int, dict[int, Order]] = {}

    # -- order entry -------------------------------------------------------------------

    def limit(
        self, side: Side, price: int, quantity: int, owner: int = BACKGROUND
    ) -> tuple[int | None, list[Fill]]:
        """Submit a limit order.

        Args:
            side: `Side.BID` to buy or `Side.ASK` to sell (or the integers 1 / -1).
            price: Limit price in ticks, an integer >= 1.
            quantity: Quantity, an integer >= 1.
            owner: Owner id; defaults to `BACKGROUND`.

        Returns:
            `(order_id, fills)`: the id of the resting remainder, or `None` if the order
            filled completely on arrival (its id still appears in `fills` as
            `taker_order_id`), and the executions in matching order.

        Raises:
            ValueError: If `side`, `price`, `quantity` or `owner` is invalid.
        """
        if side is not Side.BID and side is not Side.ASK:
            side = _as_side(side)
        price = _as_int(price, "price", 1)
        quantity = _as_int(quantity, "quantity", 1)
        if type(owner) is not int:
            owner = _as_owner(owner)
        order_id = self._next_id
        self._next_id = order_id + 1
        fills: list[Fill] = []
        remaining = self._match(side, price, quantity, order_id, owner, fills)
        if remaining == 0:
            return None, fills
        self._rest(Order(order_id, side, price, remaining, owner, self.time))
        return order_id, fills

    def market(self, side: Side, quantity: int, owner: int = BACKGROUND) -> tuple[int, list[Fill]]:
        """Submit a market order: walk the opposite side and drop any unfilled remainder.

        Args:
            side: `Side.BID` to buy or `Side.ASK` to sell (or the integers 1 / -1).
            quantity: Quantity, an integer >= 1.
            owner: Owner id; defaults to `BACKGROUND`.

        Returns:
            `(order_id, fills)`: the taker order's id and the executions in matching order
            (empty if the opposite side is empty).

        Raises:
            ValueError: If `side`, `quantity` or `owner` is invalid.
        """
        if side is not Side.BID and side is not Side.ASK:
            side = _as_side(side)
        quantity = _as_int(quantity, "quantity", 1)
        if type(owner) is not int:
            owner = _as_owner(owner)
        order_id = self._next_id
        self._next_id = order_id + 1
        fills: list[Fill] = []
        self._match(side, math.inf if side is Side.BID else 0, quantity, order_id, owner, fills)
        return order_id, fills

    def cancel(self, order_id: int) -> bool:
        """Cancel a resting order.

        Args:
            order_id: Id returned by `limit`.

        Returns:
            True if the order was resting and is now cancelled; False if the id is unknown
            or the order has already left the book.
        """
        order = self._orders.pop(order_id, None)
        if order is None:
            return False
        del self._by_owner[order.owner][order_id]
        self._unlink(order)
        return True

    def reduce(self, order_id: int, quantity: int) -> bool:
        """Reduce a resting order's remaining quantity, keeping its time priority.

        Args:
            order_id: Id returned by `limit`.
            quantity: New remaining quantity, an integer smaller than the current one;
                0 cancels the order.

        Returns:
            True if the order was resting and has been reduced (or cancelled); False if
            the id is unknown or the order has already left the book.

        Raises:
            ValueError: If `quantity` is not an integer >= 0, or is not smaller than the
                order's current remaining quantity.
        """
        quantity = _as_int(quantity, "quantity", 0)
        order = self._orders.get(order_id)
        if order is None:
            return False
        if quantity == 0:
            return self.cancel(order_id)
        if quantity >= order.quantity:
            raise ValueError(
                f"reduce() needs a quantity below the current {order.quantity}, got {quantity}."
            )
        level = (self._bid_levels if order.side is Side.BID else self._ask_levels)[order.price]
        level.volume -= order.quantity - quantity
        order.quantity = quantity
        return True

    def clear(self) -> None:
        """Remove every resting order and return the book to its freshly constructed state.

        Resets `time` to 0, `last_price` to `None` and the order-id counter to 1.
        """
        for order in self._orders.values():
            order.quantity = 0
        self.time = 0.0
        self.last_price = None
        self._next_id = 1
        self._bid_prices = []
        self._ask_prices = []
        self._bid_levels = {}
        self._ask_levels = {}
        self._orders = {}
        self._by_owner = {}

    # -- queries -----------------------------------------------------------------------

    def get(self, order_id: int) -> Order | None:
        """The resting order with this id (a live record: do not mutate it), or `None`."""
        return self._orders.get(order_id)

    def best_bid(self) -> int | None:
        """Highest resting bid price in ticks, or `None` if the bid side is empty."""
        return self._bid_prices[-1] if self._bid_prices else None

    def best_ask(self) -> int | None:
        """Lowest resting ask price in ticks, or `None` if the ask side is empty."""
        return self._ask_prices[0] if self._ask_prices else None

    def mid(self) -> float | None:
        """`(best_bid + best_ask) / 2` in ticks, or `None` unless both sides are quoted."""
        if self._bid_prices and self._ask_prices:
            return (self._bid_prices[-1] + self._ask_prices[0]) / 2
        return None

    def spread(self) -> int | None:
        """`best_ask - best_bid` in ticks, or `None` unless both sides are quoted."""
        if self._bid_prices and self._ask_prices:
            return self._ask_prices[0] - self._bid_prices[-1]
        return None

    def volume_at(self, side: Side, price: int) -> int:
        """Total resting quantity on `side` at `price` (0 if there is no such level)."""
        level = self._side_levels(_as_side(side)).get(price)
        return level.volume if level is not None else 0

    def n_levels(self, side: Side) -> int:
        """Number of non-empty price levels on `side`."""
        return len(self._bid_prices if _as_side(side) is Side.BID else self._ask_prices)

    def depth(
        self, levels: int
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
        """Aggregated depth of the best `levels` price levels on each side.

        Args:
            levels: Number of levels per side, an integer >= 0.

        Returns:
            `(bid_prices, bid_sizes, ask_prices, ask_sizes)`, each an int64 array of shape
            `(levels,)`, best level first. Missing levels have price 0 and size 0.

        Raises:
            ValueError: If `levels` is not an integer >= 0.
        """
        n = _as_int(levels, "levels", 0)
        bid_prices = np.zeros(n, dtype=np.int64)
        bid_sizes = np.zeros(n, dtype=np.int64)
        ask_prices = np.zeros(n, dtype=np.int64)
        ask_sizes = np.zeros(n, dtype=np.int64)
        k = min(n, len(self._bid_prices))
        if k:
            bids = self._bid_prices[len(self._bid_prices) - k :][::-1]
            bid_prices[:k] = bids
            bid_sizes[:k] = [self._bid_levels[p].volume for p in bids]
        k = min(n, len(self._ask_prices))
        if k:
            asks = self._ask_prices[:k]
            ask_prices[:k] = asks
            ask_sizes[:k] = [self._ask_levels[p].volume for p in asks]
        return bid_prices, bid_sizes, ask_prices, ask_sizes

    def orders_of(self, owner: int) -> list[Order]:
        """Resting orders of `owner`, oldest first (live records: do not mutate them)."""
        owned = self._by_owner.get(owner)
        return list(owned.values()) if owned else []

    def n_orders_of(self, owner: int) -> int:
        """Number of resting orders of `owner`."""
        owned = self._by_owner.get(owner)
        return len(owned) if owned else 0

    def queue_ahead(self, order_id: int) -> int:
        """Total resting quantity ahead of an order in its price level's queue.

        Args:
            order_id: Id of a resting order.

        Returns:
            The quantity that must trade at the order's price before it starts to fill.

        Raises:
            KeyError: If the order is not resting in the book.
        """
        order = self._orders.get(order_id)
        if order is None:
            raise KeyError(f"order {order_id} is not resting in the book")
        ahead = 0
        for other in self._side_levels(order.side)[order.price].orders:
            if other is order:
                break
            ahead += other.quantity  # cancelled orders hold 0
        return ahead

    def __len__(self) -> int:
        return len(self._orders)

    def __repr__(self) -> str:
        return (
            f"OrderBook(time={self.time:g}, best_bid={self.best_bid()}, "
            f"best_ask={self.best_ask()}, orders={len(self._orders)})"
        )

    # -- internals ---------------------------------------------------------------------

    def _side_levels(self, side: Side) -> dict[int, _Level]:
        return self._bid_levels if side is Side.BID else self._ask_levels

    def _match(
        self,
        side: Side,
        limit: float,
        quantity: int,
        taker_id: int,
        taker_owner: int,
        fills: list[Fill],
    ) -> int:
        """Match an incoming order against the opposite side; return the unfilled quantity."""
        is_bid = side is Side.BID
        if is_bid:
            prices, levels, best = self._ask_prices, self._ask_levels, 0
        else:
            prices, levels, best = self._bid_prices, self._bid_levels, -1
        orders, by_owner, time = self._orders, self._by_owner, self.time
        remaining = quantity
        while remaining and prices:
            price = prices[best]
            if is_bid:
                if price > limit:
                    break
            elif price < limit:
                break
            level = levels[price]
            queue = level.orders
            while remaining and level.volume:
                maker = queue[0]
                available = maker.quantity
                if available == 0:  # lazily deleted
                    queue.popleft()
                    level.dead -= 1
                    continue
                traded = available if available < remaining else remaining
                maker.quantity = available - traded
                level.volume -= traded
                remaining -= traded
                fills.append(
                    Fill(
                        price,
                        traded,
                        maker.order_id,
                        taker_id,
                        maker.owner,
                        taker_owner,
                        side,
                        time,
                    )
                )
                if traded == available:
                    queue.popleft()
                    del orders[maker.order_id]
                    del by_owner[maker.owner][maker.order_id]
            if level.volume == 0:
                del levels[price]
                del prices[best]
        if remaining != quantity:
            self.last_price = fills[-1].price
        return remaining

    def _rest(self, order: Order) -> None:
        if order.side is Side.BID:
            prices, levels = self._bid_prices, self._bid_levels
        else:
            prices, levels = self._ask_prices, self._ask_levels
        level = levels.get(order.price)
        if level is None:
            level = levels[order.price] = _Level()
            insort(prices, order.price)
        level.orders.append(order)
        level.volume += order.quantity
        self._orders[order.order_id] = order
        owned = self._by_owner.get(order.owner)
        if owned is None:
            owned = self._by_owner[order.owner] = {}
        owned[order.order_id] = order

    def _unlink(self, order: Order) -> None:
        """Remove a resting order from its level (lazily) and drop the level if empty."""
        if order.side is Side.BID:
            prices, levels = self._bid_prices, self._bid_levels
        else:
            prices, levels = self._ask_prices, self._ask_levels
        level = levels[order.price]
        level.volume -= order.quantity
        order.quantity = 0
        if level.volume == 0:
            del levels[order.price]
            del prices[bisect_left(prices, order.price)]
            return
        queue = level.orders
        level.dead += 1
        while queue[0].quantity == 0:  # a live order remains, so this terminates
            queue.popleft()
            level.dead -= 1
        if level.dead > _COMPACT_MIN_DEAD and 2 * level.dead > len(queue):
            level.orders = deque(o for o in queue if o.quantity)
            level.dead = 0


@dataclass(frozen=True)
class ZeroIntelligenceFlow:
    """Poisson background order flow for an `OrderBook`.

    In the spirit of Smith, Farmer, Gillemot & Krishnamurthy (2003) and Cont, Stoikov &
    Talreja (2010): limit orders at random distances from the opposite best quote, market
    orders, and cancellations of resting background orders, each arriving as an
    independent Poisson process. All orders are submitted with owner `BACKGROUND`.

    - **Limit orders** arrive at `limit_rate` per side. A buy is placed `k` ticks below the
      best ask and a sell `k` ticks above the best bid, with `k` in `1..max_offset` and
      `P(k)` proportional to `exp(-offset_decay * (k - 1))`, so a limit order never crosses
      on arrival (except where prices are clipped at 1 tick). If the opposite side is
      empty the reference is one tick beyond the same side's best quote, or
      `book.last_price` if the book is empty.
    - **Market orders** arrive at `market_rate` per side and take the opposite side; an
      order that finds the opposite side empty is skipped.
    - **Cancellations** remove a uniformly random resting background order, at
      `cancel_rate` per resting background order. Orders of other owners (agents) are
      never cancelled.
    - **Sizes** are `1 + Poisson(mean_size - 1)`, so their mean is `mean_size`.

    If an event leaves one side of the book empty, a background limit order is placed on
    that side at once, so the book stays two-sided.

    Volatility is governed mostly by `cancel_rate` relative to `limit_rate`. With the
    defaults the book holds about 30 resting orders, the spread is about one tick and
    the mid diffuses by roughly 0.5 ticks per unit time; `cancel_rate=1.0` gives a
    thinner book and about 1.3 ticks, while a low rate such as 0.1 lets the queues at
    the best quotes grow so long that the mid barely moves.

    Args:
        limit_rate: Limit orders per unit time, per side, > 0.
        market_rate: Market orders per unit time, per side, > 0.
        cancel_rate: Cancellations per resting background order per unit time, >= 0.
        max_offset: Largest limit-order distance from the opposite best quote in ticks, >= 1.
        offset_decay: Decay rate of the offset distribution per tick, >= 0 (0 is uniform).
        mean_size: Mean order size, >= 1.

    Raises:
        ValueError: If a parameter is out of range.
    """

    limit_rate: float = 10.0
    market_rate: float = 2.0
    cancel_rate: float = 0.5
    max_offset: int = 10
    offset_decay: float = 0.5
    mean_size: float = 3.0

    def __post_init__(self) -> None:
        for name in ("limit_rate", "market_rate"):
            if not (math.isfinite(getattr(self, name)) and getattr(self, name) > 0):
                raise ValueError(f"{name} must be finite and > 0, got {getattr(self, name)}.")
        for name in ("cancel_rate", "offset_decay"):
            if not (math.isfinite(getattr(self, name)) and getattr(self, name) >= 0):
                raise ValueError(f"{name} must be finite and >= 0, got {getattr(self, name)}.")
        if not (math.isfinite(self.mean_size) and self.mean_size >= 1):
            raise ValueError(f"mean_size must be finite and >= 1, got {self.mean_size}.")
        object.__setattr__(self, "max_offset", _as_int(self.max_offset, "max_offset", 1))

    def initialise(
        self,
        book: OrderBook,
        rng: np.random.Generator,
        *,
        mid: int,
        levels: int = 10,
        orders_per_level: int = 3,
    ) -> None:
        """Clear `book` and seed it with a symmetric two-sided background book.

        Bids rest at `mid - 1, ..., mid - levels` and asks at `mid + 1, ..., mid + levels`
        (prices clipped to >= 1 tick), each level holding `orders_per_level` background
        orders of random size. The book's time is reset to 0 and `book.last_price` is set
        to `mid` as the reference price.

        Args:
            book: The book to (re)initialise.
            rng: Source of randomness.
            mid: Initial mid price in ticks, an integer >= 1.
            levels: Price levels per side, an integer >= 1.
            orders_per_level: Orders per price level, an integer >= 1.

        Raises:
            ValueError: If `mid`, `levels` or `orders_per_level` is invalid.
        """
        mid = _as_int(mid, "mid", 1)
        levels = _as_int(levels, "levels", 1)
        orders_per_level = _as_int(orders_per_level, "orders_per_level", 1)
        book.clear()
        book.last_price = mid
        for distance in range(1, levels + 1):
            for _ in range(orders_per_level):
                book.limit(Side.BID, max(mid - distance, 1), self._draw_size(rng), BACKGROUND)
                book.limit(Side.ASK, mid + distance, self._draw_size(rng), BACKGROUND)

    def advance(self, book: OrderBook, rng: np.random.Generator, until: float) -> list[Fill]:
        """Simulate background events from `book.time` up to time `until`.

        Events arrive as competing Poisson clocks with total rate
        `2 * limit_rate + 2 * market_rate + cancel_rate * n`, where `n` is the number of
        resting background orders, recomputed after each event. `book.time` is set to
        each event's time as it happens and to `until` at the end.

        Args:
            book: The book to drive; it may also hold other owners' (agents') orders.
            rng: Source of randomness.
            until: End time, finite and >= `book.time`.

        Returns:
            Every fill that occurred, in order, including fills against non-background
            orders.

        Raises:
            ValueError: If `until` is not finite or is earlier than `book.time`, or if a
                limit order is needed while the book is empty and `book.last_price` is
                `None` (call `initialise` first).
        """
        until = float(until)
        if not math.isfinite(until):
            raise ValueError(f"until must be finite, got {until}.")
        if until < book.time:
            raise ValueError(f"until ({until}) is earlier than the book time ({book.time}).")
        lr, mr, cr = self.limit_rate, self.market_rate, self.cancel_rate
        limit_total = 2.0 * lr
        market_bid = limit_total + mr
        base = limit_total + 2.0 * mr
        fills: list[Fill] = []
        t = float(book.time)
        while True:
            n_background = book.n_orders_of(BACKGROUND)
            rate = base + cr * n_background
            t += rng.standard_exponential() / rate
            if t > until:
                break
            book.time = t
            u = rng.random() * rate
            if u < limit_total:
                self._place_limit(book, rng, Side.BID if u < lr else Side.ASK, fills)
            elif u < base:
                side = Side.BID if u < market_bid else Side.ASK
                if book.n_levels(side.opposite):
                    fills.extend(book.market(side, self._draw_size(rng), BACKGROUND)[1])
            elif n_background:
                owned = book._by_owner[BACKGROUND]
                index = int(rng.integers(n_background))
                book.cancel(next(itertools.islice(owned, index, None)))
            # Keep the book two-sided. Terminates: a placement either rests (filling the
            # empty side) or strictly consumes the finite opposite volume.
            while True:
                if not book._bid_prices:
                    self._place_limit(book, rng, Side.BID, fills)
                elif not book._ask_prices:
                    self._place_limit(book, rng, Side.ASK, fills)
                else:
                    break
        book.time = until
        return fills

    def _place_limit(
        self, book: OrderBook, rng: np.random.Generator, side: Side, fills: list[Fill]
    ) -> None:
        """Place one background limit order on `side`, appending any fills."""
        bid, ask = book.best_bid(), book.best_ask()
        if side is Side.BID:
            reference = ask if ask is not None else (bid + 1 if bid is not None else None)
        else:
            reference = bid if bid is not None else (ask - 1 if ask is not None else None)
        if reference is None:
            reference = book.last_price
            if reference is None:
                raise ValueError(
                    "the book is empty and has no reference price; call initialise() first "
                    "or set book.last_price."
                )
        offset = self._draw_offset(rng)
        price = max(reference - offset if side is Side.BID else reference + offset, 1)
        fills.extend(book.limit(side, price, self._draw_size(rng), BACKGROUND)[1])

    def _draw_offset(self, rng: np.random.Generator) -> int:
        """Sample `k` in `1..max_offset` with `P(k)` proportional to `exp(-decay * (k - 1))`."""
        u = rng.random()
        n, decay = self.max_offset, self.offset_decay
        if decay == 0:
            k = int(u * n) + 1
        else:
            # Inverse CDF of the truncated geometric distribution.
            k = math.ceil(-math.log1p(u * math.expm1(-decay * n)) / decay)
        return min(max(k, 1), n)

    def _draw_size(self, rng: np.random.Generator) -> int:
        """Sample an order size `1 + Poisson(mean_size - 1)`."""
        if self.mean_size == 1:
            return 1
        return 1 + int(rng.poisson(self.mean_size - 1.0))


def _as_int(value: Any, name: str, minimum: int) -> int:
    """`value` as a Python int >= `minimum` (numpy integers accepted, bools rejected)."""
    if type(value) is int:
        result: int = value
    elif isinstance(value, int | np.integer) and not isinstance(value, bool | np.bool_):
        result = int(value)
    else:
        raise ValueError(f"{name} must be an integer, got {value!r}.")
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}.")
    return result


def _as_owner(owner: Any) -> int:
    if isinstance(owner, bool | np.bool_):
        raise ValueError(f"owner must be an integer, got {owner!r}.")
    try:
        return operator.index(owner)
    except TypeError:
        raise ValueError(f"owner must be an integer, got {owner!r}.") from None


def _as_side(side: Any) -> Side:
    if side is Side.BID or side is Side.ASK:
        return Side(side)
    if not isinstance(side, bool | np.bool_):
        try:
            return Side(side)
        except ValueError:
            pass
    raise ValueError(f"side must be Side.BID (1) or Side.ASK (-1), got {side!r}.")
