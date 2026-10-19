"""Tests for the limit order book matching engine and the zero-intelligence background flow."""

from __future__ import annotations

import math
import pickle
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pytest

from sekai.finance.lob import BACKGROUND, Fill, Order, OrderBook, Side, ZeroIntelligenceFlow

AGENT = 0
OTHER = 1
OWNERS = (BACKGROUND, AGENT, OTHER)


def assert_consistent(book: OrderBook, owners: tuple[int, ...] = OWNERS) -> None:
    """Cross-check the book's aggregate views against its individual resting orders."""
    orders = [order for owner in owners for order in book.orders_of(owner)]
    assert len(orders) == len(book)
    volume: dict[tuple[Side, int], int] = defaultdict(int)
    for order in orders:
        assert order.quantity > 0
        assert book.get(order.order_id) is order
        volume[order.side, order.price] += order.quantity
    n = max(book.n_levels(Side.BID), book.n_levels(Side.ASK), 1)
    bid_prices, bid_sizes, ask_prices, ask_sizes = book.depth(n)
    from_depth = {}
    for side, prices, sizes in (
        (Side.BID, bid_prices, bid_sizes),
        (Side.ASK, ask_prices, ask_sizes),
    ):
        for price, size in zip(prices, sizes):
            if size:
                from_depth[side, int(price)] = int(size)
                assert book.volume_at(side, int(price)) == size
    assert from_depth == dict(volume)
    live_bids = bid_prices[bid_sizes > 0]
    live_asks = ask_prices[ask_sizes > 0]
    assert np.all(np.diff(live_bids) < 0) and np.all(np.diff(live_asks) > 0)
    bid, ask = book.best_bid(), book.best_ask()
    if bid is not None and ask is not None:
        assert bid < ask
        assert book.spread() == ask - bid and book.mid() == (bid + ask) / 2
    else:
        assert book.spread() is None and book.mid() is None


class TestSide:
    def test_opposite(self) -> None:
        assert Side.BID.opposite is Side.ASK and Side.ASK.opposite is Side.BID
        assert int(Side.BID) == 1 and int(Side.ASK) == -1


class TestOrderBook:
    def test_empty_book(self) -> None:
        book = OrderBook()
        assert len(book) == 0 and book.time == 0.0 and book.last_price is None
        assert book.best_bid() is None and book.best_ask() is None
        assert book.mid() is None and book.spread() is None
        assert book.volume_at(Side.BID, 100) == 0
        assert book.get(1) is None and book.orders_of(AGENT) == []

    def test_quotes_after_each_operation(self) -> None:
        book = OrderBook()
        bid_id, fills = book.limit(Side.BID, 99, 5)
        assert bid_id is not None and fills == []
        assert (book.best_bid(), book.best_ask(), book.mid(), book.spread()) == (
            99,
            None,
            None,
            None,
        )
        ask_id, _ = book.limit(Side.ASK, 101, 5)
        assert ask_id is not None
        assert (book.best_bid(), book.best_ask(), book.mid(), book.spread()) == (99, 101, 100.0, 2)
        better, _ = book.limit(Side.BID, 100, 2)
        assert (book.best_bid(), book.mid(), book.spread()) == (100, 100.5, 1)
        book.market(Side.ASK, 2)  # takes the whole bid at 100
        assert (book.best_bid(), book.mid(), book.spread()) == (99, 100.0, 2)
        assert book.get(better) is None
        assert book.cancel(ask_id)
        assert (book.best_ask(), book.mid(), book.spread()) == (None, None, None)
        assert book.reduce(bid_id, 1)
        assert book.best_bid() == 99 and book.volume_at(Side.BID, 99) == 1
        assert_consistent(book)

    def test_crossing_limit_fills_at_resting_price(self) -> None:
        book = OrderBook()
        book.time = 2.5
        ask_id, _ = book.limit(Side.ASK, 101, 5, owner=OTHER)
        assert ask_id is not None
        book.time = 3.0
        order_id, fills = book.limit(Side.BID, 105, 3, owner=AGENT)
        assert order_id is None  # filled completely
        assert fills == [
            Fill(
                price=101,
                quantity=3,
                maker_order_id=ask_id,
                taker_order_id=ask_id + 1,
                maker_owner=OTHER,
                taker_owner=AGENT,
                taker_side=Side.BID,
                time=3.0,
            )
        ]
        resting = book.get(ask_id)
        assert resting is not None and resting.quantity == 2 and resting.time == 2.5
        assert book.last_price == 101 and book.best_bid() is None
        assert_consistent(book)

    def test_price_then_time_priority(self) -> None:
        book = OrderBook()
        ids = []
        for t, price in enumerate([102, 101, 101, 103]):
            book.time = float(t)
            order_id, _ = book.limit(Side.ASK, price, 2, owner=OTHER)
            ids.append(order_id)
        _, fills = book.market(Side.BID, 7, owner=AGENT)
        assert [f.maker_order_id for f in fills] == [ids[1], ids[2], ids[0], ids[3]]
        assert [f.price for f in fills] == [101, 101, 102, 103]
        assert [f.quantity for f in fills] == [2, 2, 2, 1]
        remaining = book.get(ids[3])
        assert remaining is not None and remaining.quantity == 1
        assert [book.get(i) for i in ids[:3]] == [None, None, None]

    def test_bid_priority(self) -> None:
        book = OrderBook()
        first, _ = book.limit(Side.BID, 100, 1)
        better, _ = book.limit(Side.BID, 101, 1)
        second, _ = book.limit(Side.BID, 100, 1)
        _, fills = book.market(Side.ASK, 3)
        assert [f.maker_order_id for f in fills] == [better, first, second]
        assert all(f.taker_side is Side.ASK for f in fills)

    def test_partial_fill_rests_remainder(self) -> None:
        book = OrderBook()
        bid_id, _ = book.limit(Side.BID, 100, 3, owner=OTHER)
        order_id, fills = book.limit(Side.ASK, 100, 5, owner=AGENT)
        assert order_id is not None and order_id > (bid_id or 0)
        assert len(fills) == 1 and fills[0].quantity == 3
        assert fills[0].taker_order_id == order_id and fills[0].maker_order_id == bid_id
        rest = book.get(order_id)
        assert rest == Order(order_id, Side.ASK, 100, 2, AGENT, 0.0)
        assert book.best_bid() is None and book.best_ask() == 100
        assert book.orders_of(AGENT) == [rest]
        assert_consistent(book)

    def test_limit_sweeps_levels_up_to_its_price(self) -> None:
        book = OrderBook()
        for price in (101, 102, 104):
            book.limit(Side.ASK, price, 2)
        order_id, fills = book.limit(Side.BID, 102, 10, owner=AGENT)
        assert [(f.price, f.quantity) for f in fills] == [(101, 2), (102, 2)]
        assert order_id is not None and book.volume_at(Side.BID, 102) == 6
        assert book.best_bid() == 102 and book.best_ask() == 104
        assert_consistent(book)

    def test_market_walks_levels_and_drops_remainder(self) -> None:
        book = OrderBook()
        book.limit(Side.ASK, 101, 2)
        book.limit(Side.ASK, 102, 3)
        order_id, fills = book.market(Side.BID, 10, owner=AGENT)
        assert [(f.price, f.quantity) for f in fills] == [(101, 2), (102, 3)]
        assert all(f.taker_order_id == order_id for f in fills)
        assert len(book) == 0 and book.best_ask() is None and book.get(order_id) is None
        next_id, no_fills = book.market(Side.BID, 1)
        assert no_fills == [] and next_id == order_id + 1
        assert book.last_price == 102

    def test_ids_unique_and_increasing(self) -> None:
        book = OrderBook()
        book.limit(Side.ASK, 101, 1)
        ids = [
            book.limit(Side.BID, 99, 1)[0],
            book.limit(Side.BID, 101, 1)[1][0].taker_order_id,  # filled on arrival
            book.market(Side.ASK, 1)[0],
            book.limit(Side.ASK, 105, 1)[0],
        ]
        assert ids == [2, 3, 4, 5]

    def test_self_trade_allowed(self) -> None:
        book = OrderBook()
        book.limit(Side.BID, 100, 2, owner=AGENT)
        _, fills = book.limit(Side.ASK, 100, 2, owner=AGENT)
        assert len(fills) == 1 and fills[0].maker_owner == fills[0].taker_owner == AGENT

    def test_cancel(self) -> None:
        book = OrderBook()
        b1, _ = book.limit(Side.BID, 100, 2)
        b2, _ = book.limit(Side.BID, 100, 3)
        b3, _ = book.limit(Side.BID, 100, 4)
        assert b1 is not None and b2 is not None and b3 is not None
        held = book.get(b2)
        assert book.cancel(b2)
        assert held is not None and held.quantity == 0
        assert not book.cancel(b2) and not book.cancel(999)
        assert book.volume_at(Side.BID, 100) == 6 and len(book) == 2
        _, fills = book.market(Side.ASK, 5)
        assert [(f.maker_order_id, f.quantity) for f in fills] == [(b1, 2), (b3, 3)]
        assert book.cancel(b3)
        assert book.best_bid() is None and book.n_levels(Side.BID) == 0
        assert_consistent(book)

    def test_cancel_at_inner_level_keeps_sorted_levels(self) -> None:
        book = OrderBook()
        ids = {price: book.limit(Side.ASK, price, 1)[0] for price in (101, 102, 103)}
        assert book.cancel(ids[102])
        assert book.depth(3)[2].tolist() == [101, 103, 0]
        book.limit(Side.ASK, 102, 1)
        assert book.depth(3)[2].tolist() == [101, 102, 103]

    def test_lazy_deletion_compacts(self) -> None:
        book = OrderBook()
        head, _ = book.limit(Side.BID, 100, 1)
        ids = [book.limit(Side.BID, 100, 1)[0] for _ in range(200)]
        for order_id in ids[:-1]:
            assert book.cancel(order_id)
        assert book.volume_at(Side.BID, 100) == 2 and book.queue_ahead(ids[-1]) == 1
        _, fills = book.market(Side.ASK, 2)
        assert [f.maker_order_id for f in fills] == [head, ids[-1]]
        assert len(book) == 0

    def test_reduce_keeps_priority(self) -> None:
        book = OrderBook()
        a1, _ = book.limit(Side.ASK, 101, 5)
        a2, _ = book.limit(Side.ASK, 101, 5)
        assert a1 is not None and a2 is not None
        assert book.reduce(a1, 2)
        assert book.volume_at(Side.ASK, 101) == 7
        _, fills = book.market(Side.BID, 3)
        assert [(f.maker_order_id, f.quantity) for f in fills] == [(a1, 2), (a2, 1)]
        with pytest.raises(ValueError, match="below"):
            book.reduce(a2, 4)
        with pytest.raises(ValueError, match="below"):
            book.reduce(a2, 5)
        assert book.reduce(a2, 0)  # cancels
        assert book.get(a2) is None and len(book) == 0
        assert not book.reduce(a2, 1) and not book.reduce(12345, 0)

    def test_depth_with_fewer_levels_than_requested(self) -> None:
        book = OrderBook()
        book.limit(Side.BID, 100, 2)
        book.limit(Side.BID, 100, 1)
        book.limit(Side.BID, 98, 4)
        book.limit(Side.ASK, 103, 7)
        bid_prices, bid_sizes, ask_prices, ask_sizes = book.depth(4)
        for array in (bid_prices, bid_sizes, ask_prices, ask_sizes):
            assert array.shape == (4,) and array.dtype == np.int64
        assert bid_prices.tolist() == [100, 98, 0, 0]
        assert bid_sizes.tolist() == [3, 4, 0, 0]
        assert ask_prices.tolist() == [103, 0, 0, 0]
        assert ask_sizes.tolist() == [7, 0, 0, 0]
        assert book.depth(1)[0].tolist() == [100]
        assert all(a.shape == (0,) for a in book.depth(0))

    def test_queue_ahead(self) -> None:
        book = OrderBook()
        b1, _ = book.limit(Side.BID, 100, 2)
        b2, _ = book.limit(Side.BID, 100, 3)
        b3, _ = book.limit(Side.BID, 100, 4)
        other, _ = book.limit(Side.BID, 99, 10)
        assert b1 is not None and b2 is not None and b3 is not None and other is not None
        assert [book.queue_ahead(i) for i in (b1, b2, b3, other)] == [0, 2, 5, 0]
        book.cancel(b1)
        assert book.queue_ahead(b3) == 3
        book.reduce(b2, 1)
        assert book.queue_ahead(b3) == 1
        book.market(Side.ASK, 2)  # fills b2 (1) and 1 of b3
        assert book.queue_ahead(b3) == 0
        with pytest.raises(KeyError):
            book.queue_ahead(b1)

    def test_orders_of_and_len(self) -> None:
        book = OrderBook()
        a, _ = book.limit(Side.BID, 90, 1, owner=AGENT)
        book.limit(Side.ASK, 110, 1, owner=OTHER)
        b, _ = book.limit(Side.ASK, 111, 1, owner=AGENT)
        c, _ = book.limit(Side.BID, 91, 1, owner=AGENT)
        assert [o.order_id for o in book.orders_of(AGENT)] == [a, b, c]
        assert book.n_orders_of(AGENT) == 3 and book.n_orders_of(BACKGROUND) == 0
        assert len(book) == 4
        book.cancel(b)
        assert [o.order_id for o in book.orders_of(AGENT)] == [a, c]
        assert book.n_levels(Side.BID) == 2 and book.n_levels(Side.ASK) == 1

    def test_clear(self) -> None:
        book = OrderBook()
        book.time = 5.0
        order_id, _ = book.limit(Side.BID, 100, 1)
        book.limit(Side.ASK, 100, 1)
        book.limit(Side.ASK, 101, 1)
        held = book.get(3)
        book.clear()
        assert len(book) == 0 and book.time == 0.0 and book.last_price is None
        assert book.best_ask() is None and held is not None and held.quantity == 0
        assert book.limit(Side.BID, 100, 1)[0] == order_id == 1

    def test_validation(self) -> None:
        book = OrderBook()
        bad_limits = [
            (Side.BID, 0, 1),
            (Side.BID, -3, 1),
            (Side.BID, 100, 0),
            (Side.BID, 100.5, 1),
            (Side.BID, 100.0, 1),
            (Side.BID, 100, 2.0),
            (Side.BID, True, 1),
            (Side.BID, "100", 1),
            (0, 100, 1),
            (2, 100, 1),
            ("bid", 100, 1),
        ]
        for args in bad_limits:
            with pytest.raises(ValueError):
                book.limit(*args)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            book.market(Side.ASK, 0)
        with pytest.raises(ValueError):
            book.limit(True, 100, 1)  # type: ignore[arg-type]
        for owner in (1.5, True, "agent"):
            with pytest.raises(ValueError):
                book.limit(Side.BID, 100, 1, owner=owner)  # type: ignore[arg-type]
            with pytest.raises(ValueError):
                book.market(Side.BID, 1, owner=owner)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            book.reduce(1, -1)
        with pytest.raises(ValueError):
            book.depth(-1)
        assert len(book) == 0
        assert book.limit(Side.BID, 100, 1)[0] == 1  # rejected orders consume no ids

    def test_accepts_numpy_integers_and_int_sides(self) -> None:
        book = OrderBook()
        order_id, _ = book.limit(1, np.int64(100), np.int32(3), owner=np.int64(AGENT))  # type: ignore[arg-type]
        assert order_id is not None
        order = book.get(order_id)
        assert order is not None and order.side is Side.BID
        assert type(order.price) is int and type(order.quantity) is int and type(order.owner) is int
        _, fills = book.market(-1, np.uint8(2), owner=np.int32(OTHER))  # type: ignore[arg-type]
        assert fills[0].taker_side is Side.ASK and type(fills[0].price) is int
        assert type(fills[0].taker_owner) is int and fills[0].taker_owner == OTHER

    def test_pickle_round_trip(self) -> None:
        book = OrderBook()
        flow = ZeroIntelligenceFlow(cancel_rate=0.5)
        flow.initialise(book, np.random.default_rng(0), mid=1000)
        book.limit(Side.BID, 999, 2, owner=AGENT)
        for order_id in (3, 5, 7):
            book.cancel(order_id)
        copy = pickle.loads(pickle.dumps(book))
        assert repr(copy) == repr(book) and copy.last_price == book.last_price
        assert all(np.array_equal(a, b) for a, b in zip(copy.depth(20), book.depth(20)))
        assert copy.orders_of(AGENT) == book.orders_of(AGENT)
        fills = flow.advance(book, np.random.default_rng(1), 20.0)
        assert flow.advance(copy, np.random.default_rng(1), 20.0) == fills
        assert pickle.loads(pickle.dumps(fills)) == fills
        assert pickle.loads(pickle.dumps(flow)) == flow

    def test_repr(self) -> None:
        book = OrderBook()
        book.limit(Side.BID, 99, 1)
        assert repr(book) == "OrderBook(time=0, best_bid=99, best_ask=None, orders=1)"


@dataclass
class _NaiveOrder:
    order_id: int
    side: Side
    price: int
    quantity: int
    owner: int


class _NaiveBook:
    """A deliberately simple O(n log n)-per-order reference implementation."""

    def __init__(self) -> None:
        self.orders: dict[int, _NaiveOrder] = {}
        self.next_id = 1

    def submit(
        self, side: Side, price: int | None, quantity: int, owner: int
    ) -> tuple[int, list[tuple[int, int, int, int, int, int, Side]]]:
        order_id = self.next_id
        self.next_id += 1
        opposite = [o for o in self.orders.values() if o.side is not side]
        if side is Side.BID:
            opposite.sort(key=lambda o: (o.price, o.order_id))
        else:
            opposite.sort(key=lambda o: (-o.price, o.order_id))
        fills = []
        for maker in opposite:
            if quantity == 0:
                break
            if price is not None and (
                maker.price > price if side is Side.BID else maker.price < price
            ):
                break
            traded = min(quantity, maker.quantity)
            fills.append((maker.price, traded, maker.order_id, order_id, maker.owner, owner, side))
            maker.quantity -= traded
            quantity -= traded
            if maker.quantity == 0:
                del self.orders[maker.order_id]
        if price is not None and quantity:
            self.orders[order_id] = _NaiveOrder(order_id, side, price, quantity, owner)
        return order_id, fills


def test_matches_naive_reference_implementation() -> None:
    rng = np.random.default_rng(12345)
    book, naive = OrderBook(), _NaiveBook()
    ids: list[int] = []
    for step in range(3000):
        book.time = float(step)
        u = rng.random()
        side = Side.BID if rng.random() < 0.5 else Side.ASK
        owner = int(rng.integers(-1, 2))
        if u < 0.5:
            price = int(rng.integers(95, 106))
            quantity = int(rng.integers(1, 6))
            order_id, fills = book.limit(side, price, quantity, owner)
            ref_id, ref_fills = naive.submit(side, price, quantity, owner)
            assert order_id in (None, ref_id)
            assert (order_id is not None) == (ref_id in naive.orders)
            ids.append(ref_id)
        elif u < 0.65:
            quantity = int(rng.integers(1, 12))
            order_id, fills = book.market(side, quantity, owner)
            ref_id, ref_fills = naive.submit(side, None, quantity, owner)
            assert order_id == ref_id
        elif u < 0.9 and ids:
            target = ids[int(rng.integers(len(ids)))]
            assert book.cancel(target) == (naive.orders.pop(target, None) is not None)
            continue
        elif ids:
            target = ids[int(rng.integers(len(ids)))]
            ref = naive.orders.get(target)
            if ref is None:
                assert not book.reduce(target, 0)
            elif ref.quantity > 1:
                new = int(rng.integers(1, ref.quantity))
                assert book.reduce(target, new)
                ref.quantity = new
            continue
        else:
            continue
        got = [
            (
                f.price,
                f.quantity,
                f.maker_order_id,
                f.taker_order_id,
                f.maker_owner,
                f.taker_owner,
                f.taker_side,
            )
            for f in fills
        ]
        assert got == ref_fills
        assert all(f.time == float(step) for f in fills)
        if step % 50 == 0:
            assert_consistent(book)
            assert len(book) == len(naive.orders)
            for ref in naive.orders.values():
                order = book.get(ref.order_id)
                assert order is not None
                assert (order.side, order.price, order.quantity, order.owner) == (
                    ref.side,
                    ref.price,
                    ref.quantity,
                    ref.owner,
                )
                ahead = sum(
                    o.quantity
                    for o in naive.orders.values()
                    if o.side is ref.side and o.price == ref.price and o.order_id < ref.order_id
                )
                assert book.queue_ahead(ref.order_id) == ahead


class TestZeroIntelligenceFlow:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"limit_rate": 0.0},
            {"market_rate": -1.0},
            {"limit_rate": math.inf},
            {"cancel_rate": -0.1},
            {"max_offset": 0},
            {"max_offset": 2.5},
            {"offset_decay": -1.0},
            {"mean_size": 0.5},
            {"mean_size": math.nan},
        ],
    )
    def test_validation(self, kwargs: dict[str, float]) -> None:
        with pytest.raises(ValueError):
            ZeroIntelligenceFlow(**kwargs)  # type: ignore[arg-type]

    def test_zero_cancel_rate_allowed(self) -> None:
        ZeroIntelligenceFlow(cancel_rate=0.0, offset_decay=0.0, mean_size=1.0)

    def test_initialise(self) -> None:
        book = OrderBook()
        book.limit(Side.BID, 5, 1, owner=AGENT)
        book.time = 7.0
        flow = ZeroIntelligenceFlow()
        flow.initialise(book, np.random.default_rng(0), mid=1000, levels=6, orders_per_level=4)
        assert book.time == 0.0 and book.last_price == 1000 and book.orders_of(AGENT) == []
        assert len(book) == book.n_orders_of(BACKGROUND) == 2 * 6 * 4
        bid_prices, bid_sizes, ask_prices, ask_sizes = book.depth(8)
        assert bid_prices.tolist() == [999, 998, 997, 996, 995, 994, 0, 0]
        assert ask_prices.tolist() == [1001, 1002, 1003, 1004, 1005, 1006, 0, 0]
        assert np.all(bid_sizes[:6] >= 4) and np.all(ask_sizes[:6] >= 4)
        assert all(len(level) == 4 for level in _orders_by_price(book).values())
        assert book.mid() == 1000.0 and book.spread() == 2
        assert_consistent(book)

    def test_initialise_clips_prices(self) -> None:
        book = OrderBook()
        ZeroIntelligenceFlow().initialise(book, np.random.default_rng(0), mid=3, levels=5)
        assert book.depth(5)[0].tolist() == [2, 1, 0, 0, 0]
        assert book.n_orders_of(BACKGROUND) == 2 * 5 * 3
        with pytest.raises(ValueError):
            ZeroIntelligenceFlow().initialise(book, np.random.default_rng(0), mid=0)

    @pytest.mark.parametrize(
        "flow",
        [
            ZeroIntelligenceFlow(),
            ZeroIntelligenceFlow(limit_rate=5.0, market_rate=2.0, cancel_rate=1.0),
            ZeroIntelligenceFlow(limit_rate=1.0, market_rate=3.0, cancel_rate=2.0, mean_size=4.0),
        ],
        ids=["default", "thin", "very-thin"],
    )
    def test_advance_keeps_book_two_sided(self, flow: ZeroIntelligenceFlow) -> None:
        book = OrderBook()
        rng = np.random.default_rng(7)
        flow.initialise(book, rng, mid=10_000, levels=3, orders_per_level=1)
        previous = 0.0
        n_fills = 0
        for i in range(1, 1001):
            until = i * 0.25
            fills = flow.advance(book, rng, until)
            n_fills += len(fills)
            assert book.time == until
            assert all(previous < f.time <= until for f in fills)
            assert [f.time for f in fills] == sorted(f.time for f in fills)
            previous = until
            assert book.best_bid() is not None and book.best_ask() is not None
            if i % 100 == 0:
                assert_consistent(book)
        assert n_fills > 100

    def test_advance_time_validation(self) -> None:
        flow, book, rng = ZeroIntelligenceFlow(), OrderBook(), np.random.default_rng(0)
        flow.initialise(book, rng, mid=100)
        book.time = 5.0
        before = book.depth(20)
        assert flow.advance(book, rng, 5.0) == []
        assert all(np.array_equal(a, b) for a, b in zip(before, book.depth(20)))
        with pytest.raises(ValueError, match="earlier"):
            flow.advance(book, rng, 4.0)
        with pytest.raises(ValueError, match="finite"):
            flow.advance(book, rng, math.inf)

    def test_advance_on_empty_book_needs_reference(self) -> None:
        with pytest.raises(ValueError, match="reference price"):
            ZeroIntelligenceFlow().advance(OrderBook(), np.random.default_rng(0), 10.0)
        book = OrderBook()
        book.last_price = 500
        ZeroIntelligenceFlow().advance(book, np.random.default_rng(0), 10.0)
        assert book.best_bid() is not None and book.best_ask() is not None

    @pytest.mark.parametrize("seed", range(8))
    def test_advance_refills_one_sided_book(self, seed: int) -> None:
        # Market buys arriving while the ask side is empty are skipped.
        book = OrderBook()
        for price in (98, 99, 100):
            book.limit(Side.BID, price, 2)
        flow = ZeroIntelligenceFlow(limit_rate=0.5, market_rate=20.0)
        flow.advance(book, np.random.default_rng(seed), 1.0)
        bid, ask = book.best_bid(), book.best_ask()
        assert bid is not None and ask is not None and bid < ask
        assert_consistent(book)

    def test_advance_is_reproducible(self) -> None:
        def run(seed: int) -> tuple[list[Fill], list[list[int]]]:
            flow, book, rng = (
                ZeroIntelligenceFlow(cancel_rate=0.5),
                OrderBook(),
                np.random.default_rng(seed),
            )
            flow.initialise(book, rng, mid=5000)
            book.limit(Side.BID, 4999, 2, owner=AGENT)
            fills = flow.advance(book, rng, 50.0)
            return fills, [a.tolist() for a in book.depth(10)]

        assert run(3) == run(3)
        assert run(3) != run(4)

    def test_never_cancels_agent_orders(self) -> None:
        flow, book, rng = (
            ZeroIntelligenceFlow(cancel_rate=0.5),
            OrderBook(),
            np.random.default_rng(11),
        )
        flow.initialise(book, rng, mid=10_000)
        placed = {}
        for price in (9_999, 9_995, 9_990, 9_900, 1):
            order_id, _ = book.limit(Side.BID, price, 3, owner=AGENT)
            assert order_id is not None
            placed[order_id] = 3
        for price in (10_001, 10_005, 10_010, 10_100, 10**9):
            order_id, _ = book.limit(Side.ASK, price, 3, owner=AGENT)
            assert order_id is not None
            placed[order_id] = 3
        filled: dict[int, int] = defaultdict(int)
        for i in range(1, 201):
            for fill in flow.advance(book, rng, float(i)):
                if fill.maker_owner == AGENT:
                    filled[fill.maker_order_id] += fill.quantity
                assert fill.taker_owner == BACKGROUND
        for order_id, quantity in placed.items():
            order = book.get(order_id)
            remaining = order.quantity if order is not None else 0
            assert filled[order_id] + remaining == quantity
        assert any(book.get(order_id) is not None for order_id in placed)
        assert any(order_id in filled for order_id in placed)

    def test_market_order_count_matches_rate(self) -> None:
        flow = ZeroIntelligenceFlow(market_rate=2.0, cancel_rate=0.5)
        book, rng = OrderBook(), np.random.default_rng(2024)
        flow.initialise(book, rng, mid=10_000)
        horizon = 500.0
        fills = flow.advance(book, rng, horizon)
        takers = {f.taker_order_id: f.taker_side for f in fills if f.taker_owner == BACKGROUND}
        expected = 2 * flow.market_rate * horizon
        assert abs(len(takers) - expected) < 0.1 * expected
        buys = sum(side is Side.BID for side in takers.values())
        assert abs(buys - expected / 2) < 0.1 * expected

    def test_agent_order_fills_against_flow(self) -> None:
        flow, book, rng = ZeroIntelligenceFlow(), OrderBook(), np.random.default_rng(5)
        flow.initialise(book, rng, mid=10_000)
        best_bid = book.best_bid()
        assert best_bid is not None
        order_id, _ = book.limit(Side.BID, best_bid + 1, 2, owner=AGENT)
        assert order_id is not None and book.best_bid() == best_bid + 1
        agent_fills: list[Fill] = []
        t = 0.0
        while not agent_fills and t < 100.0:
            t += 0.5
            agent_fills = [f for f in flow.advance(book, rng, t) if f.maker_order_id == order_id]
        assert agent_fills, "agent order never filled"
        fill = agent_fills[0]
        assert fill.maker_owner == AGENT and fill.taker_owner == BACKGROUND
        assert fill.taker_side is Side.ASK and fill.price == best_bid + 1

    @pytest.mark.parametrize("decay", [0.0, 0.5, 2.0])
    def test_offset_distribution(self, decay: float) -> None:
        flow = ZeroIntelligenceFlow(max_offset=6, offset_decay=decay)
        rng = np.random.default_rng(0)
        draws = [flow._draw_offset(rng) for _ in range(20_000)]
        assert min(draws) >= 1 and max(draws) <= 6
        weights = [math.exp(-decay * i) for i in range(6)]
        for k, weight in enumerate(weights, start=1):
            assert abs(draws.count(k) / len(draws) - weight / sum(weights)) < 0.015

    def test_size_distribution(self) -> None:
        rng = np.random.default_rng(0)
        flow = ZeroIntelligenceFlow(mean_size=4.0)
        sizes = [flow._draw_size(rng) for _ in range(20_000)]
        assert min(sizes) >= 1 and abs(sum(sizes) / len(sizes) - 4.0) < 0.1
        assert ZeroIntelligenceFlow(mean_size=1.0)._draw_size(rng) == 1


def _orders_by_price(book: OrderBook) -> dict[tuple[Side, int], list[Order]]:
    levels: dict[tuple[Side, int], list[Order]] = defaultdict(list)
    for order in book.orders_of(BACKGROUND):
        levels[order.side, order.price].append(order)
    return levels
