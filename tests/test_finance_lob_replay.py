"""Tests for historical order-book replay, the LOBSTER reader, and order latency in the LOB envs."""

from __future__ import annotations

import math
import pickle
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import sekai
from sekai.envs.finance import LimitOrderBookEnv, LimitOrderBookMarket
from sekai.finance import (
    BACKGROUND,
    BookSnapshot,
    BookSnapshots,
    FixedStart,
    MessageReplay,
    MessageType,
    OrderBook,
    OrderBookMessages,
    SequentialStart,
    Side,
    Time,
    read_lobster,
)
from sekai.finance.lob import Fill
from tests.data.lob_pickles import CONTINUATIONS
from tests.data.lob_reference import (
    MULTI_CONFIGS,
    SINGLE_CONFIGS,
    digest,
    multi_rollout,
    single_rollout,
)

AGENT = 0
BID, ASK = int(Side.BID), int(Side.ASK)
SUBMIT, CANCEL, DELETE = MessageType.SUBMIT, MessageType.CANCEL, MessageType.DELETE
EXECUTE, HIDDEN = MessageType.EXECUTE, MessageType.EXECUTE_HIDDEN
CROSS, HALT = MessageType.CROSS, MessageType.HALT
LEVELS = 20  # snapshot depth; the simulated history never holds more than MAX_LEVELS per side
MAX_LEVELS = 15
DATA = Path(__file__).parent / "data"

Depth = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def messages_of(rows: list[tuple[float, int, int, int, int, int]]) -> OrderBookMessages:
    """Messages from `(time, event, order_id, size, price, side)` rows."""
    columns = list(zip(*rows)) if rows else [[]] * 6
    return OrderBookMessages(*(np.asarray(c) for c in columns))  # type: ignore[arg-type]


def snapshot_of(bids: list[tuple[int, int]], asks: list[tuple[int, int]]) -> BookSnapshot:
    """A snapshot from `(price, size)` levels, best first."""
    return BookSnapshot(
        np.array([p for p, _ in bids], dtype=np.int64),
        np.array([q for _, q in bids], dtype=np.int64),
        np.array([p for p, _ in asks], dtype=np.int64),
        np.array([q for _, q in asks], dtype=np.int64),
    )


def single_rollout_of(env: Any, options: dict[str, Any] | None, actions: np.ndarray) -> list[int]:
    """Digests of a reset of `env` (seed 0, with `options`) and of a step per action."""
    r = env.reset(seed=0, options=options)
    digests = [digest(r.observation, r.info)]
    for action in actions:
        s = env.step(action.astype(np.int64))
        digests.append(digest(s.observation, s.reward, s.terminated, s.truncated, s.info))
    return digests


def assert_depth(got: Depth, expected: Depth, context: Any = "") -> None:
    for name, a, b in zip(("bid prices", "bid sizes", "ask prices", "ask sizes"), got, expected):
        assert a.tolist() == b.tolist(), f"{name} differ {context}"


# ---------------------------------------------------------------------------
# A simulated history: a message stream with the book it implies, order by order
# ---------------------------------------------------------------------------


@dataclass
class History:
    messages: OrderBookMessages
    initial: BookSnapshot  # the book before the first message (every level)
    depth: list[Depth]  # the book after each message, LEVELS levels per side
    snapshots: BookSnapshots  # the same, as a LOBSTER orderbook file would hold it


class _TrueBook:
    """The historical book, order by order: the direct reconstruction the replay must match."""

    def __init__(self) -> None:
        self.orders: dict[int, list[int]] = {}  # id -> [side, price, size]
        self.queues: dict[tuple[int, int], list[int]] = defaultdict(list)

    def prices(self, side: int) -> list[int]:
        found = {p for (s, p), queue in self.queues.items() if s == side and queue}
        return sorted(found, reverse=side == BID)

    def best(self, side: int) -> int | None:
        prices = self.prices(side)
        return prices[0] if prices else None

    def volume(self, side: int) -> int:
        return sum(size for s, _, size in self.orders.values() if s == side)

    def add(self, order_id: int, side: int, price: int, size: int) -> None:
        self.orders[order_id] = [side, price, size]
        self.queues[side, price].append(order_id)

    def take(self, order_id: int, quantity: int) -> None:
        order = self.orders[order_id]
        order[2] -= quantity
        assert order[2] >= 0
        if order[2] == 0:
            self.queues[order[0], order[1]].remove(order_id)
            del self.orders[order_id]

    def depth(self, levels: int) -> Depth:
        out = []
        for side in (BID, ASK):
            prices = self.prices(side)[:levels]
            sizes = [sum(self.orders[i][2] for i in self.queues[side, p]) for p in prices]
            pad = levels - len(prices)
            out += [np.array(prices + [0] * pad, np.int64), np.array(sizes + [0] * pad, np.int64)]
        return out[0], out[1], out[2], out[3]


def simulate_history(seed: int, n: int = 1500) -> History:
    """A consistent price-time priority message stream, every message type included."""
    rng = np.random.default_rng(seed)
    book = _TrueBook()
    next_id = 1
    for distance in range(1, 6):  # the book before the first message
        for _ in range(2):
            book.add(next_id, BID, 1000 - distance, int(rng.integers(1, 6)))
            book.add(next_id + 1, ASK, 1000 + distance, int(rng.integers(1, 6)))
            next_id += 2
    initial = BookSnapshot(*book.depth(MAX_LEVELS + 1))
    rows: list[tuple[float, int, int, int, int, int]] = []
    depth: list[Depth] = []
    t = 34_200.0
    halts = [(n // 3, -1), (n // 3 + 6, 0), (n // 3 + 12, 1)]  # (after message, code)
    halted = False

    def emit(event: int, order_id: int, size: int, price: int, side: int) -> None:
        rows.append((t, event, order_id, size, price, side))
        depth.append(book.depth(LEVELS))

    while len(rows) < n:
        if rng.random() > 0.15:  # some messages share a timestamp
            t += float(rng.exponential(0.05))
        if halts and len(rows) >= halts[0][0]:
            code = halts.pop(0)[1]
            halted = code != 1
            emit(HALT, 0, 0, code, ASK)
            continue
        bid, ask = book.best(BID), book.best(ASK)
        u = rng.random()
        if bid is None or ask is None or u < 0.45:
            side = BID if bid is None else ASK if ask is None else int(rng.choice([BID, ASK]))
            reference = ask if side == BID else bid  # never crossing the opposite best
            if reference is None:  # both sides empty
                reference = 1000
            offset = int(rng.integers(1, 7))
            price = max(reference - offset if side == BID else reference + offset, 1)
            size = int(rng.integers(1, 6))
            book.add(next_id, side, price, size)
            emit(SUBMIT, next_id, size, price, side)
            next_id += 1
            prices = book.prices(side)
            if len(prices) > MAX_LEVELS:  # keep the depth within the snapshots
                deepest = book.queues[side, prices[-1]][0]
                remaining = book.orders[deepest][2]
                book.take(deepest, remaining)
                emit(DELETE, deepest, remaining, prices[-1], side)
        elif u < 0.55:
            candidates = [i for i, o in book.orders.items() if o[2] >= 2]
            if candidates:
                order_id = candidates[int(rng.integers(len(candidates)))]
                side, price, size = book.orders[order_id]
                cancelled = int(rng.integers(1, size))
                book.take(order_id, cancelled)
                emit(CANCEL, order_id, cancelled, price, side)
        elif u < 0.72:
            ids = list(book.orders)
            order_id = ids[int(rng.integers(len(ids)))]
            side, price, size = book.orders[order_id]
            book.take(order_id, size)
            emit(DELETE, order_id, size, price, side)
        elif u < 0.93 and not halted:  # an aggressor sweeps the front of one side
            side = int(rng.choice([BID, ASK]))
            wanted = min(int(rng.integers(1, 14)), book.volume(side) - 1)
            while wanted > 0:
                price = book.best(side)
                assert price is not None
                front = book.queues[side, price][0]
                traded = min(wanted, book.orders[front][2])
                book.take(front, traded)
                emit(EXECUTE, front, traded, price, side)
                wanted -= traded
        elif u < 0.98 and not halted:  # hidden liquidity inside the spread
            if ask - bid >= 2:
                side = int(rng.choice([BID, ASK]))
                emit(HIDDEN, 0, int(rng.integers(1, 5)), int(rng.integers(bid + 1, ask)), side)
        else:
            emit(CROSS, 0, int(rng.integers(1, 9)), bid, BID)
    return History(
        messages_of(rows),
        initial,
        depth,
        BookSnapshots(*(np.stack([d[k] for d in depth]) for k in range(4))),  # type: ignore[arg-type]
    )


@pytest.fixture(scope="module")
def history() -> History:
    return simulate_history(0)


# ---------------------------------------------------------------------------
# Replay without agents reproduces the historical book
# ---------------------------------------------------------------------------


class TestExactReplay:
    def test_history_covers_every_message_type(self, history: History) -> None:
        assert set(history.messages.event.tolist()) == {m.value for m in MessageType}

    def test_book_after_each_message(self, history: History) -> None:
        replay = MessageReplay(history.messages, snapshot=history.initial)
        book = OrderBook()
        replay.initialise(book)
        assert book.time == history.messages.time[0] and replay.position == 0
        initial = history.initial
        expected = (initial.bid_prices, initial.bid_sizes, initial.ask_prices, initial.ask_sizes)
        assert_depth(book.depth(MAX_LEVELS + 1), expected)
        m = history.messages
        for i in range(len(m)):
            fills = replay.step(book)
            assert book.time == m.time[i] and replay.position == i + 1
            assert_depth(book.depth(LEVELS), history.depth[i], f"after message {i}")
            bid_prices, _, ask_prices, _ = history.depth[i]
            assert book.best_bid() == (int(bid_prices[0]) or None)
            assert book.best_ask() == (int(ask_prices[0]) or None)
            if m.event[i] == EXECUTE:  # replayed as one trade with the executed order
                assert [(f.price, f.quantity) for f in fills] == [(m.price[i], m.size[i])]
                assert fills[0].maker_owner == fills[0].taker_owner == BACKGROUND
                assert book.last_price == m.price[i]
            else:  # nothing else trades in a book that follows history
                assert fills == []
            assert book.n_orders_of(BACKGROUND) == len(book)
        assert replay.done
        with pytest.raises(ValueError, match="every message"):
            replay.step(book)

    def test_advance_equals_stepping(self, history: History) -> None:
        stepped, advanced = OrderBook(), OrderBook()
        a = MessageReplay(history.messages, snapshot=history.initial)
        b = a.spawn()
        a.initialise(stepped)
        b.initialise(advanced)
        times = history.messages.time
        all_fills: list[Fill] = []
        for until in np.linspace(times[0], times[-1] + 1.0, 37):
            all_fills += b.advance(advanced, None, float(until))
            assert advanced.time == until
            # Every message stamped at or before `until` has been replayed, no later one.
            assert b.position == int(np.searchsorted(times, until, side="right"))
            while a.position < b.position:
                a.step(stepped)
            assert_depth(advanced.depth(LEVELS), stepped.depth(LEVELS), f"at {until}")
        assert sum(f.quantity for f in all_fills) == int(
            history.messages.size[history.messages.event == EXECUTE].sum()
        )

    @pytest.mark.parametrize("with_snapshots", [True, False], ids=["snapshots", "fast-forward"])
    def test_start_at_any_time(self, history: History, with_snapshots: bool) -> None:
        times = history.messages.time
        replay = MessageReplay(
            history.messages,
            snapshot=history.initial,
            snapshots=history.snapshots if with_snapshots else None,
        )
        rng = np.random.default_rng(1)
        starts = [float(times[0]), float(times[3]), float(times[-1])]
        starts += [float(t) for t in rng.uniform(times[0], times[-1], 6)]
        starts += [float(times[len(times) // 2])]  # a timestamp that several messages share
        for start in starts:
            book = OrderBook()
            replay.initialise(book, start=start)
            index = int(np.searchsorted(times, start, side="right"))
            assert replay.position == index and book.time == start
            assert_depth(book.depth(LEVELS), history.depth[index - 1], f"at start {start}")
            for i in range(index, min(index + 150, len(times))):
                replay.step(book)
                assert_depth(book.depth(LEVELS), history.depth[i], f"from {start}, message {i}")

    def test_start_before_the_first_message(self, history: History) -> None:
        replay = MessageReplay(history.messages, snapshot=history.initial)
        book = OrderBook()
        replay.initialise(book, start=history.messages.time[0] - 10.0)
        assert replay.position == 0 and book.time == history.messages.time[0] - 10.0
        assert book.n_levels(Side.BID) == 5 and book.n_levels(Side.ASK) == 5
        assert book.last_price == 1000  # the seeded book's mid

    def test_halted_state_follows_the_messages(self, history: History) -> None:
        times = history.messages.time
        halts = np.flatnonzero(history.messages.event == HALT)
        assert history.messages.price[halts].tolist() == [-1, 0, 1]
        replay = MessageReplay(history.messages, snapshots=history.snapshots)
        book = OrderBook()
        replay.initialise(book, start=float(np.nextafter(times[halts[0]], -np.inf)))
        assert not replay.halted  # just before the halt
        for halt, halted in zip(halts, (True, True, False)):  # starts on each halt message
            replay.initialise(book, start=float(times[halt]))
            assert replay.halted is halted
        replay.initialise(book, start=float(np.nextafter(times[halts[0]], -np.inf)))
        first, last = int(halts[0]), int(halts[2])
        while replay.position <= last:
            replay.step(book)
            assert replay.halted is (first <= replay.position - 1 < last)

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_other_histories(self, seed: int) -> None:
        history = simulate_history(seed, n=800)
        replay = MessageReplay(history.messages, snapshot=history.initial)
        book = OrderBook()
        replay.initialise(book)
        for i in range(len(history.messages)):
            replay.step(book)
            assert_depth(book.depth(LEVELS), history.depth[i], f"seed {seed}, message {i}")


# ---------------------------------------------------------------------------
# Replay with an agent in the book
# ---------------------------------------------------------------------------


def replay_of(
    rows: list[tuple[float, int, int, int, int, int]],
    bids: list[tuple[int, int]],
    asks: list[tuple[int, int]],
    **kwargs: Any,
) -> tuple[MessageReplay, OrderBook]:
    """A replay of `rows` seeded with a snapshot, initialised before its first message."""
    replay = MessageReplay(messages_of(rows), snapshot=snapshot_of(bids, asks), **kwargs)
    book = OrderBook()
    replay.initialise(book)
    return replay, book


class TestReplayWithAgent:
    def test_agent_ahead_in_the_queue_is_filled_first(self) -> None:
        rows = [
            (1.0, SUBMIT, 7, 4, 100, BID),  # X joins the bid at 100, behind the agent
            (2.0, EXECUTE, 7, 3, 100, BID),  # a seller takes 3 of X historically
            (3.0, CANCEL, 7, 1, 100, BID),  # X's owner cancels 1 of what is left
        ]
        for cap in (False, True):
            replay, book = replay_of(rows, [(99, 5)], [(102, 5)], cap_to_history=cap)
            agent_id, _ = book.limit(Side.BID, 100, 2, owner=AGENT)
            assert agent_id is not None
            replay.advance(book, None, 1.5)
            x = book.orders_of(BACKGROUND)[-1]
            assert (x.price, x.quantity) == (100, 4) and book.queue_ahead(x.order_id) == 2
            fills = replay.advance(book, None, 2.5)
            assert [(f.maker_owner, f.price, f.quantity) for f in fills] == [
                (AGENT, 100, 2),
                (BACKGROUND, 100, 1),
            ]
            assert all(f.taker_owner == BACKGROUND and f.taker_side is Side.ASK for f in fills)
            assert book.get(agent_id) is None
            # X kept the 2 lots the agent absorbed (historically it had 1 left); capped to
            # history, it is cut to that 1.
            assert book.volume_at(Side.BID, 100) == (1 if cap else 3)
            replay.advance(book, None, 3.5)
            assert book.volume_at(Side.BID, 100) == (0 if cap else 2)

    def test_better_priced_agent_order_is_filled_first(self) -> None:
        rows = [(1.0, EXECUTE, 555, 3, 100, BID)]  # an order resting before the snapshot
        replay, book = replay_of(rows, [(100, 5)], [(103, 5)])
        agent_id, _ = book.limit(Side.BID, 101, 1, owner=AGENT)
        fills = replay.advance(book, None, 2.0)
        assert [(f.maker_owner, f.price, f.quantity) for f in fills] == [
            (AGENT, 101, 1),
            (BACKGROUND, 100, 2),
        ]
        assert book.volume_at(Side.BID, 100) == 3 and book.get(agent_id) is None

    def test_agent_behind_the_queue_is_not_filled(self) -> None:
        rows = [(1.0, EXECUTE, 555, 5, 100, BID)]
        replay, book = replay_of(rows, [(100, 5)], [(103, 5)])
        agent_id, _ = book.limit(Side.BID, 100, 1, owner=AGENT)
        assert agent_id is not None and book.queue_ahead(agent_id) == 5  # the seeded level
        fills = replay.advance(book, None, 2.0)
        assert all(f.maker_owner == BACKGROUND for f in fills)
        assert book.queue_ahead(agent_id) == 0 and book.volume_at(Side.BID, 100) == 1

    def test_messages_about_consumed_orders(self) -> None:
        rows = [
            (1.0, SUBMIT, 7, 2, 101, ASK),  # X behind the seeded 5 at 101
            (2.0, CANCEL, 7, 1, 101, ASK),  # X is gone (the agent bought it): no-op
            (2.5, DELETE, 555, 5, 101, ASK),  # the seeded order is gone too: no-op
            (3.0, SUBMIT, 8, 3, 101, ASK),  # Y at 101
            (4.0, EXECUTE, 7, 1, 101, ASK),  # X historically executes: Y trades instead
            (5.0, DELETE, 7, 1, 101, ASK),  # X again: no-op
            (6.0, EXECUTE, 9, 4, 101, ASK),  # more than 101 holds: never trades through
        ]
        replay, book = replay_of(rows, [(99, 5)], [(101, 5), (102, 5)])
        replay.advance(book, None, 1.5)
        _, fills = book.market(Side.BID, 7, owner=AGENT)  # the seeded 5 and X's 2
        assert sum(f.quantity for f in fills) == 7 and book.best_ask() == 102
        assert replay.advance(book, None, 3.5) == []
        assert book.volume_at(Side.ASK, 101) == 3 and book.volume_at(Side.ASK, 102) == 5
        fills = replay.advance(book, None, 5.5)
        assert [(f.price, f.quantity, f.taker_side) for f in fills] == [(101, 1, Side.BID)]
        assert book.volume_at(Side.ASK, 101) == 2  # Y walked into, X's delete changed nothing
        fills = replay.advance(book, None, 7.0)
        assert [(f.price, f.quantity) for f in fills] == [(101, 2)]
        assert book.best_ask() == 102 and book.volume_at(Side.ASK, 102) == 5

    def test_submission_crosses_an_agent_quote(self) -> None:
        rows = [(1.0, SUBMIT, 7, 3, 101, BID)]  # historically below the best ask of 102
        replay, book = replay_of(rows, [(99, 5)], [(102, 5)])
        agent_id, _ = book.limit(Side.ASK, 101, 2, owner=AGENT)
        fills = replay.advance(book, None, 2.0)
        assert [(f.maker_owner, f.taker_owner, f.quantity) for f in fills] == [
            (AGENT, BACKGROUND, 2)
        ]
        assert book.get(agent_id) is None and book.volume_at(Side.BID, 101) == 1

    def test_hidden_executions_fill_only_orders_at_least_as_good(self) -> None:
        rows = [
            (1.0, HIDDEN, 0, 2, 102, BID),  # a hidden bid at 102 traded: 101 is worse
            (2.0, HIDDEN, 0, 2, 101, BID),  # a hidden bid at 101: the visible 101 goes first
            (3.0, HIDDEN, 0, 1, 103, ASK),  # a hidden ask at 103: the agent's 104 is worse
        ]
        replay, book = replay_of(rows, [(100, 5)], [(105, 5)])
        bid_id, _ = book.limit(Side.BID, 101, 1, owner=AGENT)
        ask_id, _ = book.limit(Side.ASK, 104, 1, owner=AGENT)
        assert replay.advance(book, None, 1.5) == []
        fills = replay.advance(book, None, 2.5)
        assert [(f.maker_order_id, f.price, f.quantity) for f in fills] == [(bid_id, 101, 1)]
        assert book.best_bid() == 100 and book.volume_at(Side.BID, 100) == 5  # never deeper
        assert replay.advance(book, None, 3.5) == [] and book.get(ask_id) is not None

    def test_unknown_order_beyond_the_snapshot_fills_only_better_prices(self) -> None:
        rows = [
            (1.0, SUBMIT, 7, 3, 98, BID),  # a new order at 98, where the snapshot had none
            (2.0, EXECUTE, 555, 2, 98, BID),  # an order resting at 98 before the snapshot
            (3.0, CANCEL, 556, 1, 98, BID),  # another unknown one: nothing to reduce
        ]
        replay, book = replay_of(rows, [(100, 1)], [(103, 5)])
        book.market(Side.ASK, 1, owner=AGENT)  # the agent empties the bid at 100
        agent_id, _ = book.limit(Side.BID, 99, 1, owner=AGENT)
        fills = replay.advance(book, None, 2.5)
        # The unknown order was ahead of everything at 98: only the agent's 99 is better.
        assert [(f.maker_order_id, f.quantity) for f in fills] == [(agent_id, 1)]
        assert book.volume_at(Side.BID, 98) == 3
        replay.advance(book, None, 3.5)
        assert book.volume_at(Side.BID, 98) == 3

    def test_snapshot_seeding(self) -> None:
        snapshot = snapshot_of([(100, 4), (99, 2), (97, 7)], [(101, 3), (104, 1)])
        rows = [
            (1.0, CANCEL, 900, 3, 97, BID),  # an order from before the snapshot, at 97
            (2.0, DELETE, 901, 1, 104, ASK),
            (3.0, EXECUTE, 902, 2, 100, BID),
        ]
        replay = MessageReplay(messages_of(rows), snapshot=snapshot)
        book = OrderBook()
        book.limit(Side.BID, 50, 1, owner=AGENT)  # initialise clears the book
        replay.initialise(book)
        assert_depth(
            book.depth(3),
            (
                np.array([100, 99, 97]),
                np.array([4, 2, 7]),
                np.array([101, 104, 0]),
                np.array([3, 1, 0]),
            ),
        )
        assert len(book) == book.n_orders_of(BACKGROUND) == 5  # one order per level
        assert book.last_price == 100 and book.orders_of(AGENT) == []
        replay.advance(book, None, 10.0)
        assert book.volume_at(Side.BID, 97) == 4 and book.volume_at(Side.ASK, 104) == 0
        assert book.volume_at(Side.BID, 100) == 2 and book.last_price == 100

    def test_cap_to_history_trims_seeded_levels(self) -> None:
        rows = [(1.0, EXECUTE, 555, 3, 100, BID), (2.0, CANCEL, 556, 1, 100, BID)]
        for cap, expected in ((False, [3, 2]), (True, [1, 0])):
            replay, book = replay_of(rows, [(100, 4)], [(103, 5)], cap_to_history=cap)
            book.limit(Side.BID, 101, 2, owner=AGENT)  # absorbs 2 of the 3 executed lots
            volumes = []
            for until in (1.5, 2.5):
                replay.advance(book, None, until)
                volumes.append(book.volume_at(Side.BID, 100))
            assert volumes == expected

    def test_replay_binds_one_book(self) -> None:
        replay, book = replay_of([(1.0, SUBMIT, 1, 1, 100, BID)], [], [])
        with pytest.raises(ValueError, match="another book"):
            replay.advance(OrderBook(), None, 2.0)
        with pytest.raises(ValueError, match="initialise"):
            replay.spawn().advance(book, None, 2.0)
        with pytest.raises(ValueError, match="earlier"):
            replay.advance(book, None, 0.5)
        with pytest.raises(ValueError, match="finite"):
            replay.advance(book, None, math.nan)
        with pytest.raises(ValueError, match="finite"):
            replay.initialise(book, start=math.inf)
        twin = replay.spawn()
        other = OrderBook()
        twin.initialise(other)
        twin.advance(other, None, 5.0)
        assert twin.done and not replay.done and len(other) == 1 and len(book) == 0

    def test_pickle_round_trip(self, history: History) -> None:
        replay = MessageReplay(history.messages, snapshots=history.snapshots)
        book = OrderBook()
        replay.initialise(book, start=float(history.messages.time[100]))
        replay.advance(book, None, float(history.messages.time[300]))
        copy_replay, copy_book = pickle.loads(pickle.dumps((replay, book)))
        until = float(history.messages.time[700])
        assert copy_replay.advance(copy_book, None, until) == replay.advance(book, None, until)
        assert_depth(copy_book.depth(LEVELS), book.depth(LEVELS))

    def test_repr(self, history: History) -> None:
        replay = MessageReplay(history.messages)
        assert repr(replay).startswith(f"MessageReplay({len(history.messages)} messages")
        n = len(history.messages)
        assert repr(history.messages).startswith(f"OrderBookMessages({n} messages, time=34200")
        assert repr(history.snapshots) == f"BookSnapshots({n} rows, levels={LEVELS})"
        assert "best_bid=999" in repr(history.initial)


# ---------------------------------------------------------------------------
# Data classes: validation
# ---------------------------------------------------------------------------


class TestValidation:
    GOOD = [
        (1.0, SUBMIT, 1, 2, 100, BID),
        (1.0, HALT, 0, 0, -1, ASK),
        (2.0, DELETE, 1, 2, 100, BID),
    ]

    def test_messages_are_read_only_copies(self) -> None:
        time = np.array([1.0, 2.0])
        messages = OrderBookMessages(
            time,
            np.array([1, 3]),
            np.array([5, 5]),
            np.array([1, 1]),
            np.array([9, 9]),
            np.array([1, 1]),
        )
        time[0] = 99.0
        assert messages.time[0] == 1.0 and len(messages) == 2
        assert messages.event.dtype == np.int8 and messages.side.dtype == np.int8
        assert messages.price.dtype == np.int64 and messages.time.dtype == np.float64
        with pytest.raises(ValueError):
            messages.time[0] = 3.0
        assert messages_of(self.GOOD).event.tolist() == [1, 7, 3]

    @pytest.mark.parametrize(
        "row, match",
        [
            ((0.5, SUBMIT, 2, 1, 100, BID), "earlier"),
            ((math.nan, SUBMIT, 2, 1, 100, BID), "finite"),
            ((3.0, 8, 2, 1, 100, BID), "MessageType"),
            ((3.0, 0, 2, 1, 100, BID), "MessageType"),
            ((3.0, SUBMIT, 2, 0, 100, BID), "size"),
            ((3.0, EXECUTE, 2, -1, 100, BID), "size"),
            ((3.0, SUBMIT, 2, 1, 0, BID), "price"),
            ((3.0, HALT, 0, 0, 2, BID), "halt code"),
            ((3.0, SUBMIT, 2, 1, 100, 0), "side"),
            ((3.0, SUBMIT, 1, 1, 100, BID), "submitted twice"),
        ],
    )
    def test_bad_messages(self, row: tuple[Any, ...], match: str) -> None:
        with pytest.raises(ValueError, match=match) as error:
            messages_of([*self.GOOD, row])
        assert "message 3" in str(error.value)

    def test_bad_arrays(self) -> None:
        good = [
            np.array([1.0]),
            np.array([1]),
            np.array([1]),
            np.array([1]),
            np.array([9]),
            np.array([1]),
        ]
        for k, bad in [
            (0, np.array([[1.0]])),
            (1, np.array([1, 1])),
            (3, np.array([1.5])),
            (4, np.array([True])),
            (2, np.array(["a"])),
        ]:
            arrays = list(good)
            arrays[k] = bad
            with pytest.raises(ValueError):
                OrderBookMessages(*arrays)  # type: ignore[arg-type]
        assert len(OrderBookMessages(*(a[:0] for a in good))) == 0  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bids, asks, match",
        [
            ([(100, 1), (100, 1)], [], "strictly decreasing"),
            ([(99, 1), (100, 1)], [], "strictly decreasing"),
            ([], [(101, 1), (101, 2)], "strictly increasing"),
            ([(100, 1)], [(100, 1)], "below the best ask"),
            ([(0, 1)], [], ">= 1 tick"),
            ([(100, -1)], [], ">= 0"),
            ([(100, 0), (99, 1)], [], "empty bid level comes before"),
        ],
    )
    def test_bad_snapshots(self, bids: Any, asks: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            snapshot_of(bids, asks)

    def test_snapshots(self) -> None:
        snapshot = snapshot_of([(100, 1), (7, 0)], [])
        assert snapshot.bid_prices.tolist() == [100, 0]  # an empty level's price is 0
        assert snapshot.price_levels(Side.BID) == [(100, 1)]
        assert snapshot.price_levels(Side.ASK) == []
        assert snapshot_of([], []).price_levels(Side.BID) == []
        book = OrderBook()
        book.limit(Side.BID, 50, 3)
        book.limit(Side.ASK, 52, 1)
        captured = BookSnapshot(*book.depth(3))
        assert captured.price_levels(Side.BID) == [(50, 3)]
        assert captured.price_levels(Side.ASK) == [(52, 1)]
        rows = BookSnapshots(
            np.array([[100, 99], [100, 0]]),
            np.array([[1, 2], [3, 0]]),
            np.array([[101, 0], [102, 103]]),
            np.array([[4, 0], [5, 6]]),
        )
        assert len(rows) == 2 and rows.levels == 2
        assert rows[1].price_levels(Side.ASK) == [(102, 5), (103, 6)]
        assert rows[-1].price_levels(Side.BID) == [(100, 3)]
        assert rows.crossed.tolist() == [False, False]
        # A recorded history may be locked or crossed (around halts and auctions), but such a
        # row is not the state of a book: it cannot be a BookSnapshot.
        locked = BookSnapshots(
            np.array([[100], [101], [102], [100]]),
            np.array([[1], [1], [1], [0]]),
            np.array([[101], [101], [101], [100]]),
            np.array([[1], [1], [1], [1]]),
        )
        assert locked.crossed.tolist() == [False, True, True, False]
        with pytest.raises(ValueError, match="row 1 is crossed or locked"):
            _ = locked[1]
        with pytest.raises(ValueError, match="best bid must be below the best ask"):
            snapshot_of([(101, 1)], [(101, 1)])
        empty = BookSnapshots(*(np.zeros((2, 0), np.int64) for _ in range(4)))
        assert empty.crossed.tolist() == [False, False] and empty[0].price_levels(Side.BID) == []
        with pytest.raises(ValueError, match="shape"):
            BookSnapshots(np.array([100]), np.array([1]), np.array([101]), np.array([1]))
        with pytest.raises(ValueError, match="rows"):
            BookSnapshots(
                np.array([[100]]), np.array([[1]]), np.array([[101], [102]]), np.array([[1], [1]])
            )

    def test_replay_arguments(self, history: History) -> None:
        with pytest.raises(TypeError, match="OrderBookMessages"):
            MessageReplay(history.depth)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="BookSnapshot"):
            MessageReplay(history.messages, snapshot=history.snapshots)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="BookSnapshots"):
            MessageReplay(history.messages, snapshots=history.initial)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="empty"):
            MessageReplay(messages_of([]))
        short = BookSnapshots(
            *(
                a[:10]
                for a in (
                    history.snapshots.bid_prices,
                    history.snapshots.bid_sizes,
                    history.snapshots.ask_prices,
                    history.snapshots.ask_sizes,
                )
            )
        )
        with pytest.raises(ValueError, match="after each message"):
            MessageReplay(history.messages, snapshots=short)


# ---------------------------------------------------------------------------
# LOBSTER files
# ---------------------------------------------------------------------------

TICK = 0.01
UNITS = 100  # LOBSTER price units (1/10,000 dollar) per tick of 0.01


def write_lobster(directory: Path, history: History, levels: int = LEVELS) -> tuple[Path, Path]:
    """Write `history` as a LOBSTER message file and orderbook file."""
    m = history.messages
    message_lines = []
    for t, event, order_id, size, price, side in zip(
        m.time.tolist(),
        m.event.tolist(),
        m.order_id.tolist(),
        m.size.tolist(),
        m.price.tolist(),
        m.side.tolist(),
    ):
        raw = price if event == HALT else price * UNITS
        message_lines.append(f"{t:.9f},{event},{order_id},{size},{raw},{side}")
    book_lines = []
    for bid_prices, bid_sizes, ask_prices, ask_sizes in history.depth:
        cells = []
        for k in range(levels):
            ask = ask_prices[k] * UNITS if ask_sizes[k] else 9_999_999_999
            bid = bid_prices[k] * UNITS if bid_sizes[k] else -9_999_999_999
            cells += [ask, ask_sizes[k], bid, bid_sizes[k]]
        book_lines.append(",".join(str(int(c)) for c in cells))
    message_path = directory / "TEST_2026-01-02_34200000_57600000_message_20.csv"
    orderbook_path = directory / "TEST_2026-01-02_34200000_57600000_orderbook_20.csv"
    message_path.write_text("\n".join(message_lines) + "\n")
    orderbook_path.write_text("\n".join(book_lines) + "\n")
    return message_path, orderbook_path


class TestLobster:
    def test_round_trip(self, tmp_path: Path, history: History) -> None:
        message_path, orderbook_path = write_lobster(tmp_path, history)
        messages, snapshots = read_lobster(message_path, orderbook_path, tick_size=TICK)
        assert snapshots is not None and len(snapshots) == len(messages) == len(history.messages)
        for name in ("event", "order_id", "size", "price", "side"):
            assert getattr(messages, name).tolist() == getattr(history.messages, name).tolist()
        np.testing.assert_allclose(messages.time, history.messages.time, rtol=0, atol=1e-9)
        for i in (0, 1, 500, len(messages) - 1):
            row = snapshots[i]
            assert_depth(
                (row.bid_prices, row.bid_sizes, row.ask_prices, row.ask_sizes), history.depth[i]
            )
        only_messages, nothing = read_lobster(message_path, tick_size=TICK)
        assert nothing is None and only_messages.price.tolist() == messages.price.tolist()
        _, few = read_lobster(message_path, orderbook_path, levels=3)
        assert few is not None and few.levels == 3
        assert few.bid_sizes.tolist() == snapshots.bid_sizes[:, :3].tolist()

    def test_replay_matches_the_orderbook_file(self, tmp_path: Path, history: History) -> None:
        messages, snapshots = read_lobster(*write_lobster(tmp_path, history))
        assert snapshots is not None
        replay = MessageReplay(messages, snapshots=snapshots)
        book = OrderBook()
        start = float(messages.time[200])
        replay.initialise(book, start=start)
        for i in range(replay.position, len(messages)):
            replay.step(book)
            row = snapshots[i]
            got = book.depth(LEVELS)
            assert_depth(got, (row.bid_prices, row.bid_sizes, row.ask_prices, row.ask_sizes), i)

    def test_every_event_type_and_dummy_prices(self, tmp_path: Path) -> None:
        messages_csv = "\n".join(
            [
                "34200.000000001,1,11,100,1000000,1",  # a bid at $100.00
                "34200.000000001,1,12,50,1000100,-1",  # an ask at $100.01
                "34200.5,2,11,40,1000000,1",
                "34201,4,12,10,1000100,-1",
                "34201.25,5,0,7,1000050,1",  # hidden, at the midpoint: rounded up on the bid side
                "34201.25,5,0,7,1000050,-1",  # rounded down on the ask side
                "34202,6,0,300,1000049,1",  # a cross trade: to the nearest tick
                "34203,7,0,0,-1,-1",  # trading halts
                "34204,7,0,0,0,-1",  # quoting resumes
                "34205,7,0,0,1,-1",  # trading resumes
                "34206,3,12,40,1000100,-1",
            ]
        )
        empty_ask, empty_bid = 9_999_999_999, -9_999_999_999
        rows = [
            [empty_ask, 0, 1000000, 100, empty_ask, 0, empty_bid, 0],
            [1000100, 50, 1000000, 100, empty_ask, 0, empty_bid, 0],
            [1000100, 50, 1000000, 60, empty_ask, 0, empty_bid, 0],
            [1000100, 40, 1000000, 60, empty_ask, 0, empty_bid, 0],
            *[[1000100, 40, 1000000, 60, empty_ask, 0, empty_bid, 0]] * 6,
            [empty_ask, 0, 1000000, 60, empty_ask, 0, empty_bid, 0],
        ]
        (tmp_path / "m.csv").write_text(messages_csv + "\n")
        (tmp_path / "o.csv").write_text("\n".join(",".join(map(str, r)) for r in rows) + "\n")
        messages, snapshots = read_lobster(tmp_path / "m.csv", str(tmp_path / "o.csv"))
        assert messages.event.tolist() == [1, 1, 2, 4, 5, 5, 6, 7, 7, 7, 3]
        # Halt codes stay as they are; hidden prices round away from the book's interior.
        prices = [10000, 10001, 10000, 10001, 10001, 10000, 10000, -1, 0, 1, 10001]
        assert messages.price.tolist() == prices
        assert messages.side.tolist() == [1, -1, 1, -1, 1, -1, 1, -1, -1, -1, -1]
        assert messages.time[0] == pytest.approx(34200.000000001, abs=1e-9)
        assert snapshots is not None and snapshots.levels == 2
        assert snapshots.ask_prices[0].tolist() == [0, 0]  # dummy prices: empty levels
        assert snapshots.ask_sizes[0].tolist() == [0, 0]
        assert snapshots.bid_prices[0].tolist() == [10000, 0]
        assert snapshots[0].price_levels(Side.ASK) == []
        assert snapshots[1].price_levels(Side.ASK) == [(10001, 50)]
        replay = MessageReplay(messages, snapshots=snapshots)
        book = OrderBook()
        replay.initialise(book, start=34200.0)  # before the first message: an empty book
        for i in range(len(messages)):
            replay.step(book)
            row = snapshots[i]
            expected = (row.bid_prices, row.bid_sizes, row.ask_prices, row.ask_sizes)
            assert_depth(book.depth(2), expected, f"after message {i}")
            assert replay.halted == (i in (7, 8))

    @pytest.mark.parametrize(
        "content, match",
        [
            ("34200,1,1,10,1000000\n", "6 columns"),
            ("34200,1,1,10,1000050,1\n", "not a multiple of tick_size"),
            ("34200,9,1,10,1000000,1\n", "event type 9"),
            ("34200,1,1,10,1000000,0\n", "direction 0"),
            ("34200,1,1.5,10,1000000,1\n", "order id"),
            ("34201,1,1,10,1000000,1\n34200,1,2,10,1000000,1\n", "earlier"),
            ("34200,1,1,0,1000000,1\n", "size"),
            ("", "no rows"),
        ],
    )
    def test_bad_message_files(self, tmp_path: Path, content: str, match: str) -> None:
        path = tmp_path / "messages.csv"
        path.write_text(content)
        with pytest.raises(ValueError, match=match):
            read_lobster(path)

    def test_bad_orderbook_files(self, tmp_path: Path) -> None:
        messages = tmp_path / "m.csv"
        messages.write_text("34200,1,1,10,1000000,1\n34201,1,2,10,1000100,-1\n")
        book = tmp_path / "o.csv"
        cases = [
            ("9999999999,0,1000000,10\n", "rows"),
            ("9999999999,0,1000000,10,5\n1000100,10,1000000,10,5\n", "4 columns per level"),
            ("9999999999,0,1000050,10\n1000100,10,1000000,10\n", "not a multiple"),
        ]
        for content, match in cases:
            book.write_text(content)
            with pytest.raises(ValueError, match=match):
                read_lobster(messages, book)
        book.write_text("9999999999,0,1000000,10\n1000100,10,1000000,10\n")
        with pytest.raises(ValueError, match="cannot read 2"):
            read_lobster(messages, book, levels=2)
        for tick in (0.0, -0.01, 0.00005, math.nan):
            with pytest.raises(ValueError, match="tick_size"):
                read_lobster(messages, book, tick_size=tick)
        with pytest.raises(ValueError, match="levels"):
            read_lobster(messages, book, levels=0)
        messages_1, snapshots_1 = read_lobster(messages, book, tick_size=0.0001)
        assert messages_1.price.tolist() == [1000000, 1000100] and snapshots_1 is not None

    def test_crossed_and_locked_rows(self, tmp_path: Path) -> None:
        # A halt during which a bid crosses the ask, as LOBSTER records it: the book stays
        # crossed until the reopening auction (a cross trade) clears it.
        messages_csv = "\n".join(
            [
                "1.0,1,1,5,10000,1",
                "2.0,1,2,5,10200,-1",
                "3.0,7,0,0,-1,-1",  # trading halts
                "4.0,1,3,2,10300,1",  # a bid above the ask: crossed
                "5.0,1,4,1,10100,-1",
                "6.0,7,0,0,1,-1",  # trading resumes
                "6.0,6,0,2,10200,-1",  # the reopening cross
                "6.0,3,3,2,10300,1",
                "6.0,2,2,2,10200,-1",
            ]
        )
        no_ask, no_bid = "9999999999,0", "-9999999999,0"
        book_csv = "\n".join(
            [
                f"{no_ask},10000,5,{no_ask},{no_bid}",
                f"10200,5,10000,5,{no_ask},{no_bid}",
                f"10200,5,10000,5,{no_ask},{no_bid}",
                f"10200,5,10300,2,{no_ask},10000,5",
                "10100,1,10300,2,10200,5,10000,5",
                "10100,1,10300,2,10200,5,10000,5",
                "10100,1,10300,2,10200,5,10000,5",
                f"10100,1,10000,5,10200,5,{no_bid}",
                f"10100,1,10000,5,10200,3,{no_bid}",
            ]
        )
        (tmp_path / "m.csv").write_text(messages_csv + "\n")
        (tmp_path / "o.csv").write_text(book_csv + "\n")
        messages, snapshots = read_lobster(tmp_path / "m.csv", tmp_path / "o.csv")
        assert snapshots is not None
        assert snapshots.crossed.tolist() == [False] * 3 + [True] * 4 + [False] * 2
        with pytest.raises(ValueError, match="row 4 is crossed"):
            _ = snapshots[4]
        # Starting while the recorded book is crossed, the replay is seeded from the last
        # uncrossed row (2) and replays the messages since: the crossing bid trades at once.
        # That is the book a replay of the whole history gives.
        expected: Depth = (
            np.array([100, 0]),
            np.array([5, 0]),
            np.array([101, 102]),
            np.array([1, 3]),
        )
        for replay in (
            MessageReplay(messages, snapshots=snapshots),
            MessageReplay(messages),
        ):
            book = OrderBook()
            replay.initialise(book, start=5.5)
            assert_depth(book.depth(2), expected)
            assert book.time == 5.5 and replay.position == 5 and replay.halted
        # After the auction the recorded book is uncrossed again, and seeds the book directly.
        replay = MessageReplay(messages, snapshots=snapshots)
        book = OrderBook()
        replay.initialise(book, start=7.0)
        assert_depth(book.depth(2), expected)
        assert not replay.halted and len(book) == 3  # one order per seeded level
        # With no uncrossed row before the start, the replay starts from `snapshot`.
        always = BookSnapshots(
            *(np.tile(a, (len(messages), 1)) for a in ([103, 0], [2, 0], [102, 0], [5, 0]))
        )
        replay = MessageReplay(messages, snapshots=always)
        book = OrderBook()
        replay.initialise(book, start=5.5)
        assert_depth(book.depth(2), expected)


# ---------------------------------------------------------------------------
# The environments with a replay flow
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def replay(history: History) -> MessageReplay:
    return MessageReplay(history.messages, snapshots=history.snapshots)


class TestReplayEnv:
    def test_check_env(self, replay: MessageReplay) -> None:
        env = LimitOrderBookEnv(flow=replay, session_length=20.0, warmup=2.0)
        assert sekai.check_env(env, check_determinism=True) == []
        jittery = LimitOrderBookEnv(
            flow=replay, session_length=20.0, latency=lambda rng: float(rng.uniform(0.0, 0.2))
        )
        assert sekai.check_env(jittery, check_determinism=True) == []

    def test_episode_is_a_window_of_the_history(
        self, replay: MessageReplay, history: History
    ) -> None:
        env = LimitOrderBookEnv(flow=replay, session_length=10.0, warmup=3.0, sampler=FixedStart(4))
        assert env.core.flow is not replay  # the env replays its own copy
        r = env.reset(seed=0)
        first = replay.start_time + 4 * 1.0 + 3.0
        assert r.info["time"] == pytest.approx(first) and r.info["timestamp"] == 0.0
        index = int(np.searchsorted(history.messages.time, first, side="right"))
        assert_depth(env.book.depth(LEVELS), history.depth[index - 1])
        times = [r.info["time"]]
        while not r.done:
            r = env.step(np.array([0, 0, 1]))  # no quotes: the book is the history
            times.append(r.info["time"])
            index = int(np.searchsorted(history.messages.time, r.info["time"], side="right"))
            assert_depth(env.book.depth(LEVELS), history.depth[index - 1])
        assert times == pytest.approx(first + np.arange(11)) and env.elapsed_steps == 10
        assert replay.position == 0  # the template is untouched

    def test_start_option(self, replay: MessageReplay) -> None:
        env = LimitOrderBookEnv(flow=replay, session_length=10.0, warmup=2.0)
        begin, end = replay.start_time, replay.end_time
        assert env.reset(options={"start": 5}).info["time"] == pytest.approx(begin + 5 + 2.0)
        t = begin + 7.25
        assert env.reset(options={"start": t}).info["time"] == pytest.approx(t)
        assert env.reset(options={"start": Time(t)}).info["time"] == pytest.approx(t)
        r = env.reset(options={"start": end - 10.0})
        assert r.info["time"] == pytest.approx(end - 10.0)
        for bad in (begin + 1.0, end - 9.0, -1, env.core.n_starts):
            with pytest.raises(ValueError, match="outside"):
                env.reset(options={"start": bad})
        with pytest.raises(TypeError):
            env.reset(options={"start": True})
        assert env.core.n_starts == math.floor(end - begin - 12.0 + 1e-9) + 1

    @pytest.mark.parametrize("start", [3.0, 5, Time(2.0), None])
    def test_simulated_flow_ignores_start(self, start: Any) -> None:
        # As in 0.2.0: simulated flow has no history to start in, so the option changes
        # nothing (and warns about nothing).
        actions = np.random.default_rng(0).integers(0, [6, 6, 3], size=(20, 3))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            env, plain = sekai.make("LimitOrderBook-v0"), sekai.make("LimitOrderBook-v0")
            assert single_rollout_of(env, {"start": start}, actions) == single_rollout_of(
                plain, None, actions
            )
            market, other = LimitOrderBookMarket(2), LimitOrderBookMarket(2)
            r, s = market.reset(seed=4, options={"start": start}), other.reset(seed=4)
            assert digest(r.observations, r.info) == digest(s.observations, s.info)
            step = {a: np.array([1, 1, 1]) for a in r.observations}
            r, s = market.step(step), other.step(step)
            assert digest(r.observations, r.rewards, r.info) == digest(
                s.observations, s.rewards, s.info
            )

    def test_samplers(self, replay: MessageReplay) -> None:
        env = LimitOrderBookEnv(flow=replay, session_length=10.0, warmup=0.0, sampler="sequential")
        starts = [env.reset().info["time"] - replay.start_time for _ in range(3)]
        assert starts == pytest.approx([0.0, 10.0, 20.0])  # episodes tile the history
        env = LimitOrderBookEnv(
            flow=replay, session_length=10.0, warmup=0.0, sampler=SequentialStart(stride=3)
        )
        assert [env.reset().info["time"] - replay.start_time for _ in range(2)] == pytest.approx(
            [0.0, 3.0]
        )
        env = LimitOrderBookEnv(flow=replay, session_length=10.0, decision_interval=0.5)
        times = {
            round(env.reset(seed=s).info["time"] - replay.start_time - 20.0, 6) for s in range(20)
        }
        assert len(times) > 10 and all(float(2 * t).is_integer() for t in times)

    def test_too_short_history(self, replay: MessageReplay) -> None:
        span = replay.end_time - replay.start_time
        with pytest.raises(ValueError, match="span"):
            LimitOrderBookEnv(flow=replay, session_length=span, warmup=1.0)
        env = LimitOrderBookEnv(flow=replay, session_length=span - 1.0, warmup=1.0)
        assert env.core.n_starts == 1
        r = env.reset(seed=3)
        assert r.info["time"] == pytest.approx(replay.start_time + 1.0)

    def test_agent_trades_against_history(self, replay: MessageReplay) -> None:
        env = LimitOrderBookEnv(flow=replay, session_length=30.0, warmup=5.0, inventory_penalty=0.0)
        fills = 0
        for seed in range(3):
            r = env.reset(seed=seed)
            total = 0.0
            while not r.done:
                r = env.step(np.array([1, 1, 1]))  # quote at the touch
                total += r.reward
                fills += r.info["fills"]
            assert total == pytest.approx(r.info["wealth"])
        assert fills > 10

    def test_halts_block_agent_orders(self) -> None:
        rows = [
            (0.0, SUBMIT, 1, 5, 99, BID),
            (0.5, HALT, 0, 0, -1, ASK),
            (2.5, HALT, 0, 0, 1, ASK),
            (10.0, SUBMIT, 2, 5, 90, BID),
        ]
        replay = MessageReplay(messages_of(rows), snapshot=snapshot_of([(100, 5)], [(101, 5)]))
        env = LimitOrderBookEnv(flow=replay, session_length=5.0, warmup=0.0)
        env.reset(options={"start": 0.0})
        r = env.step(np.array([1, 1, 2]))  # at 0.0: trading is open
        assert r.info["inventory"] == 1 and env.core.halted
        r = env.step(np.array([1, 1, 2]))  # at 1.0: halted, nothing is sent
        assert r.info["inventory"] == 1 and r.info["fills"] == 0
        r = env.step(np.array([0, 0, 2]))  # at 2.0: still halted
        assert r.info["inventory"] == 1 and not env.core.halted
        r = env.step(np.array([0, 0, 2]))  # at 3.0: trading again
        assert r.info["inventory"] == 2

    def test_multi_agent_market(self, replay: MessageReplay, history: History) -> None:
        market = LimitOrderBookMarket(
            2,
            flow=replay,
            decision_intervals=[1.0, 2.0],
            session_length=30.0,
            warmup=5.0,
            sampler=FixedStart(3),
        )
        r = market.reset(seed=0)
        first = replay.start_time + 3 + 5.0
        assert r.info["trader_0"]["time"] == pytest.approx(first)
        index = int(np.searchsorted(history.messages.time, first, side="right"))
        assert_depth(market.book.depth(LEVELS), history.depth[index - 1])
        fills = 0
        while not r.all_done:
            r = market.step({a: np.array([1, 1, 1]) for a in r.observations})
            fills += sum(info["fills"] for info in r.info.values())
        assert fills > 0 and r.info["trader_1"]["time"] == pytest.approx(first + 30.0)
        r = market.reset(options={"start": Time(first + 5.0)})
        assert r.info["trader_0"]["time"] == pytest.approx(first + 5.0)


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


def latency_env(
    rows: list[tuple[float, int, int, int, int, int]],
    latency: Any,
    snapshot: BookSnapshot | None = None,
    **kwargs: Any,
) -> LimitOrderBookEnv:
    """An env replaying `rows` from a two-sided snapshot, its first decision at time 0."""
    if snapshot is None:
        snapshot = snapshot_of([(100, 1), (99, 5)], [(101, 5), (102, 5)])
    # A far-away order at 0 and 20: the history runs from 0 to 20.
    rows = [(0.0, SUBMIT, 9_998, 1, 50, BID), *rows, (20.0, SUBMIT, 9_999, 1, 50, BID)]
    replay = MessageReplay(messages_of(rows), snapshot=snapshot)
    settings = {"session_length": 5.0, "warmup": 0.0, "inventory_penalty": 0.0, **kwargs}
    env = LimitOrderBookEnv(flow=replay, latency=latency, **settings)
    env.reset(options={"start": 0.0})
    return env


def trade_sides(observation: dict[str, Any]) -> list[float]:
    """The aggressor sides of the trades an observation reports."""
    trades = observation["trades"]
    return trades["values"][trades["mask"] == 1][:, 2].tolist()


class TestLatency:
    def test_market_order_arrives_after_the_price_moves(self) -> None:
        rows = [(0.02, DELETE, 900, 5, 101, ASK)]  # the best ask leaves at 0.02
        for latency, price in ((0.0, 101), (0.01, 101), (0.05, 102)):
            env = latency_env(rows, latency)
            r = env.step(np.array([0, 0, 2]))  # buy at market at time 0
            assert r.info["inventory"] == 1 and r.info["cash"] == -price, latency

    def test_cancel_arrives_after_the_fill(self) -> None:
        rows = [(1.02, EXECUTE, 900, 2, 100, BID)]  # a seller takes the seeded lot and the next
        for latency, filled in ((0.0, False), (0.05, True)):
            env = latency_env(rows, latency)
            env.step(np.array([1, 0, 1]))  # join the bid at 100 (behind the seeded lot)
            assert env.trader.bid_id is not None
            r = env.step(np.array([0, 0, 1]))  # at 1.0: cancel the bid
            assert (r.info["inventory"] == 1) is filled, latency
            assert env.book.orders_of(AGENT) == []

    def test_quote_is_priced_when_decided(self) -> None:
        # At 0.01 the bid at 100 goes and an ask arrives at 100: the price falls.
        rows = [(0.01, DELETE, 900, 1, 100, BID), (0.02, SUBMIT, 1, 3, 100, ASK)]
        env = latency_env(rows, 0.05)
        r = env.step(np.array([1, 0, 1]))  # join the best bid, 100, as observed at 0
        # The bid reaches the book at 0.05 at 100, crosses the new ask and buys as a taker.
        assert r.info["inventory"] == 1 and r.info["cash"] == -100
        assert trade_sides(r.observation) == [1.0] and env.book.volume_at(Side.ASK, 100) == 2
        env = latency_env(rows, 0.0)
        r = env.step(np.array([1, 0, 1]))  # at once: the bid rests, and the new ask sells to it
        assert r.info["inventory"] == 1 and r.info["cash"] == -100
        assert trade_sides(r.observation) == [-1.0] and env.book.volume_at(Side.ASK, 100) == 2

    def test_messages_at_the_arrival_time_come_first(self) -> None:
        rows = [(0.05, DELETE, 900, 5, 101, ASK)]
        env = latency_env(rows, 0.05)
        r = env.step(np.array([0, 0, 2]))
        assert r.info["cash"] == -102

    def test_kept_quote_is_not_resent(self) -> None:
        rows = [(1.02, EXECUTE, 900, 1, 100, BID), (1.03, EXECUTE, 901, 1, 100, BID)]
        env = latency_env(rows, 0.05, snapshot=snapshot_of([(100, 1), (99, 5)], [(102, 5)]))
        env.step(np.array([1, 0, 1]))  # rest a bid at 100, behind the seeded lot
        first = env.trader.bid_id
        r = env.step(np.array([1, 0, 1]))  # keep it: no order is sent
        # It filled at 1.03, while nothing was in flight for it; it is not placed again.
        assert r.info["inventory"] == 1 and env.trader.bid_id == first
        assert env.book.orders_of(AGENT) == []

    def test_orders_after_the_close_are_dropped(self) -> None:
        env = latency_env([], 0.6, session_length=4.5)  # the last step is half an interval
        for _ in range(4):
            r = env.step(np.array([0, 0, 2]))
        assert r.info["inventory"] == 4
        r = env.step(np.array([0, 0, 2]))  # decided at 4.0, it would arrive at 4.6
        assert r.done and r.info["inventory"] == 4 and r.info["fills"] == 0

    def test_latency_with_simulated_flow(self) -> None:
        env = LimitOrderBookEnv(latency=0.1)
        assert sekai.check_env(env, check_determinism=True) == []
        r = env.reset(seed=0)
        total = 0.0
        while not r.done:
            r = env.step(env.action_space.sample(mask=env.action_mask()))
            total += r.reward
            assert abs(r.info["inventory"]) <= env.core.max_inventory
        penalty_free = LimitOrderBookEnv(latency=0.3, inventory_penalty=0.0, session_length=50)
        r = penalty_free.reset(seed=1)
        total = 0.0
        while not r.done:
            r = penalty_free.step(np.array([1, 1, 1]))
            total += r.reward
        assert total == pytest.approx(r.info["wealth"])

    def test_random_latency(self) -> None:
        draws: list[float] = []

        def latency(rng: np.random.Generator) -> float:
            draws.append(float(rng.exponential(0.05)))
            return min(draws[-1], 1.0)

        env = LimitOrderBookEnv(latency=latency, session_length=10.0)
        env.reset(seed=0)
        for _ in range(10):
            env.step(np.array([1, 1, 1]))
        assert len(draws) == 10 and len(set(draws)) == 10

    @pytest.mark.parametrize("value", [-0.1, 1.5, math.nan, "fast"])
    def test_bad_latency(self, value: Any) -> None:
        with pytest.raises(ValueError, match="latency"):
            LimitOrderBookEnv(latency=value)
        env = LimitOrderBookEnv(latency=lambda rng: value)
        env.reset(seed=0)
        with pytest.raises(ValueError, match="latency"):
            env.step(np.array([1, 1, 1]))

    def test_latency_of_a_whole_interval_survives_rounding(self) -> None:
        # 0.1 is not exact in binary: decision time + latency can exceed the next decision
        # time by a rounding error, which must not drop the order.
        env = LimitOrderBookEnv(
            latency=0.1, decision_interval=0.1, session_length=3.0, max_inventory=100
        )
        r = env.reset(seed=0)
        while not r.done:
            r = env.step(np.array([0, 0, 2]))  # buy at market every step
        assert r.info["inventory"] == env.n_steps == 30

    def test_latency_equal_to_the_interval(self) -> None:
        rows = [(0.5, DELETE, 900, 5, 101, ASK)]
        env = latency_env(rows, 1.0)
        r = env.step(np.array([0, 0, 2]))  # arrives exactly at the next decision
        assert r.info["inventory"] == 1 and r.info["cash"] == -102


# ---------------------------------------------------------------------------
# Defaults unchanged: reference rollouts recorded before replay and latency existed
# ---------------------------------------------------------------------------
# tests/data/lob_reference.npz holds per-step digests of rollouts of LimitOrderBook-v0 and
# LimitOrderBookMarket with their default behaviour, recorded by tests/data/lob_reference.py
# on the code before this feature (see that module before regenerating it).


@pytest.fixture(scope="module")
def reference() -> dict[str, np.ndarray]:
    with np.load(DATA / "lob_reference.npz") as data:
        return {key: data[key] for key in data.files}


class TestDefaultsUnchanged:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    @pytest.mark.parametrize("config", sorted(SINGLE_CONFIGS))
    def test_single_agent(self, reference: dict[str, np.ndarray], config: str, seed: int) -> None:
        digests = single_rollout(SINGLE_CONFIGS[config], seed, reference["single_actions"])
        expected = reference[f"single_{config}_{seed}_digests"].tolist()
        assert len(digests) == len(expected)
        diverged = next((i for i, (a, b) in enumerate(zip(digests, expected)) if a != b), None)
        assert diverged is None, f"the rollout differs from step {diverged} on"

    @pytest.mark.parametrize("seed", [0, 1, 2])
    @pytest.mark.parametrize("config", sorted(MULTI_CONFIGS))
    def test_multi_agent(self, reference: dict[str, np.ndarray], config: str, seed: int) -> None:
        digests = multi_rollout(MULTI_CONFIGS[config], seed, reference["multi_actions"])
        expected = reference[f"multi_{config}_{seed}_digests"].tolist()
        assert len(digests) == len(expected)
        diverged = next((i for i, (a, b) in enumerate(zip(digests, expected)) if a != b), None)
        assert diverged is None, f"the rollout differs from step {diverged} on"

    def test_reference_digests_are_sensitive(self, reference: dict[str, np.ndarray]) -> None:
        actions = reference["single_actions"][:20]
        base = single_rollout({}, 0, actions)
        assert base == reference["single_default_0_digests"].tolist()[:21]
        assert single_rollout({}, 1, actions)[1:] != base[1:]
        assert single_rollout({}, 0, actions[::-1])[1:] != base[1:]
        assert single_rollout({"inventory_penalty": 0.02}, 0, actions)[1:] != base[1:]


# ---------------------------------------------------------------------------
# Objects pickled by sekai 0.2.0 load and continue as they did there
# ---------------------------------------------------------------------------
# tests/data/lob_pickles.npz holds order-book objects (environments as `sekai.make`
# returns them, LimitOrderBookMarket, an OrderBook with its ZeroIntelligenceFlow) pickled
# mid-run by sekai 0.2.0 under NumPy 1.24, with digests of how each run continued under
# 0.2.0. It is recorded by tests/data/lob_pickles.py, which also defines how each run
# continues (see that module before regenerating it).


@pytest.fixture(scope="module")
def pickled_by_020() -> dict[str, np.ndarray]:
    with np.load(DATA / "lob_pickles.npz") as data:
        return {key: data[key] for key in data.files}


class TestPickledBy020:
    @pytest.mark.parametrize("name", sorted(CONTINUATIONS))
    def test_continues_as_under_020(self, pickled_by_020: dict[str, np.ndarray], name: str) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # loads without a deprecation warning, too
            obj = pickle.loads(pickled_by_020[f"{name}__pickle"].tobytes())
        digests = CONTINUATIONS[name](obj)
        expected = pickled_by_020[f"{name}__digests"].tolist()
        assert len(digests) == len(expected)
        diverged = next((i for i, (a, b) in enumerate(zip(digests, expected)) if a != b), None)
        assert diverged is None, f"the run differs from step {diverged} on"

    def test_every_case_is_recorded(self, pickled_by_020: dict[str, np.ndarray]) -> None:
        kinds = ("pickle", "digests")
        assert set(pickled_by_020) == {f"{n}__{k}" for n in CONTINUATIONS for k in kinds}

    def test_a_loaded_env_snapshots_and_restores(
        self, pickled_by_020: dict[str, np.ndarray]
    ) -> None:
        env = pickle.loads(pickled_by_020["LimitOrderBook-v0__pickle"].tobytes())
        assert env.latency == 0.0 and env.core.n_starts == 0  # the defaults of 0.2.0
        snapshot = env.snapshot()
        first = [env.step(np.array([1, 1, 1])).reward for _ in range(5)]
        env.restore(snapshot)
        assert [env.step(np.array([1, 1, 1])).reward for _ in range(5)] == first
