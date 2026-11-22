"""Order-book objects pickled by sekai 0.2.0, with how their runs continued under 0.2.0.

`lob_pickles.npz`, next to this file, holds objects pickled mid-run by sekai 0.2.0:
`LimitOrderBook-v0`, `MarketMaking-v0` and `OptimalExecution-v0` as `sekai.make` returns
them (with a `TimeLimit` wrapper too), a `LimitOrderBookEnv` and a `LimitOrderBookMarket`,
and an `OrderBook` driven by a `ZeroIntelligenceFlow` with its random generator. For each,
it also holds per-step digests of how the run continued under 0.2.0: the rest of the
episode, then new episodes. `tests/test_finance_lob_replay.py` loads the objects with
the current code and continues each run the same way (`CONTINUATIONS`), so an object
pickled by 0.2.0 that no longer loads, or continues differently, shows as a failure with
the first step that differs.

This module uses only what existed in 0.2.0, and records only from it (it refuses any
other version). Record under NumPy 1.24, the oldest NumPy sekai supports, so that the
pickles load on every later NumPy as well:

    mkdir -p /tmp/sekai-020 && git archive e0eff20 sekai | tar -x -C /tmp/sekai-020
    PYTHONPATH=/tmp/sekai-020 python tests/data/lob_pickles.py tests/data/lob_pickles.npz

(with a Python environment that has NumPy 1.24, on x86-64: arm64 builds of NumPy round
`uniform()` differently). Regenerate it only for an outside reason, such as a NumPy
release that alters a random stream.
"""

from __future__ import annotations

import pickle
import sys
from collections.abc import Callable
from typing import Any

import numpy as np

import sekai
from sekai.envs.finance import LimitOrderBookEnv, LimitOrderBookMarket
from sekai.finance import OrderBook, Side, ZeroIntelligenceFlow

if __package__:  # imported from the test suite
    from tests.data.lob_reference import digest
else:  # run as a script, from this directory
    from lob_reference import digest  # type: ignore[no-redef]

PROTOCOL = 4  # loadable by every Python sekai supports


def _action(space: Any, rng: np.random.Generator) -> Any:
    """A random action of `space` (a MultiDiscrete or a bounded Box), from `rng`."""
    if hasattr(space, "nvec"):
        return rng.integers(0, np.asarray(space.nvec)).astype(np.int64)
    return rng.uniform(space.low, space.high).astype(space.dtype)


def _play(env: Any, rng: np.random.Generator, steps: int, out: list[int] | None = None) -> None:
    """Step a single-agent env for up to `steps` steps (digests appended to `out`)."""
    for _ in range(steps):
        s = env.step(_action(env.action_space, rng))
        if out is not None:
            out.append(digest(s.observation, s.reward, s.terminated, s.truncated, s.info))
        if s.terminated or s.truncated:
            return


def _play_market(
    env: LimitOrderBookMarket, rng: np.random.Generator, steps: int, out: list[int] | None = None
) -> None:
    """Step a LimitOrderBookMarket for up to `steps` steps (digests appended to `out`)."""
    for _ in range(steps):
        if not env.agents:
            return
        actions = {a: _action(env.action_spaces[a], rng) for a in sorted(env.acting)}
        r = env.step(actions)
        if out is not None:
            out.append(digest(r.observations, r.rewards, r.terminated, r.truncated, r.info))


def _continue_single(
    seed: int, steps: int, resets: tuple[int, ...], *, started: bool = True
) -> Callable[[Any], list[int]]:
    """Continue the episode for up to `steps` steps, then play an episode from each reset seed."""

    def run(env: Any) -> list[int]:
        rng = np.random.default_rng(seed)
        out: list[int] = []
        if started:
            _play(env, rng, steps, out)
        for reset_seed in resets:
            r = env.reset(seed=reset_seed)
            out.append(digest(r.observation, r.info))
            _play(env, rng, steps, out)
        return out

    return run


def _continue_market(seed: int, steps: int, resets: tuple[int, ...]) -> Callable[[Any], list[int]]:
    def run(env: LimitOrderBookMarket) -> list[int]:
        rng = np.random.default_rng(seed)
        out: list[int] = []
        _play_market(env, rng, steps, out)
        for reset_seed in resets:
            r = env.reset(seed=reset_seed)
            out.append(digest(r.observations, r.info))
            _play_market(env, rng, steps, out)
        return out

    return run


def _continue_book(state: tuple[OrderBook, ZeroIntelligenceFlow, np.random.Generator]) -> list[int]:
    """Drive the book on with its flow and generator, with an agent's orders in between."""
    book, flow, rng = state
    out: list[int] = []
    mine: int | None = None
    for k in range(40):
        fills = flow.advance(book, rng, book.time + 0.5)
        best_bid = book.best_bid()
        if k % 5 == 0 and best_bid is not None:
            mine, more = book.limit(Side.BID, best_bid, 2, owner=7)
            fills += more
        elif k % 5 == 2:
            fills += book.market(Side.ASK, 3, owner=8)[1]
        elif k % 5 == 4 and mine is not None:
            book.cancel(mine)
        resting = book.get(mine) if mine is not None else None
        trades = [
            (
                f.price,
                f.quantity,
                f.maker_order_id,
                f.taker_order_id,
                f.maker_owner,
                f.taker_owner,
                int(f.taker_side),
                f.time,
            )
            for f in fills
        ]
        out.append(
            digest(
                book.depth(5),
                trades,
                [(o.order_id, int(o.side), o.price, o.quantity, o.time) for o in book.orders_of(7)],
                book.queue_ahead(mine) if mine is not None and resting is not None else None,
                len(book),
                book.last_price,
                book.time,
            )
        )
    return out


# How each run continues from its unpickled object (the same under 0.2.0 and now).
CONTINUATIONS: dict[str, Callable[[Any], list[int]]] = {
    "LimitOrderBook-v0": _continue_single(10, 40, (4,)),
    "LimitOrderBook-v0-time-limit": _continue_single(11, 35, (6,)),
    "LimitOrderBook-v0-unstarted": _continue_single(12, 45, (7, 8), started=False),
    "LimitOrderBookEnv": _continue_single(13, 20, (9,)),
    "LimitOrderBookMarket": _continue_market(14, 200, (11,)),
    "MarketMaking-v0": _continue_single(15, 200, (13,)),
    "OptimalExecution-v0": _continue_single(16, 100, (15,)),
    "OrderBook": _continue_book,
}


def _build(name: str) -> Any:
    """The object of case `name`, run part of the way (sekai 0.2.0 only)."""
    rng = np.random.default_rng(2026)
    if name == "LimitOrderBook-v0":
        env = sekai.make("LimitOrderBook-v0")
        env.reset(seed=3)
        _play(env, rng, 25)
        return env
    if name == "LimitOrderBook-v0-time-limit":
        env = sekai.make(
            "LimitOrderBook-v0",
            max_episode_steps=30,
            flow=ZeroIntelligenceFlow(cancel_rate=1.0),
            decision_interval=0.5,
            max_inventory=3,
        )
        env.reset(seed=5)
        _play(env, rng, 20)
        return env
    if name == "LimitOrderBook-v0-unstarted":
        return sekai.make("LimitOrderBook-v0", levels=3, max_trades=4, session_length=20.0)
    if name == "LimitOrderBookEnv":
        env = LimitOrderBookEnv(session_length=15.0, inventory_penalty=0.05, max_quote_offset=3)
        env.reset(seed=8)
        _play(env, rng, 10)
        return env
    if name == "LimitOrderBookMarket":
        market = LimitOrderBookMarket(
            3, decision_intervals=[1.0, 2.0, 0.5], jitter=0.2, session_length=30.0
        )
        market.reset(seed=10)
        _play_market(market, rng, 15)
        return market
    if name == "MarketMaking-v0":
        env = sekai.make("MarketMaking-v0")
        env.reset(seed=12)
        _play(env, rng, 50)
        return env
    if name == "OptimalExecution-v0":
        env = sekai.make("OptimalExecution-v0")
        env.reset(seed=14)
        _play(env, rng, 10)
        return env
    if name == "OrderBook":
        book = OrderBook()
        flow = ZeroIntelligenceFlow(cancel_rate=0.7, mean_size=2.0)
        flow.initialise(book, rng, mid=5_000)
        flow.advance(book, rng, 10.0)
        book.limit(Side.ASK, book.best_ask() or 5_001, 4, owner=7)
        flow.advance(book, rng, 12.5)
        return book, flow, rng
    raise KeyError(name)


def record(path: str) -> None:
    """Pickle every case and record how its run continues, into the .npz file at `path`."""
    if sekai.__version__ != "0.2.0":
        raise SystemExit(f"Record with sekai 0.2.0 (see the docstring), not {sekai.__version__}.")
    out: dict[str, np.ndarray] = {}
    for name, continuation in CONTINUATIONS.items():
        obj = _build(name)
        blob = pickle.dumps(obj, protocol=PROTOCOL)
        digests = continuation(obj)
        # The pickle holds the whole state: a loaded copy continues the same way.
        assert continuation(pickle.loads(blob)) == digests, name
        out[f"{name}__pickle"] = np.frombuffer(blob, dtype=np.uint8)
        out[f"{name}__digests"] = np.array(digests, dtype="<u8")
        print(f"{name}: {len(blob)} bytes, {len(digests)} steps")
    np.savez_compressed(path, **out)


if __name__ == "__main__":
    record(sys.argv[1])
