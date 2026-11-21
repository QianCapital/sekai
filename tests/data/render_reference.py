"""Reference frames of the dashboards as sekai 0.2.0 drew them, as per-frame digests.

`render_reference.npz`, next to this file, holds 64-bit digests of every frame of short
episodes drawn by the seven finance renderers (in both themes, at two sizes) and by
dashboards built from `sekai.rendering.Dashboard` and its charts, recorded with sekai
0.2.0, before dashboards could be branded. It also holds charts, stats and themes
pickled by 0.2.0. `tests/test_rendering_reference.py` draws the same episodes with
`branding=False` (and the dashboards you build, with their default arguments, from new
and from the unpickled objects) and checks the digests, so a change to what 0.2.0 drew
shows as a failure, with the first frame that differs.

This module uses only what existed in 0.2.0, so it runs on that version. Regenerate the
file only from a checkout of it (never from later code, which would just record itself),
with NumPy 1.24 so that the pickles load on every supported NumPy, and only if the
reference must change for an outside reason such as a NumPy release that alters a random
stream:

    mkdir -p /tmp/sekai-020 && git archive v0.2.0 sekai | tar -x -C /tmp/sekai-020
    PYTHONPATH=/tmp/sekai-020 python tests/data/render_reference.py tests/data/render_reference.npz
"""

from __future__ import annotations

import hashlib
import pickle
import sys
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

import sekai
from sekai.envs.finance import (
    LimitOrderBookMarket,
    LimitOrderBookMarketRenderer,
    LimitOrderBookRenderer,
    MarketMakingRenderer,
    MarketRenderer,
    OptimalExecutionRenderer,
    PortfolioRenderer,
    TradingRenderer,
)
from sekai.finance import OrderBook, OrderBookChart, PriceChart, Side, UniverseChart
from sekai.rendering import (
    DARK,
    LIGHT,
    BarChart,
    Dashboard,
    LineChart,
    Row,
    Series,
    StackedArea,
    Stat,
)

THEMES = ("light", "dark")
SIZES = ((720, 450), (400, 250))
SEED = 0


def digest(frame: np.ndarray) -> int:
    """A 64-bit digest of a frame's shape and pixels."""
    data = np.asarray(frame.shape, dtype="<i8").tobytes() + np.ascontiguousarray(frame).tobytes()
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


# ----------------------------------------------------------------------
# The finance renderers
# ----------------------------------------------------------------------

Policy = Callable[[Any, np.random.Generator], Any]


def _uniform(low: float, high: float) -> Policy:
    def act(env: Any, rng: np.random.Generator) -> Any:
        return rng.uniform(low, high, env.action_space.shape).astype(np.float32)

    return act


FINANCE: dict[str, tuple[type, str, dict[str, Any], Policy]] = {
    "trading": (
        TradingRenderer,
        "Trading-v0",
        {"episode_length": 12},
        lambda env, rng: int(rng.integers(3)),
    ),
    "portfolio": (PortfolioRenderer, "Portfolio-v0", {"episode_length": 12}, _uniform(0, 1)),
    "market": (MarketRenderer, "Market-v0", {"episode_length": 12}, _uniform(0, 1)),
    "market_making": (
        MarketMakingRenderer,
        "MarketMaking-v0",
        {"n_steps": 12},
        _uniform(0.05, 1.5),
    ),
    "optimal_execution": (
        OptimalExecutionRenderer,
        "OptimalExecution-v0",
        {"n_steps": 10},
        _uniform(0.0, 0.3),
    ),
    "limit_order_book": (
        LimitOrderBookRenderer,
        "LimitOrderBook-v0",
        {"session_length": 12.0},
        lambda env, rng: rng.integers(0, [6, 6, 3]),
    ),
}
RENDERERS = (*FINANCE, "order_book_market")


def finance_renderer(name: str, theme: str, size: tuple[int, int], **kwargs: Any) -> Any:
    """Finance renderer `name` (a key of FINANCE, or "order_book_market")."""
    cls = LimitOrderBookMarketRenderer if name == "order_book_market" else FINANCE[name][0]
    return cls(width=size[0], height=size[1], theme=theme, **kwargs)


def finance_frames(name: str, renderer: Any) -> Iterator[Any]:
    """The frames of a short episode of finance renderer `name`'s environment."""
    if name == "order_book_market":
        market = LimitOrderBookMarket(
            n_agents=3, decision_intervals=[1.0, 1.5, 2.0], session_length=10.0
        )
        rng = np.random.default_rng(SEED)
        market.reset(seed=SEED)
        yield renderer.render(market)
        while market.agents:
            market.step({a: rng.integers(0, [6, 6, 3]) for a in sorted(market.acting)})
            yield renderer.render(market)
        return
    _, env_id, env_kwargs, policy = FINANCE[name]
    env = sekai.make(env_id, **env_kwargs)
    rng = np.random.default_rng(SEED)
    result = env.reset(seed=SEED)
    yield renderer.render(env)
    while not result.done:
        result = env.step(policy(env, rng))
        yield renderer.render(env)


# ----------------------------------------------------------------------
# Dashboards built from the generic charts
# ----------------------------------------------------------------------


class Desk(sekai.Env[np.ndarray, int]):
    """A toy trading desk: a price, a position, a book of three assets and an order book."""

    metadata = {"render_fps": 10}

    def __init__(self, episode_length: int = 12) -> None:
        self.episode_length = episode_length
        self.observation_space = sekai.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        self.action_space = sekai.Discrete(3)
        self.book = OrderBook()
        self._reset_state(SEED)

    def _reset_state(self, seed: Any) -> None:
        self.random = np.random.default_rng(seed)
        self.elapsed_steps = 0
        self.price, self.position, self.equity, self.benchmark = 100.0, 0.0, 1.0, 100.0
        self.inventory = 0.0
        self.history = 100.0 + np.cumsum(self.random.normal(0, 0.4, 8)) - 1.0
        self.weights = np.array([0.0, 0.0, 0.0, 1.0])
        self.holdings = np.zeros(3)
        self.tradable = np.ones(4, dtype=bool)
        self.listed = np.ones(3, dtype=bool)
        self.stressed = False
        self.book.clear()
        for k in range(1, 6):
            self.book.limit(Side.BID, 100 - k, k)
            self.book.limit(Side.ASK, 100 + k, k)
        self.book.limit(Side.BID, 99, 2, owner=7)

    def reset(self, *, seed: Any = None, options: dict[str, Any] | None = None) -> Any:
        self._reset_state(seed)
        return sekai.ResetResult(np.zeros(1, np.float32), {})

    def step(self, action: Any) -> Any:
        change = float(self.random.normal(0, 0.01))
        self.position = float(action) - 1.0
        self.equity *= 1 + self.position * change
        self.price *= 1 + change
        self.benchmark *= 1 + 0.5 * change
        self.inventory += float(self.random.integers(-2, 3))
        raw = self.random.uniform(-0.2, 1.0, 4)
        self.weights = raw / np.abs(raw).sum()
        self.holdings = self.weights[:3]
        self.tradable = np.array([True, self.elapsed_steps % 4 != 3, True, True])
        self.listed = np.array([True, True, self.elapsed_steps > 4])
        self.stressed = 3 <= self.elapsed_steps <= 6
        self.book.limit(Side.BID, 100 - int(self.random.integers(1, 5)), 1)
        self.book.limit(Side.ASK, 100 + int(self.random.integers(1, 5)), 1)
        self.elapsed_steps += 1
        done = self.elapsed_steps >= self.episode_length
        return sekai.StepResult(np.zeros(1, np.float32), 0.0, False, done, {})


def desk_items() -> tuple[Any, ...]:
    """A layout using every generic and market chart, with attribute sources only."""
    return (
        LineChart(
            Series("equity", label="Strategy", color="accent", fill=True),
            Series("benchmark", label="Benchmark", color="neutral", dashed=True, rebase=1.0),
            title="Equity",
            baseline=1.0,
            shade="stressed",
            shade_label="Stressed",
            height=2,
        ),
        Row(
            LineChart(
                Series("inventory", color="text_muted", step=True, signed=True),
                title="Inventory",
                baseline=0.0,
                limits=(-4.0, 4.0),
            ),
            StackedArea("weights", labels=["A", "B", "C", "Cash"], legend=True, title="Book"),
            BarChart("weights", labels=["A", "B", "C", "Cash"], available="tradable"),
        ),
        Row(
            PriceChart(
                "price",
                position="position",
                context="history",
                reference=100.0,
                reference_label="Entry",
            ),
            OrderBookChart("book", owners={7: ("Mine", "highlight")}, levels=4),
            UniverseChart("holdings", available="listed", labels=["A", "B", "C"]),
            widths=(2, 1, 1),
        ),
    )


def desk_stats() -> tuple[Stat, ...]:
    return (
        Stat("Equity", "equity", "{:.4f}", delta="relative"),
        Stat("Inventory", "inventory", "{:+.0f}", delta="absolute", higher_is_better=False),
        Stat("Price", "price"),
    )


class Spread(sekai.Env[np.ndarray, int]):
    """The guide's custom environment: trade the spread of a cointegrated pair."""

    def __init__(self, episode_length: int = 14) -> None:
        self.episode_length = episode_length
        self.observation_space = sekai.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
        self.action_space = sekai.Discrete(3)
        self.random = np.random.default_rng(SEED)
        self.elapsed_steps, self.spread, self.position, self.pnl = 0, 0.0, 0.0, 0.0

    def reset(self, *, seed: Any = None, options: dict[str, Any] | None = None) -> Any:
        self.random = np.random.default_rng(seed)
        self.elapsed_steps, self.spread, self.position, self.pnl = 0, 0.0, 0.0, 0.0
        return sekai.ResetResult(np.zeros(2, np.float32), {})

    def step(self, action: Any) -> Any:
        self.position = float(action) - 1.0
        move = -0.15 * self.spread + float(self.random.normal(0, 0.3))
        self.pnl += self.position * move
        self.spread += move
        self.elapsed_steps += 1
        done = self.elapsed_steps >= self.episode_length
        return sekai.StepResult(np.zeros(2, np.float32), 0.0, False, done, {})


def spread_dashboard(**kwargs: Any) -> Dashboard:
    """The guide's dashboard, with a title from a source and a function as a stat."""
    return Dashboard(
        PriceChart(
            "spread",
            position="position",
            reference=0.0,
            reference_label="Fair value",
            title="Spread",
            height=3,
        ),
        LineChart(
            Series("pnl", label="P&L", color="accent", fill=True),
            title="Profit and loss",
            baseline=0.0,
            height=2,
        ),
        title=lambda env: f"Spread trading, step {env.elapsed_steps}",
        subtitle="A custom environment",
        stats=[
            Stat("P&L", "pnl", "{:+.2f}"),
            Stat("Spread", lambda env: env.spread, lambda v: f"{v:+.2f}"),
        ],
        **kwargs,
    )


# Each dashboard: how to build it (from keyword arguments, which the test extends) and the
# environment it draws. "desk" has no title, so it shows the env's class name.
DASHBOARDS: dict[str, tuple[Callable[..., Dashboard], Callable[[], Any]]] = {
    "desk": (lambda **kw: Dashboard(*desk_items(), stats=desk_stats(), **kw), Desk),
    "desk_dark_small": (
        lambda **kw: Dashboard(
            *desk_items(), stats=desk_stats(), theme="dark", width=400, height=250, **kw
        ),
        Desk,
    ),
    "desk_window": (
        lambda **kw: Dashboard(*desk_items(), stats=desk_stats(), window=5, **kw),
        Desk,
    ),
    "spread": (spread_dashboard, Spread),
    "spread_dark": (lambda **kw: spread_dashboard(theme=DARK, **kw), Spread),
}


def dashboard_frames(renderer: Dashboard, env: Any) -> Iterator[Any]:
    """The frames of an episode of `env` drawn by `renderer`, acting short, flat, long."""
    result = env.reset(seed=SEED)
    yield renderer.render(env)
    i = 0
    while not result.done:
        result = env.step(i % 3)
        i += 1
        yield renderer.render(env)


def pickled_objects() -> dict[str, Any]:
    """What the test unpickles: the desk's charts and stats, and the themes."""
    return {"items": desk_items(), "stats": desk_stats(), "light": LIGHT, "dark": DARK}


def record(path: str) -> None:
    """Record every reference episode into the .npz file at `path`."""
    out: dict[str, np.ndarray] = {}
    for name in RENDERERS:
        for theme in THEMES:
            for width, height in SIZES:
                frames = finance_frames(name, finance_renderer(name, theme, (width, height)))
                key = f"finance_{name}_{theme}_{width}x{height}"
                out[key] = np.array([digest(f) for f in frames], dtype="<u8")
    for name, (build, make_env) in DASHBOARDS.items():
        frames = dashboard_frames(build(), make_env())
        out[f"dashboard_{name}"] = np.array([digest(f) for f in frames], dtype="<u8")
    out["pickle"] = np.frombuffer(pickle.dumps(pickled_objects(), protocol=4), dtype=np.uint8)
    np.savez_compressed(path, **out)


if __name__ == "__main__":
    record(sys.argv[1])
