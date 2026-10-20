"""The finance renderers and the market charts they are built from."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import sekai
from sekai.envs.finance import (
    LimitOrderBookEnv,
    LimitOrderBookMarket,
    LimitOrderBookMarketRenderer,
    LimitOrderBookRenderer,
    MarketMakingRenderer,
    MarketRenderer,
    OptimalExecutionRenderer,
    PortfolioRenderer,
    TradingRenderer,
)
from sekai.envs.finance.rendering import _chosen_position, _ExecutionPrice, _TraderPrice
from sekai.finance import OrderBook, OrderBookChart, PriceChart, Side, UniverseChart
from sekai.rendering import LIGHT, Dashboard, LineChart, RecordFrames, Series
from sekai.rendering.dashboard import View, _History

SMALL = {"width": 360, "height": 225}

CASES = [
    ("Trading-v0", {"episode_length": 20}, TradingRenderer),
    ("Portfolio-v0", {"episode_length": 20}, PortfolioRenderer),
    ("Market-v0", {"episode_length": 20}, MarketRenderer),
    ("MarketMaking-v0", {"n_steps": 20}, MarketMakingRenderer),
    ("OptimalExecution-v0", {"n_steps": 10}, OptimalExecutionRenderer),
    ("LimitOrderBook-v0", {"session_length": 20.0}, LimitOrderBookRenderer),
]


def episode(env: Any, renderer: Any, seed: int = 0) -> list[np.ndarray]:
    result = env.reset(seed=seed)
    env.action_space.seed(seed)
    frames = [renderer.render(env)]
    while not result.done:
        result = env.step(env.action_space.sample())
        frames.append(renderer.render(env))
    return frames


@pytest.mark.parametrize(("env_id", "kwargs", "renderer_cls"), CASES)
def test_renders_an_episode(env_id: str, kwargs: dict[str, Any], renderer_cls: Any) -> None:
    env = sekai.make(env_id, **kwargs)
    frames = episode(env, renderer_cls(**SMALL))
    assert len(frames) >= 2
    for frame in frames:
        assert frame.shape == (225, 360, 3) and frame.dtype == np.uint8
    assert len(np.unique(frames[-1].reshape(-1, 3), axis=0)) > 20  # a picture, not a blank
    assert not np.array_equal(frames[0], frames[-1])


@pytest.mark.parametrize(("env_id", "kwargs", "renderer_cls"), CASES)
def test_deterministic(env_id: str, kwargs: dict[str, Any], renderer_cls: Any) -> None:
    a = episode(sekai.make(env_id, **kwargs), renderer_cls(**SMALL), seed=4)
    b = episode(sekai.make(env_id, **kwargs), renderer_cls(**SMALL), seed=4)
    assert len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize(("env_id", "kwargs", "renderer_cls"), CASES)
def test_rejects_other_envs(env_id: str, kwargs: dict[str, Any], renderer_cls: Any) -> None:
    other = sekai.make("CartPole-v1")
    other.reset(seed=0)
    with pytest.raises(TypeError, match=renderer_cls.__name__):
        renderer_cls().render(other)


@pytest.mark.parametrize("renderer_cls", [TradingRenderer, PortfolioRenderer])
def test_needs_a_reset(renderer_cls: Any) -> None:
    env = sekai.make("Trading-v0" if renderer_cls is TradingRenderer else "Portfolio-v0")
    with pytest.raises(sekai.ResetRequired):
        renderer_cls().render(env)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_themes_and_sizes(theme: str) -> None:
    env = sekai.make("Trading-v0", episode_length=10)
    for width, height in ((720, 450), (1440, 900), (320, 200)):
        frame = episode(env, TradingRenderer(width=width, height=height, theme=theme))[-1]
        assert frame.shape == (height, width, 3)
        assert tuple(frame[0, 0]) == sekai.rendering.get_theme(theme).background


def test_record_frames_and_gif(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    env = RecordFrames(sekai.make("MarketMaking-v0", n_steps=8), MarketMakingRenderer(**SMALL))
    episode(env, _NoOp())
    assert len(env.frames) == 9
    assert env.save(tmp_path / "mm.gif").stat().st_size > 0


class _NoOp:
    def render(self, env: Any) -> None:
        return None


def test_window_scrolls() -> None:
    env = sekai.make("Trading-v0", episode_length=60)
    renderer = TradingRenderer(window=15, **SMALL)
    episode(env, renderer)
    assert len(renderer._histories[env.unwrapped].steps) == 16


def test_gymnasium_render_mode() -> None:
    pytest.importorskip("gymnasium")
    from sekai.compat import to_gymnasium

    env = to_gymnasium(sekai.make("Trading-v0", episode_length=5), renderer=TradingRenderer())
    env.reset(seed=0)
    env.step(2)
    assert env.render().shape == (450, 720, 3)


# ----------------------------------------------------------------------
# What the renderers read from the environments
# ----------------------------------------------------------------------


def test_trading_position_is_the_chosen_target() -> None:
    # The env reports the position after it drifted with the price; the renderer
    # recovers the target that was chosen, so trades are marked only on changes.
    env = sekai.make("Trading-v0", episode_length=30)
    env.reset(seed=1)
    positions = env.unwrapped.positions
    assert _chosen_position(env) == 0.0
    rng = np.random.default_rng(0)
    for _ in range(30):
        action = int(rng.integers(len(positions)))
        env.step(action)
        assert _chosen_position(env) == pytest.approx(positions[action], abs=1e-9)


def test_market_making_fills_are_drawn_at_their_quotes() -> None:
    env = sekai.make("MarketMaking-v0")
    env.reset(seed=0)
    chart = MarketMakingRenderer().panels[0]
    assert isinstance(chart, PriceChart)
    buys, sells = chart.buys, chart.sells
    assert buys is not None and sells is not None and callable(buys) and callable(sells)
    seen = 0
    for _ in range(50):
        result = env.step(np.array([0.3, 0.3], np.float32))
        base = env.unwrapped
        assert buys(env) == (result.info["bid"] if result.info["bid_filled"] else None)
        assert sells(env) == (result.info["ask"] if result.info["ask_filled"] else None)
        seen += base.bid_filled + base.ask_filled
    assert seen > 0


def test_execution_prices_are_recovered_from_the_shortfall() -> None:
    env = sekai.make("OptimalExecution-v0")
    renderer = OptimalExecutionRenderer()
    env.reset(seed=0)
    renderer.render(env)
    expected = []
    result = env.reset(seed=0)
    renderer.render(env)
    for action in env.unwrapped.almgren_chriss_actions():
        result = env.step(action)
        expected.append(result.info["execution_price"])
        renderer.render(env)
    chart = renderer.panels[2]
    assert isinstance(chart, _ExecutionPrice)
    view = _view(renderer, env, chart)
    sells = [price for _, price in chart.fills(view)["sells"]]
    np.testing.assert_allclose(sells, expected, rtol=1e-9)


def test_order_book_fills_follow_inventory() -> None:
    env = sekai.make("LimitOrderBook-v0", session_length=60.0)
    renderer = LimitOrderBookRenderer()
    result = env.reset(seed=3)
    renderer.render(env)
    inventory = [0]
    rng = np.random.default_rng(0)
    while not result.done:
        result = env.step(np.array([1, 1, rng.choice([0, 1, 2])]))
        renderer.render(env)
        inventory.append(result.info["inventory"])
    chart = renderer.panels[0]
    assert isinstance(chart, _TraderPrice)
    fills = chart.fills(_view(renderer, env, chart))
    changes = np.diff(inventory)
    assert len(fills["buys"]) == (changes > 0).sum()
    assert len(fills["sells"]) == (changes < 0).sum()
    mids = [p for _, p in fills["buys"] + fills["sells"]]
    assert all(9_000 < p < 11_000 for p in mids)


def _view(renderer: Dashboard, env: Any, panel: Any) -> View:
    history = renderer._histories[env.unwrapped]
    prefix = f"{renderer.panels.index(panel)}:"
    return View(env, history, prefix, LIGHT, 1.0, (0, 1), False, 0)


def test_limit_order_book_market() -> None:
    env = LimitOrderBookMarket(n_agents=3, decision_intervals=[1.0, 1.5, 2.0], session_length=15)
    renderer = LimitOrderBookMarketRenderer(**SMALL)
    env.reset(seed=0)
    frames = [renderer.render(env)]
    rng = np.random.default_rng(0)
    while env.agents:
        actions = {a: np.array([1, 1, int(rng.integers(3))]) for a in env.acting}
        env.step(actions)
        frames.append(renderer.render(env))
    assert frames[-1].shape == (225, 360, 3) and len(frames) > 5
    assert renderer._histories[env].steps[-1] == pytest.approx(15.0)  # session time
    with pytest.raises(TypeError):
        renderer.render(LimitOrderBookEnv())


# ----------------------------------------------------------------------
# The market charts on their own
# ----------------------------------------------------------------------


class Pair(sekai.Env[np.ndarray, int]):
    """A custom market env: a spread that mean-reverts, a position, a book, a universe."""

    def __init__(self) -> None:
        self.observation_space = sekai.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        self.action_space = sekai.Discrete(3)
        self.episode_length = 25
        self.elapsed_steps = 0
        self.spread = 0.0
        self.position = 0.0
        self.weights = np.zeros(3)
        self.listed = np.ones(3, dtype=bool)
        self.book = OrderBook()

    def reset(self, *, seed: Any = None, options: dict[str, Any] | None = None) -> Any:
        self._set_rng(seed)
        self.elapsed_steps, self.spread, self.position = 0, 0.0, 0.0
        self.book.clear()
        for k in range(1, 6):
            self.book.limit(Side.BID, 100 - k, k)
            self.book.limit(Side.ASK, 100 + k, k)
        self.book.limit(Side.BID, 99, 2, owner=7)
        return sekai.ResetResult(np.zeros(1, np.float32), {})

    def step(self, action: int) -> Any:
        self.position = float(action) - 1.0
        self.spread += -0.2 * self.spread + float(self.rng.normal(0, 0.5))
        self.weights = np.array([self.position, -self.position, 0.0])
        self.listed = np.array([True, True, self.elapsed_steps > 10])
        self.elapsed_steps += 1
        return sekai.StepResult(np.zeros(1, np.float32), 0.0, False, self.elapsed_steps >= 25, {})


def test_market_charts_draw_a_custom_env() -> None:
    renderer = Dashboard(
        PriceChart(
            "spread",
            position="position",
            context=lambda env: np.linspace(-1, 0, 10),
            reference=0.0,
            reference_label="Fair value",
        ),
        [
            UniverseChart("weights", available="listed", labels=["A", "B", "C"]),
            OrderBookChart("book", owners={7: ("Mine", "highlight")}, levels=4),
        ],
        LineChart(
            Series("position", step=True, signed=True),
            baseline=0.0,
            shade=lambda env: env.spread > 1.0,
            shade_label="Wide spread",
        ),
        **SMALL,
    )
    env = Pair()
    frames = episode(env, renderer)
    assert frames[-1].shape == (225, 360, 3)
    history = renderer._histories[env]
    assert history.context["0:context"].shape == (10,)
    vector_shade = Dashboard(LineChart("position", shade="listed"), **SMALL)
    env.reset(seed=0)
    with pytest.raises(ValueError, match="one true/false value"):
        vector_shade.render(env)


def test_price_chart_starts_its_axis_at_the_context() -> None:
    chart = PriceChart("spread", context=lambda env: np.arange(8.0))
    history = _History()
    history.context = {"0:context": np.arange(8.0)}
    history.record(0, {"0:price": 1.0}, replace=False)
    view = View(None, history, "0:", LIGHT, 1.0, (0, 1), False, 0)
    assert chart.x_start(view) == -7.0


def test_order_book_chart_needs_a_book() -> None:
    renderer = Dashboard(OrderBookChart(lambda env: "not a book"), **SMALL)
    env = Pair()
    env.reset(seed=0)
    with pytest.raises(TypeError, match="OrderBook"):
        renderer.render(env)


def test_empty_order_book() -> None:
    renderer = Dashboard(OrderBookChart(lambda env: OrderBook()), **SMALL)
    env = Pair()
    env.reset(seed=0)
    assert renderer.render(env).shape == (225, 360, 3)
    with pytest.raises(ValueError):
        OrderBookChart("book", levels=0)


def test_dense_trades_are_not_drawn_as_markers() -> None:
    # Trading every step for a long episode: markers would cover the price, so they are
    # left out (the shading still shows each position), and the frame stays fast.
    env = sekai.make("Trading-v0", episode_length=400)
    renderer = TradingRenderer(**SMALL)
    env.reset(seed=0)
    for t in range(400):
        env.step(2 if t % 2 else 0)
        renderer._record(env)
    dense = renderer.render(env)
    sparse_env = sekai.make("Trading-v0", episode_length=400)
    sparse = TradingRenderer(**SMALL)
    sparse_env.reset(seed=0)
    for t in range(400):
        sparse_env.step(2 if t < 200 else 0)  # one trade
        sparse._record(sparse_env)
    assert dense.shape == sparse.render(sparse_env).shape
