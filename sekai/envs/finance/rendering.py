"""RGB renderers for the finance environments.

Each renderer draws one environment's episode so far as a dashboard: a header with
the environment's name and headline numbers, then charts of its market and of the
agent (prices with positions and trades, equity, allocations, order books). They
take the env, wrapped or not, and return an `(H, W, 3)` uint8 frame, like the
classic-control renderers, and they are `sekai.rendering.Dashboard`s assembled from
the charts in `sekai.rendering` and `sekai.finance.rendering`, so you can build the
same kind of renderer for your own environment.

    env = RecordFrames(sekai.make("Trading-v0"), TradingRenderer())
    ... run an episode ...
    env.save("trading.gif")

Every renderer takes `width`, `height` (pixels; text and marks scale with the
frame), `theme` ("light", "dark" or a `sekai.rendering.Theme`) and `window`
(show only the latest steps).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from sekai.envs.finance.lob_env import LimitOrderBookEnv, LimitOrderBookMarket
from sekai.envs.finance.market_env import MarketEnv
from sekai.envs.finance.market_making import MarketMakingEnv
from sekai.envs.finance.optimal_execution import OptimalExecutionEnv
from sekai.envs.finance.portfolio import PortfolioEnv
from sekai.envs.finance.trading import TradingEnv
from sekai.finance.rendering import OrderBookChart, PriceChart, UniverseChart
from sekai.rendering.canvas import Canvas
from sekai.rendering.charts import Box, Plot, Swatch, format_number, nice_ticks, tick_labels
from sekai.rendering.dashboard import (
    BarChart,
    Column,
    Dashboard,
    LineChart,
    Panel,
    Row,
    Series,
    Source,
    StackedArea,
    Stat,
    View,
    tick_count,
)
from sekai.rendering.theme import ColorLike, Theme

__all__ = [
    "LimitOrderBookMarketRenderer",
    "LimitOrderBookRenderer",
    "MarketMakingRenderer",
    "MarketRenderer",
    "OptimalExecutionRenderer",
    "PortfolioRenderer",
    "TradingRenderer",
]


def _base(env: Any) -> Any:
    return env.unwrapped if hasattr(env, "unwrapped") else env


class _EnvRenderer(Dashboard):
    """A dashboard for one environment class (TypeError for any other)."""

    draws: ClassVar[type | tuple[type, ...]]

    def check(self, env: Any) -> None:
        base = _base(env)
        if not isinstance(base, self.draws):
            names = (
                self.draws.__name__
                if isinstance(self.draws, type)
                else "/".join(t.__name__ for t in self.draws)
            )
            raise TypeError(f"{type(self).__name__} draws {names}, got {type(base).__name__}.")


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------

_MODEL_NAMES = {
    "GBM": "geometric Brownian motion",
    "Heston": "Heston stochastic volatility",
    "MertonJumpDiffusion": "Merton jump diffusion",
    "RegimeSwitching": "regime switching",
    "GARCH": "GARCH(1, 1)",
    "MeanReverting": "mean reversion",
    "CointegratedPair": "cointegrated pair",
    "BlockBootstrap": "block bootstrap",
    "Randomized": "randomised models",
}


def _model_name(model: Any) -> str:
    if model is None:
        return _MODEL_NAMES["GBM"]
    name = type(model).__name__
    if name == "ListingProcess":
        return f"{_model_name(getattr(model, 'model', None))}, with listings"
    return _MODEL_NAMES.get(name, name)


def _date(stamp: Any) -> str | None:
    if isinstance(stamp, np.datetime64) and not np.isnat(stamp):
        return str(np.datetime_as_string(stamp, unit="D"))
    return None


def _market(env: Any) -> str:
    """The subtitle: where prices come from, and the date for dated history."""
    base = _base(env)
    if getattr(base, "data", None) is not None:
        date = _date(getattr(base, "timestamp", None))
        return f"Historical data · {date}" if date else "Historical data"
    return f"Simulated: {_model_name(getattr(base, 'model', None))}"


def _signed(value: Any, decimals: int = 2) -> str:
    return format_number(float(value), decimals, sign=True)


# ----------------------------------------------------------------------
# Trading-v0
# ----------------------------------------------------------------------


def _chosen_position(env: Any) -> float:
    """The position chosen at the last decision, before it drifted with the price.

    TradingEnv reports the drifted position `p = w (1 + r) / (1 + w r)` of the
    target `w`; inverting it recovers the target, so a trade shows only when the agent
    changed its target, not when the position merely drifted.
    """
    base = _base(env)
    if base.elapsed_steps == 0:
        return 0.0
    prices = base.price_history
    r = float(prices[-1] / prices[-2] - 1.0)
    p = float(base.position)
    denominator = 1.0 + r * (1.0 - p)
    return p / denominator if denominator > 0 else p


def _position_label(value: Any) -> str:
    v = float(value)
    if abs(v) < 1e-9:
        return "Flat"
    return f"{'Long' if v > 0 else 'Short'} {format_number(abs(v), 0, percent=True)}"


class TradingRenderer(_EnvRenderer):
    """Draws `Trading-v0` (`TradingEnv`).

    The price, with the lookback window before the episode in a muted tone; the
    position shaded behind it (long in the `up` colour, short in `down`, deeper for
    larger positions); trades as triangles at the price where they were decided; and
    the equity curve against buying and holding the asset.
    """

    draws = TradingEnv

    def __init__(
        self,
        *,
        width: int = 720,
        height: int = 450,
        theme: Theme | str = "light",
        window: int | None = None,
    ) -> None:
        super().__init__(
            PriceChart(
                "price",
                position=_chosen_position,
                context=lambda env: _base(env).price_history,
                height=3,
            ),
            LineChart(
                Series("equity", label="Strategy", color="accent", fill=True),
                Series("price", label="Buy and hold", color="neutral", rebase=1.0),
                title="Equity",
                baseline=1.0,
                height=2,
            ),
            subtitle=_market,
            stats=[
                Stat("Equity", "equity", "{:.4f}", delta="relative"),
                Stat("Position", _chosen_position, _position_label),
                Stat("Price", "price", "{:,.2f}"),
            ],
            width=width,
            height=height,
            theme=theme,
            window=window,
        )


# ----------------------------------------------------------------------
# Portfolio-v0
# ----------------------------------------------------------------------


def _asset_price(j: int) -> Source:
    def price(env: Any) -> float:
        return float(_base(env).price_history[-1, j])

    return price


def _asset_names(base: Any, n: int) -> list[str]:
    data = getattr(base, "data", None)
    if data is not None:
        return [str(a) for a in data.assets][:n]
    return [f"Asset {i + 1}" for i in range(n)]


class PortfolioRenderer(_EnvRenderer):
    """Draws `Portfolio-v0` (`PortfolioEnv`).

    The portfolio's value and every asset's price indexed to 100 at the start of the
    episode (on one axis, so they compare directly), the allocation over time, and
    the current weights. Each asset keeps its colour in every chart; cash is grey.
    """

    draws = PortfolioEnv

    def __init__(
        self,
        *,
        width: int = 720,
        height: int = 450,
        theme: Theme | str = "light",
        window: int | None = None,
    ) -> None:
        super().__init__(width=width, height=height, theme=theme, window=window, subtitle=_market)
        self._names: tuple[str, ...] | None = None

    def check(self, env: Any) -> None:
        super().check(env)
        base = _base(env)
        names = tuple(_asset_names(base, base.n_assets))
        if names != self._names:  # charts with one line and one bar per asset
            self._names = names
            self._configure(names)

    def _configure(self, names: Sequence[str]) -> None:
        n = len(names)
        colors: list[Any] = [*range(n), "neutral"]
        assets = [
            Series(_asset_price(j), label=name, color=j, width=1.25, rebase=100.0)
            for j, name in enumerate(names)
        ]
        performance = LineChart(
            Series("portfolio_value", label="Portfolio", color="price", width=2.25, rebase=100.0),
            *assets,
            title="Performance, indexed to 100",
            baseline=100.0,
            height=3,
        )
        allocation = StackedArea(
            "weights",
            labels=[*names, "Cash"],
            colors=colors,
            title="Allocation over time",
            height=2,
        )
        weights = BarChart(
            "weights", labels=[*names, "Cash"], colors=colors, limit=1.0, title="Weights"
        )
        self.set_layout(Row(Column(performance, allocation), weights, widths=(2.3, 1)))
        self.stats = (
            Stat("Value", "portfolio_value", "{:.4f}", delta="relative"),
            Stat(
                "Cash",
                lambda env: _base(env).weights[-1],
                lambda v: format_number(v, 1, percent=True),
            ),
        )


# ----------------------------------------------------------------------
# Market-v0
# ----------------------------------------------------------------------


def _tradable(env: Any) -> NDArray[np.int8]:
    base = _base(env)
    mask: NDArray[np.int8] = np.asarray(base.action_mask())[: base.max_assets]
    return mask


def _second_regime(env: Any) -> bool:
    """Whether the simulated market is in a regime other than its first (stressed)."""
    state = getattr(_base(env), "latent_state", None) or {}
    regime = state.get("regime")
    return regime is not None and float(np.asarray(regime).ravel()[0]) > 0


def _slot_labels(env: Any) -> list[str]:
    return ["\u2014" if a is None else str(a) for a in _base(env).slot_assets]


class MarketRenderer(_EnvRenderer):
    """Draws `Market-v0` (`MarketEnv`).

    Equity, with the periods the simulated market spends outside its first regime
    shaded (in the default market, the stressed one); the universe over time (one row
    per slot: filled while held, grey while tradable, empty while no asset is
    listed there); and the current holdings by slot.
    """

    draws = MarketEnv

    def __init__(
        self,
        *,
        width: int = 720,
        height: int = 450,
        theme: Theme | str = "light",
        window: int | None = None,
    ) -> None:
        equity = LineChart(
            Series("equity", label="Equity", color="accent", fill=True),
            title="Equity",
            baseline=1.0,
            shade=_second_regime,
            shade_label="Stressed regime",
            height=3,
        )
        universe = UniverseChart("weights", available=_tradable, labels=_slot_labels, height=2)
        holdings = BarChart(
            "weights", labels=_slot_labels, available=_tradable, color="accent", title="Holdings"
        )
        super().__init__(
            Row(Column(equity, universe), holdings, widths=(2.3, 1)),
            subtitle=_market,
            stats=[
                Stat("Equity", "equity", "{:.4f}", delta="relative"),
                Stat(
                    "Gross exposure",
                    lambda env: float(np.abs(_base(env).weights).sum()),
                    lambda v: format_number(v, 0, percent=True),
                ),
                Stat("Tradable", lambda env: int(_tradable(env).sum()), lambda v: f"{v} assets"),
            ],
            width=width,
            height=height,
            theme=theme,
            window=window,
        )


# ----------------------------------------------------------------------
# MarketMaking-v0
# ----------------------------------------------------------------------


def _filled(side: str) -> Source:
    def price(env: Any) -> float | None:
        base = _base(env)
        return float(getattr(base, side)) if getattr(base, f"{side}_filled") else None

    return price


class MarketMakingRenderer(_EnvRenderer):
    """Draws `MarketMaking-v0` (`MarketMakingEnv`).

    The mid price with the agent's bid and ask quotes around it and its fills
    (triangles at the price of each fill: up for a filled bid, down for a filled ask),
    its inventory, and its mark-to-market profit and loss.
    """

    draws = MarketMakingEnv

    def __init__(
        self,
        *,
        width: int = 720,
        height: int = 450,
        theme: Theme | str = "light",
        window: int | None = None,
    ) -> None:
        super().__init__(
            PriceChart(
                "mid",
                quotes=("bid", "ask"),
                buys=_filled("bid"),
                sells=_filled("ask"),
                title="Mid price and quotes",
                height=3,
            ),
            LineChart(
                Series("inventory", color="text_muted", width=1.25, step=True, signed=True),
                title="Inventory",
                baseline=0.0,
                height=1.6,
            ),
            LineChart(
                Series("wealth", color="accent", fill=True),
                title="Profit and loss",
                baseline=0.0,
                height=1.6,
            ),
            subtitle="Avellaneda\u2013Stoikov market making",
            stats=[
                Stat("P&L", "wealth", _signed, delta=None),
                Stat("Inventory", "inventory", lambda v: format_number(v, 0, sign=True)),
                Stat("Quoted spread", lambda env: _base(env).ask - _base(env).bid, "{:.2f}"),
                Stat(
                    "Time left",
                    lambda env: _base(env).time_remaining / _base(env).total_time,
                    lambda v: format_number(v, 0, percent=True),
                ),
            ],
            width=width,
            height=height,
            theme=theme,
            window=window,
        )


# ----------------------------------------------------------------------
# OptimalExecution-v0
# ----------------------------------------------------------------------


class _Holdings(Panel):
    """Shares left to sell, against the Almgren-Chriss schedule for the whole horizon."""

    def sources(self) -> Mapping[str, Source]:
        return {"inventory": "inventory"}

    def context(self) -> Mapping[str, Source]:
        return {
            "plan": lambda env: _base(env).almgren_chriss_trajectory(),
            "start": lambda env: _base(env).elapsed_steps,
        }

    def draw(self, canvas: Canvas, box: Box, view: View) -> None:
        theme, s = view.theme, view.scale
        items: list[tuple[str, ColorLike, Swatch]] = [
            ("Remaining", theme.accent, "area"),
            ("Almgren\u2013Chriss", theme.neutral, "dash"),
        ]
        plot_box = self.frame(canvas, box, view, items)
        plan = np.asarray(view.context("plan"), dtype=np.float64)
        inventory = view.values("inventory")
        top = max(float(np.nanmax(plan, initial=0.0)), float(np.nanmax(inventory, initial=0.0)))
        plot = Plot(canvas, plot_box, view.x_domain, (0.0, top * 1.06 or 1.0), theme, s)
        ticks = nice_ticks(0.0, top * 1.06 or 1.0, tick_count(plot_box.height, s))
        plot.grid(ticks, tick_labels(ticks, compact=True), avoid=[inventory[-1]])
        plot.hline(0.0, theme.axis)
        plot.line(np.arange(plan.size), plan, theme.neutral, width=1.5, dash=(5, 3))
        steps = view.steps
        plot.area(steps, inventory, theme.accent, alpha=0.1, step=True)
        plot.line(steps, inventory, theme.accent, step=True)
        plot.marker(float(steps[-1]), float(inventory[-1]), theme.accent, radius=3.5)
        plot.tag(float(inventory[-1]), format_number(inventory[-1], compact=True), theme.accent)
        self.time_axis(plot, view)


class _Sales(Panel):
    """Shares sold in each step, as columns, with the schedule's sales behind them."""

    def sources(self) -> Mapping[str, Source]:
        return {"inventory": "inventory"}

    def context(self) -> Mapping[str, Source]:
        return {"plan": lambda env: _base(env).almgren_chriss_trajectory()}

    def draw(self, canvas: Canvas, box: Box, view: View) -> None:
        theme, s = view.theme, view.scale
        items: list[tuple[str, ColorLike, Swatch]] = [
            ("Sold", theme.accent, "square"),
            ("Schedule", theme.neutral, "area"),
        ]
        plot_box = self.frame(canvas, box, view, items)
        plan = np.asarray(view.context("plan"), dtype=np.float64)
        planned = plan[:-1] - plan[1:]
        steps, inventory = view.steps, view.values("inventory")
        sold = -np.diff(inventory)
        top = max(float(np.max(planned, initial=0.0)), float(np.nanmax(sold, initial=0.0)))
        top = top * 1.1 or 1.0
        plot = Plot(canvas, plot_box, view.x_domain, (0.0, top), theme, s)
        ticks = nice_ticks(0.0, top, tick_count(plot_box.height, s))
        plot.grid(ticks, tick_labels(ticks, compact=True))
        plot.hline(0.0, theme.axis)
        step_px = plot_box.width / max(view.x_domain[1] - view.x_domain[0], 1.0)
        half = min(step_px * 0.32, 12 * s)
        floor = plot_box.bottom
        for k, amount in enumerate(planned):  # columns over the step from k to k + 1
            centre = float(plot.x(k + 0.5))
            canvas.rect(
                centre - half,
                floor,
                centre + half,
                float(plot.y(amount)),
                theme.neutral,
                alpha=0.3,
                radius=min(3 * s, half),
            )
        for i in range(1, steps.size):
            if not (math.isfinite(sold[i - 1]) and sold[i - 1] > 0):
                continue
            centre = float(plot.x(steps[i] - 0.5))
            top_px = float(plot.y(sold[i - 1]))
            r = min(3 * s, half * 0.8, max(top_px - floor, 0) / 2)
            canvas.rect(
                centre - half * 0.62, floor, centre + half * 0.62, top_px, theme.accent, radius=r
            )
            canvas.rect(
                centre - half * 0.62,
                floor,
                centre + half * 0.62,
                min(floor + r, top_px),
                theme.accent,
            )
        self.time_axis(plot, view)


class _ExecutionPrice(PriceChart):
    """The mid price, with each step's average execution price (from the shortfall)."""

    def sources(self) -> Mapping[str, Source]:
        return {**super().sources(), "inventory": "inventory", "shortfall": "shortfall"}

    def fills(self, view: View) -> dict[str, list[tuple[float, float]]]:
        inventory, shortfall = view.values("inventory"), view.values("shortfall")
        s0 = float(_base(view.env).s0)
        sells = []
        for i in range(1, view.steps.size):
            shares = inventory[i - 1] - inventory[i]
            if shares > 0:  # shortfall grows by shares * (s0 - execution price)
                price = s0 - (shortfall[i] - shortfall[i - 1]) / shares
                sells.append((float(view.steps[i]), float(price)))
        return {"buys": [], "sells": sells}

    def _legend(self, view: View) -> list[tuple[str, ColorLike, Swatch]]:
        return [
            ("Arrival price", view.theme.neutral, "dash"),
            ("Executions", view.theme.down, "down"),
        ]


class OptimalExecutionRenderer(_EnvRenderer):
    """Draws `OptimalExecution-v0` (`OptimalExecutionEnv`).

    The shares still to sell against the Almgren-Chriss optimal schedule (dashed, over
    the whole horizon), the shares sold each step against the schedule's, and the mid
    price with the average price of each step's execution and the arrival price.
    """

    draws = OptimalExecutionEnv

    def __init__(
        self,
        *,
        width: int = 720,
        height: int = 450,
        theme: Theme | str = "light",
        window: int | None = None,
    ) -> None:
        super().__init__(
            _Holdings("Inventory", height=2.4),
            _Sales("Shares sold per step", height=1.6),
            _ExecutionPrice("mid", reference="s0", title="Mid price", height=2),
            subtitle="Almgren\u2013Chriss liquidation",
            stats=[
                Stat(
                    "Shortfall",
                    lambda env: 1e4 * _base(env).shortfall / (_base(env).X0 * _base(env).s0),
                    "{:.1f} bp",
                    higher_is_better=False,
                ),
                Stat(
                    "Sold",
                    lambda env: 1 - _base(env).inventory / _base(env).X0,
                    lambda v: format_number(v, 0, percent=True),
                ),
                Stat("Mid", "mid", "{:,.3f}"),
            ],
            width=width,
            height=height,
            theme=theme,
            window=window,
        )


# ----------------------------------------------------------------------
# LimitOrderBook-v0 and LimitOrderBookMarket
# ----------------------------------------------------------------------


def _ticks(value: Any) -> str:
    if value is None:
        return "\u2014"
    return f"{value} tick" if value == 1 else f"{value} ticks"


def _best(side: str) -> Source:
    def price(env: Any) -> float | None:
        value = getattr(_base(env).book, side)()
        return None if value is None else float(value)

    return price


class _TraderPrice(PriceChart):
    """The book's mid price and spread, the agent's quotes, and its fills.

    Fills are read from the agent's inventory and cash: a step in which inventory
    rose is a buy at the average price paid, one in which it fell a sale.
    """

    def sources(self) -> Mapping[str, Source]:
        return {
            **super().sources(),
            "inventory": lambda env: _base(env).trader.inventory,
            "cash": lambda env: _base(env).trader.cash,
        }

    def fills(self, view: View) -> dict[str, list[tuple[float, float]]]:
        inventory, cash = view.values("inventory"), view.values("cash")
        out: dict[str, list[tuple[float, float]]] = {"buys": [], "sells": []}
        for i in range(1, view.steps.size):
            traded = inventory[i] - inventory[i - 1]
            if traded != 0 and math.isfinite(traded):
                price = abs((cash[i] - cash[i - 1]) / traded)
                out["buys" if traded > 0 else "sells"].append((float(view.steps[i]), price))
        return out


def _session_left(env: Any) -> float:
    core = _base(env).core
    return max(1.0 - float(core.elapsed) / float(core.session_length), 0.0)


class LimitOrderBookRenderer(_EnvRenderer):
    """Draws `LimitOrderBook-v0` (`LimitOrderBookEnv`).

    The mid price with the market's spread and the agent's fills; its inventory and
    wealth; and the order book as a depth ladder in which the agent's resting orders
    are highlighted at their place in the queue.
    """

    draws = LimitOrderBookEnv

    def __init__(
        self,
        *,
        levels: int = 8,
        width: int = 720,
        height: int = 450,
        theme: Theme | str = "light",
        window: int | None = None,
    ) -> None:
        price = _TraderPrice(
            lambda env: _base(env).core.mid(),
            spread=(_best("best_bid"), _best("best_ask")),
            title="Mid price",
            height=3,
        )
        inventory = LineChart(
            Series(
                lambda env: _base(env).trader.inventory,
                color="text_muted",
                width=1.25,
                step=True,
                signed=True,
            ),
            title="Inventory",
            baseline=0.0,
            height=1.5,
        )
        wealth = LineChart(
            Series(
                lambda env: _base(env).trader.wealth(_base(env).core.mid()),
                color="accent",
                fill=True,
            ),
            title="Wealth (ticks)",
            baseline=0.0,
            height=1.5,
        )
        book = OrderBookChart(
            "book", owners={LimitOrderBookEnv.AGENT: ("You", "highlight")}, levels=levels
        )
        super().__init__(
            Row(Column(price, inventory, wealth), book, widths=(2.2, 1)),
            subtitle="Zero-intelligence order flow",
            stats=[
                Stat(
                    "Wealth",
                    lambda env: _base(env).trader.wealth(_base(env).core.mid()),
                    lambda v: format_number(v, 1, sign=True),
                ),
                Stat(
                    "Inventory",
                    lambda env: _base(env).trader.inventory,
                    lambda v: format_number(v, 0, sign=True),
                ),
                Stat(
                    "Spread",
                    lambda env: _base(env).book.spread(),
                    _ticks,
                ),
                Stat("Session left", _session_left, lambda v: format_number(v, 0, percent=True)),
            ],
            width=width,
            height=height,
            theme=theme,
            window=window,
        )


def _trader(agent: str, quantity: str) -> Source:
    def value(env: Any) -> float:
        trader = env.traders[agent]
        if quantity == "wealth":
            return float(trader.wealth(env.core.mid()))
        return float(trader.inventory)

    return value


class LimitOrderBookMarketRenderer(Dashboard):
    """Draws `LimitOrderBookMarket`, several traders in one order book.

    The mid price and spread over the session, each trader's inventory and wealth (one
    colour per trader, in every chart), and the order book with each trader's resting
    orders highlighted in its colour. The time axis is session time, since traders act
    on their own clocks. Render it directly (`renderer.render(env)` after each step):
    `RecordFrames` wraps single-agent environments only.
    """

    def __init__(
        self,
        *,
        levels: int = 8,
        width: int = 720,
        height: int = 450,
        theme: Theme | str = "light",
        window: int | None = None,
    ) -> None:
        super().__init__(
            width=width,
            height=height,
            theme=theme,
            window=window,
            step=lambda env: env.core.elapsed,
            horizon=lambda env: env.core.session_length,
            subtitle="Zero-intelligence order flow",
        )
        self.levels = levels
        self._agents: tuple[str, ...] | None = None

    def check(self, env: Any) -> None:
        if not isinstance(env, LimitOrderBookMarket):
            raise TypeError(
                f"{type(self).__name__} draws LimitOrderBookMarket, got {type(env).__name__}."
            )
        agents = tuple(env.possible_agents)
        if agents != self._agents:
            self._agents = agents
            self._configure(env, agents)

    def _configure(self, env: LimitOrderBookMarket, agents: Sequence[str]) -> None:
        inventories = [
            Series(_trader(a, "inventory"), label=a, color=i, step=True)
            for i, a in enumerate(agents)
        ]
        wealths = [Series(_trader(a, "wealth"), label=a, color=i) for i, a in enumerate(agents)]
        price = PriceChart(
            lambda e: e.core.mid(),
            spread=(_best("best_bid"), _best("best_ask")),
            title="Mid price",
            height=3,
        )
        owners = {env.traders[a].owner: (a, i) for i, a in enumerate(agents)}
        self.set_layout(
            Row(
                Column(
                    price,
                    LineChart(*inventories, title="Inventory", baseline=0.0, tag=False, height=1.5),
                    LineChart(
                        *wealths, title="Wealth (ticks)", baseline=0.0, tag=False, height=1.5
                    ),
                ),
                OrderBookChart("book", owners=owners, levels=self.levels),
                widths=(2.2, 1),
            )
        )
        self.stats = (
            Stat("Traders", lambda e: len(e.possible_agents), "{}"),
            Stat("Spread", lambda e: e.book.spread(), _ticks),
            Stat(
                "Session left",
                lambda e: max(1 - e.core.elapsed / e.core.session_length, 0),
                lambda v: format_number(v, 0, percent=True),
            ),
        )
