"""Charts for financial-market environments, to compose into a `sekai.rendering.Dashboard`.

- `PriceChart`: a price over time with the agent's position shaded behind it, its
  trades marked, its quotes, the market's spread, and the history before the episode.
- `OrderBookChart`: a limit order book as a depth ladder, with the agent's resting
  orders shown at their place in each queue.
- `UniverseChart`: a changing universe over time, one row per slot: when each asset
  was tradable and how much of it was held.

They read the environment through sources (attribute paths or functions of the env)
like every dashboard panel, so they draw your own environments as well as the
built-in ones, whose renderers (`sekai.envs.finance.TradingRenderer`, ...) are built
from them:

    from sekai.rendering import Dashboard, LineChart, Series, Stat
    from sekai.finance.rendering import PriceChart

    renderer = Dashboard(
        PriceChart("price", position="position"),
        LineChart(Series("equity", label="Strategy", fill=True), baseline=1.0),
        stats=[Stat("Equity", "equity", "{:.4f}", delta="relative")],
        heights=(3, 2),
    )
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sekai.finance.lob import OrderBook, Side
from sekai.rendering.canvas import Canvas
from sekai.rendering.charts import (
    Box,
    Plot,
    Swatch,
    decimals_for,
    format_number,
    nice_step,
    nice_ticks,
    tick_labels,
    value_range,
)
from sekai.rendering.dashboard import ColorSpec, Panel, Source, View, resolve_color, tick_count
from sekai.rendering.text import text_width
from sekai.rendering.theme import ColorLike, mix

__all__ = ["OrderBookChart", "PriceChart", "UniverseChart"]


def _prices(value: Any) -> list[float]:
    """Trade prices from a source value: a number, a sequence of numbers, or None."""
    if value is None:
        return []
    values = np.atleast_1d(np.asarray(value, dtype=np.float64)).ravel()
    return [float(v) for v in values if math.isfinite(v)]


def _held(steps: NDArray[np.float64], values: NDArray[np.float64]) -> tuple[Any, Any]:
    """Points that hold each value over the step that just ended: (steps[i-1], steps[i]]."""
    if steps.size < 2:
        return steps, values
    return np.append(steps[:-1], steps[-1]), np.append(values[1:], values[-1])


class PriceChart(Panel):
    """A price over time, with the agent's position, trades and quotes.

    Every argument after `price` is optional:

    Args:
        price: The price per step (a mid price, for market makers).
        position: The position held over the step that just ended, signed (long > 0,
            short < 0): the background is shaded in the `up` colour while long and
            `down` while short, deeper for larger positions, and every change is
            marked as a trade (a triangle at the price it was decided at).
        quotes: `(bid, ask)` sources: the agent's own quotes over the step that just
            ended, drawn as thin lines below and above the price.
        spread: `(bid, ask)` sources: the market's best bid and ask, drawn as a band.
        buys: Prices at which the agent bought during the step that just ended (a
            number, a sequence of numbers, or None).
        sells: Likewise for sales.
        context: The prices before the episode, ending with the price at its start
            (e.g. a lookback window), read once per episode and drawn muted.
        reference: A price level drawn as a dashed line (an arrival or entry price, a
            strike): a number or a source.
        reference_label: Legend label for the reference level.
        title: Panel title.
        tag: Show the latest price in a tag on the y-axis.
        min_trade: The smallest change in `position` marked as a trade.
        width: Relative width in its row.
        height: Relative height in its column.
    """

    def __init__(
        self,
        price: Source,
        *,
        position: Source | None = None,
        quotes: tuple[Source, Source] | None = None,
        spread: tuple[Source, Source] | None = None,
        buys: Source | None = None,
        sells: Source | None = None,
        context: Source | None = None,
        reference: float | Source | None = None,
        reference_label: str | None = None,
        title: str | None = "Price",
        tag: bool = True,
        min_trade: float = 1e-6,
        width: float = 1.0,
        height: float = 1.0,
    ) -> None:
        super().__init__(title, width=width, height=height)
        self.price = price
        self.position = position
        self.quotes = quotes
        self.spread = spread
        self.buys = buys
        self.sells = sells
        self.price_context = context
        self.reference = reference
        self.reference_label = reference_label
        self.tag = tag
        self.min_trade = min_trade

    def sources(self) -> Mapping[str, Source]:
        out: dict[str, Source] = {"price": self.price}
        if self.position is not None:
            out["position"] = self.position
        if self.quotes is not None:
            out["quote_bid"], out["quote_ask"] = self.quotes
        if self.spread is not None:
            out["best_bid"], out["best_ask"] = self.spread
        if self.buys is not None:
            out["buys"] = self.buys
        if self.sells is not None:
            out["sells"] = self.sells
        if self.reference is not None and not isinstance(self.reference, (int, float)):
            out["reference"] = self.reference
        return out

    def context(self) -> Mapping[str, Source]:
        return {} if self.price_context is None else {"context": self.price_context}

    def _context(self, view: View) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        values = view.context("context")
        if values is None:
            return np.zeros(0), np.zeros(0)
        ys = np.asarray(values, dtype=np.float64).ravel()
        if ys.size < 2:
            return np.zeros(0), np.zeros(0)
        first = float(view.steps[0])
        return np.arange(first - ys.size + 1, first + 1, dtype=np.float64), ys

    def x_start(self, view: View) -> float | None:
        xs, _ = self._context(view)
        return float(xs[0]) if xs.size else None

    def _legend(self, view: View) -> list[tuple[str, ColorLike, Swatch]]:
        theme = view.theme
        items: list[tuple[str, ColorLike, Swatch]] = []
        if self.reference is not None and self.reference_label:
            items.append((self.reference_label, theme.neutral, "dash"))
        if self.spread is not None:
            items.append(("Spread", theme.neutral, "area"))
        if self.quotes is not None:
            items += [("Bid", theme.up, "line"), ("Ask", theme.down, "line")]
        if self.position is not None:
            items += [("Long", theme.up, "area"), ("Short", theme.down, "area")]
        if self.position is not None or self.buys is not None or self.sells is not None:
            items += [("Buy", theme.up, "up"), ("Sell", theme.down, "down")]
        return items

    def fills(self, view: View) -> dict[str, list[tuple[float, float]]]:
        """The agent's trades as `{"buys": [(step, price), ...], "sells": [...]}`.

        By default from the `buys` and `sells` sources; override to derive them from
        other recorded state.
        """
        return {
            key: [(float(x), p) for x, raw in zip(view.steps, view.raw(key)) for p in _prices(raw)]
            for key in ("buys", "sells")
        }

    def _reference(self, view: View) -> float | None:
        if self.reference is None:
            return None
        if isinstance(self.reference, (int, float)):
            return float(self.reference)
        value = view.latest("reference")
        return None if value is None else float(value)

    def draw(self, canvas: Canvas, box: Box, view: View) -> None:
        theme, s = view.theme, view.scale
        steps = view.steps
        price = view.values("price")
        ctx_x, ctx_y = self._context(view)
        lo, hi = view.x_domain
        visible = (ctx_x >= lo) & (ctx_x <= hi)
        extra = [ctx_y[visible]]
        for key in ("quote_bid", "quote_ask", "best_bid", "best_ask"):
            if view.raw(key):
                extra.append(view.values(key))
        fills = self.fills(view)
        extra += [np.array([p for _, p in fills.get(k, [])]) for k in ("buys", "sells")]
        reference = self._reference(view)
        if reference is not None:
            extra.append(np.array([reference]))
        plot_box = self.frame(canvas, box, view, self._legend(view))
        y_domain = value_range(price, *extra)
        plot = Plot(canvas, plot_box, view.x_domain, y_domain, theme, s)
        ticks = nice_ticks(*y_domain, tick_count(plot_box.height, s))
        labels = tick_labels(ticks)

        if self.position is not None:
            self._shade_position(plot, view, steps, view.values("position"))
        plot.grid(ticks, labels, avoid=[price[-1]] if self.tag and price.size else [])
        if reference is not None:
            plot.hline(reference, theme.neutral, width=1.25, dash=(5, 3))
        if self.spread is not None:
            plot.area(
                steps,
                view.values("best_ask"),
                theme.neutral,
                base=view.values("best_bid"),
                alpha=0.18,
            )
        if ctx_x.size:
            plot.line(ctx_x, ctx_y, theme.text_muted, width=1.5)
            plot.vline(float(steps[0]), theme.axis)
        if self.quotes is not None:
            for key, color in (("quote_bid", theme.up), ("quote_ask", theme.down)):
                xs, ys = _held(steps, view.values(key))
                plot.line(xs, ys, color, width=1.25, step=True, alpha=0.85)
        plot.line(steps, price, theme.price, width=1.75)

        # Trades: triangles under the line for buys, over it for sells. Only those in view,
        # and none once they would crowd closer than 6 px apart: the shading still shows
        # every change of position, and a carpet of triangles would hide the price.
        trades: list[tuple[float, float, bool]] = []
        if self.position is not None:
            position = view.values("position")
            change = np.diff(np.nan_to_num(position, nan=0.0))
            for i in np.flatnonzero(np.abs(change) > self.min_trade):
                trades.append((float(steps[i]), float(price[i]), bool(change[i] > 0)))
        marks = [
            (x - 0.5, p, up)
            for key, up in (("buys", True), ("sells", False))
            for x, p in fills.get(key, [])
        ]
        trades = [t for t in trades if lo <= t[0] <= hi]
        marks = [m for m in marks if lo <= m[0] <= hi]
        if len(trades) + len(marks) <= plot_box.width / (6 * s):
            for x, y, buy in trades:
                self._trade(plot, x, y, buy, 7.0)
            for x, p, up in marks:
                color = theme.up if up else theme.down
                plot.marker(x, p, color, shape="up" if up else "down", radius=3.5)
        if price.size and math.isfinite(price[-1]):
            plot.marker(float(steps[-1]), float(price[-1]), theme.price, radius=3.5)
            if self.tag:
                step = float(np.min(np.diff(ticks))) if ticks.size > 1 else 1.0
                decimals = min(decimals_for(step) + 1, 4)
                plot.tag(float(price[-1]), format_number(price[-1], decimals), theme.price)
        self.time_axis(plot, view)

    def _trade(self, plot: Plot, x: float, y: float, buy: bool, offset: float) -> None:
        theme = plot.theme
        if not (math.isfinite(x) and math.isfinite(y)):
            return
        # Offset in pixels, converted back to data units: the line stays visible.
        y0, y1 = plot.y_domain
        dy = offset * plot.scale / plot.box.height * (y1 - y0)
        plot.marker(
            x,
            y - dy if buy else y + dy,
            theme.up if buy else theme.down,
            shape="up" if buy else "down",
            radius=3.5,
        )

    def _shade_position(
        self, plot: Plot, view: View, steps: NDArray[np.float64], position: NDArray[np.float64]
    ) -> None:
        theme = view.theme
        if steps.size < 2:
            return
        # One RGBA strip, a pixel per column, stretched over the plot's height: each column
        # takes the position held over the step under its centre. One draw call, however
        # many times the position changed.
        box = plot.box
        columns = max(round(box.width), 1)
        x0, x1 = plot.x_domain
        centres = x0 + (np.arange(columns) + 0.5) / columns * (x1 - x0)
        interval = np.searchsorted(steps, centres, side="right") - 1
        inside = (interval >= 0) & (interval < steps.size - 1)
        held = np.nan_to_num(position[1:], nan=0.0)
        level = np.where(inside, held[np.clip(interval, 0, steps.size - 2)], 0.0)
        strip = np.zeros((1, columns, 4), dtype=np.uint8)
        strip[0, :, :3] = np.where((level > 0)[:, None], theme.up, theme.down)
        alpha = np.where(np.abs(level) > 1e-9, 0.05 + 0.07 * np.minimum(np.abs(level), 1), 0)
        strip[0, :, 3] = np.round(alpha * 255).astype(np.uint8)
        plot.canvas.image(strip, box.left, box.bottom, box.right, box.top)


class OrderBookChart(Panel):
    """A limit order book as a depth ladder around the mid price.

    Asks are above the mid and bids below, one row per price tick, each with a bar for
    the resting volume. The orders of highlighted owners are drawn inside the bars at
    their place in the queue (the volume ahead of them on the left), so you see how
    close an order is to being filled.

    Args:
        book: A source of the `sekai.finance.OrderBook`.
        owners: Owners whose orders to highlight: `{owner_id: (label, colour)}`, or a
            source of such a mapping. The colour is a theme role, hex, RGB or slot.
        levels: Price ticks shown on each side of the mid.
        title: Panel title.
        width: Relative width in its row.
        height: Relative height in its column.
    """

    timeseries = False

    def __init__(
        self,
        book: Source,
        *,
        owners: Mapping[int, tuple[str, ColorSpec]] | Source | None = None,
        levels: int = 8,
        title: str | None = "Order book",
        width: float = 1.0,
        height: float = 1.0,
    ) -> None:
        super().__init__(title, width=width, height=height)
        if levels < 1:
            raise ValueError("levels must be >= 1.")
        self.book = book
        self.owners = owners
        self.levels = levels

    def sources(self) -> Mapping[str, Source]:
        out: dict[str, Source] = {"book": self.book}
        if self.owners is not None and not isinstance(self.owners, Mapping):
            out["owners"] = self.owners
        return out

    def draw(self, canvas: Canvas, box: Box, view: View) -> None:
        theme, s = view.theme, view.scale
        book = view.latest("book")
        if not isinstance(book, OrderBook):
            raise TypeError(f"OrderBookChart needs an OrderBook, got {type(book).__name__}.")
        owners = self.owners if isinstance(self.owners, Mapping) else view.latest("owners") or {}
        highlight = {
            int(owner): (label, resolve_color(theme, color, i))
            for i, (owner, (label, color)) in enumerate(owners.items())
        }
        items: list[tuple[str, ColorLike, Swatch]] = [
            ("Bids", theme.up, "square"),
            ("Asks", theme.down, "square"),
        ]
        items += [(label, color, "square") for label, color in highlight.values()]
        area = self.frame(canvas, box, view, items)
        mid = book.mid()
        if mid is None:
            mid = float(book.last_price) if book.last_price is not None else None
        if mid is None:
            canvas.text(
                area.left, area.top - 10 * s, "The book is empty.", theme.text_muted, size=10 * s
            )
            return
        asks = [math.floor(mid) + 1 + k for k in range(self.levels)][::-1]
        bids = [math.ceil(mid) - 1 - k for k in range(self.levels)]
        prices = [p for p in asks + bids if p >= 1]
        volumes = [book.volume_at(Side.ASK if p > mid else Side.BID, p) for p in prices]
        top_volume = max(volumes, default=0)
        full = nice_step(top_volume, 1) if top_volume > 0 else 1.0
        gap = 18 * s  # the mid row
        row_h = (area.height - gap) / max(len(prices), 1)
        bar_h = min(max(row_h - 3 * s, 2 * s), 14 * s)
        label_w = max(text_width(format_number(p, 0), 10 * s) for p in prices) + 10 * s
        value_w = text_width("0000", 9 * s) + 6 * s
        left, right = area.left + label_w, area.right - value_w
        best_bid, best_ask = book.best_bid(), book.best_ask()
        orders = {owner: book.orders_of(owner) for owner in highlight}
        y = area.top
        for p, volume in zip(prices, volumes):
            if p < mid and y > area.top - 1 - len(asks) * row_h - gap / 2:
                y -= gap  # leave the mid row between asks and bids
                # The mid row: price and spread.
                spread = book.spread()
                text = f"mid {format_number(mid, 1)}"
                if spread is not None:
                    text += f"  ·  spread {spread}"
                canvas.text(
                    left, y + gap / 2, text, theme.text_muted, size=9.5 * s, valign="middle"
                )
            centre = y - row_h / 2
            is_ask = p > mid
            best = p in (best_bid, best_ask)
            canvas.text(
                left - 8 * s,
                centre,
                format_number(p, 0),
                theme.text if best else theme.text_muted,
                size=10 * s,
                bold=best,
                align="right",
                valign="middle",
            )
            if volume > 0:
                end = left + min(volume / full, 1.0) * (right - left)
                color = theme.down if is_ask else theme.up
                lo, hi = centre - bar_h / 2, centre + bar_h / 2
                r = min(3 * s, bar_h / 2, (end - left) / 2)
                canvas.rect(left, lo, end, hi, mix(color, theme.background, 0.15), radius=r)
                canvas.rect(left, lo, min(left + r, end), hi, mix(color, theme.background, 0.15))
                canvas.text(
                    end + 4 * s, centre, str(volume), theme.text_muted, size=9 * s, valign="middle"
                )
                for owner, own in orders.items():
                    for order in own:
                        if order.price != p or order.quantity <= 0:
                            continue
                        ahead = book.queue_ahead(order.order_id)
                        a = left + min(ahead / full, 1.0) * (right - left)
                        b = left + min((ahead + order.quantity) / full, 1.0) * (right - left)
                        mark = highlight[owner][1]
                        b = max(b, a + 3 * s)
                        canvas.rect(
                            a - 1.5 * s,
                            lo - 3 * s,
                            b + 1.5 * s,
                            hi + 3 * s,
                            theme.background,
                            radius=3 * s,
                        )
                        canvas.rect(a, lo - 1.5 * s, b, hi + 1.5 * s, mark, radius=2 * s)
            y -= row_h


class UniverseChart(Panel):
    """A changing universe over time: one row per slot, one column per step.

    A cell is filled with the slot's colour while its asset is held (deeper for a
    larger weight), drawn in the grid colour while the asset is tradable but not held,
    and left empty while the slot has no tradable asset, so listings, delistings and
    rotations show as the rows fill and empty.

    Args:
        weights: A source of the weight of each slot.
        available: A source of whether each slot is tradable now.
        labels: Slot labels (asset names), a sequence or a source; shown in the gutter.
        color: The colour of held cells (theme role, hex, RGB or slot).
        negative_color: The colour of cells held short.
        title: Panel title.
        width: Relative width in its row.
        height: Relative height in its column.
    """

    def __init__(
        self,
        weights: Source,
        *,
        available: Source | None = None,
        labels: Sequence[str] | Source | None = None,
        color: ColorSpec = "accent",
        negative_color: ColorSpec = "down",
        title: str | None = "Universe",
        width: float = 1.0,
        height: float = 1.0,
    ) -> None:
        super().__init__(title, width=width, height=height)
        self.weights = weights
        self.available = available
        self.labels = labels
        self.color = color
        self.negative_color = negative_color

    def sources(self) -> Mapping[str, Source]:
        out: dict[str, Source] = {"weights": self.weights}
        if self.available is not None:
            out["available"] = self.available
        if self.labels is not None and not isinstance(self.labels, (list, tuple)):
            out["labels"] = self.labels  # type: ignore[assignment]
        return out

    def draw(self, canvas: Canvas, box: Box, view: View) -> None:
        theme, s = view.theme, view.scale
        weights = view.values("weights")
        if weights.ndim == 1:
            weights = weights[:, None]
        n, k = weights.shape if weights.size else (0, 0)
        area = self.frame(canvas, box, view)
        if n == 0 or k == 0:
            return
        available: NDArray[np.bool_] = np.isfinite(weights)
        if self.available is not None:
            flags = np.nan_to_num(view.values("available"), nan=0.0)
            if flags.ndim == 1:
                flags = flags[:, None]
            if flags.shape[1] >= k:
                available = flags[:, :k] > 0.5
        signed = np.nan_to_num(weights, nan=0.0)
        held = np.abs(signed)
        reference = nice_step(float(held.max(initial=0.0)), 1) if held.max(initial=0.0) > 0 else 1
        steps = view.steps.astype(np.int64)
        first, last = int(steps[0]), int(steps[-1])
        columns = last - first + 1
        image = np.zeros((k, columns, 4), dtype=np.uint8)
        cols = steps - first
        empty = (*theme.grid, 255)
        back = np.asarray(theme.background, dtype=np.float64)
        long = np.asarray(view.color(self.color), dtype=np.float64)
        short = np.asarray(view.color(self.negative_color), dtype=np.float64)
        for j in range(k):
            color = np.where(signed[:, j, None] < 0, short, long)
            depth = 0.25 + 0.75 * np.clip(held[:, j] / reference, 0, 1)
            cells = back + (color - back) * depth[:, None]
            image[j, cols, :3] = np.where(held[:, j, None] > 1e-9, cells, empty[:3]).astype(
                np.uint8
            )
            image[j, cols, 3] = np.where(available[:, j], 255, 0)
        plot = Plot(canvas, area, view.x_domain, (0, k), theme, s)
        x0, x1 = float(plot.x(first - 0.5)), float(plot.x(last + 0.5))
        with canvas.clip(area.left, area.bottom, area.right, area.top):
            canvas.image(image, max(x0, area.left - 1), area.bottom, x1, area.top)
            row_h = area.height / k
            for j in range(1, k):  # a surface gap between rows
                yy = area.top - j * row_h
                canvas.rect(area.left, yy - 0.75 * s, area.right, yy + 0.75 * s, theme.background)
        if isinstance(self.labels, (list, tuple)):
            labels = [str(v) for v in self.labels]
        elif self.labels is not None:
            labels = ["" if v is None else str(v) for v in (view.latest("labels") or [])]
        else:
            labels = [str(j) for j in range(k)]
        row_h = area.height / k
        size = min(9.5, row_h / s * 0.8)
        if size >= 7:
            for j, label in enumerate(labels[:k]):
                canvas.text(
                    area.right + 6 * s,
                    area.top - (j + 0.5) * row_h,
                    label or "\u2014",
                    theme.text_muted,
                    size=size * s,
                    valign="middle",
                )
        self.time_axis(plot, view)
