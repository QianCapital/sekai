"""Chart drawing on a `Canvas`: layout boxes, nice ticks, number formats and plot marks.

These are the pieces sekai's dashboards (`sekai.rendering.Dashboard`) and finance
renderers are drawn with, and they are public so a custom renderer can draw charts in
the same style:

    canvas = Canvas(640, 360, LIGHT.background, antialias=True)
    plot = Plot(canvas, Box(16, 24, 580, 340), (0, 100), (0.9, 1.2), LIGHT)
    plot.grid(nice_ticks(0.9, 1.2))
    plot.line(steps, equity, LIGHT.accent)

Marks follow fixed specs: 2 px lines with round caps and joins, hairline (1 px)
solid gridlines one step off the surface, area fills as a 10% wash of the series
colour, end markers with a 2 px ring in the surface colour, and text in the ink
colours rather than series colours. Sizes scale with `scale`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sekai.rendering.canvas import Canvas
from sekai.rendering.text import text_width
from sekai.rendering.theme import Color, ColorLike, Theme, rgb

MINUS = "\u2212"
"""The typographic minus sign, used for negative numbers."""

Marker = Literal["circle", "up", "down", "diamond"]
Swatch = Literal["line", "dash", "area", "dot", "square", "up", "down"]


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    """An axis-aligned rectangle in canvas coordinates (origin bottom-left, y up)."""

    left: float
    bottom: float
    right: float
    top: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.top - self.bottom

    def inset(
        self, left: float = 0.0, bottom: float = 0.0, right: float = 0.0, top: float = 0.0
    ) -> Box:
        """The box shrunk by the given margins."""
        return Box(self.left + left, self.bottom + bottom, self.right - right, self.top - top)

    def rows(self, weights: Sequence[float], gap: float = 0.0) -> list[Box]:
        """Split into stacked boxes, top first, sized by `weights`, `gap` apart."""
        edges = _split(self.height, weights, gap)
        return [Box(self.left, self.top - b, self.right, self.top - a) for a, b in edges]

    def columns(self, weights: Sequence[float], gap: float = 0.0) -> list[Box]:
        """Split into side-by-side boxes, left first, sized by `weights`, `gap` apart."""
        edges = _split(self.width, weights, gap)
        return [Box(self.left + a, self.bottom, self.left + b, self.top) for a, b in edges]

    def snapped(self) -> Box:
        """The box with its edges on whole pixels (for crisp hairlines and fills)."""
        return Box(round(self.left), round(self.bottom), round(self.right), round(self.top))


def _split(length: float, weights: Sequence[float], gap: float) -> list[tuple[float, float]]:
    total = float(sum(weights))
    if total <= 0 or not weights:
        raise ValueError("Layout weights must be positive.")
    usable = max(length - gap * (len(weights) - 1), 0.0)
    edges, start = [], 0.0
    for w in weights:
        end = start + usable * w / total
        edges.append((start, end))
        start = end + gap
    return edges


# ----------------------------------------------------------------------
# Ticks and number formats
# ----------------------------------------------------------------------


def nice_step(span: float, count: int = 4, *, integer: bool = False) -> float:
    """A round step (1, 2, 2.5 or 5 times a power of ten) giving about `count` intervals.

    `integer=True` keeps the step a whole number (1, 2, 5, 10, ...), for counts.
    """
    raw = span / max(count, 1)
    if not (math.isfinite(raw) and raw > 0):
        return 1.0
    if integer and raw <= 1:
        return 1.0
    magnitude = 10.0 ** math.floor(math.log10(raw))
    steps = (1.0, 2.0, 5.0, 10.0) if integer else (1.0, 2.0, 2.5, 5.0, 10.0)
    for m in steps:
        if m * magnitude >= raw * (1 - 1e-9):
            return m * magnitude
    return 10 * magnitude


def nice_ticks(
    low: float, high: float, count: int = 4, *, integer: bool = False
) -> NDArray[np.float64]:
    """Round tick values inside [low, high], about `count` intervals apart.

    `integer=True` places ticks on whole numbers only.
    """
    if not (math.isfinite(low) and math.isfinite(high)):
        return np.zeros(0)
    if high <= low:
        return np.array([low])
    for attempt in range(4):  # refine the step until at least two ticks are in view
        step = nice_step(high - low, count * 2**attempt, integer=integer)
        first = math.ceil(low / step - 1e-9)
        last = math.floor(high / step + 1e-9)
        if last > first:
            break
    ticks: NDArray[np.float64] = np.arange(first, last + 1, dtype=np.float64) * step
    return ticks


def decimals_for(step: float) -> int:
    """Decimal places needed to write multiples of `step` exactly (at most 8)."""
    if not (math.isfinite(step) and step > 0):
        return 0
    for d in range(9):
        scaled = step * 10**d
        if abs(scaled - round(scaled)) < 1e-6 * max(scaled, 1):
            return d
    return 8


def format_number(
    value: float,
    decimals: int = 2,
    *,
    sign: bool = False,
    percent: bool = False,
    compact: bool = False,
) -> str:
    """A number for display: thousands separators, a true minus sign, optional `+`.

    `percent=True` multiplies by 100 and appends `%`; `compact=True` writes large
    magnitudes as 12.5K, 3.4M, 1.2B. NaN and None are written as an em dash.
    """
    if value is None or not math.isfinite(value):
        return "\u2014"
    v = float(value) * (100.0 if percent else 1.0)
    suffix = "%" if percent else ""
    if compact and abs(v) >= 1000:
        for limit, letter in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
            if abs(v) >= limit:
                v, suffix = v / limit, letter + suffix
                decimals = 0 if abs(v) >= 100 else 1 if abs(v) >= 10 else 2
                break
    text = f"{abs(v):,.{max(decimals, 0)}f}"
    if float(text.replace(",", "")) == 0:
        return text + suffix  # no sign on zero
    if v < 0:
        return MINUS + text + suffix
    return ("+" if sign else "") + text + suffix


def tick_labels(
    ticks: ArrayLike, *, percent: bool = False, compact: bool | None = None
) -> list[str]:
    """Labels for evenly spaced ticks, all with the same number of decimals."""
    values = np.asarray(ticks, dtype=np.float64)
    if values.size == 0:
        return []
    step = float(np.min(np.diff(values))) if values.size > 1 else abs(float(values[0])) or 1.0
    if compact is None:
        compact = not percent and float(np.max(np.abs(values))) >= 10_000
    decimals = decimals_for(step * (100 if percent else 1))
    if compact:
        labels = [
            format_number(v, decimals, compact=True)
            if abs(v) < 1000
            else format_number(v, compact=True)
            for v in values
        ]
        if len(set(labels)) == len(labels):
            return labels
        # Ticks closer together than compact labels can tell apart (prices near 10,000).
    return [format_number(v, decimals, percent=percent) for v in values]


def value_range(
    *arrays: ArrayLike, include: Sequence[float] = (), pad: float = 0.08
) -> tuple[float, float]:
    """The finite range of some data (plus `include`), padded by `pad` of its span.

    A flat or empty range is widened so a scale can always be built from it.
    """
    parts = [np.asarray(a, dtype=np.float64).ravel() for a in arrays]
    parts.append(np.asarray(include, dtype=np.float64).ravel())
    values = np.concatenate(parts) if parts else np.zeros(0)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0, 1.0
    low, high = float(np.min(values)), float(np.max(values))
    span = high - low
    if span <= 1e-12 * max(abs(low), abs(high), 1.0):
        half = max(abs(low) * 0.01, 1e-6)
        return low - half, high + half
    return low - span * pad, high + span * pad


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------


class Plot:
    """A box of the canvas mapped to data coordinates, with methods that draw marks.

    Marks are clipped to the box; labels and value tags may extend into the space
    to its right (the y-axis gutter, where tick labels go).

    Args:
        canvas: The canvas to draw on (anti-aliased canvases look best).
        box: The plot area.
        x_domain: Data range shown across the box.
        y_domain: Data range shown from bottom to top.
        theme: Colours.
        scale: Size multiplier for line widths, markers and text (1 at 720 x 450).
    """

    def __init__(
        self,
        canvas: Canvas,
        box: Box,
        x_domain: tuple[float, float],
        y_domain: tuple[float, float],
        theme: Theme,
        scale: float = 1.0,
    ) -> None:
        self.canvas = canvas
        self.box = box
        self.theme = theme
        self.scale = scale
        x0, x1 = (float(v) for v in x_domain)
        y0, y1 = (float(v) for v in y_domain)
        self.x_domain = (x0, x1 if x1 != x0 else x0 + 1.0)
        self.y_domain = (y0, y1 if y1 != y0 else y0 + 1.0)

    def x(self, values: ArrayLike) -> NDArray[np.float64]:
        """Canvas x of data x."""
        x0, x1 = self.x_domain
        out: NDArray[np.float64] = (
            self.box.left + (np.asarray(values, dtype=np.float64) - x0) / (x1 - x0) * self.box.width
        )
        return out

    def y(self, values: ArrayLike) -> NDArray[np.float64]:
        """Canvas y of data y."""
        y0, y1 = self.y_domain
        out: NDArray[np.float64] = (
            self.box.bottom
            + (np.asarray(values, dtype=np.float64) - y0) / (y1 - y0) * self.box.height
        )
        return out

    def _clip(self) -> object:
        b = self.box
        return self.canvas.clip(b.left, b.bottom, b.right, b.top)

    # Axes -----------------------------------------------------------------

    def grid(
        self,
        ticks: ArrayLike,
        labels: Sequence[str] | None = None,
        *,
        size: float = 10,
        avoid: Sequence[float | None] = (),
    ) -> None:
        """Hairline gridlines at y `ticks`, labelled in the gutter right of the box.

        Labels within a tag's height of a data y in `avoid` are left out, so a value
        tag drawn there (`tag()`) does not cover one.
        """
        values = np.asarray(ticks, dtype=np.float64)
        labels = tick_labels(values) if labels is None else labels
        b, s = self.box, self.scale
        keep_clear = [float(self.y(v)) for v in avoid if v is not None and math.isfinite(v)]
        for value, label in zip(values, labels):
            yy = float(self.y(value))
            if not (b.bottom - 0.5 <= yy <= b.top + 0.5):
                continue
            row = math.floor(min(yy, b.top - 1))
            self.canvas.rect(b.left, row, b.right, row + 1, self.theme.grid)
            clear = (size + 9) * s
            if any(abs(yy - min(max(c, b.bottom), b.top)) < clear for c in keep_clear):
                continue
            self.canvas.text(
                b.right + 6 * s,
                row + 0.5,
                label,
                self.theme.text_muted,
                size=size * s,
                valign="middle",
            )

    def x_ticks(self, ticks: ArrayLike, labels: Sequence[str], *, size: float = 10) -> None:
        """Labels under the box at x `ticks` (no gridlines: time runs left to right)."""
        b, s = self.box, self.scale
        for value, label in zip(np.asarray(ticks, dtype=np.float64), labels):
            xx = float(self.x(value))
            if b.left - 0.5 <= xx <= b.right + 0.5:
                width = text_width(label, size * s)
                xx = min(max(xx, b.left + width / 2), b.right - width / 2)
                self.canvas.text(
                    xx,
                    b.bottom - 5 * s,
                    label,
                    self.theme.text_muted,
                    size=size * s,
                    align="center",
                    valign="top",
                )

    def hline(
        self,
        value: float,
        color: ColorLike,
        *,
        width: float = 1.0,
        dash: tuple[float, float] | None = None,
        alpha: float = 1.0,
    ) -> None:
        """A horizontal line across the box at data y `value` (a baseline or reference)."""
        yy = float(self.y(value))
        b, s = self.box, self.scale
        if not (b.bottom - 1 <= yy <= b.top + 1):
            return
        if dash is None and width <= 1.0:
            row = math.floor(min(yy, b.top - 1))
            self.canvas.rect(b.left, row, b.right, row + 1, rgb(color), alpha=alpha)
            return
        with self._clip():  # type: ignore[attr-defined]
            self.canvas.polyline(
                [(b.left, yy), (b.right, yy)],
                rgb(color),
                width * s,
                alpha=alpha,
                dash=None if dash is None else (dash[0] * s, dash[1] * s),
            )

    def vline(
        self,
        value: float,
        color: ColorLike,
        *,
        width: float = 1.0,
        alpha: float = 1.0,
        dash: tuple[float, float] | None = None,
    ) -> None:
        """A vertical line across the box at data x `value` (e.g. the start of an episode)."""
        xx = float(self.x(value))
        b, s = self.box, self.scale
        if not (b.left - 1 <= xx <= b.right + 1):
            return
        with self._clip():  # type: ignore[attr-defined]
            if dash is None and width <= 1.0:
                col = math.floor(xx)
                self.canvas.rect(col, b.bottom, col + 1, b.top, rgb(color), alpha=alpha)
            else:
                self.canvas.polyline(
                    [(xx, b.bottom), (xx, b.top)],
                    rgb(color),
                    width * s,
                    alpha=alpha,
                    dash=None if dash is None else (dash[0] * s, dash[1] * s),
                )

    # Marks -----------------------------------------------------------------

    def _points(
        self, xs: ArrayLike, ys: ArrayLike, step: bool
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        px = self.x(xs).ravel()
        py = self.y(ys).ravel()
        if step and px.size > 1:  # hold each value until the next sample
            px = np.repeat(px, 2)[1:]
            py = np.repeat(py, 2)[:-1]
            px[1::2] += 1e-9  # keep x strictly increasing for interpolation
        return px, py

    def line(
        self,
        xs: ArrayLike,
        ys: ArrayLike,
        color: ColorLike,
        *,
        width: float = 2.0,
        dash: tuple[float, float] | None = None,
        alpha: float = 1.0,
        step: bool = False,
    ) -> None:
        """A series as a line (NaN leaves a gap); `step=True` holds values between samples."""
        px, py = self._points(xs, ys, step)
        s = self.scale
        with self._clip():  # type: ignore[attr-defined]
            self.canvas.polyline(
                np.column_stack([px, py]),
                rgb(color),
                width * s,
                alpha=alpha,
                dash=None if dash is None else (dash[0] * s, dash[1] * s),
            )

    def area(
        self,
        xs: ArrayLike,
        ys: ArrayLike,
        color: ColorLike,
        *,
        base: ArrayLike | float | None = None,
        alpha: float = 0.1,
        step: bool = False,
    ) -> None:
        """Fill between a series and `base` (default: the bottom of the box)."""
        px, py = self._points(xs, ys, step)
        if base is None:
            pb: NDArray[np.float64] | float = self.box.bottom
        elif np.ndim(base) == 0:
            pb = float(self.y(float(base)))  # type: ignore[arg-type]
        else:
            pb = self._points(xs, base, step)[1]
        with self._clip():  # type: ignore[attr-defined]
            self.canvas.fill_between(px, pb, py, rgb(color), alpha=alpha)

    def shade(self, x0: float, x1: float, color: ColorLike, *, alpha: float = 0.08) -> None:
        """A vertical band over the box's full height, from data x `x0` to `x1`."""
        a, b = float(self.x(x0)), float(self.x(x1))
        with self._clip():  # type: ignore[attr-defined]
            self.canvas.rect(a, self.box.bottom, b, self.box.top, rgb(color), alpha=alpha)

    def marker(
        self,
        x: float,
        y: float,
        color: ColorLike,
        *,
        shape: Marker = "circle",
        radius: float = 4.0,
        ring: bool = True,
    ) -> None:
        """A marker (radius >= 4 px, i.e. at least 8 px across) with a surface ring."""
        px, py = float(self.x(x)), float(self.y(y))
        if not (math.isfinite(px) and math.isfinite(py)):
            return
        r = radius * self.scale
        if ring:
            _shape(self.canvas, shape, px, py, r + 2 * self.scale, self.theme.background)
        _shape(self.canvas, shape, px, py, r, rgb(color))

    def tag(self, y: float, text: str, color: ColorLike, *, size: float = 10) -> None:
        """A value tag: a pill in the gutter at data y `y`, filled with `color`."""
        yy = float(self.y(y))
        if not math.isfinite(yy):
            return
        s, b = self.scale, self.box
        yy = min(max(yy, b.bottom), b.top)
        width = text_width(text, size * s, True) + 10 * s
        height = (size + 7) * s
        fill = rgb(color)
        left = b.right + 2 * s
        self.canvas.rect(
            left, yy - height / 2, left + width, yy + height / 2, fill, radius=height / 2
        )
        self.canvas.text(
            left + 5 * s,
            yy,
            text,
            self.theme.ink_on(fill),
            size=size * s,
            bold=True,
            valign="middle",
        )


def _shape(canvas: Canvas, shape: Marker, x: float, y: float, r: float, color: Color) -> None:
    if shape == "circle":
        canvas.circle(x, y, r, color)
    elif shape == "diamond":
        canvas.polygon(
            [(x, y - r * 1.2), (x + r * 1.2, y), (x, y + r * 1.2), (x - r * 1.2, y)], color
        )
    else:  # an equilateral triangle centred on (x, y), pointing up or down
        k = 1.0 if shape == "up" else -1.0
        h = r * 1.5
        canvas.polygon(
            [(x - r * 1.25, y - k * h / 2), (x + r * 1.25, y - k * h / 2), (x, y + k * h / 2)],
            color,
        )


# ----------------------------------------------------------------------
# Labels
# ----------------------------------------------------------------------


def legend(
    canvas: Canvas,
    x: float,
    y: float,
    items: Sequence[tuple[str, ColorLike, Swatch]],
    theme: Theme,
    *,
    scale: float = 1.0,
    size: float = 10,
    align: Literal["left", "right"] = "right",
) -> float:
    """Legend entries (label, colour, swatch kind) on one line; returns its width.

    `x` is the right edge (align="right") or left edge of the legend, `y` the
    middle of its line. Labels are in secondary ink: the swatch carries identity.
    """
    s = scale
    swatch_w, gap, spacing = 12 * s, 5 * s, 12 * s
    widths = [swatch_w + gap + text_width(label, size * s) for label, _, _ in items]
    total = legend_width(items, scale=scale, size=size)
    cursor = x - total if align == "right" else x
    for (label, color, kind), width in zip(items, widths):
        c = rgb(color)
        mid = cursor + swatch_w / 2
        if kind == "line":
            canvas.polyline([(cursor, y), (cursor + swatch_w, y)], c, 2 * s)
        elif kind == "dash":
            canvas.polyline([(cursor, y), (cursor + swatch_w, y)], c, 2 * s, dash=(3 * s, 2 * s))
        elif kind == "dot":
            canvas.circle(mid, y, 3.5 * s, c)
        elif kind in ("up", "down"):
            _shape(canvas, kind, mid, y, 3.5 * s, c)
        else:  # area / square
            alpha = 0.35 if kind == "area" else 1.0
            canvas.rect(
                mid - 4 * s, y - 4 * s, mid + 4 * s, y + 4 * s, c, radius=2 * s, alpha=alpha
            )
            if kind == "area":
                canvas.polyline(
                    [(mid - 4 * s, y + 3.5 * s), (mid + 4 * s, y + 3.5 * s)], c, 1.5 * s
                )
        canvas.text(
            cursor + swatch_w + gap, y, label, theme.text_secondary, size=size * s, valign="middle"
        )
        cursor += width + spacing
    return total


def legend_width(
    items: Sequence[tuple[str, ColorLike, Swatch]], *, scale: float = 1.0, size: float = 10
) -> float:
    """The width `legend()` would draw these entries in."""
    s = scale
    widths = [17 * s + text_width(label, size * s) for label, _, _ in items]
    return sum(widths) + 12 * s * max(len(items) - 1, 0)


def triangle(canvas: Canvas, x: float, y: float, r: float, color: ColorLike, up: bool) -> None:
    """A small direction triangle (for deltas), centred on (x, y)."""
    _shape(canvas, "up" if up else "down", x, y, r, rgb(color))


__all__ = [
    "MINUS",
    "Box",
    "Plot",
    "decimals_for",
    "format_number",
    "legend",
    "legend_width",
    "nice_step",
    "nice_ticks",
    "tick_labels",
    "triangle",
    "value_range",
]
