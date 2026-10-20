"""Dashboards: renderers assembled from charts that follow an environment through an episode.

A `Dashboard` is a `Renderer`. Each call to `render(env)` reads the values its charts
need from the environment (attribute names or functions of the env), records them
for the current episode, and draws the episode so far: a header with the
environment's name and headline numbers, then charts laid out in rows and columns.
It needs nothing from the environment but readable state, so it works for any env,
and it is how sekai's finance renderers are built.

    renderer = Dashboard(
        LineChart(Series("equity", label="Strategy", fill=True), baseline=1.0),
        BarChart("weights", labels=["A", "B", "Cash"], title="Allocation"),
        title="My portfolio",
        stats=[Stat("Equity", "equity", "{:.3f}", delta="relative")],
    )
    env = RecordFrames(my_env, renderer)

**Sources.** A chart reads its data through sources: an attribute path on the
unwrapped env (`"equity"`, `"book.best_bid"`, a method is not called) or a function
`f(env)` of the env as passed to `render()` (wrapped or not).

**Episodes.** The dashboard keeps one history per environment. It reads the
episode clock from `step` (default: the env's `elapsed_steps`, found through
wrappers): a clock at 0, or one that goes backwards, starts a new history; the same
value again redraws the latest sample. Without a clock, every render is the next
step and histories never restart (call `reset()`).
"""

from __future__ import annotations

import math
import numbers
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from sekai.rendering.canvas import Canvas
from sekai.rendering.charts import (
    Box,
    Plot,
    Swatch,
    decimals_for,
    format_number,
    legend,
    legend_width,
    nice_step,
    nice_ticks,
    tick_labels,
    triangle,
    value_range,
)
from sekai.rendering.renderer import BaseRenderer
from sekai.rendering.text import text_width
from sekai.rendering.theme import Color, ColorLike, Theme, get_theme, mix, rgb

Source: TypeAlias = str | Callable[[Any], Any]
"""Where a value comes from: an attribute path on the unwrapped env, or `f(env)`."""

ColorSpec: TypeAlias = ColorLike | int | None
"""A colour: a theme role ("accent", "up", ...), hex string, RGB tuple, or palette slot."""

ROLES = ("text", "price", "accent", "neutral", "up", "down", "highlight", "text_muted")


def read(env: Any, source: Source) -> Any:
    """The value of `source` for `env`: `source(env)`, or an attribute path on the base env.

    A single attribute missing from the base env is also looked up through the wrappers
    (`env.get_attr`), so wrapper state such as `elapsed_steps` is readable.
    """
    if callable(source):
        return source(env)
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    parts = source.split(".")
    try:
        target = base
        for part in parts:
            target = getattr(target, part)
        return target
    except AttributeError:
        if len(parts) == 1 and hasattr(env, "has_attr") and env.has_attr(source):
            return env.get_attr(source)
        raise


def resolve_color(theme: Theme, spec: ColorSpec, slot: int) -> Color:
    """A concrete colour: a role name, hex, RGB, a palette slot (int), or slot `slot`."""
    if spec is None:
        return theme.color(slot)
    if isinstance(spec, bool):
        raise TypeError("A colour cannot be a bool.")
    if isinstance(spec, int):
        return theme.color(spec)
    if isinstance(spec, str) and not spec.startswith("#"):
        if spec not in ROLES:
            raise ValueError(f"Unknown colour role {spec!r}; use one of {ROLES} or a hex string.")
        color: Color = getattr(theme, spec)
        return color
    return rgb(spec)


def _positive_int(value: Any) -> int | None:
    """`value` as an int if it is a positive integer (not a bool), else None."""
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool) and value > 0:
        return int(value)
    return None


def _number(value: Any) -> float:
    """A recorded value as a float (NaN for None, a non-number or a non-finite value)."""
    if value is None:
        return math.nan
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number


# ----------------------------------------------------------------------
# Data specifications
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Series:
    """One line of a `LineChart`.

    Args:
        source: The value per step.
        label: Legend label (charts with two or more labelled series show a legend).
        color: Theme role, hex, RGB or palette slot; default: the slot of its position.
        fill: Shade the area under the line (a 10% wash, down to the chart's baseline).
        signed: Shade the area in the `up` colour above the baseline and `down` below.
        step: Hold each value until the next step (for inventories, positions).
        dashed: Draw dashed (benchmarks, plans and other references).
        width: Line width in pixels (at 720 x 450).
        rebase: Divide by the episode's first value and multiply by this (index the
            series, e.g. `rebase=100`), so differently scaled series share an axis.
    """

    source: Source
    label: str | None = None
    color: ColorSpec = None
    fill: bool = False
    signed: bool = False
    step: bool = False
    dashed: bool = False
    width: float = 2.0
    rebase: float | None = None


@dataclass(frozen=True)
class Stat:
    """A headline number in a dashboard's header.

    Args:
        label: What it is ("Equity").
        source: Its value.
        format: A format string (`"{:.2f}"`) or a function of the value returning text.
        delta: Also show its change since the episode began: "relative" (percent) or
            "absolute", with a direction triangle.
        higher_is_better: Whether a rise is good: it picks the delta's colour.
    """

    label: str
    source: Source
    format: str | Callable[[Any], str] = "{:,.2f}"
    delta: Literal["relative", "absolute"] | None = None
    higher_is_better: bool = True

    def text(self, value: Any) -> str:
        if callable(self.format):
            return str(self.format(value))
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return "\u2014"
        text = self.format.format(value)
        return text.replace("-", "\u2212") if text.startswith("-") else text


# ----------------------------------------------------------------------
# Recording
# ----------------------------------------------------------------------


class _History:
    """The recorded samples of one environment's current episode."""

    def __init__(self) -> None:
        self.steps: list[float] = []
        self.values: dict[str, list[Any]] = {}
        self.context: dict[str, Any] = {}

    def record(self, step: float, samples: Mapping[str, Any], replace: bool) -> None:
        if replace and self.steps:
            for key, value in samples.items():
                self.values.setdefault(key, [math.nan] * len(self.steps))[-1] = value
            return
        self.steps.append(step)
        for key, value in samples.items():
            self.values.setdefault(key, [math.nan] * (len(self.steps) - 1)).append(value)

    def trim(self, keep: int) -> None:
        if len(self.steps) > keep:
            cut = len(self.steps) - keep
            del self.steps[:cut]
            for series in self.values.values():
                del series[:cut]


def _snapshot(value: Any) -> Any:
    """A recorded copy of a value: arrays are copied (envs may update them in place)."""
    if isinstance(value, np.ndarray):
        return value.copy()
    if (
        isinstance(value, (list, tuple))
        and value
        and all(isinstance(v, numbers.Number) for v in value)
    ):
        return np.asarray(value, dtype=np.float64)
    return value


class View:
    """What a panel sees when it draws: the recorded episode, the time axis and the style.

    Attributes:
        env: The environment being rendered (as passed to `render`).
        theme: The colours.
        scale: Size multiplier (1 at 720 x 450).
        steps: The episode clock of each recorded sample.
        x_domain: The time axis shared by every time-series panel.
        show_x: Whether this panel draws the time axis labels (the lowest in its column).
        gutter: Width reserved right of time-series plots for y-axis labels.
    """

    def __init__(
        self,
        env: Any,
        history: _History,
        prefix: str,
        theme: Theme,
        scale: float,
        x_domain: tuple[float, float],
        show_x: bool,
        gutter: float,
    ) -> None:
        self.env = env
        self._history = history
        self._prefix = prefix
        self.theme = theme
        self.scale = scale
        self.steps: NDArray[np.float64] = np.asarray(history.steps, dtype=np.float64)
        self.x_domain = x_domain
        self.show_x = show_x
        self.gutter = gutter

    @property
    def step(self) -> float:
        """The current value of the episode clock."""
        return self._history.steps[-1]

    def raw(self, key: str) -> list[Any]:
        """Every recorded value of a source, as recorded."""
        return self._history.values.get(self._prefix + key, [])

    def latest(self, key: str) -> Any:
        """The current value of a source."""
        values = self.raw(key)
        return values[-1] if values else None

    def values(self, key: str) -> NDArray[np.float64]:
        """A source's recorded values as floats: `(n,)` for numbers, `(n, k)` for vectors.

        Missing values are NaN; vectors of different lengths are padded with NaN.
        """
        raw = self.raw(key)
        if not raw:
            return np.zeros(0)
        if not any(isinstance(v, np.ndarray) and v.ndim > 0 for v in raw):
            return np.array([_number(v) for v in raw], dtype=np.float64)
        rows = [
            np.atleast_1d(np.asarray(v, dtype=np.float64))
            if v is not None and not (isinstance(v, float) and math.isnan(v))
            else np.zeros(0)
            for v in raw
        ]
        width = max(r.size for r in rows)
        out = np.full((len(rows), width), np.nan)
        for i, r in enumerate(rows):
            out[i, : r.size] = r.ravel()
        return out

    def context(self, key: str) -> Any:
        """A context source, read once at the start of the episode."""
        return self._history.context.get(self._prefix + key)

    def color(self, spec: ColorSpec, slot: int = 0) -> Color:
        return resolve_color(self.theme, spec, slot)

    def size(self, px: float) -> float:
        """A size in pixels at 720 x 450, scaled to this frame."""
        return px * self.scale


# ----------------------------------------------------------------------
# Panels
# ----------------------------------------------------------------------


class Panel:
    """One chart of a dashboard. Subclass it to draw your own.

    A panel declares the values it needs (`sources()`, recorded every frame, and
    `context()`, read once per episode) and draws them in `draw(canvas, box, view)`,
    reading them back from the `View` with the same keys. Time-series panels
    (`timeseries = True`) share the dashboard's time axis; the helper `frame()`
    draws the title row and legend and returns the plot area.

    Args:
        title: Shown at the panel's top left.
        width: Relative width in its row.
        height: Relative height in its column.
    """

    timeseries: ClassVar[bool] = True

    def __init__(self, title: str | None = None, *, width: float = 1.0, height: float = 1.0):
        if width <= 0 or height <= 0:
            raise ValueError("Panel width and height must be positive.")
        self.title = title
        self.width = width
        self.height = height

    def sources(self) -> Mapping[str, Source]:
        """Values to record every frame, by key."""
        return {}

    def context(self) -> Mapping[str, Source]:
        """Values to read once, at the first frame of an episode, by key."""
        return {}

    def x_start(self, view: View) -> float | None:
        """The earliest time this panel shows, if before the first recorded step."""
        return None

    def draw(self, canvas: Canvas, box: Box, view: View) -> None:
        raise NotImplementedError

    def frame(
        self,
        canvas: Canvas,
        box: Box,
        view: View,
        legend_items: Sequence[tuple[str, ColorLike, Swatch]] = (),
        *,
        gutter: bool | None = None,
    ) -> Box:
        """Draw the title row (title left, legend right) and return the plot area.

        Time-series panels reserve the dashboard's y-axis gutter on the right and,
        when they show the time axis, a band underneath for its labels.
        """
        s = view.scale
        theme = view.theme
        title_h = 18 * s if (self.title or legend_items) else 0.0
        mid = box.top - 7 * s
        if self.title:
            canvas.text(
                box.left,
                mid,
                self.title,
                theme.text_secondary,
                size=11 * s,
                bold=True,
                valign="middle",
            )
        reserve = view.gutter if (self.timeseries if gutter is None else gutter) else 0.0
        items = list(legend_items)
        if items:
            room = box.width - reserve
            beside = room - (text_width(self.title, 11 * s, True) + 16 * s if self.title else 0)
            if legend_width(items, scale=s) <= beside:
                legend(canvas, box.right - reserve, mid, items, theme, scale=s)
            else:  # lines of their own under the title
                lines: list[list[tuple[str, ColorLike, Swatch]]] = [[]]
                for item in items:
                    if lines[-1] and legend_width([*lines[-1], item], scale=s) > room:
                        lines.append([])
                    lines[-1].append(item)
                for line in lines:
                    title_h += 16 * s
                    legend(
                        canvas,
                        box.left,
                        box.top - title_h + 2 * s,
                        line,
                        theme,
                        scale=s,
                        align="left",
                    )
        bottom = 16 * s if (self.timeseries and view.show_x) else 0.0
        return box.inset(top=title_h + 4 * s if title_h else 0.0, right=reserve, bottom=bottom)

    def time_axis(self, plot: Plot, view: View) -> None:
        """Label the time axis under `plot`, if this panel shows it."""
        if not view.show_x:
            return
        lo, hi = view.x_domain
        ticks = nice_ticks(lo, hi, max(int(plot.box.width / (70 * view.scale)), 2))
        decimals = decimals_for(float(np.min(np.diff(ticks)))) if ticks.size > 1 else 0
        plot.x_ticks(ticks, [format_number(t, decimals) for t in ticks])


class Row:
    """Panels or columns side by side. `widths` overrides the children's own widths."""

    def __init__(
        self, *children: Any, widths: Sequence[float] | None = None, height: float = 1.0
    ) -> None:
        self.children = children
        self.widths = None if widths is None else tuple(widths)
        self.width = 1.0
        self.height = height

    def __repr__(self) -> str:
        return f"Row({', '.join(map(repr, self.children))})"


class Column:
    """Panels or rows stacked top to bottom. `heights` overrides the children's own."""

    def __init__(
        self, *children: Any, heights: Sequence[float] | None = None, width: float = 1.0
    ) -> None:
        self.children = children
        self.heights = None if heights is None else tuple(heights)
        self.width = width
        self.height = 1.0

    def __repr__(self) -> str:
        return f"Column({', '.join(map(repr, self.children))})"


def _as_node(item: Any) -> Any:
    if isinstance(item, (Panel, Row, Column)):
        return item
    if isinstance(item, (list, tuple)):
        return Row(*item)
    raise TypeError(f"A dashboard holds Panels, Rows and Columns, got {type(item).__name__}.")


def _panels(node: Any) -> list[Panel]:
    if isinstance(node, Panel):
        return [node]
    return [p for child in node.children for p in _panels(_as_node(child))]


def _layout(node: Any, box: Box, gap: float, out: list[tuple[Panel, Box]]) -> None:
    if isinstance(node, Panel):
        out.append((node, box))
        return
    children = [_as_node(c) for c in node.children]
    if not children:
        return
    if isinstance(node, Row):
        weights = node.widths or tuple(c.width for c in children)
        boxes = box.columns(weights, gap * 1.3)
    else:
        weights = node.heights or tuple(c.height for c in children)
        boxes = box.rows(weights, gap)
    for child, child_box in zip(children, boxes):
        _layout(child, child_box, gap, out)


# ----------------------------------------------------------------------
# Dashboard
# ----------------------------------------------------------------------


class Dashboard(BaseRenderer):
    """A renderer that draws an environment's episode as a dashboard of charts.

    Args:
        *items: The layout, top to bottom: Panels, `Row`s / `Column`s of them, or
            lists (a list is a Row).
        title: The heading (default: the env's registered id or class name); a string
            or a source.
        subtitle: A line under the title (a string or a source), e.g. the date.
        stats: Headline numbers shown at the top right.
        heights: Relative heights of the top-level items (default: their own `height`).
        width: Frame width in pixels.
        height: Frame height in pixels.
        theme: "light", "dark" or a `Theme`.
        step: The episode clock, which is also the time axis (default:
            `elapsed_steps`; any number works, e.g. a session time); None counts renders.
        horizon: The clock's value at the end of an episode, which fixes the time axis
            so the episode draws from left to right: a number, a source, None (the
            axis grows with the episode) or "auto" (the env's `episode_length` or
            `n_steps`, else the `max_episode_steps` of its spec).
        window: Show only the latest `window` steps (a scrolling chart).
    """

    def __init__(
        self,
        *items: Any,
        title: str | Source | None = None,
        subtitle: str | Source | None = None,
        stats: Sequence[Stat] = (),
        heights: Sequence[float] | None = None,
        width: int = 720,
        height: int = 450,
        theme: Theme | str = "light",
        step: Source | None = "elapsed_steps",
        horizon: float | Source | None = "auto",
        window: int | None = None,
    ) -> None:
        if width < 160 or height < 120:
            raise ValueError("A dashboard needs at least 160 x 120 pixels.")
        if window is not None and window < 1:
            raise ValueError("window must be >= 1.")
        self.set_layout(*items, heights=heights)
        self.title = title
        self.subtitle = subtitle
        self.stats = tuple(stats)
        self.width = width
        self.height = height
        self.theme = get_theme(theme)
        self.step = step
        self.horizon = horizon
        self.window = window
        self.scale = min(width / 720, height / 450)
        self._histories: weakref.WeakKeyDictionary[Any, _History] = weakref.WeakKeyDictionary()
        self._unhashable: dict[int, _History] = {}

    def set_layout(self, *items: Any, heights: Sequence[float] | None = None) -> None:
        """Replace the charts (and forget recorded histories): the same arguments as above.

        A renderer that sizes its charts from the env (one line per trader, say) can
        call this from `check()` the first time it sees the env.
        """
        self.root = Column(*items, heights=heights)
        self.panels = _panels(self.root)
        if hasattr(self, "_histories"):
            self.reset()

    # Hooks ------------------------------------------------------------------

    def check(self, env: Any) -> None:
        """Raise if `env` cannot be drawn (subclasses check its type and that it was reset)."""

    def reset(self, env: Any = None) -> None:
        """Forget the recorded history of `env` (of every env if None)."""
        if env is None:
            self._histories.clear()
            self._unhashable.clear()
            return
        base = env.unwrapped if hasattr(env, "unwrapped") else env
        self._histories.pop(base, None)
        self._unhashable.pop(id(base), None)

    # Recording ----------------------------------------------------------------

    def _store(self, base: Any) -> tuple[_History | None, Callable[[_History], None]]:
        try:
            history = self._histories.get(base)

            def put(h: _History) -> None:
                self._histories[base] = h

        except TypeError:  # not weak-referenceable
            history = self._unhashable.get(id(base))

            def put(h: _History) -> None:
                self._unhashable[id(base)] = h

        return history, put

    def _clock(self, env: Any) -> float | None:
        if self.step is None:
            return None
        try:
            value = read(env, self.step)
        except AttributeError:
            if self.step == "elapsed_steps":
                return None
            raise
        return None if value is None else float(value)

    def _sources(self) -> tuple[dict[str, Source], dict[str, Source]]:
        sources: dict[str, Source] = {}
        context: dict[str, Source] = {}
        for i, panel in enumerate(self.panels):
            sources.update({f"{i}:{k}": v for k, v in panel.sources().items()})
            context.update({f"{i}:{k}": v for k, v in panel.context().items()})
        for i, stat in enumerate(self.stats):
            sources[f"stat{i}"] = stat.source
        return sources, context

    def _record(self, env: Any) -> _History:
        base = env.unwrapped if hasattr(env, "unwrapped") else env
        history, put = self._store(base)
        clock = self._clock(env)
        sources, context = self._sources()
        new = history is None or (
            clock is not None and (clock == 0 or (history.steps and clock < history.steps[-1]))
        )
        if new:
            history = _History()
            history.context = {key: _snapshot(read(env, src)) for key, src in context.items()}
            put(history)
        assert history is not None
        if clock is None:
            clock = history.steps[-1] + 1 if history.steps else 0
        replace = bool(history.steps) and clock == history.steps[-1]
        samples = {key: _snapshot(read(env, src)) for key, src in sources.items()}
        history.record(clock, samples, replace)
        if self.window is not None:
            history.trim(self.window + 1)
        return history

    # Drawing --------------------------------------------------------------------

    def _horizon(self, env: Any) -> float | None:
        horizon = self.horizon
        if horizon is None:
            return None
        if isinstance(horizon, (int, float)):
            return float(horizon)
        if horizon != "auto":
            value = read(env, horizon)
            return None if value is None else float(value)
        base = env.unwrapped if hasattr(env, "unwrapped") else env
        for name in ("episode_length", "n_steps"):
            steps = _positive_int(getattr(base, name, None))
            if steps is not None:
                return float(steps)
        for target in (env, base):
            steps = _positive_int(getattr(getattr(target, "spec", None), "max_episode_steps", None))
            if steps is not None:
                return float(steps)
        return None

    def _title(self, env: Any) -> str:
        if self.title is not None:
            return self.title if isinstance(self.title, str) else str(read(env, self.title))
        base = env.unwrapped if hasattr(env, "unwrapped") else env
        for target in (env, base):
            spec_id = getattr(getattr(target, "spec", None), "id", None)
            if spec_id:
                return str(spec_id)
        return type(base).__name__

    def render(self, env: Any) -> NDArray[np.uint8]:
        """Record the env's current state and draw the episode so far."""
        self.check(env)
        history = self._record(env)
        theme, s = self.theme, self.scale
        canvas = Canvas(self.width, self.height, theme.background, antialias=True)
        frame = Box(0, 0, self.width, self.height).inset(18 * s, 14 * s, 18 * s, 14 * s)
        header = Box(frame.left, frame.top - 40 * s, frame.right, frame.top)
        body = frame.inset(top=header.height + 18 * s)
        horizon = self._horizon(env)
        self._header(canvas, header, env, history, horizon)

        placed: list[tuple[Panel, Box]] = []
        _layout(self.root, body, 16 * s, placed)
        steps = history.steps
        first, last = float(steps[0]), float(steps[-1])
        lo = first
        views = []
        for panel, _ in placed:
            view = View(env, history, f"{self.panels.index(panel)}:", theme, s, (0, 1), False, 0)
            start = panel.x_start(view) if panel.timeseries else None
            if start is not None:
                lo = min(lo, start)
            views.append(view)
        if self.window is not None:
            lo = max(lo, last - self.window)
            hi = max(last, lo + self.window)
        else:
            hi = max(float(horizon) if horizon else last, last)
        if hi <= lo:
            hi = lo + 1.0
        for (panel, box), view in zip(placed, views):
            view.x_domain = (lo, hi)
            view.gutter = 52 * s
            view.show_x = panel.timeseries and not any(
                other.timeseries
                and other_box.top <= box.bottom + 1e-6
                and other_box.left < box.right
                and other_box.right > box.left
                for other, other_box in placed
                if other is not panel
            )
            panel.draw(canvas, box, view)
        return canvas.to_array()

    def _header(
        self, canvas: Canvas, box: Box, env: Any, history: _History, horizon: float | None
    ) -> None:
        theme, s = self.theme, self.scale
        title = self._title(env)
        canvas.text(
            box.left, box.top - 13 * s, title, theme.text, size=16 * s, bold=True, valign="middle"
        )
        if self.subtitle is not None:
            sub = self.subtitle if isinstance(self.subtitle, str) else read(env, self.subtitle)
            if sub:
                canvas.text(
                    box.left,
                    box.top - 31 * s,
                    str(sub),
                    theme.text_muted,
                    size=10.5 * s,
                    valign="middle",
                )
        # Stats, right-aligned: label over value, with an optional change since the start.
        right = box.right
        for i in reversed(range(len(self.stats))):
            stat = self.stats[i]
            values = history.values.get(f"stat{i}", [])
            value = values[-1] if values else None
            text = stat.text(value)
            width = text_width(text, 15 * s, True)
            delta_text, delta_up = None, True
            if stat.delta is not None and values:
                v0, v1 = _number(values[0]), _number(value)
                change = (v1 / v0 - 1) if stat.delta == "relative" and v0 else v1 - v0
                if math.isfinite(change):
                    if stat.delta == "relative":
                        delta_text = format_number(abs(change), 2, percent=True)
                    else:
                        delta_text = stat.text(abs(change))
                    delta_up = change >= 0
            delta_w = text_width(delta_text, 10 * s) + 11 * s if delta_text else 0.0
            label_w = text_width(stat.label, 10 * s)
            column = max(width + delta_w, label_w)
            left = right - column
            canvas.text(
                right,
                box.top - 6 * s,
                stat.label,
                theme.text_muted,
                size=10 * s,
                align="right",
                valign="middle",
            )
            canvas.text(
                right - delta_w,
                box.top - 24 * s,
                text,
                theme.text,
                size=15 * s,
                bold=True,
                align="right",
                valign="middle",
            )
            if delta_text:
                good = delta_up == stat.higher_is_better
                color = theme.up if good else theme.down
                tx = right - delta_w + 9 * s
                triangle(canvas, tx, box.top - 24 * s, 3 * s, color, delta_up)
                canvas.text(
                    tx + 5 * s,
                    box.top - 24 * s,
                    delta_text,
                    theme.text_secondary,
                    size=10 * s,
                    valign="middle",
                )
            right = left - 24 * s
        # Episode progress: a hairline track along the bottom of the header.
        track_y = box.bottom - 6 * s
        canvas.rect(box.left, track_y, box.right, track_y + 2 * s, theme.grid, radius=1 * s)
        if horizon:
            done = min(max(history.steps[-1] / horizon, 0.0), 1.0)
            if done > 0:
                canvas.rect(
                    box.left,
                    track_y,
                    box.left + done * box.width,
                    track_y + 2 * s,
                    theme.accent,
                    radius=1 * s,
                )


# ----------------------------------------------------------------------
# Generic panels
# ----------------------------------------------------------------------


def shade_steps(
    plot: Plot,
    steps: NDArray[np.float64],
    on: NDArray[np.bool_],
    color: ColorLike,
    alpha: float = 0.08,
) -> None:
    """Shade the steps where `on` holds, each over the step that ended at it."""
    run_start: float | None = None
    for i in range(1, len(steps) + 1):
        active = i < len(steps) and bool(on[i])
        if active and run_start is None:
            run_start = float(steps[i - 1])
        elif not active and run_start is not None:
            plot.shade(run_start, float(steps[i - 1]), color, alpha=alpha)
            run_start = None


def tick_count(height: float, scale: float) -> int:
    """How many y-axis intervals suit a plot `height` pixels tall (one per ~28 px, >= 3)."""
    return max(int(height / (28 * scale)), 3)


def _series_values(view: View, key: str, series: Series) -> NDArray[np.float64]:
    values = view.values(key)
    if values.ndim > 1:
        values = values[:, 0]
    if series.rebase is not None:
        finite = values[np.isfinite(values)]
        base = finite[0] if finite.size else math.nan
        values = values / base * series.rebase if base else values * math.nan
    return values


class LineChart(Panel):
    """Series over time, on one shared y-axis.

    Args:
        *series: `Series`, or sources (each becomes a plain Series).
        title: Panel title.
        baseline: A reference value drawn as a baseline and kept in view (e.g. 0, or
            1.0 for equity that starts at 1).
        percent: Label the y-axis (and the value tag) as percentages.
        tag: Show the first series' latest value in a tag on the y-axis.
        limits: Reference values drawn as dashed lines (e.g. inventory limits).
        shade: A source that is true while a condition holds (a regime, a drawdown,
            market hours): those steps are shaded behind the series.
        shade_label: Legend label for the shading.
        shade_color: Colour of the shading (default: the `down` role).
        width: Relative width in its row.
        height: Relative height in its column.
    """

    def __init__(
        self,
        *series: Series | Source,
        title: str | None = None,
        baseline: float | None = None,
        percent: bool = False,
        tag: bool = True,
        limits: Sequence[float] = (),
        shade: Source | None = None,
        shade_label: str | None = None,
        shade_color: ColorSpec = "down",
        width: float = 1.0,
        height: float = 1.0,
    ) -> None:
        super().__init__(title, width=width, height=height)
        if not series:
            raise ValueError("LineChart needs at least one series.")
        self.series = tuple(s if isinstance(s, Series) else Series(s) for s in series)
        self.baseline = baseline
        self.percent = percent
        self.tag = tag
        self.limits = tuple(limits)
        self.shade = shade
        self.shade_label = shade_label
        self.shade_color = shade_color

    def sources(self) -> Mapping[str, Source]:
        out = {f"s{i}": s.source for i, s in enumerate(self.series)}
        if self.shade is not None:
            out["shade"] = self.shade
        return out

    def draw(self, canvas: Canvas, box: Box, view: View) -> None:
        theme, s = view.theme, view.scale
        colors = [view.color(sr.color, i) for i, sr in enumerate(self.series)]
        data = [_series_values(view, f"s{i}", sr) for i, sr in enumerate(self.series)]
        labelled = [(sr, c) for sr, c in zip(self.series, colors) if sr.label]
        items: list[tuple[str, ColorLike, Swatch]] = []
        if len(labelled) >= 2:
            items = [
                (sr.label or "", c, "dash" if sr.dashed else "area" if sr.fill else "line")
                for sr, c in labelled
            ]
        shade_color = view.color(self.shade_color)
        shaded = None
        if self.shade is not None:
            flags = view.values("shade")
            if flags.ndim != 1:
                raise ValueError("LineChart(shade=...) needs one true/false value per step.")
            shaded = flags > 0.5
        if shaded is not None and self.shade_label and shaded[1:].any():
            items.insert(0, (self.shade_label, shade_color, "area"))
        plot_box = self.frame(canvas, box, view, items)
        include = [self.baseline] if self.baseline is not None else []
        y_domain = value_range(*data, include=[*include, *self.limits])
        plot = Plot(canvas, plot_box, view.x_domain, y_domain, theme, s)
        values = np.concatenate([d[np.isfinite(d)] for d in data])
        whole = not self.percent and values.size > 0 and bool(np.all(values == np.round(values)))
        ticks = nice_ticks(*y_domain, tick_count(plot_box.height, s), integer=whole)
        last = data[0][-1] if self.tag and data[0].size else None
        if shaded is not None:
            shade_steps(plot, view.steps, shaded, shade_color)
        plot.grid(ticks, tick_labels(ticks, percent=self.percent), avoid=[last])
        if self.baseline is not None:
            plot.hline(self.baseline, theme.axis)
        for limit in self.limits:
            plot.hline(limit, theme.neutral, dash=(3, 3))
        steps = view.steps
        for i in reversed(range(len(self.series))):
            sr, ys, color = self.series[i], data[i], colors[i]
            if sr.signed:
                base = self.baseline if self.baseline is not None else 0.0
                plot.area(
                    steps,
                    np.where(ys >= base, ys, base),
                    theme.up,
                    base=base,
                    alpha=0.2,
                    step=sr.step,
                )
                plot.area(
                    steps,
                    np.where(ys < base, ys, base),
                    theme.down,
                    base=base,
                    alpha=0.2,
                    step=sr.step,
                )
            elif sr.fill:
                plot.area(steps, ys, color, alpha=0.1, step=sr.step)
            plot.line(
                steps, ys, color, width=sr.width, dash=(5, 3) if sr.dashed else None, step=sr.step
            )
            if ys.size and math.isfinite(ys[-1]):
                plot.marker(steps[-1], ys[-1], color, radius=3.5)
        if self.tag and data[0].size and math.isfinite(data[0][-1]):
            step = float(np.min(np.diff(ticks))) if ticks.size > 1 else 1.0
            decimals = min(decimals_for(step * (100 if self.percent else 1)) + 1, 4)
            if whole:
                decimals = 0  # counts, inventories: whole numbers
            text = format_number(
                data[0][-1], decimals, percent=self.percent, compact=abs(data[0][-1]) >= 100_000
            )
            plot.tag(data[0][-1], text, colors[0])
        self.time_axis(plot, view)


class BarChart(Panel):
    """The current values of a vector as labelled horizontal bars (weights, holdings).

    Args:
        source: A vector per step.
        labels: Bar labels: a sequence, or a source giving one per frame.
        colors: One colour per bar (theme roles, hex, RGB or slots); default: slot i
            for bar i, when bars are separate entities shown elsewhere in those colours.
        color: One colour for every bar instead (a single series identified by its
            labels), e.g. "accent". Negative values use `negative_color`.
        negative_color: Colour of negative bars when `color` is given.
        available: A source of a boolean per bar; unavailable bars are greyed out.
        percent: Write values as percentages.
        limit: The value a full-length bar stands for (default: a round number just
            above the largest magnitude).
        title: Panel title.
        width: Relative width in its row.
        height: Relative height in its column.
    """

    timeseries = False

    def __init__(
        self,
        source: Source,
        *,
        labels: Sequence[str] | Source | None = None,
        colors: Sequence[ColorSpec] | None = None,
        color: ColorSpec = None,
        negative_color: ColorSpec = "down",
        available: Source | None = None,
        percent: bool = True,
        limit: float | None = None,
        title: str | None = None,
        width: float = 1.0,
        height: float = 1.0,
    ) -> None:
        super().__init__(title, width=width, height=height)
        self.source = source
        self.labels = labels
        self.colors = None if colors is None else tuple(colors)
        self.color = color
        self.negative_color = negative_color
        self.available = available
        self.percent = percent
        self.limit = limit

    def sources(self) -> Mapping[str, Source]:
        out: dict[str, Source] = {"values": self.source}
        if self.labels is not None and not isinstance(self.labels, (list, tuple)):
            out["labels"] = self.labels  # type: ignore[assignment]
        if self.available is not None:
            out["available"] = self.available
        return out

    def draw(self, canvas: Canvas, box: Box, view: View) -> None:
        theme, s = view.theme, view.scale
        values = np.atleast_1d(np.asarray(view.latest("values"), dtype=np.float64)).ravel()
        n = values.size
        if isinstance(self.labels, (list, tuple)):
            labels = [str(v) for v in self.labels]
        elif self.labels is not None:
            labels = [("" if v is None else str(v)) for v in (view.latest("labels") or [])]
        else:
            labels = [str(i) for i in range(n)]
        labels = (labels + [""] * n)[:n]
        available = (
            np.ones(n, dtype=bool)
            if self.available is None
            else np.asarray(view.latest("available"), dtype=bool).ravel()[:n]
        )
        area = self.frame(canvas, box, view)
        if n == 0:
            return
        finite = np.where(np.isfinite(values), values, 0.0)
        top = self.limit if self.limit is not None else float(np.max(np.abs(finite), initial=0))
        if self.limit is None:
            top = nice_step(top, 1) if top > 0 else 1.0
        negative = bool((finite < 0).any())
        row_h = area.height / n
        bar_h = min(14 * s, row_h * 0.55)
        stacked = row_h >= bar_h + 22 * s  # room for the label above its bar
        label_w = 0.0 if stacked else max(text_width(t, 10 * s) for t in labels) + 10 * s
        value_w = max(text_width(self._fmt(v), 10 * s) for v in finite) + 8 * s
        left = area.left + label_w
        right = area.right - value_w
        zero = (left + right) / 2 if negative else left
        span = (right - zero) if not negative else (right - left) / 2
        for i, (value, label) in enumerate(zip(finite, labels)):
            row_top = area.top - i * row_h
            if stacked:
                bar_mid = row_top - row_h + (row_h - 16 * s) / 2
                canvas.text(
                    area.left,
                    bar_mid + bar_h / 2 + 9 * s,
                    label,
                    theme.text_secondary,
                    size=10 * s,
                    valign="middle",
                )
            else:
                bar_mid = row_top - row_h / 2
                canvas.text(
                    area.left, bar_mid, label, theme.text_secondary, size=10 * s, valign="middle"
                )
            if self.color is not None:
                spec = self.color if value >= 0 else self.negative_color
            else:
                spec = self.colors[i] if self.colors and i < len(self.colors) else None
            color = resolve_color(theme, spec, i)
            if not available[i]:
                canvas.rect(
                    zero,
                    bar_mid - bar_h / 2,
                    right,
                    bar_mid + bar_h / 2,
                    theme.grid,
                    radius=2 * s,
                    alpha=0.6,
                )
                canvas.text(
                    zero + 6 * s,
                    bar_mid,
                    "not tradable",
                    theme.text_muted,
                    size=9 * s,
                    valign="middle",
                )
                continue
            # Track, then the bar: rounded at its data end, square at the baseline.
            canvas.rect(
                left,
                bar_mid - bar_h / 2,
                right,
                bar_mid + bar_h / 2,
                theme.grid,
                radius=2 * s,
                alpha=0.55,
            )
            end = zero + min(abs(value) / top, 1.0) * span * (1 if value >= 0 else -1)
            if abs(end - zero) >= 0.5:
                lo_x, hi_x = min(zero, end), max(zero, end)
                r = min(4 * s, bar_h / 2, (hi_x - lo_x) / 2)
                canvas.rect(lo_x, bar_mid - bar_h / 2, hi_x, bar_mid + bar_h / 2, color, radius=r)
                if value >= 0:
                    canvas.rect(
                        zero, bar_mid - bar_h / 2, min(zero + r, hi_x), bar_mid + bar_h / 2, color
                    )
                else:
                    canvas.rect(
                        max(zero - r, lo_x), bar_mid - bar_h / 2, zero, bar_mid + bar_h / 2, color
                    )
            canvas.text(
                area.right,
                bar_mid,
                self._fmt(value),
                theme.text,
                size=10 * s,
                align="right",
                valign="middle",
            )
        if negative:
            canvas.rect(round(zero), area.bottom, round(zero) + 1, area.top, theme.axis)

    def _fmt(self, value: float) -> str:
        if self.percent:
            return format_number(value, 1, percent=True)
        return format_number(value, 2 if abs(value) < 100 else 0, compact=abs(value) >= 100_000)


class StackedArea(Panel):
    """A vector over time as stacked areas (an allocation's composition, say).

    Positive parts stack up from zero, negative parts down. Layers are separated by a
    thin line in the surface colour.

    Args:
        source: A vector per step.
        labels: Layer labels for the legend.
        colors: Layer colours; default: slot i for layer i.
        percent: Label the axis as percentages.
        title: Panel title.
        width: Relative width in its row.
        height: Relative height in its column.
    """

    def __init__(
        self,
        source: Source,
        *,
        labels: Sequence[str] | None = None,
        colors: Sequence[ColorSpec] | None = None,
        percent: bool = True,
        legend: bool = False,
        title: str | None = None,
        width: float = 1.0,
        height: float = 1.0,
    ) -> None:
        super().__init__(title, width=width, height=height)
        self.source = source
        self.labels = None if labels is None else tuple(labels)
        self.colors = None if colors is None else tuple(colors)
        self.percent = percent
        self.legend = legend

    def sources(self) -> Mapping[str, Source]:
        return {"values": self.source}

    def draw(self, canvas: Canvas, box: Box, view: View) -> None:
        theme, s = view.theme, view.scale
        values = view.values("values")
        if values.ndim == 1:
            values = values[:, None]
        k = values.shape[1] if values.size else 0
        colors = [
            resolve_color(
                theme, self.colors[j] if self.colors and j < len(self.colors) else None, j
            )
            for j in range(k)
        ]
        items: list[tuple[str, ColorLike, Swatch]] = []
        if self.legend and self.labels:
            items = [(lab, c, "square") for lab, c in zip(self.labels, colors)]
        plot_box = self.frame(canvas, box, view, items)
        clean = np.nan_to_num(values, nan=0.0)
        up = np.cumsum(np.clip(clean, 0, None), axis=1)
        down = np.cumsum(np.clip(clean, None, 0), axis=1)
        top = float(np.max(up[:, -1], initial=0.0)) if k else 1.0
        bottom = float(np.min(down[:, -1], initial=0.0)) if k else 0.0
        if self.percent and top <= 1.0 + 1e-6 and top > 0.5:
            top = 1.0
        y_domain = (bottom, top if top > bottom else bottom + 1.0)
        plot = Plot(canvas, plot_box, view.x_domain, y_domain, theme, s)
        ticks = nice_ticks(*y_domain, tick_count(plot_box.height, s))
        plot.grid(ticks, tick_labels(ticks, percent=self.percent))
        steps = view.steps
        if k and steps.size:
            lower_up = np.zeros(len(steps))
            lower_down = np.zeros(len(steps))
            for j in range(k):
                for lower, upper in ((lower_up, up[:, j]), (lower_down, down[:, j])):
                    if np.allclose(upper, lower):
                        continue
                    fill = mix(colors[j], theme.background, 0.38)
                    plot.area(steps, upper, fill, base=lower, alpha=1.0)
                    plot.line(steps, upper, theme.background, width=1.0)
                lower_up, lower_down = up[:, j], down[:, j]
        self.time_axis(plot, view)


__all__ = [
    "BarChart",
    "Column",
    "Dashboard",
    "LineChart",
    "Panel",
    "Row",
    "Series",
    "Source",
    "StackedArea",
    "Stat",
    "View",
    "read",
    "resolve_color",
]
