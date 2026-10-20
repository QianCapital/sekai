"""Colour themes for sekai's charts and dashboards.

A `Theme` names colours by the job they do, never by hue: surface and ink, the
polarity pair `up` / `down` (long and short, buys and sells, bids and asks), roles
such as `accent` (the series a chart is about) and `highlight` (the agent's own
orders), and an eight-colour categorical palette for identity (one colour per asset
or trader, assigned in order and never cycled).

`LIGHT` and `DARK` use the Qian Capital house palette: a warm paper or forest-ink
surface with forest-green and gold accents. Their categorical palettes were validated
for colour-vision deficiency (protanopia and deuteranopia, adjacent slots at least
8.9 apart in OKLab x100; the first three slots apart from each other in every pair)
and for normal vision (adjacent slots at least 15.6 apart), each against its own
surface. The polarity pairs are separated for protanopes and deuteranopes as well,
and charts pair them with shape and position (triangles up and down, bids below
asks), so colour is never the only cue.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, TypeAlias

Color: TypeAlias = tuple[int, int, int]
ColorLike: TypeAlias = Color | str


def rgb(color: ColorLike) -> Color:
    """An `(r, g, b)` tuple from a tuple or a `"#rrggbb"` / `"#rgb"` hex string."""
    if isinstance(color, str):
        value = color.lstrip("#")
        if len(value) == 3:
            value = "".join(ch * 2 for ch in value)
        if len(value) != 6:
            raise ValueError(f"Expected a colour like '#1f271b', got {color!r}.")
        return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))
    r, g, b = color
    return (int(r), int(g), int(b))


def mix(color: ColorLike, other: ColorLike, amount: float) -> Color:
    """`color` moved `amount` (0 to 1) of the way towards `other`."""
    a, b = rgb(color), rgb(other)
    return (
        round(a[0] + (b[0] - a[0]) * amount),
        round(a[1] + (b[1] - a[1]) * amount),
        round(a[2] + (b[2] - a[2]) * amount),
    )


def luminance(color: ColorLike) -> float:
    """WCAG relative luminance in [0, 1]."""

    def channel(c: int) -> float:
        v = c / 255
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = (channel(c) for c in rgb(color))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


@dataclass(frozen=True)
class Theme:
    """The colours of a chart, by role.

    Attributes:
        name: A short name ("light", "dark", ...).
        background: The frame and plot surface.
        grid: Hairline gridlines, one step off the surface.
        axis: Baselines and the progress-bar track.
        text: Primary ink: titles and values.
        text_secondary: Panel titles and legends.
        text_muted: Axis labels and stat labels.
        price: The main price line.
        accent: The series a chart is about, e.g. the agent's equity.
        neutral: De-emphasised series (benchmarks, cash) and reference lines.
        up: Long positions, buys, bids, gains.
        down: Short positions, sells, asks, losses.
        highlight: The agent's own orders and quotes.
        series: The categorical palette, in assignment order.
    """

    name: str
    background: Color
    grid: Color
    axis: Color
    text: Color
    text_secondary: Color
    text_muted: Color
    price: Color
    accent: Color
    neutral: Color
    up: Color
    down: Color
    highlight: Color
    series: tuple[Color, ...]

    def color(self, slot: int) -> Color:
        """Categorical colour `slot` (0-based); slots past the palette get `neutral`.

        Colours are never reused: a ninth series is folded into the neutral grey
        rather than given a colour that could be mistaken for the first.
        """
        return self.series[slot] if 0 <= slot < len(self.series) else self.neutral

    def ink_on(self, fill: ColorLike) -> Color:
        """The text colour (light or dark ink) that reads best on a filled `fill`."""
        light, dark = (self.background, self.text)
        if luminance(light) < luminance(dark):
            light, dark = dark, light
        return dark if luminance(fill) > 0.32 else light

    def with_colors(
        self, *, series: Sequence[ColorLike] | None = None, **colors: ColorLike
    ) -> Theme:
        """A copy with some roles changed, e.g. `LIGHT.with_colors(up="#2a78d6")`.

        `series` replaces the categorical palette (validate a new one for colour-vision
        deficiency before you rely on it).
        """
        changes: dict[str, Any] = {role: rgb(value) for role, value in colors.items()}
        if series is not None:
            changes["series"] = tuple(rgb(c) for c in series)
        return replace(self, **changes)


def _theme(name: str, series: tuple[str, ...], **roles: str) -> Theme:
    return Theme(
        name=name,
        series=tuple(rgb(c) for c in series),
        **{role: rgb(value) for role, value in roles.items()},
    )


LIGHT = _theme(
    "light",
    # forest, gold, plum, terracotta, sea, olive, slate, rose
    ("#107c5a", "#bd8928", "#82498b", "#c45e3d", "#0098a9", "#7b8935", "#3d73b6", "#cc6b83"),
    background="#f8f6f2",
    grid="#eae3d7",
    axis="#cfc7b8",
    text="#1f271b",
    text_secondary="#4d5545",
    text_muted="#6f6c5f",
    price="#1f271b",
    accent="#107c5a",
    neutral="#a8a294",
    up="#439458",
    down="#a12f2f",
    highlight="#bd8928",
)
"""Warm paper surface, forest ink: the Qian Capital light palette."""

DARK = _theme(
    "dark",
    ("#3b9a76", "#bc8726", "#a86cb1", "#cf6746", "#0ba5b6", "#808f3b", "#578ed4", "#d16f88"),
    background="#161c13",
    grid="#262f22",
    axis="#3b4636",
    text="#ece3d0",
    text_secondary="#c4bfae",
    text_muted="#8e9483",
    price="#ece3d0",
    accent="#3b9a76",
    neutral="#6f7566",
    up="#3aa273",
    down="#b94642",
    highlight="#c0aa81",
)
"""Forest-ink surface, gold-white ink: the Qian Capital dark palette."""

THEMES = {"light": LIGHT, "dark": DARK}


def get_theme(theme: Theme | str) -> Theme:
    """A Theme, or the built-in one named "light" or "dark"."""
    if isinstance(theme, Theme):
        return theme
    try:
        return THEMES[theme]
    except KeyError:
        raise ValueError(f"Unknown theme {theme!r}; use 'light', 'dark' or a Theme.") from None


__all__ = ["DARK", "LIGHT", "THEMES", "Color", "ColorLike", "Theme", "get_theme", "mix", "rgb"]
