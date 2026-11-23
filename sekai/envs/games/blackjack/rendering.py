"""Renderers for blackjack: the felt table as RGB frames, and a text summary.

`BlackjackRenderer` draws the table: a green felt with a rounded top, a dark rail and a
gold line, "BLACKJACK PAYS 3 TO 2" printed on an arc (in the env's rules), the dealer at
the top with the hole card face down, the shoe gauge with the Hi-Lo count, and the seat
at the bottom with its hands, totals, outcomes, bet spot and name plate. It reads only
the public table view, so it never shows the shoe order or an unrevealed hole card.

`BlackjackTextRenderer` returns the same information as a few lines of text.

Both draw a `BlackjackEnv` (or any wrapper around one) or a `BlackjackTable`, which may
have any number of seats.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from sekai.envs.games._cards import card_height, draw_card
from sekai.envs.games._draw import Painter, gaussian_blur, rounded_box_coverage
from sekai.envs.games.blackjack.env import BlackjackEnv
from sekai.envs.games.blackjack.rules import Rules
from sekai.envs.games.blackjack.table import BlackjackTable
from sekai.envs.games.rendering import GameRenderer, GameTheme, Status, Tone, unwrap
from sekai.error import ResetRequired
from sekai.rendering.canvas import Canvas
from sekai.rendering.renderer import BaseRenderer
from sekai.rendering.text import face
from sekai.rendering.theme import ColorLike, rgb

Corners = list[tuple[float, float]]
"""Per-corner radii `(rx, ry)`: top-left, top-right, bottom-right, bottom-left."""

# The table's colours.
FELT_STOPS = ((0.0, "#3f6a52"), (0.5, "#2c4d3b"), (1.0, "#1d3428"))
RAIL = "#2a1d14"
BRASS = "#c0aa81"
GOLD = "#c9b48b"
IVORY = "#f3ecdd"
LINEN = "#eae3d7"  # text on the felt, at some opacity
PILL = "#0a120c"  # totals, at .65
BUST = "#a03c2d"  # busts and losses, at .85
WIN = "#86ad93"
PUSH = "#3a3f37"
INK = "#13200f"
CHIP_RED = "#b8352b"
PLATE = "#0e140d"  # name plates, at .9
MARKER = "#d98573"

OUTCOMES = {
    "win": "Win",
    "lose": "Lose",
    "push": "Push",
    "blackjack": "Blackjack",
    "surrender": "Surrender",
}

LINE = 1.55
"""The line height (in ems), which sets the height of a line of text."""

MONO_SPACE = 0.4
"""The advance of a space in figures (in ems): figures are set in a monospaced face with
wide spaces; the sans face's own space is narrow."""


# ----------------------------------------------------------------------
# Numbers and labels, as the table shows them
# ----------------------------------------------------------------------


def format_signed(value: float, digits: int = 0) -> str:
    """`+10`, `0`, or a negative number with a minus sign (U+2212), halves rounding up."""
    q = Decimal(abs(value)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    if q == 0:
        return f"{Decimal(0).quantize(Decimal(1).scaleb(-digits)):f}"
    return ("+" if value > 0 else "\u2212") + f"{q:f}"


def format_chips(value: float) -> str:
    """Whole chips with thousands separators (`1022.5` is `1,023`: halves round up)."""
    whole = math.floor(value)
    if value - whole >= 0.5:
        whole += 1
    return f"{whole:,}"


def total_label(hand: Mapping[str, Any]) -> str:
    """A hand's total as the table shows it: `Blackjack`, `Bust 24`, `Soft 17`, `20`."""
    if not hand.get("cards"):
        return ""
    if hand.get("blackjack"):
        return "Blackjack"
    total = hand["total"]
    if hand.get("bust") or total > 21:
        return f"Bust {total}"
    return f"Soft {total}" if hand.get("soft") and total < 21 else str(total)


def dealer_label(dealer: Mapping[str, Any]) -> str:
    """The dealer's total: `Shows 10` before the reveal, then `17`, `Bust 24`, `Blackjack`."""
    if not dealer["cards"]:
        return ""
    if dealer["blackjack"]:
        return "Blackjack"
    if dealer["bust"]:
        return f"Bust {dealer['total']}"
    return str(dealer["total"]) if dealer["revealed"] else f"Shows {dealer['total']}"


def outcome_label(hand: Mapping[str, Any]) -> str:
    """A settled hand's result: `Win +10`, `Push`, `Blackjack +15`, `Lose` and a minus..."""
    text = OUTCOMES.get(hand["outcome"], str(hand["outcome"]))
    payout = hand.get("payout")
    return f"{text} {format_signed(payout, 0)}" if payout else text


def initials(name: str) -> str:
    """The initials an avatar shows: `"PL"` for `"Player"`, `"BS"` for `"Basic strategy"`."""
    words = "".join(c if c.isascii() and c.isalnum() else " " for c in name).split()
    if not words:
        return "?"
    if len(words) == 1:
        return words[0][:2].upper()
    last = words[-1]
    if last.isdigit() or set(last) <= set("IVX"):
        return (words[0][0] + last)[:3].upper()
    return (words[0][0] + last[0]).upper()


def print_lines(rules: Rules) -> tuple[str, str]:
    """The words printed on the felt for these rules: the payout, and the dealer's rule."""
    ratio = rules.blackjack_ratio
    main = f"BLACKJACK PAYS {ratio.numerator} TO {ratio.denominator}"
    dealer = "DEALER HITS SOFT 17" if rules.dealer_hits_soft_17 else "DEALER STANDS ON ALL 17s"
    return main, f"{dealer} \u00b7 NO INSURANCE"


def _up_label(card: str | None) -> str:
    if not card:
        return "?"
    return "10" if card[0] == "T" else card[0]


# ----------------------------------------------------------------------
# Drawing helpers the table needs beyond the shared painter
# ----------------------------------------------------------------------


def _blurred_box(
    left: float, top: float, right: float, bottom: float, radius: Corners, sigma: float
) -> tuple[NDArray[np.float64], int, int]:
    """A rounded box's coverage blurred by `sigma`, on a grid wide enough for the blur, and
    the grid's top-left pixel."""
    pad = math.ceil(3 * sigma) + 2
    x0, y0 = math.floor(left) - pad, math.floor(top) - pad
    x1, y1 = math.ceil(right) + pad, math.ceil(bottom) + pad
    xs = np.arange(x0, x1, dtype=np.float64)[None, :] + 0.5
    ys = np.arange(y0, y1, dtype=np.float64)[:, None] + 0.5
    sharp = rounded_box_coverage(xs, ys, left, top, right, bottom, radius)
    return gaussian_blur(sharp, max(sigma, 1e-6)), x0, y0


def _fill_grid(
    painter: Painter, values: NDArray[np.float64], x0: int, y0: int, color: ColorLike, alpha: float
) -> None:
    """Composite `color` weighted by `values`, a grid whose top-left pixel is `(x0, y0)`."""
    h, w = values.shape
    cx0, cy0 = max(x0, 0), max(y0, 0)
    cx1, cy1 = min(x0 + w, painter.width), min(y0 + h, painter.height)
    if cx0 < cx1 and cy0 < cy1:
        crop = values[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]
        painter._fill((cx0, cy0, cx1, cy1), crop, rgb(color), alpha)


def box_shadow(
    painter: Painter,
    left: float,
    top: float,
    right: float,
    bottom: float,
    radius: Corners,
    *,
    blur: float,
    offset: tuple[float, float] = (0.0, 0.0),
    spread: float = 0.0,
    color: ColorLike = (0, 0, 0),
    alpha: float = 0.5,
) -> None:
    """A CSS `box-shadow` of a box with per-corner radii (blurred beyond the canvas too, so
    it fades the same where the frame cuts it)."""
    x0, y0 = left + offset[0] - spread, top + offset[1] - spread
    x1, y1 = right + offset[0] + spread, bottom + offset[1] + spread
    if x1 <= x0 or y1 <= y0:
        return
    radii = [(max(rx + spread, 0.0), max(ry + spread, 0.0)) for rx, ry in radius]
    soft, gx, gy = _blurred_box(x0, y0, x1, y1, radii, blur / 2)
    _fill_grid(painter, soft, gx, gy, color, alpha)


def inset_ring(
    painter: Painter,
    left: float,
    top: float,
    right: float,
    bottom: float,
    radius: Corners,
    width: float,
    color: ColorLike,
    *,
    alpha: float = 1.0,
) -> None:
    """A CSS `box-shadow: inset 0 0 0 <width>`: a band along the inside of the box's edge."""
    box = painter._box(left, top, right, bottom, pad=0.0)
    if box is None:
        return
    xs, ys = painter._centres(box)
    outer = rounded_box_coverage(xs, ys, left, top, right, bottom, radius)
    inner_radius = [(max(rx - width, 0.0), max(ry - width, 0.0)) for rx, ry in radius]
    inner = rounded_box_coverage(
        xs, ys, left + width, top + width, right - width, bottom - width, inner_radius
    )
    painter._fill(box, np.clip(outer - inner, 0.0, 1.0), rgb(color), alpha)


def inset_shadow(
    painter: Painter,
    left: float,
    top: float,
    right: float,
    bottom: float,
    radius: Corners,
    *,
    blur: float,
    color: ColorLike = (0, 0, 0),
    alpha: float = 0.5,
) -> None:
    """A CSS `box-shadow: inset 0 0 <blur>`: a soft darkening inside the box's edge."""
    inside, gx, gy = _blurred_box(left, top, right, bottom, radius, blur / 2)
    h, w = inside.shape
    xs = np.arange(gx, gx + w, dtype=np.float64)[None, :] + 0.5
    ys = np.arange(gy, gy + h, dtype=np.float64)[:, None] + 0.5
    within = rounded_box_coverage(xs, ys, left, top, right, bottom, radius)
    _fill_grid(painter, np.clip(1.0 - inside, 0.0, 1.0) * within, gx, gy, color, alpha)


def spaced_text(
    painter: Painter,
    x: float,
    y: float,
    text: str,
    color: ColorLike,
    *,
    size: float,
    bold: bool = False,
    spacing: float = 0.0,
    align: str = "left",
    alpha: float = 1.0,
) -> float:
    """Text with CSS `letter-spacing` (after every character), its capitals centred on `y`.

    Returns its width, the trailing spacing included, as CSS lays it out.
    """
    typeface = face(size, bold)
    widths = [typeface.width(ch) for ch in text]
    total = sum(widths) + spacing * len(text)
    left = x - total / 2 if align == "center" else (x - total if align == "right" else x)
    for ch, w in zip(text, widths):
        painter.text(left, y, ch, color, size=size, bold=bold, valign="middle", alpha=alpha)
        left += w + spacing
    return total


def mono_width(text: str, size: float, bold: bool = False) -> float:
    """The width of `mono_text`: the words, and the monospaced font's wide spaces."""
    typeface = face(size, bold)
    words = text.split(" ")
    return sum(typeface.width(w) for w in words) + MONO_SPACE * size * (len(words) - 1)


def mono_text(
    painter: Painter,
    x: float,
    y: float,
    text: str,
    color: ColorLike,
    *,
    size: float,
    bold: bool = False,
    align: str = "left",
    alpha: float = 1.0,
) -> float:
    """Figures in a monospaced face (tabular digits, spaces as wide as a digit), its
    capitals centred on `y`. Returns its width."""
    total = mono_width(text, size, bold)
    left = x - total / 2 if align == "center" else (x - total if align == "right" else x)
    typeface = face(size, bold)
    for word in text.split(" "):
        if word:
            painter.text(left, y, word, color, size=size, bold=bold, valign="middle", alpha=alpha)
        left += typeface.width(word) + MONO_SPACE * size
    return total


def curve_text(
    painter: Painter,
    points: NDArray[np.float64],
    text: str,
    color: ColorLike,
    *,
    size: float,
    bold: bool = False,
    serif: bool = False,
    letter_spacing: float = 0.0,
    alpha: float = 1.0,
) -> None:
    """`Painter.text_on_curve` at any size: text smaller than the fonts' smallest size (9 px)
    is drawn larger on a scratch canvas and scaled down, so it keeps its size."""
    if size >= 9:
        painter.text_on_curve(
            points,
            text,
            color,
            size=size,
            bold=bold,
            serif=serif,
            letter_spacing=letter_spacing,
            alpha=alpha,
        )
        return
    k = math.ceil(9 / max(size, 0.5))
    pad = 3 * size
    x0 = math.floor(float(np.min(points[:, 0])) - pad)
    y0 = math.floor(float(np.min(points[:, 1])) - pad)
    x1 = math.ceil(float(np.max(points[:, 0])) + pad)
    y1 = math.ceil(float(np.max(points[:, 1])) + pad)
    scratch = Canvas((x1 - x0) * k, (y1 - y0) * k, (0, 0, 0), antialias=True)
    Painter(scratch).text_on_curve(
        (points - np.array([x0, y0], dtype=np.float64)) * k,
        text,
        (255, 255, 255),
        size=size * k,
        bold=bold,
        serif=serif,
        letter_spacing=letter_spacing * k,
    )
    big = scratch.pixels[..., 0].astype(np.float64) / 255.0
    painter.mask(big.reshape(y1 - y0, k, x1 - x0, k).mean(axis=(1, 3)), x0, y0, color, alpha=alpha)


def _quadratic(
    p0: tuple[float, float], c: tuple[float, float], p2: tuple[float, float], n: int = 96
) -> NDArray[np.float64]:
    """Points along a quadratic Bezier curve (an SVG `Q` path)."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    a, b, e = (np.asarray(v, dtype=np.float64)[None, :] for v in (p0, c, p2))
    out: NDArray[np.float64] = (1 - t) ** 2 * a + 2 * (1 - t) * t * b + t**2 * e
    return out


def _fade(
    painter: Painter, box: tuple[int, int, int, int], before: NDArray[np.uint8], opacity: float
) -> None:
    """CSS `opacity`: blend what was drawn into `box` since `before` was copied."""
    x0, y0, x1, y1 = box
    now = painter.canvas.pixels[y0:y1, x0:x1].astype(np.float32)
    old = before.astype(np.float32)
    painter.canvas.pixels[y0:y1, x0:x1] = (old + (now - old) * opacity + 0.5).astype(np.uint8)


# ----------------------------------------------------------------------
# Sprites: cards, chips and plates drawn once, then composited
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class _Sprite:
    """An RGBA image as premultiplied colour and opacity, both `uint8`, `(h, w, 3)` and
    `(h, w, 1)`, with its top-left `(dx, dy)` from the point it is drawn at."""

    premultiplied: NDArray[np.uint8]
    alpha: NDArray[np.uint8]
    dx: int
    dy: int

    def draw(self, painter: Painter, x: int, y: int) -> None:
        x0, y0 = x + self.dx, y + self.dy
        h, w = self.alpha.shape[:2]
        cx0, cy0 = max(x0, 0), max(y0, 0)
        cx1, cy1 = min(x0 + w, painter.width), min(y0 + h, painter.height)
        if cx0 >= cx1 or cy0 >= cy1:
            return
        rows, cols = slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0)
        a = self.alpha[rows, cols].astype(np.float32) * np.float32(1 / 255)
        view = painter.canvas.pixels[cy0:cy1, cx0:cx1]
        view[...] = (view * (1 - a) + self.premultiplied[rows, cols] + 0.5).astype(np.uint8)


def _sprite(on_black: NDArray[np.uint8], on_white: NDArray[np.uint8], dx: int, dy: int) -> _Sprite:
    """The sprite that, drawn over black and over white, gives these two images."""
    spread = on_white.astype(np.float64) - on_black.astype(np.float64)
    alpha = np.clip(255.0 - np.mean(spread, axis=-1, keepdims=True), 0.0, 255.0)
    premultiplied = on_black.copy()
    opacity = np.round(alpha).astype(np.uint8)
    premultiplied.flags.writeable = False
    opacity.flags.writeable = False
    return _Sprite(premultiplied, opacity, dx, dy)


def _drawn(width: int, height: int, dx: int, dy: int, draw: Any) -> _Sprite:
    """A sprite of whatever `draw(painter)` draws on a `width` x `height` canvas."""
    layers = []
    for background in ((0, 0, 0), (255, 255, 255)):
        canvas = Canvas(width, height, background, antialias=True)
        draw(Painter(canvas))
        layers.append(canvas.pixels)
    return _sprite(layers[0], layers[1], dx, dy)


@functools.lru_cache(maxsize=256)
def _card_sprite(card: str | None, width: float, brightness: float) -> _Sprite:
    """A card and its shadow, drawn at the point given as its top-left corner."""
    pad = math.ceil(0.25 * width) + 2
    sprite = _drawn(
        math.ceil(width) + 2 * pad,
        math.ceil(card_height(width)) + 2 * pad,
        -pad,
        -pad,
        lambda p: draw_card(p, pad, pad, width, card),
    )
    if brightness == 1.0:
        return sprite
    # CSS brightness(): the shadow is black, so only the card changes.
    dimmed = np.round(sprite.premultiplied * brightness).astype(np.uint8)
    dimmed.flags.writeable = False
    return _Sprite(dimmed, sprite.alpha, sprite.dx, sprite.dy)


@functools.lru_cache(maxsize=16)
def _chip_sprite(diameter: float, scale: float) -> _Sprite:
    """The bet chip and its shadow, drawn at the point given as its centre."""
    r = diameter / 2
    pad = math.ceil(6 * scale) + 2
    size = math.ceil(diameter) + 2 * pad
    centre = pad + r

    def draw(p: Painter) -> None:
        c = centre
        p.shadow(
            c - r, c - r, c + r, c + r, blur=4 * scale, offset=(0, 2 * scale), radius=r, alpha=0.4
        )
        # radial-gradient(circle, red 0 52%, ivory 53% 60%, red 61%), farthest-corner sized
        stops = (
            (0.0, CHIP_RED),
            (0.52, CHIP_RED),
            (0.53, IVORY),
            (0.60, IVORY),
            (0.61, CHIP_RED),
            (1.0, CHIP_RED),
        )
        p.radial_disc(c, c, r, stops, spread=math.sqrt(0.5))

    return _drawn(size, size, -round(centre), -round(centre), draw)


@functools.lru_cache(maxsize=32)
def _plate_sprite(
    name: str, width: float, height: float, scale: float, turn: bool, theme: GameTheme
) -> _Sprite:
    """A seat's name plate, its avatar and name (the bankroll is drawn on it each frame),
    with its shadow, drawn at the point given as its top-left corner."""
    s, dark = scale, theme.name == "dark"
    pad = math.ceil(70 * s) + 2
    radius = height / 2

    def draw(p: Painter) -> None:
        left, top = float(pad), float(pad)
        right, bottom = left + width, top + height
        p.shadow(  # --shadow: 0 12px 36px -12px
            left,
            top,
            right,
            bottom,
            blur=36 * s,
            offset=(0, 12 * s),
            spread=(-12 if dark else -14) * s,
            radius=radius,
            color=theme.shadow,
            alpha=0.6 if dark else 0.28,
        )
        if turn:  # 0 0 0 3px rgba(201, 180, 139, .25)
            p.rounded_outline(
                left - 1.5 * s,
                top - 1.5 * s,
                right + 1.5 * s,
                bottom + 1.5 * s,
                3 * s,
                GOLD,
                radius=radius + 1.5 * s,
                alpha=0.25,
            )
        p.rect(left, top, right, bottom, PLATE, alpha=0.9, radius=radius)
        border = s
        p.rounded_outline(
            left + border / 2,
            top + border / 2,
            right - border / 2,
            bottom - border / 2,
            border,
            GOLD,
            radius=radius - border / 2,
            alpha=1.0 if turn else 0.35,
        )
        # The avatar (28 px): gold initials on a dark green linear-gradient(145deg, ...).
        r = 14 * s
        ax, ay = left + 7 * s + r, top + height / 2
        p.linear_rect(
            ax - r,
            ay - r,
            ax + r,
            ay + r,
            ((0.0, "#2c3b29"), (1.0, "#1d271a")),
            start=(0.100, -0.071),
            end=(0.900, 1.071),
            radius=r,
        )
        p.ring(ax, ay, r - 0.5 * s, s, "#46563f")
        p.text(
            ax, ay, initials(name), BRASS, size=10 * s, bold=True, align="center", valign="middle"
        )
        line = (13 + 12) * s * LINE
        name_y = top + (height - line) / 2 + 13 * s * LINE / 2
        p.text(ax + r + 9 * s, name_y, name, IVORY, size=13 * s, bold=True, valign="middle")

    w, h = math.ceil(width) + 2 * pad, math.ceil(height) + 2 * pad
    return _drawn(w, h, -pad, -pad, draw)


# ----------------------------------------------------------------------
# The renderers
# ----------------------------------------------------------------------


def _scene(env: Any) -> tuple[dict[str, Any], bool, BlackjackTable]:
    """The public view to draw, whether the env's round is over, and the table."""
    if isinstance(env, BlackjackEnv):
        if env.table is None:
            raise ResetRequired()
        return env.table.view(0), env.done, env.table
    table: BlackjackTable = env
    return table.view(), False, table


def _seat_names(names: str | Sequence[str] | Mapping[Any, str] | None) -> dict[int, str]:
    if names is None:
        return {}
    if isinstance(names, str):
        return {0: names}
    if isinstance(names, Mapping):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(n) for i, n in enumerate(names)}


class _Named:
    seat_names: dict[int, str]

    def seat_name(self, seat: int, seats: int) -> str:
        """A seat's name: the one given, else `Player` (or `Player 2` at a bigger table)."""
        if seat in self.seat_names:
            return self.seat_names[seat]
        return "Player" if seats == 1 else f"Player {seat + 1}"


@dataclass(frozen=True)
class _SeatLayout:
    width: float
    row_height: float
    hands: list[tuple[float, float]]
    compact: bool
    plate: tuple[float, float]


class BlackjackRenderer(_Named, GameRenderer):
    """The blackjack table as RGB frames.

    The table is `width` wide and `0.56 * width` tall, under a header with the round's
    status; the frame adds margins. Lengths are in the unit `u = width / 100` (cards are
    `7.2u` wide), and pixel sizes scale with `width / 640`. The env's seat sits centred at
    the bottom; a `BlackjackTable` with more seats spreads them out.

    Args:
        width: The table's width in pixels (at least 200).
        theme: `"dark"` (the default), `"light"` or a `GameTheme`. The felt is the same in
            both; the page around it and the shadows follow the theme.
        header: Draw the header (the game's name and status).
        names: The seat's name (`"Player"` by default), or one name per seat of a table
            (a sequence, or a mapping from seat index to name).
    """

    title: ClassVar[str] = "Blackjack"
    env_types: ClassVar[tuple[type, ...]] = (BlackjackEnv, BlackjackTable)

    def __init__(
        self,
        width: int = 640,
        *,
        theme: GameTheme | str = "dark",
        header: bool = True,
        names: str | Sequence[str] | Mapping[Any, str] | None = None,
    ) -> None:
        if width < 200:
            raise ValueError(f"width must be at least 200 pixels, got {width}.")
        super().__init__(theme=theme, header=header)
        self.seat_names = _seat_names(names)
        self.names = {str(k): v for k, v in self.seat_names.items()}
        self.width = int(width)

    # -- layout -----------------------------------------------------------------------

    @property
    def scale(self) -> float:
        return self.width / 640

    @property
    def unit(self) -> float:
        """The layout's unit `u`: a hundredth of the table's width."""
        return self.width / 100

    def table_height(self) -> float:
        """The table's height: `0.56 * width`."""
        return 0.56 * self.width

    def frame_size(self) -> tuple[int, int]:
        m = self.margin()
        return self.width + 2 * round(m), round(2 * m + self.header_height() + self.table_height())

    def table_box(self) -> tuple[float, float, float, float]:
        """`(left, top, right, bottom)` of the table."""
        left = round(self.margin())
        top = round(self.margin() + self.header_height())
        return left, top, left + self.width, top + self.table_height()

    def corners(self) -> Corners:
        """The table's corner radii: `46% 46% 18px 18px / 30% 30% 18px 18px`."""
        w, h, s = self.width, self.table_height(), self.scale
        return [(0.46 * w, 0.30 * h), (0.46 * w, 0.30 * h), (18 * s, 18 * s), (18 * s, 18 * s)]

    # -- the header -------------------------------------------------------------------

    def check_ready(self, env: Any) -> None:
        if isinstance(env, BlackjackEnv) and env.table is None:
            raise ResetRequired()

    def status(self, env: Any) -> Status:
        view, done, _ = _scene(env)
        seats = view["seats"]
        if done:
            last = view["last_round"]
            net = last["seats"][0]["net"] if last else 0
            if last and last["seats"][0].get("forfeit"):
                how = "No cards dealt" if not last["seats"][0]["hands"] else "Every hand lost"
                bankroll = format_chips(seats[0]["bankroll"])
                sub = f"An illegal action. {how} \u00b7 bankroll {bankroll}"
                return Status(f"Round forfeited: {format_signed(net, 0)}", sub, "loss")
            dealer = last["dealer"] if last else view["dealer"]
            if dealer["blackjack"]:
                sub = "Dealer blackjack"
            elif dealer["bust"]:
                sub = f"Dealer busts with {dealer['total']}"
            else:
                sub = f"Dealer {dealer['total']}"
            sub += f" \u00b7 bankroll {format_chips(seats[0]['bankroll'])}"
            tone: Tone = "win" if net > 0 else ("loss" if net < 0 else "draw")
            return Status(f"Round over: {format_signed(net, 0)}", sub, tone)
        seat = view["to_act"]
        if seat is None:
            return Status("Every seat is out of chips", f"Round {view['round_no']}", "loss")
        many = len(seats) > 1
        name = self.seat_name(seat, len(seats))
        if view["phase"] == "betting":
            last = view["last_round"]
            played = last["seats"][seat] if last else None
            if played and played["played"]:
                sub = f"Last round {format_signed(played['net'], 0)}"
            else:
                sub = f"Round {view['round_no']}"
            return Status(f"{name} to bet" if many else "Place your bet", sub, "yours")
        hands = seats[seat]["hands"]
        h = view["active_hand"] or 0
        up = _up_label(view["dealer"]["cards"][0])
        parts = [name] if many else []
        if len(hands) > 1:
            parts.append(f"hand {h + 1} of {len(hands)}")
        parts.append(f"round {view['round_no']}")
        sub = " \u00b7 ".join(parts)
        text = f"{total_label(hands[h])} against the dealer's {up}"
        return Status(text, sub[:1].upper() + sub[1:], "yours")

    # -- the felt (cached) --------------------------------------------------------------

    def static_key(self, env: Any) -> Any:
        return print_lines(_table(env).rules)

    def draw_static(self, painter: Painter, env: Any) -> None:
        s, u, theme = self.scale, self.unit, self.theme
        left, top, right, bottom = self.table_box()
        corners = self.corners()
        box_shadow(  # --shadow-lg: 0 30px 80px -24px
            painter,
            left,
            top,
            right,
            bottom,
            corners,
            blur=80 * s,
            offset=(0, 30 * s),
            spread=(-24 if theme.name == "dark" else -28) * s,
            color=theme.shadow,
            alpha=theme.shadow_alpha,
        )
        painter.radial_rect(
            left, top, right, bottom, FELT_STOPS, centre=(0.5, 0.18), radius=corners
        )
        inset_shadow(painter, left, top, right, bottom, corners, blur=7 * u, alpha=0.45)
        inset_ring(painter, left, top, right, bottom, corners, 1.3 * u, BRASS)
        inset_ring(painter, left, top, right, bottom, corners, 1.1 * u, RAIL)
        # The printed arcs, laid out on a 1000 x 560 grid and scaled to the table.
        k = self.width / 1000
        main, sub = print_lines(_table(env).rules)
        arc = _quadratic((220, 176), (500, 286), (780, 176)) * k + (left, top)
        curve_text(
            painter,
            arc,
            main,
            GOLD,
            size=25 * k,
            serif=True,
            letter_spacing=0.16 * 25 * k,
            alpha=0.52,
        )
        arc = _quadratic((262, 204), (500, 300), (738, 204)) * k + (left, top)
        curve_text(
            painter,
            arc,
            sub,
            GOLD,
            size=11.5 * k,
            bold=True,
            letter_spacing=0.24 * 11.5 * k,
            alpha=0.4,
        )

    # -- the dealer, the shoe and the seats -----------------------------------------------

    def draw_surface(self, painter: Painter, env: Any) -> None:
        view, done, _ = _scene(env)
        settled = view["phase"] == "betting"
        self._draw_dealer(painter, view, settled)
        self._draw_shoe(painter, view)
        self._draw_seats(painter, view, done, settled)

    def _pill(
        self,
        painter: Painter,
        left: float,
        cy: float,
        text: str,
        *,
        size: float,
        padding: float,
        color: ColorLike,
        background: ColorLike,
        background_alpha: float,
        mono: bool,
    ) -> None:
        """A pill `2.6u` tall, its left edge at `left`."""
        height = 2.6 * self.unit
        width = self._pill_width(text, size, padding, mono)
        painter.rect(
            left,
            cy - height / 2,
            left + width,
            cy + height / 2,
            background,
            alpha=background_alpha,
            radius=height / 2,
        )
        if mono:
            mono_text(
                painter, left + width / 2, cy, text, color, size=size, bold=True, align="center"
            )
        else:
            painter.text(
                left + width / 2,
                cy,
                text,
                color,
                size=size,
                bold=True,
                align="center",
                valign="middle",
            )

    @staticmethod
    def _pill_width(text: str, size: float, padding: float, mono: bool) -> float:
        width = mono_width(text, size, True) if mono else face(size, True).width(text)
        return width + 2 * padding

    def _draw_cards(
        self,
        painter: Painter,
        cards: Sequence[str | None],
        centre: float,
        top: float,
        width: float,
        overlap: float,
        settled: bool,
    ) -> None:
        step = width - overlap
        left = centre - (width + step * (len(cards) - 1)) / 2
        for i, card in enumerate(cards):
            sprite = _card_sprite(card, round(width, 2), 0.86 if settled else 1.0)
            sprite.draw(painter, round(left + i * step), round(top))

    def _draw_dealer(self, painter: Painter, view: Mapping[str, Any], settled: bool) -> None:
        u = self.unit
        left, top, right, _ = self.table_box()
        cx = (left + right) / 2
        y = top + 0.06 * self.table_height()
        label = 1.35 * u * LINE
        spaced_text(
            painter,
            cx,
            y + label / 2,
            "DEALER",
            LINEN,
            size=1.35 * u,
            bold=True,
            spacing=0.2 * 1.35 * u,
            align="center",
            alpha=0.6,
        )
        y += label + 0.8 * u
        dealer = view["dealer"]
        shown = dealer["cards"]
        if not shown and view["last_round"]:
            shown = view["last_round"]["dealer"]["cards"]
        if shown:
            self._draw_cards(painter, shown, cx, y, 7.2 * u, 3.2 * u, settled)
            y += card_height(7.2 * u)
        text = dealer_label(dealer)
        if text:
            size, padding = 1.5 * u, 1.1 * u
            bust = dealer["bust"]
            self._pill(
                painter,
                cx - self._pill_width(text, size, padding, True) / 2,
                y + 0.8 * u + 1.3 * u,
                text,
                size=size,
                padding=padding,
                color=IVORY,
                background=BUST if bust else PILL,
                background_alpha=0.85 if bust else 0.65,
                mono=True,
            )

    def _draw_shoe(self, painter: Painter, view: Mapping[str, Any]) -> None:
        s, u = self.scale, self.unit
        left, top, _, _ = self.table_box()
        shoe = view["shoe"]
        right = left + 0.89 * self.width
        x0, bar = right - 15 * u, 15 * u
        y = top + 0.17 * self.table_height()
        height = 6 * s
        painter.rect(x0, y, right, y + height, (0, 0, 0), alpha=0.3, radius=height / 2)
        box = painter._box(x0, y, right, y + height, pad=0.0)
        if box is not None:  # the fill and the reshuffle mark, inside the rounded bar
            xs, ys = painter._centres(box)
            inside = rounded_box_coverage(xs, ys, x0, y, right, y + height, height / 2)
            fill = np.clip(x0 + bar * shoe["remaining"] / shoe["total"] - (xs - 0.5), 0, 1)
            painter._fill(box, inside * fill, rgb(GOLD), 0.8)
            mark = x0 + bar * (1 - shoe["penetration"])
            marker = np.clip(np.minimum(xs + 0.5, mark + 2 * s) - np.maximum(xs - 0.5, mark), 0, 1)
            painter._fill(box, inside * marker, rgb(MARKER), 1.0)
        size = 1.3 * u
        y += height + 4 * s + size * LINE / 2
        count = (
            f"Count {format_signed(view['running_count'], 0)} \u00b7 "
            f"True {format_signed(view['true_count'], 1)}"
        )
        mono_text(painter, right, y, count, LINEN, size=size, align="right", alpha=0.7)
        if shoe["reshuffle_next"]:
            y += size * LINE + 4 * s
            painter.text(
                right,
                y,
                "Shuffle after this round",
                LINEN,
                size=size,
                align="right",
                valign="middle",
                alpha=0.7,
            )

    def _hand_size(self, hand: Mapping[str, Any], compact: bool) -> tuple[float, float]:
        s, u = self.scale, self.unit
        width = (5.4 if compact else 7.2) * u
        step = width - (3.4 if compact else 3.2) * u
        cards = width + step * (len(hand["cards"]) - 1)
        items = self._meta(hand)
        meta = sum(w for _, w in items) + 5 * s * (len(items) - 1)
        return max(cards, meta) + 8 * s, 14 * s + card_height(width) + 2.6 * u

    def _meta(self, hand: Mapping[str, Any]) -> list[tuple[str, float]]:
        """What the hand shows under its cards, and their widths."""
        u = self.unit
        items = [("total", self._pill_width(total_label(hand), 1.5 * u, 1.1 * u, True))]
        if hand["doubled"]:
            items.append(("flag", mono_width("\u00d72", 1.3 * u, True)))
        if hand["outcome"]:
            items.append(("outcome", self._pill_width(outcome_label(hand), 1.35 * u, u, False)))
        return items

    def _plate_size(self, name: str, bankroll: str) -> tuple[float, float]:
        s = self.scale
        text = max(face(13 * s, True).width(name), mono_width(bankroll, 12 * s))
        # 1px border, 5px 14px 5px 6px padding, the 28px avatar and a 9px gap
        return 59 * s + text, (13 + 12) * s * LINE + 12 * s

    def _seat_layout(self, seat: Mapping[str, Any], name: str, seats: int) -> _SeatLayout:
        s, u = self.scale, self.unit
        hands = seat["hands"]
        compact = len(hands) > 2
        sizes = [self._hand_size(h, compact) for h in hands]
        if sizes:
            row = sum(w for w, _ in sizes) + (0.8 if compact else 2.0) * u * (len(sizes) - 1)
        else:
            row = face(12 * s).width("Out of chips") if seat["out"] else 0.0
        plate = self._plate_size(name, format_chips(seat["bankroll"]))
        # min-width: 30% (less at a table whose seats would not fit side by side)
        width = max(row, 5 * u, plate[0], 0.88 * self.width * min(0.3, 1 / seats))
        row_height = max([15 * u, *(h for _, h in sizes)])
        return _SeatLayout(width, row_height, sizes, compact, plate)

    def _draw_seats(
        self, painter: Painter, view: Mapping[str, Any], done: bool, settled: bool
    ) -> None:
        left, top, _, _ = self.table_box()
        seats = view["seats"]
        names = [self.seat_name(i, len(seats)) for i in range(len(seats))]
        layouts = [self._seat_layout(seat, names[i], len(seats)) for i, seat in enumerate(seats)]
        free = 0.88 * self.width - sum(lay.width for lay in layouts)
        # justify-content: space-around (centred when the seats overflow)
        margin = max(free, 0.0) / len(layouts) / 2
        x = left + 0.06 * self.width + (margin if free >= 0 else free / 2)
        bottom = top + 0.95 * self.table_height()
        for i, seat in enumerate(seats):
            lay = layouts[i]
            is_turn = view["to_act"] == i and not done
            active = view["active_hand"] if view["phase"] == "playing" and is_turn else None
            self._draw_seat(
                painter, seat, names[i], lay, x + lay.width / 2, bottom, is_turn, active, settled
            )
            x += lay.width + 2 * margin

    def _draw_seat(
        self,
        painter: Painter,
        seat: Mapping[str, Any],
        name: str,
        lay: _SeatLayout,
        centre: float,
        bottom: float,
        is_turn: bool,
        active: int | None,
        settled: bool,
    ) -> None:
        s, u = self.scale, self.unit
        plate_w, plate_h = lay.plate
        plate_top = bottom - plate_h
        spot_top = plate_top - 0.9 * u - 5 * u
        row_bottom = spot_top - 0.9 * u
        before: NDArray[np.uint8] | None = None
        box = None
        if seat["out"]:  # .is-out: the seat at 55% opacity
            margin = 80 * s
            box = painter._box(
                centre - lay.width / 2 - margin,
                row_bottom - lay.row_height - margin,
                centre + lay.width / 2 + margin,
                bottom + margin,
                pad=0.0,
            )
            if box is not None:
                before = painter.canvas.pixels[box[1] : box[3], box[0] : box[2]].copy()
        hands = seat["hands"]
        if hands:
            gap = (0.8 if lay.compact else 2.0) * u
            x = centre - (sum(w for w, _ in lay.hands) + gap * (len(hands) - 1)) / 2
            for h, (hand, (hw, hh)) in enumerate(zip(hands, lay.hands)):
                highlight = active == h and len(hands) > 1
                self._draw_hand(
                    painter, hand, x, row_bottom - hh, hw, hh, lay.compact, highlight, settled
                )
                x += hw + gap
        elif seat["out"]:
            painter.text(
                centre,
                row_bottom - 6 * s * LINE,
                "Out of chips",
                LINEN,
                size=12 * s,
                align="center",
                valign="middle",
                alpha=0.5,
            )
        self._draw_spot(painter, seat["bet"], centre, spot_top + 2.5 * u)
        plate_left = centre - plate_w / 2
        sprite = _plate_sprite(name, round(plate_w, 2), round(plate_h, 2), s, is_turn, self.theme)
        sprite.draw(painter, round(plate_left), round(plate_top))
        line = (13 + 12) * s * LINE
        y = round(plate_top) + (plate_h - line) / 2 + 13 * s * LINE + 12 * s * LINE / 2
        mono_text(
            painter,
            round(plate_left) + 44 * s,
            y,
            format_chips(seat["bankroll"]),
            GOLD,
            size=12 * s,
        )
        if box is not None and before is not None:
            _fade(painter, box, before, 0.55)

    def _draw_hand(
        self,
        painter: Painter,
        hand: Mapping[str, Any],
        left: float,
        top: float,
        width: float,
        height: float,
        compact: bool,
        highlight: bool,
        settled: bool,
    ) -> None:
        s, u = self.scale, self.unit
        if highlight:  # .is-active
            radius = 10 * s
            painter.rect(left, top, left + width, top + height, GOLD, alpha=0.16, radius=radius)
            painter.rounded_outline(
                left - 0.5,
                top - 0.5,
                left + width + 0.5,
                top + height + 0.5,
                1.0,
                GOLD,
                radius=radius + 0.5,
                alpha=0.5,
            )
        card_w = (5.4 if compact else 7.2) * u
        cx = left + width / 2
        self._draw_cards(
            painter, hand["cards"], cx, top + 4 * s, card_w, (3.4 if compact else 3.2) * u, settled
        )
        cy = top + 10 * s + card_height(card_w) + 1.3 * u
        items = self._meta(hand)
        x = cx - (sum(w for _, w in items) + 5 * s * (len(items) - 1)) / 2
        for kind, w in items:
            if kind == "total":
                if hand["blackjack"]:
                    bg, alpha, ink = BRASS, 1.0, "#1a2117"
                elif hand["bust"]:
                    bg, alpha, ink = BUST, 0.85, IVORY
                else:
                    bg, alpha, ink = PILL, 0.65, IVORY
                self._pill(
                    painter,
                    x,
                    cy,
                    total_label(hand),
                    size=1.5 * u,
                    padding=1.1 * u,
                    color=ink,
                    background=bg,
                    background_alpha=alpha,
                    mono=True,
                )
            elif kind == "flag":
                mono_text(painter, x, cy, "\u00d72", GOLD, size=1.3 * u, bold=True)
            else:
                bg, alpha, ink = {
                    "win": (WIN, 1.0, INK),
                    "blackjack": (BRASS, 1.0, INK),
                    "lose": (BUST, 0.85, "#ffffff"),
                }.get(hand["outcome"], (PUSH, 1.0, "#e6dfd2"))
                self._pill(
                    painter,
                    x,
                    cy,
                    outcome_label(hand),
                    size=1.35 * u,
                    padding=u,
                    color=ink,
                    background=bg,
                    background_alpha=alpha,
                    mono=False,
                )
            x += w + 5 * s

    def _draw_spot(self, painter: Painter, bet: int | None, cx: float, cy: float) -> None:
        s, u = self.scale, self.unit
        r = 2.5 * u
        if bet:
            _chip_sprite(round(2 * r, 2), round(s, 4)).draw(painter, round(cx), round(cy))
            mono_text(
                painter,
                round(cx),
                round(cy),
                format_chips(bet),
                "#ffffff",
                size=1.35 * u,
                bold=True,
                align="center",
            )
        else:  # an empty ring: border 1.5px rgba(201, 180, 139, .45)
            painter.ring(cx, cy, r - 0.75 * s, 1.5 * s, GOLD, alpha=0.45)


def _table(env: Any) -> BlackjackTable:
    table: BlackjackTable | None = env.table if isinstance(env, BlackjackEnv) else env
    if table is None:
        raise ResetRequired()
    return table


def _plain_signed(value: float, digits: int = 0) -> str:
    return format_signed(value, digits).replace("\u2212", "-")


def _cards_text(cards: Sequence[str | None]) -> str:
    return " ".join(c if c else "??" for c in cards)


class BlackjackTextRenderer(_Named, BaseRenderer):
    """The table as a few lines of text: the round, the dealer, each seat, the shoe.

    Between rounds it shows the seats' bankrolls and last results rather than the cards of
    the round before. Reads only the public table view.

    Args:
        names: The seat's name (`"Player"` by default), or one name per seat of a table.
    """

    def __init__(self, names: str | Sequence[str] | Mapping[Any, str] | None = None) -> None:
        self.seat_names = _seat_names(names)

    def render(self, env: Any) -> str:
        base = unwrap(env, (BlackjackEnv, BlackjackTable))
        view, done, table = _scene(base)
        last = view["last_round"]
        seats = view["seats"]
        playing = view["phase"] == "playing"
        if done:
            net = last["seats"][0]["net"] if last else 0
            state = f"round {last['round'] if last else view['round_no']} over, net "
            state += _plain_signed(net)
            if last and last["seats"][0].get("forfeit"):
                state += " (forfeited: an illegal action)"
        elif view["to_act"] is None and not playing:
            state = f"round {view['round_no']}, every seat is out of chips"
        else:
            state = f"round {view['round_no']}, {view['phase']}"
        lines = [f"Blackjack ({table.rules.label}), {state}"]
        if (playing or done) and view["dealer"]["cards"]:  # (none in a round forfeited unplayed)
            dealer = view["dealer"]
            label = dealer_label(dealer).lower()
            lines.append(f"Dealer: {_cards_text(dealer['cards'])} ({label})")
        for i, seat in enumerate(seats):
            parts = [f"bankroll {format_chips(seat['bankroll'])}"]
            if seat["bet"] is not None:
                parts.append(f"bet {seat['bet']}")
            if not (playing or done) and last and last["seats"][i]["played"]:
                parts.append(f"last round {_plain_signed(last['seats'][i]['net'])}")
            if seat["out"]:
                parts.append("out")
            if view["to_act"] == i and not done:
                parts.append("to act" if playing else "to bet")
            lines.append(f"{self.seat_name(i, len(seats))}: " + ", ".join(parts))
            if not (playing or done):
                continue
            for h, hand in enumerate(seat["hands"]):
                acting = playing and view["to_act"] == i and view["active_hand"] == h
                text = f"  hand {h + 1}{'*' if acting else ''}: {_cards_text(hand['cards'])}"
                text += f" ({total_label(hand).lower()}), bet {hand['bet']}"
                if hand["doubled"]:
                    text += " doubled"
                if hand["outcome"]:
                    payout = hand["payout"] or 0
                    text += f", {hand['outcome']}" + (f" {_plain_signed(payout)}" if payout else "")
                lines.append(text)
        shoe = view["shoe"]
        lines.append(
            f"Shoe: {shoe['remaining']}/{shoe['total']} cards left, running count "
            f"{_plain_signed(view['running_count'])}, true count "
            f"{_plain_signed(view['true_count'], 1)}"
            + (", shuffle after this round" if shoe["reshuffle_next"] else "")
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return "BlackjackTextRenderer()"


__all__ = [
    "BlackjackRenderer",
    "BlackjackTextRenderer",
    "dealer_label",
    "format_chips",
    "format_signed",
    "outcome_label",
    "print_lines",
    "total_label",
]
