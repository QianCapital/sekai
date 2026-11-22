"""Playing cards, chips and seat plates for the poker and blackjack renderers.

A card has an ivory face with the rank over the suit in two corners (the lower one turned
upside down) and a large pip in the middle, and a forest-green back with a gold border and
cross-hatching; dashed slots mark where community cards will come. Suits are drawn as
shapes rather than font glyphs, so they need no font.

Sizes are in pixels; a card is `width` wide and `1.4 * width` tall.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from sekai.envs.games._draw import Painter, rounded_box_coverage
from sekai.rendering.text import face
from sekai.rendering.theme import ColorLike, rgb

CARD_FACE = "#fbfaf6"
CARD_EDGE = "#d9d3c4"
CARD_RED = "#b8352b"
CARD_BLACK = "#1b2118"
BACK_FACE = "#22382c"
BACK_FIELD = "#1f3428"
BACK_EDGE = "#b39a6c"
GOLD = "#c9b48b"
BRAND_GOLD = "#c0aa81"
ASPECT = 1.4

RED_SUITS = frozenset("hd")


def suit_color(suit: str) -> str:
    """The ink of a suit (`"s"`, `"h"`, `"d"` or `"c"`)."""
    return CARD_RED if suit in RED_SUITS else CARD_BLACK


def rank_text(rank: str) -> str:
    """The rank as printed on a card: `"T"` is `"10"`."""
    return "10" if rank in ("T", "t") else rank.upper()


# ----------------------------------------------------------------------
# Suits, as unions of discs and polygons in a unit box (y down)
# ----------------------------------------------------------------------

_Disc = tuple[float, float, float]
_Poly = tuple[tuple[float, float], ...]

_SUITS: dict[str, tuple[tuple[_Disc, ...], tuple[_Poly, ...]]] = {
    "h": (
        ((0.285, 0.33, 0.255), (0.715, 0.33, 0.255)),
        (((0.045, 0.40), (0.5, 0.97), (0.955, 0.40), (0.5, 0.30)),),
    ),
    "s": (
        ((0.285, 0.58, 0.235), (0.715, 0.58, 0.235)),
        (
            ((0.5, 0.02), (0.06, 0.52), (0.5, 0.72), (0.94, 0.52)),
            ((0.5, 0.62), (0.30, 0.99), (0.70, 0.99)),
        ),
    ),
    "d": ((), (((0.5, 0.0), (0.86, 0.5), (0.5, 1.0), (0.14, 0.5)),)),
    "c": (
        ((0.5, 0.27, 0.215), (0.265, 0.6, 0.215), (0.735, 0.6, 0.215), (0.5, 0.55, 0.13)),
        (((0.5, 0.52), (0.31, 0.99), (0.69, 0.99)),),
    ),
}


def _polygon_coverage(
    xs: NDArray[np.float64], ys: NDArray[np.float64], points: Sequence[tuple[float, float]]
) -> NDArray[np.float64]:
    """Anti-aliased coverage of a polygon at pixel centres (signed distance to its edges)."""
    pts = np.asarray(points, dtype=np.float64)
    shape = (ys.shape[0], xs.shape[1])
    X, Y = np.broadcast_to(xs, shape), np.broadcast_to(ys, shape)
    inside = np.zeros(shape, dtype=bool)
    distance = np.full(shape, np.inf)
    xj, yj = pts[-1]
    for xi, yi in pts:
        with np.errstate(divide="ignore", invalid="ignore"):
            crosses = (yi > Y) != (yj > Y)
            x_at = (xj - xi) * (Y - yi) / (yj - yi) + xi
        inside ^= crosses & (x_at > X)
        dx, dy = xj - xi, yj - yi
        length2 = dx * dx + dy * dy
        t = np.clip(((X - xi) * dx + (Y - yi) * dy) / length2, 0, 1) if length2 else 0.0
        distance = np.minimum(distance, np.hypot(X - (xi + t * dx), Y - (yi + t * dy)))
        xj, yj = xi, yi
    signed = np.where(inside, -distance, distance)
    return np.asarray(np.clip(0.5 - signed, 0, 1), dtype=np.float64)


def draw_suit(
    painter: Painter,
    cx: float,
    cy: float,
    size: float,
    suit: str,
    color: ColorLike,
    *,
    alpha: float = 1.0,
) -> None:
    """A suit symbol `size` pixels tall centred on `(cx, cy)`."""
    discs, polys = _SUITS[suit]
    width = size * (0.86 if suit == "d" else 1.0)
    left, top = cx - width / 2, cy - size / 2
    box = painter._box(left, top, left + width, top + size)
    if box is None:
        return
    xs, ys = painter._centres(box)
    coverage = np.zeros((ys.shape[0], xs.shape[1]))
    for dx, dy, r in discs:
        d = np.hypot(xs - (left + dx * width), ys - (top + dy * size))
        coverage = np.maximum(coverage, np.clip(r * width + 0.5 - d, 0, 1))
    for poly in polys:
        scaled = [(left + x * width, top + y * size) for x, y in poly]
        coverage = np.maximum(coverage, _polygon_coverage(xs, ys, scaled))
    painter._fill(box, coverage, rgb(color), alpha)


# ----------------------------------------------------------------------
# Cards
# ----------------------------------------------------------------------


def card_height(width: float) -> float:
    return width * ASPECT


def draw_card(
    painter: Painter,
    left: float,
    top: float,
    width: float,
    card: str | None,
    *,
    highlight: bool = False,
    dim: bool = False,
    corners_only: bool = False,
) -> None:
    """A playing card with its top-left corner at `(left, top)`.

    `card` is rank then suit (`"As"`, `"Td"`) or `None` for a face-down card. `highlight`
    lifts the card and rings it in gold (the winning hand); `dim` darkens it (a folded
    hand). `corners_only` drops the centre pip and the lower corner (small cards).
    """
    height = card_height(width)
    if highlight:
        top -= 0.04 * height
    radius = 0.1 * width
    right, bottom = left + width, top + height
    painter.shadow(
        left,
        top,
        right,
        bottom,
        blur=0.12 * width,
        offset=(0, 0.04 * width),
        radius=radius,
        alpha=0.35,
    )
    if highlight:
        painter.shadow(
            left, top, right, bottom, blur=18 * width / 60, radius=radius, color=GOLD, alpha=0.55
        )
    if card is None:
        _draw_back(painter, left, top, width, height, radius)
    else:
        _draw_face(painter, left, top, width, height, radius, card, corners_only)
    if highlight:
        painter.rounded_outline(
            left - 1, top - 1, right + 1, bottom + 1, 2.0, BRAND_GOLD, radius=radius + 1
        )
    if dim:
        painter.adjust(left - 1, top - 1, right + 1, bottom + 1, brightness=0.62, saturate=0.6)


def _draw_face(
    painter: Painter,
    left: float,
    top: float,
    width: float,
    height: float,
    radius: float,
    card: str,
    corners_only: bool,
) -> None:
    right, bottom = left + width, top + height
    painter.rect(left, top, right, bottom, CARD_EDGE, radius=radius)
    painter.rect(left + 1, top + 1, right - 1, bottom - 1, CARD_FACE, radius=max(radius - 1, 0))
    painter.line(left + radius, top + 1.5, right - radius, top + 1.5, "#ffffff", 1.0, alpha=0.6)
    rank, suit = rank_text(card[0]), card[1].lower()
    ink = suit_color(suit)
    rank_size = 0.32 * width
    suit_size = 0.2 * width
    rank_face = face(rank_size, True)
    rank_w = rank_face.width(rank)
    column = max(rank_w, suit_size * 0.9)
    cx = left + 0.08 * width + column / 2
    caps_top = top + 0.05 * height + 0.045 * width
    _corner(painter, cx, caps_top, rank, suit, ink, rank_size, suit_size, flip=False)
    if corners_only:
        return
    _corner_flipped(painter, left, top, width, height, rank, suit, ink, rank_size, suit_size)
    painter_size = 0.42 * width
    draw_suit(painter, left + width / 2, top + height / 2 + 0.03 * width, painter_size, suit, ink)


def _corner(
    painter: Painter,
    cx: float,
    caps_top: float,
    rank: str,
    suit: str,
    ink: str,
    rank_size: float,
    suit_size: float,
    *,
    flip: bool,
) -> None:
    typeface = face(rank_size, True)
    painter.text(cx, caps_top, rank, ink, size=rank_size, bold=True, align="center", valign="top")
    suit_top = caps_top + typeface.cap_height + 0.22 * suit_size
    draw_suit(painter, cx, suit_top + suit_size / 2, suit_size, suit, ink)


def _corner_flipped(
    painter: Painter,
    left: float,
    top: float,
    width: float,
    height: float,
    rank: str,
    suit: str,
    ink: str,
    rank_size: float,
    suit_size: float,
) -> None:
    """The lower corner: the upper one turned through 180 degrees about the card's centre."""
    x0, y0 = math.floor(left), math.floor(top)
    x1, y1 = math.ceil(left + width), math.ceil(top + height)
    # Draw the upper corner on a scratch painter with the card's face behind it, then
    # composite its ink, rotated, into the lower corner.
    from sekai.rendering.canvas import Canvas

    w, h = x1 - x0, y1 - y0
    scratch = Canvas(w, h, rgb(CARD_FACE), antialias=True)
    sp = Painter(scratch)
    rank_face = face(rank_size, True)
    column = max(rank_face.width(rank), suit_size * 0.9)
    cx = (left - x0) + 0.08 * width + column / 2
    caps_top = (top - y0) + 0.05 * height + 0.045 * width
    _corner(sp, cx, caps_top, rank, suit, ink, rank_size, suit_size, flip=False)
    pixels = scratch.pixels.astype(np.float64)
    paper = np.asarray(rgb(CARD_FACE), dtype=np.float64)
    ink_rgb = np.asarray(rgb(ink), dtype=np.float64)
    # Coverage of the ink over the paper, recovered channel by channel.
    span = ink_rgb - paper
    channel = int(np.argmax(np.abs(span)))
    coverage = np.clip((pixels[..., channel] - paper[channel]) / span[channel], 0, 1)
    rotated = coverage[::-1, ::-1]
    xs = np.arange(x0, x1, dtype=np.float64)[None, :] + 0.5
    ys = np.arange(y0, y1, dtype=np.float64)[:, None] + 0.5
    inside = rounded_box_coverage(xs, ys, left, top, left + width, top + height, 0.1 * width)
    painter.texture(x0, y0, rotated * inside, ink)


def _draw_back(
    painter: Painter, left: float, top: float, width: float, height: float, radius: float
) -> None:
    right, bottom = left + width, top + height
    painter.rect(left, top, right, bottom, BACK_EDGE, radius=radius)
    painter.rect(left + 1, top + 1, right - 1, bottom - 1, BACK_FACE, radius=max(radius - 1, 0))
    il, it = left + 0.07 * width, top + 0.07 * height
    ir, ib = right - 0.07 * width, bottom - 0.07 * height
    inner_r = 0.06 * width
    painter.rect(il, it, ir, ib, BACK_FIELD, radius=inner_r)
    box = painter._box(il, it, ir, ib, pad=0.0)
    if box is not None:
        xs, ys = painter._centres(box)
        unit = max(width / 60.0, 0.5)
        period, line = 7.0 * unit, 2.0 * unit
        a = ((xs + ys) / math.sqrt(2)) % period < line
        b = ((xs - ys) / math.sqrt(2)) % period < line
        hatch = (a | b).astype(np.float64)
        inside = rounded_box_coverage(xs, ys, il, it, ir, ib, inner_r)
        painter._fill(box, hatch * inside, rgb(GOLD), 0.18)
    cx, cy = (il + ir) / 2, (it + ib) / 2
    dot = 0.12 * math.hypot(ir - il, ib - it) / 2
    painter.circle(cx, cy, dot, GOLD, alpha=0.55)
    painter.rounded_outline(il, it, ir, ib, 1.0, GOLD, radius=inner_r, alpha=0.55)


def draw_slot(painter: Painter, left: float, top: float, width: float) -> None:
    """An empty place for a card (a community card not dealt yet)."""
    height = card_height(width)
    radius = 0.1 * width
    painter.rect(left, top, left + width, top + height, (0, 0, 0), alpha=0.12, radius=radius)
    _dashed_outline(painter, left, top, left + width, top + height, radius, "#eae3d7", 0.22)


def _dashed_outline(
    painter: Painter,
    left: float,
    top: float,
    right: float,
    bottom: float,
    radius: float,
    color: ColorLike,
    alpha: float,
) -> None:
    box = painter._box(left - 1, top - 1, right + 1, bottom + 1, pad=0.0)
    if box is None:
        return
    xs, ys = painter._centres(box)
    outer = rounded_box_coverage(xs, ys, left, top, right, bottom, radius)
    inner = rounded_box_coverage(xs, ys, left + 1, top + 1, right - 1, bottom - 1, radius - 1)
    ring = outer - inner
    # Dashes of about 3 px, alternating along the outline (by distance from the centre).
    cx, cy = (left + right) / 2, (top + bottom) / 2
    along = np.abs(xs - cx) + np.abs(ys - cy)
    on = (np.floor(along / 3.0) % 2) == 0
    painter._fill(box, ring * on, rgb(color), alpha)


# ----------------------------------------------------------------------
# Chips, pills, plates and buttons
# ----------------------------------------------------------------------


def draw_chip(
    painter: Painter, cx: float, cy: float, r: float, color: ColorLike, *, unit: float = 1.0
) -> None:
    """A casino chip: a coloured disc with a dashed white edge and a small shadow."""
    painter.shadow(
        cx - r, cy - r, cx + r, cy + r, blur=2 * unit, offset=(0, unit), radius=r, alpha=0.5
    )
    painter.circle(cx, cy, r, color)
    edge = 3 * unit
    painter.ring(cx, cy, r - edge / 2, edge, "#ffffff", alpha=0.85, dash=(2.6 * unit, 2.6 * unit))


def draw_pill(
    painter: Painter,
    x: float,
    cy: float,
    text: str,
    *,
    size: float,
    color: ColorLike,
    background: ColorLike,
    background_alpha: float = 1.0,
    height: float | None = None,
    padding: float | None = None,
    align: str = "center",
    bold: bool = False,
    border: ColorLike | None = None,
    border_alpha: float = 1.0,
) -> tuple[float, float]:
    """A rounded label; `x` is its centre (or left/right edge). Returns its left and right."""
    height = height if height is not None else size * 1.9
    padding = padding if padding is not None else size * 0.8
    width = face(size, bold).width(text) + 2 * padding
    left = x - width / 2 if align == "center" else (x if align == "left" else x - width)
    top = cy - height / 2
    painter.rect(
        left,
        top,
        left + width,
        top + height,
        background,
        alpha=background_alpha,
        radius=height / 2,
    )
    if border is not None:
        painter.rounded_outline(
            left,
            top,
            left + width,
            top + height,
            1.0,
            border,
            radius=height / 2,
            alpha=border_alpha,
        )
    painter.text(
        left + width / 2, cy, text, color, size=size, bold=bold, align="center", valign="middle"
    )
    return left, left + width


__all__ = [
    "ASPECT",
    "CARD_FACE",
    "CARD_RED",
    "CARD_BLACK",
    "card_height",
    "draw_card",
    "draw_chip",
    "draw_pill",
    "draw_slot",
    "draw_suit",
    "rank_text",
    "suit_color",
]
