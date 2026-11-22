"""Renderers of heads-up hold'em: the felt table, and a text table.

`HoldemRenderer` draws the table: an oval of green felt in a wooden rail, the board and
the pot in the middle, and a seat at the top and one at the bottom, each with its two
hole cards, a plate with its name and stack (and the dealer button), its last action on
the street in a speech bubble and its bet in chips. When the hand is over a banner says
who won what, the winning five cards light up at a showdown and a folded hand is dimmed.

Most lengths are fractions of the table's width (`u` is 1% of it) and the rest are pixels
at a 640-pixel table (`scale` is `width / 640`).

`HoldemTextRenderer` gives the same table as text, for terminals and logs.
"""

from __future__ import annotations

import functools
import math
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from sekai.envs.games._cards import card_height, draw_card, draw_slot
from sekai.envs.games._draw import Painter, VAlign, gaussian_blur, rounded_box_coverage
from sekai.envs.games.poker.cards import CARD_STRINGS, card_from_str
from sekai.envs.games.poker.env import HoldemEnv
from sekai.envs.games.poker.holdem import COMPLETE, SHOWDOWN, HoldemHand, SeatView
from sekai.envs.games.rendering import GameRenderer, GameTheme, Status, Tone, unwrap
from sekai.error import ResetRequired
from sekai.rendering.canvas import Canvas
from sekai.rendering.renderer import BaseRenderer
from sekai.rendering.text import face
from sekai.rendering.theme import Color, ColorLike, rgb

Reveal = Literal["all", "perspective"]

SIDES = ("p1", "p2")
SEAT_LABELS = ("Seat 1", "Seat 2")
STREET_NAMES = {
    "preflop": "Preflop",
    "flop": "Flop",
    "turn": "Turn",
    "river": "River",
    SHOWDOWN: "Showdown",
    COMPLETE: "Hand over",
}

# The table's colours (the same in both themes).
FELT_STOPS = ((0.0, "#3f6a52"), (0.45, "#2c4d3b"), (1.0, "#1d3428"))
RAILS = ((2.4, "#3a281b"), (1.6, "#c0aa81"), (1.4, "#2a1d14"))  # outermost first, in u
FELT_LINE = (201, 180, 139)
CREAM = "#f3ecdd"
GOLD = "#c9b48b"
GREEN = "#86ad93"
BRAND_GOLD = "#c0aa81"
INK = "#1a2117"
PAPER = "#f6f1e4"
SOFT = (234, 227, 215)
CHIP_COLORS = ("#b8352b", "#1f3b5c", "#2a2a2a")

# The line height of text, in ems.
LINE = 1.55

# The avatar's linear-gradient(145deg, ...): its gradient line through a square, in box units.
_A = math.radians(145)
_HALF = (abs(math.sin(_A)) + abs(math.cos(_A))) / 2
_AVATAR_START = (0.5 - math.sin(_A) * _HALF, 0.5 + math.cos(_A) * _HALF)
_AVATAR_END = (0.5 + math.sin(_A) * _HALF, 0.5 - math.cos(_A) * _HALF)


@dataclass(frozen=True)
class _Shadows:
    """A theme's shadows: `(y offset, blur, spread)` in pixels, colour, opacity."""

    color: Color
    small: tuple[tuple[float, float, float], float]
    medium: tuple[tuple[float, float, float], float]
    large: tuple[tuple[float, float, float], float]


_SHADOWS = {
    "dark": _Shadows((0, 0, 0), ((1, 2, 0), 0.35), ((12, 36, -12), 0.6), ((30, 80, -24), 0.75)),
    "light": _Shadows(
        rgb("#344e41"), ((1, 2, 0), 0.12), ((12, 36, -14), 0.28), ((30, 80, -28), 0.38)
    ),
}


def _shadows(theme: GameTheme) -> _Shadows:
    known = _SHADOWS.get(theme.name)
    if known is not None and theme.shadow == known.color:
        return known
    a = theme.shadow_alpha  # a custom theme: its colour, the dark theme's proportions
    return _Shadows(
        rgb(theme.shadow),
        ((1, 2, 0), 0.47 * a),
        ((12, 36, -12), 0.8 * a),
        ((30, 80, -24), a),
    )


# ----------------------------------------------------------------------
# What a frame shows
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class _Table:
    """The public state of a hand plus the hole cards the frame may show."""

    view: SeatView
    holes: tuple[tuple[int, ...] | None, tuple[int, ...] | None]
    hand_no: int


def _table(hand: HoldemHand, bottom: int, reveal: Reveal, hand_no: int) -> _Table:
    if hand.is_complete:
        view = hand.reveal()  # both hands once the hand is over, folded ones too
        return _Table(view, view.holes, hand_no)
    if reveal == "perspective":
        view = hand.view(bottom)  # the bottom seat's information set, nothing more
        return _Table(view, view.holes, hand_no)
    view = hand.public_view()
    holes = (tuple(hand.hole(0)), tuple(hand.hole(1)))
    return _Table(replace(view, holes=holes), holes, hand_no)


def _chips(amount: int) -> str:
    return f"{amount:,}"


def _pot_shown(view: SeatView) -> int:
    """The pot in the middle of the table: without the street's bets, which sit in front of
    the seats, unless that leaves nothing (preflop); the final pot once the hand is over."""
    if view.result is not None:
        return view.result.pot
    middle = view.pot - view.street_bets[0] - view.street_bets[1]
    return middle if middle > 0 else view.pot


def _seat_name(names: Mapping[str, str], seat: int) -> str:
    return names.get(SIDES[seat], SEAT_LABELS[seat])


def _last_actions(view: SeatView) -> dict[int, str]:
    """Each seat's latest action on the current street, as the speech bubbles say it."""
    out: dict[int, str] = {}
    for event in view.events:
        if event["type"] == "deal":
            out.clear()
        elif event["type"] == "action":
            out[event["seat"]] = _action_text(event)
    return out


def _action_text(event: Mapping[str, Any]) -> str:
    kind, all_in = event["kind"], event["all_in"]
    if kind == "fold":
        return "Fold"
    if kind == "check":
        return "Check"
    if kind == "call":
        return "Call · all-in" if all_in else f"Call {_chips(event['added'])}"
    if kind == "bet":
        return f"All-in {_chips(event['to'])}" if all_in else f"Bet {_chips(event['to'])}"
    return f"All-in {_chips(event['to'])}" if all_in else f"Raise to {_chips(event['to'])}"


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def _result_lines(view: SeatView, names: Mapping[str, str]) -> tuple[str, str | None]:
    """The banner's line ("Seat 1 wins 600 with ...") and its second line, if any."""
    result = view.result
    assert result is not None
    if len(result.winners) == 2:
        first = result.hands.get(0)
        return "Split pot" + (f" · {first['rank']}" if first else ""), None
    winner = result.winners[0]
    text = _wins(_seat_name(names, winner), result.pot)
    if result.showdown and winner in result.hands:
        text += f" with {_lower_first(result.hands[winner]['rank'])}"
    if not result.showdown:
        return text, f"{_seat_name(names, 1 - winner)} folded"
    return text, None


def _wins(name: str, pot: int) -> str:
    """`"Seat 1 wins 600"` (`"You win 600"` for a seat named "You")."""
    return f"{name} {'win' if name == 'You' else 'wins'} {_chips(pot)}"


def _initials(name: str) -> str:
    """The avatar's letters ("Seat 1" -> "S1", "Agent" -> "AG")."""
    cleaned = "".join(ch if ch.isascii() and ch.isalnum() else " " for ch in name)
    words = cleaned.split()
    if not words:
        return "?"
    if len(words) == 1:
        return words[0][:2].upper()
    last = words[-1]
    if last.isdigit() or set(last) <= set("IVX"):
        return (words[0][0] + last)[:3].upper()
    return (words[0][0] + last[0]).upper()


# ----------------------------------------------------------------------
# Cached pieces: shadows and card sprites
# ----------------------------------------------------------------------

Box = tuple[int, int, int, int]


@functools.lru_cache(maxsize=128)
def _shadow_mask(
    size: tuple[int, int],
    box: tuple[float, float, float, float],
    radius: float,
    blur: float,
    offset: float,
    spread: float,
    clip: bool,
) -> tuple[Box, NDArray[np.float64]] | None:
    """The coverage of a CSS outer `box-shadow` (offset downwards), on a canvas of `size`.

    With `clip` the shadow is cut out under the box itself, as CSS draws it (it shows
    through a translucent background otherwise).
    """
    width, height = size
    left, top, right, bottom = box
    sigma = max(blur / 2, 1e-6)
    x0, y0 = left - spread, top + offset - spread
    x1, y1 = right + spread, bottom + offset + spread
    if x1 <= x0 or y1 <= y0:
        return None
    pad = 3 * sigma + 1
    px0, py0 = max(math.floor(x0 - pad), 0), max(math.floor(y0 - pad), 0)
    px1, py1 = min(math.ceil(x1 + pad), width), min(math.ceil(y1 + pad), height)
    if px0 >= px1 or py0 >= py1:
        return None
    xs = np.arange(px0, px1, dtype=np.float64)[None, :] + 0.5
    ys = np.arange(py0, py1, dtype=np.float64)[:, None] + 0.5
    sharp = rounded_box_coverage(xs, ys, x0, y0, x1, y1, max(radius + spread, 0.0))
    soft = gaussian_blur(sharp, sigma)
    if clip:
        soft = soft * (1.0 - rounded_box_coverage(xs, ys, left, top, right, bottom, radius))
    soft.flags.writeable = False
    return (px0, py0, px1, py1), soft


def _shadow(
    painter: Painter,
    box: tuple[float, float, float, float],
    radius: float,
    token: tuple[tuple[float, float, float], float],
    color: Color,
    scale: float,
    *,
    clip: bool = True,
) -> None:
    (offset, blur, spread), alpha = token
    mask = _shadow_mask(
        (painter.width, painter.height),
        box,
        radius,
        blur * scale,
        offset * scale,
        spread * scale,
        clip,
    )
    if mask is not None:
        painter._fill(mask[0], mask[1], color, alpha)


Sprite = tuple[NDArray[np.float32], NDArray[np.float32], int, int]
"""A drawing to composite: premultiplied colour (0-255), `1 - alpha`, and the canvas pixel
of its top-left corner (or its offset from a reference pixel)."""


def _capture(width: int, height: int, draw: Callable[[Painter], None]) -> NDArray[np.float32]:
    """Premultiplied RGBA in [0, 1] of what `draw` paints, drawn on black and on white."""
    black = Canvas(width, height, (0, 0, 0), antialias=True)
    white = Canvas(width, height, (255, 255, 255), antialias=True)
    draw(Painter(black))
    draw(Painter(white))
    on_black = black.pixels.astype(np.float32) / 255.0
    on_white = white.pixels.astype(np.float32) / 255.0
    rgba = np.empty((height, width, 4), dtype=np.float32)
    rgba[..., 3] = np.clip(1.0 - np.mean(on_white - on_black, axis=-1), 0.0, 1.0)
    rgba[..., :3] = np.minimum(on_black, rgba[..., 3:4])
    return rgba


def _sprite_of(rgba: NDArray[np.float32], x: int, y: int) -> Sprite:
    """A sprite of premultiplied RGBA whose top-left is at `(x, y)`, cropped to its pixels."""
    rows = np.flatnonzero(np.max(rgba[..., 3], axis=1) > 0.5 / 255)
    cols = np.flatnonzero(np.max(rgba[..., 3], axis=0) > 0.5 / 255)
    if rows.size == 0:
        empty = np.zeros((0, 0, 3), dtype=np.float32)
        return empty, np.ones((0, 0, 1), dtype=np.float32), x, y
    y0, y1, x0, x1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    crop = rgba[y0:y1, x0:x1]
    color = np.ascontiguousarray(crop[..., :3] * 255.0)
    keep = np.ascontiguousarray(1.0 - crop[..., 3:4])
    color.flags.writeable = False
    keep.flags.writeable = False
    return color, keep, x + int(x0), y + int(y0)


def _blit(painter: Painter, sprite: Sprite, dx: int = 0, dy: int = 0) -> None:
    """Composite a sprite onto the canvas, moved by `(dx, dy)`."""
    color, keep, x0, y0 = sprite
    x0, y0 = x0 + dx, y0 + dy
    h, w = keep.shape[:2]
    cx0, cy0 = max(x0, 0), max(y0, 0)
    cx1, cy1 = min(x0 + w, painter.width), min(y0 + h, painter.height)
    if cx0 >= cx1 or cy0 >= cy1:
        return
    view = painter.canvas.pixels[cy0:cy1, cx0:cx1]
    rows, cols = slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0)
    out = view * keep[rows, cols] + color[rows, cols]
    out += 0.5
    view[...] = out.astype(np.uint8)


def _dim(rgba: NDArray[np.float32]) -> NDArray[np.float32]:
    """CSS `filter: brightness(.62) saturate(.6)` on premultiplied RGBA."""
    s = 0.6
    matrix = np.array(
        [
            [0.213 + 0.787 * s, 0.715 - 0.715 * s, 0.072 - 0.072 * s],
            [0.213 - 0.213 * s, 0.715 + 0.285 * s, 0.072 - 0.072 * s],
            [0.213 - 0.213 * s, 0.715 - 0.715 * s, 0.072 + 0.928 * s],
        ],
        dtype=np.float32,
    )
    alpha = rgba[..., 3:4]
    color = np.divide(rgba[..., :3], alpha, out=np.zeros_like(rgba[..., :3]), where=alpha > 0)
    color = np.clip(color @ matrix.T * 0.62, 0.0, 1.0)
    out: NDArray[np.float32] = rgba.copy()
    out[..., :3] = color * alpha
    return out


def _rotate(rgba: NDArray[np.float32], angle: float, cx: float, cy: float) -> NDArray[np.float32]:
    """`rgba` turned by `angle` degrees (clockwise on screen) about `(cx, cy)`, same size."""
    h, w = rgba.shape[:2]
    a = math.radians(angle)
    c, s = math.cos(a), math.sin(a)
    xs = np.arange(w, dtype=np.float64)[None, :] + 0.5 - cx
    ys = np.arange(h, dtype=np.float64)[:, None] + 0.5 - cy
    sx = xs * c + ys * s + cx - 0.5  # inverse rotation, into pixel indices
    sy = -xs * s + ys * c + cy - 0.5
    ix, iy = np.floor(sx).astype(np.intp), np.floor(sy).astype(np.intp)
    fx = (sx - ix)[..., None].astype(np.float32)
    fy = (sy - iy)[..., None].astype(np.float32)
    padded = np.pad(rgba, ((1, 1), (1, 1), (0, 0)))
    ix = np.clip(ix + 1, 0, w)
    iy = np.clip(iy + 1, 0, h)
    out = (
        padded[iy, ix] * (1 - fx) * (1 - fy)
        + padded[iy, ix + 1] * fx * (1 - fy)
        + padded[iy + 1, ix] * (1 - fx) * fy
        + padded[iy + 1, ix + 1] * fx * fy
    )
    return np.asarray(out, dtype=np.float32)


_SPRITES: dict[tuple[Any, ...], Sprite] = {}
"""Drawings shared by the renderers (plates, bubbles, chips, ...), by what they depend on."""

_STATIC_LAYERS: dict[tuple[Any, ...], NDArray[np.uint8]] = {}
"""The empty tables, by renderer settings and frame size."""


def _card(
    painter: Painter,
    card: str | None,
    width: float,
    cx: float,
    cy: float,
    *,
    angle: float = 0.0,
    highlight: bool = False,
    lift: bool = False,
    dim: bool = False,
) -> None:
    """A card of `width` centred on `(cx, cy)`, turned by `angle` degrees (clockwise).

    `highlight` rings the card in gold; `lift` raises it as a highlighted board card is;
    `dim` darkens it (a folded hand, or a card that does not play at a showdown).
    """
    ix, iy = math.floor(cx), math.floor(cy)
    _blit(painter, _card_sprite(card, width, angle, highlight, lift, dim, cx - ix, cy - iy), ix, iy)


@functools.lru_cache(maxsize=160)
def _card_sprite(
    card: str | None,
    width: float,
    angle: float,
    highlight: bool,
    lift: bool,
    dim: bool,
    fx: float,
    fy: float,
) -> Sprite:
    """A card centred `(fx, fy)` past a pixel corner: a sprite, placed from that corner."""
    height = card_height(width)
    pad = 0.55 * width + 3
    half = math.ceil(math.hypot(width, height) / 2 + pad)
    lx, ly = half + fx, half + fy  # the centre in the sprite
    top = ly - height / 2 + (0.0 if lift or not highlight else 0.04 * height)

    def draw(p: Painter) -> None:
        draw_card(p, lx - width / 2, top, width, card, highlight=highlight)

    rgba = _capture(2 * half + 1, 2 * half + 1, draw)
    if dim:
        rgba = _dim(rgba)
    if angle:
        rgba = _rotate(rgba, angle, lx, ly)
    return _sprite_of(rgba, -half, -half)


# ----------------------------------------------------------------------
# The table renderer
# ----------------------------------------------------------------------


class HoldemRenderer(GameRenderer):
    """Heads-up hold'em at a felt table, as RGB frames.

    Args:
        width: The table's width in pixels; it is `0.6 * width` tall (the frame adds the
            header and margins).
        perspective: The seat drawn at the bottom: `0` or `1` (or `"p1"`, `"p2"`).
        reveal: Whose hole cards are face up during a hand: `"all"` (both, as a spectator
            of two agents sees them) or `"perspective"` (only the bottom seat's; the
            other seat's stay face down until the hand is over). Both hands are shown
            once it is over, folded ones included. Undealt cards are never drawn.
        theme: `"dark"` (the default), `"light"` or a `GameTheme`.
        header: Draw the header (the game's name and status).
        names: Display names by side (`{"p1": "Agent", "p2": "Random"}`); seats without
            one are "Seat 1" and "Seat 2".

    The pot says "Hand n": the number of hands this renderer has drawn so far, counting
    from 1.
    """

    title: ClassVar[str] = "Hold'em"
    env_types: ClassVar[tuple[type, ...]] = (HoldemEnv,)

    def __init__(
        self,
        width: int = 640,
        *,
        perspective: int | str = 0,
        reveal: Reveal = "all",
        theme: GameTheme | str = "dark",
        header: bool = True,
        names: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(theme=theme, header=header, names=names)
        if width < 320:
            raise ValueError(f"width must be at least 320 pixels, got {width}.")
        if reveal not in ("all", "perspective"):
            raise ValueError(f"reveal must be 'all' or 'perspective', got {reveal!r}.")
        self.width = int(width)
        self.perspective = _seat_of(perspective)
        self.reveal: Reveal = reveal
        self._hand_numbers: weakref.WeakKeyDictionary[HoldemHand, int] = weakref.WeakKeyDictionary()
        self._hands_drawn = 0

    # -- layout -------------------------------------------------------------------------

    @property
    def scale(self) -> float:
        return self.width / 640

    @property
    def unit(self) -> float:
        """1% of the table's width, the unit of most lengths."""
        return self.width / 100

    def _band(self) -> float:
        """Room above and below the table for the "Folded" tags."""
        return 22 * self.scale

    def frame_size(self) -> tuple[int, int]:
        m = self.margin()
        height = 2 * m + self.header_height() + 2 * self._band() + 0.6 * self.width
        return self.width + 2 * round(m), round(height)

    def stage(self) -> tuple[float, float, float, float]:
        """`(left, top, right, bottom)` of the table area."""
        left = round(self.margin())
        top = self.margin() + self.header_height() + self._band()
        return left, top, left + self.width, top + 0.6 * self.width

    def felt(self) -> tuple[float, float, float, float]:
        """`(left, top, right, bottom)` of the felt (inset 9% and 2.5% of the stage)."""
        left, top, right, bottom = self.stage()
        w, h = right - left, bottom - top
        return left + 0.025 * w, top + 0.09 * h, right - 0.025 * w, bottom - 0.09 * h

    # -- the parts that do not change ---------------------------------------------------------

    def draw_static(self, painter: Painter, env: Any) -> None:
        # Every renderer with these settings draws the same felt: share it.
        key = (self.width, self.header, self.theme, painter.width, painter.height)
        pixels = _STATIC_LAYERS.get(key)
        if pixels is None:
            self._draw_table(painter)
            if len(_STATIC_LAYERS) >= 8:
                del _STATIC_LAYERS[next(iter(_STATIC_LAYERS))]
            _STATIC_LAYERS[key] = painter.canvas.pixels.copy()
        else:
            painter.canvas.pixels[...] = pixels

    def _draw_table(self, painter: Painter) -> None:
        """The table without anything on it: its shadow, the rails and the felt."""
        u = self.unit
        left, top, right, bottom = self.felt()
        radius = (bottom - top) / 2
        shadows = _shadows(self.theme)
        _shadow(
            painter, (left, top, right, bottom), radius, shadows.large, shadows.color, self.scale
        )
        for spread, color in RAILS:
            d = spread * u
            painter.rect(left - d, top - d, right + d, bottom + d, color, radius=radius + d)
        painter.radial_rect(left, top, right, bottom, FELT_STOPS, centre=(0.5, 0.4), radius=radius)
        self._inset_shadow(painter, (left, top, right, bottom), radius, 6 * u, 0.45)
        w, h = right - left, bottom - top
        il, it, ir, ib = left + 0.05 * w, top + 0.05 * h, right - 0.05 * w, bottom - 0.05 * h
        line = max(self.scale, 1.0)
        painter.rounded_outline(
            il + line / 2,
            it + line / 2,
            ir - line / 2,
            ib - line / 2,
            line,
            FELT_LINE,
            radius=(ib - it) / 2 - line / 2,
            alpha=0.18,
        )

    @staticmethod
    def _inset_shadow(
        painter: Painter,
        box: tuple[float, float, float, float],
        radius: float,
        blur: float,
        alpha: float,
    ) -> None:
        """A CSS `inset 0 0 blur` box-shadow: darkness creeping in from the edges."""
        left, top, right, bottom = box
        x0, y0 = math.floor(left), math.floor(top)
        x1, y1 = math.ceil(right), math.ceil(bottom)
        xs = np.arange(x0, x1, dtype=np.float64)[None, :] + 0.5
        ys = np.arange(y0, y1, dtype=np.float64)[:, None] + 0.5
        inside = rounded_box_coverage(xs, ys, left, top, right, bottom, radius)
        soft = gaussian_blur(inside, max(blur / 2, 1e-6))
        painter._fill((x0, y0, x1, y1), inside * (1.0 - soft), (0, 0, 0), alpha)

    # -- status ---------------------------------------------------------------------------

    def check_ready(self, env: Any) -> None:
        if env._hand is None:
            raise ResetRequired()

    def _hand_no(self, hand: HoldemHand) -> int:
        number = self._hand_numbers.get(hand)
        if number is None:
            self._hands_drawn += 1
            number = self._hand_numbers[hand] = self._hands_drawn
        return number

    def _names(self) -> dict[str, str]:
        return {side: _seat_name(self.names, s) for s, side in enumerate(SIDES)}

    def status(self, env: Any) -> Status:
        hand: HoldemHand = env.hand
        names = self._names()
        view = hand.public_view()
        hand_no = self._hand_no(hand)
        if view.result is None:
            assert view.to_act is not None
            tone: Tone = (
                "yours"
                if self.reveal == "perspective" and view.to_act == self.perspective
                else "idle"
            )
            text = f"{names[SIDES[view.to_act]]} to act"
            return Status(text, f"{STREET_NAMES[view.street]} · Hand {hand_no}", tone)
        result = view.result
        if len(result.winners) == 2:
            text = "Split pot"
            sub = result.hands[0]["rank"] if result.hands else None
        else:
            winner = result.winners[0]
            text = _wins(names[SIDES[winner]], result.pot)
            if result.showdown:
                sub = f"Showdown · {result.hands[winner]['rank']}"
            else:
                sub = f"{names[SIDES[1 - winner]]} folded"
        if self.reveal == "perspective":
            delta = result.deltas[self.perspective]
            tone = "win" if delta > 0 else "loss" if delta < 0 else "draw"
        else:
            tone = "draw"
        return Status(text, sub, tone)

    # -- the table ------------------------------------------------------------------------

    def draw_surface(self, painter: Painter, env: Any) -> None:
        hand: HoldemHand = env.hand
        table = _table(hand, self.perspective, self.reveal, self._hand_no(hand))
        view = table.view
        result = view.result
        highlight: set[int] = set()
        if result is not None and result.showdown:
            for w in result.winners:
                highlight |= {_card_int(c) for c in result.hands[w]["best"]}
        top_seat = 1 - self.perspective
        # Paint from the bottom up: the centre, the bets, the seats, the banner on top.
        self._draw_centre(painter, table, highlight)
        if result is None:
            self._draw_bet(painter, view.street_bets[top_seat], view.big_blind, top=True)
            self._draw_bet(painter, view.street_bets[self.perspective], view.big_blind, top=False)
        bubbles = _last_actions(view)
        for seat, at_top in ((top_seat, True), (self.perspective, False)):
            self._draw_seat(painter, table, seat, at_top, bubbles.get(seat), highlight)
        if result is not None:
            self._draw_banner(painter, table)

    def _draw_cached(
        self,
        painter: Painter,
        key: tuple[Any, ...],
        box: tuple[float, float, float, float],
        pad: float,
        draw: Callable[[Painter, float, float], None],
    ) -> None:
        """Draw what `draw(painter, left, top)` paints for `box`, through a cached sprite.

        `draw` paints the box with its top-left corner at `(left, top)`, and nothing
        further than `pad` outside it; `key` says everything else it depends on.
        """
        left, top, right, bottom = box
        ix, iy = math.floor(left - pad), math.floor(top - pad)
        full = (key, self.width, self.theme, left - ix, top - iy, right - left, bottom - top)
        sprite = _SPRITES.get(full)
        if sprite is None:
            width, height = math.ceil(right + pad) - ix + 1, math.ceil(bottom + pad) - iy + 1
            rgba = _capture(width, height, lambda p: draw(p, left - ix, top - iy))
            sprite = _sprite_of(rgba, 0, 0)
            if len(_SPRITES) >= 256:
                del _SPRITES[next(iter(_SPRITES))]
            _SPRITES[full] = sprite
        _blit(painter, sprite, ix, iy)

    def _shadow_pad(self) -> float:
        """How far a box's drop shadow reaches beyond the box."""
        return 58 * self.scale + 2

    # The centre column: board, pot and (at the end) the banner, centred in the felt.

    def _centre_layout(self, table: _Table) -> tuple[float, float, float]:
        """The tops of the board, the pot and the banner."""
        u = self.unit
        _, top, _, bottom = self.felt()
        board_h = card_height(7.2 * u)
        pot_h = self._pot_height()
        gap = 1.4 * u
        total = board_h + gap + pot_h
        if table.view.result is not None:
            total += gap + self._banner_layout(table)[1]
        board_top = (top + bottom) / 2 - total / 2
        pot_top = board_top + board_h + gap
        return board_top, pot_top, pot_top + pot_h + gap

    def _pot_height(self) -> float:
        u = self.unit
        return 1.75 * u * LINE + 2 * 0.5 * u

    def _draw_centre(self, painter: Painter, table: _Table, highlight: set[int]) -> None:
        board_top, pot_top, _ = self._centre_layout(table)
        u = self.unit
        left, _, right, _ = self.felt()
        cx = (left + right) / 2
        width, gap = 7.2 * u, 0.9 * u
        height = card_height(width)
        x = cx - (5 * width + 4 * gap) / 2
        board = table.view.board
        for i in range(5):
            if i < len(board):
                lit = board[i] in highlight
                dark = bool(highlight) and not lit
                _card(
                    painter,
                    _card_str(board[i]),
                    width,
                    x + width / 2,
                    board_top + height / 2,
                    highlight=lit,
                    lift=True,
                    dim=dark,
                )
            else:
                self._draw_cached(
                    painter,
                    ("slot", width),
                    (x, board_top, x + width, board_top + height),
                    2,
                    lambda p, sx, sy: draw_slot(p, sx, sy, width),
                )
            x += width + gap
        self._draw_pot(painter, table, cx, pot_top)

    def _draw_pot(self, painter: Painter, table: _Table, cx: float, top: float) -> None:
        u, s = self.unit, self.scale
        view = table.view
        amount = _pot_shown(view)
        size = 1.75 * u
        label_size, street_size = 0.82 * size, 0.78 * size
        label, value = "POT", _chips(amount)
        street = f"{STREET_NAMES[view.street]} · Hand {table.hand_no}"
        spacing = 0.08 * label_size
        label_w = _spaced_width(label, label_size, True, spacing)
        value_w = face(size).width(value)
        street_w = face(street_size).width(street)
        gap = 8 * s
        width = 2 * 1.6 * u + label_w + gap + value_w + gap + 4 * s + street_w
        height = self._pot_height()
        left = cx - width / 2
        painter.rect(
            left, top, left + width, top + height, (10, 18, 12), alpha=0.45, radius=height / 2
        )
        baseline = top + height / 2 + face(size).cap_height / 2  # the items share a baseline
        x = left + 1.6 * u
        _spaced_text(
            painter,
            x,
            baseline,
            label,
            SOFT,
            label_size,
            True,
            spacing,
            alpha=0.65,
            valign="baseline",
        )
        x += label_w + gap
        painter.text(x, baseline, value, CREAM, size=size)
        x += value_w + gap + 4 * s
        painter.text(x, baseline, street, SOFT, size=street_size, alpha=0.5)

    def _banner_lines(self, table: _Table) -> list[tuple[str, bool]]:
        """The banner's lines, `(text, main)`, as a spectator sees them."""
        view = table.view
        text, sub = _result_lines(view, self.names)
        return [(text, True)] if sub is None else [(text, True), (sub, False)]

    def _banner_layout(self, table: _Table) -> tuple[float, float, tuple[tuple[str, bool], ...]]:
        """The banner's width and height, and its lines once wrapped to its width."""
        u, s = self.unit, self.scale
        left, _, right, _ = self.felt()
        felt_w = right - left
        pad_x = 2.2 * u
        inner = 0.7 * felt_w - 2 * pad_x - 2
        lines: list[tuple[str, bool]] = []
        for text, main in self._banner_lines(table):
            size = 2 * u if main else 1.45 * u
            lines += [(line, main) for line in _wrap(text, size, main, inner)]
        widest = max(face(2 * u if m else 1.45 * u, serif=m).width(t) for t, m in lines)
        width = min(max(widest + 2 * pad_x + 2, 0.46 * felt_w), 0.7 * felt_w)
        height = 2 + 2 * 0.9 * u
        height += sum(2 * u * 1.2 if main else 1.45 * u * LINE + 2 * s for _, main in lines)
        return width, height, tuple(lines)

    def _draw_banner(self, painter: Painter, table: _Table) -> None:
        u, s = self.unit, self.scale
        left, _, right, _ = self.felt()
        width, height, lines = self._banner_layout(table)
        top = self._centre_layout(table)[2]
        cx = (left + right) / 2
        shadows = _shadows(self.theme)

        def draw(p: Painter, x: float, y: float) -> None:
            box = (x, y, x + width, y + height)
            radius = 12 * s
            _shadow(p, box, radius, shadows.medium, shadows.color, s)
            p.rect(*box, (12, 20, 14), alpha=0.82, radius=radius)
            _border(p, box, radius, FELT_LINE, 0.55, s)
            middle = x + width / 2
            line_y = y + 1 + 0.9 * u
            for text, main in lines:
                if main:
                    line_h = 2 * u * 1.2
                    p.text(
                        middle,
                        line_y + line_h / 2,
                        text,
                        "#f7f1e3",
                        size=2 * u,
                        serif=True,
                        align="center",
                        valign="middle",
                    )
                else:
                    line_y += 2 * s
                    line_h = 1.45 * u * LINE
                    p.text(
                        middle,
                        line_y + line_h / 2,
                        text,
                        SOFT,
                        size=1.45 * u,
                        align="center",
                        valign="middle",
                        alpha=0.72,
                    )
                line_y += line_h

        box = (cx - width / 2, top, cx + width / 2, top + height)
        key = ("banner", lines)
        self._draw_cached(painter, key, box, self._shadow_pad(), draw)

    # Bets: chips beside the centre, for each seat's bet on the street.

    def _draw_bet(self, painter: Painter, amount: int, big_blind: int, *, top: bool) -> None:
        if not amount:
            return
        u, s = self.unit, self.scale
        f_left, f_top, f_right, f_bottom = self.felt()
        height = 26 * s
        offset = 0.21 * (f_bottom - f_top)
        y = f_top + offset if top else f_bottom - offset - height
        left = (f_left + f_right) / 2 + 11 * u
        text = _chips(amount)
        size = 12.5 * s
        width = (4 + 20 + 7 + 10) * s + face(size).width(text)
        painter.rect(left, y, left + width, y + height, (10, 18, 12), alpha=0.5, radius=height / 2)
        count = min(4, max(1, math.floor(math.log2(amount / big_blind + 1) + 0.5)))
        stack_left, stack_top = left + 4 * s, y + 3 * s

        def chips(p: Painter, x: float, sy: float) -> None:
            r = 10 * s
            for i in range(count):
                _chip(p, x + r, sy + r - 3 * s * i, r, CHIP_COLORS[i % 3], s)

        box = (stack_left, stack_top, stack_left + 20 * s, stack_top + 20 * s)
        self._draw_cached(
            painter,
            ("chips", count),
            (box[0], box[1] - 9 * s, box[2], box[3]),
            4 * s + 2,
            lambda p, x, sy: chips(p, x, sy + 9 * s),
        )
        painter.text(
            left + (4 + 20 + 7) * s, y + height / 2, text, CREAM, size=size, valign="middle"
        )

    # Seats: hole cards, the plate (name, stack, dealer button), bubble and tag.

    def _plate_size(self, name: str, stack: str) -> tuple[float, float]:
        s = self.scale
        text_w = max(face(13 * s, True).width(name), face(12 * s).width(stack))
        width = max(17 * self.unit, (1 + 6 + 28 + 9 + 14 + 1) * s + text_w)
        height = ((13 + 12) * LINE + 10 + 2) * s
        return width, height

    def _draw_seat(
        self,
        painter: Painter,
        table: _Table,
        seat: int,
        at_top: bool,
        bubble: str | None,
        highlight: set[int],
    ) -> None:
        u, s = self.unit, self.scale
        view = table.view
        result = view.result
        stage_left, stage_top, stage_right, stage_bottom = self.stage()
        cx = (stage_left + stage_right) / 2
        name = _seat_name(self.names, seat)
        stack = _chips(view.stacks[seat])
        plate_w, plate_h = self._plate_size(name, stack)
        card_w = 8.6 * u
        card_h = card_height(card_w)
        cards_w = 2 * card_w + 0.6 * u
        gap = 0.8 * u
        seat_w = max(plate_w, cards_w)
        if at_top:  # the plate at the table's edge, the cards towards the middle
            plate_top = stage_top
            cards_top = plate_top + plate_h + gap
            seat_top, seat_bottom = stage_top, cards_top + card_h
        else:
            plate_top = stage_bottom - plate_h
            cards_top = plate_top - gap - card_h
            seat_top, seat_bottom = cards_top, stage_bottom
        folded = result is not None and not result.showdown and seat not in result.winners
        winner = result is not None and seat in result.winners
        turn = result is None and view.to_act == seat

        # The hole cards, tilted a little.
        cards = table.holes[seat]
        angles = (-4.0, 4.0) if at_top else (-3.0, 3.0)
        x = cx - cards_w / 2
        for i in range(2):
            card = None if cards is None else cards[i]
            ccx, ccy = x + card_w / 2, cards_top + card_h / 2
            if not at_top:  # translateY(4%), turned with the card
                a = math.radians(angles[i])
                ccx -= 0.04 * card_h * math.sin(a)
                ccy += 0.04 * card_h * math.cos(a)
            lit = card is not None and card in highlight
            text = None if card is None else _card_str(card)
            _card(painter, text, card_w, ccx, ccy, angle=angles[i], highlight=lit, dim=folded)
            x += card_w + 0.6 * u

        # The plate, then its name and stack.
        box = (cx - plate_w / 2, plate_top, cx + plate_w / 2, plate_top + plate_h)
        state = "turn" if turn else "winner" if winner else ""
        initials = _initials(name)
        self._draw_cached(
            painter,
            ("plate", state, initials),
            box,
            self._shadow_pad(),
            lambda p, px, py: self._plate(p, px, py, plate_w, plate_h, state, initials),
        )
        text_x = box[0] + (1 + 6 + 28 + 9) * s
        block_top = (box[1] + box[3]) / 2 - (13 + 12) * LINE * s / 2
        name_mid = block_top + 13 * LINE * s / 2
        stack_mid = block_top + 13 * LINE * s + 12 * LINE * s / 2
        painter.text(text_x, name_mid, name, CREAM, size=13 * s, bold=True, valign="middle")
        painter.text(text_x, stack_mid, stack, GOLD, size=12 * s, valign="middle")
        if view.button == seat:  # the dealer button, on the plate's corner
            r = 11 * s
            dcx, dcy = box[2] + 10 * s - r, box[1] - 8 * s + r
            self._draw_cached(
                painter,
                ("dealer",),
                (dcx - r, dcy - r, dcx + r, dcy + r),
                8 * s + 2,
                lambda p, dx, dy: self._dealer(p, dx + r, dy + r),
            )

        if bubble is not None:
            self._draw_bubble(painter, cx - seat_w / 2 - 12 * s, seat_bottom - 8 * s, bubble)
        if folded:
            size = 10.5 * s
            spacing = 0.12 * size
            line_h = size * LINE
            tag_top = seat_top - 18 * s if at_top else seat_bottom + 20 * s - line_h
            width = _spaced_width("FOLDED", size, True, spacing)
            _spaced_text(
                painter,
                cx - width / 2,
                tag_top + line_h / 2,
                "FOLDED",
                self.theme.muted,
                size,
                True,
                spacing,
            )

    def _plate(
        self,
        painter: Painter,
        x: float,
        y: float,
        width: float,
        height: float,
        state: str,
        initials: str,
    ) -> None:
        """A seat's plate without its name and stack: shadow, ring, pill, border, avatar."""
        s = self.scale
        box = (x, y, x + width, y + height)
        radius = height / 2
        shadows = _shadows(self.theme)
        _shadow(painter, box, radius, shadows.medium, shadows.color, s)
        border: ColorLike = (201, 180, 139)
        border_alpha = 0.35
        if state:
            ring, ring_alpha = ((201, 180, 139), 0.25) if state == "turn" else (rgb(GREEN), 0.3)
            d = 3 * s  # a 3px ring around the plate (a spread box-shadow)
            painter.rounded_outline(
                x - d / 2,
                y - d / 2,
                x + width + d / 2,
                y + height + d / 2,
                d,
                ring,
                radius=radius + d / 2,
                alpha=ring_alpha,
            )
            border, border_alpha = (GOLD if state == "turn" else GREEN), 1.0
        painter.rect(*box, (14, 20, 13), alpha=0.9, radius=radius)
        _border(painter, box, radius, border, border_alpha, s)
        size = 28 * s  # the avatar: the seat's initials on a dark disc
        line = max(s, 1.0)
        acx, acy = x + (1 + 6) * s + size / 2, y + height / 2
        painter.circle(acx, acy, size / 2, "#46563f")
        painter.linear_rect(
            acx - size / 2 + line,
            acy - size / 2 + line,
            acx + size / 2 - line,
            acy + size / 2 - line,
            ((0.0, "#2c3b29"), (1.0, "#1d271a")),
            start=_AVATAR_START,
            end=_AVATAR_END,
            radius=size / 2 - line,
        )
        painter.text(
            acx,
            acy,
            initials,
            BRAND_GOLD,
            size=0.36 * size,
            bold=True,
            align="center",
            valign="middle",
        )

    def _dealer(self, painter: Painter, cx: float, cy: float) -> None:
        s = self.scale
        r = 11 * s
        shadows = _shadows(self.theme)
        _shadow(painter, (cx - r, cy - r, cx + r, cy + r), r, shadows.small, shadows.color, s)
        painter.circle(cx, cy, r, BRAND_GOLD)
        painter.circle(cx, cy, r - 2 * s, PAPER)
        painter.text(cx, cy, "D", INK, size=11 * s, bold=True, align="center", valign="middle")

    def _draw_bubble(self, painter: Painter, right: float, bottom: float, text: str) -> None:
        s = self.scale
        size = 12.5 * s
        fold = text == "Fold"
        width = face(size, True).width(text) + 2 * 11 * s
        height = size * LINE + 2 * 5 * s
        shadows = _shadows(self.theme)

        def draw(p: Painter, x: float, y: float) -> None:
            box = (x, y, x + width, y + height)
            radius = 10 * s
            _shadow(p, box, radius, shadows.medium, shadows.color, s)
            p.rect(*box, "#3a3f37" if fold else PAPER, radius=radius)
            p.text(
                x + width / 2,
                y + height / 2,
                text,
                "#e6dfd2" if fold else INK,
                size=size,
                bold=True,
                align="center",
                valign="middle",
            )

        box = (right - width, bottom - height, right, bottom)
        self._draw_cached(painter, ("bubble", text), box, self._shadow_pad(), draw)

    def __repr__(self) -> str:
        return (
            f"HoldemRenderer(width={self.width}, perspective={self.perspective}, "
            f"reveal={self.reveal!r}, theme={self.theme.name!r})"
        )


def _border(
    painter: Painter,
    box: tuple[float, float, float, float],
    radius: float,
    color: ColorLike,
    alpha: float,
    scale: float,
) -> None:
    """A CSS 1px border inside a (rounded) box."""
    line = max(scale, 1.0)
    left, top, right, bottom = box
    painter.rounded_outline(
        left + line / 2,
        top + line / 2,
        right - line / 2,
        bottom - line / 2,
        line,
        color,
        radius=radius - line / 2,
        alpha=alpha,
    )


def _chip(painter: Painter, cx: float, cy: float, r: float, color: str, s: float) -> None:
    """A chip: a disc with a dashed white edge and a small shadow."""
    box = (cx - r, cy - r, cx + r, cy + r)
    _shadow(painter, box, r, ((1, 2, 0), 0.5), (0, 0, 0), s, clip=False)
    painter.circle(cx, cy, r, color)
    edge = 3 * s
    painter.ring(cx, cy, r - edge / 2, edge, "#ffffff", alpha=0.85, dash=(2.6 * s, 2.6 * s))


def _seat_of(value: int | str) -> int:
    if isinstance(value, str) and value.lower() in SIDES:
        return SIDES.index(value.lower())
    if not isinstance(value, bool) and value in (0, 1):
        return int(value)
    raise ValueError(f"perspective must be 0, 1, 'p1' or 'p2', got {value!r}.")


def _card_str(card: int) -> str:
    return CARD_STRINGS[card]


def _card_int(text: str) -> int:
    return card_from_str(text)


def _spaced_width(text: str, size: float, bold: bool, spacing: float) -> float:
    typeface = face(size, bold)
    return sum(typeface.width(ch) for ch in text) + spacing * (len(text) - 1)


def _spaced_text(
    painter: Painter,
    x: float,
    y: float,
    text: str,
    color: ColorLike,
    size: float,
    bold: bool,
    spacing: float,
    *,
    alpha: float = 1.0,
    valign: VAlign = "middle",
) -> None:
    """Text with extra space between its letters (CSS `letter-spacing`), from `x`."""
    typeface = face(size, bold)
    for ch in text:
        painter.text(x, y, ch, color, size=size, bold=bold, valign=valign, alpha=alpha)
        x += typeface.width(ch) + spacing


def _wrap(text: str, size: float, serif: bool, width: float) -> list[str]:
    """`text` broken into lines no wider than `width` (at spaces), as a browser wraps it."""
    typeface = face(size, serif=serif)
    if typeface.width(text) <= width:
        return [text]
    lines: list[str] = []
    line = ""
    for word in text.split(" "):
        trial = f"{line} {word}" if line else word
        if line and typeface.width(trial) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    lines.append(line)
    return lines


# ----------------------------------------------------------------------
# The text renderer
# ----------------------------------------------------------------------


class HoldemTextRenderer(BaseRenderer):
    """Heads-up hold'em as a small text table.

    The board (`--` for cards still to come) and the pot, then one line per seat, the top
    seat of `HoldemRenderer` first: name, position, stack, bet on the street, hole cards
    (`?? ??` when hidden), last action on the street, and who is to act; and the result
    once the hand is over.

    Args:
        perspective: The seat listed last (the bottom seat of the picture): `0` or `1`
            (or `"p1"`, `"p2"`).
        reveal: `"all"` (both hands) or `"perspective"` (the other seat's cards stay
            hidden until the hand is over).
        names: Display names by side (`{"p1": ..., "p2": ...}`).
    """

    def __init__(
        self,
        *,
        perspective: int | str = 0,
        reveal: Reveal = "all",
        names: Mapping[str, str] | None = None,
    ) -> None:
        if reveal not in ("all", "perspective"):
            raise ValueError(f"reveal must be 'all' or 'perspective', got {reveal!r}.")
        self.perspective = _seat_of(perspective)
        self.reveal: Reveal = reveal
        self.names = dict(names or {})

    def render(self, env: Any) -> str:
        base = unwrap(env, HoldemEnv)
        if base._hand is None:
            raise ResetRequired()
        hand: HoldemHand = base.hand
        table = _table(hand, self.perspective, self.reveal, 1)
        view = table.view
        board = [_card_str(c) for c in view.board] + ["--"] * (5 - len(view.board))
        lines = [
            f"Hold'em {view.small_blind}/{view.big_blind} · {STREET_NAMES[view.street]}",
            f"Board  {' '.join(board)}    Pot {_chips(_pot_shown(view))}",
        ]
        bubbles = _last_actions(view)
        seats = (1 - self.perspective, self.perspective)
        names = [_seat_name(self.names, s) for s in seats]
        width = max(len(n) for n in names)
        stacks = [_chips(view.stacks[s]) for s in seats]
        stack_w = max(len(t) for t in stacks)
        bets = [_chips(view.street_bets[s]) for s in seats]
        bet_w = max(len(t) for t in bets)
        for k, seat in enumerate(seats):
            cards = table.holes[seat]
            shown = "?? ??" if cards is None else " ".join(_card_str(c) for c in cards)
            role = "D" if view.button == seat else " "
            line = (
                f"{role} {names[k]:<{width}}  stack {stacks[k]:>{stack_w}}  "
                f"bet {bets[k]:>{bet_w}}  [{shown}]"
            )
            if seat in bubbles:
                line += f"  {bubbles[seat]}"
            if view.result is None and view.to_act == seat:
                line += "  <- to act"
            lines.append(line.rstrip())
        if view.result is not None:
            text, sub = _result_lines(view, self.names)
            lines.append(text + (f" ({sub})" if sub else ""))
        return "\n".join(lines) + "\n"

    def __repr__(self) -> str:
        return f"HoldemTextRenderer(perspective={self.perspective}, reveal={self.reveal!r})"


__all__ = ["HoldemRenderer", "HoldemTextRenderer"]
