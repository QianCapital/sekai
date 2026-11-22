"""Renderers for xiangqi: the wooden board, and a text diagram.

`XiangqiRenderer` draws the board in board units (one unit between two points):
the 8 x 9 grid of points inside a padding of 1.18, so the board is `8 + 2 * 1.18` wide and
`9 + 2 * 1.18` tall, on a wood gradient; the river with 楚 河 / 漢 界; the file numbers
(Red's 九..一 and Black's 1..9) along the two edges; and the pieces as discs with their
character. The last move's two points are ringed, and a general in check glows. The
header says whose move it is or how the game ended, and each player strip lists the pieces
that side has captured and its material lead.

The board and its markings are drawn once per layout (`draw_static`), and every piece,
ring and glow is a small pre-rendered sprite, so a frame costs a few milliseconds.

`XiangqiTextRenderer` returns the engine's text diagram.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from sekai.envs.games._assets import glyph
from sekai.envs.games._draw import Painter
from sekai.envs.games.rendering import (
    BoardRenderer,
    GameTheme,
    Status,
    tone_for_result,
    unwrap,
)
from sekai.envs.games.xiangqi.engine import (
    ADVISOR,
    CANNON,
    CHARIOT,
    ELEPHANT,
    HORSE,
    KING,
    NUM_FILES,
    NUM_RANKS,
    SOLDIER,
    move_from_uci,
)
from sekai.envs.games.xiangqi.env import XiangqiEnv
from sekai.rendering.renderer import BaseRenderer
from sekai.rendering.text import face
from sekai.rendering.theme import ColorLike, rgb

Orientation = Literal["red", "black", "mover"]

PAD = 1.18
"""Padding around the grid of points, in point units."""
VIEW_WIDTH = 8 + 2 * PAD
VIEW_HEIGHT = 9 + 2 * PAD

WOOD = ("#edd4a0", "#dcb877")
LINE = "#6a4526"
INK = {"red": "#b3302a", "black": "#1f2a1e"}
"""The colour of each side's characters."""
EDGE = {"red": "#9c3a2a", "black": "#2b3528"}
RING_ALPHA = {"red": 0.75, "black": 0.70}
FACE_STOPS = ((0.0, "#fbf3df"), (0.7, "#efdfb9"), (1.0, "#d9c08d"))
SHADOW = ((60, 35, 10), 0.35)
LAST_MOVE = (70, 110, 85)
CHECK = ((214, 70, 50), 0.45)
MINI_FACE, MINI_BORDER = "#efdfb9", "#b89c66"

GLYPHS = {
    "red": {
        KING: "帥",
        ADVISOR: "仕",
        ELEPHANT: "相",
        HORSE: "傌",
        CHARIOT: "俥",
        CANNON: "炮",
        SOLDIER: "兵",
    },
    "black": {
        KING: "將",
        ADVISOR: "士",
        ELEPHANT: "象",
        HORSE: "馬",
        CHARIOT: "車",
        CANNON: "砲",
        SOLDIER: "卒",
    },
}
"""The characters on the pieces (傌 and 俥 for Red's horse and chariot)."""

RED_FILES = "九八七六五四三二一"
VALUES = {CHARIOT: 9.0, CANNON: 4.5, HORSE: 4.0, ELEPHANT: 2.0, ADVISOR: 2.0, SOLDIER: 1.0}
"""Material values for the strips' lead, in the order captured pieces are listed."""
START_COUNTS = {CHARIOT: 2, CANNON: 2, HORSE: 2, ELEPHANT: 2, ADVISOR: 2, SOLDIER: 5}

ENDINGS = {
    "CHECKMATE": "checkmate",
    "STALEMATE": "stalemate (having no legal move loses)",
    "PERPETUAL_CHECK": "perpetual check, which is forbidden",
    "REPETITION": "repetition",
    "NO_CAPTURE": "sixty moves without a capture",
    "INSUFFICIENT_MATERIAL": "insufficient material",
    "ILLEGAL_MOVE": "an illegal move",
    "MAX_PLIES": "the move limit",
}
"""How the header's second line words each ending ("Red wins by checkmate.")."""

_PHASES = 4  # sprites are cached at quarter-pixel offsets
_CJK_SPACE = 0.25  # the advance of a space in the CJK font, in ems
# Browsers centre CJK text (SVG `dominant-baseline: central`, or a CSS line box) halfway
# between the font's ascent and descent (1.151 and 0.286 em), which is 0.0525 em below the
# centre of its em box: the em box is drawn that much lower than its centring point.
_CENTRAL = (1.151 - 0.286) / 2 - (0.88 - 0.12) / 2
_TEXT_GAMMA = 0.8


# ----------------------------------------------------------------------
# Coverage and sprites
# ----------------------------------------------------------------------


def _grid(width: int, height: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pixel centres of a `width` x `height` patch: a row of x and a column of y."""
    xs = np.arange(width, dtype=np.float64)[None, :] + 0.5
    ys = np.arange(height, dtype=np.float64)[:, None] + 0.5
    return xs, ys


def _disc(
    xs: NDArray[np.float64], ys: NDArray[np.float64], cx: float, cy: float, r: float
) -> NDArray[np.float64]:
    """Anti-aliased coverage of a disc (as the canvas draws one)."""
    cover: NDArray[np.float64] = np.clip(r + 0.5 - np.hypot(xs - cx, ys - cy), 0.0, 1.0)
    return cover


def _ring(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    cx: float,
    cy: float,
    r: float,
    width: float,
    dash: tuple[float, float] | None = None,
) -> NDArray[np.float64]:
    """A circle's stroke `width` wide on radius `r`, dashed `(on, off)` from the top, clockwise.

    The dashes start at the top of the circle, where an SVG rect with fully rounded
    corners starts its outline.
    """
    d = np.hypot(xs - cx, ys - cy)
    outer, inner = r + width / 2, max(r - width / 2, 0.0)
    cover = np.clip(outer + 0.5 - d, 0.0, 1.0) - np.clip(inner + 0.5 - d, 0.0, 1.0)
    if dash is not None:
        on, off = dash
        angle = np.arctan2(ys - cy, xs - cx) + math.pi / 2  # 0 at the top, clockwise on screen
        along = np.mod(angle, 2 * math.pi) * r
        cover = cover * (np.mod(along, on + off) < on)
    result: NDArray[np.float64] = cover
    return result


def _overlaps(start: float, step: float, count: int, pixels: int) -> NDArray[np.float64]:
    """`(pixels, count)` lengths by which source cells `[start + j step, start + (j+1) step)`
    overlap the pixels `[i, i + 1)`: the weights of an exact box-filter resampling."""
    edges = start + step * np.arange(count + 1, dtype=np.float64)
    left = np.arange(pixels, dtype=np.float64)[:, None]
    weights: NDArray[np.float64] = np.clip(
        np.minimum(left + 1.0, edges[None, 1:]) - np.maximum(left, edges[None, :-1]), 0.0, None
    )
    return weights


def _glyph(
    ch: str, em: float, left: float, top: float, width: int, height: int
) -> NDArray[np.float64]:
    """Coverage, on a `width` x `height` patch, of character `ch`'s em box `em` pixels square
    with its top-left corner at `(left, top)` (both fractional): an exact area resampling of
    the glyph atlas."""
    source = glyph(ch, 96)
    n = source.shape[0]
    wx = _overlaps(left, em / n, n, width)
    wy = _overlaps(top, em / n, n, height)
    # A 0.8 power draws the characters a little heavier than their plain coverage, as
    # browsers draw text (their text gamma).
    cover: NDArray[np.float64] = np.clip(wy @ source @ wx.T, 0.0, 1.0) ** _TEXT_GAMMA
    return cover


@dataclass(frozen=True)
class _Sprite:
    """A premultiplied RGBA patch: `color` is colour times alpha, `keep` is `1 - alpha`."""

    color: NDArray[np.float32]
    keep: NDArray[np.float32]
    half: int
    """The patch's anchor sits `half` pixels (plus the phase) from its top-left corner."""

    def blit(self, pixels: NDArray[np.uint8], x0: int, y0: int) -> None:
        """Composite onto an `(H, W, 3)` image with the patch's top-left at `(x0, y0)`."""
        h, w = self.keep.shape[:2]
        height, width = pixels.shape[:2]
        cx0, cy0 = max(x0, 0), max(y0, 0)
        cx1, cy1 = min(x0 + w, width), min(y0 + h, height)
        if cx0 >= cx1 or cy0 >= cy1:
            return
        view = pixels[cy0:cy1, cx0:cx1]
        keep = self.keep[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]
        color = self.color[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]
        view[...] = (view * keep + color + np.float32(0.5)).astype(np.uint8)


class _Layers:
    """Builds a sprite from layers painted back to front ("over" compositing)."""

    def __init__(self, size: int, half: int) -> None:
        self.size, self.half = size, half
        self.color = np.zeros((size, size, 3), dtype=np.float64)
        self.alpha = np.zeros((size, size), dtype=np.float64)
        self.xs, self.ys = _grid(size, size)

    def paint(
        self, cover: NDArray[np.float64], color: ColorLike | NDArray[np.float64], alpha: float = 1.0
    ) -> None:
        a = np.clip(cover, 0.0, 1.0) * alpha
        ink = np.asarray(rgb(color) if isinstance(color, (str, tuple)) else color, np.float64)
        self.color = ink * a[..., None] + self.color * (1.0 - a[..., None])
        self.alpha = a + self.alpha * (1.0 - a)

    def sprite(self) -> _Sprite:
        return _Sprite(
            color=self.color.astype(np.float32),
            keep=(1.0 - self.alpha)[..., None].astype(np.float32),
            half=self.half,
        )


def _gradient(t: NDArray[np.float64], stops: tuple[tuple[float, str], ...]) -> NDArray[np.float64]:
    """Colours at gradient positions `t`, padded with the end colours, shape `t.shape + (3,)`."""
    positions = [p for p, _ in stops]
    colors = np.asarray([rgb(c) for _, c in stops], dtype=np.float64)
    return np.stack([np.interp(t, positions, colors[:, k]) for k in range(3)], axis=-1)


def _square_cover(
    left: float, top: float, right: float, bottom: float, x0: int, y0: int, w: int, h: int
) -> NDArray[np.float64]:
    """Exact area coverage of a box on the pixels `[x0, x0 + w) x [y0, y0 + h)`."""
    xs = np.arange(x0, x0 + w, dtype=np.float64)
    ys = np.arange(y0, y0 + h, dtype=np.float64)
    cx = np.clip(np.minimum(xs + 1.0, right) - np.maximum(xs, left), 0.0, 1.0)
    cy = np.clip(np.minimum(ys + 1.0, bottom) - np.maximum(ys, top), 0.0, 1.0)
    cover: NDArray[np.float64] = cy[:, None] * cx[None, :]
    return cover


# ----------------------------------------------------------------------
# The renderers
# ----------------------------------------------------------------------


class XiangqiRenderer(BoardRenderer):
    """The xiangqi board as an RGB array.

    Args:
        size: The board's width in pixels (its height follows the board's proportions).
        orientation: `"red"` (Red at the bottom, the default), `"black"`, or `"mover"`
            (the side to move at the bottom). The bottom player strip is that side's.
        theme: `"dark"` (the default), `"light"` or a `GameTheme`; the board is the same
            in both.
        header: Draw the header (the title and the game's status).
        strips: Draw the player strips (name, colour, captured pieces, material lead).
        names: Display names by side (`{"red": "Agent", "black": "Random"}`).

    `render(env)` takes a `XiangqiEnv`, or any wrapper of one, and returns an
    `(H, W, 3)` uint8 array.
    """

    title: ClassVar[str] = "Xiangqi"
    env_types: ClassVar[tuple[type, ...]] = (XiangqiEnv,)
    sides: ClassVar[tuple[str, str]] = ("red", "black")
    labels: ClassVar[dict[str, str]] = {"red": "Red", "black": "Black"}

    def __init__(
        self,
        size: int = 480,
        *,
        orientation: Orientation = "red",
        theme: GameTheme | str = "dark",
        header: bool = True,
        strips: bool = True,
        names: Mapping[str, str] | None = None,
    ) -> None:
        if orientation not in ("red", "black", "mover"):
            raise ValueError(f"orientation must be 'red', 'black' or 'mover', got {orientation!r}.")
        super().__init__(size, theme=theme, header=header, strips=strips, names=names)
        self.orientation: Orientation = orientation
        self._sprites: dict[tuple[Any, ...], _Sprite] = {}

    # -- layout -----------------------------------------------------------------------

    def board_height(self) -> int:
        return round(self.size * VIEW_HEIGHT / VIEW_WIDTH)

    def _units(self) -> tuple[float, float, float]:
        """Pixels per point unit, and the pixel position of the point `(0, 0)`.

        As SVG fits a viewBox (`xMidYMid meet`): the larger scale that fits both sides,
        centred in the board's box.
        """
        left, top, right, bottom = self.board_box()
        width, height = right - left, bottom - top
        unit = min(width / VIEW_WIDTH, height / VIEW_HEIGHT)
        ox = left + (width - VIEW_WIDTH * unit) / 2 + PAD * unit
        oy = top + (height - VIEW_HEIGHT * unit) / 2 + PAD * unit
        return unit, ox, oy

    def _point(self, x: float, y: float) -> tuple[float, float]:
        unit, ox, oy = self._units()
        return ox + x * unit, oy + y * unit

    def bottom_side(self, env: Any) -> str:
        if self.orientation == "mover":
            return str(env.turn)
        return self.orientation

    def turn(self, env: Any) -> str | None:
        return None if env.done or self.time_limit_reached else str(env.turn)

    def static_key(self, env: Any) -> Any:
        return self.bottom_side(env)

    @staticmethod
    def _xy(square: int, black_bottom: bool) -> tuple[int, int]:
        """The board column and row (from the top) of an absolute square."""
        file, rank = square % NUM_FILES, square // NUM_FILES
        return (8 - file, rank) if black_bottom else (file, 9 - rank)

    # -- status -----------------------------------------------------------------------

    def status(self, env: Any) -> Status:
        outcome = env.outcome
        if outcome is None and self.time_limit_reached:
            return Status("Draw", f"Drawn by {ENDINGS['MAX_PLIES']}.", tone_for_result(None))
        if outcome is None:
            sub = "In check." if env.is_check() else None
            return Status(f"{self.player(env.turn).name} to move", sub, "idle")
        how = ENDINGS.get(outcome.termination, outcome.termination.replace("_", " ").lower())
        if outcome.winner is None:
            return Status("Draw", f"Drawn by {how}.", tone_for_result(None))
        player = self.player(outcome.winner)
        sub = f"{player.label} wins by {how}."
        return Status(f"{player.name} won", sub, tone_for_result(outcome.winner))

    # -- static layer: the board ------------------------------------------------------

    def draw_static(self, painter: Painter, env: Any) -> None:
        s = self.scale
        unit, ox, oy = self._units()

        def at(x: float, y: float) -> tuple[float, float]:
            return ox + x * unit, oy + y * unit

        self.draw_board_shadow(painter, radius=8 * s)
        x0, y0 = at(-PAD, -PAD)
        x1, y1 = at(8 + PAD, 9 + PAD)
        stops = ((0.0, WOOD[0]), (1.0, WOOD[1]))
        painter.linear_rect(x0, y0, x1, y1, stops, radius=0.28 * unit)
        self._frame(painter, *at(-0.18, -0.18), *at(8.18, 9.18), 0.07 * unit)

        half = 0.035 * unit / 2
        for row in range(NUM_RANKS):
            (lx, ly), (rx, _) = at(0, row), at(8, row)
            painter.rect(lx, ly - half, rx, ly + half, LINE)
        for col in range(NUM_FILES):
            spans = ((0, 9),) if col in (0, 8) else ((0, 4), (5, 9))
            for a, b in spans:
                (tx, ty), (_, by) = at(col, a), at(col, b)
                painter.rect(tx - half, ty, tx + half, by, LINE)
        for (ax, ay), (bx, by) in (
            ((3, 0), (5, 2)),
            ((5, 0), (3, 2)),
            ((3, 7), (5, 9)),
            ((5, 7), (3, 9)),
        ):
            painter.line(*at(ax, ay), *at(bx, by), LINE, 0.035 * unit)
        for row in (2, 7):
            for col in (1, 7):
                self._corners(painter, at, col, row, unit)
        for row in (3, 6):
            for col in (0, 2, 4, 6, 8):
                self._corners(painter, at, col, row, unit)

        river = 0.56 * unit
        for text, col in (("楚 河", 2), ("漢 界", 6)):
            _cjk_text(painter, text, *at(col, 4.52), river, 0.3 * river, LINE, 0.82)

        black_bottom = self.bottom_side(env) == "black"
        for x in range(NUM_FILES):
            file = 8 - x if black_bottom else x
            red, black = RED_FILES[file], str(file + 1)
            bottom, top = (black, red) if black_bottom else (red, black)
            _cjk_text(painter, bottom, *at(x, 9 + PAD - 0.34), 0.34 * unit, 0.0, LINE, 0.75)
            _cjk_text(painter, top, *at(x, -PAD + 0.34), 0.34 * unit, 0.0, LINE, 0.75)

    @staticmethod
    def _frame(
        painter: Painter, left: float, top: float, right: float, bottom: float, width: float
    ) -> None:
        """A rectangle's stroke with square (mitred) corners, composited once."""
        h = width / 2
        x0, y0 = math.floor(left - h), math.floor(top - h)
        w, ht = math.ceil(right + h) - x0, math.ceil(bottom + h) - y0
        outer = _square_cover(left - h, top - h, right + h, bottom + h, x0, y0, w, ht)
        inner = _square_cover(left + h, top + h, right - h, bottom - h, x0, y0, w, ht)
        painter.mask(outer - inner, x0, y0, LINE)

    @staticmethod
    def _corners(
        painter: Painter,
        at: Callable[[float, float], tuple[float, float]],
        col: int,
        row: int,
        unit: float,
    ) -> None:
        """The L-shaped marks around a cannon or soldier point (none outside the board)."""
        d, length, w = 0.1, 0.22, 0.03 * unit / 2
        for sx in (-1, 1):
            if (col == 0 and sx < 0) or (col == 8 and sx > 0):
                continue
            for sy in (-1, 1):
                px, py = at(col + sx * d, row + sy * d)
                ex, ey = px + sx * length * unit, py + sy * length * unit
                # The stroke of M(ex, py) H px V ey, mitred at (px, py), as one polygon.
                painter.polygon(
                    [
                        (ex, py - sy * w),
                        (px - sx * w, py - sy * w),
                        (px - sx * w, ey),
                        (px + sx * w, ey),
                        (px + sx * w, py + sy * w),
                        (ex, py + sy * w),
                    ],
                    LINE,
                )

    # -- per frame: pieces and highlights ---------------------------------------------

    def draw_board(self, painter: Painter, env: Any) -> None:
        unit, ox, oy = self._units()
        black_bottom = self.bottom_side(env) == "black"
        pixels = painter.canvas.pixels
        last = env.last_move
        if last is not None:
            move = move_from_uci(last)
            for square in (move // 90, move % 90):
                x, y = self._xy(square, black_bottom)
                self._blit(pixels, ("last", unit), ox + x * unit, oy + y * unit)
        position = env.position
        board = position.board
        checked = position.kings[position.side] if position.checked else -1
        for rank in range(NUM_RANKS - 1, -1, -1):  # the FEN's order
            for file in range(NUM_FILES):
                square = rank * NUM_FILES + file
                piece = board[square]
                if not piece:
                    continue
                x, y = self._xy(square, black_bottom)
                px, py = ox + x * unit, oy + y * unit
                if square == checked:
                    self._blit(pixels, ("check", unit), px, py)
                self._blit(pixels, ("piece", unit, piece), px, py)

    def _blit(self, pixels: NDArray[np.uint8], key: tuple[Any, ...], x: float, y: float) -> None:
        """Draw the sprite `key` with its anchor at `(x, y)` (cached per quarter pixel)."""
        ix, iy = math.floor(x), math.floor(y)
        qx, qy = round((x - ix) * _PHASES), round((y - iy) * _PHASES)
        if qx == _PHASES:
            ix, qx = ix + 1, 0
        if qy == _PHASES:
            iy, qy = iy + 1, 0
        full = (*key, qx, qy)
        sprite = self._sprites.get(full)
        if sprite is None:
            if len(self._sprites) > 4096:
                self._sprites.clear()
            sprite = _make_sprite(key, qx / _PHASES, qy / _PHASES)
            self._sprites[full] = sprite
        sprite.blit(pixels, ix - sprite.half, iy - sprite.half)

    # -- strips: captured pieces and the lead -----------------------------------------

    def strip_info(self, painter: Painter, env: Any, side: str, right: float, cy: float) -> None:
        captured, score = _material(env.position.board)
        taken = captured[side]
        if not taken:
            return
        s = self.scale
        lead = score if side == "red" else -score
        other = "black" if side == "red" else "red"
        mini, gap, limit = 19 * s, 2 * s, 220 * s
        items: list[tuple[str, float]] = [("mini", mini)] * len(taken)
        lead_text = f"+{lead:g}"
        lead_size = 12 * s
        if lead > 0:
            items.append(("lead", face(lead_size).width(lead_text) + 4 * s))
        lines: list[list[tuple[str, float]]] = [[]]
        used = 0.0
        for item in items:  # flex-wrap: as many as fit in 220 px per line
            width = item[1] if not lines[-1] else used + gap + item[1]
            if lines[-1] and width > limit:
                lines.append([])
                width = item[1]
            lines[-1].append(item)
            used = width
        heights = [mini if any(k == "mini" for k, _ in line) else 15 * s for line in lines]
        total = sum(heights) + gap * (len(lines) - 1)
        top = cy - total / 2
        pixels = painter.canvas.pixels
        index = 0
        for line, height in zip(lines, heights):
            width = sum(w for _, w in line) + gap * (len(line) - 1)
            x = right - width
            mid = top + height / 2
            for kind, w in line:
                if kind == "mini":
                    key = ("mini", s, other, taken[index])
                    self._blit(pixels, key, x + mini / 2, mid)
                    index += 1
                else:
                    painter.text(
                        x + 4 * s,
                        mid,
                        lead_text,
                        self.theme.muted,
                        size=lead_size,
                        valign="middle",
                    )
                x += w + gap
            top += height + gap


class XiangqiTextRenderer(BaseRenderer):
    """The board as text: the engine's diagram (Black at the top, Red in capitals)."""

    def render(self, env: Any) -> str:
        base = unwrap(env, XiangqiEnv)
        return str(base.position.ascii())


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _material(board: list[int]) -> tuple[dict[str, list[int]], float]:
    """The pieces each side has captured (by type, in the strips' order) and Red's lead."""
    counts = {"red": dict.fromkeys(VALUES, 0), "black": dict.fromkeys(VALUES, 0)}
    for piece in board:
        if piece and abs(piece) != KING:
            counts["red" if piece > 0 else "black"][abs(piece)] += 1
    captured: dict[str, list[int]] = {"red": [], "black": []}
    score = 0.0
    for kind, value in VALUES.items():
        captured["red"] += [kind] * max(0, START_COUNTS[kind] - counts["black"][kind])
        captured["black"] += [kind] * max(0, START_COUNTS[kind] - counts["red"][kind])
        score += (counts["red"][kind] - counts["black"][kind]) * value
    return captured, score


def _cjk_text(
    painter: Painter,
    text: str,
    x: float,
    y: float,
    em: float,
    spacing: float,
    color: ColorLike,
    alpha: float,
) -> None:
    """CJK text centred on `(x, y)` as SVG sets it with `text-anchor: middle` and
    `dominant-baseline: central`: em boxes centred on `y`, the advance (with the letter
    spacing after every character, the last one too) centred on `x`."""
    advances = [(_CJK_SPACE if ch == " " else 1.0) * em for ch in text]
    pen = x - (sum(advances) + spacing * len(text)) / 2
    top = y - em / 2 + _CENTRAL * em
    for ch, advance in zip(text, advances):
        if ch != " ":
            left = pen + (advance - em) / 2
            x0, y0 = math.floor(left), math.floor(top)
            w, h = math.ceil(left + em) - x0, math.ceil(top + em) - y0
            cover = _glyph(ch, em, left - x0, top - y0, w, h)
            painter.mask(cover, x0, y0, color, alpha=alpha)
        pen += advance + spacing


def _make_sprite(key: tuple[Any, ...], fx: float, fy: float) -> _Sprite:
    """Render sprite `key` with its anchor at the fractional offset `(fx, fy)`."""
    kind = key[0]
    if kind == "piece":
        return _piece_sprite(float(key[1]), int(key[2]), fx, fy)
    if kind == "last":
        unit = float(key[1])
        half = math.ceil(0.5 * unit) + 2
        layers = _Layers(2 * half, half)
        cx, cy = half + fx, half + fy
        tint = LAST_MOVE
        layers.paint(_disc(layers.xs, layers.ys, cx, cy, 0.47 * unit), tint, 0.22)
        ring = _ring(
            layers.xs, layers.ys, cx, cy, 0.47 * unit, 0.045 * unit, (0.14 * unit, 0.1 * unit)
        )
        layers.paint(ring, tint, 0.6)
        return layers.sprite()
    if kind == "check":
        unit = float(key[1])
        half = math.ceil(0.56 * unit) + 2
        layers = _Layers(2 * half, half)
        color, alpha = CHECK
        layers.paint(_disc(layers.xs, layers.ys, half + fx, half + fy, 0.56 * unit), color, alpha)
        return layers.sprite()
    if kind == "mini":
        return _mini_sprite(float(key[1]), str(key[2]), int(key[3]), fx, fy)
    raise KeyError(key)


def _piece_sprite(unit: float, piece: int, fx: float, fy: float) -> _Sprite:
    """A piece: drop shadow, gradient face with its edge, inner ring and character."""
    side = "red" if piece > 0 else "black"
    half = math.ceil(0.5 * unit) + 2
    layers = _Layers(2 * half, half)
    xs, ys = layers.xs, layers.ys
    cx, cy = half + fx, half + fy
    color, alpha = SHADOW
    layers.paint(_disc(xs, ys, cx, cy + 0.05 * unit, 0.45 * unit), color, alpha)
    r = 0.44 * unit
    # radialGradient cx=40% cy=35% r=75% of the face's bounding box (2r square)
    t = np.hypot(xs - (cx - r + 0.8 * r), ys - (cy - r + 0.7 * r)) / (1.5 * r)
    layers.paint(_disc(xs, ys, cx, cy, r), _gradient(t, FACE_STOPS))
    layers.paint(_ring(xs, ys, cx, cy, r, 0.045 * unit), EDGE[side])
    layers.paint(_ring(xs, ys, cx, cy, 0.37 * unit, 0.035 * unit), INK[side], RING_ALPHA[side])
    em = 0.5 * unit
    size = 2 * half
    top = cy + 0.02 * unit - em / 2 + _CENTRAL * em
    cover = _glyph(GLYPHS[side][abs(piece)], em, cx - em / 2, top, size, size)
    layers.paint(cover, INK[side])
    return layers.sprite()


def _mini_sprite(scale: float, side: str, kind: int, fx: float, fy: float) -> _Sprite:
    """A captured piece in a strip: a 19 px disc with a 1 px border and an 11 px character."""
    r = 9.5 * scale
    half = math.ceil(r) + 2
    layers = _Layers(2 * half, half)
    xs, ys = layers.xs, layers.ys
    cx, cy = half + fx, half + fy
    layers.paint(_disc(xs, ys, cx, cy, r), MINI_BORDER)
    layers.paint(_disc(xs, ys, cx, cy, r - scale), MINI_FACE)
    em = 11 * scale
    size = 2 * half
    cover = _glyph(GLYPHS[side][kind], em, cx - em / 2, cy - em / 2 + _CENTRAL * em, size, size)
    layers.paint(cover, INK[side])
    return layers.sprite()


__all__ = ["ENDINGS", "Orientation", "XiangqiRenderer", "XiangqiTextRenderer"]
