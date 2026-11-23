"""Renderers for Go: the wooden board, and the position as text.

`GoRenderer` draws the board in board units (one unit is the distance between two lines):
a rounded wooden board with a grain, the grid with a heavier edge, the star points,
coordinates on all four sides, stones with soft shadows and shading, a ring on the last
move and, once the game is over, the territory each side scores and the stones judged
dead (`scoring.final_score`). The board sits between the two player strips (captured
stones, and komi on White's), under a header with whose move it is or the result.

`GoTextRenderer` returns the position as text with its coordinates, for terminals and logs.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from sekai.envs.games._draw import Painter, fractal_noise, rounded_box_coverage
from sekai.envs.games.go.actions import COLUMNS
from sekai.envs.games.go.board import BLACK, EMPTY, WHITE
from sekai.envs.games.go.env import REASON_ILLEGAL_MOVE, GoEnv
from sekai.envs.games.go.scoring import GoScore, final_score
from sekai.envs.games.rendering import (
    BoardRenderer,
    GameTheme,
    Status,
    ready_or_raise,
    tone_for_result,
    unwrap,
)
from sekai.rendering.renderer import BaseRenderer
from sekai.rendering.text import face, font_sizes
from sekai.rendering.theme import ColorLike, rgb

# The board, in board units.
PAD = 1.2
"""Wood around the grid."""
CORNER = 0.25
WOOD = ((0.0, "#dcbd82"), (1.0, "#c99e5d"))
GRAIN_FREQUENCY = (0.04, 0.9)
"""The grain's base frequency, cycles per unit along x and y."""
GRAIN_COLOR = (0.35, 0.22, 0.08)
"""The grain's colour, in linear RGB; it is drawn in its sRGB encoding."""
GRAIN_ALPHA = 0.09
LINE = "#4a3824"
STONE_RADIUS = 0.47
SHADOW_OFFSET = (0.045, 0.07)
SHADOW = ((40, 24, 8), 0.35)
BLACK_STONE = ((0.35, 0.32), 0.70, ((0.0, "#5d625c"), (0.45, "#24271f"), (1.0, "#0d0f0b")))
WHITE_STONE = ((0.36, 0.30), 0.75, ((0.0, "#ffffff"), (0.65, "#ece9e1"), (1.0, "#c9c4b7")))
"""Stone shading: the radial gradient's focus and radius (fractions of the stone's box)
and its stops."""
LAST_MOVE = {BLACK: "#f1ece1", WHITE: "#24271f"}
"""The last-move ring's colour on a black and on a white stone."""
TERRITORY = {BLACK: ("#1c1f18", None), WHITE: ("#f7f4ec", "#524331")}
"""Territory squares: fill and outline."""
DEAD_OPACITY = 0.42
SCORING_PLAYOUTS = {9: 300, 13: 300, 19: 200}
"""Playouts for the dead-stone estimate of a finished game, by board size (200 otherwise)."""
SCORING_SEED = 0
"""The playouts' seed, so a finished position is always scored (and drawn) the same way."""

_PHASES = 4
"""Sub-pixel positions per pixel at which stone and territory sprites are drawn."""


def _num(value: float) -> str:
    """A number as JavaScript prints one: `7.5`, `81`."""
    value = float(value)
    return str(int(value)) if value.is_integer() else repr(value)


def _linear_to_srgb(value: float) -> int:
    if value <= 0.0031308:
        encoded = 12.92 * value
    else:
        encoded = 1.055 * value ** (1 / 2.4) - 0.055
    return round(min(max(encoded, 0.0), 1.0) * 255)


def star_points(size: int) -> list[tuple[int, int]]:
    """The `(row, column)` of the star points the board shows (9x9, 13x13 and 19x19)."""
    if size == 9:
        return [(2, 2), (2, 6), (6, 2), (6, 6), (4, 4)]
    if size == 13:
        return [(3, 3), (3, 9), (9, 3), (9, 9), (6, 6)]
    if size == 19:
        return [(r, c) for r in (3, 9, 15) for c in (3, 9, 15)]
    return []


class _Sprite:
    """A premultiplied RGBA patch: `color` `(h, w, 3)` and `keep` (`1 - alpha`), float32."""

    __slots__ = ("color", "half", "keep")

    def __init__(self, color: NDArray[np.float32], alpha: NDArray[np.float32], half: int) -> None:
        self.color = color
        self.keep = (1.0 - alpha)[..., None]
        self.half = half

    def draw(self, pixels: NDArray[np.uint8], x: float, y: float) -> None:
        """Composite the sprite with its centre at `(x, y)` (snapped to its phase)."""
        x0 = math.floor(x * _PHASES + 0.5) // _PHASES - self.half
        y0 = math.floor(y * _PHASES + 0.5) // _PHASES - self.half
        h, w = self.keep.shape[:2]
        height, width = pixels.shape[:2]
        sx0, sy0 = max(0, -x0), max(0, -y0)
        sx1, sy1 = min(w, width - x0), min(h, height - y0)
        if sx0 >= sx1 or sy0 >= sy1:
            return
        view = pixels[y0 + sy0 : y0 + sy1, x0 + sx0 : x0 + sx1]
        blended = view * self.keep[sy0:sy1, sx0:sx1] + self.color[sy0:sy1, sx0:sx1]
        view[...] = (blended + 0.5).astype(np.uint8)


def _phase(value: float) -> float:
    """The sub-pixel phase a sprite drawn at `value` uses."""
    return (math.floor(value * _PHASES + 0.5) % _PHASES) / _PHASES


def _grid(n: int, cx: float, cy: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pixel centres of an `n x n` patch, relative to a point at `(cx, cy)` in it."""
    xs = np.arange(n, dtype=np.float64)[None, :] + 0.5 - cx
    ys = np.arange(n, dtype=np.float64)[:, None] + 0.5 - cy
    return xs, ys


def _gradient(t: NDArray[np.float64], stops: Any) -> NDArray[np.float64]:
    positions = np.asarray([p for p, _ in stops], dtype=np.float64)
    colors = np.asarray([rgb(c) for _, c in stops], dtype=np.float64)
    return np.stack([np.interp(t, positions, colors[:, k]) for k in range(3)], axis=-1)


def _stone_sprite(color: int, unit: float, fx: float, fy: float, opacity: float) -> _Sprite:
    """A stone with its shadow (the shadow disc, then the shaded disc), as a group drawn
    at `opacity`."""
    r = STONE_RADIUS * unit
    dx, dy = SHADOW_OFFSET[0] * unit, SHADOW_OFFSET[1] * unit
    half = math.ceil(r + max(dx, dy)) + 2
    n = 2 * half + 1
    xs, ys = _grid(n, half + fx, half + fy)
    stone = np.clip(r + 0.5 - np.hypot(xs, ys), 0.0, 1.0)
    shadow_rgb, shadow_alpha = SHADOW
    shadow = shadow_alpha * np.clip(r + 0.5 - np.hypot(xs - dx, ys - dy), 0.0, 1.0)
    focus, spread, stops = BLACK_STONE if color == BLACK else WHITE_STONE
    gx, gy = -r + focus[0] * 2 * r, -r + focus[1] * 2 * r
    shade = _gradient(np.hypot(xs - gx, ys - gy) / (spread * 2 * r), stops)
    under = shadow * (1.0 - stone)
    alpha = stone + under
    premultiplied = shade * stone[..., None] + np.asarray(shadow_rgb) * under[..., None]
    return _Sprite(
        (premultiplied * opacity).astype(np.float32), (alpha * opacity).astype(np.float32), half
    )


def _territory_sprite(color: int, unit: float, fx: float, fy: float) -> _Sprite:
    """A territory square (a rounded square, White's with an outline)."""
    side, rx, stroke = 0.34 * unit, 0.04 * unit, 0.03 * unit
    half = math.ceil(side / 2 + stroke) + 2
    n = 2 * half + 1
    xs, ys = _grid(n, half + fx, half + fy)
    a = side / 2
    fill = rounded_box_coverage(xs, ys, -a, -a, a, a, rx)
    fill_hex, outline_hex = TERRITORY[color]
    color_f = np.asarray(rgb(fill_hex), dtype=np.float64) * fill[..., None]
    alpha = fill
    if outline_hex is not None:
        w = stroke / 2
        outer = rounded_box_coverage(xs, ys, -a - w, -a - w, a + w, a + w, rx + w)
        inner = rounded_box_coverage(xs, ys, -a + w, -a + w, a - w, a - w, max(rx - w, 0.0))
        ring = np.clip(outer - inner, 0.0, 1.0)
        color_f = color_f * (1 - ring[..., None]) + np.asarray(rgb(outline_hex)) * ring[..., None]
        alpha = ring + alpha * (1 - ring)
    return _Sprite(color_f.astype(np.float32), alpha.astype(np.float32), half)


def _box_weights(n_in: int, scale: float) -> NDArray[np.float64]:
    """Area-averaging weights `(n_out, n_in)` that shrink `n_in` pixels by `scale` (< 1)."""
    n_out = max(math.ceil(n_in * scale), 1)
    edges = np.arange(n_in + 1, dtype=np.float64) * scale
    lo = np.arange(n_out, dtype=np.float64)[:, None]
    overlap = np.minimum(edges[None, 1:], lo + 1) - np.maximum(edges[None, :-1], lo)
    weights: NDArray[np.float64] = np.clip(overlap, 0.0, None)
    return weights


def _small_text(
    painter: Painter,
    x: float,
    baseline: float,
    text: str,
    color: ColorLike,
    size: float,
    alpha: float,
) -> None:
    """Bold text centred on `x`, smaller than the smallest stored size: drawn at that size
    and shrunk (area-averaged)."""
    typeface = face(min(font_sizes()), bold=True)
    scale = size / typeface.size
    placed, width = typeface.layout(text)
    glyphs = [(g, round(pen) + g.left) for g, pen in placed if g.mask.size]
    if not glyphs:
        return
    left = min(gx for _, gx in glyphs)
    right = max(gx + g.mask.shape[1] for g, gx in glyphs)
    top = min(g.top for g, _ in glyphs)
    bottom = max(g.top + g.mask.shape[0] for g, _ in glyphs)
    mask = np.zeros((bottom - top, right - left), dtype=np.float64)
    for g, gx in glyphs:
        h, w = g.mask.shape
        rows = slice(g.top - top, g.top - top + h)
        mask[rows, gx - left : gx - left + w] += g.mask
    small = (
        _box_weights(mask.shape[0], scale)
        @ np.clip(mask, 0, 1)
        @ _box_weights(mask.shape[1], scale).T
    )
    painter.mask(small, x + (left - width / 2) * scale, baseline + top * scale, color, alpha=alpha)


class GoRenderer(BoardRenderer):
    """The wooden Go board between the two player strips (RGB frames).

    The bottom strip shows Black by default (the board itself is never turned). Once the
    game is over the board shows the territory each side scores and the stones judged
    dead, and the header the result, both from `scoring.final_score` with dead stones
    estimated by a fixed number of seeded playouts (so a position is always scored the
    same way). That score, which counts dead stones as captured, can differ from the
    environment's own Tromp-Taylor result (`info["result"]`).

    Args:
        size: The board's width in pixels, its wooden border included.
        theme: `"dark"` (the default), `"light"` or a `GameTheme`.
        header: Draw the header (the game's name and status).
        strips: Draw the player strips above and below the board.
        names: Display names by side (`{"black": "Agent", "white": "Random"}`); sides
            without a name show their label.
        perspective: The side whose strip is at the bottom, `"black"` or `"white"`.
    """

    title: ClassVar[str] = "Go"
    env_types: ClassVar[tuple[type, ...]] = (GoEnv,)
    sides: ClassVar[tuple[str, str]] = ("black", "white")
    labels: ClassVar[dict[str, str]] = {"black": "Black", "white": "White"}

    def __init__(
        self,
        size: int = 480,
        *,
        theme: GameTheme | str = "dark",
        header: bool = True,
        strips: bool = True,
        names: Mapping[str, str] | None = None,
        perspective: str = "black",
    ) -> None:
        super().__init__(size, theme=theme, header=header, strips=strips, names=names)
        if perspective not in self.sides:
            raise ValueError(f"perspective must be 'black' or 'white', got {perspective!r}.")
        self.perspective = perspective
        self._sprites: dict[tuple[Any, ...], _Sprite] = {}
        self._scores: dict[tuple[Any, ...], GoScore] = {}

    # -- layout -----------------------------------------------------------------------

    def unit(self, size: int) -> float:
        """Pixels per board unit (the line spacing) on a `size x size` board."""
        return self.size / (size - 1 + 2 * PAD)

    def point(self, size: int, row: float, col: float) -> tuple[float, float]:
        """Pixel coordinates of a point of the grid (row 0 at the top)."""
        left, top, _, _ = self.board_box()
        unit = self.unit(size)
        return left + (col + PAD) * unit, top + (row + PAD) * unit

    def corner_radius(self, size: int) -> float:
        """The wood's corner radius in pixels: the board's own, or the page's 8 px clip."""
        return max(CORNER * self.unit(size), 8 * self.scale)

    # -- the game -----------------------------------------------------------------------

    def check_ready(self, env: Any) -> None:
        ready_or_raise(env._has_reset)

    def bottom_side(self, env: Any) -> str:
        return self.perspective

    def turn(self, env: Any) -> str | None:
        return None if self.finished(env) else str(env.to_play)

    def final(self, env: GoEnv) -> GoScore:
        """The score a finished game shows (dead stones estimated), cached per position."""
        key = (env.size, env.board_string(), env.to_play, env.komi)
        score = self._scores.get(key)
        if score is None:
            if len(self._scores) >= 8:
                self._scores.pop(next(iter(self._scores)))
            score = final_score(
                env,
                estimate_dead=True,
                playouts=SCORING_PLAYOUTS.get(env.size, 200),
                rng=SCORING_SEED,
            )
            self._scores[key] = score
        return score

    def finished(self, env: Any) -> bool:
        """Whether the game is over: ended by the env or stopped by a `TimeLimit`."""
        return bool(env.done) or self.time_limit_reached

    def status(self, env: Any) -> Status:
        outcome = env.outcome
        if outcome is not None and outcome.reason == REASON_ILLEGAL_MOVE:
            winner = self.player(outcome.winner).name
            return Status(
                f"{winner} won",
                f"{self.labels[outcome.winner]} wins by an illegal move.",
                tone_for_result(outcome.winner),
            )
        if self.finished(env):
            score = self.final(env)
            margin = abs(score.margin)
            points = (
                f"Black {_num(score.black)} · White {_num(score.white)} (komi {_num(env.komi)})"
            )
            if not env.done:
                return Status("Stopped", f"Stopped at the move limit. {points}.", "draw")
            if score.winner is None:
                return Status("Draw", f"A draw. {points}.", tone_for_result(None))
            winner = self.player(score.winner).name
            plural = "" if margin == 1 else "s"
            wins = f"{winner} wins by {_num(margin)} point{plural}. {points}."
            return Status(f"{winner} won", wins, tone_for_result(score.winner))
        side = str(env.to_play)
        passed = None
        if env.last_move == "pass":
            passer = self.sides[1] if side == self.sides[0] else self.sides[0]
            passed = f"{self.player(passer).name} passed."
        return Status(f"{self.player(side).name} to move", passed, "idle")

    def strip_info(self, painter: Painter, env: Any, side: str, right: float, cy: float) -> None:
        text = f"{env.captures[side]} captured"
        if side == "white" and env.komi:
            text += f" · komi {_num(env.komi)}"
        size = 12.5 * self.scale
        painter.text(right, cy, text, self.theme.muted, size=size, align="right", valign="middle")

    # -- drawing ------------------------------------------------------------------------

    def static_key(self, env: Any) -> Any:
        return env.size

    def draw_static(self, painter: Painter, env: Any) -> None:
        size = env.size
        unit = self.unit(size)
        left, top, right, bottom = self.board_box()
        radius = self.corner_radius(size)
        self.draw_board_shadow(painter, 8 * self.scale)
        painter.linear_rect(left, top, right, bottom, WOOD, radius=radius)

        width, height = round(right - left), round(bottom - top)
        frequency = (GRAIN_FREQUENCY[0] / unit, GRAIN_FREQUENCY[1] / unit)
        grain = fractal_noise(width, height, frequency, octaves=2, seed=7)
        xs = np.arange(width, dtype=np.float64)[None, :] + 0.5 + left
        ys = np.arange(height, dtype=np.float64)[:, None] + 0.5 + top
        inside = rounded_box_coverage(xs, ys, left, top, right, bottom, radius)
        red, green, blue = (_linear_to_srgb(v) for v in GRAIN_COLOR)
        grain_rgb = (red, green, blue)
        painter.texture(round(left), round(top), grain, grain_rgb, alpha=GRAIN_ALPHA, mask=inside)

        line = (0.035 if size >= 19 else 0.04) * unit
        x0, y0 = self.point(size, 0, 0)
        x1, y1 = self.point(size, size - 1, size - 1)
        for i in range(size):
            x, y = self.point(size, i, i)
            painter.line(x0, y, x1, y, LINE, line)
            painter.line(x, y0, x, y1, LINE, line)
        painter.rounded_outline(x0, y0, x1, y1, 1.8 * line, LINE)
        star = (0.11 if size >= 19 else 0.12) * unit
        for r, c in star_points(size):
            painter.circle(*self.point(size, r, c), star, LINE)
        self._draw_coordinates(painter, size)

    def _draw_coordinates(self, painter: Painter, size: int) -> None:
        unit = self.unit(size)
        text_size = (0.3 if size >= 19 else 0.32 if size >= 13 else 0.3) * unit
        smallest = min(font_sizes())
        small = round(text_size) < smallest
        if small:
            typeface = face(smallest, bold=True)
            x_height = typeface.x_height * text_size / typeface.size
        else:
            x_height = face(text_size, bold=True).x_height
        far = size - 1 + PAD
        labels = []
        for i in range(size):
            letter, number = COLUMNS[i], str(size - i)
            labels += [
                (i, -PAD + 0.42, letter),
                (i, far - 0.4, letter),
                (-PAD + 0.4, i + 0.02, number),
                (far - 0.4, i + 0.02, number),
            ]
        for u, v, text in labels:
            x, y = self.point(size, v, u)
            # The middle of the x-height on the point.
            baseline = y + x_height / 2
            if small:
                _small_text(painter, x, baseline, text, LINE, text_size, 0.7)
            else:
                painter.text(
                    x, baseline, text, LINE, size=text_size, bold=True, align="center", alpha=0.7
                )

    def _sprite(self, kind: str, color: int, unit: float, x: float, y: float) -> _Sprite:
        fx, fy = _phase(x), _phase(y)
        key = (kind, color, unit, fx, fy)
        sprite = self._sprites.get(key)
        if sprite is None:
            if len(self._sprites) > 512:
                self._sprites.clear()
            if kind == "territory":
                sprite = _territory_sprite(color, unit, fx, fy)
            else:
                opacity = DEAD_OPACITY if kind == "dead" else 1.0
                sprite = _stone_sprite(color, unit, fx, fy, opacity)
            self._sprites[key] = sprite
        return sprite

    def draw_board(self, painter: Painter, env: Any) -> None:
        size = env.size
        unit = self.unit(size)
        left, top, _, _ = self.board_box()
        xs = [left + (c + PAD) * unit for c in range(size)]
        ys = [top + (r + PAD) * unit for r in range(size)]
        pixels = painter.canvas.pixels
        dead: set[int] = set()
        if self.finished(env):  # in order: territory, stones, then the last move's ring
            score = self.final(env)
            for owner, color in (("black", BLACK), ("white", WHITE)):
                for vertex in score.territory.get(owner, ()):
                    r, c = divmod(env.encode_vertex(vertex), size)
                    x, y = xs[c], ys[r]
                    self._sprite("territory", color, unit, x, y).draw(pixels, x, y)
            dead = {env.encode_vertex(v) for v in score.dead_stones}
        colors = env.board.color
        for p, color in enumerate(colors):
            if color == EMPTY:
                continue
            r, c = divmod(p, size)
            x, y = xs[c], ys[r]
            self._sprite("dead" if p in dead else "stone", color, unit, x, y).draw(pixels, x, y)
        last = env.last_move
        if last is not None and last != "pass":
            p = env.encode_vertex(last)
            if colors[p] != EMPTY:
                r, c = divmod(p, size)
                painter.ring(xs[c], ys[r], 0.2 * unit, 0.07 * unit, LAST_MOVE[colors[p]])


class GoTextRenderer(BaseRenderer):
    """The position as text: the board with its coordinates (`Board.pretty()`), then a
    line with whose move it is (or the result), the captures and the komi.

    `X` is Black, `O` White and `.` an empty point:

           A B C D E F G H J
         9 . . . . . . . . .  9
         ...
           A B C D E F G H J
        White to move. Last move: Black E5. Captured: Black 0, White 0. Komi 7.5.
    """

    def render(self, env: Any) -> str:
        base: GoEnv = unwrap(env, (GoEnv,))
        ready_or_raise(base._has_reset)
        captures = base.captures
        tail = (
            f"Captured: Black {captures['black']}, White {captures['white']}. "
            f"Komi {_num(base.komi)}."
        )
        outcome = base.outcome
        if outcome is not None:
            head = f"Game over ({outcome.reason}): {outcome.result}."
        else:
            head = f"{base.to_play.title()} to move."
            last = base.last_move
            if last is not None:
                mover = "White" if base.to_play == "black" else "Black"
                head += f" Last move: {mover} {last}."
        return f"{base.board.pretty()}\n{head} {tail}"


__all__ = ["GoRenderer", "GoTextRenderer", "star_points"]
