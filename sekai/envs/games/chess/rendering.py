"""Renderers for the chess environment.

`ChessRenderer` draws the board as an RGB frame: cream and green squares,
Colin M.L. Burnett's pieces, the last move tinted gold, a red glow under a king in check,
coordinates in the corner squares, and a strip for each player with the material they
have captured. `ChessTextRenderer` returns the board as text with chess symbols.

    renderer = ChessRenderer(orientation="mover")
    frame = renderer.render(env)  # (H, W, 3) uint8
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any, ClassVar, Literal

import numpy as np

from sekai.envs.games._assets import chess_piece
from sekai.envs.games._draw import RGBA, Painter
from sekai.envs.games.chess.board import (
    BISHOP,
    BLACK,
    FILE_NAMES,
    KNIGHT,
    PAWN,
    QUEEN,
    ROOK,
    WHITE,
    Board,
    Color,
)
from sekai.envs.games.chess.env import ChessEnv
from sekai.envs.games.rendering import BoardRenderer, GameTheme, Status
from sekai.error import ResetRequired
from sekai.rendering.renderer import BaseRenderer
from sekai.rendering.theme import rgb

Orientation = Literal["white", "black", "mover"]

LIGHT_SQUARE = "#ebe3d3"
DARK_SQUARE = {"dark": "#557161", "light": "#6a8a76"}
LAST_MOVE = ((201, 180, 139), 0.58)
CHECK = (214, 83, 62)

_SYMBOLS = "pnbrqk"
_VALUES = {PAWN: 1, KNIGHT: 3, BISHOP: 3, ROOK: 5, QUEEN: 9}
_START = {PAWN: 8, KNIGHT: 2, BISHOP: 2, ROOK: 2, QUEEN: 1}
_TERMINATIONS = {
    "CHECKMATE": "checkmate",
    "STALEMATE": "stalemate",
    "INSUFFICIENT_MATERIAL": "insufficient material",
    "FIFTY_MOVES": "the fifty-move rule",
    "SEVENTYFIVE_MOVES": "the seventy-five-move rule",
    "THREEFOLD_REPETITION": "threefold repetition",
    "FIVEFOLD_REPETITION": "fivefold repetition",
    "ILLEGAL_MOVE": "an illegal move",
}


def _piece_code(kind: int, color: Color) -> str:
    return ("w" if color else "b") + _SYMBOLS[kind - 1].upper()


def material(board: Board) -> tuple[dict[Color, list[int]], int]:
    """The pieces each side has captured (by type, most valuable first) and White's lead
    in pawns (queen 9, rook 5, bishop and knight 3, pawn 1)."""
    counts = {WHITE: dict.fromkeys(_START, 0), BLACK: dict.fromkeys(_START, 0)}
    for kind, color in board.piece_map().values():
        if kind in _START:
            counts[color][kind] += 1
    captured: dict[Color, list[int]] = {WHITE: [], BLACK: []}
    score = 0
    for kind in (QUEEN, ROOK, BISHOP, KNIGHT, PAWN):
        captured[WHITE] += [kind] * max(0, _START[kind] - counts[BLACK][kind])
        captured[BLACK] += [kind] * max(0, _START[kind] - counts[WHITE][kind])
        score += (counts[WHITE][kind] - counts[BLACK][kind]) * _VALUES[kind]
    return captured, score


@functools.lru_cache(maxsize=32)
def _ink_piece(code: str, size: int, color: tuple[int, int, int], alpha: float) -> RGBA:
    """A piece's silhouette filled with one colour (the captured pieces of a strip)."""
    sprite = chess_piece(code, size)
    out = np.empty_like(sprite)
    out[..., :3] = color
    out[..., 3] = np.round(sprite[..., 3] * alpha).astype(np.uint8)
    out.flags.writeable = False
    return out


class ChessRenderer(BoardRenderer):
    """The chess board as an RGB frame, with a header and a strip for each player.

    Args:
        size: The board's width (and height) in pixels.
        orientation: `"white"` (White at the bottom, the default), `"black"`, or
            `"mover"` (the side to move at the bottom, as the agent sees the board).
        coordinates: Draw the files and ranks in the edge squares.
        theme: `"dark"` (the default), `"light"` or a `GameTheme`.
        header: Draw the header (the game's status).
        strips: Draw the player strips (names, captured material, whose turn it is).
        names: Display names by side (`{"white": ..., "black": ...}`).
    """

    title: ClassVar[str] = "Chess"
    env_types: ClassVar[tuple[type, ...]] = (ChessEnv,)
    sides: ClassVar[tuple[str, str]] = ("white", "black")
    labels: ClassVar[dict[str, str]] = {"white": "White", "black": "Black"}

    def __init__(
        self,
        size: int = 480,
        *,
        orientation: Orientation = "white",
        coordinates: bool = True,
        theme: GameTheme | str = "dark",
        header: bool = True,
        strips: bool = True,
        names: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(size, theme=theme, header=header, strips=strips, names=names)
        if orientation not in ("white", "black", "mover"):
            raise ValueError(
                f"orientation must be 'white', 'black' or 'mover', got {orientation!r}."
            )
        self.orientation: Orientation = orientation
        self.coordinates = bool(coordinates)

    # -- layout -----------------------------------------------------------------------

    def check_ready(self, env: Any) -> None:
        if not env._has_reset:
            raise ResetRequired()

    def _view(self, env: ChessEnv) -> Color:
        if self.orientation == "mover":
            return env.board.turn
        return self.orientation == "white"

    def bottom_side(self, env: Any) -> str:
        return "white" if self._view(env) else "black"

    def static_key(self, env: Any) -> Any:
        return self._view(env)

    def _cell(self) -> float:
        return self.size / 8

    def _square_box(self, sq: int, view: Color) -> tuple[float, float, float, float]:
        left, top, _, _ = self.board_box()
        file, rank = sq & 7, sq >> 3
        col, row = (file, 7 - rank) if view else (7 - file, rank)
        c = self._cell()
        return left + col * c, top + row * c, left + (col + 1) * c, top + (row + 1) * c

    # -- drawing ------------------------------------------------------------------------

    def draw_static(self, painter: Painter, env: Any) -> None:
        radius = 8 * self.scale
        self.draw_board_shadow(painter, radius)
        left, top, right, bottom = self.board_box()
        dark = DARK_SQUARE.get(self.theme.name, DARK_SQUARE["dark"])
        painter.rect(left, top, right, bottom, LIGHT_SQUARE, radius=radius)
        view = self._view(env)
        for sq in range(64):
            if ((sq >> 3) + (sq & 7)) % 2 == 0:  # a1 is dark
                x0, y0, x1, y1 = self._square_box(sq, view)
                corner = self._corner_radius(sq, view, radius)
                painter.rect(x0, y0, x1, y1, dark, radius=corner)

    def _corner_radius(self, sq: int, view: Color, radius: float) -> Any:
        """Round a square's outer corner if it is a corner of the board."""
        x0, y0, _, _ = self._square_box(sq, view)
        left, top, _, _ = self.board_box()
        c = self._cell()
        col, row = round((x0 - left) / c), round((y0 - top) / c)
        corners = [(0.0, 0.0)] * 4  # top-left, top-right, bottom-right, bottom-left
        index = {(0, 0): 0, (7, 0): 1, (7, 7): 2, (0, 7): 3}.get((col, row))
        if index is None:
            return 0.0
        corners[index] = (radius, radius)
        return corners

    def draw_board(self, painter: Painter, env: Any) -> None:
        board: Board = env.board
        view = self._view(env)
        cell = self._cell()
        # The last move, tinted gold.
        if board.move_stack:
            move = board.peek()
            tint, alpha = LAST_MOVE
            for sq in (move.from_square, move.to_square):
                x0, y0, x1, y1 = self._square_box(sq, view)
                corner = self._corner_radius(sq, view, 8 * self.scale)
                painter.rect(x0, y0, x1, y1, tint, alpha=alpha, radius=corner)
        # A red glow under the king in check.
        if board.is_check():
            king = board.king(board.turn)
            if king is not None:
                x0, y0, x1, y1 = self._square_box(king, view)
                self._check_glow(painter, (x0 + x1) / 2, (y0 + y1) / 2, cell)
        if self.coordinates:
            self._draw_coordinates(painter, view)
        size = max(round(cell), 1)
        for sq, (kind, color) in board.piece_map().items():
            x0, y0, _, _ = self._square_box(sq, view)
            image = chess_piece(_piece_code(kind, color), size)
            painter.sprite(image, round(x0), round(y0), anchor="top-left")

    def _check_glow(self, painter: Painter, cx: float, cy: float, cell: float) -> None:
        """CSS `radial-gradient(circle, red .95 0%, red .5 55%, transparent 76%)` on a square."""
        far = cell * 0.7071  # the gradient's radius: centre to corner
        box = painter._box(cx - cell / 2, cy - cell / 2, cx + cell / 2, cy + cell / 2, pad=0.0)
        if box is None:
            return
        xs, ys = painter._centres(box)
        t = np.hypot(xs - cx, ys - cy) / far
        alpha = np.interp(t, [0.0, 0.55, 0.76], [0.95, 0.5, 0.0])
        inside = ((np.abs(xs - cx) <= cell / 2 + 0.5) & (np.abs(ys - cy) <= cell / 2 + 0.5)).astype(
            np.float64
        )
        painter._fill(box, alpha * inside, CHECK, 1.0)

    def _draw_coordinates(self, painter: Painter, view: Color) -> None:
        left, top, _, bottom = self.board_box()
        cell = self._cell()
        size = max(11 * self.scale, 8.0)
        light = rgb(LIGHT_SQUARE)
        dark = rgb(DARK_SQUARE.get(self.theme.name, DARK_SQUARE["dark"]))
        pad = 2.5 * self.scale
        for col in range(8):
            file = col if view else 7 - col
            square_dark = (file + (0 if view else 7)) % 2 == 0
            ink = light if square_dark else dark
            painter.text(
                left + (col + 1) * cell - pad,
                bottom - pad,
                FILE_NAMES[file],
                ink,
                size=size,
                bold=True,
                align="right",
                valign="bottom",
            )
        for row in range(8):
            rank = 7 - row if view else row
            square_dark = (rank + (0 if view else 7)) % 2 == 0
            ink = light if square_dark else dark
            painter.text(
                left + pad,
                top + row * cell + pad,
                str(rank + 1),
                ink,
                size=size,
                bold=True,
                valign="top",
            )

    # -- strips and status ----------------------------------------------------------------

    def turn(self, env: Any) -> str | None:
        if env.done or self.time_limit_reached:
            return None
        return "white" if env.board.turn else "black"

    def strip_info(self, painter: Painter, env: Any, side: str, right: float, cy: float) -> None:
        color = side == "white"
        captured, score = material(env.board)
        lead = score if color else -score
        s = self.scale
        x = right
        if lead > 0:
            x -= painter.text(
                x, cy, f"+{lead}", self.theme.muted, size=12 * s, align="right", valign="middle"
            )
            x -= 6 * s
        pieces = captured[color]
        if not pieces:
            return
        size = max(round(19 * s), 8)
        step = size * 0.66
        x -= size
        for kind in reversed(pieces):
            if color:  # White's captures are Black's pieces: filled in the text colour
                image = _ink_piece(_piece_code(kind, BLACK), size, self.theme.text_strong, 0.8)
            else:
                image = chess_piece(_piece_code(kind, WHITE), size)
            painter.sprite(image, round(x), round(cy - size / 2), anchor="top-left")
            x -= step

    def status(self, env: Any) -> Status:
        names = {side: self.side_name(side, label) for side, label in self.labels.items()}
        outcome = env.outcome
        if outcome is not None:
            how = _TERMINATIONS.get(outcome.termination, outcome.termination.lower())
            if outcome.winner is None:
                return Status("Draw", f"Drawn by {how}.", "draw")
            side = "white" if outcome.winner else "black"
            return Status(f"{names[side]} won", f"{self.labels[side]} wins by {how}.", "draw")
        if self.time_limit_reached:
            return Status("Draw", "Drawn by the move limit.", "draw")
        side = "white" if env.board.turn else "black"
        return Status(
            f"{names[side]} to move",
            "Check." if env.board.is_check() else f"Move {env.board.fullmove_number}",
            "idle",
        )


class ChessTextRenderer(BaseRenderer):
    """The board as text with chess symbols, the last move and the status.

    Args:
        orientation: `"white"`, `"black"` or `"mover"` (as for `ChessRenderer`).
        coordinates: Label the files and ranks.
    """

    def __init__(self, *, orientation: Orientation = "white", coordinates: bool = True) -> None:
        if orientation not in ("white", "black", "mover"):
            raise ValueError(
                f"orientation must be 'white', 'black' or 'mover', got {orientation!r}."
            )
        self.orientation: Orientation = orientation
        self.coordinates = bool(coordinates)

    def render(self, env: Any) -> str:
        base = env.unwrapped if hasattr(env, "unwrapped") else env
        if not isinstance(base, ChessEnv):
            raise TypeError(f"This renderer draws ChessEnv, got {type(base).__name__}.")
        if not base._has_reset:
            raise ResetRequired()
        board = base.board
        view = board.turn if self.orientation == "mover" else self.orientation == "white"
        rows = board.unicode(orientation=view).splitlines()
        if self.coordinates:
            ranks = "87654321" if view else "12345678"
            files = " ".join(FILE_NAMES if view else FILE_NAMES[::-1])
            rows = [f"{rank} {row}" for rank, row in zip(ranks, rows)] + [f"  {files}"]
        outcome = base.outcome
        if outcome is not None:
            rows.append(f"{outcome.result()} ({outcome.termination.lower().replace('_', ' ')})")
        else:
            side = "White" if board.turn else "Black"
            rows.append(f"{side} to move" + (", in check" if board.is_check() else ""))
        return "\n".join(rows)


__all__ = ["ChessRenderer", "ChessTextRenderer", "Orientation", "material"]
