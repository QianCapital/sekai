"""AlphaZero-style board planes.

An observation is a `float32` array of shape `(14 * T + 7, 8, 8)` with values in
`[0, 1]`, channel-first (PyTorch convention) and indexed `[plane, rank, file]` in the
**side-to-move's frame**: when Black is to move the board is mirrored vertically and the
colours are swapped, so "our" pieces always start at the bottom.

For each of the `T` most recent positions (current position first, then one ply back,
and so on; missing history at the start of a game is zero-filled):

| plane | meaning |
|---|---|
| 0-5 | our pawns, knights, bishops, rooks, queens, king |
| 6-11 | their pawns, knights, bishops, rooks, queens, king |
| 12 | 1.0 if that position had occurred at least twice |
| 13 | 1.0 if that position had occurred at least three times |

followed by 7 constant planes:

| plane | meaning |
|---|---|
| 0 | colour: 1.0 if White is to move |
| 1 | total move count, `fullmove_number / MOVE_COUNT_SCALE` clipped to 1 |
| 2, 3 | our kingside / queenside castling rights |
| 4, 5 | their kingside / queenside castling rights |
| 6 | no-progress count, `halfmove_clock / NO_PROGRESS_SCALE` clipped to 1 |

`T = 8` gives the 119-plane input of the AlphaZero paper. `T = 1` (21 planes) is a good
lightweight choice for early experiments.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from sekai.envs.games.chess.board import (
    BISHOP,
    KING,
    KNIGHT,
    PAWN,
    QUEEN,
    ROOK,
    WHITE,
    Board,
    Color,
)
from sekai.spaces.box import Box

PIECE_TYPES = (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)
NUM_PIECE_PLANES = 12
PLANES_PER_POSITION = NUM_PIECE_PLANES + 2  # + 2 repetition planes
NUM_CONSTANT_PLANES = 7
DEFAULT_HISTORY_LENGTH = 8

MOVE_COUNT_SCALE = 256.0  # 512 plies, the default episode cap
NO_PROGRESS_SCALE = 100.0  # 100 half-moves without progress = fifty-move rule

Planes = NDArray[np.float32]

# Board codes of white then black pieces, in plane order (see `Board.squares`).
_CODES = np.array([1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14], dtype=np.uint8)[:, None]
# Plane order seen from Black: their pieces first, then ours, then the repetition planes.
_SWAP = np.r_[6:12, 0:6, 12:14]


def position_planes(board: Board, repetitions: int = 1) -> Planes:
    """Piece and repetition planes for `board` in the **absolute** frame (White at the bottom).

    `repetitions` is how many times this position has occurred in the game so far,
    including the current occurrence. Shape `(14, 8, 8)`.
    """
    cells = np.frombuffer(bytes(board.squares), dtype=np.uint8)
    planes = np.zeros((PLANES_PER_POSITION, 64), dtype=np.float32)
    planes[:NUM_PIECE_PLANES] = cells[None, :] == _CODES
    if repetitions >= 2:
        planes[12] = 1.0
    if repetitions >= 3:
        planes[13] = 1.0
    return planes.reshape(PLANES_PER_POSITION, 8, 8)


def orient_planes(planes: Planes, turn: Color) -> Planes:
    """Convert absolute-frame position planes into the side-to-move's frame."""
    if turn == WHITE:
        return planes
    return planes[_SWAP, ::-1, :]


def constant_values(board: Board) -> list[float]:
    """The values of the 7 constant planes (each plane is filled with one value)."""
    rights = board.clean_castling_rights()  # bits: 1 K, 2 Q, 4 k, 8 q
    ours, theirs = (rights, rights >> 2) if board.turn == WHITE else (rights >> 2, rights)
    return [
        1.0 if board.turn == WHITE else 0.0,
        min(board.fullmove_number / MOVE_COUNT_SCALE, 1.0),
        float(ours & 1),
        float(ours >> 1 & 1),
        float(theirs & 1),
        float(theirs >> 1 & 1),
        min(board.halfmove_clock / NO_PROGRESS_SCALE, 1.0),
    ]


_ONES = np.ones((1, 64), dtype=np.float32)


def constant_planes(board: Board) -> Planes:
    """Colour, move count, castling rights and no-progress planes. Shape `(7, 8, 8)`."""
    values = np.array(constant_values(board), dtype=np.float32)[:, None]
    return (values * _ONES).reshape(NUM_CONSTANT_PLANES, 8, 8)


def frames(planes: Planes) -> tuple[Planes, Planes]:
    """Position planes in White's frame and in Black's (contiguous), to encode either side."""
    return planes, np.ascontiguousarray(planes[_SWAP, ::-1, :])


class ObservationEncoder:
    """Stateless encoder turning a board plus position history into a network input."""

    def __init__(self, history_length: int = DEFAULT_HISTORY_LENGTH) -> None:
        if history_length < 1:
            raise ValueError("history_length must be at least 1.")
        self.history_length = history_length
        self.num_planes = PLANES_PER_POSITION * history_length + NUM_CONSTANT_PLANES
        self.shape: tuple[int, int, int] = (self.num_planes, 8, 8)

    def space(self) -> Box:
        return Box(0.0, 1.0, shape=self.shape, dtype=np.float32)

    def encode(self, board: Board, history: Sequence[Planes]) -> Planes:
        """Build the observation for `board`'s side to move.

        `history` holds absolute-frame `position_planes`, oldest first, with the
        **current** position last (the natural order of an append-only deque). Only the
        most recent `history_length` entries are used.
        """
        n = len(history)
        if n == 0:
            raise ValueError("history must contain at least the current position.")
        turn = board.turn
        obs = np.zeros(self.shape, dtype=np.float32)
        for i in range(min(n, self.history_length)):
            planes = history[n - 1 - i]  # current position first
            start = i * PLANES_PER_POSITION
            obs[start : start + PLANES_PER_POSITION] = orient_planes(planes, turn)
        obs[PLANES_PER_POSITION * self.history_length :] = constant_planes(board)
        return obs

    def encode_frames(self, board: Board, history: Sequence[tuple[Planes, Planes]]) -> Planes:
        """`encode` from `frames()` of each history entry (oldest first): the same array,
        without re-orienting the history at every step."""
        n = len(history)
        if n == 0:
            raise ValueError("history must contain at least the current position.")
        side = 0 if board.turn == WHITE else 1
        t = self.history_length
        parts = [history[n - 1 - i][side] for i in range(min(n, t))]
        if n < t:
            parts.append(np.zeros(((t - n) * PLANES_PER_POSITION, 8, 8), dtype=np.float32))
        parts.append(constant_planes(board))
        return np.concatenate(parts, axis=0)

    def encode_board(self, board: Board) -> Planes:
        """Encode a bare board with no history (older slots zero-filled, repetitions unknown).

        Handy for one-shot evaluation of arbitrary positions, e.g. inside a tree search.
        """
        return self.encode(board, [position_planes(board)])

    def __repr__(self) -> str:
        return f"ObservationEncoder(history_length={self.history_length}, shape={self.shape})"


__all__ = [
    "DEFAULT_HISTORY_LENGTH",
    "MOVE_COUNT_SCALE",
    "NO_PROGRESS_SCALE",
    "NUM_CONSTANT_PLANES",
    "NUM_PIECE_PLANES",
    "ObservationEncoder",
    "PIECE_TYPES",
    "PLANES_PER_POSITION",
    "constant_planes",
    "constant_values",
    "frames",
    "orient_planes",
    "position_planes",
]
