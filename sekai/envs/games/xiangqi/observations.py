"""AlphaZero-style board planes for Xiangqi.

An observation is a `float32` array of shape `(16 * T + 3, 10, 9)` with values in
`[0, 1]`, channel-first (PyTorch convention) and indexed `[plane, rank, file]` in the
**mover's frame** (the same frame as the actions, see `sekai.envs.games.xiangqi.actions`):
for Red that is the absolute board (rank 0 = Red's back rank, file 0 = file `a`); when
Black is to move the board is rotated by 180 degrees and the colours are swapped, so "our"
pieces always start at the bottom (rank index 0) and advance towards rank index 9.

For each of the `T` most recent positions (current position first, then one ply back,
and so on; missing history at the start of an episode is zero-filled):

| plane | meaning                                                                 |
|-------|-------------------------------------------------------------------------|
| 0-6   | our general, advisors, elephants, horses, chariots, cannons, soldiers   |
| 7-13  | their general, advisors, elephants, horses, chariots, cannons, soldiers |
| 14    | 1.0 if that position had occurred at least twice                        |
| 15    | 1.0 if that position had occurred at least three times                  |

"Our" always means the side to move in the *current* position, also in older history
slots. A position's repetition count is the number of times its placement and side to move
had occurred in the game up to and including that moment.

followed by 3 constant planes:

| plane | meaning                                                                    |
|-------|----------------------------------------------------------------------------|
| 0     | colour: 1.0 if Red is to move                                              |
| 1     | game length, `ply / PLY_SCALE` clipped to 1                                |
| 2     | no-capture count, `halfmove / NO_CAPTURE_SCALE` clipped to 1 (120 = draw) |

where `ply` is the number of half-moves since the start of the game (derived from the FEN
move number). `T = 8` (the default) gives 131 planes; `T = 1` gives 19.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from sekai.envs.games.xiangqi.engine import (
    NO_CAPTURE_LIMIT,
    NUM_FILES,
    NUM_RANKS,
    NUM_SQUARES,
    RED,
    Position,
)
from sekai.spaces.box import Box

NUM_PIECE_TYPES = 7
NUM_PIECE_PLANES = 2 * NUM_PIECE_TYPES  # 14
PLANES_PER_POSITION = NUM_PIECE_PLANES + 2  # + 2 repetition planes
NUM_CONSTANT_PLANES = 3
DEFAULT_HISTORY_LENGTH = 8

PLY_SCALE = 400.0  # the registered episode cap
NO_CAPTURE_SCALE = float(NO_CAPTURE_LIMIT)

Planes = NDArray[np.float32]
Frames = NDArray[np.float32]
"""A position's planes in both frames, shape `(2, 16, 10, 9)`: `[0]` in Red's frame (the
absolute frame, as `position_planes`), `[1]` in Black's (as `orient_planes` turns them)."""


def position_planes(position: Position, repetitions: int = 1) -> Planes:
    """Piece and repetition planes in the **absolute** frame (Red at the bottom, Red planes first).

    Planes 0-6 hold Red's pieces, 7-13 Black's, 14/15 the repetition flags; `repetitions`
    is how many times this position has occurred so far, this occurrence included.
    Shape `(16, 10, 9)`.
    """
    flat = np.zeros((PLANES_PER_POSITION, NUM_RANKS * NUM_FILES), dtype=np.float32)
    board = position.board
    for sq in position.pieces[1]:
        flat[board[sq] - 1, sq] = 1.0
    for sq in position.pieces[2]:
        flat[NUM_PIECE_TYPES - board[sq] - 1, sq] = 1.0
    if repetitions >= 2:
        flat[14] = 1.0
    if repetitions >= 3:
        flat[15] = 1.0
    return flat.reshape(PLANES_PER_POSITION, NUM_RANKS, NUM_FILES)


def orient_planes(planes: Planes, side: int) -> Planes:
    """Convert absolute-frame position planes into the frame of `side` (`RED`/`BLACK`)."""
    if side == RED:
        return planes
    rotated = planes[:, ::-1, ::-1]
    return np.concatenate(
        [rotated[NUM_PIECE_TYPES:NUM_PIECE_PLANES], rotated[:NUM_PIECE_TYPES], rotated[14:]],
        axis=0,
    )


def constant_planes(position: Position) -> Planes:
    """Colour, game-length and no-capture planes. Shape `(3, 10, 9)`."""
    planes = np.zeros((NUM_CONSTANT_PLANES, NUM_RANKS, NUM_FILES), dtype=np.float32)
    if position.side == RED:
        planes[0] = 1.0
    planes[1] = min(position.ply / PLY_SCALE, 1.0)
    planes[2] = min(position.halfmove / NO_CAPTURE_SCALE, 1.0)
    return planes


def _frame_offsets() -> tuple[list[list[int]], list[list[int]]]:
    """Flat offsets into a `(16 * 90,)` plane block of each signed piece on each square:
    in Red's frame (`[piece][sq]`) and in Black's (rotated, our pieces first)."""
    red = [[0] * NUM_SQUARES for _ in range(15)]
    black = [[0] * NUM_SQUARES for _ in range(15)]
    for t in range(1, NUM_PIECE_TYPES + 1):
        for sq in range(NUM_SQUARES):
            rotated = NUM_SQUARES - 1 - sq
            red[t][sq] = (t - 1) * NUM_SQUARES + sq
            red[-t][sq] = (NUM_PIECE_TYPES + t - 1) * NUM_SQUARES + sq
            black[t][sq] = (NUM_PIECE_TYPES + t - 1) * NUM_SQUARES + rotated
            black[-t][sq] = (t - 1) * NUM_SQUARES + rotated
    return red, black


_RED_OFFSETS, _BLACK_OFFSETS = _frame_offsets()
_FRAMES_SHAPE = (2, PLANES_PER_POSITION, NUM_RANKS, NUM_FILES)
_REPEATED = 14 * NUM_SQUARES  # flat offsets of the two repetition planes
_THRICE = 15 * NUM_SQUARES


def position_frames(position: Position, repetitions: int = 1) -> Frames:
    """`position_planes` in both frames at once (see `Frames`).

    The array is read-only, so a history of them can be shared between copies of an
    environment.
    """
    board = position.board
    red_offsets, black_offsets = _RED_OFFSETS, _BLACK_OFFSETS
    squares = [*position.pieces[1], *position.pieces[2]]
    pieces = [board[sq] for sq in squares]
    flat = np.zeros((2, PLANES_PER_POSITION * NUM_SQUARES), dtype=np.float32)
    flat[0, [red_offsets[p][sq] for p, sq in zip(pieces, squares)]] = 1.0
    flat[1, [black_offsets[p][sq] for p, sq in zip(pieces, squares)]] = 1.0
    if repetitions >= 2:
        flat[:, _REPEATED:_THRICE] = 1.0
    if repetitions >= 3:
        flat[:, _THRICE:] = 1.0
    flat.flags.writeable = False
    return flat.reshape(_FRAMES_SHAPE)


def advance_frames(
    frames: Frames, piece: int, fr: int, to: int, captured: int, repetitions: int
) -> Frames:
    """The `position_frames` after `piece` moved from `fr` to `to`, from those before it.

    `captured` is the piece that stood on `to` (0 if none) and `repetitions` the new
    position's repetition count. Equal to `position_frames` of the new position, at the
    cost of a copy and a few writes.
    """
    flat: Frames = frames.reshape(2, PLANES_PER_POSITION * NUM_SQUARES).copy()
    red, black = _RED_OFFSETS[piece], _BLACK_OFFSETS[piece]
    flat[0, red[fr]] = 0.0
    flat[1, black[fr]] = 0.0
    if captured:
        flat[0, _RED_OFFSETS[captured][to]] = 0.0
        flat[1, _BLACK_OFFSETS[captured][to]] = 0.0
    flat[0, red[to]] = 1.0
    flat[1, black[to]] = 1.0
    flat[:, _REPEATED:_THRICE] = 1.0 if repetitions >= 2 else 0.0
    flat[:, _THRICE:] = 1.0 if repetitions >= 3 else 0.0
    flat.flags.writeable = False
    return flat.reshape(_FRAMES_SHAPE)


class ObservationEncoder:
    """Stateless encoder turning a position plus position history into a network input."""

    def __init__(self, history_length: int = DEFAULT_HISTORY_LENGTH) -> None:
        if history_length < 1:
            raise ValueError("history_length must be at least 1.")
        self.history_length = history_length
        self.num_planes = PLANES_PER_POSITION * history_length + NUM_CONSTANT_PLANES
        self.shape: tuple[int, int, int] = (self.num_planes, NUM_RANKS, NUM_FILES)

    def space(self) -> Box:
        return Box(0.0, 1.0, shape=self.shape, dtype=np.float32)

    def encode(self, position: Position, history: Sequence[Planes]) -> Planes:
        """Build the observation for `position`'s side to move.

        `history` holds absolute-frame `position_planes`, oldest first, with the
        **current** position last (the natural order of an append-only deque). Only the most
        recent `history_length` entries are used.
        """
        if len(history) == 0:
            raise ValueError("history must contain at least the current position.")
        side = position.side
        recent = list(history)[-self.history_length :][::-1]  # current position first
        obs = np.zeros(self.shape, dtype=np.float32)
        for i, planes in enumerate(recent):
            obs[i * PLANES_PER_POSITION : (i + 1) * PLANES_PER_POSITION] = orient_planes(
                planes, side
            )
        obs[PLANES_PER_POSITION * self.history_length :] = constant_planes(position)
        return obs

    def encode_frames(self, position: Position, frames: Iterable[Frames]) -> Planes:
        """`encode` from `position_frames` entries, newest first (what `XiangqiEnv` uses).

        Gives the same array as `encode` with the matching absolute planes, without
        re-orienting the history at every step.
        """
        k = 0 if position.side == RED else 1
        n = PLANES_PER_POSITION
        stacked = n * self.history_length
        obs = np.empty(self.shape, dtype=np.float32)
        filled = 0
        for i, entry in enumerate(frames):
            if i == self.history_length:
                break
            obs[i * n : (i + 1) * n] = entry[k]
            filled = (i + 1) * n
        if filled == 0:
            raise ValueError("frames must contain at least the current position.")
        if filled < stacked:
            obs[filled:stacked] = 0.0
        obs[stacked] = 1.0 if k == 0 else 0.0
        obs[stacked + 1] = min(position.ply / PLY_SCALE, 1.0)
        obs[stacked + 2] = min(position.halfmove / NO_CAPTURE_SCALE, 1.0)
        return obs

    def encode_position(self, position: Position) -> Planes:
        """Encode a bare position: no older history, repetitions taken from its own move stack.

        Handy for one-shot evaluation of arbitrary positions, e.g. inside a tree search.
        """
        return self.encode(position, [position_planes(position, position.repetition_count())])

    def __repr__(self) -> str:
        return f"ObservationEncoder(history_length={self.history_length}, shape={self.shape})"


__all__ = [
    "NUM_PIECE_TYPES",
    "NUM_PIECE_PLANES",
    "PLANES_PER_POSITION",
    "NUM_CONSTANT_PLANES",
    "DEFAULT_HISTORY_LENGTH",
    "PLY_SCALE",
    "NO_CAPTURE_SCALE",
    "Frames",
    "position_planes",
    "position_frames",
    "advance_frames",
    "orient_planes",
    "constant_planes",
    "ObservationEncoder",
]
