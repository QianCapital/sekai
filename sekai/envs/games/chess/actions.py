"""AlphaZero-style move encoding for chess.

Every move is an integer in `[0, 4672)`: a (from-square, move-plane) pair with 64
from-squares and 73 move planes, exactly as in the AlphaZero paper (Silver et al., 2018):

    action = from_square * 73 + plane

Move planes (73):

* `0-55`  queen-like moves: 8 directions x 7 distances, `plane = direction * 7 + (distance - 1)`.
  Directions in order N, NE, E, SE, S, SW, W, NW (see `QUEEN_DIRECTIONS`).
* `56-63` knight moves, in the order of `KNIGHT_DELTAS`.
* `64-72` underpromotions: 3 directions (capture-left, push, capture-right) x 3 pieces
  (knight, bishop, rook), `plane = 64 + direction * 3 + piece`.

Queen promotions are encoded as ordinary queen-like pawn moves onto the last rank; the
decoder adds the queen promotion whenever the moving piece is a pawn.

Everything is expressed from the perspective of the side to move: when Black is to move
the board is mirrored vertically (`a1 <-> a8`) so that pawns always advance "up" the
board. This is what lets a single policy network play both colours, and it means that
symmetric moves share an index (`e2e4` for White and `e7e5` for Black are the same
action).

Of the 4672 indices only `NUM_VALID_ACTIONS` (1858) correspond to a geometrically
possible move; the rest are never legal and `decode_action` returns `None` for them.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sekai.envs.games.chess.board import (
    BISHOP,
    KNIGHT,
    PAWN,
    QUEEN,
    ROOK,
    WHITE,
    Board,
    Color,
    Move,
    PieceType,
    Square,
    square,
    square_file,
    square_mirror,
    square_rank,
)

NUM_SQUARES = 64
NUM_QUEEN_PLANES = 56
NUM_KNIGHT_PLANES = 8
NUM_UNDERPROMOTION_PLANES = 9
NUM_MOVE_PLANES = NUM_QUEEN_PLANES + NUM_KNIGHT_PLANES + NUM_UNDERPROMOTION_PLANES  # 73
NUM_ACTIONS = NUM_SQUARES * NUM_MOVE_PLANES  # 4672

# (file delta, rank delta) in the oriented frame, where "N" is towards the 8th rank.
QUEEN_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),  # N
    (1, 1),  # NE
    (1, 0),  # E
    (1, -1),  # SE
    (0, -1),  # S
    (-1, -1),  # SW
    (-1, 0),  # W
    (-1, 1),  # NW
)
KNIGHT_DELTAS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
)
UNDERPROMOTION_DIRECTIONS: tuple[int, ...] = (-1, 0, 1)  # file delta; rank delta is always +1
UNDERPROMOTION_PIECES: tuple[PieceType, ...] = (KNIGHT, BISHOP, ROOK)

# (from_square, to_square, promotion) in the oriented frame. Queen promotions use None.
MoveKey = tuple[int, int, int | None]


def _build_tables() -> tuple[tuple[MoveKey | None, ...], dict[MoveKey, int]]:
    action_to_move: list[MoveKey | None] = [None] * NUM_ACTIONS
    move_to_action: dict[MoveKey, int] = {}
    for from_square in range(NUM_SQUARES):
        from_file, from_rank = square_file(from_square), square_rank(from_square)
        for plane in range(NUM_MOVE_PLANES):
            promotion: int | None = None
            if plane < NUM_QUEEN_PLANES:
                direction, dist = divmod(plane, 7)
                dfile, drank = QUEEN_DIRECTIONS[direction]
                to_file, to_rank = from_file + dfile * (dist + 1), from_rank + drank * (dist + 1)
            elif plane < NUM_QUEEN_PLANES + NUM_KNIGHT_PLANES:
                dfile, drank = KNIGHT_DELTAS[plane - NUM_QUEEN_PLANES]
                to_file, to_rank = from_file + dfile, from_rank + drank
            else:
                if from_rank != 6:  # underpromotions only happen from the 7th rank
                    continue
                direction, piece = divmod(plane - NUM_QUEEN_PLANES - NUM_KNIGHT_PLANES, 3)
                to_file, to_rank = from_file + UNDERPROMOTION_DIRECTIONS[direction], from_rank + 1
                promotion = UNDERPROMOTION_PIECES[piece]
            if not (0 <= to_file < 8 and 0 <= to_rank < 8):
                continue
            key: MoveKey = (from_square, square(to_file, to_rank), promotion)
            action = from_square * NUM_MOVE_PLANES + plane
            action_to_move[action] = key
            move_to_action[key] = action
    return tuple(action_to_move), move_to_action


ACTION_TO_MOVE, MOVE_TO_ACTION = _build_tables()
NUM_VALID_ACTIONS = len(MOVE_TO_ACTION)  # 1858


def _code_tables() -> dict[Color, dict[int, int]]:
    """The action of each move code (see `Board.legal_move_codes`), for each side to move."""
    tables: dict[Color, dict[int, int]] = {}
    for color in (True, False):
        table: dict[int, int] = {}
        for (fr, to, promo), action in MOVE_TO_ACTION.items():
            absolute_from = fr if color else square_mirror(fr)
            absolute_to = to if color else square_mirror(to)
            code = absolute_from | absolute_to << 6
            if promo is None:
                table[code] = action
                if square_rank(to) == 7 and square_rank(fr) == 6:
                    table[code | QUEEN << 12] = action  # a queen promotion
            else:
                table[code | promo << 12] = action
        tables[color] = table
    return tables


ACTION_OF_CODE = _code_tables()
"""`ACTION_OF_CODE[side_to_move][move_code]`: the action index of a legal move code."""


def orient_square(sq: Square, turn: Color) -> Square:
    """Map an absolute square to the side-to-move's frame (identity for White, mirror for Black)."""
    return sq if turn == WHITE else square_mirror(sq)


def encode_move(move: Move, turn: Color = WHITE) -> int:
    """Encode `move`, played by `turn`, as an action index.

    Raises `ValueError` for moves that have no encoding (the null move).
    """
    from_square = orient_square(move.from_square, turn)
    to_square = orient_square(move.to_square, turn)
    promotion = None if move.promotion in (None, QUEEN) else move.promotion
    try:
        return MOVE_TO_ACTION[(from_square, to_square, promotion)]
    except KeyError:
        raise ValueError(f"Move {move.uci()} has no AlphaZero action encoding.") from None


def decode_action(action: int, board: Board) -> Move | None:
    """Decode an action index into a move for `board`'s side to move.

    Returns `None` when the index has no geometric meaning (e.g. moving off the board).
    The returned move is *not* checked for legality; use `board.is_legal(move)`.
    """
    if not 0 <= action < NUM_ACTIONS:
        raise ValueError(f"Action {action} is out of range [0, {NUM_ACTIONS}).")
    key = ACTION_TO_MOVE[action]
    if key is None:
        return None
    from_oriented, to_oriented, promotion = key
    turn = board.turn
    from_square = orient_square(from_oriented, turn)
    to_square = orient_square(to_oriented, turn)
    if (
        promotion is None
        and square_rank(to_oriented) == 7
        and board.piece_type_at(from_square) == PAWN
    ):
        promotion = QUEEN
    return Move(from_square, to_square, promotion)


def legal_actions(board: Board) -> NDArray[np.int64]:
    """Action indices of all legal moves in `board`, sorted."""
    table = ACTION_OF_CODE[board.turn]
    return np.sort(np.fromiter((table[m] for m in board.legal_move_codes()), dtype=np.int64))


def legal_action_mask(board: Board) -> NDArray[np.int8]:
    """Length-4672 `int8` mask with `1` at every legal action.

    `int8` matches the mask format expected by `sekai.Discrete.sample(mask)`; use
    `mask.astype(bool)` for boolean indexing.
    """
    mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
    table = ACTION_OF_CODE[board.turn]
    codes = board.legal_move_codes()
    if codes:
        mask[[table[m] for m in codes]] = 1
    return mask


def action_to_uci(action: int, board: Board) -> str | None:
    """Human-readable UCI string for `action` in `board`, or `None` if it is off-board."""
    move = decode_action(action, board)
    return None if move is None else move.uci()


__all__ = [
    "ACTION_OF_CODE",
    "ACTION_TO_MOVE",
    "KNIGHT_DELTAS",
    "MOVE_TO_ACTION",
    "NUM_ACTIONS",
    "NUM_MOVE_PLANES",
    "NUM_SQUARES",
    "NUM_VALID_ACTIONS",
    "QUEEN_DIRECTIONS",
    "UNDERPROMOTION_DIRECTIONS",
    "UNDERPROMOTION_PIECES",
    "action_to_uci",
    "decode_action",
    "encode_move",
    "legal_action_mask",
    "legal_actions",
    "orient_square",
]
