"""Move encoding for Xiangqi: one action per (from-square, to-square) pair.

Every move is an integer in `[0, 8100)`:

    action = from_square * 90 + to_square

where squares are numbered in the **mover's frame**:

    square = rank * 9 + file      # file 0..8 = a..i, rank 0 = the mover's own back rank

For Red the mover's frame is the absolute board frame (`a0 = 0`, `i9 = 89`). For Black
the board is rotated by 180 degrees, `square -> 89 - square`, so Black's back rank is rank
0 and its left-hand file (as Black sees the board) is file 0. Every piece therefore always
"advances" towards higher ranks and a single policy network plays both colours: Red's
central-cannon opening `h2e2` and Black's mirror reply `b7e7` are the same action
(`25 * 90 + 22 = 2272`).

Most of the 8100 indices are never legal (`from == to`, or no piece moves that way);
`NUM_VALID_ACTIONS` counts the (from, to) pairs that some piece could ever play in the
mover's frame. `decode_action` decodes every index (it never returns `None`); use the
legal-action mask to restrict a policy to legal moves.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sekai.envs.games.xiangqi import engine as _engine
from sekai.envs.games.xiangqi.engine import (
    ADVISOR,
    ELEPHANT,
    KING,
    LEGAL_SQUARES,
    NUM_SQUARES,
    RED,
    SOLDIER,
    Color,
    Position,
    move_from_uci,
    move_to_uci,
    parse_color,
)

NUM_ACTIONS = NUM_SQUARES * NUM_SQUARES  # 8100

Side = Color | int


def orient_square(sq: int, side: Side) -> int:
    """Map an absolute square to the mover's frame (identity for Red, `89 - sq` for Black).

    The map is its own inverse, so it also converts mover-frame squares back to absolute.
    """
    return sq if parse_color(side) == RED else NUM_SQUARES - 1 - sq


def move_to_action(move: int, side: Side) -> int:
    """Action index of the absolute move int `move` played by `side`."""
    if parse_color(side) == RED:
        return move
    fr, to = divmod(move, NUM_SQUARES)
    return (NUM_SQUARES - 1 - fr) * NUM_SQUARES + (NUM_SQUARES - 1 - to)


def action_to_move(action: int, side: Side) -> int:
    """Absolute move int for `action` played by `side` (the inverse of `move_to_action`)."""
    action = int(action)
    if not 0 <= action < NUM_ACTIONS:
        raise ValueError(f"Action {action} is out of range [0, {NUM_ACTIONS}).")
    return move_to_action(action, side)  # the 180-degree rotation is an involution


def encode_move(uci: str, turn: Side = "red") -> int:
    """Action index of the UCCI move `uci` (e.g. `"h2e2"`) played by `turn`.

    Raises `InvalidMoveFormat` for malformed strings. Legality is not checked.
    """
    return move_to_action(move_from_uci(uci), turn)


def decode_action(action: int, turn: Side = "red") -> str:
    """UCCI string of `action` played by `turn`. Legality is not checked."""
    return move_to_uci(action_to_move(action, turn))


def sorted_actions(moves: list[int] | tuple[int, ...], side: int) -> NDArray[np.int64]:
    """Sorted action indices of the absolute moves `moves` played by `side` (`RED`/`BLACK`).

    Black's action of a move is `8099 - move` (both squares rotated), so sorting the moves
    once serves both colours.
    """
    ordered = np.asarray(sorted(moves), dtype=np.int64)
    if side == RED:
        return ordered
    return np.asarray(NUM_ACTIONS - 1 - ordered[::-1], dtype=np.int64)


def legal_actions(position: Position) -> NDArray[np.int64]:
    """Sorted action indices of all legal moves of `position`'s side to move."""
    return sorted_actions(position.legal_moves(), position.side)


def legal_action_mask(position: Position) -> NDArray[np.int8]:
    """Length-8100 `int8` mask with `1` at every legal action.

    `int8` matches the mask format expected by `sekai.Discrete.sample(mask)`; use
    `mask.astype(bool)` for boolean indexing.
    """
    mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
    mask[legal_actions(position)] = 1
    return mask


def _valid_actions() -> frozenset[int]:
    """(from, to) pairs, in Red's (= the mover's) frame, that some Red piece can ever play."""
    pairs: set[int] = set()
    for fr in range(NUM_SQUARES):
        targets: set[int] = set()
        for ray in _engine.RAYS[fr]:  # chariot / cannon / general lines
            targets.update(ray)
        targets.update(to for to, _ in _engine.HORSE_MOVES[fr])
        if fr in LEGAL_SQUARES[RED][KING]:
            targets.update(_engine.KING_MOVES[RED][fr])
        if fr in LEGAL_SQUARES[RED][ADVISOR]:
            targets.update(_engine.ADVISOR_MOVES[RED][fr])
        if fr in LEGAL_SQUARES[RED][ELEPHANT]:
            targets.update(to for to, _ in _engine.ELEPHANT_MOVES[RED][fr])
        if fr in LEGAL_SQUARES[RED][SOLDIER]:
            targets.update(_engine.SOLDIER_MOVES[RED][fr])
        pairs.update(fr * NUM_SQUARES + to for to in targets)
    return frozenset(pairs)


VALID_ACTIONS = _valid_actions()
"""Actions that some piece could ever play in the mover's frame."""
NUM_VALID_ACTIONS = len(VALID_ACTIONS)

__all__ = [
    "NUM_ACTIONS",
    "NUM_VALID_ACTIONS",
    "VALID_ACTIONS",
    "orient_square",
    "move_to_action",
    "action_to_move",
    "encode_move",
    "decode_action",
    "sorted_actions",
    "legal_actions",
    "legal_action_mask",
]
