"""Human-readable Xiangqi move notation: WXF (English) and traditional Chinese.

Both notations describe a move relative to the mover, and both are computed in the
position *before* the move. A move is written as four parts:

    <piece> <file> <operator> <number>        e.g.  C2=5   炮二平五

- **Piece**: WXF letters `K A E H R C P` (general, advisor, elephant, horse, chariot,
  cannon, soldier). Chinese: Red `帥 仕 相 馬 車 炮 兵`, Black `將 士 象 馬 車 砲 卒`.
- **File** of the moving piece, counted 1-9 from the mover's right: for Red file 1 is file
  `i` (Red's right-hand edge), for Black file 1 is file `a`. Chinese uses `一`..`九`
  for Red and the digits `1`..`9` for Black.
- **Operator**: `+` / 進 advance (towards the opponent), `-` / 退 retreat,
  `=` / 平 traverse (sideways along the rank).
- **Number**: for pieces that move in straight lines (general, chariot, cannon, soldier)
  the number of ranks moved after `+`/`-` and the destination file after `=`. For
  pieces that move diagonally (horse, elephant, advisor) always the destination file.

Examples from the start position: `h2e2` = `C2=5` = 炮二平五, `h9g7` = `H8+7` =
馬8進7, `i0h0` (after `h0g2`) = `R1=2` = 車一平二, `c3c4` = `P7+1` = 兵七進一.

## Tandem pieces

When the moving piece shares its file with other pieces of the same type and colour, a
position marker **replaces the file number** and is written **before** the piece, mirroring
the Chinese 前/後 ("front" is the piece further advanced towards the opponent):

| pieces | WXF markers (front to rear) | Chinese markers (front to rear)  |
|--------|-----------------------------|----------------------------------|
| 2      | `+`, `-`                    | 前, 後                           |
| 3      | `+`, `=`, `-`               | 前, 中, 後                       |
| 4 or 5 | `1`, `2`, ... `5`           | 一, 二, 三, 四, 五 (both colours) |

e.g. `+R+3` / 前車進三, `-C=5` / 後炮平五, `=P=4` / 中兵平四. Three or more can only
happen with soldiers. If soldiers stand in tandem on **more than one** file, the marker is
followed by the file number in place of the piece (`+7=6` / 前七平六, Black `-3+1` /
後3進1), as in Chinese practice.

Advisors and elephants never use markers: two of them on one file can always be told apart
by the operator (the rear one can only advance, the front one only retreat), so they keep
the ordinary form, e.g. `A6+5` / 仕六進五 and `A6-5` / 仕六退五.
"""

from __future__ import annotations

from dataclasses import dataclass

from sekai.envs.games.xiangqi.engine import (
    ADVISOR,
    BLACK,
    CANNON,
    CHARIOT,
    ELEPHANT,
    HORSE,
    KING,
    NUM_FILES,
    RED,
    SOLDIER,
    Position,
    move_from_uci,
    move_to_uci,
)
from sekai.envs.games.xiangqi.errors import IllegalMove

WXF_LETTERS = {
    KING: "K",
    ADVISOR: "A",
    ELEPHANT: "E",
    HORSE: "H",
    CHARIOT: "R",
    CANNON: "C",
    SOLDIER: "P",
}
CHINESE_PIECES = {
    RED: {
        KING: "帥",
        ADVISOR: "仕",
        ELEPHANT: "相",
        HORSE: "馬",
        CHARIOT: "車",
        CANNON: "炮",
        SOLDIER: "兵",
    },
    BLACK: {
        KING: "將",
        ADVISOR: "士",
        ELEPHANT: "象",
        HORSE: "馬",
        CHARIOT: "車",
        CANNON: "砲",
        SOLDIER: "卒",
    },
}
CHINESE_NUMERALS = "一二三四五六七八九"
CHINESE_OPERATORS = {"+": "進", "-": "退", "=": "平"}

WXF_MARKERS = {2: "+-", 3: "+=-", 4: "1234", 5: "12345"}
CHINESE_MARKERS = {2: "前後", 3: "前中後", 4: "一二三四", 5: "一二三四五"}

_DIAGONAL_MOVERS = (HORSE, ELEPHANT, ADVISOR)


def file_number(file: int, side: int) -> int:
    """File `0..8` (`a`..`i`) as numbered 1-9 from `side`'s own right-hand edge."""
    return NUM_FILES - file if side == RED else file + 1


def _number(n: int, side: int) -> str:
    return CHINESE_NUMERALS[n - 1] if side == RED else str(n)


@dataclass(frozen=True, slots=True)
class MoveDescription:
    """The notation-independent parts of a move, from the mover's point of view."""

    side: int
    """`RED` or `BLACK`."""

    kind: int
    """Piece type (`KING` .. `SOLDIER`)."""

    file: int
    """File number 1-9 of the moving piece, counted from the mover's right."""

    operator: str
    """`"+"` advance, `"-"` retreat or `"="` traverse."""

    number: int
    """Ranks moved (straight movers advancing/retreating) or destination file number."""

    tandem: int = 1
    """Pieces of this type and colour on the moving piece's file when a marker is needed, else 1."""

    index: int = 0
    """Position of the moving piece among them, 0 = front."""

    multi_file: bool = False
    """Soldiers also stand in tandem on another file: the marker is followed by the file."""


def describe_move(position: Position, move: int | str) -> MoveDescription:
    """Analyse a legal move (UCCI string or absolute move int) for notation.

    Raises `IllegalMove` if the move is not legal.
    """
    if isinstance(move, str):
        uci = move
        move = move_from_uci(move)
    else:
        uci = move_to_uci(move) if 0 <= move < 8100 else str(move)
    if not position.is_legal(move):
        raise IllegalMove(uci, position.fen(), reason="notation needs a legal move")
    board = position.board
    fr, to = divmod(move, 90)
    piece = board[fr]
    side = RED if piece > 0 else BLACK
    kind = abs(piece)
    from_file, from_rank = fr % 9, fr // 9
    to_file, to_rank = to % 9, to // 9

    advance = (to_rank - from_rank) * side
    operator = "+" if advance > 0 else "-" if advance < 0 else "="
    if kind in _DIAGONAL_MOVERS or operator == "=":
        number = file_number(to_file, side)
    else:
        number = abs(advance)
    description = MoveDescription(side, kind, file_number(from_file, side), operator, number)

    if kind in (KING, ADVISOR, ELEPHANT):
        return description
    ranks = [r for r in range(10) if board[r * 9 + from_file] == piece]
    if len(ranks) == 1:
        return description
    ranks.sort(reverse=side == RED)  # front = furthest advanced towards the opponent
    multi_file = kind == SOLDIER and any(
        sum(1 for r in range(10) if board[r * 9 + f] == piece) > 1
        for f in range(NUM_FILES)
        if f != from_file
    )
    return MoveDescription(
        side,
        kind,
        description.file,
        operator,
        number,
        tandem=len(ranks),
        index=ranks.index(from_rank),
        multi_file=multi_file,
    )


def to_wxf(position: Position, move: int | str) -> str:
    """WXF notation of `move` (UCCI string or absolute move int) in `position`.

    Raises `IllegalMove` if the move is not legal.
    """
    d = describe_move(position, move)
    letter = WXF_LETTERS[d.kind]
    tail = f"{d.operator}{d.number}"
    if d.tandem == 1:
        return f"{letter}{d.file}{tail}"
    marker = WXF_MARKERS[d.tandem][d.index]
    return f"{marker}{d.file}{tail}" if d.multi_file else f"{marker}{letter}{tail}"


def to_chinese(position: Position, move: int | str) -> str:
    """Traditional Chinese notation of `move` in `position`, e.g. `"炮二平五"`.

    Raises `IllegalMove` if the move is not legal.
    """
    d = describe_move(position, move)
    name = CHINESE_PIECES[d.side][d.kind]
    tail = CHINESE_OPERATORS[d.operator] + _number(d.number, d.side)
    if d.tandem == 1:
        return f"{name}{_number(d.file, d.side)}{tail}"
    marker = CHINESE_MARKERS[d.tandem][d.index]
    if d.multi_file:
        return f"{marker}{_number(d.file, d.side)}{tail}"
    return f"{marker}{name}{tail}"


__all__ = [
    "WXF_LETTERS",
    "CHINESE_PIECES",
    "CHINESE_NUMERALS",
    "CHINESE_OPERATORS",
    "WXF_MARKERS",
    "CHINESE_MARKERS",
    "MoveDescription",
    "file_number",
    "describe_move",
    "to_wxf",
    "to_chinese",
]
