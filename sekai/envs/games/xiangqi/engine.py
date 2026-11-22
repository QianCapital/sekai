"""A fast, dependency-free Xiangqi (Chinese chess) rules engine.

## Board and squares

The board has 9 files `a`..`i` (left to right as seen by Red) and 10 ranks `0`..`9`
(Red's back rank is 0, Black's is 9). Squares are integers in the **absolute frame**:

    square = rank * 9 + file          # a0 = 0, i0 = 8, a1 = 9, ..., i9 = 89

The board is a flat `list` of 90 ints. A piece is a signed int: positive for Red, negative
for Black, and its absolute value is the piece type:

| type | constant   | letter | Chinese (Red/Black) |
|------|------------|--------|---------------------|
| 1    | `KING`     | `K`    | 帥 / 將 (general)   |
| 2    | `ADVISOR`  | `A`    | 仕 / 士             |
| 3    | `ELEPHANT` | `B`    | 相 / 象             |
| 4    | `HORSE`    | `N`    | 馬 / 馬             |
| 5    | `CHARIOT`  | `R`    | 車 / 車             |
| 6    | `CANNON`   | `C`    | 炮 / 砲             |
| 7    | `SOLDIER`  | `P`    | 兵 / 卒             |

Sides are `RED = 1` and `BLACK = -1`, so `piece * side > 0` means "own piece". Per-side
tables are 3-element lists `[unused, red, black]` indexed by the side itself (Python's
negative indexing makes `table[-1]` the Black entry).

## Moves

A move is the int `from_square * 90 + to_square` in the absolute frame; its UCCI string is
the two square names, e.g. `"h2e2"`. (The environment's *actions* use the same layout but in
the mover's frame, see `sekai.envs.games.xiangqi.actions`.)

## FEN

UCCI-engine FEN (Pikafish / xqbase): ranks from 9 down to 0 separated by `/`, uppercase
Red, lowercase Black, letters `K A B N R C P` (`E` and `H` are accepted for the
elephant and horse on input; `B`/`N` are always emitted), digits for runs of empty
points, then the side to move (`w` or `r` for Red, `b` for Black), two `-`
placeholders, the no-capture half-move clock and the full-move number:

    rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1

Missing trailing fields default to `w - - 0 1`. Parsing validates the position: exactly
one general per side, at most the initial number of every other piece, every piece on a
point its type can legally occupy (palace for generals and advisors, the seven elephant
points, unmoved soldiers only on their starting files), and the side *not* to move must not
be in check (this includes the two generals facing each other).

## Rules implemented

Standard piece movement (elephant eye, horse leg, cannon screen, palace, river), the flying
general rule (a move leaving the generals facing on an open file is illegal), and no move may
leave one's own general attacked. Game-ending conditions (no legal move = loss, repetition,
perpetual check, the 120-ply no-capture rule and insufficient material) are exposed as
queries here and applied by `sekai.envs.games.xiangqi.XiangqiEnv`.

## Performance

`Position` keeps piece-square sets, general locations, a 64-bit Zobrist hash and an
incremental material + piece-square score, all updated by `Position.push` /
`Position.pop`. Move generation uses precomputed per-square tables; legality testing
skips the expensive make/check/unmake step for moves that provably cannot expose the
general (not in check, and the move neither starts on the general's rank, file or diagonal
neighbourhood nor lands on its rank or file). Repetitions are detected by comparing Zobrist
hashes of the positions since the last capture (64-bit keys: collisions are negligible).
Start-position perft(3) (79,666 leaves) takes about 0.05 s and perft(4) (3,290,240) about
1.5-2 s on one core.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from typing import Final, Literal, TypeVar

from sekai.envs.games.xiangqi.errors import IllegalMove, InvalidFen, InvalidMoveFormat

Color = Literal["red", "black"]
_T = TypeVar("_T")

RED: Final = 1
BLACK: Final = -1

EMPTY: Final = 0
KING: Final = 1
ADVISOR: Final = 2
ELEPHANT: Final = 3
HORSE: Final = 4
CHARIOT: Final = 5
CANNON: Final = 6
SOLDIER: Final = 7

PIECE_TYPES: Final = (KING, ADVISOR, ELEPHANT, HORSE, CHARIOT, CANNON, SOLDIER)
PIECE_NAMES: Final = ("", "general", "advisor", "elephant", "horse", "chariot", "cannon", "soldier")
FEN_LETTERS: Final = "KABNRCP"  # FEN_LETTERS[type - 1]
FEN_ALIASES: Final = {"E": "B", "H": "N"}
MAX_COUNTS: Final = (0, 1, 2, 2, 2, 2, 2, 5)  # per type, per side

NUM_FILES: Final = 9
NUM_RANKS: Final = 10
NUM_SQUARES: Final = 90
FILE_NAMES: Final = "abcdefghi"

NO_CAPTURE_LIMIT: Final = 120
"""Plies without a capture after which the game is drawn."""

START_FEN: Final = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"


# ----------------------------------------------------------------------
# Squares, moves and colours
# ----------------------------------------------------------------------


def square(file: int, rank: int) -> int:
    """Absolute square index of `(file, rank)`."""
    return rank * NUM_FILES + file


def square_file(sq: int) -> int:
    return sq % NUM_FILES


def square_rank(sq: int) -> int:
    return sq // NUM_FILES


def square_name(sq: int) -> str:
    """UCCI name of a square, e.g. `square_name(7 + 2 * 9) == "h2"`."""
    return FILE_NAMES[sq % NUM_FILES] + str(sq // NUM_FILES)


def parse_square(name: str) -> int:
    """Inverse of `square_name`. Raises `ValueError` on a malformed name."""
    if len(name) != 2 or name[0] not in FILE_NAMES or not "0" <= name[1] <= "9":
        raise ValueError(f"{name!r} is not a Xiangqi square (files a-i, ranks 0-9).")
    return int(name[1]) * NUM_FILES + FILE_NAMES.index(name[0])


def make_move(from_sq: int, to_sq: int) -> int:
    return from_sq * NUM_SQUARES + to_sq


def move_from(move: int) -> int:
    return move // NUM_SQUARES


def move_to(move: int) -> int:
    return move % NUM_SQUARES


def move_to_uci(move: int) -> str:
    """UCCI string of an absolute move int, e.g. `"h2e2"`."""
    return square_name(move // NUM_SQUARES) + square_name(move % NUM_SQUARES)


def move_from_uci(uci: str) -> int:
    """Absolute move int for a UCCI string. Raises `InvalidMoveFormat` if malformed."""
    if not isinstance(uci, str):
        raise InvalidMoveFormat(uci)
    text = uci.strip().lower()
    if len(text) != 4:
        raise InvalidMoveFormat(uci)
    try:
        return parse_square(text[:2]) * NUM_SQUARES + parse_square(text[2:])
    except ValueError:
        raise InvalidMoveFormat(uci) from None


def color_name(side: int) -> Color:
    """`"red"` for `RED` (1), `"black"` for `BLACK` (-1)."""
    return "red" if side == RED else "black"


def parse_color(color: object) -> int:
    """Side constant for `"red"`/`"black"` (also `"r"`, `"w"`, `"b"`) or `1`/`-1`."""
    if isinstance(color, str):
        key = color.strip().lower()
        if key in ("red", "r", "w", "white"):
            return RED
        if key in ("black", "b"):
            return BLACK
    elif isinstance(color, int) and not isinstance(color, bool) and color in (RED, BLACK):
        return int(color)
    raise ValueError(f"Unknown side {color!r}; expected 'red' or 'black'.")


# ----------------------------------------------------------------------
# Precomputed geometry
# ----------------------------------------------------------------------


def _on_board(f: int, r: int) -> bool:
    return 0 <= f < NUM_FILES and 0 <= r < NUM_RANKS


def _in_palace(f: int, r: int, side: int) -> bool:
    return 3 <= f <= 5 and (0 <= r <= 2 if side == RED else 7 <= r <= 9)


def _own_half(r: int, side: int) -> bool:
    return 0 <= r <= 4 if side == RED else 5 <= r <= 9


def _per_side(build: Callable[[int], _T]) -> list[_T]:
    """`[red, red, black]`: index with the side itself (1 or -1); slot 0 is never used."""
    red = build(RED)
    return [red, red, build(BLACK)]


def _king_moves(side: int) -> tuple[tuple[int, ...], ...]:
    table = []
    for sq in range(NUM_SQUARES):
        f, r = sq % 9, sq // 9
        targets: list[int] = []
        if _in_palace(f, r, side):
            for df, dr in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                if _in_palace(f + df, r + dr, side):
                    targets.append(square(f + df, r + dr))
        table.append(tuple(targets))
    return tuple(table)


def _advisor_moves(side: int) -> tuple[tuple[int, ...], ...]:
    table = []
    for sq in range(NUM_SQUARES):
        f, r = sq % 9, sq // 9
        targets: list[int] = []
        if _in_palace(f, r, side):
            for df, dr in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                if _in_palace(f + df, r + dr, side):
                    targets.append(square(f + df, r + dr))
        table.append(tuple(targets))
    return tuple(table)


def _elephant_moves(side: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    table = []
    for sq in range(NUM_SQUARES):
        f, r = sq % 9, sq // 9
        targets: list[tuple[int, int]] = []
        if _own_half(r, side):
            for df, dr in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                tf, tr = f + 2 * df, r + 2 * dr
                if _on_board(tf, tr) and _own_half(tr, side):
                    targets.append((square(tf, tr), square(f + df, r + dr)))
        table.append(tuple(targets))
    return tuple(table)


# (file delta, rank delta) of the jump and of the horse's leg.
_HORSE_JUMPS = (
    ((1, 2), (0, 1)),
    ((-1, 2), (0, 1)),
    ((1, -2), (0, -1)),
    ((-1, -2), (0, -1)),
    ((2, 1), (1, 0)),
    ((2, -1), (1, 0)),
    ((-2, 1), (-1, 0)),
    ((-2, -1), (-1, 0)),
)


def _horse_moves() -> tuple[tuple[tuple[int, int], ...], ...]:
    table = []
    for sq in range(NUM_SQUARES):
        f, r = sq % 9, sq // 9
        targets = []
        for (df, dr), (lf, lr) in _HORSE_JUMPS:
            if _on_board(f + df, r + dr):
                targets.append((square(f + df, r + dr), square(f + lf, r + lr)))
        table.append(tuple(targets))
    return tuple(table)


def _soldier_moves(side: int) -> tuple[tuple[int, ...], ...]:
    table = []
    for sq in range(NUM_SQUARES):
        f, r = sq % 9, sq // 9
        targets = []
        if _on_board(f, r + side):
            targets.append(square(f, r + side))
        if not _own_half(r, side):  # crossed the river: sideways moves too
            for df in (-1, 1):
                if _on_board(f + df, r):
                    targets.append(square(f + df, r))
        table.append(tuple(targets))
    return tuple(table)


def _rays() -> tuple[tuple[tuple[int, ...], ...], ...]:
    table = []
    for sq in range(NUM_SQUARES):
        f, r = sq % 9, sq // 9
        rays = []
        for df, dr in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            ray = []
            tf, tr = f + df, r + dr
            while _on_board(tf, tr):
                ray.append(square(tf, tr))
                tf, tr = tf + df, tr + dr
            if ray:
                rays.append(tuple(ray))
        table.append(tuple(rays))
    return tuple(table)


KING_MOVES = _per_side(_king_moves)
ADVISOR_MOVES = _per_side(_advisor_moves)
ELEPHANT_MOVES = _per_side(_elephant_moves)
SOLDIER_MOVES = _per_side(_soldier_moves)
HORSE_MOVES = _horse_moves()
RAYS = _rays()
"""`RAYS[sq]`: the squares in each of the four orthogonal directions, nearest first."""


def _horse_attackers() -> tuple[tuple[tuple[int, int], ...], ...]:
    table: list[list[tuple[int, int]]] = [[] for _ in range(NUM_SQUARES)]
    for sq in range(NUM_SQUARES):
        for to, leg in HORSE_MOVES[sq]:
            table[to].append((sq, leg))
    return tuple(tuple(entries) for entries in table)


def _soldier_attackers(side: int) -> tuple[tuple[int, ...], ...]:
    table: list[list[int]] = [[] for _ in range(NUM_SQUARES)]
    for sq in range(NUM_SQUARES):
        for to in SOLDIER_MOVES[side][sq]:
            table[to].append(sq)
    return tuple(tuple(entries) for entries in table)


HORSE_ATTACKERS = _horse_attackers()
"""`HORSE_ATTACKERS[sq]`: `(horse_square, leg_square)` pairs from which a horse hits `sq`."""
SOLDIER_ATTACKERS = _per_side(_soldier_attackers)
OWN_HALF = _per_side(lambda side: tuple(_own_half(sq // 9, side) for sq in range(NUM_SQUARES)))


def _lines(k: int) -> tuple[bool, ...]:
    return tuple(sq % 9 == k % 9 or sq // 9 == k // 9 for sq in range(NUM_SQUARES))


def _from_sensitive(k: int) -> tuple[bool, ...]:
    kf, kr = k % 9, k // 9
    return tuple(
        sq % 9 == kf or sq // 9 == kr or (abs(sq % 9 - kf) == 1 and abs(sq // 9 - kr) == 1)
        for sq in range(NUM_SQUARES)
    )


# Squares whose vacating (FROM) or occupation (TO) can expose a general standing on k: its
# rank and file (chariot / cannon / flying-general lines, cannon screens) and, for FROM, its
# diagonal neighbours (the legs of horses attacking it).
LINES = tuple(_lines(k) for k in range(NUM_SQUARES))
FROM_SENSITIVE = tuple(_from_sensitive(k) for k in range(NUM_SQUARES))


def _ray_through() -> tuple[tuple[tuple[int, ...] | None, ...], ...]:
    table = []
    for k in range(NUM_SQUARES):
        row: list[tuple[int, ...] | None] = [None] * NUM_SQUARES
        for ray in RAYS[k]:
            for sq in ray:
                row[sq] = ray
        table.append(tuple(row))
    return tuple(table)


def _leg_horses() -> tuple[tuple[tuple[int, ...], ...], ...]:
    table = []
    for k in range(NUM_SQUARES):
        row: list[list[int]] = [[] for _ in range(NUM_SQUARES)]
        for hsq, leg in HORSE_ATTACKERS[k]:
            row[leg].append(hsq)
        table.append(tuple(tuple(squares) for squares in row))
    return tuple(table)


# RAY_THROUGH[k][sq]: the ray of RAYS[k] holding sq (None off k's rank and file).
# LEG_HORSES[k][leg]: the points from which a horse hits k with its leg on `leg`.
RAY_THROUGH = _ray_through()
LEG_HORSES = _leg_horses()


def _closure(starts: Iterable[int], moves: tuple[tuple[int, ...], ...]) -> frozenset[int]:
    seen = set(starts)
    frontier = list(seen)
    while frontier:
        sq = frontier.pop()
        for to in moves[sq]:
            if to not in seen:
                seen.add(to)
                frontier.append(to)
    return frozenset(seen)


def _legal_squares(side: int) -> tuple[frozenset[int], ...]:
    back = 0 if side == RED else 9
    soldier_rank = 3 if side == RED else 6
    elephant_targets = tuple(tuple(to for to, _ in t) for t in ELEPHANT_MOVES[side])
    everywhere = frozenset(range(NUM_SQUARES))
    return (
        frozenset(),
        frozenset(sq for sq in range(NUM_SQUARES) if _in_palace(sq % 9, sq // 9, side)),
        _closure((square(3, back), square(5, back)), ADVISOR_MOVES[side]),
        _closure((square(2, back), square(6, back)), elephant_targets),
        everywhere,
        everywhere,
        everywhere,
        _closure((square(f, soldier_rank) for f in (0, 2, 4, 6, 8)), SOLDIER_MOVES[side]),
    )


LEGAL_SQUARES = _per_side(_legal_squares)
"""`LEGAL_SQUARES[side][type]`: every point a piece of that type and side can ever occupy."""


# ----------------------------------------------------------------------
# Zobrist keys and evaluation tables
# ----------------------------------------------------------------------

_zobrist_rng = random.Random(0x58_49_41_4E_47_51_49)  # "XIANGQI"
ZOBRIST: Final = [[_zobrist_rng.getrandbits(64) for _ in range(NUM_SQUARES)] for _ in range(15)]
"""`ZOBRIST[piece][square]` for signed pieces (negative indices are Black's rows)."""
ZOBRIST_SIDE: Final = _zobrist_rng.getrandbits(64)
"""XORed into the hash whenever Black is to move."""

MATERIAL: Final = (0, 0, 200, 200, 400, 900, 450, 100)
"""Base piece values in centipawns by type (soldiers are worth 200 once across the river)."""


def _red_square_value(t: int, f: int, r: int) -> int:
    """Material plus a small positional bonus for a Red piece of type `t` on `(f, r)`."""
    centre = 4 - abs(f - 4)  # 0 on the edge files, 4 on the central file
    if t == KING:
        return -(8 if r > 0 else 0) - (4 if f != 4 else 0)
    if t == ADVISOR:
        return 200 + (6 if (f, r) == (4, 1) else 0)
    if t == ELEPHANT:
        return 200 + (6 if (f, r) == (4, 2) else 0) - (4 if r == 4 else 0)
    if t == HORSE:
        bonus = 3 * centre + 3 * min(r, 7) - (12 if f in (0, 8) else 0) - (6 if r == 0 else 0)
        return 400 + bonus
    if t == CHARIOT:
        return (
            900 + 2 * min(r, 6) + (4 if f in (3, 5) else 0) - (6 if r == 0 and f in (0, 8) else 0)
        )
    if t == CANNON:
        return 450 + (10 if f == 4 else 0) + (4 if r in (2, 7) else 0) + centre
    # Soldier: 100 before the river, 200 after (less on the last rank, where it is stuck).
    if r <= 4:
        return 100 + (4 if f == 4 else 0)
    if r == 9:
        return 150 + 2 * centre
    return 200 + 5 * centre + 6 * (r - 5)


def _piece_square_tables() -> list[list[int]]:
    tables = [[0] * NUM_SQUARES for _ in range(15)]
    for t in PIECE_TYPES:
        for sq in range(NUM_SQUARES):
            f, r = sq % 9, sq // 9
            tables[t][sq] = _red_square_value(t, f, r)
            tables[-t][sq] = -_red_square_value(t, f, 9 - r)  # mirrored for Black
    return tables


PST: Final = _piece_square_tables()
"""`PST[piece][square]`: signed value (Red positive) of a piece on a square, in centipawns."""


def piece_value(piece: int, sq: int) -> int:
    """Unsigned material value of `piece` on `sq` (soldier value depends on the river)."""
    t = abs(piece)
    if t == SOLDIER and not OWN_HALF[1 if piece > 0 else -1][sq]:
        return 200
    return MATERIAL[t]


# FEN placement: the letter of each signed piece ("1" for an empty point, so that runs of
# "1" are collapsed into digits afterwards), indexed like ZOBRIST.
_FEN_CHARS: Final = ["1", *FEN_LETTERS, *FEN_LETTERS.lower()[::-1]]
_FEN_RUNS: Final = tuple(("1" * n, str(n)) for n in range(NUM_FILES, 1, -1))


def _placement(board: list[int]) -> str:
    """The FEN placement field of a board (ranks 9 to 0)."""
    chars = _FEN_CHARS
    text = "".join([chars[piece] for piece in board])
    placement = "/".join([text[i : i + NUM_FILES] for i in range(81, -1, -NUM_FILES)])
    for run, digit in _FEN_RUNS:  # longest first: each run becomes one digit
        placement = placement.replace(run, digit)
    return placement


# ----------------------------------------------------------------------
# FEN parsing
# ----------------------------------------------------------------------


def _parse_fen(fen: object) -> tuple[list[int], int, int, int]:
    if not isinstance(fen, str):
        raise InvalidFen(fen, "expected a string")
    parts = fen.split()
    if not 1 <= len(parts) <= 6:
        raise InvalidFen(fen, "expected 1-6 space-separated fields")
    rows = parts[0].split("/")
    if len(rows) != NUM_RANKS:
        raise InvalidFen(fen, f"expected 10 ranks separated by '/', got {len(rows)}")
    board = [EMPTY] * NUM_SQUARES
    for i, row in enumerate(rows):
        rank = 9 - i
        f = 0
        for ch in row:
            if "1" <= ch <= "9":
                f += int(ch)
            else:
                letter = FEN_ALIASES.get(ch.upper(), ch.upper())
                if letter not in FEN_LETTERS or not ch.isalpha():
                    raise InvalidFen(fen, f"unknown piece letter {ch!r} on rank {rank}")
                if f >= NUM_FILES:
                    raise InvalidFen(fen, f"rank {rank} has more than 9 points")
                t = FEN_LETTERS.index(letter) + 1
                board[rank * 9 + f] = t if ch.isupper() else -t
                f += 1
            if f > NUM_FILES:
                raise InvalidFen(fen, f"rank {rank} has more than 9 points")
        if f != NUM_FILES:
            raise InvalidFen(fen, f"rank {rank} has {f} points instead of 9")

    side_field = parts[1].lower() if len(parts) > 1 else "w"
    if side_field in ("w", "r"):
        side = RED
    elif side_field == "b":
        side = BLACK
    else:
        raise InvalidFen(fen, f"side to move must be 'w', 'r' or 'b', got {parts[1]!r}")

    try:
        halfmove = int(parts[4]) if len(parts) > 4 else 0
        fullmove = int(parts[5]) if len(parts) > 5 else 1
    except ValueError:
        raise InvalidFen(fen, "half-move clock and move number must be integers") from None
    if halfmove < 0 or fullmove < 0:
        raise InvalidFen(fen, "half-move clock and move number must be non-negative")
    fullmove = max(fullmove, 1)

    for colour in (RED, BLACK):
        name = color_name(colour)
        counts = [0] * 8
        for sq, piece in enumerate(board):
            if piece * colour > 0:
                t = abs(piece)
                counts[t] += 1
                if sq not in LEGAL_SQUARES[colour][t]:
                    raise InvalidFen(
                        fen, f"{name} {PIECE_NAMES[t]} on {square_name(sq)} is not a legal point"
                    )
        if counts[KING] != 1:
            raise InvalidFen(fen, f"{name} must have exactly one general, found {counts[KING]}")
        for t in PIECE_TYPES[1:]:
            if counts[t] > MAX_COUNTS[t]:
                raise InvalidFen(
                    fen, f"{name} has {counts[t]} {PIECE_NAMES[t]}s (at most {MAX_COUNTS[t]})"
                )
    return board, side, halfmove, fullmove


def mirror_fen(fen: str) -> str:
    """The same position seen from the other side: board rotated by 180 degrees, colours and
    side to move swapped (counters kept). The rules are invariant under this map, and the
    mover-frame observations and actions of the two positions are identical.
    """
    board, side, halfmove, fullmove = _parse_fen(fen)
    mirrored = [-board[NUM_SQUARES - 1 - sq] for sq in range(NUM_SQUARES)]
    turn = "b" if side == RED else "w"
    return f"{_placement(mirrored)} {turn} - - {halfmove} {fullmove}"


# ----------------------------------------------------------------------
# Position
# ----------------------------------------------------------------------


class Position:
    """A Xiangqi position plus the move history needed for the repetition rules.

    `Position()` is the initial position; `Position(fen)` parses a FEN (see the module
    docstring) and raises `InvalidFen` if it is malformed or illegal.

    Mutate it only through `push` / `pop`; use `copy` to search without disturbing the
    original. Attributes are public for speed and read-only by convention:

    - `board`: list of 90 signed pieces (absolute frame).
    - `side`: `RED` (1) or `BLACK` (-1), the side to move.
    - `kings[side]` / `pieces[side]`: the general's square / the set of occupied squares.
    - `hash`: 64-bit Zobrist key of placement + side to move.
    - `score`: material + piece-square evaluation in centipawns, from Red's point of view.
    - `halfmove`: plies since the last capture. `ply`: half-moves since the start of the
      game (derived from the FEN move number). `checked`: whether `side` is in check.
    """

    __slots__ = (
        "_checks",
        "_hashes",
        "_undo",
        "board",
        "checked",
        "halfmove",
        "hash",
        "kings",
        "pieces",
        "ply",
        "score",
        "side",
    )

    def __init__(self, fen: str | None = None) -> None:
        board, side, halfmove, fullmove = _parse_fen(START_FEN if fen is None else fen)
        self.board: list[int] = board
        self.side: int = side
        self.kings: list[int] = [0, 0, 0]
        self.pieces: list[set[int]] = [set(), set(), set()]
        h = ZOBRIST_SIDE if side == BLACK else 0
        score = 0
        for sq, piece in enumerate(board):
            if piece:
                colour = RED if piece > 0 else BLACK
                self.pieces[colour].add(sq)
                if abs(piece) == KING:
                    self.kings[colour] = sq
                h ^= ZOBRIST[piece][sq]
                score += PST[piece][sq]
        self.hash: int = h
        self.score: int = score
        self.halfmove: int = halfmove
        self.ply: int = 2 * (fullmove - 1) + (1 if side == BLACK else 0)
        self._undo: list[tuple[int, int, int, int, int, bool]] = []
        self._hashes: list[int] = []
        self._checks: list[bool] = []
        if self._king_attacked(self.kings[-side], -side):
            what = (
                "the generals face each other on an open file"
                if self._generals_facing()
                else f"the side not to move ({color_name(-side)}) is in check"
            )
            raise InvalidFen(fen, what)
        self.checked: bool = self._king_attacked(self.kings[side], side)

    @classmethod
    def from_fen(cls, fen: str) -> Position:
        return cls(fen)

    def copy(self) -> Position:
        """Independent copy, including the move history used for repetition detection."""
        new = Position.__new__(Position)
        new.board = self.board[:]
        new.side = self.side
        new.kings = self.kings[:]
        new.pieces = [set(), set(self.pieces[1]), set(self.pieces[2])]
        new.hash = self.hash
        new.score = self.score
        new.halfmove = self.halfmove
        new.ply = self.ply
        new.checked = self.checked
        new._undo = self._undo[:]
        new._hashes = self._hashes[:]
        new._checks = self._checks[:]
        return new

    # ------------------------------------------------------------------
    # Basic queries
    # ------------------------------------------------------------------

    @property
    def turn(self) -> Color:
        """Side to move as `"red"` or `"black"`."""
        return color_name(self.side)

    @property
    def fullmove(self) -> int:
        return self.ply // 2 + 1

    @property
    def move_stack(self) -> list[int]:
        """Moves pushed since this position was created from a FEN, oldest first."""
        return [entry[0] for entry in self._undo]

    def piece_at(self, sq: int) -> int:
        return self.board[sq]

    def fen(self) -> str:
        side = "w" if self.side == RED else "b"
        return f"{_placement(self.board)} {side} - - {self.halfmove} {self.ply // 2 + 1}"

    def in_check(self, side: int | None = None) -> bool:
        """Whether `side` (default: the side to move) is in check, flying general included."""
        if side is None or side == self.side:
            return self.checked
        return self._king_attacked(self.kings[side], side)

    def _generals_facing(self) -> bool:
        red, black = self.kings[RED], self.kings[BLACK]
        if red % 9 != black % 9:
            return False
        return all(self.board[sq] == EMPTY for sq in range(red + 9, black, 9))

    def _king_attacked(self, k: int, s: int) -> bool:
        """Is the general of side `s` standing on `k` attacked (board-only test)?"""
        b = self.board
        enemy_rook = -s * CHARIOT
        enemy_cannon = -s * CANNON
        enemy_king = -s * KING
        for ray in RAYS[k]:
            screened = False
            for sq in ray:
                q = b[sq]
                if q:
                    if screened:
                        if q == enemy_cannon:
                            return True
                        break
                    if q == enemy_rook or q == enemy_king:  # noqa: SIM109 (hot loop)
                        return True
                    screened = True
        enemy_horse = -s * HORSE
        for hsq, leg in HORSE_ATTACKERS[k]:
            if b[hsq] == enemy_horse and not b[leg]:
                return True
        enemy_soldier = -s * SOLDIER
        for psq in SOLDIER_ATTACKERS[-s][k]:  # noqa: SIM110 (faster than any())
            if b[psq] == enemy_soldier:
                return True
        return False

    def attackers(self, sq: int, by: int) -> list[int]:
        """Squares of the pieces of side `by` that attack `sq` (pseudo-legally).

        "Attack" means the piece could move to `sq` if it held an enemy piece, ignoring
        whether that move would expose its own general. A general attacks along an open file
        only when `sq` holds the enemy general (the flying-general rule).
        """
        b = self.board
        found: list[int] = []
        rook, cannon, king = by * CHARIOT, by * CANNON, by * KING
        for ray in RAYS[sq]:
            screened = False
            for s2 in ray:
                q = b[s2]
                if q:
                    if screened:
                        if q == cannon:
                            found.append(s2)
                        break
                    if q == rook or (q == king and b[sq] == -king):
                        found.append(s2)
                    screened = True
        horse = by * HORSE
        for hsq, leg in HORSE_ATTACKERS[sq]:
            if b[hsq] == horse and not b[leg]:
                found.append(hsq)
        soldier = by * SOLDIER
        for psq in SOLDIER_ATTACKERS[by][sq]:
            if b[psq] == soldier:
                found.append(psq)
        advisor = by * ADVISOR
        for asq in ADVISOR_MOVES[by][sq]:
            if b[asq] == advisor:
                found.append(asq)
        if OWN_HALF[by][sq]:
            elephant = by * ELEPHANT
            for esq, eye in ELEPHANT_MOVES[by][sq]:
                if b[esq] == elephant and not b[eye]:
                    found.append(esq)
        for ksq in KING_MOVES[by][sq]:
            if b[ksq] == king:
                found.append(ksq)
        return found

    def is_attacked(self, sq: int, by: int) -> bool:
        """Whether any piece of side `by` attacks `sq` (see `attackers`)."""
        return bool(self.attackers(sq, by))

    # ------------------------------------------------------------------
    # Move generation
    # ------------------------------------------------------------------

    def pseudo_legal_moves(self, captures_only: bool = False) -> list[int]:
        """Moves obeying piece movement rules, ignoring whether they expose the general."""
        b = self.board
        s = self.side
        out: list[int] = []
        append = out.append
        quiet = not captures_only
        for fr in self.pieces[s]:
            t = b[fr] * s
            base = fr * 90
            if t == CHARIOT:
                for ray in RAYS[fr]:
                    for to in ray:
                        q = b[to]
                        if not q:
                            if quiet:
                                append(base + to)
                        else:
                            if q * s < 0:
                                append(base + to)
                            break
            elif t == CANNON:
                for ray in RAYS[fr]:
                    it = iter(ray)
                    for to in it:
                        if not b[to]:
                            if quiet:
                                append(base + to)
                        else:
                            for to2 in it:
                                q = b[to2]
                                if q:
                                    if q * s < 0:
                                        append(base + to2)
                                    break
                            break
            elif t == HORSE:
                for to, leg in HORSE_MOVES[fr]:
                    if not b[leg]:
                        q = b[to] * s
                        if q < 0 or (quiet and not q):
                            append(base + to)
            elif t == SOLDIER:
                for to in SOLDIER_MOVES[s][fr]:
                    q = b[to] * s
                    if q < 0 or (quiet and not q):
                        append(base + to)
            elif t == ELEPHANT:
                for to, eye in ELEPHANT_MOVES[s][fr]:
                    if not b[eye]:
                        q = b[to] * s
                        if q < 0 or (quiet and not q):
                            append(base + to)
            elif t == ADVISOR:
                for to in ADVISOR_MOVES[s][fr]:
                    q = b[to] * s
                    if q < 0 or (quiet and not q):
                        append(base + to)
            else:  # KING
                for to in KING_MOVES[s][fr]:
                    q = b[to] * s
                    if q < 0 or (quiet and not q):
                        append(base + to)
        return out

    def _safe(self, move: int) -> bool:
        """Does the pseudo-legal `move` leave the mover's general unattacked?"""
        fr = move // 90
        to = move - fr * 90
        b = self.board
        s = self.side
        piece = b[fr]
        captured = b[to]
        b[to] = piece
        b[fr] = EMPTY
        k = to if piece == s * KING else self.kings[s]
        attacked = self._king_attacked(k, s)
        b[fr] = piece
        b[to] = captured
        return not attacked

    def _safe_unchecked(self, move: int) -> bool:
        """`_safe` for a side that is not in check, looking only at what the move changes.

        With its general unattacked before the move, a move by another piece can only
        expose it along the rank or file through the vacated or the occupied point (line
        pieces, cannon screens, the flying general) or by unblocking a horse's leg on the
        vacated point: the answer is exactly `_safe`'s.
        """
        fr = move // 90
        to = move - fr * 90
        b = self.board
        s = self.side
        piece = b[fr]
        if piece == s * KING:
            return self._safe(move)
        k = self.kings[s]
        captured = b[to]
        b[to] = piece
        b[fr] = EMPTY
        enemy_rook = -s * CHARIOT
        enemy_cannon = -s * CANNON
        enemy_king = -s * KING
        attacked = False
        through = RAY_THROUGH[k]
        first = through[fr]
        second = through[to]
        for ray in (first, None if second is first else second):
            if ray is None:
                continue
            screened = False
            for sq in ray:
                q = b[sq]
                if q:
                    if screened:
                        attacked = q == enemy_cannon
                        break
                    if q == enemy_rook or q == enemy_king:  # noqa: SIM109 (hot loop)
                        attacked = True
                        break
                    screened = True
            if attacked:
                break
        if not attacked:
            enemy_horse = -s * HORSE
            for hsq in LEG_HORSES[k][fr]:
                if b[hsq] == enemy_horse:
                    attacked = True
                    break
        b[fr] = piece
        b[to] = captured
        return not attacked

    def _filter_legal(self, pseudo: list[int]) -> list[int]:
        if self.checked:
            return [m for m in pseudo if self._safe(m)]
        s = self.side
        k = self.kings[s]
        b = self.board
        # The points whose vacating or occupation can expose the general (a subset of its
        # rank, file and diagonal neighbours): the rays from it that hold an enemy chariot,
        # cannon or general, the legs of enemy horses aimed at it, and its own point (its
        # moves are always tested). Other moves are safe.
        exposed = [False] * NUM_SQUARES
        exposed[k] = True
        rook, cannon, king = -s * CHARIOT, -s * CANNON, -s * KING
        for ray in RAYS[k]:
            for sq in ray:
                q = b[sq]
                if q == rook or q == cannon or q == king:  # noqa: SIM109 (hot loop)
                    for on_ray in ray:
                        exposed[on_ray] = True
                    break
        horse = -s * HORSE
        for hsq, leg in HORSE_ATTACKERS[k]:
            if b[hsq] == horse:
                exposed[leg] = True
        safe = self._safe_unchecked
        out = []
        for m in pseudo:
            fr = m // 90
            if (exposed[fr] or exposed[m - fr * 90]) and not safe(m):
                continue
            out.append(m)
        return out

    def legal_moves(self) -> list[int]:
        """All legal moves (absolute move ints) for the side to move, in generation order."""
        return self._filter_legal(self.pseudo_legal_moves())

    def legal_captures(self) -> list[int]:
        """Legal capturing moves only (used by quiescence search)."""
        return self._filter_legal(self.pseudo_legal_moves(captures_only=True))

    def is_pseudo_legal(self, move: int) -> bool:
        """Whether `move` obeys the movement rules for the side to move (general safety aside)."""
        if not 0 <= move < NUM_SQUARES * NUM_SQUARES:
            return False
        fr = move // 90
        piece = self.board[fr]
        if piece * self.side <= 0:
            return False
        to = move - fr * 90
        if self.board[to] * self.side > 0:
            return False
        # Generate only this piece's moves: cheap enough and guaranteed consistent.
        saved = self.pieces[self.side]
        self.pieces[self.side] = {fr}
        try:
            return move in self.pseudo_legal_moves()
        finally:
            self.pieces[self.side] = saved

    def is_legal(self, move: int) -> bool:
        """Whether the absolute move int `move` is legal for the side to move."""
        return self.is_pseudo_legal(move) and self._safe(move)

    def gives_check(self, move: int) -> bool:
        """Whether the legal `move` puts the opponent in check."""
        self.push(move)
        try:
            return self.checked
        finally:
            self.pop()

    def has_legal_move(self) -> bool:
        """Whether the side to move has any legal move (cheaper than `legal_moves()`)."""
        pseudo = self.pseudo_legal_moves()
        if not self.checked:
            k = self.kings[self.side]
            fs, ls = FROM_SENSITIVE[k], LINES[k]
            for m in pseudo:
                fr = m // 90
                if not (fs[fr] or ls[m - fr * 90]):
                    return True
        return any(self._safe(m) for m in pseudo)

    # ------------------------------------------------------------------
    # Making and unmaking moves
    # ------------------------------------------------------------------

    def push(self, move: int) -> None:
        """Play `move` (assumed legal; use `push_uci` for validated input)."""
        fr = move // 90
        to = move - fr * 90
        b = self.board
        s = self.side
        piece = b[fr]
        captured = b[to]
        h = self.hash
        self._undo.append((move, captured, self.halfmove, h, self.score, self.checked))
        self._hashes.append(h)
        zp = ZOBRIST[piece]
        pp = PST[piece]
        h ^= zp[fr] ^ zp[to] ^ ZOBRIST_SIDE
        score = self.score - pp[fr] + pp[to]
        if captured:
            h ^= ZOBRIST[captured][to]
            score -= PST[captured][to]
            self.pieces[-s].discard(to)
            self.halfmove = 0
        else:
            self.halfmove += 1
        own = self.pieces[s]
        own.discard(fr)
        own.add(to)
        b[to] = piece
        b[fr] = EMPTY
        if piece == s * KING:
            self.kings[s] = to
        self.side = -s
        self.hash = h
        self.score = score
        self.ply += 1
        checked = self._king_attacked(self.kings[-s], -s)
        self.checked = checked
        self._checks.append(checked)

    def pop(self) -> int:
        """Undo the last `push` and return its move."""
        move, captured, halfmove, h, score, checked = self._undo.pop()
        self._hashes.pop()
        self._checks.pop()
        fr = move // 90
        to = move - fr * 90
        b = self.board
        s = -self.side  # the side that made the move
        piece = b[to]
        b[fr] = piece
        b[to] = captured
        own = self.pieces[s]
        own.discard(to)
        own.add(fr)
        if captured:
            self.pieces[-s].add(to)
        if piece == s * KING:
            self.kings[s] = fr
        self.side = s
        self.hash = h
        self.score = score
        self.halfmove = halfmove
        self.ply -= 1
        self.checked = checked
        return move

    def push_uci(self, uci: str) -> int:
        """Validate and play a UCCI move; returns the move int. Raises `IllegalMove`."""
        move = move_from_uci(uci)
        if not self.is_legal(move):
            raise IllegalMove(move_to_uci(move), self.fen())
        self.push(move)
        return move

    # ------------------------------------------------------------------
    # Game-ending conditions
    # ------------------------------------------------------------------

    def repetition_count(self) -> int:
        """How many times the current position (placement + side) has occurred, this one included.

        Only positions since the last capture can repeat, and only positions reached through
        `push` are known (a position loaded from a FEN has no history).
        """
        h = self.hash
        hashes = self._hashes
        n = len(hashes)
        lowest = n - min(self.halfmove, n)
        count = 1
        for i in range(n - 2, lowest - 1, -2):
            if hashes[i] == h:
                count += 1
        return count

    def repetition_start(self, occurrence: int = 3) -> int:
        """History index of an earlier occurrence if the current position is a repeat, else -1.

        Returns the index `i` (into the pushed-move history) of the earliest of the last
        `occurrence - 1` earlier occurrences, so that the current position is (at least)
        its `occurrence`-th occurrence and moves `i ..` form the repetition cycle.
        `occurrence=3` is the game rule; search uses `occurrence=2`.
        """
        h = self.hash
        hashes = self._hashes
        n = len(hashes)
        lowest = n - min(self.halfmove, n)
        needed = occurrence - 1
        found = 0
        for i in range(n - 2, lowest - 1, -2):
            if hashes[i] == h:
                found += 1
                if found >= needed:
                    return i
        return -1

    def perpetual_checker(self, start: int) -> int:
        """Side that loses for perpetual check over moves `start ..` (`RED`/`BLACK`), or 0.

        A side loses when every one of its moves in the cycle gave check while the other side
        did not check with every one of its moves.
        """
        checks = self._checks
        n = len(checks)
        side_now = self.side
        all_checks = {RED: True, BLACK: True}
        moved = {RED: False, BLACK: False}
        for i in range(start, n):
            mover = side_now if (n - i) % 2 == 0 else -side_now
            moved[mover] = True
            if not checks[i]:
                all_checks[mover] = False
        red = moved[RED] and all_checks[RED]
        black = moved[BLACK] and all_checks[BLACK]
        if red and not black:
            return RED
        if black and not red:
            return BLACK
        return 0

    def insufficient_material(self) -> bool:
        """True when neither side has a chariot, horse, cannon or soldier left."""
        b = self.board
        for colour in (RED, BLACK):
            for sq in self.pieces[colour]:
                if abs(b[sq]) >= HORSE:  # HORSE, CHARIOT, CANNON, SOLDIER
                    return False
        return True

    # ------------------------------------------------------------------
    # Debugging aids
    # ------------------------------------------------------------------

    def perft(self, depth: int) -> int:
        """Number of leaf nodes of the legal move tree of the given depth."""
        if depth <= 0:
            return 1
        moves = self.legal_moves()
        if depth == 1:
            return len(moves)
        total = 0
        for m in moves:
            self.push(m)
            total += self.perft(depth - 1)
            self.pop()
        return total

    def ascii(self) -> str:
        """Text diagram, Black at the top, uppercase Red."""
        lines = []
        for rank in range(9, -1, -1):
            cells = []
            for f in range(NUM_FILES):
                piece = self.board[rank * 9 + f]
                if piece == EMPTY:
                    cells.append(".")
                else:
                    letter = FEN_LETTERS[abs(piece) - 1]
                    cells.append(letter if piece > 0 else letter.lower())
            lines.append(f"{rank} " + " ".join(cells))
            if rank == 5:
                lines.append("  " + "~" * 17)
        lines.append("  " + " ".join(FILE_NAMES))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.ascii()

    def __repr__(self) -> str:
        return f"Position('{self.fen()}')"


__all__ = [
    "Color",
    "RED",
    "BLACK",
    "EMPTY",
    "KING",
    "ADVISOR",
    "ELEPHANT",
    "HORSE",
    "CHARIOT",
    "CANNON",
    "SOLDIER",
    "PIECE_TYPES",
    "PIECE_NAMES",
    "FEN_LETTERS",
    "NUM_FILES",
    "NUM_RANKS",
    "NUM_SQUARES",
    "FILE_NAMES",
    "NO_CAPTURE_LIMIT",
    "START_FEN",
    "MATERIAL",
    "PST",
    "LEGAL_SQUARES",
    "square",
    "square_file",
    "square_rank",
    "square_name",
    "parse_square",
    "make_move",
    "move_from",
    "move_to",
    "move_to_uci",
    "move_from_uci",
    "color_name",
    "parse_color",
    "piece_value",
    "mirror_fen",
    "Position",
]
