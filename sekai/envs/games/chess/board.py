"""A fast, dependency-free chess rules engine.

Squares are numbered like python-chess: `a1 = 0`, `b1 = 1`, ..., `h8 = 63`, so
`square = rank * 8 + file`. Colours are booleans, `WHITE = True` and `BLACK = False`, and
piece types the integers `PAWN = 1` ... `KING = 6`.

`Board` holds a position and the moves that led to it. It generates the legal moves
directly (from the pins and checks on the king, so a move is never made and unmade to test
it, except en passant captures), plays and takes back moves (`push`, `pop`), reads and
writes FEN, writes and parses SAN, and decides how a game ends. Its rules follow
python-chess exactly where they can differ between implementations:

- FEN parsing accepts what python-chess accepts and `is_valid()` rejects what its
  `status()` flags (missing or extra kings, more than 16 pieces or 8 pawns of a colour,
  pawns on the back ranks, castling rights without their king and rook in place, an en
  passant square no double push can have left, the side not to move in check, impossible
  checks).
- Castling rights are lost when the king moves, and when anything moves from or to a
  rook's original square.
- `fen()` writes the en passant square only when an en passant capture is legal.
- `outcome()` checks checkmate, insufficient material, stalemate, the seventy-five-move
  rule and fivefold repetition, in that order; insufficient material follows
  python-chess's rules (a lone minor piece cannot win, unless the opponent has pieces
  that make a self-mate possible).

Moves are `Move(from_square, to_square, promotion)`; castling is the king's two-square
move (`e1g1`), as in UCI.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final

Square = int
Color = bool
PieceType = int

WHITE: Final = True
BLACK: Final = False
COLORS: Final = (WHITE, BLACK)

PAWN: Final = 1
KNIGHT: Final = 2
BISHOP: Final = 3
ROOK: Final = 4
QUEEN: Final = 5
KING: Final = 6
PIECE_TYPES: Final = (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)
PIECE_SYMBOLS: Final = (None, "p", "n", "b", "r", "q", "k")
PIECE_NAMES: Final = (None, "pawn", "knight", "bishop", "rook", "queen", "king")

FILE_NAMES: Final = "abcdefgh"
RANK_NAMES: Final = "12345678"
SQUARE_NAMES: Final = tuple(f + r for r in RANK_NAMES for f in FILE_NAMES)

STARTING_FEN: Final = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
STARTING_BOARD_FEN: Final = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

# Piece codes on the board: 0 empty, 1-6 a white piece of that type, 9-14 a black one.
_BLACK_BIT = 8

# Castling rights as bits, by the rook's original square.
_WK, _WQ, _BK, _BQ = 1, 2, 4, 8
_RIGHT_OF_SQUARE = {7: _WK, 0: _WQ, 63: _BK, 56: _BQ}
_LOSE_RIGHTS = [15] * 64  # rights kept when a move touches a square
for _sq, _bit in _RIGHT_OF_SQUARE.items():
    _LOSE_RIGHTS[_sq] = 15 & ~_bit
_LOSE_RIGHTS_KING = {WHITE: 15 & ~(_WK | _WQ), BLACK: 15 & ~(_BK | _BQ)}

FEN_CASTLING_REGEX: Final = re.compile(r"^(?:-|[KQABCDEFGH]{0,2}[kqabcdefgh]{0,2})\Z")
SAN_REGEX: Final = re.compile(
    r"^([NBKRQ])?([a-h])?([1-8])?[\-x]?([a-h][1-8])(=?[nbrqkNBRQK])?[\+#]?\Z"
)


def square(file: int, rank: int) -> Square:
    """The square on `file` (0-7, a-h) and `rank` (0-7, 1-8)."""
    return rank * 8 + file


def square_file(sq: Square) -> int:
    return sq & 7


def square_rank(sq: Square) -> int:
    return sq >> 3


def square_name(sq: Square) -> str:
    """`"e4"` for square 28."""
    return SQUARE_NAMES[sq]


def parse_square(name: str) -> Square:
    """The square named `name` (`"e4"`); raises `ValueError` for anything else."""
    try:
        return SQUARE_NAMES.index(name)
    except ValueError:
        raise ValueError(f"{name!r} is not a square name (a1 to h8).") from None


def square_mirror(sq: Square) -> Square:
    """The square mirrored vertically (`a1 <-> a8`)."""
    return sq ^ 56


# ----------------------------------------------------------------------
# Errors (all ValueError, as python-chess's are)
# ----------------------------------------------------------------------


class InvalidMoveError(ValueError):
    """A move string that is not well formed (UCI or SAN)."""


class IllegalMoveError(ValueError):
    """A move that is not legal in the position."""


class AmbiguousMoveError(ValueError):
    """A SAN move that matches more than one legal move."""


# ----------------------------------------------------------------------
# Moves
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Move:
    """A move from one square to another, promoting to `promotion` (a piece type) or not.

    Castling is written as the king's move (`e1g1`). `Move.null()` is the null move
    (`0000`), which only `Board.push` accepts.
    """

    from_square: Square
    to_square: Square
    promotion: PieceType | None = None

    def uci(self) -> str:
        """The move in UCI notation: `"e2e4"`, `"e7e8q"`, `"0000"` for the null move."""
        if not self:
            return "0000"
        text = SQUARE_NAMES[self.from_square] + SQUARE_NAMES[self.to_square]
        if self.promotion:
            text += PIECE_SYMBOLS[self.promotion]  # type: ignore[operator]
        return text

    def __bool__(self) -> bool:
        return bool(self.from_square or self.to_square or self.promotion)

    def __str__(self) -> str:
        return self.uci()

    def __repr__(self) -> str:
        return f"Move.from_uci({self.uci()!r})"

    @classmethod
    def from_uci(cls, uci: str) -> Move:
        """Parse UCI notation; raises `InvalidMoveError` if it is not well formed."""
        if uci == "0000":
            return cls.null()
        if len(uci) not in (4, 5):
            raise InvalidMoveError(f"expected uci string to be of length 4 or 5: {uci!r}")
        try:
            from_square = SQUARE_NAMES.index(uci[0:2])
            to_square = SQUARE_NAMES.index(uci[2:4])
        except ValueError:
            raise InvalidMoveError(f"invalid uci: {uci!r}") from None
        promotion = None
        if len(uci) == 5:
            if uci[4] not in "pnbrqk":
                raise InvalidMoveError(f"invalid uci: {uci!r}")
            promotion = PIECE_SYMBOLS.index(uci[4])
        if from_square == to_square:
            raise InvalidMoveError(f"invalid uci (use 0000 for null moves): {uci!r}")
        return cls(from_square, to_square, promotion)

    @classmethod
    def null(cls) -> Move:
        return cls(0, 0, None)


# ----------------------------------------------------------------------
# Geometry tables
# ----------------------------------------------------------------------

# Directions N, NE, E, SE, S, SW, W, NW as (file delta, rank delta).
_DIRECTIONS: Final = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))
_ORTHOGONAL: Final = (0, 2, 4, 6)
_DIAGONAL: Final = (1, 3, 5, 7)
_KNIGHT_DELTAS: Final = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))


def _build_rays() -> tuple[tuple[tuple[Square, ...], ...], ...]:
    rays = []
    for sq in range(64):
        f, r = sq & 7, sq >> 3
        per_dir = []
        for df, dr in _DIRECTIONS:
            ray = []
            tf, tr = f + df, r + dr
            while 0 <= tf < 8 and 0 <= tr < 8:
                ray.append(tr * 8 + tf)
                tf, tr = tf + df, tr + dr
            per_dir.append(tuple(ray))
        rays.append(tuple(per_dir))
    return tuple(rays)


def _build_leaper(deltas: tuple[tuple[int, int], ...]) -> tuple[tuple[Square, ...], ...]:
    out = []
    for sq in range(64):
        f, r = sq & 7, sq >> 3
        out.append(
            tuple((r + dr) * 8 + f + df for df, dr in deltas if 0 <= f + df < 8 and 0 <= r + dr < 8)
        )
    return tuple(out)


RAYS: Final = _build_rays()
"""`RAYS[sq][d]`: the squares from `sq` in direction `d` (N, NE, E, SE, S, SW, W, NW)."""
KNIGHT_TARGETS: Final = _build_leaper(_KNIGHT_DELTAS)
KING_TARGETS: Final = _build_leaper(_DIRECTIONS)
# Squares a pawn of each colour on a square attacks.
PAWN_ATTACKS: Final = {
    WHITE: _build_leaper(((-1, 1), (1, 1))),
    BLACK: _build_leaper(((-1, -1), (1, -1))),
}


def _build_direction() -> tuple[tuple[int, ...], ...]:
    table = [[-1] * 64 for _ in range(64)]
    for sq in range(64):
        for d, ray in enumerate(RAYS[sq]):
            for t in ray:
                table[sq][t] = d
    return tuple(tuple(row) for row in table)


DIRECTION: Final = _build_direction()
"""`DIRECTION[a][b]`: the direction from `a` to `b` if they share a line, else -1."""

_DARK_SQUARES: Final = frozenset(sq for sq in range(64) if (sq >> 3) % 2 == (sq & 7) % 2)


def _between(a: Square, b: Square) -> tuple[Square, ...]:
    """The squares strictly between `a` and `b` on a shared line (empty if not aligned)."""
    d = DIRECTION[a][b]
    if d < 0:
        return ()
    out = []
    for sq in RAYS[a][d]:
        if sq == b:
            break
        out.append(sq)
    return tuple(out)


# ----------------------------------------------------------------------
# The board
# ----------------------------------------------------------------------

_UNDO = tuple[int, int, int, int | None, int, int, int, int]
"""(move, captured code, capture square, ep square, castling, halfmove, flags, piece)."""


class Board:
    """A chess position plus the moves that led to it (see the module docstring).

    `Board()` is the standard starting position; `Board(fen)` parses a FEN (raises
    `ValueError` if it is malformed; `is_valid()` checks it describes a legal position).
    Mutate it only with `push` and `pop`; `copy()` gives an independent board.
    """

    __slots__ = (
        "_checkers",
        "_legal",
        "_undo",
        "castling",
        "ep_square",
        "fullmove_number",
        "halfmove_clock",
        "kings",
        "move_stack",
        "squares",
        "turn",
    )

    def __init__(self, fen: str | None = STARTING_FEN) -> None:
        self.squares = bytearray(64)
        """Piece code per square: 0 empty, 1-6 a white piece of that type, 9-14 black."""
        self.turn: Color = WHITE
        self.castling = 0
        self.ep_square: Square | None = None
        self.halfmove_clock = 0
        self.fullmove_number = 1
        self.kings: list[Square] = [-1, -1]
        """King squares, indexed by colour (`kings[WHITE]`), -1 if there is none."""
        self.move_stack: list[Move] = []
        self._undo: list[_UNDO] = []
        self._legal: list[int] | None = None
        self._checkers: list[Square] | None = None
        if fen is not None:
            self.set_fen(fen)

    # -- construction ----------------------------------------------------------

    @classmethod
    def empty(cls) -> Board:
        return cls(None)

    def copy(self, *, stack: bool | int = True) -> Board:
        """An independent copy. `stack=False` drops the move history, an int keeps that many
        of the most recent moves (enough for `pop()` that far back)."""
        new = Board.__new__(Board)
        new.squares = bytearray(self.squares)
        new.turn = self.turn
        new.castling = self.castling
        new.ep_square = self.ep_square
        new.halfmove_clock = self.halfmove_clock
        new.fullmove_number = self.fullmove_number
        new.kings = self.kings[:]
        if stack is True:
            keep = len(self.move_stack)
        elif stack is False:
            keep = 0
        else:
            keep = min(int(stack), len(self.move_stack))
        new.move_stack = self.move_stack[len(self.move_stack) - keep :]
        new._undo = self._undo[len(self._undo) - keep :]
        new._legal = self._legal
        new._checkers = self._checkers
        return new

    def set_fen(self, fen: str) -> None:
        """Set the position from a FEN; raises `ValueError` if it is malformed.

        Missing trailing fields default to white to move, no castling, no en passant
        square, `0` and `1`. Use `is_valid()` to check the position is a legal one.
        """
        parts = fen.split()
        if not parts:
            raise ValueError("empty fen")
        board_part = parts.pop(0)
        turn = WHITE
        if parts:
            turn_part = parts.pop(0)
            if turn_part == "w":
                turn = WHITE
            elif turn_part == "b":
                turn = BLACK
            else:
                raise ValueError(f"expected 'w' or 'b' for turn part of fen: {fen!r}")
        castling_part = "-"
        if parts:
            castling_part = parts.pop(0)
            if not FEN_CASTLING_REGEX.match(castling_part):
                raise ValueError(f"invalid castling part in fen: {fen!r}")
        ep_square: Square | None = None
        if parts:
            ep_part = parts.pop(0)
            if ep_part != "-":
                if ep_part not in SQUARE_NAMES:
                    raise ValueError(f"invalid en passant part in fen: {fen!r}")
                ep_square = SQUARE_NAMES.index(ep_part)
        halfmove = 0
        if parts:
            try:
                halfmove = int(parts.pop(0))
            except ValueError:
                raise ValueError(f"invalid half-move clock in fen: {fen!r}") from None
            if halfmove < 0:
                raise ValueError(f"half-move clock cannot be negative: {fen!r}")
        fullmove = 1
        if parts:
            try:
                fullmove = int(parts.pop(0))
            except ValueError:
                raise ValueError(f"invalid fullmove number in fen: {fen!r}") from None
            if fullmove < 0:
                raise ValueError(f"fullmove number cannot be negative: {fen!r}")
            fullmove = max(fullmove, 1)
        if parts:
            raise ValueError(f"fen string has more parts than expected: {fen!r}")

        self._set_board_fen(board_part)
        self.turn = turn
        self.castling = self._castling_from_fen(castling_part)
        self.ep_square = ep_square
        self.halfmove_clock = halfmove
        self.fullmove_number = fullmove
        self.move_stack = []
        self._undo = []
        self._invalidate()

    def _set_board_fen(self, fen: str) -> None:
        fen = fen.strip()
        if " " in fen:
            raise ValueError(f"expected position part of fen, got multiple parts: {fen!r}")
        rows = fen.split("/")
        if len(rows) != 8:
            raise ValueError(f"expected 8 rows in position part of fen: {fen!r}")
        squares = bytearray(64)
        for i, row in enumerate(rows):
            rank = 7 - i
            field = 0
            previous_was_digit = False
            previous_was_piece = False
            for c in row:
                if c in "12345678":
                    if previous_was_digit:
                        raise ValueError(f"two subsequent digits in position part of fen: {fen!r}")
                    field += int(c)
                    previous_was_digit, previous_was_piece = True, False
                elif c == "~":
                    if not previous_was_piece:
                        raise ValueError(f"'~' not after piece in position part of fen: {fen!r}")
                    previous_was_digit, previous_was_piece = False, False
                elif c.lower() in "pnbrqk":
                    if field < 8:
                        code = "pnbrqk".index(c.lower()) + 1
                        squares[rank * 8 + field] = code if c.isupper() else code | _BLACK_BIT
                    field += 1
                    previous_was_digit, previous_was_piece = False, True
                else:
                    raise ValueError(f"invalid character in position part of fen: {fen!r}")
            if field != 8:
                raise ValueError(f"expected 8 columns per row in position part of fen: {fen!r}")
        self.squares = squares
        self.kings = [-1, -1]
        for sq, code in enumerate(squares):
            if code & 7 == KING:
                self.kings[not code & _BLACK_BIT] = sq

    def _castling_from_fen(self, text: str) -> int:
        """Castling rights as written (python-chess's reading), before checking them."""
        rights = 0
        if text == "-":
            return 0
        for flag in text:
            color = flag.isupper()
            low = flag.lower()
            rank = 0 if color else 7
            if low in ("k", "h"):
                rights |= _WK if color else _BK
            elif low in ("q", "a"):
                rights |= _WQ if color else _BQ
            else:
                # A rook on another file: castling rights sekai (standard chess) never has.
                rights |= 1 << (16 + rank + FILE_NAMES.index(low))
        return rights

    # -- basic queries ---------------------------------------------------------

    def piece_at(self, sq: Square) -> tuple[PieceType, Color] | None:
        """`(piece_type, color)` on `sq`, or `None`."""
        code = self.squares[sq]
        if not code:
            return None
        return code & 7, not code & _BLACK_BIT

    def piece_type_at(self, sq: Square) -> PieceType | None:
        code = self.squares[sq]
        return code & 7 if code else None

    def color_at(self, sq: Square) -> Color | None:
        code = self.squares[sq]
        return (not code & _BLACK_BIT) if code else None

    def piece_map(self) -> dict[Square, tuple[PieceType, Color]]:
        """Every piece on the board, by square."""
        return {
            sq: (code & 7, not code & _BLACK_BIT) for sq, code in enumerate(self.squares) if code
        }

    def king(self, color: Color) -> Square | None:
        sq = self.kings[color]
        return None if sq < 0 else sq

    def clean_castling_rights(self) -> int:
        """The castling rights that are real: a rook on its corner and the king on e1/e8."""
        rights = self.castling & 15
        squares = self.squares
        if squares[4] != KING:
            rights &= ~(_WK | _WQ)
        if squares[60] != KING | _BLACK_BIT:
            rights &= ~(_BK | _BQ)
        if squares[7] != ROOK:
            rights &= ~_WK
        if squares[0] != ROOK:
            rights &= ~_WQ
        if squares[63] != ROOK | _BLACK_BIT:
            rights &= ~_BK
        if squares[56] != ROOK | _BLACK_BIT:
            rights &= ~_BQ
        return rights

    def has_kingside_castling_rights(self, color: Color) -> bool:
        return bool(self.clean_castling_rights() & (_WK if color else _BK))

    def has_queenside_castling_rights(self, color: Color) -> bool:
        return bool(self.clean_castling_rights() & (_WQ if color else _BQ))

    def has_castling_rights(self, color: Color) -> bool:
        return bool(self.clean_castling_rights() & ((_WK | _WQ) if color else (_BK | _BQ)))

    # -- attacks -----------------------------------------------------------------

    def is_attacked_by(self, color: Color, sq: Square) -> bool:
        """Whether a piece of `color` attacks `sq` (pinned pieces count)."""
        return self._attacked(sq, color, self.squares)

    @staticmethod
    def _attacked(sq: Square, by: Color, squares: bytearray) -> bool:
        side = 0 if by else _BLACK_BIT
        knight, king, pawn = KNIGHT | side, KING | side, PAWN | side
        rook, bishop, queen = ROOK | side, BISHOP | side, QUEEN | side
        for t in KNIGHT_TARGETS[sq]:
            if squares[t] == knight:
                return True
        for t in KING_TARGETS[sq]:
            if squares[t] == king:
                return True
        for t in PAWN_ATTACKS[not by][sq]:
            if squares[t] == pawn:
                return True
        rays = RAYS[sq]
        for d in _ORTHOGONAL:
            for t in rays[d]:
                p = squares[t]
                if p:
                    if p in (rook, queen):
                        return True
                    break
        for d in _DIAGONAL:
            for t in rays[d]:
                p = squares[t]
                if p:
                    if p in (bishop, queen):
                        return True
                    break
        return False

    def attackers(self, color: Color, sq: Square) -> list[Square]:
        """Squares of the pieces of `color` that attack `sq`."""
        squares = self.squares
        side = 0 if color else _BLACK_BIT
        out = [t for t in KNIGHT_TARGETS[sq] if squares[t] == KNIGHT | side]
        out += [t for t in KING_TARGETS[sq] if squares[t] == KING | side]
        out += [t for t in PAWN_ATTACKS[not color][sq] if squares[t] == PAWN | side]
        rays = RAYS[sq]
        for d in range(8):
            sliders = (ROOK | side, QUEEN | side) if d % 2 == 0 else (BISHOP | side, QUEEN | side)
            for t in rays[d]:
                p = squares[t]
                if p:
                    if p in sliders:
                        out.append(t)
                    break
        return sorted(out)

    def checkers(self) -> list[Square]:
        """Squares of the pieces that give check to the side to move."""
        if self._checkers is None:
            self._generate()
        assert self._checkers is not None
        return list(self._checkers)

    def is_check(self) -> bool:
        if self._checkers is None:
            self._generate()
        return bool(self._checkers)

    def was_into_check(self) -> bool:
        """Whether the side not to move is in check (an illegal position)."""
        king = self.kings[not self.turn]
        return king >= 0 and self._attacked(king, self.turn, self.squares)

    # -- legal moves ---------------------------------------------------------------

    def _invalidate(self) -> None:
        self._legal = None
        self._checkers = None

    @property
    def legal_moves(self) -> list[Move]:
        """Every legal move, as `Move` objects."""
        return [_to_move(m) for m in self.legal_move_codes()]

    def legal_move_codes(self) -> list[int]:
        """Every legal move as an int: `from | to << 6 | promotion << 12` (shared, read-only)."""
        if self._legal is None:
            self._generate()
        assert self._legal is not None
        return self._legal

    def is_legal(self, move: Move) -> bool:
        return _to_code(move) in self.legal_move_codes()

    def has_legal_en_passant(self) -> bool:
        """Whether an en passant capture is legal."""
        ep = self.ep_square
        if ep is None:
            return False
        squares = self.squares
        for m in self.legal_move_codes():
            fr = m & 63
            if (m >> 6) & 63 == ep and squares[fr] & 7 == PAWN and fr & 7 != ep & 7:
                return not squares[ep]  # a diagonal pawn move to the empty en passant square
        return False

    def _generate(self) -> None:
        """Compute the legal moves and the checkers of the position (cached until changed)."""
        squares = self.squares
        us = self.turn
        side = 0 if us else _BLACK_BIT
        enemy = _BLACK_BIT if us else 0
        moves: list[int] = []
        add = moves.append
        king = self.kings[us]
        if king < 0:  # no king (invalid positions only): every pseudo-legal move
            self._checkers = []
            self._pseudo(moves, squares, us, side, enemy, -1, {}, None)
            self._legal = moves
            return

        # Checkers and pins, from the king outwards.
        checkers: list[Square] = []
        block: set[Square] = set()  # squares that block (or capture) a single checker
        pins: dict[Square, int] = {}  # pinned piece -> direction of its pin
        e_rook, e_bishop, e_queen = ROOK | enemy, BISHOP | enemy, QUEEN | enemy
        rays = RAYS[king]
        for d in range(8):
            straight = d % 2 == 0
            own = -1
            for i, t in enumerate(rays[d]):
                p = squares[t]
                if not p:
                    continue
                if (p & _BLACK_BIT) == side:
                    if own >= 0:
                        break
                    own = t
                    continue
                if p == e_queen or p == (e_rook if straight else e_bishop):
                    if own >= 0:
                        pins[own] = d
                    else:
                        checkers.append(t)
                        block.update(rays[d][: i + 1])
                break
        e_knight, e_pawn = KNIGHT | enemy, PAWN | enemy
        for t in KNIGHT_TARGETS[king]:
            if squares[t] == e_knight:
                checkers.append(t)
                block.add(t)
        for t in PAWN_ATTACKS[us][king]:
            if squares[t] == e_pawn:
                checkers.append(t)
                block.add(t)
        self._checkers = checkers

        # King moves: never onto an attacked square (tested with the king lifted off).
        squares[king] = 0
        attacked = Board._attacked
        for t in KING_TARGETS[king]:
            p = squares[t]
            if (not p or (p & _BLACK_BIT) == enemy) and not attacked(t, not us, squares):
                add(king | t << 6)
        squares[king] = KING | side
        if len(checkers) > 1:
            self._legal = moves
            return
        target = block if checkers else None
        self._pseudo(moves, squares, us, side, enemy, king, pins, target)
        if not checkers:
            self._castling(moves, squares, us, side, king)
        self._legal = moves

    def _pseudo(
        self,
        moves: list[int],
        squares: bytearray,
        us: Color,
        side: int,
        enemy: int,
        king: Square,
        pins: dict[Square, int],
        target: set[Square] | None,
    ) -> None:
        """Moves of every piece but the king, respecting pins and (if `target`) a check."""
        add = moves.append
        direction = DIRECTION[king] if king >= 0 else None
        ep = self.ep_square
        forward, start_rank, last_rank = (8, 1, 7) if us else (-8, 6, 0)
        for sq in range(64):
            p = squares[sq]
            if not p or (p & _BLACK_BIT) != side:
                continue
            kind = p & 7
            if kind == KING:
                if king < 0:
                    for t in KING_TARGETS[sq]:
                        q = squares[t]
                        if not q or (q & _BLACK_BIT) == enemy:
                            add(sq | t << 6)
                continue
            pin = pins.get(sq, -1)
            if kind == PAWN:
                one = sq + forward
                if 0 <= one < 64 and not squares[one]:
                    self._pawn_to(moves, sq, one, pin, direction, target, last_rank)
                    two = one + forward
                    if (
                        sq >> 3 == start_rank
                        and not squares[two]
                        and (target is None or two in target)
                        and (pin < 0 or direction[two] == pin)  # type: ignore[index]
                    ):
                        add(sq | two << 6)
                for t in PAWN_ATTACKS[us][sq]:
                    q = squares[t]
                    if q and (q & _BLACK_BIT) == enemy:
                        self._pawn_to(moves, sq, t, pin, direction, target, last_rank)
                    elif t == ep and not q and self._ep_legal(sq, t, us, king):
                        add(sq | t << 6)
                continue
            if kind == KNIGHT:
                if pin >= 0:
                    continue
                for t in KNIGHT_TARGETS[sq]:
                    q = squares[t]
                    if (not q or (q & _BLACK_BIT) == enemy) and (target is None or t in target):
                        add(sq | t << 6)
                continue
            dirs = _ORTHOGONAL if kind == ROOK else _DIAGONAL if kind == BISHOP else range(8)
            rays = RAYS[sq]
            for d in dirs:
                if pin >= 0 and d != pin and d != (pin + 4) % 8:
                    continue
                for t in rays[d]:
                    q = squares[t]
                    if q:
                        if (q & _BLACK_BIT) == enemy and (target is None or t in target):
                            add(sq | t << 6)
                        break
                    if target is None or t in target:
                        add(sq | t << 6)

    @staticmethod
    def _pawn_to(
        moves: list[int],
        sq: Square,
        t: Square,
        pin: int,
        direction: tuple[int, ...] | None,
        target: set[Square] | None,
        last_rank: int,
    ) -> None:
        if target is not None and t not in target:
            return
        if pin >= 0 and direction is not None and direction[t] != pin:
            return
        base = sq | t << 6
        if t >> 3 == last_rank:
            moves.append(base | QUEEN << 12)
            moves.append(base | ROOK << 12)
            moves.append(base | BISHOP << 12)
            moves.append(base | KNIGHT << 12)
        else:
            moves.append(base)

    def _ep_legal(self, sq: Square, t: Square, us: Color, king: Square) -> bool:
        """Whether the en passant capture `sq -> t` leaves the king safe (tried on the board)."""
        if king < 0:
            return True
        squares = self.squares
        captured = t - 8 if us else t + 8
        pawn, victim = squares[sq], squares[captured]
        squares[t], squares[sq], squares[captured] = pawn, 0, 0
        safe = not Board._attacked(king, not us, squares)
        squares[sq], squares[t], squares[captured] = pawn, 0, victim
        return safe

    def _castling(
        self, moves: list[int], squares: bytearray, us: Color, side: int, king: Square
    ) -> None:
        rights = self.clean_castling_rights()
        home = 4 if us else 60
        if king != home:
            return
        attacked = Board._attacked
        them = not us
        for bit, rook_sq, king_to, rook_to in (
            ((_WK, 7, 6, 5) if us else (_BK, 63, 62, 61)),
            ((_WQ, 0, 2, 3) if us else (_BQ, 56, 58, 59)),
        ):
            if not rights & bit or squares[rook_sq] != ROOK | side:
                continue
            path = set(_between(home, king_to)) | set(_between(rook_sq, rook_to))
            path |= {king_to, rook_to}
            path -= {home, rook_sq}
            if any(squares[s] for s in path):
                continue
            squares[home] = 0
            unsafe = any(attacked(s, them, squares) for s in (home, *_between(home, king_to)))
            if not unsafe:
                squares[rook_sq], squares[rook_to] = 0, ROOK | side
                unsafe = attacked(king_to, them, squares)
                squares[rook_sq], squares[rook_to] = ROOK | side, 0
            squares[home] = KING | side
            if not unsafe:
                moves.append(home | king_to << 6)

    # -- making moves ----------------------------------------------------------------

    def push(self, move: Move) -> None:
        """Play `move` (it must be legal, or the null move)."""
        self.push_code(_to_code(move) if move else -1)
        self.move_stack.append(move)

    def push_code(self, m: int) -> None:
        """Play a legal move given as a move code (see `legal_move_codes`); -1 is a null move.

        Unlike `push`, this does not append to `move_stack` (`pop` still works).
        """
        squares = self.squares
        us = self.turn
        side = 0 if us else _BLACK_BIT
        undo_ep = self.ep_square
        undo_castling = self.castling
        undo_half = self.halfmove_clock
        self.ep_square = None
        self.halfmove_clock += 1
        if not us:
            self.fullmove_number += 1
        if m < 0:
            self._undo.append((m, 0, 0, undo_ep, undo_castling, undo_half, 0, 0))
            self.turn = not us
            self._invalidate()
            return
        fr, to, promo = m & 63, (m >> 6) & 63, m >> 12
        piece = squares[fr]
        kind = piece & 7
        captured = squares[to]
        capture_sq = to
        flags = 0
        if kind == PAWN:
            self.halfmove_clock = 0
            diff = to - fr
            if diff == 16 or diff == -16:
                self.ep_square = fr + diff // 2
            elif (
                to == undo_ep
                and not captured
                and (diff == 7 or diff == 9 or diff == -7 or diff == -9)
            ):
                capture_sq = to - 8 if us else to + 8
                captured = squares[capture_sq]
                squares[capture_sq] = 0
                flags = 1  # en passant
        if captured:
            self.halfmove_clock = 0
        castling = self.castling & _LOSE_RIGHTS[fr] & _LOSE_RIGHTS[to]
        squares[fr] = 0
        if kind == KING:
            castling &= _LOSE_RIGHTS_KING[us]
            self.kings[us] = to
            if to - fr == 2:  # castle short
                squares[fr + 3], squares[fr + 1] = 0, ROOK | side
                flags = 2
            elif fr - to == 2:  # castle long
                squares[fr - 4], squares[fr - 1] = 0, ROOK | side
                flags = 3
        squares[to] = (promo | side) if promo else piece
        self.castling = castling
        self._undo.append(
            (m, captured, capture_sq, undo_ep, undo_castling, undo_half, flags, piece)
        )
        self.turn = not us
        self._invalidate()

    def pop(self) -> Move:
        """Take back the last move; returns it."""
        m, captured, capture_sq, ep, castling, half, flags, piece = self._undo.pop()
        move = self.move_stack.pop() if self.move_stack else _to_move(m)
        self._pop_code(m, captured, capture_sq, ep, castling, half, flags, piece)
        return move

    def pop_code(self) -> int:
        """Take back the last move played with `push_code`; returns its code."""
        m, captured, capture_sq, ep, castling, half, flags, piece = self._undo.pop()
        self._pop_code(m, captured, capture_sq, ep, castling, half, flags, piece)
        return m

    def _pop_code(
        self,
        m: int,
        captured: int,
        capture_sq: int,
        ep: Square | None,
        castling: int,
        half: int,
        flags: int,
        piece: int,
    ) -> None:
        self.turn = us = not self.turn
        if not us:
            self.fullmove_number -= 1
        self.ep_square = ep
        self.castling = castling
        self.halfmove_clock = half
        self._invalidate()
        if m < 0:
            return
        squares = self.squares
        fr, to = m & 63, (m >> 6) & 63
        side = 0 if us else _BLACK_BIT
        squares[fr] = piece
        squares[to] = 0
        if captured:
            squares[capture_sq] = captured
        if piece & 7 == KING:
            self.kings[us] = fr
            if flags == 2:
                squares[fr + 1], squares[fr + 3] = 0, ROOK | side
            elif flags == 3:
                squares[fr - 1], squares[fr - 4] = 0, ROOK | side

    def peek(self) -> Move:
        """The last move played (raises `IndexError` if there is none)."""
        return self.move_stack[-1]

    # -- FEN -----------------------------------------------------------------------------

    def board_fen(self) -> str:
        """The piece placement part of the FEN."""
        squares = bytes(self.squares)
        cache = _FEN_ROWS
        rows = []
        for start in (56, 48, 40, 32, 24, 16, 8, 0):
            cells = squares[start : start + 8]
            row = cache.get(cells)
            if row is None:
                row = _fen_row(cells)
                if len(cache) < 65536:  # bounded: rows that recur are few
                    cache[cells] = row
            rows.append(row)
        return "/".join(rows)

    def castling_fen(self) -> str:
        rights = self.clean_castling_rights()
        text = "".join(
            ch for bit, ch in ((_WK, "K"), (_WQ, "Q"), (_BK, "k"), (_BQ, "q")) if rights & bit
        )
        return text or "-"

    def fen(self) -> str:
        """The position as FEN (the en passant square only when a capture there is legal)."""
        ep = SQUARE_NAMES[self.ep_square] if self.has_legal_en_passant() else "-"  # type: ignore[index]
        turn = "w" if self.turn else "b"
        return (
            f"{self.board_fen()} {turn} {self.castling_fen()} {ep} "
            f"{self.halfmove_clock} {self.fullmove_number}"
        )

    # -- validity ----------------------------------------------------------------------

    def status_errors(self) -> list[str]:
        """Why the position is not a legal one (empty when it is), as python-chess's flags."""
        squares = self.squares
        errors: list[str] = []
        codes = [c for c in squares if c]
        if not codes:
            errors.append("EMPTY")
        white_kings = sum(1 for c in codes if c == KING)
        black_kings = sum(1 for c in codes if c == KING | _BLACK_BIT)
        if not white_kings:
            errors.append("NO_WHITE_KING")
        if not black_kings:
            errors.append("NO_BLACK_KING")
        if white_kings + black_kings > 2:
            errors.append("TOO_MANY_KINGS")
        if sum(1 for c in codes if not c & _BLACK_BIT) > 16:
            errors.append("TOO_MANY_WHITE_PIECES")
        if sum(1 for c in codes if c & _BLACK_BIT) > 16:
            errors.append("TOO_MANY_BLACK_PIECES")
        if sum(1 for c in codes if c == PAWN) > 8:
            errors.append("TOO_MANY_WHITE_PAWNS")
        if sum(1 for c in codes if c == PAWN | _BLACK_BIT) > 8:
            errors.append("TOO_MANY_BLACK_PAWNS")
        if any(squares[s] & 7 == PAWN for s in (*range(8), *range(56, 64))):
            errors.append("PAWNS_ON_BACKRANK")
        if self.castling != self._clean_fen_castling():
            errors.append("BAD_CASTLING_RIGHTS")
        valid_ep = self._valid_ep_square()
        if self.ep_square != valid_ep:
            errors.append("INVALID_EP_SQUARE")
        if self.was_into_check():
            errors.append("OPPOSITE_CHECK")
        king = self.kings[self.turn] if white_kings + black_kings <= 2 else -1
        if king >= 0:
            checkers = self.attackers(not self.turn, king)
            if checkers:
                if len(checkers) > 2:
                    errors.append("TOO_MANY_CHECKERS")
                if valid_ep is not None:
                    pushed_to = valid_ep ^ 8
                    pushed_from = valid_ep ^ 24
                    before = bytearray(squares)
                    before[pushed_from], before[pushed_to] = squares[pushed_to], 0
                    if len(checkers) > 1 or (
                        max(checkers) != pushed_to and self._attacked_with(king, before)
                    ):
                        errors.append("IMPOSSIBLE_CHECK")
                elif len(checkers) > 2 or (
                    len(checkers) == 2 and self._on_line(min(checkers), max(checkers), king)
                ):
                    errors.append("IMPOSSIBLE_CHECK")
        return errors

    def _attacked_with(self, king: Square, occupancy: bytearray) -> bool:
        """python-chess's `_attacked_for_king`: attackers are the current pieces, the lines
        they attack along are blocked by `occupancy`."""
        squares = self.squares
        them = not self.turn
        side = 0 if them else _BLACK_BIT
        for t in KNIGHT_TARGETS[king]:
            if squares[t] == KNIGHT | side:
                return True
        for t in KING_TARGETS[king]:
            if squares[t] == KING | side:
                return True
        for t in PAWN_ATTACKS[not them][king]:
            if squares[t] == PAWN | side:
                return True
        for d in range(8):
            sliders = (ROOK | side, QUEEN | side) if d % 2 == 0 else (BISHOP | side, QUEEN | side)
            for t in RAYS[king][d]:
                if squares[t] in sliders:
                    return True
                if occupancy[t]:
                    break
        return False

    @staticmethod
    def _on_line(a: Square, b: Square, c: Square) -> bool:
        """Whether `c` lies on the full line through `a` and `b`."""
        d = DIRECTION[a][b]
        if d < 0:
            return False
        if c in (a, b):
            return True
        return DIRECTION[a][c] in (d, (d + 4) % 8) or DIRECTION[b][c] in (d, (d + 4) % 8)

    def _clean_fen_castling(self) -> int:
        return self.clean_castling_rights() if not self.castling >> 16 else -1

    def _valid_ep_square(self) -> Square | None:
        ep = self.ep_square
        if not ep:
            return None
        if self.turn:
            ep_rank, pawn_sq, seventh = 5, ep - 8, ep + 8
        else:
            ep_rank, pawn_sq, seventh = 2, ep + 8, ep - 8
        if ep >> 3 != ep_rank:
            return None
        if self.squares[pawn_sq] != (PAWN | (_BLACK_BIT if self.turn else 0)):
            return None
        if self.squares[ep]:
            return None
        if self.squares[seventh]:
            return None
        return ep

    def is_valid(self) -> bool:
        """Whether the position passes python-chess's validity checks (see `status_errors`)."""
        return not self.status_errors()

    # -- game end ------------------------------------------------------------------------

    def is_checkmate(self) -> bool:
        return self.is_check() and not self.legal_move_codes()

    def is_stalemate(self) -> bool:
        return not self.is_check() and not self.legal_move_codes()

    def has_insufficient_material(self, color: Color) -> bool:
        """Whether `color` cannot possibly mate (python-chess's rules, see the module docs)."""
        squares = self.squares
        side = 0 if color else _BLACK_BIT
        own = [c & 7 for c in squares if c and (c & _BLACK_BIT) == side]
        if any(k in (PAWN, ROOK, QUEEN) for k in own):
            return False
        if KNIGHT in own:
            theirs = [c & 7 for c in squares if c and (c & _BLACK_BIT) != side]
            return len(own) <= 2 and not any(k not in (KING, QUEEN) for k in theirs)
        if BISHOP in own:
            bishops = [sq for sq, c in enumerate(squares) if c & 7 == BISHOP]
            same_color = all(sq in _DARK_SQUARES for sq in bishops) or not any(
                sq in _DARK_SQUARES for sq in bishops
            )
            return (
                same_color
                and not any(c & 7 == PAWN for c in squares)
                and not any(c & 7 == KNIGHT for c in squares)
            )
        return True

    def is_insufficient_material(self) -> bool:
        return self.has_insufficient_material(WHITE) and self.has_insufficient_material(BLACK)

    # -- notation ----------------------------------------------------------------------

    def san(self, move: Move) -> str:
        """The legal `move` in standard algebraic notation (`"Nf3"`, `"exd5"`, `"Qh7#"`)."""
        san = self._san_without_suffix(move)
        self.push(move)
        try:
            if self.is_check():
                return san + ("#" if not self.legal_move_codes() else "+")
            return san
        finally:
            self.pop()

    def _san_without_suffix(self, move: Move) -> str:
        if not move:
            return "--"
        fr, to = move.from_square, move.to_square
        kind = self.piece_type_at(fr)
        if kind is None:
            raise IllegalMoveError(f"san() needs a legal move, got {move.uci()} in {self.fen()}")
        if kind == KING and abs(to - fr) == 2:
            return "O-O-O" if to < fr else "O-O"
        capture = bool(self.squares[to]) or (kind == PAWN and to == self.ep_square)
        san = "" if kind == PAWN else "pnbrqk"[kind - 1].upper()
        if kind != PAWN:
            others = [
                _to_move(m).from_square
                for m in self.legal_move_codes()
                if (m >> 6) & 63 == to and m & 63 != fr and self.squares[m & 63] & 7 == kind
            ]
            if others:
                row = column = False
                if any(o >> 3 == fr >> 3 for o in others):
                    column = True
                if any(o & 7 == fr & 7 for o in others):
                    row = True
                else:
                    column = True
                if column:
                    san += FILE_NAMES[fr & 7]
                if row:
                    san += RANK_NAMES[fr >> 3]
        elif capture:
            san += FILE_NAMES[fr & 7]
        if capture:
            san += "x"
        san += SQUARE_NAMES[to]
        if move.promotion:
            san += "=" + "pnbrqk"[move.promotion - 1].upper()
        return san

    def parse_san(self, san: str) -> Move:
        """The legal move written `san` in standard algebraic notation.

        Accepts what python-chess accepts (castling as `O-O`/`0-0`, check marks,
        overspecified and long algebraic moves). Raises `InvalidMoveError`,
        `IllegalMoveError` or `AmbiguousMoveError` (all `ValueError`).
        """
        if san in ("O-O", "O-O+", "O-O#", "0-0", "0-0+", "0-0#"):
            return self._castle(short=True, san=san)
        if san in ("O-O-O", "O-O-O+", "O-O-O#", "0-0-0", "0-0-0+", "0-0-0#"):
            return self._castle(short=False, san=san)
        match = SAN_REGEX.match(san)
        if not match:
            if san in ("--", "Z0", "0000", "@@@@"):
                return Move.null()
            if "," in san:
                raise InvalidMoveError(f"unsupported multi-leg move: {san!r}")
            raise InvalidMoveError(f"invalid san: {san!r}")
        to = SQUARE_NAMES.index(match.group(4))
        p = match.group(5)
        promotion = PIECE_SYMBOLS.index(p[-1].lower()) if p else None
        from_file = FILE_NAMES.index(match.group(2)) if match.group(2) else None
        from_rank = int(match.group(3)) - 1 if match.group(3) else None
        squares = self.squares
        if not match.group(1) and from_file is not None and from_rank is not None:
            # Fully specified moves (long algebraic, castling as e1g1), as python-chess.
            fr = square(from_file, from_rank)
            implied = promotion
            if implied is None and squares[fr] & 7 == PAWN and to >> 3 in (0, 7):
                implied = QUEEN
            move = Move(fr, to, implied)
            if not self.is_legal(move):
                raise IllegalMoveError(f"no matching legal move for {move.uci()} in {self.fen()}")
            if implied != promotion:
                raise IllegalMoveError(f"missing promotion piece type: {san!r} in {self.fen()}")
            return move
        kind = PIECE_SYMBOLS.index(match.group(1).lower()) if match.group(1) else PAWN
        if kind == PAWN and from_file is None:
            from_file = to & 7
        side = 0 if self.turn else _BLACK_BIT
        matched: Move | None = None
        for m in self.legal_move_codes():
            fr, t, promo = m & 63, (m >> 6) & 63, (m >> 12) or None
            if t != to or squares[fr] != kind | side or promo != promotion:
                continue
            if from_file is not None and fr & 7 != from_file:
                continue
            if from_rank is not None and fr >> 3 != from_rank:
                continue
            if matched is not None:
                raise AmbiguousMoveError(f"ambiguous san: {san!r} in {self.fen()}")
            matched = Move(fr, t, promo)
        if matched is None:
            raise IllegalMoveError(f"illegal san: {san!r} in {self.fen()}")
        return matched

    def _castle(self, *, short: bool, san: str) -> Move:
        home = 4 if self.turn else 60
        to = home + 2 if short else home - 2
        if self.squares[home] & 7 == KING and (home | to << 6) in self.legal_move_codes():
            return Move(home, to)
        raise IllegalMoveError(f"illegal san: {san!r} in {self.fen()}")

    def unicode(self, *, orientation: Color = WHITE, empty_square: str = "·") -> str:
        """The board as text with chess symbols, `orientation`'s side at the bottom."""
        symbols = "·♙♘♗♖♕♔··♟♞♝♜♛♚"
        lines = []
        ranks = range(7, -1, -1) if orientation else range(8)
        files = range(8) if orientation else range(7, -1, -1)
        for rank in ranks:
            cells = []
            for file in files:
                code = self.squares[rank * 8 + file]
                cells.append(symbols[code] if code else empty_square)
            lines.append(" ".join(cells))
        return "\n".join(lines)

    def __str__(self) -> str:
        lines = []
        for rank in range(7, -1, -1):
            cells = []
            for sq in range(rank * 8, rank * 8 + 8):
                code = self.squares[sq]
                letter = "pnbrqk"[(code & 7) - 1] if code else "."
                cells.append(letter if code & _BLACK_BIT or not code else letter.upper())
            lines.append(" ".join(cells))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Board({self.fen()!r})"

    def __iter__(self) -> Iterator[Square]:
        return (sq for sq in range(64) if self.squares[sq])


_FEN_ROWS: dict[bytes, str] = {}


def _fen_row(cells: bytes) -> str:
    """One rank of a FEN board part (`cells` are the eight piece codes, a-file first)."""
    row = ""
    empty = 0
    for code in cells:
        if not code:
            empty += 1
            continue
        if empty:
            row += str(empty)
            empty = 0
        letter = "pnbrqk"[(code & 7) - 1]
        row += letter if code & _BLACK_BIT else letter.upper()
    if empty:
        row += str(empty)
    return row


def _to_code(move: Move) -> int:
    return move.from_square | move.to_square << 6 | (move.promotion or 0) << 12


def _to_move(m: int) -> Move:
    return Move(m & 63, (m >> 6) & 63, (m >> 12) or None)


def move_to_code(move: Move) -> int:
    """The int code of a move (`from | to << 6 | promotion << 12`)."""
    return _to_code(move)


def code_to_move(code: int) -> Move:
    """The move of an int code."""
    return _to_move(code)


__all__ = [
    "AmbiguousMoveError",
    "BISHOP",
    "BLACK",
    "Board",
    "Color",
    "FILE_NAMES",
    "IllegalMoveError",
    "InvalidMoveError",
    "KING",
    "KNIGHT",
    "Move",
    "PAWN",
    "PIECE_NAMES",
    "PIECE_SYMBOLS",
    "PIECE_TYPES",
    "PieceType",
    "QUEEN",
    "RANK_NAMES",
    "ROOK",
    "SQUARE_NAMES",
    "STARTING_BOARD_FEN",
    "STARTING_FEN",
    "Square",
    "WHITE",
    "code_to_move",
    "move_to_code",
    "parse_square",
    "square",
    "square_file",
    "square_mirror",
    "square_name",
    "square_rank",
]
