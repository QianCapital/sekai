"""Incremental Go board: groups, liberties, captures, simple ko, Zobrist hashing and fast
eye-avoiding random playouts.

Points are the environment's action indices, `p = r * size + c` with `r = 0` the top row
and `c = 0` the left column; `PASS` is `size * size` (see `sekai.envs.games.go.actions`).
Colours are `EMPTY` `0`, `BLACK` `1` and `WHITE` `2`, so the opponent of `c` is `3 - c`.

Representation
--------------
Sets of points are Python ints used as bitsets (bit `p` set means point `p` is in the
set). The board keeps, per point, its colour and the *root* point of its group; per root, the
group's stones and liberties as bitsets. Placing a stone merges the smaller adjacent friendly
group into the larger one, removes the point from the adjacent enemy groups' liberties and
captures those left with none. Bitset liberties make the common queries O(1) ("more than
one liberty?" is `libs & (libs - 1)`, "in atari at p?" is `libs == 1 << p`) and make
`Board.copy()` a handful of shallow list copies, because ints are immutable.

The board knows the side to move and the **simple-ko** point, but has no game history: it
is the fast path used by tree search and playouts. Positional superko, move history and
observations live in `sekai.envs.games.go.GoEnv`, which uses `Board.hash_after()` and
`Board.colors_after()` to check repetitions.

Rules implemented here: captures of adjacent enemy groups left without liberties, then
suicide (self-capture, including multi-stone suicide) is illegal; a capturing move is never
suicide. Simple ko forbids immediately retaking a single stone that just captured a single
stone. Scoring is Tromp-Taylor area: stones plus empty points that reach only one colour.

Eyes
----
For playouts and simple policies, an *eye* of colour `c` is an empty point whose
orthogonal neighbours are all `c` stones and whose diagonal neighbours contain at most one
opponent stone (none if the point is on the edge or in a corner). Random playouts never
fill their own eyes, so living groups survive and playouts terminate.
"""

from __future__ import annotations

import functools
import random
from collections.abc import Iterable, Sequence
from itertools import compress

from sekai.envs.games.go.actions import COLUMNS, action_to_vertex, check_size
from sekai.envs.games.go.errors import IllegalMove

EMPTY = 0
BLACK = 1
WHITE = 2
NO_POINT = -1
COLOR_NAMES = ("empty", "black", "white")
STONE_CHARS = ".XO"
"""Board-string characters indexed by colour: `.` empty, `X` black, `O` white."""

# Hash keys only ever identify positions inside one process (superko checks), so their
# values are never part of an observation, a mask or an info dict.
_ZOBRIST_SEED = "sekai.go.zobrist.v1"


def bits_to_points(x: int) -> list[int]:
    """Points of a bitset, in increasing order."""
    out: list[int] = []
    while x:
        low = x & -x
        out.append(low.bit_length() - 1)
        x ^= low
    return out


def points_to_bits(points: Iterable[int]) -> int:
    """The bitset of some points."""
    x = 0
    for p in points:
        x |= 1 << p
    return x


class Geometry:
    """Immutable per-size lookup tables, shared by every board of that size.

    Attributes:
        size: The board size.
        n: Points on the board, `size * size` (also the pass point).
        pass_point: The pass point, `n`.
        nbrs: The orthogonal neighbours of every point.
        diags: The diagonal neighbours of every point.
        edge: 1 for points on the first line (edge or corner), else 0.
        line: Distance to the nearest edge plus one: 1 for the first line, 2 for the second, ...
        bit: `1 << p` for every point.
        nbr_bits: The neighbours of every point as a bitset.
        full: The bitset of every point.
        not_left: Every point but those of the left column.
        not_right: Every point but those of the right column.
        zobrist: Zobrist keys indexed `[colour][point]` (row 0, for EMPTY, is all zeros).
        near: Bitset of the points within Manhattan distance 3 of each point.
    """

    __slots__ = (
        "bit",
        "diags",
        "edge",
        "full",
        "line",
        "n",
        "nbr_bits",
        "nbrs",
        "near",
        "not_left",
        "not_right",
        "pass_point",
        "size",
        "zobrist",
    )

    def __init__(self, size: int) -> None:
        size = check_size(size)
        n = size * size
        self.size = size
        self.n = n
        self.pass_point = n
        nbrs: list[tuple[int, ...]] = []
        diags: list[tuple[int, ...]] = []
        edge: list[int] = []
        line: list[int] = []
        for p in range(n):
            r, c = divmod(p, size)
            nbrs.append(
                tuple(
                    rr * size + cc
                    for rr, cc in ((r - 1, c), (r, c - 1), (r, c + 1), (r + 1, c))
                    if 0 <= rr < size and 0 <= cc < size
                )
            )
            diags.append(
                tuple(
                    rr * size + cc
                    for rr, cc in ((r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1))
                    if 0 <= rr < size and 0 <= cc < size
                )
            )
            edge.append(1 if r in (0, size - 1) or c in (0, size - 1) else 0)
            line.append(1 + min(r, c, size - 1 - r, size - 1 - c))
        self.nbrs: tuple[tuple[int, ...], ...] = tuple(nbrs)
        self.diags: tuple[tuple[int, ...], ...] = tuple(diags)
        self.edge: tuple[int, ...] = tuple(edge)
        self.line: tuple[int, ...] = tuple(line)
        self.bit: tuple[int, ...] = tuple(1 << p for p in range(n))
        self.nbr_bits: tuple[int, ...] = tuple(points_to_bits(nb) for nb in nbrs)
        self.full = (1 << n) - 1
        self.not_left = points_to_bits(p for p in range(n) if p % size != 0)
        self.not_right = points_to_bits(p for p in range(n) if p % size != size - 1)
        rnd = random.Random(f"{_ZOBRIST_SEED}.{size}")
        self.zobrist: tuple[tuple[int, ...], ...] = (
            (0,) * n,
            tuple(rnd.getrandbits(64) for _ in range(n)),
            tuple(rnd.getrandbits(64) for _ in range(n)),
        )
        self.near: tuple[int, ...] = tuple(
            points_to_bits(
                q for q in range(n) if abs(q // size - p // size) + abs(q % size - p % size) <= 3
            )
            for p in range(n)
        )

    def adjacent(self, x: int) -> int:
        """Bitset of every point orthogonally adjacent to some point of `x`."""
        s = self.size
        return (
            ((x << 1) & self.not_left)
            | ((x >> 1) & self.not_right)
            | ((x << s) & self.full)
            | (x >> s)
        )

    def reach(self, seeds: int, through: int) -> int:
        """Points of `through` connected to `seeds` by paths inside `through`."""
        r = self.adjacent(seeds) & through
        while True:
            nr = (r | self.adjacent(r)) & through
            if nr == r:
                return r
            r = nr

    def area_bits(self, black: int, white: int) -> tuple[int, int]:
        """Tromp-Taylor areas `(black_area, white_area)` as bitsets.

        A point belongs to a colour if it holds that colour's stone, or is empty and every
        empty region path from it reaches stones of that colour only.
        """
        empty = self.full & ~(black | white)
        reach_b = self.reach(black, empty)
        reach_w = self.reach(white, empty)
        return black | (reach_b & ~reach_w), white | (reach_w & ~reach_b)

    def __repr__(self) -> str:
        return f"Geometry(size={self.size})"


@functools.cache
def geometry(size: int) -> Geometry:
    """The shared `Geometry` for `size` (built once per size)."""
    return Geometry(size)


# Fixed handicap placement (GTP "fixed_handicap"), as (column, row-from-bottom) offsets
# from the corners: "lo" = the 3-3 point on 9x9 and the 4-4 point otherwise, "mid" = centre.
_HANDICAP_PATTERN: dict[int, tuple[str, ...]] = {
    2: ("lo lo", "hi hi"),
    3: ("lo lo", "hi hi", "lo hi"),
    4: ("lo lo", "hi hi", "lo hi", "hi lo"),
    5: ("lo lo", "hi hi", "lo hi", "hi lo", "mid mid"),
    6: ("lo lo", "hi hi", "lo hi", "hi lo", "lo mid", "hi mid"),
    7: ("lo lo", "hi hi", "lo hi", "hi lo", "lo mid", "hi mid", "mid mid"),
    8: ("lo lo", "hi hi", "lo hi", "hi lo", "lo mid", "hi mid", "mid lo", "mid hi"),
    9: ("lo lo", "hi hi", "lo hi", "hi lo", "lo mid", "hi mid", "mid lo", "mid hi", "mid mid"),
}
HANDICAP_SIZES = (9, 13, 19)


def handicap_points(size: int, stones: int) -> list[int]:
    """Points of the standard fixed handicap placement (GTP `fixed_handicap`).

    On 19x19, 2 stones are D4 Q16, 3 add D16, 4 add Q4, 5 add the centre K10, 6 are the
    four corners plus D10 Q10, 7 add K10, 8 are the six plus K4 K16, 9 add K10. 13x13 and
    9x9 use the same pattern on their star points (4-4 and 3-3 points respectively).
    Only 9x9, 13x13 and 19x19 boards and 2..9 stones are supported.
    """
    if size not in HANDICAP_SIZES:
        raise ValueError(f"Fixed handicap is only defined for sizes {HANDICAP_SIZES}, got {size}.")
    if isinstance(stones, bool) or stones not in _HANDICAP_PATTERN:
        raise ValueError(f"Handicap must be 0 or an integer in [2, 9], got {stones!r}.")
    lo = 2 if size == 9 else 3
    coord = {"lo": lo, "mid": size // 2, "hi": size - 1 - lo}
    points = []
    for spec in _HANDICAP_PATTERN[stones]:
        col_key, row_key = spec.split()
        col, row_from_bottom = coord[col_key], coord[row_key]
        points.append((size - 1 - row_from_bottom) * size + col)
    return points


class Board:
    """A Go position with incremental group/liberty tracking (no game history).

    Attributes are public for speed-critical callers but should be treated as read-only;
    mutate only through `play()`, `play_unchecked()` and `pass_move()`.

    Args:
        size: The board size, `2..19`.
        to_play: The side to move, `BLACK` (1) or `WHITE` (2).

    Attributes:
        geo: The board's `Geometry`.
        color: Colour of every point.
        gid: Root point of the group containing each stone (meaningless for empty points).
        gstones: Stones bitset of the group rooted at each root point (0 for non-roots).
        glibs: Liberties bitset of the group rooted at each root point (0 for non-roots).
        empties: Empty points, in no particular order.
        eidx: Index of each empty point in `empties`.
        stones: Stone bitsets indexed by colour (index 0 unused).
        hash: Zobrist hash of the stone configuration (side to move and ko excluded).
        ko: Point the side to move may not play because of simple ko, or `NO_POINT`.
        to_play: The side to move.
    """

    __slots__ = (
        "color",
        "eidx",
        "empties",
        "geo",
        "gid",
        "glibs",
        "gstones",
        "hash",
        "ko",
        "stones",
        "to_play",
    )

    def __init__(self, size: int = 9, to_play: int = BLACK) -> None:
        geo = geometry(size)
        n = geo.n
        if to_play not in (BLACK, WHITE):
            raise ValueError(f"to_play must be BLACK (1) or WHITE (2), got {to_play!r}.")
        self.geo = geo
        self.color: list[int] = [EMPTY] * n
        self.gid: list[int] = [0] * n
        self.gstones: list[int] = [0] * n
        self.glibs: list[int] = [0] * n
        self.empties: list[int] = list(range(n))
        self.eidx: list[int] = list(range(n))
        self.stones: list[int] = [0, 0, 0]
        self.hash = 0
        self.ko = NO_POINT
        self.to_play = to_play

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_colors(cls, colors: Sequence[int], size: int, to_play: int = BLACK) -> Board:
        """Board with the given colour per point (row-major from the top-left).

        Raises `ValueError` if a group has no liberties.
        """
        board = cls(size, to_play)
        geo = board.geo
        n = geo.n
        if len(colors) != n:
            raise ValueError(f"Expected {n} colours for a {size}x{size} board, got {len(colors)}.")
        color = [int(c) for c in colors]
        if any(c not in (EMPTY, BLACK, WHITE) for c in color):
            raise ValueError("Colours must be EMPTY (0), BLACK (1) or WHITE (2).")
        board.color = color
        board.empties = [p for p in range(n) if color[p] == EMPTY]
        for i, p in enumerate(board.empties):
            board.eidx[p] = i
        seen = [False] * n
        for p in range(n):
            c = color[p]
            if c == EMPTY or seen[p]:
                continue
            stones = libs = 0
            stack = [p]
            seen[p] = True
            while stack:
                s = stack.pop()
                stones |= geo.bit[s]
                board.gid[s] = p
                for q in geo.nbrs[s]:
                    cq = color[q]
                    if cq == EMPTY:
                        libs |= geo.bit[q]
                    elif cq == c and not seen[q]:
                        seen[q] = True
                        stack.append(q)
            if not libs:
                vertex = action_to_vertex(p, size)
                raise ValueError(f"The {COLOR_NAMES[c]} group at {vertex} has no liberties.")
            board.gstones[p] = stones
            board.glibs[p] = libs
            board.stones[c] |= stones
            board.hash ^= _xor_keys(geo.zobrist[c], stones)
        return board

    @classmethod
    def from_string(cls, text: str, to_play: int = BLACK) -> Board:
        """Board from a board string (`.` empty, `X` black, `O` white; whitespace ignored).

        The size is inferred from the number of points, which must be a perfect square.
        """
        chars = "".join(text.split())
        size = round(len(chars) ** 0.5)
        if size * size != len(chars):
            raise ValueError(f"A board string needs size*size characters, got {len(chars)}.")
        lookup = {".": EMPTY, "+": EMPTY, "X": BLACK, "x": BLACK, "O": WHITE, "o": WHITE}
        try:
            colors = [lookup[ch] for ch in chars]
        except KeyError as exc:
            raise ValueError(f"Unexpected character {exc.args[0]!r} in board string.") from None
        return cls.from_colors(colors, size, to_play)

    def copy(self) -> Board:
        """Independent copy (cheap: shallow list copies; geometry tables are shared)."""
        new = Board.__new__(Board)
        new.geo = self.geo
        new.color = self.color[:]
        new.gid = self.gid[:]
        new.gstones = self.gstones[:]
        new.glibs = self.glibs[:]
        new.empties = self.empties[:]
        new.eidx = self.eidx[:]
        new.stones = self.stones[:]
        new.hash = self.hash
        new.ko = self.ko
        new.to_play = self.to_play
        return new

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """The board size."""
        return self.geo.size

    @property
    def n(self) -> int:
        """Number of points, `size * size` (also the pass point)."""
        return self.geo.n

    def stone_count(self) -> int:
        """Stones on the board."""
        return self.geo.n - len(self.empties)

    def group_root(self, p: int) -> int:
        """Root point of the group at `p` (`p` must hold a stone)."""
        return self.gid[p]

    def group_stones(self, p: int) -> list[int]:
        """Stones of the group containing `p` (empty list for an empty point)."""
        if self.color[p] == EMPTY:
            return []
        return bits_to_points(self.gstones[self.gid[p]])

    def liberties(self, p: int) -> list[int]:
        """Liberties of the group containing `p` (empty list for an empty point)."""
        if self.color[p] == EMPTY:
            return []
        return bits_to_points(self.glibs[self.gid[p]])

    def liberty_count(self, p: int) -> int:
        """Liberties of the group containing `p` (0 for an empty point)."""
        if self.color[p] == EMPTY:
            return 0
        return self.glibs[self.gid[p]].bit_count()

    def groups(self, color: int | None = None) -> list[tuple[int, int, int, int]]:
        """`(root, colour, stones, liberties)` bitsets of every group (optionally one colour)."""
        out = []
        gid, col = self.gid, self.color
        for p in range(self.geo.n):
            c = col[p]
            if c != EMPTY and gid[p] == p and (color is None or c == color):
                out.append((p, c, self.gstones[p], self.glibs[p]))
        return out

    def is_eye(self, p: int, color: int | None = None) -> bool:
        """Whether empty point `p` is an eye of `color` (default: the side to move)."""
        c = self.to_play if color is None else color
        col = self.color
        if col[p] != EMPTY:
            return False
        for q in self.geo.nbrs[p]:
            if col[q] != c:
                return False
        opp = 3 - c
        bad = self.geo.edge[p]
        for d in self.geo.diags[p]:
            if col[d] == opp:
                bad += 1
        return bad < 2

    def is_legal(self, p: int, *, ignore_ko: bool = False) -> bool:
        """Legality of `p` for the side to move under simple ko (pass is always legal)."""
        geo = self.geo
        if p == geo.n:
            return True
        if not 0 <= p < geo.n:
            return False
        color = self.color
        if color[p] != EMPTY or (p == self.ko and not ignore_ko):
            return False
        c = self.to_play
        bp = geo.bit[p]
        glibs, gid = self.glibs, self.gid
        for q in geo.nbrs[p]:
            cq = color[q]
            if cq == EMPTY:
                return True
            libs = glibs[gid[q]]
            if cq == c:
                if libs & (libs - 1):
                    return True
            elif libs == bp:
                return True
        return False

    def illegal_reason(self, p: int, *, ignore_ko: bool = False) -> str | None:
        """`None` if `p` is legal for the side to move (simple ko), else a reason code."""
        n = self.geo.n
        if p == n:
            return None
        if not 0 <= p < n:
            return "off-board"
        if self.color[p] != EMPTY:
            return "occupied"
        if p == self.ko and not ignore_ko:
            return "ko"
        if not self.is_legal(p, ignore_ko=True):
            return "suicide"
        return None

    def legal_points(self, *, ignore_ko: bool = False) -> list[int]:
        """Every legal non-pass point for the side to move under simple ko, ascending."""
        geo = self.geo
        color, glibs, gid, nbrs, bit = self.color, self.glibs, self.gid, geo.nbrs, geo.bit
        c = self.to_play
        ko = NO_POINT if ignore_ko else self.ko
        out = []
        for p in self.empties:
            if p == ko:
                continue
            bp = bit[p]
            for q in nbrs[p]:
                cq = color[q]
                if cq == EMPTY:
                    out.append(p)
                    break
                libs = glibs[gid[q]]
                if cq == c:
                    if libs & (libs - 1):
                        out.append(p)
                        break
                elif libs == bp:
                    out.append(p)
                    break
        out.sort()
        return out

    def legal_bits(self, *, ignore_ko: bool = False) -> tuple[int, int]:
        """`legal_points()` as a bitset, and the bitset of those of them that capture.

        Computed on whole bitsets: an empty point with an empty neighbour is legal, and one
        without is legal if it touches a group of its own with another liberty or is the
        last liberty of an enemy group (a capture).
        """
        geo = self.geo
        stones = self.stones
        empty = geo.full ^ (stones[BLACK] | stones[WHITE])
        c = self.to_play
        color, glibs, gstones = self.color, self.glibs, self.gstones
        safe = atari = 0  # own groups with a spare liberty; enemy groups' last liberties
        for g in compress(range(geo.n), glibs):  # the groups' roots
            libs = glibs[g]
            if color[g] == c:
                if libs & (libs - 1):
                    safe |= gstones[g]
            elif not libs & (libs - 1):
                atari |= libs
        free = empty & geo.adjacent(empty)
        legal = free | (empty & ~free & (geo.adjacent(safe) | atari))
        if not ignore_ko and self.ko != NO_POINT:
            legal &= ~geo.bit[self.ko]
        return legal, legal & atari

    def captured_by(self, p: int) -> int:
        """Bitset of the enemy stones that playing `p` would capture (side to move)."""
        geo = self.geo
        color, glibs, gid, gstones = self.color, self.glibs, self.gid, self.gstones
        opp = 3 - self.to_play
        bp = geo.bit[p]
        captured = 0
        for q in geo.nbrs[p]:
            if color[q] == opp:
                g = gid[q]
                if glibs[g] == bp:
                    captured |= gstones[g]
        return captured

    def group_after(self, p: int) -> tuple[int, int, int]:
        """`(stones, liberties, captured)` bitsets of the group formed by playing `p`.

        Computed without modifying the board, for the side to move; `p` must be empty.
        """
        geo = self.geo
        color, glibs, gid, gstones, bit = self.color, self.glibs, self.gid, self.gstones, geo.bit
        c = self.to_play
        bp = bit[p]
        stones = bp
        libs = captured = 0
        for q in geo.nbrs[p]:
            cq = color[q]
            if cq == EMPTY:
                libs |= bit[q]
            elif cq == c:
                g = gid[q]
                stones |= gstones[g]
                libs |= glibs[g]
            else:
                g = gid[q]
                if glibs[g] == bp:
                    captured |= gstones[g]
        libs &= ~bp
        if captured:
            libs |= geo.adjacent(stones) & captured
        return stones, libs, captured

    def is_self_atari(self, p: int) -> bool:
        """Whether playing empty point `p` leaves a group of 2+ stones with one liberty.

        Single-stone self-ataris (throw-ins, killing moves inside eyes) do not count.
        """
        color = self.color
        empty = 0
        for q in self.geo.nbrs[p]:
            if not color[q]:
                empty += 1
        if empty >= 2:
            return False
        stones, libs, _captured = self.group_after(p)
        return bool(libs and not libs & (libs - 1) and stones & (stones - 1))

    def hash_after(self, p: int) -> int:
        """Zobrist hash of the stone configuration after the side to move plays `p`."""
        h = self.hash
        if p == self.geo.n:
            return h
        geo = self.geo
        h ^= geo.zobrist[self.to_play][p]
        captured = self.captured_by(p)
        if captured:
            h ^= _xor_keys(geo.zobrist[3 - self.to_play], captured)
        return h

    def colors_after(self, p: int) -> bytes:
        """Colour per point (as `bytes`) after the side to move plays legal point `p`."""
        col = bytearray(self.color)
        if p == self.geo.n:
            return bytes(col)
        col[p] = self.to_play
        for s in bits_to_points(self.captured_by(p)):
            col[s] = EMPTY
        return bytes(col)

    def config(self) -> bytes:
        """Colour per point as `bytes` (the exact position identity used for superko)."""
        return bytes(self.color)

    def area_score(self) -> tuple[int, int]:
        """Tromp-Taylor `(black_points, white_points)`, without komi."""
        b, w = self.geo.area_bits(self.stones[BLACK], self.stones[WHITE])
        return b.bit_count(), w.bit_count()

    def area_bits(self) -> tuple[int, int]:
        """Tromp-Taylor areas `(black, white)` as bitsets."""
        return self.geo.area_bits(self.stones[BLACK], self.stones[WHITE])

    def to_string(self) -> str:
        """`size*size` characters row-major from the top-left: `.` empty, `X`, `O`."""
        return "".join([STONE_CHARS[c] for c in self.color])

    def pretty(self) -> str:
        """Multi-line diagram with GTP coordinates, for debugging and logs."""
        size = self.geo.size
        letters = " ".join(COLUMNS[:size])
        lines = [f"   {letters}"]
        for r in range(size):
            row = " ".join(STONE_CHARS[c] for c in self.color[r * size : (r + 1) * size])
            lines.append(f"{size - r:2d} {row} {size - r:2d}")
        lines.append(f"   {letters}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Moves
    # ------------------------------------------------------------------

    def pass_move(self) -> None:
        """Pass: clears the ko point and hands the move to the opponent."""
        self.ko = NO_POINT
        self.to_play = 3 - self.to_play

    def play(self, p: int) -> int:
        """Play `p` (or pass) for the side to move, checking legality under simple ko.

        Returns the number of stones captured. Raises `IllegalMove` otherwise.
        """
        if p == self.geo.n:
            self.pass_move()
            return 0
        reason = self.illegal_reason(p)
        if reason is not None:
            vertex = action_to_vertex(p, self.geo.size) if 0 <= p < self.geo.n else str(p)
            raise IllegalMove(p, vertex, reason, self.to_string())
        return self.play_unchecked(p)

    def play_unchecked(self, p: int) -> int:
        """Place a stone at `p` for the side to move without any legality check.

        The fast path for search: the caller guarantees `p` is empty, not the ko point and
        not suicide (e.g. via `is_legal()`). Returns the number of stones captured.
        """
        geo = self.geo
        color = self.color
        gid = self.gid
        gstones = self.gstones
        glibs = self.glibs
        empties = self.empties
        eidx = self.eidx
        nbrs = geo.nbrs
        bit = geo.bit
        c = self.to_play
        opp = 3 - c
        bp = bit[p]
        nb = geo.full ^ bp

        k = eidx[p]
        last = empties.pop()
        if last != p:
            empties[k] = last
            eidx[last] = k
        color[p] = c
        gid[p] = p
        gstones[p] = bp
        glibs[p] = 0
        h = self.hash ^ geo.zobrist[c][p]
        root = p
        libs = 0
        captured = 0
        for q in nbrs[p]:
            cq = color[q]
            if cq == 0:
                libs |= bit[q]
            elif cq == c:
                g = gid[q]
                if g != root:
                    sr = gstones[root]
                    sg = gstones[g]
                    if sg.bit_count() >= sr.bit_count():
                        small, big = root, g
                    else:
                        small, big = g, root
                    x = gstones[small]
                    while x:
                        low = x & -x
                        gid[low.bit_length() - 1] = big
                        x ^= low
                    gstones[big] = sr | sg
                    glibs[big] |= glibs[small]
                    gstones[small] = 0
                    glibs[small] = 0
                    root = big
            else:
                g = gid[q]
                gl = glibs[g] & nb
                glibs[g] = gl
                if not gl:
                    x = gstones[g]
                    captured |= x
                    gstones[g] = 0
                    zo = geo.zobrist[opp]
                    while x:
                        low = x & -x
                        s = low.bit_length() - 1
                        x ^= low
                        color[s] = 0
                        eidx[s] = len(empties)
                        empties.append(s)
                        h ^= zo[s]
                        for t in nbrs[s]:
                            if color[t] == c:
                                glibs[gid[t]] |= low
        glibs[root] = (glibs[root] | libs) & nb
        stones = self.stones
        stones[c] |= bp
        self.hash = h
        self.to_play = opp
        if captured:
            stones[opp] ^= captured
            if root == p and not (captured & (captured - 1)) and glibs[p] == captured:
                self.ko = captured.bit_length() - 1
            else:
                self.ko = NO_POINT
            return captured.bit_count()
        self.ko = NO_POINT
        return 0

    # ------------------------------------------------------------------
    # Playouts
    # ------------------------------------------------------------------

    def random_playout(
        self,
        rnd: random.Random,
        *,
        max_moves: int | None = None,
        passes: int = 0,
        amaf: list[int] | None = None,
        avoid_self_atari: bool = False,
    ) -> int:
        """Play uniformly random legal non-eye-filling moves until two consecutive passes.

        Modifies the board in place; a side passes only when it has no such move. `passes`
        is the number of consecutive passes immediately before this position. Uses simple
        ko only. If `amaf` (a list with one slot per point, 0 = unplayed) is given, the
        colour that first plays each point is recorded in it, for RAVE. Returns the number
        of moves (including passes) played; stops early after `max_moves` (default
        `3 * size * size`).

        With `avoid_self_atari=True` moves that would leave a group of two or more stones
        in atari are skipped as well (see `is_self_atari()`). Ownership estimation uses
        this so that a seki is not destroyed just because the side to move has no other move
        left; the plain policy is the fast default used by tree search.
        """
        geo = self.geo
        color = self.color
        gid = self.gid
        glibs = self.glibs
        nbrs = geo.nbrs
        diags = geo.diags
        edge = geo.edge
        bit = geo.bit
        empties = self.empties
        play = self.play_unchecked
        self_atari = self.is_self_atari
        rand = rnd.random
        limit = 3 * geo.n if max_moves is None else max_moves
        moves = 0
        while moves < limit:
            c = self.to_play
            ko = self.ko
            cand = empties
            copied = False
            i = len(cand)
            chosen = -1
            while i:
                j = int(rand() * i)
                p = cand[j]
                if p != ko:
                    nbp = nbrs[p]
                    for q in nbp:
                        if not color[q]:
                            chosen = p
                            break
                    else:
                        ok = False
                        own = True
                        bp = bit[p]
                        for q in nbp:
                            libs = glibs[gid[q]]
                            if color[q] == c:
                                if libs & (libs - 1):
                                    ok = True
                            else:
                                own = False
                                if libs == bp:
                                    ok = True
                        if ok:
                            if own:
                                opp = 3 - c
                                bad = edge[p]
                                for d in diags[p]:
                                    if color[d] == opp:
                                        bad += 1
                                if bad >= 2:
                                    chosen = p
                            else:
                                chosen = p
                    if chosen >= 0:
                        if not avoid_self_atari or not self_atari(chosen):
                            break
                        chosen = -1
                i -= 1
                if not copied:
                    cand = cand[:]
                    copied = True
                cand[j] = cand[i]
            moves += 1
            if chosen < 0:
                passes += 1
                self.ko = NO_POINT
                self.to_play = 3 - c
                if passes >= 2:
                    break
            else:
                passes = 0
                if amaf is not None and not amaf[chosen]:
                    amaf[chosen] = c
                play(chosen)
        return moves

    def __repr__(self) -> str:
        return f"Board(size={self.geo.size}, to_play={COLOR_NAMES[self.to_play]})"


def _xor_keys(keys: tuple[int, ...], x: int) -> int:
    h = 0
    while x:
        low = x & -x
        h ^= keys[low.bit_length() - 1]
        x ^= low
    return h


__all__ = [
    "EMPTY",
    "BLACK",
    "WHITE",
    "NO_POINT",
    "COLOR_NAMES",
    "STONE_CHARS",
    "HANDICAP_SIZES",
    "Geometry",
    "geometry",
    "handicap_points",
    "bits_to_points",
    "points_to_bits",
    "Board",
]
