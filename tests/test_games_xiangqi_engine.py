"""Xiangqi rules engine: perft, piece movement, legality, FEN, and a brute-force cross-check."""

from __future__ import annotations

import numpy as np
import pytest

from sekai.envs.games.xiangqi.engine import (
    BLACK,
    RED,
    START_FEN,
    Position,
    mirror_fen,
    move_from_uci,
    move_to_uci,
    parse_square,
    square_name,
)
from sekai.envs.games.xiangqi.errors import IllegalMove, InvalidFen, InvalidMoveFormat


def legal_set(fen: str) -> set[str]:
    return {move_to_uci(m) for m in Position(fen).legal_moves()}


def moves_from(fen: str, sq: str) -> set[str]:
    return {m[2:] for m in legal_set(fen) if m[:2] == sq}


# ----------------------------------------------------------------------
# Perft
# ----------------------------------------------------------------------


@pytest.mark.parametrize(("depth", "nodes"), [(1, 44), (2, 1_920), (3, 79_666)])
def test_start_position_perft(depth: int, nodes: int) -> None:
    assert Position().perft(depth) == nodes


def test_start_position_perft_depth_4() -> None:  # 3.3 million leaves, under a second
    assert Position().perft(4) == 3_290_240


# Well-known Xiangqi perft test positions (chessprogramming.org / Fairy-Stockfish suites).
PERFT_SUITE = [
    ("r1ba1a3/4kn3/2n1b4/pNp1p1p1p/4c4/6P2/P1P2R2P/1CcC5/9/2BAKAB2 w - - 0 1", [38, 1128, 43929]),
    ("1cbak4/9/n2a5/2p1p3p/5cp2/2n2N3/6PCP/3AB4/2C6/3A1K1N1 w - - 0 1", [7, 281, 8620]),
    ("5a3/3k5/3aR4/9/5r3/5n3/9/3A1A3/5K3/2BC2B2 w - - 0 1", [25, 424, 9850]),
    ("CRN1k1b2/3ca4/4ba3/9/2nr5/9/9/4B4/4A4/4KA3 w - - 0 1", [28, 516, 14808]),
    ("R1N1k1b2/9/3aba3/9/2nr5/2B6/9/4B4/4A4/4KA3 w - - 0 1", [21, 364, 7626]),
    ("C1nNk4/9/9/9/9/9/n1pp5/B3C4/9/3A1K3 w - - 0 1", [28, 222, 6241]),
    ("4ka3/4a4/9/9/4N4/p8/9/4C3c/7n1/2BK5 w - - 0 1", [23, 345, 8124]),
    ("2b1ka3/9/b3N4/4n4/9/9/9/4C4/2p6/2BK5 w - - 0 1", [21, 195, 3883]),
]


@pytest.mark.parametrize(("fen", "expected"), PERFT_SUITE)
def test_perft_suite(fen: str, expected: list[int]) -> None:
    position = Position(fen)
    assert [position.perft(d) for d in (1, 2, 3)] == expected
    assert position.fen() == fen  # push/pop restored everything


def test_push_pop_restores_state() -> None:
    position = Position(PERFT_SUITE[0][0])
    before = (position.fen(), position.hash, position.score, position.checked)
    for move in position.legal_moves():
        position.push(move)
        position.pop()
        assert (position.fen(), position.hash, position.score, position.checked) == before


def test_incremental_hash_and_score_match_recomputation() -> None:
    rng = np.random.default_rng(0)
    position = Position()
    for _ in range(80):
        moves = position.legal_moves()
        if not moves:
            break
        position.push(int(rng.choice(moves)))
        fresh = Position(position.fen())
        assert position.hash == fresh.hash
        assert position.score == fresh.score
        assert position.checked == fresh.checked


def test_copy_is_independent() -> None:
    position = Position()
    position.push_uci("h2e2")
    copy = position.copy()
    copy.push_uci("h9g7")
    assert position.fen() != copy.fen() and position.move_stack == [move_from_uci("h2e2")]
    assert copy.pop() == move_from_uci("h9g7") and copy.fen() == position.fen()


# ----------------------------------------------------------------------
# Squares and moves
# ----------------------------------------------------------------------


def test_square_and_move_strings() -> None:
    assert parse_square("a0") == 0 and parse_square("i9") == 89 and parse_square("h2") == 25
    assert square_name(25) == "h2"
    assert move_to_uci(move_from_uci("h2e2")) == "h2e2"
    for bad in ("h2e", "j2e2", "h2e10", "", "h2-e2", 42):
        with pytest.raises(InvalidMoveFormat):
            move_from_uci(bad)  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Piece movement
# ----------------------------------------------------------------------


def test_cannon_moves_like_a_rook_but_captures_over_one_screen() -> None:
    # Red cannon e4. Up the file: red soldier e6 is a screen for the black horse e8.
    # To the left: black soldier b4 is a screen for the black chariot a4.
    fen = "3k5/4n4/9/4P4/9/rp2C4/9/9/9/5K3 w - - 0 1"
    targets = moves_from(fen, "e4")
    quiet = {"e5", "e3", "e2", "e1", "e0", "d4", "c4", "f4", "g4", "h4", "i4"}
    assert targets == quiet | {"e8", "a4"}
    assert "b4" not in targets  # the adjacent screen itself cannot be captured
    assert "e7" not in targets  # no quiet move past a screen


def test_cannon_cannot_capture_over_two_screens() -> None:
    # Cannon e1, red soldier e3 (screen), black soldier e4, black chariot e6.
    fen = "3k5/9/9/4r4/9/4p4/4P4/9/4C4/5K3 w - - 0 1"
    targets = moves_from(fen, "e1")
    assert {"e2", "e4"} <= targets  # e4 is the first piece beyond exactly one screen
    assert "e3" not in targets and "e6" not in targets  # e6 is behind two screens


def test_horse_leg_blocking() -> None:
    assert moves_from("4k4/9/9/9/9/9/9/9/9/1N1K5 w - - 0 1", "b0") == {"a2", "c2", "d1"}
    # A chariot on b1 blocks both upward jumps.
    assert moves_from("4k4/9/9/9/9/9/9/9/1R7/1N1K5 w - - 0 1", "b0") == {"d1"}
    # An elephant on c0 blocks the jump to d1.
    assert moves_from("3k5/9/9/9/9/9/9/9/9/1NB1K4 w - - 0 1", "b0") == {"a2", "c2"}


def test_horse_attack_needs_free_leg() -> None:
    # Black horse f2 attacks the red general on e0 through the free leg f1.
    assert Position("3k5/9/9/9/9/9/9/5n3/9/4K4 w - - 0 1").checked
    assert not Position("3k5/9/9/9/9/9/9/5n3/5R3/4K4 w - - 0 1").checked
    # A horse orthogonally or diagonally adjacent gives no check.
    assert not Position("3k5/9/9/9/9/9/9/9/5n3/4K4 w - - 0 1").checked
    assert not Position("3k5/9/9/9/9/9/9/9/9/4Kn3 w - - 0 1").checked


def test_elephant_eye_and_river() -> None:
    assert moves_from("4k4/9/9/9/9/2B6/9/9/9/3K5 w - - 0 1", "c4") == {"a2", "e2"}
    # A chariot on d3 blocks the eye of the c4-e2 move.
    assert moves_from("4k4/9/9/9/9/2B6/3R5/9/9/3K5 w - - 0 1", "c4") == {"a2"}
    # Black's elephant on c5 never crosses the river to rank 3.
    assert moves_from("3k5/9/9/9/2b6/9/9/9/9/4K4 b - - 0 1", "c5") == {"a7", "e7"}


def test_palace_limits_for_general_and_advisors() -> None:
    assert moves_from("5k3/9/9/9/9/9/9/9/9/3K5 w - - 0 1", "d0") == {"d1", "e0"}  # not c0
    assert moves_from("5k3/9/9/9/9/9/9/3K5/9/9 w - - 0 1", "d2") == {"d1", "e2"}  # not c2/d3
    assert moves_from("3k5/9/9/9/9/9/9/9/9/3AK4 w - - 0 1", "d0") == {"e1"}
    # f0 holds the general; c-file and rank 3 are outside the palace.
    assert moves_from("4k4/9/9/9/9/9/9/9/4A4/5K3 w - - 0 1", "e1") == {"d0", "d2", "f2"}


def test_soldier_before_and_after_the_river() -> None:
    assert moves_from("4k4/9/9/9/9/9/4P4/9/9/3K5 w - - 0 1", "e3") == {"e4"}
    assert moves_from("4k4/9/9/9/4P4/9/9/9/9/3K5 w - - 0 1", "e5") == {"e6", "d5", "f5"}
    assert moves_from("3k5/8P/9/9/9/9/9/9/9/4K4 w - - 0 1", "i8") == {"i9", "h8"}
    assert moves_from("3k4P/9/9/9/9/9/9/9/9/4K4 w - - 0 1", "i9") == {"h9"}  # never back
    assert moves_from("4k4/9/9/9/9/4p4/9/9/9/3K5 b - - 0 1", "e4") == {"e3", "d4", "f4"}
    assert moves_from("4k4/9/9/4p4/9/9/9/9/9/3K5 b - - 0 1", "e6") == {"e5"}


def test_flying_general() -> None:
    # The only piece between the generals is pinned: a horse cannot move at all, a cannon
    # only along the file. (With a second blocker on e1 the cannon is free.)
    assert moves_from("4k4/9/9/9/4N4/9/9/9/9/4K4 w - - 0 1", "e5") == set()
    pinned = moves_from("4k4/9/9/9/4C4/9/9/9/9/4K4 w - - 0 1", "e5")
    assert pinned == {"e1", "e2", "e3", "e4", "e6", "e7", "e8"}
    assert "a5" in moves_from("4k4/9/9/9/4C4/9/9/9/4A4/4K4 w - - 0 1", "e5")
    # The general may not step onto an open file facing the enemy general.
    assert moves_from("4k4/9/9/9/9/9/9/9/9/3K5 w - - 0 1", "d0") == {"d1"}
    # Facing generals in a FEN are rejected.
    with pytest.raises(InvalidFen, match="face each other"):
        Position("4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1")


def test_pinned_pieces_and_cannon_screens() -> None:
    # Horse e2 pinned by the black chariot e7: it cannot move at all.
    assert moves_from("3k5/9/4r4/9/9/9/9/4N4/9/4K4 w - - 0 1", "e2") == set()
    # Two red screens between the general and a black cannon: removing either one exposes it.
    fen = "3k5/4c4/9/9/9/4R4/9/4N4/9/4K4 w - - 0 1"
    assert moves_from(fen, "e4") == {"e3", "e5", "e6", "e7", "e8"}
    assert moves_from(fen, "e2") == set()
    # Moving a piece *onto* the open line creates a screen: horse d2-e4 is illegal.
    created = "3k5/4c4/9/9/9/9/9/3N5/9/4K4 w - - 0 1"
    assert "e4" not in moves_from(created, "d2")
    assert {"c4", "b3", "f3", "b1", "f1", "c0"} <= moves_from(created, "d2")


def test_check_must_be_answered() -> None:
    # Black chariot e7 checks; Red can block or step aside (f0 would face the black general).
    fen = "5k3/9/4r4/9/9/9/9/2R6/9/4K4 w - - 0 1"
    assert Position(fen).checked
    moves = legal_set(fen)
    assert moves == {"c2e2", "e0d0"}


# ----------------------------------------------------------------------
# FEN
# ----------------------------------------------------------------------


@pytest.mark.parametrize("fen", [START_FEN] + [fen for fen, _ in PERFT_SUITE])
def test_fen_round_trip(fen: str) -> None:
    assert Position(fen).fen() == fen
    assert mirror_fen(mirror_fen(fen)) == fen


def test_fen_aliases_defaults_and_counters() -> None:
    aliased = "rheakaehr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RHEAKAEHR r - - 0 1"
    assert Position(aliased).fen() == START_FEN
    assert Position(START_FEN.split()[0]).fen() == START_FEN
    position = Position("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR b - - 7 12")
    assert position.side == BLACK and position.halfmove == 7 and position.fullmove == 12
    assert position.ply == 23
    position.push(move_from_uci("h9g7"))
    assert position.fen().endswith("w - - 8 13")


@pytest.mark.parametrize(
    ("fen", "reason"),
    [
        ("", "1-6"),
        ("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/RNBAKABNR w", "10 ranks"),
        ("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNRR w", "more than 9"),
        ("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABN w", "8 points"),
        ("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNX w", "unknown piece"),
        ("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR x", "side to move"),
        (START_FEN.replace(" 0 1", " a 1"), "integers"),
        (START_FEN.replace(" 0 1", " -1 1"), "non-negative"),
        ("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBA1ABNR w", "exactly one general"),
        ("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/4K4/RNBAKABNR w", "exactly one"),
        ("4k4/9/9/9/9/9/9/9/9/2K6 w", "general on c0"),
        ("4k4/9/9/9/9/9/9/9/9/3KA4 w", "advisor on e0"),
        ("4k4/9/9/9/9/9/4B4/9/9/3K5 w", "elephant on e3"),
        ("4k4/9/9/9/2B6/9/9/9/9/3K5 w", "elephant on c5"),
        ("4k4/9/9/9/9/9/1P7/9/9/3K5 w", "soldier on b3"),
        ("4k4/9/9/9/9/9/9/4P4/9/3K5 w", "soldier on e2"),
        ("4k4/9/9/9/9/9/9/RRR6/9/3K5 w", "3 chariots"),
        ("4k4/9/9/9/9/9/9/4R4/9/3K5 w", "not to move"),
    ],
)
def test_invalid_fens_are_rejected(fen: str, reason: str) -> None:
    with pytest.raises(InvalidFen, match=reason):
        Position(fen)


def test_invalid_fen_is_a_value_error() -> None:
    with pytest.raises(ValueError):
        Position("not a fen")


def test_push_uci_validates() -> None:
    position = Position()
    assert move_to_uci(position.push_uci("h2e2")) == "h2e2"
    with pytest.raises(IllegalMove):
        position.push_uci("e2e9")  # it is Black's move
    with pytest.raises(IllegalMove):
        position.push_uci("a9a5")  # chariot cannot jump over the soldier


# ----------------------------------------------------------------------
# Brute-force reference move generator
# ----------------------------------------------------------------------

Grid = list[list[str]]


def parse_grid(fen: str) -> tuple[Grid, bool]:
    rows = fen.split()[0].split("/")
    grid = [["."] * 9 for _ in range(10)]
    for i, row in enumerate(rows):
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
            else:
                grid[9 - i][f] = ch
                f += 1
    return grid, fen.split()[1] == "w"


def in_palace(f: int, r: int, red: bool) -> bool:
    return 3 <= f <= 5 and (0 <= r <= 2 if red else 7 <= r <= 9)


def targets(grid: Grid, f: int, r: int) -> list[tuple[int, int]]:
    """Pseudo-legal destinations of the piece on (f, r), written straight from the rules."""
    p = grid[r][f]
    red = p.isupper()
    kind = p.upper()

    def on(f2: int, r2: int) -> bool:
        return 0 <= f2 < 9 and 0 <= r2 < 10

    def own(f2: int, r2: int) -> bool:
        q = grid[r2][f2]
        return q != "." and q.isupper() == red

    out: list[tuple[int, int]] = []
    if kind == "K":
        for df, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if in_palace(f + df, r + dr, red) and not own(f + df, r + dr):
                out.append((f + df, r + dr))
    elif kind == "A":
        for df, dr in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            if in_palace(f + df, r + dr, red) and not own(f + df, r + dr):
                out.append((f + df, r + dr))
    elif kind == "B":
        for df, dr in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            f2, r2 = f + 2 * df, r + 2 * dr
            home = r2 <= 4 if red else r2 >= 5
            if on(f2, r2) and home and grid[r + dr][f + df] == "." and not own(f2, r2):
                out.append((f2, r2))
    elif kind == "N":
        for (df, dr), (lf, lr) in (
            ((1, 2), (0, 1)), ((-1, 2), (0, 1)), ((1, -2), (0, -1)), ((-1, -2), (0, -1)),
            ((2, 1), (1, 0)), ((2, -1), (1, 0)), ((-2, 1), (-1, 0)), ((-2, -1), (-1, 0)),
        ):  # fmt: skip
            if on(f + df, r + dr) and grid[r + lr][f + lf] == "." and not own(f + df, r + dr):
                out.append((f + df, r + dr))
    elif kind in "RC":
        for df, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            f2, r2 = f + df, r + dr
            jumped = False
            while on(f2, r2):
                q = grid[r2][f2]
                if not jumped:
                    if q == ".":
                        out.append((f2, r2))
                    elif kind == "R":
                        if not own(f2, r2):
                            out.append((f2, r2))
                        break
                    else:
                        jumped = True
                elif q != ".":
                    if not own(f2, r2):
                        out.append((f2, r2))
                    break
                f2, r2 = f2 + df, r2 + dr
    elif kind == "P":
        forward = 1 if red else -1
        crossed = r >= 5 if red else r <= 4
        steps = [(0, forward)] + ([(1, 0), (-1, 0)] if crossed else [])
        for df, dr in steps:
            if on(f + df, r + dr) and not own(f + df, r + dr):
                out.append((f + df, r + dr))
    return out


def general_attacked(grid: Grid, red: bool) -> bool:
    king = "K" if red else "k"
    kf, kr = next((f, r) for r in range(10) for f in range(9) if grid[r][f] == king)
    other = "k" if red else "K"
    of, orank = next((f, r) for r in range(10) for f in range(9) if grid[r][f] == other)
    if kf == of and all(grid[r][kf] == "." for r in range(min(kr, orank) + 1, max(kr, orank))):
        return True
    for r in range(10):
        for f in range(9):
            q = grid[r][f]
            if q != "." and q.isupper() != red and (kf, kr) in targets(grid, f, r):
                return True
    return False


def reference_legal_moves(fen: str) -> set[str]:
    grid, red = parse_grid(fen)
    legal = set()
    for r in range(10):
        for f in range(9):
            q = grid[r][f]
            if q == "." or q.isupper() != red:
                continue
            for f2, r2 in targets(grid, f, r):
                after = [row[:] for row in grid]
                after[r2][f2] = q
                after[r][f] = "."
                if not general_attacked(after, red):
                    legal.add(f"{'abcdefghi'[f]}{r}{'abcdefghi'[f2]}{r2}")
    return legal


def sample_positions(n_games: int, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    fens = [START_FEN] + [fen for fen, _ in PERFT_SUITE]
    for _ in range(n_games):
        position = Position()
        for ply in range(160):
            moves = position.legal_moves()
            if not moves:
                break
            if ply % 7 == 3:
                fens.append(position.fen())
            position.push(int(rng.choice(moves)))
        fens.append(position.fen())
    return fens


def test_legal_moves_match_brute_force_reference() -> None:
    for fen in sample_positions(n_games=4, seed=1):
        assert legal_set(fen) == reference_legal_moves(fen), fen


def test_is_legal_agrees_with_move_generation_on_every_pair() -> None:
    for fen in sample_positions(n_games=1, seed=2)[::4]:
        position = Position(fen)
        generated = set(position.legal_moves())
        accepted = {m for m in range(8100) if position.is_legal(m)}
        assert accepted == generated, fen


def test_checked_flag_matches_reference() -> None:
    for fen in sample_positions(n_games=3, seed=3):
        grid, red = parse_grid(fen)
        assert Position(fen).checked == general_attacked(grid, red), fen


def test_has_legal_move_agrees_with_move_generation() -> None:
    for fen in sample_positions(n_games=2, seed=5):
        position = Position(fen)
        assert position.has_legal_move() == bool(position.legal_moves()), fen


def test_attackers_matches_reference() -> None:
    for fen in sample_positions(n_games=2, seed=4)[::3]:
        position = Position(fen)
        grid, _ = parse_grid(fen)
        for by in (RED, BLACK):
            for sq in range(90):
                f, r = sq % 9, sq // 9
                target = grid[r][f]
                if target != "." and target.isupper() == (by == RED):
                    continue  # attacks are defined on empty and enemy-occupied points
                # A piece attacks sq if it could capture an enemy piece standing there.
                probe = [row[:] for row in grid]
                if target == ".":
                    probe[r][f] = "p" if by == RED else "P"
                expected = set()
                for r2 in range(10):
                    for f2 in range(9):
                        q = grid[r2][f2]
                        if q == "." or q.isupper() != (by == RED):
                            continue
                        if (f, r) in targets(probe, f2, r2):
                            expected.add(r2 * 9 + f2)
                        elif (
                            q.upper() == "K"
                            and target.upper() == "K"
                            and f == f2
                            and all(grid[x][f] == "." for x in range(min(r, r2) + 1, max(r, r2)))
                        ):
                            expected.add(r2 * 9 + f2)  # flying general
                assert set(position.attackers(sq, by)) == expected, (fen, sq, by)


def test_errors_are_the_games_errors() -> None:
    from sekai.envs.games import errors as games
    from sekai.envs.games.xiangqi import errors

    assert issubclass(errors.XiangqiError, games.GameError)
    assert issubclass(errors.IllegalMove, games.IllegalMove)
    assert issubclass(errors.IllegalMove, errors.XiangqiError)
    assert issubclass(errors.GameOver, games.GameOver)
    assert issubclass(errors.GameOver, errors.XiangqiError)
    assert issubclass(errors.InvalidFen, ValueError)
    assert issubclass(errors.InvalidMoveFormat, ValueError)
    error = errors.IllegalMove("h2h9", START_FEN, action=7, reason="test")
    assert (error.move, error.fen, error.action, error.reason) == ("h2h9", START_FEN, 7, "test")
    assert str(error).startswith("Action 7 decodes to move h2h9, which is illegal")
