"""The Go engine: captures, suicide, ko, eyes, hashing, playouts and legal-move bitsets."""

from __future__ import annotations

import random

import pytest

from sekai.envs.games.go import action_to_vertex, vertex_to_action
from sekai.envs.games.go.board import (
    BLACK,
    EMPTY,
    WHITE,
    Board,
    bits_to_points,
    geometry,
    handicap_points,
)
from sekai.envs.games.go.errors import IllegalMove


def board_from(rows: list[str], to_play: int = BLACK) -> Board:
    return Board.from_string("".join(r.replace(" ", "") for r in rows), to_play)


def v(vertex: str, size: int = 9) -> int:
    return vertex_to_action(vertex, size)


def assert_consistent(board: Board) -> None:
    """Incremental state must equal a from-scratch rebuild of the same colours."""
    ref = Board.from_colors(board.color, board.size, board.to_play)
    assert sorted(board.empties) == sorted(ref.empties)
    for i, p in enumerate(board.empties):
        assert board.eidx[p] == i
    for p in range(board.n):
        if board.color[p] != EMPTY:
            assert board.gstones[board.gid[p]] == ref.gstones[ref.gid[p]]
            assert board.glibs[board.gid[p]] == ref.glibs[ref.gid[p]]
    assert board.hash == ref.hash
    assert board.stones == ref.stones


# ----------------------------------------------------------------------
# Captures
# ----------------------------------------------------------------------


def test_single_stone_capture_in_the_middle() -> None:
    board = board_from(
        [
            ".....",
            "..X..",
            ".XO..",
            "..X..",
            ".....",
        ]
    )
    assert board.liberty_count(v("C3", 5)) == 1
    captured = board.play(v("D3", 5))
    assert captured == 1
    assert board.color[v("C3", 5)] == EMPTY
    assert board.to_play == WHITE
    assert_consistent(board)


def test_multi_stone_capture() -> None:
    board = board_from(
        [
            ".XX..",
            "XOOX.",
            ".XO..",
            "..X..",
            ".....",
        ]
    )
    # White group B4 C4 C3 has its last liberty at D3.
    assert board.liberties(v("C3", 5)) == [v("D3", 5)]
    assert board.play(v("D3", 5)) == 3
    for vertex in ("B4", "C4", "C3"):
        assert board.color[v(vertex, 5)] == EMPTY
    # D3 joins D4; the merged group regains C3 and C4 among its six liberties.
    assert board.liberties(v("D3", 5)) == sorted(
        v(x, 5) for x in ("C4", "E4", "D5", "C3", "E3", "D2")
    )
    assert_consistent(board)


def test_edge_and_corner_captures() -> None:
    edge = board_from(
        [
            ".....",
            ".....",
            ".....",
            ".....",
            ".XOX.",
        ]
    )
    assert edge.liberties(v("C1", 5)) == [v("C2", 5)]
    assert edge.play(v("C2", 5)) == 1
    assert edge.color[v("C1", 5)] == EMPTY

    corner = board_from(
        [
            ".....",
            ".....",
            ".....",
            ".....",
            "OX...",
        ]
    )
    assert corner.play(v("A2", 5)) == 1
    assert corner.color[v("A1", 5)] == EMPTY
    assert_consistent(corner)


def test_capture_of_two_groups_frees_liberties_for_every_neighbour() -> None:
    board = board_from(
        [
            "X.X..",
            "OXO..",
            ".O...",
            ".....",
            ".....",
        ],
        to_play=WHITE,
    )
    # White B5 has no empty neighbour but takes the last liberty of both A5 and B4.
    b5 = v("B5", 5)
    assert board.is_legal(b5)
    assert board.play(b5) == 2
    assert board.color[v("A5", 5)] == EMPTY and board.color[v("B4", 5)] == EMPTY
    for white in ("A4", "C4", "B3", "B5"):
        assert v("B4", 5) in board.liberties(v(white, 5))
    assert_consistent(board)


# ----------------------------------------------------------------------
# Suicide and capture-not-suicide
# ----------------------------------------------------------------------


def test_single_stone_suicide_is_illegal() -> None:
    board = board_from(
        [
            ".O...",
            "O....",
            ".....",
            ".....",
            ".....",
        ]
    )
    a5 = v("A5", 5)
    assert not board.is_legal(a5)
    assert board.illegal_reason(a5) == "suicide"
    with pytest.raises(IllegalMove) as excinfo:
        board.play(a5)
    assert excinfo.value.reason == "suicide"
    assert board.color[a5] == EMPTY and board.to_play == BLACK


def test_multi_stone_suicide_is_illegal() -> None:
    board = board_from(
        [
            "X.O..",
            "XO...",
            "O....",
            ".....",
            ".....",
        ]
    )
    # Black A5 A4 has one liberty at B5; filling it captures nothing (C5 and B4 have
    # other liberties) and would leave the merged group with none.
    b5 = v("B5", 5)
    assert board.liberties(v("A5", 5)) == [b5]
    assert board.illegal_reason(b5) == "suicide"


def test_capture_is_not_suicide() -> None:
    board = board_from(
        [
            ".OX..",
            "OX...",
            "X....",
            ".....",
            ".....",
        ]
    )
    # A5 has no empty neighbour, but it is the last liberty of white B5 and of white A4.
    a5 = v("A5", 5)
    assert board.is_legal(a5)
    assert board.play(a5) == 2
    assert board.color[a5] == BLACK
    assert board.liberty_count(a5) == 2
    assert_consistent(board)


# ----------------------------------------------------------------------
# Simple ko
# ----------------------------------------------------------------------


KO_ROWS = [
    ".XO..",
    "XO.O.",
    ".XO..",
    ".....",
    ".....",
]


def test_simple_ko_forbids_immediate_recapture() -> None:
    board = board_from(KO_ROWS)
    c4, b4 = v("C4", 5), v("B4", 5)
    assert board.play(c4) == 1  # black takes the ko
    assert board.ko == b4
    assert board.to_play == WHITE
    assert not board.is_legal(b4)
    assert board.illegal_reason(b4) == "ko"
    assert board.is_legal(b4, ignore_ko=True)
    # A ko threat exchange clears the ko.
    board.play(v("E1", 5))
    assert board.ko == -1
    board.play(v("A1", 5))
    assert board.is_legal(b4)
    assert board.play(b4) == 1
    assert board.ko == c4


def test_pass_clears_ko() -> None:
    board = board_from(KO_ROWS)
    board.play(v("C4", 5))
    board.pass_move()
    assert board.ko == -1 and board.to_play == BLACK


def test_capture_leaving_extra_liberties_is_not_ko() -> None:
    board = board_from(
        [
            ".XOO.",
            "XO..O",
            ".XOO.",
            ".....",
            ".....",
        ]
    )
    # Black C4 captures the single stone B4 but also keeps the liberty D4: no ko.
    assert board.play(v("C4", 5)) == 1
    assert board.ko == -1


# ----------------------------------------------------------------------
# Eyes
# ----------------------------------------------------------------------


def test_eye_definition_centre_edge_corner() -> None:
    centre = board_from(
        [
            ".....",
            "..X..",
            ".X.X.",
            "..X..",
            ".....",
        ]
    )
    c3 = v("C3", 5)
    assert centre.is_eye(c3, BLACK)
    assert not centre.is_eye(c3, WHITE)
    one_bad = board_from(
        [
            ".....",
            ".OX..",
            ".X.X.",
            "..X..",
            ".....",
        ]
    )
    assert one_bad.is_eye(c3, BLACK)  # one opponent diagonal is fine in the centre
    two_bad = board_from(
        [
            ".....",
            ".OXO.",
            ".X.X.",
            "..X..",
            ".....",
        ]
    )
    assert not two_bad.is_eye(c3, BLACK)

    edge = board_from(
        [
            ".....",
            ".....",
            ".....",
            "..X..",
            ".X.X.",
        ]
    )
    c1 = v("C1", 5)
    assert edge.is_eye(c1, BLACK)
    edge_bad = board_from(
        [
            ".....",
            ".....",
            ".....",
            ".OX..",
            ".X.X.",
        ]
    )
    assert not edge_bad.is_eye(c1, BLACK)  # none allowed on the edge

    corner = board_from(
        [
            ".....",
            ".....",
            ".....",
            "X....",
            ".X...",
        ]
    )
    a1 = v("A1", 5)
    assert corner.is_eye(a1, BLACK)
    corner_bad = board_from(
        [
            ".....",
            ".....",
            ".....",
            "XO...",
            ".X...",
        ]
    )
    assert not corner_bad.is_eye(a1, BLACK)


def test_suicide_of_two_stones_and_self_atari_detection() -> None:
    closed = board_from(
        [
            ".....",
            ".OO..",
            "O.XO.",
            ".OO..",
            ".....",
        ]
    )
    assert closed.illegal_reason(v("B3", 5)) == "suicide"

    board = board_from(
        [
            ".....",
            ".OO..",
            "..XO.",
            ".OO..",
            ".O...",
        ]
    )
    assert board.is_self_atari(v("B3", 5))  # B3 + C3 keep only A3
    assert not board.is_self_atari(v("A1", 5))  # a lone stone in atari does not count
    assert not board.is_self_atari(v("E5", 5))


# ----------------------------------------------------------------------
# Hashing, copies, scoring
# ----------------------------------------------------------------------


def test_incremental_state_matches_rebuild_in_random_games() -> None:
    rnd = random.Random(7)
    for size in (2, 3, 5, 7):
        for _game in range(12):
            board = Board(size)
            for _ in range(3 * size * size):
                legal = board.legal_points()
                if not legal or rnd.random() < 0.05:
                    board.pass_move()
                    continue
                p = rnd.choice(legal)
                expected_hash = board.hash_after(p)
                expected_colors = board.colors_after(p)
                expected_capture = board.captured_by(p).bit_count()
                assert board.play(p) == expected_capture
                assert board.hash == expected_hash
                assert board.config() == expected_colors
                assert_consistent(board)


def test_hash_ignores_move_order() -> None:
    a = Board(9)
    for vertex in ("C3", "G7", "E5"):
        a.play(v(vertex))
    b = Board(9)
    for vertex in ("E5", "G7", "C3"):
        b.play(v(vertex))
    assert a.hash == b.hash  # same stones (black C3 E5, white G7)
    c = Board(9)
    for vertex in ("C3", "E5", "G7"):
        c.play(v(vertex))
    assert c.hash != a.hash  # E5 is white here


def test_copy_is_independent() -> None:
    board = Board(9)
    board.play(v("E5"))
    clone = board.copy()
    clone.play(v("E6"))
    assert board.color[v("E6")] == EMPTY
    assert board.to_play == WHITE and clone.to_play == BLACK
    assert board.geo is clone.geo
    assert_consistent(board)
    assert_consistent(clone)


def test_area_score() -> None:
    assert Board(9).area_score() == (0, 0)
    board = Board(9)
    board.play(v("E5"))
    assert board.area_score() == (81, 0)
    board.play(v("E6"))
    assert board.area_score() == (1, 1)
    split = board_from(
        [
            "..XO.",
            "..XO.",
            "..XO.",
            "..XO.",
            "..XO.",
        ]
    )
    assert split.area_score() == (15, 10)
    dame = board_from(
        [
            "..X.O",
            "..X.O",
            "..X.O",
            "..X.O",
            "..X.O",
        ]
    )
    assert dame.area_score() == (15, 5)


def test_random_playout_ends_with_two_passes() -> None:
    rnd = random.Random(3)
    for size in (5, 9, 13):
        board = Board(size)
        moves = board.random_playout(rnd)
        assert 0 < moves < 3 * size * size
        assert_consistent(board)
        # The side that passed last did so in exactly this position (ko already cleared).
        last_passer = 3 - board.to_play
        board.to_play = last_passer
        assert board.ko == -1
        assert [p for p in board.legal_points() if not board.is_eye(p, last_passer)] == []


def test_random_playout_records_amaf_and_respects_seed() -> None:
    first, second = Board(9), Board(9)
    amaf1, amaf2 = [0] * 81, [0] * 81
    first.random_playout(random.Random(11), amaf=amaf1)
    second.random_playout(random.Random(11), amaf=amaf2)
    assert first.to_string() == second.to_string()
    assert amaf1 == amaf2
    assert set(amaf1) <= {0, BLACK, WHITE} and BLACK in amaf1 and WHITE in amaf1


def test_group_after_matches_play() -> None:
    rnd = random.Random(5)
    board = Board(7)
    board.random_playout(rnd, max_moves=40)
    for p in board.legal_points():
        stones, libs, captured = board.group_after(p)
        trial = board.copy()
        trial.play(p)
        assert stones == trial.gstones[trial.gid[p]]
        assert libs == trial.glibs[trial.gid[p]]
        assert captured == board.captured_by(p)
        assert captured.bit_count() == board.n - len(board.empties) + 1 - trial.stone_count()


def test_from_string_rejects_dead_groups_and_bad_input() -> None:
    with pytest.raises(ValueError):
        board_from(["OX...", "X....", ".....", ".....", "....."])  # A5 has no liberties
    with pytest.raises(ValueError):
        Board.from_string("." * 10)
    with pytest.raises(ValueError):
        Board.from_string("." * 8 + "Z")


def test_geometry_tables() -> None:
    geo = geometry(9)
    assert geometry(9) is geo
    assert len(geo.nbrs[0]) == 2 and len(geo.nbrs[40]) == 4 and len(geo.nbrs[4]) == 3
    assert geo.edge[0] == 1 and geo.edge[40] == 0
    assert geo.line[40] == 5
    assert bits_to_points(geo.adjacent(1 << 40)) == sorted(geo.nbrs[40])
    assert bits_to_points(geo.adjacent(1 << 8)) == sorted(geo.nbrs[8])  # no wrap-around


def test_handicap_points() -> None:
    def vertices(size: int, stones: int) -> list[str]:
        return [action_to_vertex(p, size) for p in handicap_points(size, stones)]

    assert vertices(9, 2) == ["C3", "G7"]
    assert vertices(13, 4) == ["D4", "K10", "D10", "K4"]
    assert vertices(19, 3) == ["D4", "Q16", "D16"]
    assert vertices(19, 5) == ["D4", "Q16", "D16", "Q4", "K10"]
    assert sorted(vertices(19, 9)) == sorted(
        ["D4", "Q16", "D16", "Q4", "D10", "Q10", "K4", "K16", "K10"]
    )
    assert "K10" not in vertices(19, 6) and "K10" not in vertices(19, 8)
    for bad_size in (7, 11, 18):
        with pytest.raises(ValueError):
            handicap_points(bad_size, 2)
    for bad_count in (0, 1, 10):
        with pytest.raises(ValueError):
            handicap_points(19, bad_count)


def test_legal_bits_match_legal_points_in_random_games() -> None:
    rnd = random.Random(11)
    for size in (2, 3, 5, 9):
        for _game in range(6):
            board = Board(size)
            for _ in range(4 * size * size):
                for ignore in (False, True):
                    legal, captures = board.legal_bits(ignore_ko=ignore)
                    points = board.legal_points(ignore_ko=ignore)
                    assert bits_to_points(legal) == points
                    assert bits_to_points(captures) == [p for p in points if board.captured_by(p)]
                legal_points = board.legal_points()
                if not legal_points or rnd.random() < 0.05:
                    board.pass_move()
                else:
                    board.play(rnd.choice(legal_points))
