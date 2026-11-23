"""Scoring of Go positions (Tromp-Taylor counts, dead-stone estimation) and SGF records."""

from __future__ import annotations

import datetime
import random
import time

import numpy as np
import pytest

from sekai.envs.games.go import (
    Board,
    GoEnv,
    InvalidVertex,
    area_score,
    estimate_ownership,
    final_score,
    to_sgf,
)
from sekai.envs.games.go.scoring import GoScore, dead_groups
from sekai.envs.games.go.sgf import sgf_point, sgf_to_action


def make_env(rows: list[str], to_play: str = "black", komi: float = 7.5) -> GoEnv:
    env = GoEnv(len(rows), komi=komi)
    env.reset(options={"board": "".join(r.replace(" ", "") for r in rows), "to_play": to_play})
    return env


# Black owns the bottom, White the top; each has a clearly dead stone inside the other's
# area (black D7 among white stones, white C3 among black stones).
DEAD_ROWS = [
    ". O . O . O O . O",
    "O O O O O O . O O",
    "O . . X . O O O O",
    "O O O O O O O O O",
    "X X X X X X X X X",
    "X . . X X . X X X",
    "X . O X . X . X X",
    "X X X X X X X . X",
    ". X . X . X X X .",
]

# A seki: the black ring and the white group inside it share the liberties C7 and E7 and
# neither can fill them without being captured. All other points are settled.
SEKI_ROWS = [
    "OOOOOOO.O",
    "OXXXXXOOO",
    "OX.O.XO.O",
    "OXOOOXOOO",
    "OXXXXXOOO",
    "OOOOOOOOO",
    "XXXXXXXXX",
    "X.X.X.X.X",
    ".X.X.X.X.",
]


def test_area_score_accepts_env_and_board() -> None:
    env = make_env(DEAD_ROWS)
    assert area_score(env) == area_score(env.board) == env.board.area_score()
    # Black: 33 stones + 9 territory (the region around C3 is shared with white).
    assert area_score(env) == (42, 33)


def test_final_score_without_estimation_is_tromp_taylor() -> None:
    env = make_env(DEAD_ROWS)
    score = final_score(env, estimate_dead=False)
    assert isinstance(score, GoScore)
    assert (score.black, score.white) == (42.0, 40.5)
    assert score.result == "B+1.5" and score.winner == "black" and score.margin == 1.5
    assert score.dead_stones == []
    assert "Tromp-Taylor" in score.method
    # Matches the environment's own result when the game ends now.
    env.step(81)
    env.step(81)
    assert env.outcome is not None and env.outcome.result == score.result
    assert (env.outcome.black, env.outcome.white) == (score.black, score.white)


@pytest.mark.parametrize("to_play", ["black", "white"])
def test_final_score_removes_clearly_dead_stones(to_play: str) -> None:
    env = make_env(DEAD_ROWS, to_play=to_play)
    before = env.board_string()
    score = final_score(env, rng=0)
    assert env.board_string() == before  # the env is not modified
    assert score.dead_stones == ["D7", "C3"]
    assert (score.black, score.white) == (45.0, 43.5)
    assert score.result == "B+1.5"
    assert "C3" in score.territory["black"] and "B3" in score.territory["black"]
    assert "D7" in score.territory["white"] and "A9" in score.territory["white"]
    assert not set(score.territory["black"]) & set(score.territory["white"])
    assert "200 playouts" in score.method


def test_dead_stones_change_the_winner() -> None:
    rows = list(DEAD_ROWS)
    rows[2] = "O . . . . O O O O"  # no dead black stone: only the white one remains
    env = make_env(rows, komi=7.5)
    tt = final_score(env, estimate_dead=False)
    estimated = final_score(env, rng=1)
    assert tt.result == "W+3.5"  # the dead white stone spoils black's area
    assert estimated.dead_stones == ["C3"]
    assert estimated.result == "B+1.5"
    assert (estimated.black, estimated.white) == (tt.black + 4, tt.white - 1)


@pytest.mark.parametrize("to_play", ["black", "white"])
def test_seki_is_kept_alive(to_play: str) -> None:
    env = make_env(SEKI_ROWS, to_play=to_play)
    score = final_score(env, rng=0)
    assert score.dead_stones == []
    tt = final_score(env, estimate_dead=False)
    assert score.result == tt.result
    ownership = estimate_ownership(env, 100, rng=0)
    c7, e7 = env.encode_vertex("C7"), env.encode_vertex("E7")
    assert ownership[c7] == 0 and ownership[e7] == 0  # shared liberties stay neutral


def test_estimate_ownership_properties() -> None:
    env = GoEnv(9)
    env.reset(options={"moves": ["E5", "C3", "G7"]})
    before = (env.board_string(), env.moves, env.board.hash)
    own = estimate_ownership(env, 50, rng=3)
    assert own.shape == (81,) and own.dtype == np.float32
    assert np.all(own >= -1) and np.all(own <= 1)
    np.testing.assert_array_equal(own, estimate_ownership(env, 50, rng=3))
    assert (env.board_string(), env.moves, env.board.hash) == before
    assert own[env.encode_vertex("E5")] > 0 and own[env.encode_vertex("C3")] < 0


def test_final_score_on_a_bare_board_and_timing() -> None:
    board = Board.from_string("".join(r.replace(" ", "") for r in DEAD_ROWS))
    assert final_score(board, komi=7.5, rng=0).dead_stones == ["D7", "C3"]
    assert final_score(board, estimate_dead=False).white == 33.0  # no komi by default
    for size in (9, 19):
        env = GoEnv(size)
        env.reset()
        start = time.perf_counter()
        score = final_score(env, rng=0)
        elapsed = time.perf_counter() - start
        assert score.result == "W+7.5"
        assert elapsed < (2.0 if size == 9 else 6.0)  # loose: the machine is shared


# ----------------------------------------------------------------------
# SGF
# ----------------------------------------------------------------------


def test_simple_game_record() -> None:
    sgf = to_sgf(9, 7.5, ["E5", "C3", "pass", "pass"], result="W+5.5", date="2026-10-18")
    assert sgf == (
        "(;GM[1]FF[4]CA[UTF-8]AP[sekai]SZ[9]KM[7.5]RU[Chinese]PB[Black]PW[White]"
        "DT[2026-10-18]RE[W+5.5];B[ee];W[cg];B[];W[])\n"
    )
    other = to_sgf(9, 7.5, [], date="x", application="Kifu]Writer")
    assert other.startswith("(;GM[1]FF[4]CA[UTF-8]AP[Kifu\\]Writer]SZ[9]")


def test_coordinates_are_from_the_top_left() -> None:
    assert sgf_point(0, 19) == "aa"  # A19
    assert sgf_point(18, 19) == "sa"  # T19
    assert sgf_point(342, 19) == "as"  # A1
    assert sgf_point(361, 19) == ""
    for action in range(82):
        assert sgf_to_action(sgf_point(action, 9), 9) == action
    assert sgf_to_action("tt", 19) == 361
    with pytest.raises(InvalidVertex):
        sgf_to_action("zz", 9)
    with pytest.raises(InvalidVertex):
        sgf_point(81 + 1, 9)


def test_handicap_names_dates_and_integer_moves() -> None:
    sgf = to_sgf(
        19,
        0.5,
        ["Q3", "R16"],
        black_name="Visitor [guest]",
        white_name="Tree\\Search",
        date=datetime.date(2026, 1, 2),
        handicap=2,
    )
    assert "HA[2]AB[dp][pd]" in sgf
    assert ";W[pq];B[qd])" in sgf  # White moves first after the handicap
    assert "PB[Visitor [guest\\]]" in sgf and "PW[Tree\\\\Search]" in sgf
    assert "DT[2026-01-02]" in sgf and "KM[0.5]" in sgf and "RE[" not in sgf
    assert "DT[2026-01-02]" in to_sgf(9, 7, [], date=datetime.datetime(2026, 1, 2, 13, 5))
    assert to_sgf(9, 7, [40, 81], date="x").count(";B[ee];W[]") == 1
    assert "KM[7]" in to_sgf(9, 7, [], date="x")
    assert "DT[" in to_sgf(9, 7.5, [])


def test_env_game_round_trip() -> None:
    env = GoEnv(9)
    env.reset(options={"handicap": 2, "moves": ["E5", "D4", "pass", "pass"]})
    sgf = to_sgf(9, env.komi, env.moves, result=env.outcome.result, handicap=env.handicap)
    assert "AB[cg][gc]" in sgf
    assert sgf.endswith(";W[ee];B[df];W[];B[])\n")
    assert f"RE[{env.outcome.result}]" in sgf


def test_scoring_rngs_and_dead_groups() -> None:
    env = make_env(DEAD_ROWS)
    # A seed, a Generator seeded alike and a random.Random give reproducible estimates.
    a = estimate_ownership(env, 30, rng=5)
    b = estimate_ownership(env, 30, rng=np.random.default_rng(5))
    np.testing.assert_array_equal(a, b)
    c = estimate_ownership(env, 30, rng=random.Random(9))
    np.testing.assert_array_equal(c, estimate_ownership(env, 30, rng=random.Random(9)))
    dead = dead_groups(env.board, a)
    assert [env.decode_action(p) for p in dead] == ["D7", "C3"]
    assert final_score(env, rng=0, playouts=1).method.endswith("from 1 playouts")
    with pytest.raises(TypeError):
        final_score(object())
