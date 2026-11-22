"""The Go environment: rules, observations, info, reset options, superko, scoring of
finished games, registration, check_env, OpponentWrapper and golden rollout digests."""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

import numpy as np
import pytest

import sekai
from sekai.envs.games import OpponentWrapper, RandomPolicy
from sekai.envs.games import errors as game_errors
from sekai.envs.games.go import (
    GameOver,
    GoEnv,
    GoEnvError,
    GoResult,
    IllegalMove,
    InvalidVertex,
    action_to_vertex,
    vertex_to_action,
)
from sekai.envs.games.go.env import replay

INFO_KEYS = [
    "legal_action_mask",
    "legal_actions",
    "to_play",
    "last_move",
    "ply",
    "captures",
    "passes",
    "komi",
    "result",
    "winner",
]


def rows_to_board(rows: list[str]) -> str:
    return "".join(r.replace(" ", "") for r in rows)


def make_env(rows: list[str], to_play: str = "black", **kwargs: Any) -> GoEnv:
    env = GoEnv(len(rows), **kwargs)
    env.reset(options={"board": rows_to_board(rows), "to_play": to_play})
    return env


# ----------------------------------------------------------------------
# Golden digests: seeded random games, every output hashed
# ----------------------------------------------------------------------


def rollout_digest(
    env: Any, seed: int, options: dict[str, Any] | None = None, pass_every: int = 40
) -> tuple[str, int]:
    """sha256 over everything `reset` and `step` return in a game chosen by a fixed LCG
    (observation bytes, mask bytes, reward, flags and the info as JSON), and the plies.

    The game passes about once in `pass_every` moves (and when it has no other move)."""
    digest = hashlib.sha256()

    def record(result: Any) -> None:
        info = result.info
        digest.update(result.observation.dtype.str.encode())
        digest.update(repr(result.observation.shape).encode())
        digest.update(result.observation.tobytes())
        digest.update(info["legal_action_mask"].dtype.str.encode())
        digest.update(info["legal_action_mask"].tobytes())
        digest.update(info["legal_actions"].dtype.str.encode())
        flags = (getattr(result, "reward", None), getattr(result, "terminated", None))
        digest.update(repr((*flags, getattr(result, "truncated", None))).encode())
        digest.update(json.dumps(info, default=lambda a: a.tolist()).encode())

    state = seed
    result = env.reset(seed=seed, options=options)
    record(result)
    plies = 0
    n = env.unwrapped.num_points
    while not env.unwrapped.done:
        state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
        legal = [int(a) for a in result.info["legal_actions"] if a < n]
        if not legal or (state >> 33) % pass_every == 0:
            action = n
        else:
            action = legal[(state >> 11) % len(legal)]
        result = env.step(action)
        record(result)
        plies += 1
    return digest.hexdigest(), plies


GOLDEN: list[tuple[dict[str, Any], int, dict[str, Any] | None, int, str, int]] = [
    # max moves, B+2.5
    (
        {"size": 9},
        0,
        None,
        40,
        "e97940c3a598336aee621ff7bda3082675e0eb3bf026ddf96700232e3f7f94c5",
        243,
    ),
    # two passes, W+7.5
    (
        {"size": 9},
        1,
        None,
        6,
        "0c330764a9897650278d15492e08ca5aefc53c728cc209ba70d207d4a38dd3e1",
        4,
    ),
    # two passes, W+38
    (
        {"size": 9, "komi": 6},
        7,
        {"moves": ["E5", "C3", "pass", "G7"]},
        12,
        "9694a18b6fd92966b3f9b679c83be17f5cdcdefaacb2888207b5c533444d7f20",
        143,
    ),
    # max moves, W+4.5
    (
        {"size": 13},
        3,
        {"handicap": 4, "komi": 0.5},
        40,
        "54a7c04868af7ec56c7dcb1bc2d6649be5ca3b13a718e04d77cd7f28820cca58",
        507,
    ),
    # two passes, W+18.5
    (
        {"size": 13, "superko": False},
        8,
        None,
        15,
        "e6a53994097dbf370a4e7502b77d216ed62186cffb40c6898cb85420f1782695",
        184,
    ),
    # max moves, W+6.5
    (
        {"size": 19, "max_moves": 150, "history_length": 4},
        11,
        None,
        40,
        "7447b198066cac9389e29a7660f9433d4548264b5364c4c01d3c441b66d4a1c2",
        150,
    ),
    # two passes, B+3.5
    (
        {"size": 19},
        12,
        {"handicap": 9},
        25,
        "e4b6cb0ef73f32890c0d1332d690a467b89edba7002bdcd71a8c47e94512165d",
        202,
    ),
    # max moves, W+21.5
    (
        {"size": 5, "superko": False},
        5,
        {"board": ".XO..XO.O..XO............", "to_play": "white"},
        40,
        "0e429844970a76f64ce3a5c872988dbc21d10c0beed730849db327b6783bdd25",
        75,
    ),
    # two passes, a draw
    (
        {"size": 7, "komi": 0},
        2,
        None,
        8,
        "ee30e969c408524766db68dd34d7242a69dc385d866a808aa248c27d6c484829",
        4,
    ),
    # max moves, W+10.5
    (
        {"size": 4},
        9,
        None,
        40,
        "1b1a0db5f95207862add3a0a6d4cf9e81273b94c04ca1d6e175a7d368ff6182e",
        48,
    ),
]
"""(constructor kwargs, seed, reset options, pass_every, digest, plies)."""


@pytest.mark.parametrize("kwargs, seed, options, pass_every, expected, plies", GOLDEN)
def test_golden_digests(
    kwargs: dict[str, Any],
    seed: int,
    options: dict[str, Any] | None,
    pass_every: int,
    expected: str,
    plies: int,
) -> None:
    env = GoEnv(**kwargs)
    assert rollout_digest(env, seed, options, pass_every) == (expected, plies)


def test_golden_digest_survives_reuse_and_midgame_resets() -> None:
    # One env, reset in the middle of games and after a failed reset, replays the same
    # games as fresh envs do: nothing cached outlives a reset.
    env = GoEnv(9)
    for kwargs, seed, options, every, expected, plies in GOLDEN:
        if kwargs != {"size": 9}:
            continue
        env.reset(seed=3, options={"moves": ["E5", "D4", "C3"]})
        env.legal_action_mask()
        with pytest.raises(ValueError):
            env.reset(options={"handicap": 1})
        assert rollout_digest(env, seed, options, every) == (expected, plies)


# ----------------------------------------------------------------------
# A deliberately naive reference implementation (flood fill, full history)
# ----------------------------------------------------------------------


def naive_neighbours(p: int, size: int) -> list[int]:
    r, c = divmod(p, size)
    out = []
    for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= rr < size and 0 <= cc < size:
            out.append(rr * size + cc)
    return out


def naive_group(colors: list[int], p: int, size: int) -> tuple[set[int], set[int]]:
    color = colors[p]
    stones, libs, stack = {p}, set(), [p]
    while stack:
        s = stack.pop()
        for q in naive_neighbours(s, size):
            if colors[q] == 0:
                libs.add(q)
            elif colors[q] == color and q not in stones:
                stones.add(q)
                stack.append(q)
    return stones, libs


def naive_play(
    colors: tuple[int, ...], size: int, p: int, color: int
) -> tuple[tuple[int, ...], set[int]] | None:
    """(new_colors, captured_points) or None if occupied/suicide."""
    if colors[p] != 0:
        return None
    new = list(colors)
    new[p] = color
    captured: set[int] = set()
    for q in naive_neighbours(p, size):
        if new[q] == 3 - color:
            stones, libs = naive_group(new, q, size)
            if not libs:
                captured |= stones
    for s in captured:
        new[s] = 0
    _, libs = naive_group(new, p, size)
    if not libs:
        return None
    return tuple(new), captured


class NaiveGame:
    def __init__(self, size: int, superko: bool) -> None:
        self.size = size
        self.superko = superko
        self.colors = tuple([0] * (size * size))
        self.to_play = 1
        self.history = {self.colors}
        self.ko = -1
        self.captures = {1: 0, 2: 0}

    def legal_mask(self) -> np.ndarray:
        n = self.size * self.size
        mask = np.zeros(n + 1, dtype=np.int8)
        mask[n] = 1
        for p in range(n):
            result = naive_play(self.colors, self.size, p, self.to_play)
            if result is None:
                continue
            if self.superko and result[0] in self.history:
                continue
            if not self.superko and p == self.ko:
                continue
            mask[p] = 1
        return mask

    def play(self, action: int) -> None:
        n = self.size * self.size
        if action == n:
            self.ko = -1
        else:
            result = naive_play(self.colors, self.size, action, self.to_play)
            assert result is not None
            new, captured = result
            self.captures[self.to_play] += len(captured)
            stones, libs = naive_group(list(new), action, self.size)
            if len(captured) == 1 and len(stones) == 1 and libs == captured:
                self.ko = next(iter(captured))
            else:
                self.ko = -1
            self.colors = new
            self.history.add(new)
        self.to_play = 3 - self.to_play

    def board_string(self) -> str:
        return "".join(".XO"[c] for c in self.colors)


@pytest.mark.parametrize("size,superko", [(4, True), (5, True), (5, False), (7, True)])
def test_mask_matches_naive_reference(size: int, superko: bool) -> None:
    rnd = random.Random(size * 10 + superko)
    games = 8 if size <= 5 else 3
    for _ in range(games):
        env = GoEnv(size, superko=superko, history_length=2)
        result: Any = env.reset()
        ref = NaiveGame(size, superko)
        previous = None
        while result.info["result"] == "*":
            mask = result.info["legal_action_mask"]
            np.testing.assert_array_equal(mask, ref.legal_mask())
            np.testing.assert_array_equal(env.action_mask(), mask)
            assert env.board_string() == ref.board_string()
            assert result.info["captures"] == {"black": ref.captures[1], "white": ref.captures[2]}
            # Observation planes against the reference history.
            obs = result.observation
            me = 1 if result.info["to_play"] == "black" else 2
            cur = np.array(ref.colors).reshape(size, size)
            np.testing.assert_array_equal(obs[0], cur == me)
            np.testing.assert_array_equal(obs[2], cur == 3 - me)
            if previous is not None:
                np.testing.assert_array_equal(obs[1], previous == me)
                np.testing.assert_array_equal(obs[3], previous == 3 - me)
            legal = np.flatnonzero(mask)
            stones = legal[legal < size * size]
            if len(stones) and rnd.random() > 0.05:
                action = int(rnd.choice(stones))
            else:
                action = size * size
            previous = cur
            ref.play(action)
            result = env.step(action)
        assert result.done


# ----------------------------------------------------------------------
# API basics
# ----------------------------------------------------------------------


def test_spaces_and_check_env() -> None:
    env = GoEnv(9)
    assert env.action_space == sekai.Discrete(82)
    assert isinstance(env.observation_space, sekai.Box)
    assert env.observation_space.shape == (17, 9, 9)
    assert env.observation_space.dtype == np.float32
    assert sekai.check_env(GoEnv(9)) == []
    assert sekai.check_env(GoEnv(5), n_steps=400, check_determinism=True) == []
    assert GoEnv(19, history_length=1).observation_space.shape == (3, 19, 19)


def test_sekai_conventions() -> None:
    env = GoEnv(9)
    assert GoEnv.sides == GoEnv.players == ("black", "white")
    assert env.metadata["render_fps"] == 4
    result = env.reset(seed=0, options={"moves": ["E5"]})
    np.testing.assert_array_equal(env.action_mask(), result.info["legal_action_mask"])
    assert env.action_mask().dtype == np.int8
    env.action_mask()[:] = 0  # a copy: the env is not affected
    assert env.action_mask().sum() == 81
    # Errors: each game's own, under the shared games errors and sekai's.
    assert issubclass(IllegalMove, game_errors.IllegalMove)
    assert issubclass(IllegalMove, sekai.InvalidAction)
    assert issubclass(IllegalMove, GoEnvError)
    assert issubclass(GameOver, game_errors.GameOver)
    assert issubclass(GoEnvError, game_errors.GameError)
    assert issubclass(InvalidVertex, ValueError)
    with pytest.raises(game_errors.IllegalMove) as excinfo:
        env.step(40)
    import pickle

    copy = pickle.loads(pickle.dumps(excinfo.value))
    assert (copy.action, copy.vertex, copy.reason) == (40, "E5", "occupied")
    assert str(copy) == str(excinfo.value)


def test_reset_info() -> None:
    env = GoEnv(9)
    result = env.reset(seed=0)
    info = result.info
    assert list(info) == INFO_KEYS
    assert info["to_play"] == "black" and env.to_play == "black"
    assert info["last_move"] is None and info["ply"] == 0 and info["passes"] == 0
    assert info["captures"] == {"black": 0, "white": 0}
    assert info["result"] == "*" and info["winner"] is None
    assert info["legal_action_mask"].dtype == np.int8
    assert int(info["legal_action_mask"].sum()) == 82
    assert info["legal_actions"].dtype == np.int64 and len(info["legal_actions"]) == 82
    assert env.board_string() == "." * 81
    assert not env.done and env.outcome is None


def test_step_updates_state_and_info() -> None:
    env = GoEnv(9)
    env.reset()
    result = env.step(env.encode_vertex("E5"))
    assert result.reward == 0.0 and not result.terminated and not result.truncated
    info = result.info
    assert info["to_play"] == "white" and info["last_move"] == "E5" and info["ply"] == 1
    assert info["legal_action_mask"][vertex_to_action("E5", 9)] == 0
    assert env.board_string()[40] == "X"
    result = env.step(81)
    assert result.info["last_move"] == "pass" and result.info["passes"] == 1
    assert env.moves == ["E5", "pass"] and env.ply == 2 and env.consecutive_passes == 1
    env.step(env.encode_vertex("D4"))
    assert env.consecutive_passes == 0
    # NumPy integers are actions too.
    assert env.step(np.int64(0)).info["last_move"] == "A9"


def test_step_errors() -> None:
    env = GoEnv(5)
    with pytest.raises(sekai.ResetRequired):
        env.step(0)
    env.reset()
    with pytest.raises(sekai.InvalidAction):
        env.step(26)
    with pytest.raises(sekai.InvalidAction):
        env.step(-1)
    env.step(0)
    with pytest.raises(IllegalMove) as excinfo:
        env.step(0)
    assert excinfo.value.reason == "occupied"
    assert isinstance(excinfo.value, sekai.InvalidAction)
    assert "A5" in str(excinfo.value)
    assert env.ply == 1  # the failed step changed nothing
    with pytest.raises(ValueError):
        env.reset(options={"nonsense": 1})


def test_vertex_round_trips() -> None:
    for size in (9, 13, 19):
        for action in range(size * size + 1):
            vertex = action_to_vertex(action, size)
            assert vertex_to_action(vertex, size) == action
            assert vertex_to_action(vertex.lower(), size) == action
    assert vertex_to_action("A9", 9) == 0
    assert vertex_to_action("J9", 9) == 8
    assert vertex_to_action("A1", 9) == 72
    assert vertex_to_action("J1", 9) == 80
    assert vertex_to_action("E5", 9) == 40
    assert vertex_to_action("PASS", 9) == 81 and vertex_to_action(" Pass ", 9) == 81
    assert vertex_to_action("T19", 19) == 18 and vertex_to_action("A1", 19) == 342
    assert vertex_to_action("K10", 19) == 9 * 19 + 9
    assert action_to_vertex(361, 19) == "pass"
    assert "I" not in "".join(action_to_vertex(a, 19)[0] for a in range(19))
    for bad in ("I5", "Z1", "A0", "A10", "K1", "", "pass1", "5A", "A-1"):
        with pytest.raises(InvalidVertex):
            vertex_to_action(bad, 9)
    with pytest.raises(ValueError):  # InvalidVertex is a ValueError
        vertex_to_action("A20", 19)
    with pytest.raises(InvalidVertex):
        action_to_vertex(82, 9)
    with pytest.raises(ValueError):
        GoEnv(20)
    with pytest.raises(ValueError):
        GoEnv(1)
    env = GoEnv(13)
    assert env.decode_action(env.encode_vertex("N13")) == "N13"


def test_observation_planes() -> None:
    env = GoEnv(5, history_length=3)
    obs = env.reset().observation
    assert obs.shape == (7, 5, 5) and obs.dtype == np.float32
    assert not obs[:6].any() and obs[6].all()  # black to move
    env.step(vertex_to_action("A5", 5))  # black
    obs = env.step(vertex_to_action("E1", 5)).observation  # white; black to move again
    assert obs[6].all()
    # Planes 0..2: black (to move) now, one ply ago, two plies ago.
    assert obs[0, 0, 0] == 1 and obs[0].sum() == 1
    assert obs[1, 0, 0] == 1 and obs[1].sum() == 1
    assert obs[2].sum() == 0
    # Planes 3..5: white now, one ply ago (before E1), two plies ago.
    assert obs[3, 4, 4] == 1 and obs[3].sum() == 1
    assert obs[4].sum() == 0 and obs[5].sum() == 0
    obs = env.step(vertex_to_action("C3", 5)).observation  # white to move
    assert not obs[6].any()
    assert obs[0, 4, 4] == 1 and obs[0].sum() == 1  # white stones now
    assert obs[3, 0, 0] == 1 and obs[3, 2, 2] == 1 and obs[3].sum() == 2
    assert obs[4].sum() == 1 and obs[5].sum() == 1  # black one and two plies ago
    np.testing.assert_array_equal(env.observe(), obs)
    # The encoder gives the same planes from the positions.
    from sekai.envs.games.go import ObservationEncoder, decode_observation

    history = [np.frombuffer(bytes(c), dtype=np.int8) for c in env._history]
    np.testing.assert_array_equal(ObservationEncoder(5, 3).encode(history, 2), obs)
    colours, to_play = decode_observation(obs)
    assert to_play == 2 and "".join(".XO"[c] for c in colours) == env.board_string()
    with pytest.raises(ValueError):
        GoEnv(5).observe()  # no position yet


def test_clone_is_independent() -> None:
    env = GoEnv(9)
    env.reset(options={"moves": ["E5", "C3"]})
    snapshot = (env.board_string(), env.moves, env.ply, env.legal_action_mask().tolist())
    clone = env.clone()
    clone.step(clone.encode_vertex("G7"))
    clone.step(81)
    clone.step(81)
    assert clone.done and not env.done
    assert (env.board_string(), env.moves, env.ply, env.legal_action_mask().tolist()) == snapshot
    np.testing.assert_array_equal(clone.clone().observe(), clone.observe())
    assert clone.board is not env.board and clone.board.geo is env.board.geo
    # The original moves on, or starts again, without touching the clone.
    other = env.clone()
    env.step(env.encode_vertex("D4"))
    env.reset(options={"handicap": 2})
    assert other.board_string() == snapshot[0] and other.moves == snapshot[1]
    np.testing.assert_array_equal(other.legal_action_mask(), np.array(snapshot[3], np.int8))
    # Same moves from a clone and from a replay give the same game.
    fresh = replay(9, ["E5", "C3", "D4"])
    other.step(other.encode_vertex("D4"))
    np.testing.assert_array_equal(fresh.observe(), other.observe())
    np.testing.assert_array_equal(fresh.legal_action_mask(), other.legal_action_mask())


def test_snapshot_restore() -> None:
    env = GoEnv(9)
    env.reset(seed=0, options={"moves": ["E5", "C3"]})
    snap = env.snapshot()
    before = env.step(env.encode_vertex("D4"))
    env.step(81)
    env.restore(snap)
    after = env.step(env.encode_vertex("D4"))
    np.testing.assert_array_equal(before.observation, after.observation)
    np.testing.assert_array_equal(before.info["legal_action_mask"], after.info["legal_action_mask"])
    assert env.moves == ["E5", "C3", "D4"]


def test_moves_option_replays_a_game() -> None:
    moves = ["E5", "E4", "D4", "D5", "F4", "pass", "E3", "F5", "E6"]
    stepped = GoEnv(9)
    stepped.reset()
    for move in moves:
        stepped.step(stepped.encode_vertex(move))
    replayed = GoEnv(9)
    result = replayed.reset(options={"moves": moves})
    assert replayed.board_string() == stepped.board_string()
    assert replayed.moves == stepped.moves == moves
    assert replayed.captures == stepped.captures
    assert result.info["ply"] == len(moves) and result.info["last_move"] == "E6"
    np.testing.assert_array_equal(result.observation, stepped.observe())
    np.testing.assert_array_equal(result.info["legal_action_mask"], stepped.legal_action_mask())
    # Integers work too, and an illegal replay raises.
    assert GoEnv(9).reset(options={"moves": [40, 81]}).info["passes"] == 1
    with pytest.raises(IllegalMove):
        GoEnv(9).reset(options={"moves": ["E5", "E5"]})
    with pytest.raises(GameOver):
        GoEnv(9).reset(options={"moves": ["pass", "pass", "E5"]})
    broken = GoEnv(9)
    broken.reset()
    with pytest.raises(IllegalMove):
        broken.reset(options={"moves": ["E5", "E5"]})
    with pytest.raises(sekai.ResetRequired):  # a failed reset leaves no usable game
        broken.step(0)


def test_seed_reproducibility() -> None:
    def play(seed: int) -> list[str]:
        env = GoEnv(9)
        result: Any = env.reset(seed=seed)
        policy = RandomPolicy(seed=seed)
        while not env.done:
            result = env.step(policy.act(result.observation, result.info))
        return env.moves

    assert play(3) == play(3)
    assert play(3) != play(4)
    env = GoEnv(9)
    np.testing.assert_array_equal(env.reset(seed=1).observation, env.reset(seed=1).observation)


@pytest.mark.parametrize("size", [9, 13, 19])
def test_registered(size: int) -> None:
    env = sekai.make(f"Go{size}x{size}-v0")
    assert isinstance(env.unwrapped, GoEnv)
    assert env.unwrapped.size == size
    assert env.action_space.n == size * size + 1
    assert env.observation_space.shape == (17, size, size)
    result = env.reset(seed=0)
    assert result.info["to_play"] == "black"
    assert sekai.make("Go9x9-v0", komi=6.5).unwrapped.komi == 6.5
    if size == 9:
        assert sekai.check_env(sekai.make("Go9x9-v0"), n_steps=300) == []


# ----------------------------------------------------------------------
# Ko and positional superko
# ----------------------------------------------------------------------


KO_ROWS = [
    ".XO..",
    "XO.O.",
    ".XO..",
    ".....",
    ".....",
]


def test_simple_ko_in_env() -> None:
    env = make_env(KO_ROWS)
    result = env.step(env.encode_vertex("C4"))  # black takes the ko
    b4 = env.encode_vertex("B4")
    assert env.ko_point == "B4"
    assert result.info["legal_action_mask"][b4] == 0
    assert result.info["captures"] == {"black": 1, "white": 0}
    with pytest.raises(IllegalMove) as excinfo:
        env.step(b4)
    assert excinfo.value.reason == "ko"
    env.step(env.encode_vertex("E1"))  # ko threat
    assert env.ko_point is None
    env.step(env.encode_vertex("A1"))  # answer
    assert env.legal_action_mask()[b4] == 1
    env.step(b4)
    assert env.ko_point == "C4"
    for superko in (True, False):
        plain = make_env(KO_ROWS, superko=superko)
        plain.step(plain.encode_vertex("C4"))
        assert plain.legal_action_mask()[b4] == 0


# Three kos (A at row 8, B at row 5, C at row 2). In this position Black can take A,
# White B, Black C, White A, Black B; White retaking C would then recreate the start
# position. Simple ko does not forbid that (the ko point is B's), positional superko does.
TRIPLE_KO_ROWS = [
    ".XO......",
    "XO.O.....",
    ".XO......",
    ".XO......",
    "X.XO.....",
    ".XO......",
    ".XO......",
    "XO.O.....",
    ".XO......",
]
TRIPLE_KO_CYCLE = ["C8", "B5", "C2", "B8", "C5"]


def test_positional_superko_beyond_simple_ko() -> None:
    env = make_env(TRIPLE_KO_ROWS)
    start = env.board_string()
    for move in TRIPLE_KO_CYCLE:
        env.step(env.encode_vertex(move))
    b2 = env.encode_vertex("B2")
    assert env.ko_point == "B5"  # simple ko only forbids B5, so it would allow B2...
    assert env.board.is_legal(b2)
    assert env.legal_action_mask()[b2] == 0  # ...positional superko does not
    assert not env.is_legal(b2)
    with pytest.raises(IllegalMove) as excinfo:
        env.step(b2)
    assert excinfo.value.reason == "superko"
    # The check is exact: the forbidden move would recreate the start position.
    assert env.board.colors_after(b2) == bytes(".XO".index(ch) for ch in start)

    simple = make_env(TRIPLE_KO_ROWS, superko=False)
    for move in TRIPLE_KO_CYCLE:
        simple.step(simple.encode_vertex(move))
    assert simple.legal_action_mask()[b2] == 1
    simple.step(b2)
    assert simple.board_string() == start


def test_superko_counts_the_starting_position_and_uses_hashes() -> None:
    env = make_env(TRIPLE_KO_ROWS)
    assert env.board.hash in env._seen
    seen_before = len(env._seen)
    env.step(env.encode_vertex("C8"))
    env.step(81)  # a pass repeats the position; nothing new is recorded
    assert len(env._seen) == seen_before + 1


def test_superko_forbids_a_repeat_that_captures_nothing() -> None:
    # White C5 takes the two black stones A5 B5, Black B5 takes C5 back, White passes:
    # Black A5 now captures nothing but would recreate the starting position.
    rows = [
        "XX.X.",
        "OOX..",
        ".....",
        ".....",
        ".....",
    ]
    for superko in (True, False):
        env = make_env(rows, to_play="white", superko=superko)
        start = env.board_string()
        env.step(env.encode_vertex("C5"))
        env.step(env.encode_vertex("B5"))
        env.step(25)
        assert env.captures == {"black": 1, "white": 2}
        a5 = env.encode_vertex("A5")
        assert env.board.is_legal(a5) and env.board.captured_by(a5) == 0
        assert env.legal_action_mask()[a5] == (0 if superko else 1)
        if superko:
            with pytest.raises(IllegalMove) as excinfo:
                env.step(a5)
            assert excinfo.value.reason == "superko"
        else:
            env.step(a5)
            assert env.board_string() == start


# ----------------------------------------------------------------------
# Passing, scoring, truncation, handicap
# ----------------------------------------------------------------------


def test_two_passes_end_the_game_with_tromp_taylor_score() -> None:
    env = GoEnv(9, komi=7.5)
    env.reset()
    env.step(env.encode_vertex("E5"))
    first = env.step(81)
    assert not first.done and first.reward == 0.0
    last = env.step(81)
    assert last.terminated and not last.truncated
    assert last.reward == 1.0  # black, who passed last, wins B+73.5
    assert last.info["result"] == "B+73.5" and last.info["winner"] == "black"
    assert last.info["score"] == {"black": 81.0, "white": 7.5}
    assert last.info["termination"] == "two passes"
    assert list(last.info) == [*INFO_KEYS, "score", "termination"]
    assert env.done and env.outcome == GoResult("black", 81.0, 7.5, 73.5, "B+73.5", "two passes")
    assert int(last.info["legal_action_mask"].sum()) == 0
    with pytest.raises(GameOver):
        env.step(81)


def test_white_wins_on_komi_and_reward_sign() -> None:
    env = make_env(
        [
            "..XO.",
            "..XO.",
            "..XO.",
            "..XO.",
            "..XO.",
        ],
        to_play="white",
        komi=5.5,
    )
    assert env.step(25).reward == 0.0  # white passes
    result = env.step(25)  # black passes and loses: 15 vs 10 + 5.5
    assert result.info["result"] == "W+0.5"
    assert result.reward == -1.0
    assert result.info["score"] == {"black": 15.0, "white": 15.5}


def test_integer_komi_allows_draws() -> None:
    env = GoEnv(5, komi=0)
    env.reset()
    env.step(25)
    result = env.step(25)
    assert result.info["result"] == "Draw" and result.info["winner"] is None
    assert result.reward == 0.0 and result.terminated
    env = make_env(
        [
            "..XO.",
            "..XO.",
            "..XO.",
            "..XO.",
            "..XO.",
        ],
        komi=5,
    )
    env.step(25)
    assert env.step(25).info["result"] == "Draw"
    assert env.outcome is not None and env.outcome.margin == 0.0


def test_komi_option_is_per_episode() -> None:
    env = GoEnv(9, komi=7.5)
    assert env.reset(options={"komi": 0.5}).info["komi"] == 0.5
    assert env.komi == 0.5
    assert env.reset().info["komi"] == 7.5
    with pytest.raises(ValueError):
        env.reset(options={"komi": "lots"})


def test_max_moves_truncates_and_scores() -> None:
    env = GoEnv(5, max_moves=3)
    env.reset()
    env.step(vertex_to_action("C3", 5))
    env.step(vertex_to_action("A1", 5))
    result = env.step(vertex_to_action("E5", 5))
    assert result.truncated and not result.terminated
    assert result.info["termination"] == "max moves"
    assert env.outcome is not None and env.outcome.reason == "max moves"
    assert result.info["result"] == "W+6.5"  # 2 vs 1 + 7.5 (no territory: all mixed)
    assert result.reward == -1.0
    assert GoEnv(9).max_moves == 243


def test_handicap_placement() -> None:
    env = GoEnv(19)
    result = env.reset(options={"handicap": 4})
    assert result.info["to_play"] == "white"
    assert env.handicap == 4
    assert env.handicap_stones == ["D4", "Q16", "D16", "Q4"]
    board = env.board_string()
    for vertex in ("D4", "Q16", "D16", "Q4"):
        assert board[env.encode_vertex(vertex)] == "X"
    assert board.count("X") == 4 and board.count("O") == 0
    assert result.observation[-1].sum() == 0  # white to move
    env9 = GoEnv(9)
    env9.reset(options={"handicap": 9})
    assert env9.board_string().count("X") == 9
    assert env9.board_string()[40] == "X"
    with pytest.raises(ValueError):
        GoEnv(7).reset(options={"handicap": 2})
    with pytest.raises(ValueError):
        GoEnv(9).reset(options={"handicap": 1})
    with pytest.raises(ValueError):
        GoEnv(9).reset(options={"handicap": 10})
    assert GoEnv(9).reset(options={"handicap": 0}).info["to_play"] == "black"


def test_board_option_and_captures_property() -> None:
    env = make_env(
        [
            ".....",
            "..X..",
            ".XO..",
            "..X..",
            ".....",
        ]
    )
    env.step(env.encode_vertex("D3"))
    assert env.captures == {"black": 1, "white": 0}
    assert env.board_string() == rows_to_board([".....", "..X..", ".X.X.", "..X..", "....."])
    with pytest.raises(ValueError):
        GoEnv(9).reset(options={"board": "." * 25})
    with pytest.raises(ValueError):
        GoEnv(5).reset(options={"to_play": "white"})


def test_reset_mid_game_starts_afresh() -> None:
    env = GoEnv(5)
    env.reset(seed=0)
    for move in ("C3", "C4", "B4", "D4", "B3"):
        env.step(env.encode_vertex(move))
    env.legal_action_mask()  # computed and kept for this position
    rows = [".XO..", "XO.O.", ".XO..", ".....", "....."]
    result = env.reset(options={"board": rows_to_board(rows)})
    fresh = make_env(rows)
    np.testing.assert_array_equal(result.info["legal_action_mask"], fresh.legal_action_mask())
    np.testing.assert_array_equal(result.observation, fresh.observe())
    assert env.moves == [] and env.captures == {"black": 0, "white": 0}


# ----------------------------------------------------------------------
# Against a fixed opponent
# ----------------------------------------------------------------------


@pytest.mark.parametrize("agent_color", ["black", "white", "random"])
def test_opponent_wrapper(agent_color: str) -> None:
    env = OpponentWrapper(GoEnv(5), RandomPolicy(seed=1), agent_color=agent_color)
    agent = RandomPolicy(seed=2)
    colours = set()
    for seed in range(6):
        result: Any = env.reset(seed=seed)
        colour = result.info["agent_color"]
        colours.add(colour)
        if agent_color != "random":
            assert colour == agent_color
        total = 0.0
        while True:
            assert result.info["to_play"] == colour
            result = env.step(agent.act(result.observation, result.info))
            total += result.reward
            if result.done:
                break
        winner = result.info["winner"]
        expected = 0.0 if winner is None else (1.0 if winner == colour else -1.0)
        assert total == result.reward == expected
    if agent_color == "random":
        assert colours == {"black", "white"}
    else:
        assert colours == {agent_color}


def test_opponent_wrapper_moves_first_for_white_agent() -> None:
    env = OpponentWrapper(GoEnv(9), RandomPolicy(seed=0), agent_color="white")
    result = env.reset(seed=0)
    assert result.info["to_play"] == "white" and result.info["ply"] == 1
    assert result.info["opponent_move"] == env.unwrapped.moves[0]
