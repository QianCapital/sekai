"""Chess: the rules engine, the encodings, the environment, PGN and the renderers."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pytest

import sekai
from sekai.envs.games import OpponentWrapper, RandomPolicy
from sekai.envs.games.chess import (
    BLACK,
    NUM_ACTIONS,
    NUM_VALID_ACTIONS,
    QUEEN,
    WHITE,
    AmbiguousMoveError,
    Board,
    ChessEnv,
    ChessRenderer,
    ChessTextRenderer,
    GameOver,
    IllegalMove,
    IllegalMoveError,
    InvalidMoveError,
    Move,
    ObservationEncoder,
    decode_action,
    encode_move,
    env_pgn,
    legal_action_mask,
    legal_actions,
    to_pgn,
)
from sekai.envs.games.chess.board import STARTING_FEN

# ---------------------------------------------------------------------------
# The rules engine
# ---------------------------------------------------------------------------

KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"

# Published perft counts (chessprogramming.org), which cover castling, en passant,
# promotions, pins and checks.
PERFT = [
    (STARTING_FEN, [20, 400, 8902]),
    (KIWIPETE, [48, 2039]),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", [14, 191, 2812]),
    ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", [6, 264, 9467]),
    ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", [44, 1486]),
    ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", [46, 2079]),
]


def perft(board: Board, depth: int) -> int:
    moves = board.legal_move_codes()
    if depth == 1:
        return len(moves)
    total = 0
    for m in list(moves):
        board.push_code(m)
        total += perft(board, depth - 1)
        board.pop_code()
    return total


@pytest.mark.parametrize(("fen", "counts"), PERFT)
def test_perft(fen: str, counts: list[int]) -> None:
    board = Board(fen)
    for depth, expected in enumerate(counts, 1):
        assert perft(board, depth) == expected
    assert board.fen() == fen  # push and pop restore everything


# (fen, (is_valid, fen()) or None when parsing raises ValueError), as python-chess gives.
FEN_CASES: list[tuple[str, tuple[bool, str] | None]] = [
    ("8/8/8/8/8/8/8/8 w - - 0 1", (False, "8/8/8/8/8/8/8/8 w - - 0 1")),
    ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", (True, "4k3/8/8/8/8/8/8/4K3 w - - 0 1")),
    ("4k3/8/8/8/8/8/8/4K2R w K - 0 1", (True, "4k3/8/8/8/8/8/8/4K2R w K - 0 1")),
    ("4k3/8/8/8/8/8/8/4K1R1 w K - 0 1", (False, "4k3/8/8/8/8/8/8/4K1R1 w - - 0 1")),
    ("4k3/8/8/8/8/8/8/5K1R w K - 0 1", (False, "4k3/8/8/8/8/8/8/5K1R w - - 0 1")),
    ("r3k2r/8/8/8/8/8/8/R3K2R w AHah - 0 1", (True, "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")),
    ("r3k2r/8/8/8/8/8/8/R3K2R w BHah - 0 1", (False, "r3k2r/8/8/8/8/8/8/R3K2R w Kkq - 0 1")),
    ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", (True, "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")),
    ("4k3/8/8/3pP3/8/8/8/4K3 w - e6 0 1", (False, "4k3/8/8/3pP3/8/8/8/4K3 w - - 0 1")),
    ("4k3/8/8/3p4/4P3/8/8/4K3 w - d6 0 1", (True, "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")),
    ("4kq2/8/8/8/8/8/8/4K3 w - - 0 1", (True, "4kq2/8/8/8/8/8/8/4K3 w - - 0 1")),
    ("8/8/8/8/8/8/8/4K2k w - - 0 1", (True, "8/8/8/8/8/8/8/4K2k w - - 0 1")),
    (
        "Pnbqkbnr/pppppppp/8/8/8/8/1PPPPPPP/RNBQKBNR w - - 0 1",
        (False, "Pnbqkbnr/pppppppp/8/8/8/8/1PPPPPPP/RNBQKBNR w - - 0 1"),
    ),
    ("4k3/8/8/8/8/8/PPPPPPPP/P3K3 w - - 0 1", (False, "4k3/8/8/8/8/8/PPPPPPPP/P3K3 w - - 0 1")),
    ("8/8/8/2k5/3Pp3/8/8/4K3 b - d3 0 1", (True, "8/8/8/2k5/3Pp3/8/8/4K3 b - d3 0 1")),
    ("8/8/8/8/k2Pp2Q/8/8/4K3 b - d3 0 1", (True, "8/8/8/8/k2Pp2Q/8/8/4K3 b - - 0 1")),
    ("4k3/8/8/8/8/8/8/4K3 w - - 0 0", (True, "4k3/8/8/8/8/8/8/4K3 w - - 0 1")),
    ("4k3/8/8/8/8/8/8/4K3", (True, "4k3/8/8/8/8/8/8/4K3 w - - 0 1")),
    ("4k3/8/8/8/8/8/8/4KQ~2 w - - 0 1", (True, "4k3/8/8/8/8/8/8/4KQ2 w - - 0 1")),
    ("4k3/8/8/8/8/8/8/4K3 w - - -1 1", None),
    ("4k3/8/8/8/8/8/8/4K3 x", None),
    ("4k3/8/8/8/8/8/8/4K3 w kQ", None),
    ("4k3/8/8/8/8/8/8/44K3 w - - 0 1", None),
    ("4k3/8/8/8/8/8/8/4K3 w - - 0 1 extra", None),
    ("4k3/9/8/8/8/8/8/4K3 w - - 0 1", None),
    ("4k3/8/8/8/8/8/8/4K3 w - z9 0 1", None),
    ("4k3/8/8/8/8/8/8/4K3~ w - - 0 1", None),
]


@pytest.mark.parametrize(("fen", "expected"), FEN_CASES)
def test_fen_parsing_and_validity(fen: str, expected: tuple[bool, str] | None) -> None:
    if expected is None:
        with pytest.raises(ValueError):
            Board(fen)
        return
    board = Board(fen)
    assert (board.is_valid(), board.fen()) == expected


# (fen, {uci: san}, {san: uci or the error parsing raises}), as python-chess gives them.
SAN_CASES: list[tuple[str, dict[str, str], dict[str, Any]]] = [
    (
        "r3k2r/1P6/8/3Np3/8/2N5/8/R3K2R w KQkq - 0 1",
        {
            "b7b8q": "b8=Q+",
            "b7a8n": "bxa8=N",
            "e1g1": "O-O",
            "e1c1": "O-O-O",
            "c3b5": "Nb5",
            "d5b4": "Nb4",
            "a1a8": "Rxa8+",
        },
        {
            "0-0": "e1g1",
            "Nb5": "c3b5",
            "Ke3": IllegalMoveError,
            "hello": InvalidMoveError,
            "b7b8": IllegalMoveError,
        },
    ),
    (
        "4k3/8/8/8/8/8/4K3/R6R w - - 0 1",
        {"a1d1": "Rad1", "h1d1": "Rhd1"},
        {"Rd1": AmbiguousMoveError, "Rad1": "a1d1"},
    ),
    (
        "4k3/8/8/R7/8/8/4K3/R7 w - - 0 1",
        {"a1a3": "R1a3", "a5a3": "R5a3"},
        {"Ra3": AmbiguousMoveError, "R1a3": "a1a3"},
    ),
    (
        "4k3/8/8/8/2N3N1/8/2N5/4K3 w - - 0 1",
        {"c2e3": "N2e3", "c4e3": "Nc4e3", "g4e3": "Nge3"},
        {"Ne3": AmbiguousMoveError, "Nc2e3": "c2e3", "Nce3": AmbiguousMoveError},
    ),
    ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", {"e5d6": "exd6"}, {"exd6": "e5d6", "ed6": "e5d6"}),
    ("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", {"a1a8": "Ra8#"}, {"Ra8#": "a1a8"}),
]


@pytest.mark.parametrize(("fen", "written", "parsed"), SAN_CASES)
def test_san_writing_and_parsing(fen: str, written: dict[str, str], parsed: dict[str, Any]) -> None:
    board = Board(fen)
    for uci, san in written.items():
        assert board.san(Move.from_uci(uci)) == san
    for san, expected in parsed.items():
        if isinstance(expected, str):
            assert board.parse_san(san) == Move.from_uci(expected)
        else:
            with pytest.raises(expected):
                board.parse_san(san)
    assert board.fen() == fen


def test_outcomes_follow_the_rules() -> None:
    assert Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").is_stalemate()
    assert Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1").is_valid()
    assert Board("8/8/8/4k3/8/8/8/4KB2 w - - 0 1").is_insufficient_material()
    # Bishops on squares of one colour are not enough; on both colours they could mate.
    assert Board("8/8/8/4k3/8/8/8/3BKB2 w - - 0 1").is_insufficient_material()
    assert not Board("8/8/8/4k3/8/8/8/2B1KB2 w - - 0 1").is_insufficient_material()
    # A lone knight cannot win, unless the opponent has pieces to self-mate with.
    assert Board("8/8/8/4k3/8/8/8/4KN2 w - - 0 1").is_insufficient_material()
    assert not Board("8/8/8/4k3/8/8/4r3/4KN2 w - - 0 1").has_insufficient_material(WHITE)


def test_castling_rights_follow_rook_and_king_moves() -> None:
    board = Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    board.push(Move.from_uci("h1h8"))  # the rook captures the other rook: both rights go
    assert board.castling_fen() == "Qq"
    board.pop()
    board.push(Move.from_uci("e1e2"))
    assert board.castling_fen() == "kq"


# ---------------------------------------------------------------------------
# Encodings
# ---------------------------------------------------------------------------


def test_action_encoding_round_trips_over_random_games() -> None:
    assert NUM_ACTIONS == 4672 and NUM_VALID_ACTIONS == 1858
    rng = np.random.default_rng(0)
    for _ in range(20):
        board = Board()
        for _ in range(80):
            moves = board.legal_moves
            if not moves:
                break
            mask = legal_action_mask(board)
            actions = legal_actions(board)
            assert sorted(actions.tolist()) == np.flatnonzero(mask).tolist()
            for move in moves:
                action = encode_move(move, board.turn)
                assert mask[action] == 1
                assert decode_action(action, board) == move
            board.push(moves[int(rng.integers(len(moves)))])


def test_moves_are_encoded_from_the_movers_side() -> None:
    assert encode_move(Move.from_uci("e2e4"), WHITE) == encode_move(Move.from_uci("e7e5"), BLACK)
    queen = encode_move(Move.from_uci("a7a8q"), WHITE)
    assert queen == encode_move(Move.from_uci("a7a8"), WHITE)  # queen promotions are implicit
    assert encode_move(Move.from_uci("a7a8n"), WHITE) != queen
    with pytest.raises(ValueError):
        encode_move(Move.null())


def test_observation_planes() -> None:
    encoder = ObservationEncoder(8)
    obs = encoder.encode_board(Board())
    assert obs.shape == (119, 8, 8) and obs.dtype == np.float32
    assert obs[0, 1].sum() == 8 and obs[6, 6].sum() == 8  # our pawns, their pawns
    assert obs[14:112].sum() == 0  # no history yet
    assert (obs[112] == 1).all() and (obs[114:118] == 1).all()  # White to move, all castling
    flipped = encoder.encode_board(
        Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
    )
    np.testing.assert_array_equal(flipped[:12], obs[:12])  # Black sees the same from its side
    assert flipped[112].sum() == 0


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------


def _feed(h: Any, result: Any) -> None:
    h.update(np.ascontiguousarray(result.observation).tobytes())
    info = result.info
    h.update(np.ascontiguousarray(info["legal_action_mask"]).tobytes())
    h.update(np.ascontiguousarray(info["legal_actions"]).tobytes())
    keys = ("to_play", "fen", "last_move", "ply", "is_check", "result", "winner", "termination")
    h.update(json.dumps([info[k] for k in keys]).encode())
    if hasattr(result, "reward"):
        flags = (float(result.reward), bool(result.terminated), bool(result.truncated))
        h.update(repr(flags).encode())


def _digest(seed: int, steps: int, **kwargs: Any) -> str:
    env = ChessEnv(**kwargs)
    rng = np.random.default_rng(seed)
    h = hashlib.sha256()
    result: Any = env.reset(seed=seed)
    _feed(h, result)
    for _ in range(steps):
        if result.done:
            result = env.reset()
            _feed(h, result)
            continue
        result = env.step(int(rng.choice(np.flatnonzero(result.info["legal_action_mask"]))))
        _feed(h, result)
    return h.hexdigest()


# Golden digests of random games: every observation, mask, reward, flag and info value.
GOLDEN = [
    (0, 400, {}, "944b4758ce3de1065638e2d36252e800e9c9647d2c64232e02f69bf55a13aae1"),
    (
        1,
        400,
        {"history_length": 1, "claim_draw": False},
        "2f5abae844db59187e20e259bf9e16b144c137bcf62dc0261d10cfb54cbaf749",
    ),
    (
        2,
        300,
        {"starting_fen": KIWIPETE},
        "1ae31473b3de2e93456f1fcb5ffed5ea6afebf1b1c6b401dfb9e69dd43621f0a",
    ),
]


@pytest.mark.parametrize(("seed", "steps", "kwargs", "expected"), GOLDEN)
def test_golden_rollouts(seed: int, steps: int, kwargs: dict[str, Any], expected: str) -> None:
    assert _digest(seed, steps, **kwargs) == expected


def test_reset_and_info() -> None:
    env = ChessEnv()
    result = env.reset(seed=0)
    assert result.observation.shape == (119, 8, 8)
    info = result.info
    assert info["legal_actions"].size == 20 and info["legal_action_mask"].dtype == np.int8
    assert info["to_play"] is True and info["result"] == "*" and info["fen"] == STARTING_FEN
    assert info["last_move"] is None and info["ply"] == 0 and info["termination"] is None
    np.testing.assert_array_equal(env.action_mask(), info["legal_action_mask"])
    assert env.start_fen == STARTING_FEN


def test_checkmate_rewards_the_mover() -> None:
    env = ChessEnv()
    env.reset()
    for uci in ["f2f3", "e7e5", "g2g4"]:
        assert env.step(env.encode_move(uci)).reward == 0.0
    result = env.step(env.encode_move("Qh4#"))
    assert result.reward == 1.0 and result.terminated
    assert result.info["termination"] == "CHECKMATE" and result.info["winner"] is False
    assert result.info["result"] == "0-1" and result.info["legal_action_mask"].sum() == 0
    with pytest.raises(GameOver):
        env.step(0)


def test_illegal_actions() -> None:
    env = ChessEnv()
    with pytest.raises(sekai.ResetRequired):
        env.step(0)
    env.reset()
    with pytest.raises(IllegalMove):
        env.step(0)
    with pytest.raises(sekai.InvalidAction):
        env.step(NUM_ACTIONS)
    lose = ChessEnv(illegal_action="lose")
    lose.reset()
    result = lose.step(0)
    assert result.reward == -1.0 and result.terminated
    assert result.info["termination"] == "ILLEGAL_MOVE" and result.info["illegal_action"] == 0
    with pytest.raises(ValueError):
        ChessEnv(illegal_action="ignore")  # type: ignore[arg-type]


def test_repetitions_end_the_game() -> None:
    shuffle = ["g1f3", "g8f6", "f3g1", "f6g8"] * 4
    for claim_draw, ending, plies in (
        (True, "THREEFOLD_REPETITION", 8),
        (False, "FIVEFOLD_REPETITION", 16),
    ):
        env = ChessEnv(claim_draw=claim_draw)
        result: Any = env.reset()
        played = 0
        for uci in shuffle:
            result = env.step(env.encode_move(uci))
            played += 1
            if result.done:
                break
        assert result.info["termination"] == ending and played == plies
        assert result.observation[12].all() and result.observation[13].all()


def test_reset_options_and_clone() -> None:
    env = ChessEnv(starting_fen=KIWIPETE)
    assert env.reset().info["fen"] == KIWIPETE
    assert env.reset(options={"fen": STARTING_FEN}).info["fen"] == STARTING_FEN
    with pytest.raises(ValueError):
        env.reset(options={"fen": "8/8/8/8/8/8/8/8 w - - 0 1"})
    env.reset(options={"fen": STARTING_FEN})
    env.step(env.encode_move("e4"))
    clone = env.clone(full_stack=False)
    clone.step(clone.encode_move("e5"))
    assert env.ply == 1 and clone.ply == 2 and env.turn is BLACK
    np.testing.assert_array_equal(env.observe(), env.clone().observe())
    assert env.decode_action(env.encode_move("e7e5")) == Move.from_uci("e7e5")


def test_reset_mid_game_starts_afresh() -> None:
    env, fresh = ChessEnv(starting_fen=KIWIPETE), ChessEnv()
    env.reset()
    env.step(int(env.legal_actions()[0]))
    for _ in range(2):
        a, b = env.reset(options={"fen": STARTING_FEN}), fresh.reset()
        np.testing.assert_array_equal(a.observation, b.observation)
        np.testing.assert_array_equal(a.info["legal_action_mask"], b.info["legal_action_mask"])
    assert (
        env.step(env.encode_move("e4")).info["fen"]
        == fresh.step(fresh.encode_move("e4")).info["fen"]
    )


def test_history_length_changes_the_planes() -> None:
    env = ChessEnv(history_length=1)
    assert env.reset().observation.shape == (21, 8, 8)
    with pytest.raises(ValueError):
        ChessEnv(history_length=0)


def test_check_env_and_registered_ids() -> None:
    assert sekai.check_env(ChessEnv(), n_steps=60) == []
    env = sekai.make("Chess-v0")
    assert env.spec.max_episode_steps == 512 and isinstance(env.unwrapped, ChessEnv)


def test_chess_vs_random() -> None:
    env = sekai.make("ChessVsRandom-v0")
    colors = set()
    for seed in range(8):
        result: Any = env.reset(seed=seed)
        agent = result.info["agent_color"]
        colors.add(agent)
        assert result.info["to_play"] == agent
        steps = 0
        while not result.done and steps < 20:
            result = env.step(int(np.flatnonzero(result.info["legal_action_mask"])[0]))
            steps += 1
            if not result.done:
                assert result.info["to_play"] == agent and "opponent_move" in result.info
    assert colors == {True, False}
    with pytest.raises(ValueError):
        OpponentWrapper(ChessEnv(), agent_color="green")


def test_opponent_wrapper_rewards_are_the_agents() -> None:
    # Fool's mate played by the opponent: the agent (White) loses.
    class Script:
        def __init__(self) -> None:
            self.moves = iter(["e7e5", "d8h4"])

        def act(self, observation: Any, info: dict[str, Any]) -> int:
            return int(base.encode_move(next(self.moves)))

    base = ChessEnv()
    env = OpponentWrapper(base, Script(), agent_color="white")
    env.reset()
    assert env.step(base.encode_move("f2f3")).reward == 0.0
    result = env.step(base.encode_move("g2g4"))
    assert result.terminated and result.reward == -1.0 and result.info["opponent_move"] == "d8h4"
    random_env = OpponentWrapper(ChessEnv(), RandomPolicy(0), agent_color="black")
    assert random_env.reset(seed=1).info["to_play"] is False


# ---------------------------------------------------------------------------
# PGN and rendering
# ---------------------------------------------------------------------------


def test_pgn() -> None:
    env = ChessEnv()
    env.reset()
    for uci in ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"]:
        env.step(env.encode_move(uci))
    pgn = env_pgn(env, white="A", black="B")
    assert '[White "A"]' in pgn and '[Result "1-0"]' in pgn and '[Termination "checkmate"]' in pgn
    assert pgn.strip().endswith("1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0")
    black_first = to_pgn(
        ["e7e5"], start_fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    )
    assert '[SetUp "1"]' in black_first and "1... e5 *" in black_first
    with pytest.raises(ValueError):
        to_pgn(["e2e5"])
    with pytest.raises(ValueError):
        env_pgn(env.clone(full_stack=False))


@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("orientation", ["white", "black", "mover"])
def test_renderer(theme: str, orientation: Any) -> None:
    renderer = ChessRenderer(240, theme=theme, orientation=orientation)
    env = sekai.make("Chess-v0")
    with pytest.raises(sekai.ResetRequired):
        renderer.render(env)
    env.reset(seed=0)
    first = renderer.render(env)
    assert first.dtype == np.uint8 and first.ndim == 3 and first.shape[1] == 240 + 2 * 8
    env.step(env.unwrapped.encode_move("e2e4"))
    second = renderer.render(env)
    assert second.shape == first.shape and (second != first).any()
    np.testing.assert_array_equal(second, renderer.render(env))  # deterministic


def test_renderer_options_and_errors() -> None:
    env = ChessEnv()
    env.reset()
    plain = ChessRenderer(200, header=False, strips=False, coordinates=False).render(env)
    assert plain.shape[0] < ChessRenderer(200).render(env).shape[0]
    with pytest.raises(TypeError):
        ChessRenderer().render(sekai.make("CartPole-v1"))
    with pytest.raises(ValueError):
        ChessRenderer(orientation="up")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        ChessRenderer(32)


def test_renderer_after_a_time_limit() -> None:
    env = sekai.make("Chess-v0", max_episode_steps=2)
    env.reset(seed=0)
    for move in ("e2e4", "e7e5"):
        env.step(env.unwrapped.encode_move(move))
    renderer = ChessRenderer(240)
    renderer.render(env)
    status = renderer.status(env.unwrapped)
    assert (status.text, status.sub) == ("Draw", "Drawn by the move limit.")
    assert renderer.turn(env.unwrapped) is None
    assert ChessEnv.metadata["render_fps"] == 2


def test_text_renderer() -> None:
    env = ChessEnv()
    env.reset()
    text = ChessTextRenderer().render(env)
    assert text.splitlines()[0] == "8 ♜ ♞ ♝ ♛ ♚ ♝ ♞ ♜" and text.endswith("White to move")
    flipped = ChessTextRenderer(orientation="black", coordinates=False).render(env)
    assert flipped.splitlines()[0] == "♖ ♘ ♗ ♔ ♕ ♗ ♘ ♖"
    assert QUEEN == 5
