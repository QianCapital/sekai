"""XiangqiEnv: the sekai API, info, game-ending rules, actions, observations, clone, replay."""

from __future__ import annotations

import hashlib
import json
import pickle
from typing import Any

import numpy as np
import pytest

import sekai
from sekai.envs.games import OpponentWrapper, RandomPolicy
from sekai.envs.games.xiangqi import (
    NUM_ACTIONS,
    GameOver,
    IllegalMove,
    InvalidFen,
    Position,
    XiangqiEnv,
    XiangqiResult,
    decode_action,
    encode_move,
)
from sekai.envs.games.xiangqi.actions import (
    NUM_VALID_ACTIONS,
    VALID_ACTIONS,
    action_to_move,
    legal_action_mask,
    move_to_action,
    orient_square,
)
from sekai.envs.games.xiangqi.engine import (
    BLACK,
    RED,
    START_FEN,
    mirror_fen,
    move_from_uci,
    move_to_uci,
)
from sekai.envs.games.xiangqi.observations import (
    NUM_CONSTANT_PLANES,
    PLANES_PER_POSITION,
    ObservationEncoder,
    orient_planes,
    position_frames,
    position_planes,
)

INFO_KEYS = {
    "legal_action_mask",
    "legal_actions",
    "to_play",
    "fen",
    "last_move",
    "ply",
    "is_check",
    "result",
    "winner",
    "termination",
}

MATE_IN_ONE = "4k4/R8/9/9/9/9/9/9/9/1R1K5 w - - 0 1"  # b0b9#
STALEMATE_IN_ONE = "3k5/9/8R/9/9/9/9/9/9/4K4 w - - 0 1"  # i7i8 leaves Black without a move
PERPETUAL = "3k5/9/9/9/4R4/9/9/9/9/5K3 w - - 0 1"
PERPETUAL_CYCLE = ["e5d5", "d9e9", "d5e5", "e9d9"]
MIDGAME = "r1ba1a3/4kn3/2n1b4/pNp1p1p1p/4c4/6P2/P1P2R2P/1CcC5/9/2BAKAB2 w - - 0 1"


def play(env: XiangqiEnv, moves: list[str]) -> sekai.StepResult[Any]:
    result = None
    for uci in moves:
        result = env.step(env.encode_move(uci))
    assert result is not None
    return result


def random_action(info: dict[str, Any], rng: np.random.Generator) -> int:
    legal = info["legal_actions"]
    return int(legal[rng.integers(len(legal))])


# ----------------------------------------------------------------------
# sekai API and info
# ----------------------------------------------------------------------


def test_check_env_is_clean() -> None:
    assert sekai.check_env(XiangqiEnv()) == []
    env = XiangqiEnv(history_length=2, illegal_action="lose")
    assert sekai.check_env(env, n_steps=50, check_determinism=True) == []


def test_spaces() -> None:
    env = XiangqiEnv()
    assert env.action_space == sekai.Discrete(NUM_ACTIONS) and NUM_ACTIONS == 8100
    assert isinstance(env.observation_space, sekai.Box)
    assert env.observation_space.shape == (16 * 8 + 3, 10, 9)
    assert env.observation_space.dtype == np.float32
    assert XiangqiEnv(history_length=1).observation_space.shape == (19, 10, 9)


def test_sides_players_and_metadata() -> None:
    assert XiangqiEnv.sides == XiangqiEnv.players == ("red", "black")
    assert XiangqiEnv.metadata["name"] == "Xiangqi-v0"
    assert XiangqiEnv.metadata["render_fps"] > 0
    assert str(XiangqiEnv()) == f"<Xiangqi-v0 history_length=8 fen='{START_FEN}'>"


def test_reset_info_and_determinism() -> None:
    env = XiangqiEnv()
    first = env.reset(seed=7)
    second = env.reset(seed=7)
    np.testing.assert_array_equal(first.observation, second.observation)
    info = first.info
    assert set(info) == INFO_KEYS
    assert info["to_play"] == "red" and info["fen"] == START_FEN
    assert info["ply"] == 0 and info["last_move"] is None and info["is_check"] is False
    assert info["result"] == "*" and info["winner"] is None and info["termination"] is None
    assert info["legal_action_mask"].dtype == np.int8
    assert info["legal_action_mask"].shape == (NUM_ACTIONS,)
    assert int(info["legal_action_mask"].sum()) == 44
    assert info["legal_actions"].dtype == np.int64 and len(info["legal_actions"]) == 44
    np.testing.assert_array_equal(info["legal_actions"], np.flatnonzero(info["legal_action_mask"]))


def test_step_info_and_helpers() -> None:
    env = XiangqiEnv()
    env.reset()
    result = env.step(env.encode_move("h2e2"))
    assert set(result.info) == INFO_KEYS
    assert result.reward == 0.0 and not result.terminated and not result.truncated
    assert result.info["to_play"] == "black" and env.turn == "black"
    assert result.info["last_move"] == "h2e2" and env.last_move == "h2e2"
    assert result.info["ply"] == 1 and env.ply == 1
    assert (
        env.fen
        == result.info["fen"]
        == ("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C2C4/9/RNBAKABNR b - - 1 1")
    )
    assert env.moves == ["h2e2"] and env.initial_fen == START_FEN
    assert env.is_legal("h9g7") and not env.is_legal("h2e2") and not env.is_legal("zz")
    assert env.legal_moves() == sorted(env.legal_moves()) and len(env.legal_moves()) == 45
    assert "h9g7" in env.legal_moves()
    assert env.move_wxf("h9g7") == "H8+7" and env.move_chinese("h9g7") == "馬8進7"
    assert env.decode_action(env.encode_move("h9g7")) == "h9g7"
    assert not env.is_check() and env.outcome is None and not env.done
    assert isinstance(env.position, Position)
    assert "r n b a k a b n r" in env.render()


def test_action_mask_is_the_legal_action_mask() -> None:
    env = XiangqiEnv(starting_fen=MATE_IN_ONE)
    result = env.reset()
    np.testing.assert_array_equal(env.action_mask(), result.info["legal_action_mask"])
    np.testing.assert_array_equal(env.action_mask(), env.legal_action_mask())
    assert sekai.get_action_mask(env) is not None
    env.step(env.encode_move("b0b9"))
    assert env.action_mask().dtype == np.int8 and not env.action_mask().any()


def test_returned_arrays_are_fresh() -> None:
    env = XiangqiEnv()
    result = env.reset()
    result.info["legal_action_mask"][:] = 0  # the caller's arrays are its own
    result.info["legal_actions"][:] = 0
    env.legal_action_mask()[:] = 1
    env.legal_actions()[:] = 0
    assert int(env.legal_action_mask().sum()) == 44
    assert env.legal_actions()[0] != env.legal_actions()[-1]
    assert int(env.step(env.encode_move("h2e2")).info["legal_action_mask"].sum()) == 45
    assert env.observe() is not env.observe()


def test_step_before_reset_and_after_game_over() -> None:
    env = XiangqiEnv(starting_fen=MATE_IN_ONE)
    with pytest.raises(sekai.ResetRequired):
        env.step(0)
    env.reset()
    env.step(env.encode_move("b0b9"))
    with pytest.raises(GameOver):
        env.step(0)
    assert env.legal_moves() == [] and env.legal_actions().size == 0
    assert int(env.legal_action_mask().sum()) == 0 and not env.is_legal("e9e8")


def test_illegal_action_raises_or_loses() -> None:
    env = XiangqiEnv()
    env.reset()
    with pytest.raises(IllegalMove, match="e2e9"):
        env.step(encode_move("e2e9", "red"))
    with pytest.raises(sekai.InvalidAction):
        env.step(NUM_ACTIONS)
    assert env.ply == 0  # nothing changed

    forfeit = XiangqiEnv(illegal_action="lose")
    forfeit.reset()
    result = forfeit.step(encode_move("e2e9", "red"))
    assert result.terminated and result.reward == -1.0
    assert result.info["termination"] == "ILLEGAL_MOVE" and result.info["winner"] == "black"
    assert result.info["illegal_action"] == encode_move("e2e9", "red")
    assert not result.info["legal_action_mask"].any() and result.info["legal_actions"].size == 0
    with pytest.raises(ValueError):
        XiangqiEnv(illegal_action="ignore")  # type: ignore[arg-type]


def test_numpy_actions_are_accepted() -> None:
    env = XiangqiEnv()
    env.reset()
    assert env.step(np.int64(env.encode_move("h2e2"))).info["last_move"] == "h2e2"
    assert env.step(np.intp(env.encode_move("h9g7"))).info["last_move"] == "h9g7"


def test_errors_pickle() -> None:
    env = XiangqiEnv()
    env.reset()
    with pytest.raises(IllegalMove) as raised:
        env.step(encode_move("e2e9", "red"))
    copy = pickle.loads(pickle.dumps(raised.value))
    assert type(copy) is IllegalMove and str(copy) == str(raised.value)
    assert copy.action == raised.value.action and copy.move == "e2e9"


# ----------------------------------------------------------------------
# Game end
# ----------------------------------------------------------------------


def test_checkmate_rewards_the_mover() -> None:
    env = XiangqiEnv(starting_fen=MATE_IN_ONE)
    env.reset()
    result = env.step(env.encode_move("b0b9"))
    assert result.terminated and not result.truncated and result.reward == 1.0
    info = result.info
    assert info["result"] == "1-0" and info["winner"] == "red"
    assert info["termination"] == "CHECKMATE" and info["is_check"]
    assert env.outcome == XiangqiResult(winner="red", termination="CHECKMATE")
    assert env.outcome.result() == "1-0"


def test_black_checkmate() -> None:
    env = XiangqiEnv()
    env.reset(options={"fen": "1r1k5/9/9/9/9/9/9/9/r8/4K4 b - - 0 1"})
    result = env.step(env.encode_move("b9b0"))
    assert result.reward == 1.0 and result.info["result"] == "0-1"
    assert result.info["winner"] == "black" and result.info["termination"] == "CHECKMATE"


def test_stalemate_is_a_loss() -> None:
    env = XiangqiEnv(starting_fen=STALEMATE_IN_ONE)
    env.reset()
    result = env.step(env.encode_move("i7i8"))
    assert result.terminated and result.reward == 1.0
    assert not result.info["is_check"]
    assert result.info["termination"] == "STALEMATE" and result.info["winner"] == "red"
    # A position that is already stalemated ends at reset.
    reset = XiangqiEnv().reset(options={"fen": "3k5/8R/9/9/9/9/9/9/9/4K4 b - - 0 1"})
    assert reset.info["result"] == "1-0" and reset.info["termination"] == "STALEMATE"


def test_threefold_repetition_is_a_draw() -> None:
    env = XiangqiEnv()
    env.reset()
    shuffle = ["b0c2", "b9c7", "c2b0", "c7b9"]
    result = play(env, shuffle)
    assert not result.done and env.position.repetition_count() == 2
    assert result.observation[14].all() and not result.observation[15].any()
    result = play(env, shuffle[:3])
    assert not result.done
    result = env.step(env.encode_move(shuffle[3]))
    assert result.terminated and result.reward == 0.0
    assert result.info["termination"] == "REPETITION" and result.info["result"] == "1/2-1/2"
    assert result.observation[15].all()


def test_perpetual_check_loses() -> None:
    env = XiangqiEnv(starting_fen=PERPETUAL)
    env.reset()
    result = play(env, PERPETUAL_CYCLE * 2)
    # Red checked with every move, Black never: Red loses; Black's last move gets +1.
    assert result.terminated and result.reward == 1.0
    assert result.info["termination"] == "PERPETUAL_CHECK" and result.info["winner"] == "black"


def test_perpetual_check_cycle_closed_by_the_checker() -> None:
    env = XiangqiEnv(starting_fen="3k5/9/9/9/3R5/9/9/9/9/5K3 b - - 0 1")
    env.reset()
    result = play(env, ["d9e9", "d5e5", "e9d9", "e5d5"] * 2)
    assert result.terminated and result.reward == -1.0  # Red's own move completed the cycle
    assert result.info["termination"] == "PERPETUAL_CHECK" and result.info["result"] == "0-1"


def test_no_capture_rule() -> None:
    fen = START_FEN.replace(" 0 1", " 119 60")
    env = XiangqiEnv()
    env.reset(options={"fen": fen})
    result = env.step(env.encode_move("h2e2"))
    assert result.terminated and result.reward == 0.0
    assert result.info["termination"] == "NO_CAPTURE" and result.info["result"] == "1/2-1/2"
    # A capture resets the clock.
    env.reset(options={"fen": fen.replace("119 60", "118 60")})
    result = play(env, ["h2e2", "e6e5"])  # Black's move is the 120th quiet ply
    assert result.info["termination"] == "NO_CAPTURE"
    env.reset(options={"fen": fen.replace("119 60", "118 60")})
    result = play(env, ["b2b9", "a9b9"])  # cannon takes horse, chariot recaptures
    assert not result.done and env.position.halfmove == 0


def test_insufficient_material() -> None:
    env = XiangqiEnv()
    env.reset(options={"fen": "3k5/9/9/9/9/9/9/9/4p4/4K4 w - - 0 1"})
    assert env.is_check()
    result = env.step(env.encode_move("e0e1"))  # the general takes the last attacker
    assert result.terminated and result.reward == 0.0
    assert result.info["termination"] == "INSUFFICIENT_MATERIAL"
    bare = env.reset(options={"fen": "3k1a3/4a4/9/9/9/9/9/4B4/4A4/4KA3 w - - 0 1"})
    assert bare.info["termination"] == "INSUFFICIENT_MATERIAL"


def test_truncation_through_the_registry() -> None:
    env = sekai.make("Xiangqi-v0")
    assert env.spec.max_episode_steps == 400
    assert isinstance(env, sekai.TimeLimit) and isinstance(env.unwrapped, XiangqiEnv)
    result = env.reset(seed=3)
    policy = RandomPolicy(seed=3)
    steps = 0
    while True:
        result = env.step(policy.act(result.observation, result.info))
        steps += 1
        if result.done:
            break
    assert steps <= 400
    assert result.terminated or (result.truncated and steps == 400)


def test_registered() -> None:
    spec = sekai.spec("Xiangqi-v0")
    assert spec.entry_point == "sekai.envs.games.xiangqi.env:XiangqiEnv"
    assert spec.max_episode_steps == 400
    assert sekai.envs.games.XiangqiEnv is XiangqiEnv


# ----------------------------------------------------------------------
# Actions
# ----------------------------------------------------------------------


def test_action_round_trips_for_both_colours() -> None:
    for side, name in ((RED, "red"), (BLACK, "black")):
        for action in range(NUM_ACTIONS):
            uci = decode_action(action, name)
            assert encode_move(uci, name) == action
            assert move_to_action(action_to_move(action, side), side) == action


def test_black_actions_are_rotated_180_degrees() -> None:
    assert orient_square(0, "red") == 0 and orient_square(0, "black") == 89
    assert orient_square(move_from_uci("e9e8") // 90, "black") == 4  # e9 -> mover's e0
    # Red's h2e2 and Black's b7e7 are the same move in the mover's frame.
    assert encode_move("h2e2", "red") == encode_move("b7e7", "black") == 25 * 90 + 22
    assert decode_action(25 * 90 + 22, "black") == "b7e7"
    assert encode_move("h9g7", "black") == encode_move("b0c2", "red")
    with pytest.raises(ValueError):
        decode_action(NUM_ACTIONS, "red")
    with pytest.raises(ValueError):
        encode_move("h2e2", "green")  # type: ignore[arg-type]


def test_valid_action_count() -> None:
    assert 0 < NUM_VALID_ACTIONS < NUM_ACTIONS
    env = XiangqiEnv(history_length=1)
    policy = RandomPolicy(seed=0)
    result = env.reset(seed=0)
    for _ in range(150):
        if env.done:
            result = env.reset()
        assert set(result.info["legal_actions"].tolist()) <= VALID_ACTIONS
        result = env.step(policy.act(result.observation, result.info))


def test_mask_matches_brute_force_legality() -> None:
    env = XiangqiEnv(history_length=1)
    policy = RandomPolicy(seed=11)
    result = env.reset(seed=11)
    for _ in range(120):
        if env.done:
            result = env.reset()
        position = env.position
        mask = result.info["legal_action_mask"]
        expected = np.zeros(NUM_ACTIONS, dtype=np.int8)
        for action in range(NUM_ACTIONS):
            if position.is_legal(action_to_move(action, position.side)):
                expected[action] = 1
        np.testing.assert_array_equal(mask, expected)
        np.testing.assert_array_equal(mask, legal_action_mask(position))
        result = env.step(policy.act(result.observation, result.info))


# ----------------------------------------------------------------------
# Observations
# ----------------------------------------------------------------------


def test_start_observation_planes() -> None:
    env = XiangqiEnv()
    obs = env.reset().observation
    assert obs.shape == (131, 10, 9) and obs.dtype == np.float32
    assert np.min(obs) >= 0.0 and np.max(obs) <= 1.0
    assert obs[0, 0, 4] == 1.0 and obs[0].sum() == 1  # our general on e0
    assert obs[7, 9, 4] == 1.0 and obs[7].sum() == 1  # their general on e9
    assert obs[4].sum() == 2 and obs[4, 0, 0] == obs[4, 0, 8] == 1.0  # our chariots
    assert obs[6].sum() == 5 and obs[6, 3, ::2].sum() == 5  # our soldiers
    assert obs[5, 2, 1] == obs[5, 2, 7] == 1.0  # our cannons
    assert not obs[14:16].any()
    assert not obs[16:128].any()  # history zero-filled
    constants = obs[128:]
    assert constants.shape[0] == NUM_CONSTANT_PLANES
    assert constants[0].all() and not constants[1].any() and not constants[2].any()


def test_black_observation_is_rotated_and_swapped() -> None:
    env = XiangqiEnv(history_length=2)
    env.reset()
    obs = env.step(env.encode_move("h2e2")).observation
    # Black to move: black general e9 appears at mover-frame (rank 0, file 4).
    assert obs[0, 0, 4] == 1.0 and obs[7, 9, 4] == 1.0
    # Red's cannons are now on e2 and b2; rotated: e2 -> (7, 4), b2 -> (7, 7).
    assert obs[7 + 5, 7, 4] == 1.0 and obs[7 + 5, 7, 7] == 1.0 and obs[12].sum() == 2
    # Our (Black's) cannons b7/h7 -> (2, 7) and (2, 1).
    assert obs[5, 2, 7] == 1.0 and obs[5, 2, 1] == 1.0
    # One ply back: the start position, also from Black's point of view.
    previous = obs[PLANES_PER_POSITION : 2 * PLANES_PER_POSITION]
    assert previous[12, 7, 1] == 1.0 and previous[12, 7, 7] == 1.0  # red cannons b2, h2
    constants = obs[2 * PLANES_PER_POSITION :]
    assert not constants[0].any()  # Black to move
    np.testing.assert_allclose(constants[1], 1 / 400)
    np.testing.assert_allclose(constants[2], 1 / 120)


@pytest.mark.parametrize(
    "fen",
    [
        START_FEN,
        "r1ba1a3/4kn3/2n1b4/pNp1p1p1p/4c4/6P2/P1P2R2P/1CcC5/9/2BAKAB2 w - - 3 9",
        "1cbak4/9/n2a5/2p1p3p/5cp2/2n2N3/6PCP/3AB4/2C6/3A1K1N1 w - - 0 20",
    ],
)
def test_colour_flipped_positions_look_identical_to_the_mover(fen: str) -> None:
    red, black = Position(fen), Position(mirror_fen(fen))
    assert black.side == BLACK and mirror_fen(mirror_fen(fen)) == fen
    encoder = ObservationEncoder(1)
    red_obs, black_obs = encoder.encode_position(red), encoder.encode_position(black)
    np.testing.assert_array_equal(red_obs[:PLANES_PER_POSITION], black_obs[:PLANES_PER_POSITION])
    assert red_obs[PLANES_PER_POSITION].all() and not black_obs[PLANES_PER_POSITION].any()
    # ... and have exactly the same legal actions.
    np.testing.assert_array_equal(legal_action_mask(red), legal_action_mask(black))


def test_position_planes_absolute_frame() -> None:
    planes = position_planes(Position(), repetitions=3)
    assert planes.shape == (16, 10, 9)
    assert planes[0, 0, 4] == 1.0 and planes[7, 9, 4] == 1.0
    assert planes[14].all() and planes[15].all()


def test_position_frames_match_the_oriented_planes() -> None:
    position = Position(MIDGAME)
    for repetitions in (1, 2, 3):
        frames = position_frames(position, repetitions)
        assert frames.shape == (2, 16, 10, 9) and not frames.flags.writeable
        planes = position_planes(position, repetitions)
        np.testing.assert_array_equal(frames[0], planes)
        np.testing.assert_array_equal(frames[1], orient_planes(planes, BLACK))


# ----------------------------------------------------------------------
# Clone, replay and resets
# ----------------------------------------------------------------------


def test_clone_is_independent() -> None:
    env = XiangqiEnv()
    env.reset(seed=1)
    play(env, ["h2e2", "h9g7"])
    clone = env.clone()
    before = (env.fen, env.ply, env.moves, env.observe().copy(), env.legal_action_mask().copy())
    result = play(clone, ["h0g2", "i9h9", "c3c4"])
    assert clone.ply == 5 and result.info["fen"] != before[0]
    assert (env.fen, env.ply, env.moves) == before[:3]
    np.testing.assert_array_equal(env.observe(), before[3])
    np.testing.assert_array_equal(env.legal_action_mask(), before[4])
    # The original keeps playing normally, and so does a clone that was reset.
    assert env.step(env.encode_move("h0g2")).info["ply"] == 3
    clone.reset(options={"fen": MATE_IN_ONE})
    assert env.ply == 3 and env.fen != clone.fen


def test_clone_preserves_repetition_history() -> None:
    env = XiangqiEnv()
    env.reset()
    play(env, ["b0c2", "b9c7", "c2b0", "c7b9", "b0c2", "b9c7", "c2b0"])
    clone = env.clone()
    assert clone.step(clone.encode_move("c7b9")).info["termination"] == "REPETITION"
    assert not env.done


def test_clone_steps_like_the_original() -> None:
    rng = np.random.default_rng(4)
    env = XiangqiEnv(history_length=3)
    result = env.reset()
    for _ in range(40):
        if env.done:
            break
        clone = env.clone()
        action = random_action(result.info, rng)
        result, cloned = env.step(action), clone.step(action)
        np.testing.assert_array_equal(result.observation, cloned.observation)
        assert result.info["fen"] == cloned.info["fen"] and result.reward == cloned.reward


def test_snapshot_and_restore() -> None:
    rng = np.random.default_rng(6)
    env = XiangqiEnv()
    result = env.reset()
    for _ in range(10):
        result = env.step(random_action(result.info, rng))
    snapshot = env.snapshot()
    actions, observations = [], []
    for _ in range(10):
        if result.done:
            break
        actions.append(random_action(result.info, rng))
        result = env.step(actions[-1])
        observations.append(result.observation)
    env.restore(snapshot)
    for action, observation in zip(actions, observations):
        np.testing.assert_array_equal(env.step(action).observation, observation)


def test_moves_replay_matches_stepping() -> None:
    moves = ["h2e2", "h9g7", "h0g2", "i9h9", "i0h0", "b9c7", "c3c4", "b7b3"]
    stepped = XiangqiEnv()
    stepped.reset()
    last = play(stepped, moves)
    replayed = XiangqiEnv()
    reset = replayed.reset(options={"moves": moves})
    np.testing.assert_array_equal(reset.observation, last.observation)
    assert reset.info["fen"] == last.info["fen"] and reset.info["ply"] == len(moves)
    assert reset.info["last_move"] == "b7b3" and replayed.moves == moves
    np.testing.assert_array_equal(reset.info["legal_action_mask"], last.info["legal_action_mask"])
    # A space-separated string works too, and so does replaying from a FEN.
    assert replayed.reset(options={"moves": " ".join(moves)}).info["fen"] == last.info["fen"]
    mid = XiangqiEnv().reset(options={"fen": stepped.fen, "moves": ["a0a1"]})
    assert mid.info["last_move"] == "a0a1"


def test_bad_replay_or_fen_leaves_env_unchanged() -> None:
    env = XiangqiEnv()
    env.reset()
    play(env, ["h2e2"])
    with pytest.raises(IllegalMove, match=r"options\['moves'\]\[1\]"):
        env.reset(options={"moves": ["h2e2", "h2e2"]})
    assert env.moves == ["h2e2"] and env.ply == 1
    with pytest.raises(InvalidFen):
        env.reset(options={"fen": "nonsense"})
    with pytest.raises(GameOver):
        env.reset(options={"fen": MATE_IN_ONE, "moves": ["b0b9", "e9e8"]})
    assert env.moves == ["h2e2"]
    assert env.step(env.encode_move("h9g7")).info["ply"] == 2


def test_resets_mid_game_match_a_fresh_env() -> None:
    """One env reused across games and resets behaves like a fresh env for each reset."""
    rng = np.random.default_rng(9)
    env = XiangqiEnv(history_length=4)
    result = env.reset()
    options_list: list[dict[str, Any] | None] = [
        {"fen": MIDGAME},
        None,
        {"moves": ["h2e2", "h9g7", "h0g2"]},
        {"fen": PERPETUAL, "moves": ["e5d5"]},
        {},
    ]
    for options in options_list:
        for _ in range(int(rng.integers(5, 25))):
            if result.done:
                break
            result = env.step(random_action(result.info, rng))
        with pytest.raises(InvalidFen):  # a failed reset first: nothing may change
            env.reset(options={"fen": "4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1"})
        result = env.reset(options=options)
        fresh = XiangqiEnv(history_length=4)
        expected = fresh.reset(options=options)
        np.testing.assert_array_equal(result.observation, expected.observation)
        assert result.info["fen"] == expected.info["fen"]
        np.testing.assert_array_equal(
            result.info["legal_action_mask"], expected.info["legal_action_mask"]
        )
        action = random_action(result.info, rng)
        result, stepped = env.step(action), fresh.step(action)
        np.testing.assert_array_equal(result.observation, stepped.observation)
        assert result.info["legal_actions"].tolist() == stepped.info["legal_actions"].tolist()


def test_starting_fen_and_bad_constructor_fen() -> None:
    env = XiangqiEnv(starting_fen=PERPETUAL)
    assert env.reset().info["fen"] == PERPETUAL
    assert env.reset(options={"fen": None}).info["fen"] == PERPETUAL
    with pytest.raises(InvalidFen):
        XiangqiEnv(starting_fen="4k4/9/9/9/9/9/9/9/9/4K4 w - - 0 1")


def test_random_games_terminate_consistently() -> None:
    env = XiangqiEnv(history_length=1)
    policy = RandomPolicy(seed=5)
    for seed in range(4):
        result = env.reset(seed=seed)
        while not env.done and env.ply < 400:
            result = env.step(policy.act(result.observation, result.info))
        if env.done:
            assert result.info["termination"] is not None
            assert result.info["result"] in ("1-0", "0-1", "1/2-1/2")
            assert move_to_uci(move_from_uci(result.info["last_move"])) == result.info["last_move"]


# ----------------------------------------------------------------------
# Playing against a fixed opponent
# ----------------------------------------------------------------------


@pytest.mark.parametrize("agent_color", ["red", "black", "random"])
def test_opponent_wrapper(agent_color: str) -> None:
    env = OpponentWrapper(XiangqiEnv(history_length=1), RandomPolicy(seed=2), agent_color)
    result = env.reset(seed=2)
    side = result.info["agent_color"]
    assert side in ("red", "black") and result.info["to_play"] == side
    if agent_color != "random":
        assert side == agent_color
    if side == "black":
        assert result.info["ply"] == 1 and "opponent_move" in result.info
    rng = np.random.default_rng(2)
    while not result.done and env.unwrapped.ply < 300:
        result = env.step(random_action(result.info, rng))
        assert result.reward in (-1.0, 0.0, 1.0)
        if not result.done:
            assert result.info["to_play"] == side
            assert result.info["opponent_move"] == result.info["last_move"]
    if result.terminated and result.info["winner"] is not None:
        assert result.reward == (1.0 if result.info["winner"] == side else -1.0)


# ----------------------------------------------------------------------
# Golden digests: seeded rollouts whose every step is pinned
# ----------------------------------------------------------------------


def _update(hasher: Any, observation: Any, info: dict[str, Any], step: Any = None) -> None:
    hasher.update(f"{observation.dtype.str}{observation.shape}".encode())
    hasher.update(observation.tobytes())
    hasher.update(info["legal_action_mask"].tobytes())
    rest = {k: v for k, v in info.items() if k != "legal_action_mask"}
    rest["legal_actions"] = rest["legal_actions"].tolist()
    hasher.update(json.dumps(rest, ensure_ascii=False).encode())
    if step is not None:
        hasher.update(json.dumps(step).encode())


def rollout_digest(
    env_type: Any,
    *,
    seed: int,
    plies: int,
    history_length: int = 8,
    illegal_action: str = "raise",
    options: dict[str, Any] | None = None,
    script: list[str] | None = None,
    illegal_at: int | None = None,
) -> str:
    """sha256 over every observation, mask, reward, flag and info of a rollout.

    Random legal moves from `seed` (or the UCCI moves of `script`), an illegal action at
    ply `illegal_at` if given.
    """
    env = env_type(history_length=history_length, illegal_action=illegal_action)
    result = env.reset(seed=seed, options=options)
    hasher = hashlib.sha256()
    _update(hasher, result.observation, result.info)
    rng = np.random.default_rng(seed)
    for ply in range(plies):
        if result.info["termination"] is not None:
            break
        if script is not None:
            if ply == len(script):
                break
            action = env.encode_move(script[ply])
        elif ply == illegal_at:
            action = int(np.flatnonzero(result.info["legal_action_mask"] == 0)[ply])
        else:
            action = random_action(result.info, rng)
        result = env.step(action)
        step = [result.reward, result.terminated, result.truncated]
        _update(hasher, result.observation, result.info, step)
    return hasher.hexdigest()


GOLDEN_ROLLOUTS: dict[str, dict[str, Any]] = {
    "start-h8": {"seed": 0, "plies": 400},
    "start-h1": {"seed": 1, "plies": 400, "history_length": 1},
    "midgame-h8": {"seed": 2, "plies": 300, "options": {"fen": MIDGAME}},
    "replay-h2": {
        "seed": 3,
        "plies": 200,
        "history_length": 2,
        "options": {"moves": ["h2e2", "h9g7", "h0g2", "i9h9"]},
    },
    "repetition-h8": {
        "seed": 4,
        "plies": 12,
        "script": ["b0c2", "b9c7", "c2b0", "c7b9"] * 3,
    },
    "perpetual-h2": {
        "seed": 5,
        "plies": 8,
        "history_length": 2,
        "options": {"fen": PERPETUAL},
        "script": PERPETUAL_CYCLE * 2,
    },
    "forfeit-h1": {
        "seed": 6,
        "plies": 40,
        "history_length": 1,
        "illegal_action": "lose",
        "illegal_at": 21,
    },
    "no-capture-h8": {
        "seed": 7,
        "plies": 80,
        "options": {"fen": START_FEN.replace(" 0 1", " 100 30")},
    },
    "insufficient-h1": {
        "seed": 7,
        "plies": 120,
        "history_length": 1,
        "options": {"fen": "2bk5/4a4/9/9/9/9/9/4p4/4A4/4K4 w - - 0 1"},
    },
}

GOLDEN_DIGESTS = {
    "forfeit-h1": "dd50132f8afa8eb2b6409310ab78a9e77c6ab6b63dd2569ef4089a9d62c140dc",
    "insufficient-h1": "7915456aa97983f8f1d182fc1ab3f64462ad807e61b6f4c768f2ad027de5713e",
    "midgame-h8": "8e50f59435f70b2f51a36a19c3486c6f3f5337ddecdb391c34f78ad650d0c7c0",
    "no-capture-h8": "7b55a6b8ee4df75a1c70ad11a323d26d9163e30b97c097bd07b5cc5609bf7635",
    "perpetual-h2": "4bb7872f13c1c86f79f18f08cbfc93fe45b5ece1182192c7d37a583ed03da8ee",
    "repetition-h8": "09975b00a00a306453af6522114312185e101b75195e13b03d70552e7c1a55ca",
    "replay-h2": "4c5802fec08967bc6f9b6dc3f87d44ddc4505f0712aca9b28abad2a590d3b71d",
    "start-h1": "59a0ef95c567900a7f58f56c6cbd577c8531822a5ad65fecc0e0d14efd9a8d10",
    "start-h8": "fdf5c456801bae20ba9b6d04d43e274fee226bf7c55a78acf72800f6f7724c04",
}
"""Pinned: any change to an observation, mask, reward, flag or info breaks one."""


@pytest.mark.parametrize("name", sorted(GOLDEN_ROLLOUTS))
def test_golden_digests(name: str) -> None:
    assert rollout_digest(XiangqiEnv, **GOLDEN_ROLLOUTS[name]) == GOLDEN_DIGESTS[name]
