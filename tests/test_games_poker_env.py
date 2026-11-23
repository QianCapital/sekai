"""Heads-up hold'em: the environment's API, observations, rewards, info and seeding."""

from __future__ import annotations

import hashlib
import json
import struct
from typing import Any

import numpy as np
import pytest

import sekai
from sekai.envs.games import OpponentWrapper, RandomPolicy
from sekai.envs.games.errors import GameError, GameOver, IllegalMove
from sekai.envs.games.poker import (
    ALL_IN,
    CHECK_CALL,
    FOLD,
    HandComplete,
    HoldemEnv,
    IllegalAction,
    PokerError,
    parse_cards,
)
from sekai.envs.games.poker import observations as O
from sekai.wrappers import FrameStackObservation, NormalizeObservation, TimeLimit

DECK = "AsKdQhQc2c7d9hJs3s"


def test_make_and_spaces() -> None:
    env = sekai.make("HeadsUpHoldem-v0")
    assert isinstance(env.unwrapped, HoldemEnv)
    assert env.action_space == sekai.Discrete(7)
    assert env.observation_space.shape == (125,)
    deep = sekai.make("HeadsUpHoldem-v0", history_length=2, small_blind=1, big_blind=2, stack=200)
    assert deep.observation_space.shape == (137,) and deep.unwrapped.big_blind == 2
    assert HoldemEnv.sides == ("p1", "p2") and HoldemEnv.players == (0, 1)
    assert HoldemEnv.metadata["render_fps"] > 0
    assert str(HoldemEnv()) == "<HeadsUpHoldem-v0 blinds=50/100 stack=10000>"


def test_errors_derive_from_the_games_errors() -> None:
    assert issubclass(IllegalAction, IllegalMove) and issubclass(IllegalAction, PokerError)
    assert issubclass(IllegalAction, sekai.InvalidAction)
    assert issubclass(HandComplete, GameOver) and issubclass(HandComplete, PokerError)
    assert issubclass(PokerError, GameError) and issubclass(PokerError, sekai.SekaiError)


@pytest.mark.parametrize("history_length", [0, 3])
def test_check_env(history_length: int) -> None:
    env = HoldemEnv(history_length=history_length)
    assert sekai.check_env(env, n_steps=300, check_determinism=True) == []


def test_reset_and_info() -> None:
    env = HoldemEnv()
    r = env.reset(seed=0, options={"button": 1, "deck": DECK})
    assert isinstance(r, sekai.ResetResult)
    obs, info = r.observation, r.info
    assert obs.dtype == np.float32 and obs.shape == (125,)
    assert env.observation_space.contains(obs)
    assert list(info) == [
        "legal_action_mask",
        "legal_actions",
        "to_play",
        "observer",
        "street",
        "payoffs",
        "view",
        "last_action",
        "result",
    ]
    assert info["to_play"] == 1 and info["observer"] == 1 and info["street"] == "preflop"
    assert info["legal_action_mask"].dtype == np.int8
    assert info["legal_action_mask"].tolist() == [1, 1, 1, 0, 1, 1, 1]
    assert info["legal_actions"].dtype == np.int64
    assert info["legal_actions"].tolist() == [0, 1, 2, 4, 5, 6]
    assert info["payoffs"] == [0.0, 0.0] and info["result"] is None
    assert info["view"].seat == 1 and info["view"].hole == tuple(parse_cards("QhQc"))
    assert info["last_action"] is None
    assert np.array_equal(env.action_mask(), info["legal_action_mask"])
    assert np.array_equal(env.legal_action_mask(), info["legal_action_mask"])
    assert np.array_equal(env.legal_actions(), info["legal_actions"])
    assert env.to_play == 1 and not env.done


def test_masks_are_fresh_arrays() -> None:
    env = HoldemEnv()
    r = env.reset(seed=0)
    r.info["legal_action_mask"][:] = 0
    r.info["legal_actions"][:] = -1
    assert env.action_mask().sum() > 0 and env.legal_actions().min() >= 0
    env.action_mask()[:] = 0
    assert env.legal_action_mask().sum() > 0


def test_observation_layout() -> None:
    env = HoldemEnv()
    obs = env.reset(options={"button": 0, "deck": DECK}).observation
    hole = np.flatnonzero(obs[O.HOLE_OFFSET : O.HOLE_OFFSET + 52]).tolist()
    assert hole == sorted(parse_cards("AsKd"))
    assert obs[O.BOARD_OFFSET : O.BOARD_OFFSET + 52].sum() == 0
    assert obs[O.STREET_OFFSET : O.STREET_OFFSET + 4].tolist() == [1, 0, 0, 0]
    assert obs[O.BUTTON_SLOT] == 1.0
    assert obs[O.POT_SLOT] == pytest.approx(150 / 10000)
    assert obs[O.OWN_STACK_SLOT] == pytest.approx(9950 / 10000)
    assert obs[O.OPP_STACK_SLOT] == pytest.approx(9900 / 10000)
    assert obs[O.OWN_BET_SLOT] == pytest.approx(50 / 10000)
    assert obs[O.OPP_BET_SLOT] == pytest.approx(100 / 10000)
    assert obs[O.TO_CALL_SLOT] == pytest.approx(50 / 10000)
    assert obs[O.MIN_RAISE_SLOT] == pytest.approx(200 / 10000)
    assert obs[O.BIG_BLIND_SLOT] == pytest.approx(100 / 10000)
    assert obs[O.BET_COUNT_OFFSET : O.BET_COUNT_OFFSET + 8].sum() == 0

    r = env.step(4)  # seat 0 raises pot (to 300)
    obs = r.observation  # seat 1's perspective now
    assert r.info["to_play"] == 1 and r.reward == 0.0 and not r.terminated
    assert np.flatnonzero(obs[:52]).tolist() == sorted(parse_cards("QhQc"))
    assert obs[O.BUTTON_SLOT] == 0.0
    assert obs[O.TO_CALL_SLOT] == pytest.approx(200 / 10000)
    counts = obs[O.BET_COUNT_OFFSET : O.BET_COUNT_OFFSET + 8].tolist()
    assert counts == [0, 0.25, 0, 0, 0, 0, 0, 0]  # the opponent raised once preflop

    r = env.step(1)  # call: flop dealt, seat 1 (big blind) to act
    obs = r.observation
    board = np.flatnonzero(obs[O.BOARD_OFFSET : O.BOARD_OFFSET + 52]).tolist()
    assert board == sorted(parse_cards("2c7d9h"))
    assert obs[O.STREET_OFFSET : O.STREET_OFFSET + 4].tolist() == [0, 1, 0, 0]
    assert obs[O.MIN_RAISE_SLOT] == pytest.approx(100 / 10000)
    assert r.info["last_action"] == "Calls 200"


def test_observation_never_contains_opponent_cards() -> None:
    env = HoldemEnv()
    rng = np.random.default_rng(5)
    for episode in range(40):
        r = env.reset(seed=episode)
        while True:
            seat = r.info["observer"]
            opp_cards = env.hand.hole(1 - seat)
            if env.hand.street != "showdown":
                assert r.observation[opp_cards].sum() == 0
            if r.info.get("result") is not None:
                break
            mask = r.info["legal_action_mask"]
            r = env.step(int(rng.choice(np.flatnonzero(mask))))


def test_terminal_reward_and_payoffs() -> None:
    env = HoldemEnv()
    env.reset(options={"button": 0, "deck": DECK})
    env.step(1)  # seat 0 calls
    r = env.step(4)  # seat 1 raises pot: to 300
    r = env.step(0)  # seat 0 folds
    assert r.terminated and not r.truncated
    assert r.reward == pytest.approx(-1.0)  # seat 0 loses its 100
    assert r.info["payoffs"] == [-1.0, 1.0]
    assert r.info["to_play"] is None and r.info["observer"] == 0
    assert r.info["legal_action_mask"].sum() == 0 and r.info["street"] == "complete"
    assert r.info["last_action"] == "Folds"
    assert env.observation_space.contains(r.observation)
    assert env.done and env.legal_actions().size == 0
    with pytest.raises(HandComplete):
        env.step(1)


def test_showdown_payoffs_from_winner_perspective() -> None:
    env = HoldemEnv()
    env.reset(options={"button": 0, "deck": DECK})
    env.step(6)  # seat 0 all-in
    r = env.step(1)  # seat 1 calls with queens and wins
    assert r.terminated and r.reward == pytest.approx(100.0)
    assert r.info["payoffs"] == [-100.0, 100.0]
    assert r.info["result"].showdown
    assert r.info["result"] is r.info["view"].result


def test_illegal_actions() -> None:
    env = HoldemEnv()
    with pytest.raises(sekai.ResetRequired):
        env.step(0)
    with pytest.raises(sekai.ResetRequired):
        env.action_mask()
    env.reset(options={"button": 0, "deck": DECK})
    with pytest.raises(IllegalAction, match="half pot"):
        env.step(3)  # half pot duplicates the min-raise preflop
    for bad in (7, -1, 1.0, True, "1"):
        with pytest.raises(sekai.InvalidAction):
            env.step(bad)  # type: ignore[arg-type]
    env.step(np.int64(1))
    with pytest.raises(IllegalAction):
        env.step(0)  # folding is illegal when checking is free


def test_seeding_and_options() -> None:
    env = HoldemEnv()
    a = env.reset(seed=3)
    b = env.reset(seed=3)
    assert np.array_equal(a.observation, b.observation)
    other = HoldemEnv()
    other.reset(seed=3)
    assert other.hand.hole(0) == env.hand.hole(0) and other.hand.button == env.hand.button
    r = env.reset(seed=1, options={"stacks": (5000, 20000), "button": 1})
    assert env.hand.starting_stacks == (5000, 20000) and env.hand.button == 1
    assert r.observation[O.OWN_STACK_SLOT] == pytest.approx(20000 / 20000 - 50 / 20000)
    with pytest.raises(ValueError):
        env.reset(options={"stacks": (100, 10000)})
    with pytest.raises(ValueError):
        HoldemEnv(small_blind=200, big_blind=100)
    with pytest.raises(ValueError):
        HoldemEnv(history_length=-1)


def test_random_draws_button_then_shuffle() -> None:
    env = HoldemEnv()
    env.reset(seed=9)
    rng = np.random.default_rng(9)
    button = int(rng.integers(2))
    deck = rng.permutation(52).tolist()
    assert env.hand.button == button
    assert env.hand.hole(0) == deck[0:2] and env.hand.hole(1) == deck[2:4]
    # A given button draws nothing for it; a fixed deck draws nothing at all.
    env.reset(seed=9, options={"button": 0})
    assert env.hand.hole(0) == np.random.default_rng(9).permutation(52).tolist()[0:2]
    env.reset(seed=9, options={"button": 0, "deck": DECK})
    assert env.rng.bit_generator.state == np.random.default_rng(9).bit_generator.state


def test_history_length_extends_observation() -> None:
    env = HoldemEnv(history_length=3)
    assert env.observation_space.shape == (125 + 18,)
    env.reset(options={"button": 0, "deck": DECK})
    env.step(4)  # seat 0: raise to 300 (adds 250)
    obs = env.step(1).observation  # seat 1 calls; flop, seat 1 to act
    first = obs[O.HISTORY_OFFSET : O.HISTORY_OFFSET + 6].tolist()  # most recent: own call
    second = obs[O.HISTORY_OFFSET + 6 : O.HISTORY_OFFSET + 12].tolist()
    assert first == pytest.approx([1, 0, 0, 1, 0, 200 / 10000])
    assert second == pytest.approx([0, 1, 0, 0, 1, 250 / 10000])
    assert obs[O.HISTORY_OFFSET + 12 :].sum() == 0


def test_random_rollouts_terminate_and_conserve_chips() -> None:
    env = HoldemEnv()
    for episode in range(200):
        r = env.reset(seed=episode)
        steps = 0
        total = 0.0
        while True:
            assert env.observation_space.contains(r.observation)
            action = env.action_space.sample(mask=r.info["legal_action_mask"])
            r = env.step(action)
            steps += 1
            total += r.reward
            if r.terminated:
                break
            assert r.reward == 0.0
        payoffs = r.info["payoffs"]
        assert payoffs[0] + payoffs[1] == 0
        assert sum(env.hand.stacks) == 20000
        assert steps < 200
        assert total == payoffs[r.info["observer"]]  # only the terminal step pays


def test_acting_on_the_hand_directly_updates_the_mask() -> None:
    env = HoldemEnv()
    env.reset(options={"button": 0, "deck": DECK})
    assert env.action_mask().tolist() == [1, 1, 1, 0, 1, 1, 1]
    env.hand.act("call")  # a trusted caller playing the hand itself
    assert env.action_mask().tolist() == [0, 1, 1, 0, 1, 1, 1]
    r = env.step(CHECK_CALL)  # the big blind checks: the flop
    assert r.info["street"] == "flop" and r.info["last_action"] == "Checks"


def test_reset_mid_hand_and_failed_resets() -> None:
    env = HoldemEnv()
    env.reset(seed=1, options={"button": 0, "deck": DECK})
    env.step(4)
    r = env.reset(options={"button": 1, "deck": DECK})  # mid-hand: a new hand
    assert r.info["to_play"] == 1 and r.info["legal_actions"].tolist() == [0, 1, 2, 4, 5, 6]
    with pytest.raises(ValueError):
        env.reset(options={"stacks": (50, 10000)})
    # A reset that fails leaves the hand in play.
    r2 = env.step(CHECK_CALL)
    assert r2.info["street"] == "preflop" and r2.info["to_play"] == 0


def test_snapshot_and_restore() -> None:
    env = HoldemEnv(history_length=2)
    env.reset(seed=4, options={"button": 0, "deck": DECK})
    env.step(CHECK_CALL)  # seat 0 limps
    snapshot = env.snapshot()
    first = env.step(4)
    env.restore(snapshot)
    assert env.action_mask().tolist() == [0, 1, 1, 0, 1, 1, 1]
    again = env.step(4)
    assert np.array_equal(first.observation, again.observation)
    assert first.info["view"].to_dict() == again.info["view"].to_dict()


# ----------------------------------------------------------------------
# Golden digests: seeded random rollouts, observation by observation
# ----------------------------------------------------------------------

GOLDEN_DECK = [51, 45, 42, 40, 0, 21, 30, 39, 3, 17, 8, 26]

GOLDEN_CASES: list[tuple[dict[str, Any], int, dict[str, Any] | None, int, str]] = [
    # (constructor kwargs, first seed, reset options, hands, digest)
    ({}, 0, None, 120, "cf20fa4e95c16da96023592f015afb8e00be27d6704ec1a8302c040ded1d14e5"),
    (
        {"history_length": 4},
        1,
        {"button": 1},
        80,
        "b355062bb8e74d2b2e77c4e9d11524e5dd1ea66bd5ace625f6d47e3e01b53524",
    ),
    (
        {"small_blind": 1, "big_blind": 2, "stack": 40},
        2,
        {"stacks": (5, 37)},
        80,
        "11153fb496267f8bca27e839df1e9dcd6c83ebe6f2a90482b9074e4f54cb6033",
    ),
    (
        {},
        3,
        {"deck": GOLDEN_DECK, "button": 0},
        60,
        "d0721f9517fa98a18aac06864e14c1a4ae9dfab7adf7bcb8e506fc5aa804a2ca",
    ),
    (
        {"small_blind": 25, "big_blind": 50, "stack": 2500, "history_length": 2},
        4,
        {"stacks": (2500, 7000), "button": 0},
        80,
        "8ecc29d22da0cc0515b5c99e8982bd3cc9fdeb9268144cce23cc7bbf23351dd7",
    ),
    (
        {"small_blind": 50, "big_blind": 100, "stack": 101},
        5,
        None,
        40,
        "9cf001b4cea33a8d0e3ff0f396fd2938f39c7814e66ca234ca5730f5707d5956",
    ),
]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"dtype": value.dtype.str, "data": value.tolist()}
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def rollout_digest(env: HoldemEnv, seed: int, options: dict[str, Any] | None, hands: int) -> str:
    """sha256 over every observation, mask, reward, flag and info of a seeded rollout.

    The first hand is reset with `seed`, the others continue its stream; actions are
    uniform among the legal ones, drawn from a generator seeded with `seed + 1000`.
    """
    digest = hashlib.sha256()
    picks = np.random.default_rng(seed + 1000)

    def absorb(result: Any) -> None:
        observation, info = result.observation, result.info
        digest.update(observation.dtype.str.encode() + observation.tobytes())
        digest.update(info["legal_action_mask"].tobytes() + info["legal_actions"].tobytes())
        if isinstance(result, sekai.StepResult):
            flags = (result.reward, result.terminated, result.truncated)
            digest.update(struct.pack("<d??", *flags))
        data = {key: _jsonable(value) for key, value in info.items()}
        digest.update(json.dumps(data).encode())

    for hand in range(hands):
        result: Any = env.reset(seed=seed if hand == 0 else None, options=options)
        absorb(result)
        while True:
            legal = result.info["legal_actions"]
            result = env.step(int(legal[int(picks.integers(len(legal)))]))
            absorb(result)
            if result.terminated:
                break
    return digest.hexdigest()


@pytest.mark.parametrize(("kwargs", "seed", "options", "hands", "digest"), GOLDEN_CASES)
def test_golden_digests(
    kwargs: dict[str, Any], seed: int, options: dict[str, Any] | None, hands: int, digest: str
) -> None:
    assert rollout_digest(HoldemEnv(**kwargs), seed, options, hands) == digest


# ----------------------------------------------------------------------
# Against a fixed opponent
# ----------------------------------------------------------------------


class CheckCall:
    """Checks or calls, always."""

    def act(self, observation: Any, info: dict[str, Any]) -> int:
        return CHECK_CALL


class Folder:
    """Folds to any bet, checks otherwise."""

    def act(self, observation: Any, info: dict[str, Any]) -> int:
        return FOLD if info["legal_action_mask"][FOLD] else CHECK_CALL


def test_opponent_plays_until_the_agents_turn_across_streets() -> None:
    env = OpponentWrapper(HoldemEnv(), CheckCall(), agent_color="p1")
    r = env.reset(options={"button": 0, "deck": DECK})
    assert r.info["agent_color"] == 0 and r.info["to_play"] == 0  # the agent's button
    r = env.step(CHECK_CALL)  # the big blind checks its option, then leads the flop
    assert r.info["to_play"] == 0 and r.info["street"] == "flop"
    actions = [e for e in env.unwrapped.hand.events if e["type"] == "action"]
    assert [(e["seat"], e["street"], e["kind"]) for e in actions] == [
        (0, "preflop", "call"),
        (1, "preflop", "check"),
        (1, "flop", "check"),
    ]
    assert r.info["opponent_action"] == CHECK_CALL and r.info["opponent_move"] == "Checks"
    assert r.info["observer"] == 0 and r.reward == 0.0 and not r.done


def test_opponent_rewards_are_the_agents_in_big_blinds() -> None:
    # The agent (p2, seat 1) has the queens; the opponent on the button limps.
    env = OpponentWrapper(HoldemEnv(), CheckCall(), agent_color="p2")
    r = env.reset(options={"button": 0, "deck": DECK})
    assert r.info["to_play"] == 1 and r.info["opponent_move"] == "Calls 50"
    r = env.step(ALL_IN)  # the opponent calls: the queens hold
    assert r.terminated and r.reward == pytest.approx(100.0)
    assert r.info["payoffs"] == [-100.0, 100.0]

    env = OpponentWrapper(HoldemEnv(), Folder(), agent_color="p1")
    env.reset(options={"button": 0, "deck": DECK})
    r = env.step(ALL_IN)  # the big blind folds
    assert r.terminated and r.reward == pytest.approx(1.0)

    env.reset(options={"button": 0, "deck": DECK})
    r = env.step(FOLD)  # the agent's own fold ends the hand
    assert r.terminated and r.reward == pytest.approx(-0.5)


def test_a_hand_the_opponent_ends_ends_in_the_agents_own_view() -> None:
    # The env's last observation is the last actor's private view; when that is the
    # opponent (it folds, or calls to a showdown), the agent gets its own seat's.
    for opponent, deck in ((Folder(), DECK), (CheckCall(), DECK)):
        env = OpponentWrapper(HoldemEnv(), opponent, agent_color="p2")
        env.reset(options={"button": 1, "deck": deck})  # the agent's button: it acts first
        r = env.step(ALL_IN)
        base = env.unwrapped
        assert r.terminated and r.info["observer"] == 1 and r.info["view"].seat == 1
        np.testing.assert_array_equal(r.observation, base.observe(1))
        assert r.info["view"].holes[1] == tuple(base.hand.hole(1))
        if isinstance(opponent, Folder):  # the folded cards stay hidden
            assert r.info["view"].holes[0] is None
    env = OpponentWrapper(HoldemEnv(), CheckCall(), agent_color="p2")
    env.reset(options={"button": 1, "deck": DECK})
    r = env.step(FOLD)  # the agent's own fold: its own view, as before
    assert r.info["observer"] == 1


def test_a_hand_a_time_limit_stops_ends_in_the_agents_own_view() -> None:
    # Stopped after the agent's own action, the env's observation is the next seat's.
    env = OpponentWrapper(TimeLimit(HoldemEnv(), max_episode_steps=1), CheckCall(), "p2")
    env.reset(options={"button": 1, "deck": DECK})  # the agent's button: it acts first
    r = env.step(CHECK_CALL)
    base = env.unwrapped
    assert r.truncated and not r.terminated and base.hand.to_act == 0
    assert r.info["observer"] == 1 and r.info["view"].seat == 1
    assert r.info["view"].holes[0] is None
    np.testing.assert_array_equal(r.observation, base.observe(1))


def test_observation_wrappers_go_outside_the_opponent_wrapper() -> None:
    # Inside, they would see both seats' private views, one after the other.
    for inner in (FrameStackObservation(HoldemEnv(), 2), NormalizeObservation(HoldemEnv())):
        with pytest.raises(TypeError, match="both seats' private views"):
            OpponentWrapper(inner)
    outside = FrameStackObservation(OpponentWrapper(HoldemEnv(), agent_color="p1"), 2)
    assert outside.reset(seed=0).observation.shape == (2, 125)
    OpponentWrapper(TimeLimit(HoldemEnv(), max_episode_steps=50))  # not an observation's


def test_opponent_wrapper_deals_again_when_the_opponent_ends_the_hand() -> None:
    # The folding opponent ends every hand in which it has the button and acts first.
    env = OpponentWrapper(HoldemEnv(), Folder(), agent_color="p2")
    skipped: list[float] = []
    for seed in range(20):
        r = env.reset(seed=seed)
        assert r.info["to_play"] == 1 and not env.unwrapped.done
        skipped += r.info.get("skipped_rewards", [])
    assert skipped and all(reward == pytest.approx(0.5) for reward in skipped)
    with pytest.raises(RuntimeError, match="in a row"):
        env.reset(options={"button": 0})


def test_opponent_wrapper_random_play() -> None:
    env = OpponentWrapper(HoldemEnv(), RandomPolicy(seed=3), agent_color="random")
    for seed in range(30):
        r = env.reset(seed=seed)
        while not r.done:
            assert r.info["to_play"] == r.info["agent_color"]
            r = env.step(int(env.action_space.sample(mask=r.info["legal_action_mask"])))
