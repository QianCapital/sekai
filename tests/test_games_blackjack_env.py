"""Blackjack: the environment (spaces, episodes, rewards, masks, persistence, registration)."""

from __future__ import annotations

import hashlib
import json
import pickle
import struct
from typing import Any

import numpy as np
import pytest

import sekai
from sekai.envs.games import GameError, GameOver, IllegalMove
from sekai.envs.games.blackjack import BlackjackEnv, BlackjackTable, Rules, Shoe
from sekai.envs.games.blackjack.errors import (
    BlackjackError,
    IllegalAction,
    NotYourTurn,
    RoundOver,
)
from sekai.envs.games.blackjack.observations import (
    OBS_SIZE,
    SLOT_BANKROLL,
    SLOT_CAN_DOUBLE,
    SLOT_CAN_SPLIT,
    SLOT_CAN_SURRENDER,
    SLOT_COMPOSITION,
    SLOT_DEALER_UP,
    SLOT_NUM_CARDS,
    SLOT_PAIR,
    SLOT_SOFT,
    SLOT_TOTAL,
    SLOT_TRUE_COUNT,
    encode_view,
)


def play_round(env: BlackjackEnv, rng: np.random.Generator) -> sekai.StepResult:
    result = env.reset()
    while True:
        legal = result.info["legal_actions"]
        result = env.step(int(rng.choice(legal)))
        if result.done:
            return result


def test_spaces() -> None:
    env = BlackjackEnv()
    assert env.action_space == sekai.Discrete(9)
    assert env.observation_space.shape == (OBS_SIZE,) == (43,)
    assert env.observation_space.dtype == np.float32
    assert BlackjackEnv(bet_levels=(1, 5)).action_space.n == 7


def test_reset_is_betting_phase() -> None:
    env = BlackjackEnv()
    r = env.reset(seed=0)
    assert isinstance(r, sekai.ResetResult)
    assert env.observation_space.contains(r.observation)
    info = r.info
    assert info["phase"] == "betting" and info["to_play"] == 0
    assert info["legal_action_mask"].dtype == np.int8
    assert info["legal_action_mask"].tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1]
    assert info["legal_actions"].dtype == np.int64
    assert info["legal_actions"].tolist() == [5, 6, 7, 8]
    obs = r.observation
    assert obs[0] == 1 and obs[1] == 0 and obs[SLOT_BANKROLL] == 1.0
    assert obs[SLOT_TOTAL : SLOT_TRUE_COUNT - 2].sum() == 0
    assert obs[SLOT_COMPOSITION : SLOT_COMPOSITION + 10].sum() == pytest.approx(1.0)


def test_round_reward_is_net_over_min_bet() -> None:
    env = BlackjackEnv()
    rng = np.random.default_rng(0)
    env.reset(seed=1)
    for _ in range(300):
        result = play_round(env, rng)
        before = 1000  # every episode starts from the initial bankroll
        after = result.info["view"]["seats"][0]["bankroll"]
        assert result.terminated and not result.truncated
        assert result.reward == pytest.approx((after - before) / 10)
        assert result.reward == pytest.approx(result.info["net"] / 10)
        assert result.info["legal_action_mask"].sum() == 0
        assert env.observation_space.contains(result.observation)


def test_intermediate_rewards_are_zero_and_observations_valid() -> None:
    env = BlackjackEnv()
    rng = np.random.default_rng(1)
    env.reset(seed=2)
    for _ in range(100):
        info = env.reset().info
        while True:
            step = env.step(int(rng.choice(info["legal_actions"])))
            assert env.observation_space.contains(step.observation)
            if step.done:
                break
            assert step.reward == 0.0 and step.info["phase"] == "playing"
            info = step.info


def test_playing_observation_slots() -> None:
    shoe = Shoe(6, seed=0)
    shoe.stack_top(["8h", "6c", "8d", "Ts"])
    table = BlackjackTable(seats=1, shoe=shoe)
    table.act("bet:20")
    obs = encode_view(table.view(0), 0, 1000)
    assert obs[1] == 1 and obs[0] == 0
    assert obs[SLOT_TOTAL] == pytest.approx(16 / 21) and obs[SLOT_SOFT] == 0
    assert obs[SLOT_PAIR + 7] == 1 and obs[SLOT_PAIR : SLOT_PAIR + 10].sum() == 1
    assert obs[SLOT_NUM_CARDS] == pytest.approx(0.2)
    assert obs[SLOT_CAN_DOUBLE] == obs[SLOT_CAN_SPLIT] == obs[SLOT_CAN_SURRENDER] == 1
    assert obs[SLOT_DEALER_UP + 5] == 1 and obs[SLOT_DEALER_UP : SLOT_DEALER_UP + 10].sum() == 1
    assert obs[28] == 0 and obs[29] == pytest.approx(0.25)
    assert obs[SLOT_BANKROLL] == pytest.approx(0.98)
    table.act("split")
    obs = encode_view(table.view(0), 0, 1000)
    assert obs[29] == pytest.approx(0.5) and obs[SLOT_CAN_SURRENDER] == 0


def test_illegal_actions_raise() -> None:
    env = BlackjackEnv()
    with pytest.raises(sekai.ResetRequired):
        env.step(5)
    r = env.reset(seed=0)
    with pytest.raises(IllegalAction):
        env.step(1)  # hit before betting
    with pytest.raises(sekai.InvalidAction):
        env.step(42)
    r = env.step(5)
    while not r.done:
        with pytest.raises(IllegalAction):
            env.step(6)  # bet while playing
        r = env.step(0)
    with pytest.raises(RoundOver):
        env.step(0)


def test_illegal_action_details() -> None:
    env = BlackjackEnv()
    env.reset(seed=0)
    with pytest.raises(IllegalAction) as caught:
        env.step(1)
    error = caught.value
    assert error.action == 1 and error.legal == [5, 6, 7, 8]
    assert error.reason == "hit is not possible before the bet is placed"
    assert str(error) == (
        "Illegal blackjack action 1: hit is not possible before the bet is placed. "
        "Legal: [5, 6, 7, 8]."
    )
    for bad in (True, 1.0, "1", None, -1, 9):
        with pytest.raises(sekai.InvalidAction):
            env.step(bad)  # type: ignore[arg-type]
    assert env.step(np.int64(5)).info["phase"] in ("playing", "betting")


def test_error_hierarchy() -> None:
    assert issubclass(BlackjackError, GameError) and issubclass(GameError, sekai.SekaiError)
    assert issubclass(IllegalAction, IllegalMove) and issubclass(IllegalAction, BlackjackError)
    assert issubclass(IllegalAction, sekai.InvalidAction)
    assert issubclass(RoundOver, GameOver) and issubclass(RoundOver, BlackjackError)
    assert issubclass(NotYourTurn, BlackjackError)
    for error in (IllegalAction("hit", "no", ["stand"]), NotYourTurn(1, 0), RoundOver(3)):
        copy = pickle.loads(pickle.dumps(error))
        assert type(copy) is type(error) and str(copy) == str(error)
    assert str(RoundOver(3)) == (
        "Round 3 is over and has been settled. Call reset() to play the next round."
    )
    assert str(NotYourTurn(1, None)) == "Seat 1 cannot act now: nobody can act."


def test_unaffordable_bet_levels_are_masked() -> None:
    env = BlackjackEnv(bankroll=50)
    r = env.reset(seed=0)
    assert r.info["legal_actions"].tolist() == [5, 6, 7]  # 10, 20, 40 but not 80


def test_shoe_persists_across_resets() -> None:
    env = BlackjackEnv()
    rng = np.random.default_rng(3)
    env.reset(seed=4)
    fractions = []
    for _ in range(10):
        play_round(env, rng)
        fractions.append(env.view()["shoe"]["dealt_fraction"])
    assert fractions == sorted(fractions) and fractions[-1] > 0.05
    env.reset(options={"new_shoe": True})
    assert env.view()["shoe"]["dealt_fraction"] == 0.0


def test_bankroll_resets_unless_kept() -> None:
    env = BlackjackEnv()
    rng = np.random.default_rng(5)
    env.reset(seed=6)
    for _ in range(5):
        play_round(env, rng)
        assert env.reset().info["bankroll"] == 1000
    result = play_round(env, rng)
    carried = result.info["bankroll"]
    assert env.reset(options={"keep_bankroll": True}).info["bankroll"] == carried
    assert env.reset(options={"bankroll": 250}).info["bankroll"] == 250
    r = env.reset(options={"bankroll": 5})
    assert r.info["rebuy"] and r.info["bankroll"] == 1000


def test_reset_mid_round_finishes_the_round_first() -> None:
    # 16 against the dealer's 9 and hole card 5, who then draws a ten: 24, bust.
    for options, bankroll in (({}, 1000), ({"keep_bankroll": True}, 1010)):
        env = BlackjackEnv()
        env.reset(seed=0)
        env.table._shoe.stack_top(["Th", "9c", "6d", "5s", "Tc"])  # type: ignore[union-attr]
        env.step(5)
        played = env.table.rounds_played  # type: ignore[union-attr]
        r = env.reset(options=options)  # stands on the 16
        assert r.info["phase"] == "betting" and r.info["bankroll"] == bankroll
        assert env.table.rounds_played == played + 1  # type: ignore[union-attr]
        assert env.table.last_round["seats"][0]["net"] == 10  # type: ignore[union-attr]
    # So a time limit and the vector envs' autoreset work as with any env.
    env = sekai.make("Blackjack-v0", max_episode_steps=1)
    result = env.reset(seed=3)
    for _ in range(20):
        result = env.step(5)
        assert result.done
        result = env.reset()
    vec = sekai.make_vec("Blackjack-v0", num_envs=2, vectorization_mode="sync", max_episode_steps=1)
    vec.reset(seed=0)
    for _ in range(10):
        vec.step(np.array([5, 5]))


def test_illegal_action_lose_loses_the_rounds_stake() -> None:
    env = BlackjackEnv(illegal_action="lose")
    env.reset(seed=0)
    env.table._shoe.stack_top(["8h", "9c", "8d", "7s", "3c", "4d"])  # type: ignore[union-attr]
    env.step(5)  # bet 10
    env.step(3)  # split the eights against the nine
    r = env.step(7)  # a bet while playing: both hands are lost
    assert r.terminated and r.reward == -2.0 and r.info["net"] == -20
    assert r.info["illegal_action"] == 7
    assert [h["outcome"] for h in r.info["round"]["seats"][0]["hands"]] == ["lose", "lose"]
    with pytest.raises(RoundOver):
        env.step(5)
    r = env.reset(options={"keep_bankroll": True})
    played = env.table.rounds_played  # type: ignore[union-attr]
    r = env.step(0)  # a stand before the bet: the smallest bet is lost, no cards dealt
    assert r.terminated and r.reward == -1.0 and r.info["net"] == -10
    assert r.info["phase"] == "betting" and r.info["round"]["seats"][0]["forfeit"]
    assert r.info["round"]["seats"][0]["hands"] == [] and not r.info["round"]["dealer"]["cards"]
    assert env.table.rounds_played == played + 1  # type: ignore[union-attr]
    assert env.reset(options={"keep_bankroll": True}).info["bankroll"] == 970
    table = env.table
    assert sum(table.bankrolls) + table.chips_on_table + table.house_net == pytest.approx(  # type: ignore[union-attr]
        table.chips_in  # type: ignore[union-attr]
    )
    with pytest.raises(ValueError, match="illegal_action"):
        BlackjackEnv(illegal_action="ignore")  # type: ignore[arg-type]


def test_bet_levels_the_table_cannot_take_are_refused() -> None:
    with pytest.raises(ValueError, match="above the table maximum"):
        BlackjackEnv(rules=Rules(max_bet=50), bet_levels=(8, 16))
    env = BlackjackEnv(rules=Rules(max_bet=50), bet_levels=(1, 8))  # 80 is simply never legal
    assert env.reset(seed=0).info["legal_action_mask"].tolist() == [0, 0, 0, 0, 0, 1, 0]


def test_seeded_episodes_are_reproducible() -> None:
    def trajectory(seed: int) -> list[np.ndarray]:
        env = BlackjackEnv()
        rng = np.random.default_rng(0)
        out = []
        for episode in range(20):
            r = env.reset(seed=seed) if episode == 0 else env.reset()
            out.append(r.observation)
            info = r.info
            while True:
                step = env.step(int(rng.choice(info["legal_actions"])))
                out.append(step.observation)
                info = step.info
                if step.done:
                    break
        return out

    a, b = trajectory(7), trajectory(7)
    assert len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))
    c = trajectory(8)
    assert not (len(a) == len(c) and all(np.array_equal(x, y) for x, y in zip(a, c)))


def test_info_view_hides_hole_card() -> None:
    env = BlackjackEnv()
    env.reset(seed=0)
    for _ in range(50):
        r = env.reset()
        r = env.step(5)
        if not r.done:
            assert r.info["view"]["dealer"]["cards"][1] is None
            assert not r.info["view"]["dealer"]["revealed"]
            while not r.done:
                r = env.step(0)


def test_registration() -> None:
    env = sekai.make("Blackjack-v0")
    assert isinstance(env.unwrapped, BlackjackEnv)
    assert env.spec.id == "Blackjack-v0" and env.spec.max_episode_steps is None
    r = env.reset(seed=0)
    assert r.info["legal_action_mask"].shape == (9,)
    env2 = sekai.make("Blackjack-v0", rules=Rules(decks=2), bet_levels=(1, 2))
    assert env2.unwrapped.rules.decks == 2 and env2.action_space.n == 7
    assert sekai.envs.games.BlackjackEnv is BlackjackEnv


# ----------------------------------------------------------------------------------------
# sekai conventions
# ----------------------------------------------------------------------------------------


def test_check_env() -> None:
    assert sekai.check_env(BlackjackEnv(), n_steps=500, check_determinism=True) == []
    assert sekai.check_env(sekai.make("Blackjack-v0"), n_steps=200) == []


def test_action_mask_and_metadata() -> None:
    env = BlackjackEnv()
    assert env.action_mask().tolist() == [0] * 9  # nothing is legal before a reset
    r = env.reset(seed=3)
    assert np.array_equal(env.action_mask(), r.info["legal_action_mask"])
    assert env.action_mask().dtype == np.int8
    assert sekai.utils.masking.supports_action_mask(env)
    r = env.step(int(r.info["legal_actions"][0]))
    assert np.array_equal(env.action_mask(), r.info["legal_action_mask"])
    assert np.array_equal(env.legal_actions(), np.flatnonzero(env.action_mask()))
    assert env.metadata["render_fps"] == 2
    assert str(env) == "<Blackjack-v0 6D S17 DAS LS 3:2 bet_levels=(1, 2, 4, 8)>"
    assert not hasattr(env, "players") and not hasattr(env, "sides")


def test_returned_arrays_are_fresh() -> None:
    env = BlackjackEnv()
    r = env.reset(seed=0)
    r.info["legal_action_mask"][:] = 0
    r.info["legal_actions"][:] = 0
    r2 = env.reset()
    assert r2.info["legal_action_mask"].tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1]
    assert r2.info["legal_actions"].tolist() == [5, 6, 7, 8]
    assert r2.info["legal_action_mask"].flags.writeable
    r2.info["view"]["seats"][0]["bankroll"] = -1
    assert env.view()["seats"][0]["bankroll"] == 1000


def test_observe_and_view() -> None:
    env = BlackjackEnv()
    with pytest.raises(sekai.ResetRequired):
        env.view()
    with pytest.raises(sekai.ResetRequired):
        env.observe()
    r = env.reset(seed=2)
    assert np.array_equal(env.observe(), r.observation)
    assert env.view() == r.info["view"]


def test_snapshot_restores_the_shoe() -> None:
    env = BlackjackEnv()
    rng = np.random.default_rng(0)
    env.reset(seed=11)
    for _ in range(5):
        play_round(env, rng)
    snap = env.snapshot()

    def rounds() -> list[Any]:
        out = []
        picks = np.random.default_rng(1)
        for _ in range(8):
            result = play_round(env, picks)
            out.append((result.reward, result.info["view"]["dealer"]["cards"]))
        return out

    first = rounds()
    env.restore(snap)
    assert rounds() == first


# ----------------------------------------------------------------------------------------
# Golden digests: seeded random rollouts, every output hashed
# ----------------------------------------------------------------------------------------
# Each digest is a SHA-256 over every reset and step of a rollout: the observation's
# bytes, the mask's bytes, the reward, the flags and a JSON dump of `info` (key order
# included), and the type and message of every error raised on the way. The rollout
# plays random legal actions, tries illegal ones, and resets with every option, also in
# the middle of a round, over one persistent shoe.

GOLDEN_CONFIGS: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    "default": ({}, {}),
    "single_deck_deep": (
        {"decks": 1, "penetration": 0.95, "resplit_aces": True, "max_hands": 8},
        {},
    ),
    "h17_no_peek_six_five": (
        {
            "decks": 2,
            "dealer_hits_soft_17": True,
            "dealer_peeks": False,
            "blackjack_payout": 1.2,
            "surrender": None,
            "double_after_split": False,
        },
        {},
    ),
    "small_bankroll": (
        {"min_bet": 5, "max_bet": 100, "bet_increment": 5, "hit_split_aces": True},
        {"bankroll": 60, "bet_levels": (1, 3, 7)},
    ),
}

GOLDEN_DIGESTS: dict[tuple[str, int], tuple[str, int]] = {
    ("default", 0): ("002277ebd88e27ae08ced93f07afb43a095c4da17931f23ef25fecd9d5b8136b", 533),
    ("default", 7): ("abf8f482484fcd79ed21c7f45205249a55de62b186272cc819e1597e5d036cd9", 533),
    ("single_deck_deep", 0): (
        "08adafd0e05cc373e7539af104e2ec32df54545a9f8c97dd5215555f48755b97",
        539,
    ),
    ("single_deck_deep", 7): (
        "477d211647e244b58ea19ed0c068071b67dd319f6fdb7efede528243e00bdff0",
        537,
    ),
    ("h17_no_peek_six_five", 0): (
        "c07c574e550822f50dbe8653842731fca27b2024138a7ff4c327801d4815db34",
        588,
    ),
    ("h17_no_peek_six_five", 7): (
        "f84b556b2224853a7102907503fbabe8800b3ed9d9e282b3088844446f76673b",
        579,
    ),
    ("small_bankroll", 0): (
        "12bd00b518d69ff76ace0bc825368e8f0ff803959704d7dc853dae01658801ff",
        545,
    ),
    ("small_bankroll", 7): (
        "4f8d1b906468b874b4de68a1ab39d9d71d8555834ca08d94ccde7ce39388201f",
        537,
    ),
}
"""Digest and step count of `golden_rollout(env, seed, 250)` by `(config, seed)`."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "data": value.tolist()}
    raise TypeError(f"Not JSON-friendly: {value!r}")


def _feed(digest: Any, observation: np.ndarray, info: dict[str, Any]) -> None:
    assert observation.dtype == np.float32 and info["legal_action_mask"].dtype == np.int8
    digest.update(observation.astype("<f4").tobytes())
    digest.update(info["legal_action_mask"].tobytes())
    digest.update(json.dumps(info, default=_jsonable).encode())


def _reset_options(rng: np.random.Generator) -> dict[str, Any] | None:
    u = rng.random()
    if u < 0.6:
        return None
    if u < 0.7:
        return {"keep_bankroll": True}
    if u < 0.8:
        return {"new_shoe": True}
    if u < 0.9:
        return {"bankroll": [5, 50, 250.5, 2000][int(rng.integers(4))]}
    return {"new_shoe": True, "keep_bankroll": True}


def golden_rollout(env: Any, seed: int, rounds: int) -> tuple[str, int]:
    """The digest of a seeded rollout of `env` over `rounds` rounds, and its step count."""
    digest = hashlib.sha256()
    rng = np.random.default_rng(seed + 1)
    r = env.reset(seed=seed)
    _feed(digest, r.observation, r.info)
    info = r.info
    steps = played = 0

    def failed(error: Exception) -> None:
        digest.update(f"{type(error).__name__}: {error}".encode())

    while played < rounds:
        u = rng.random()
        if u < 0.02:  # a reset in the middle of a round
            try:
                r = env.reset(options=_reset_options(rng))
            except Exception as error:
                failed(error)
            else:
                _feed(digest, r.observation, r.info)
                info = r.info
            continue
        mask = info["legal_action_mask"]
        if u < 0.05:
            action = int(rng.choice(np.flatnonzero(mask == 0)))
        else:
            action = int(rng.choice(np.flatnonzero(mask)))
        try:
            s = env.step(action)
        except Exception as error:
            failed(error)
            continue
        steps += 1
        _feed(digest, s.observation, s.info)
        digest.update(struct.pack("<d?", float(s.reward), s.terminated))
        digest.update(f"{type(s.reward).__name__} {s.truncated}".encode())
        info = s.info
        if s.done:
            played += 1
            if rng.random() < 0.1:
                try:
                    env.step(0)
                except Exception as error:
                    failed(error)
            r = env.reset(options=_reset_options(rng))
            _feed(digest, r.observation, r.info)
            info = r.info
    return digest.hexdigest(), steps


def _golden_env(name: str) -> BlackjackEnv:
    rules, kwargs = GOLDEN_CONFIGS[name]
    return BlackjackEnv(rules=Rules(**rules), **kwargs)


@pytest.mark.parametrize(("name", "seed"), sorted(GOLDEN_DIGESTS))
def test_golden_digests(name: str, seed: int) -> None:
    expected, steps = GOLDEN_DIGESTS[name, seed]
    assert golden_rollout(_golden_env(name), seed, 250) == (expected, steps)


def test_golden_digests_are_sensitive() -> None:
    base = golden_rollout(_golden_env("default"), 0, 20)
    assert golden_rollout(_golden_env("default"), 1, 20) != base
    assert golden_rollout(BlackjackEnv(rules=Rules(decks=5)), 0, 20) != base
    assert golden_rollout(BlackjackEnv(bet_levels=(1, 2, 4, 9)), 0, 20) != base
