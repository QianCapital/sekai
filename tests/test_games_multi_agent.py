"""MultiAgentGame: the two-player games as sekai multi-agent environments."""

from __future__ import annotations

import copy
import pickle
from typing import Any

import numpy as np
import pytest

import sekai
from sekai.envs.games import ChessRenderer, HoldemRenderer, IllegalMove, MultiAgentGame
from sekai.envs.games.chess import ChessEnv
from sekai.envs.games.poker import ALL_IN, CHECK_CALL, FOLD, HoldemEnv
from sekai.wrappers import (
    ActionRepeat,
    AutoReset,
    FlattenObservation,
    FrameStackObservation,
    RecordEpisodeStatistics,
    TimeLimit,
)

GAMES = ["Chess-v0", "Go9x9-v0", "Xiangqi-v0", "HeadsUpHoldem-v0"]
DECK = "AsKdQhQc2c7d9hJs3s"


def play(env: MultiAgentGame, seed: int) -> list[Any]:
    """A random game, checking the protocol at every step; the results, reset first."""
    results: list[Any] = [env.reset(seed=seed)]
    for agent in env.possible_agents:
        env.action_space(agent).seed(seed)
    while not results[-1].all_done:
        result = results[-1]
        (agent,) = result.observations  # only the agent to move
        assert env.observation_space(agent).contains(result.observations[agent])
        other = next(a for a in env.possible_agents if a != agent)
        assert np.array_equal(env.action_mask(agent), result.info[agent]["legal_action_mask"])
        assert not env.action_mask(other).any() and not result.info[other]["legal_actions"].size
        action = env.action_space(agent).sample(mask=env.action_mask(agent))
        results.append(env.step({agent: action}))
        step = results[-1]
        assert set(step.rewards) == set(step.terminated) == set(step.info) == {agent, other}
        assert step.rewards[other] == -step.rewards[agent]
    return results


@pytest.mark.parametrize("env_id", GAMES)
def test_a_game_is_played_one_agent_at_a_time(env_id: str) -> None:
    env = MultiAgentGame(sekai.make(env_id))
    assert env.possible_agents == list(env.unwrapped.sides) and env.agents == []
    for seed in range(3):
        results = play(env, seed)
        last = results[-1]
        assert set(last.observations) == set(env.possible_agents) and env.agents == []
        assert sum(last.rewards.values()) == 0.0
        with pytest.raises(sekai.ResetRequired):
            env.step({env.possible_agents[0]: 0})


@pytest.mark.parametrize("env_id", GAMES)
def test_its_moves_and_rewards_are_the_games(env_id: str) -> None:
    # The same seeded game, self-play and multi-agent: the same observations and rewards.
    game, env = sekai.make(env_id), MultiAgentGame(sekai.make(env_id))
    rng = np.random.default_rng(0)
    single, multi = game.reset(seed=4), env.reset(seed=4)
    while not single.done:
        (agent,) = multi.observations
        np.testing.assert_array_equal(multi.observations[agent], single.observation)
        action = int(rng.choice(np.flatnonzero(single.info["legal_action_mask"])))
        single, multi = game.step(action), env.step({agent: action})
        assert multi.rewards[agent] == single.reward
    assert multi.all_done
    assert multi.terminated == dict.fromkeys(env.possible_agents, single.terminated)


def test_poker_agents_see_only_their_own_seat() -> None:
    env = MultiAgentGame(HoldemEnv())
    result = env.reset(options={"button": 0, "deck": DECK})
    base = env.unwrapped
    for agent, seat in (("p1", 0), ("p2", 1)):
        assert result.info[agent]["observer"] == seat and result.info[agent]["view"].seat == seat
        assert result.info[agent]["view"].holes[1 - seat] is None
    result = env.step({"p1": CHECK_CALL})
    result = env.step({"p2": FOLD if env.action_mask("p2")[FOLD] else ALL_IN})
    if not result.all_done:
        result = env.step({"p1": FOLD})
    assert result.all_done
    for agent, seat in (("p1", 0), ("p2", 1)):  # the final observations are each seat's own
        np.testing.assert_array_equal(result.observations[agent], base.observe(seat))
        assert result.info[agent]["observer"] == seat
    assert result.rewards["p1"] == -result.rewards["p2"] != 0.0


def test_errors_change_nothing() -> None:
    env = MultiAgentGame(sekai.make("Chess-v0"))
    result = env.reset(seed=0)
    fen = env.unwrapped.board.fen()
    with pytest.raises(ValueError, match="white's turn"):
        env.step({"black": 0})
    with pytest.raises(ValueError, match="white's turn"):
        env.step({"white": 0, "black": 0})
    with pytest.raises(IllegalMove):
        env.step({"white": 0})  # a rook blocked by its pawn
    assert env.unwrapped.board.fen() == fen and env.agents == ["white", "black"]
    mask = result.info["white"]["legal_action_mask"]
    np.testing.assert_array_equal(env.action_mask("white"), mask)
    with pytest.raises(KeyError):
        env.action_mask("red")
    # A forfeit in "lose" mode ends the game for both agents.
    lose = MultiAgentGame(ChessEnv(illegal_action="lose"))
    lose.reset(seed=0)
    result = lose.step({"white": 0})
    assert result.all_done and result.rewards == {"white": -1.0, "black": 1.0}


def test_time_limits_resets_into_finished_games_and_copies() -> None:
    env = MultiAgentGame(TimeLimit(ChessEnv(), max_episode_steps=3))
    env.reset(seed=0)
    for agent in ("white", "black", "white"):
        result = env.step({agent: int(np.flatnonzero(env.action_mask(agent))[0])})
    assert result.all_done and result.truncated == {"white": True, "black": True}
    # A mated side cannot move: nobody is to move.
    mated = env.reset(options={"fen": "7k/6Q1/6K1/8/8/8/8/8 b - - 0 1"})
    assert mated.observations == {} and env.agents == [] and not env.action_mask("black").any()
    env.reset(seed=1)
    env.step({"white": int(np.flatnonzero(env.action_mask("white"))[0])})
    twin, dup = copy.deepcopy(env), pickle.loads(pickle.dumps(env))
    action = int(np.flatnonzero(env.action_mask("black"))[-1])
    expected = env.step({"black": action})
    for other in (twin, dup):
        result = other.step({"black": action})
        np.testing.assert_array_equal(result.observations["white"], expected.observations["white"])


def test_renderers_draw_it_and_poker_refuses_observation_wrappers_inside() -> None:
    chess = MultiAgentGame(sekai.make("Chess-v0"))
    chess.reset(seed=0)
    assert ChessRenderer().render(chess).ndim == 3 and "MultiAgentGame" in str(chess)
    poker = MultiAgentGame(HoldemEnv())
    poker.reset(seed=0)
    assert HoldemRenderer().render(poker).dtype == np.uint8
    with pytest.raises(TypeError, match="two-player game"):
        MultiAgentGame(sekai.make("Blackjack-v0"))


def test_wrappers_that_cannot_work_inside_are_refused() -> None:
    # Those that work across steps would act on both players' moves (or hide the games'
    # ends), and in poker any that changes the observations sees both seats' views.
    across = [
        AutoReset(ChessEnv()),
        RecordEpisodeStatistics(ChessEnv()),
        FrameStackObservation(ChessEnv(), 2),
        ActionRepeat(HoldemEnv(), 2),
    ]
    for inner in across:
        with pytest.raises(TypeError, match=r"would act on both players' moves.*its learner"):
            MultiAgentGame(inner)
    with pytest.raises(TypeError, match="both seats' private views"):
        MultiAgentGame(FlattenObservation(HoldemEnv()))
    with pytest.raises(TypeError, match="needs the game itself"):
        MultiAgentGame(sekai.make("ChessVsRandom-v0"))
    flat = MultiAgentGame(FlattenObservation(ChessEnv()))  # a perfect-information game's own
    assert flat.reset(seed=0).observations["white"].shape == (119 * 8 * 8,)


def test_a_reset_that_raises_leaves_no_game() -> None:
    env = MultiAgentGame(sekai.make("Go9x9-v0"))
    env.reset(seed=0)
    with pytest.raises(IllegalMove):
        env.reset(options={"moves": ["D4", "D4"]})
    assert env.agents == [] and not env.action_mask("black").any()
    with pytest.raises(sekai.ResetRequired):
        env.step({"black": 0})


def test_to_pettingzoo() -> None:
    pytest.importorskip("pettingzoo")
    from sekai.compat import to_pettingzoo

    env = to_pettingzoo(MultiAgentGame(sekai.make("Go9x9-v0")))
    observations, infos = env.reset(seed=0)
    assert list(observations) == ["black"] and env.possible_agents == ["black", "white"]
    while env.agents:
        (agent,) = observations
        mask = infos[agent]["legal_action_mask"]
        observations, rewards, _, _, infos = env.step(
            {agent: int(np.flatnonzero(mask)[-1])}  # passing ends the game soon
        )
    assert set(rewards) == {"black", "white"} and sum(rewards.values()) == 0.0
