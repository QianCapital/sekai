"""PettingZoo interop (skipped without pettingzoo)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

pytest.importorskip("pettingzoo")

from gymnasium import spaces as gspaces
from gymnasium.utils import seeding
from pettingzoo.utils.env import ParallelEnv

from sekai.compat import from_pettingzoo, to_pettingzoo
from sekai.envs.finance import LimitOrderBookMarket


class CoinFlips(ParallelEnv):
    """A PettingZoo env seeded the usual way, through gymnasium's seeding (int seeds only)."""

    metadata = {"name": "coin_flips"}

    def __init__(self) -> None:
        self.possible_agents = ["p0", "p1"]
        self.agents: list[str] = []

    def observation_space(self, agent: str) -> gspaces.Space[Any]:
        return gspaces.Discrete(2)

    def action_space(self, agent: str) -> gspaces.Space[Any]:
        return gspaces.Discrete(2)

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.np_random, _ = seeding.np_random(seed)
        self.agents = list(self.possible_agents)
        obs = {a: int(self.np_random.integers(2)) for a in self.agents}
        return obs, {a: {} for a in self.agents}

    def step(self, actions: dict[str, Any]) -> tuple[dict[str, Any], ...]:
        obs = {a: int(self.np_random.integers(2)) for a in self.agents}
        done = dict.fromkeys(self.agents, True)
        infos: dict[str, Any] = {a: {} for a in self.agents}
        self.agents = []
        return obs, dict.fromkeys(obs, 1.0), done, dict.fromkeys(obs, False), infos


def test_round_trip_through_pettingzoo() -> None:
    from pettingzoo.test import parallel_api_test

    from tests.conftest_ma import QuoteAndTrade

    pz = to_pettingzoo(QuoteAndTrade())
    parallel_api_test(pz, num_cycles=50)
    back = from_pettingzoo(pz)
    r = back.reset(seed=0)
    assert set(r.observations) == {"maker", "trader"}
    step = back.step({a: back.action_space(a).sample() for a in back.agents})
    assert set(step.rewards) == {"maker", "trader"}


def test_event_driven_market_through_pettingzoo() -> None:
    pz = to_pettingzoo(LimitOrderBookMarket(2, decision_intervals=[1.0, 2.0], session_length=10))
    observations, _ = pz.reset(seed=0)
    assert set(observations) == {"trader_0", "trader_1"}
    while pz.agents:
        actions = {a: pz.action_space(a).sample() for a in observations}
        observations, rewards, terminations, _, _ = pz.step(actions)
        assert set(rewards) == {"trader_0", "trader_1"}
    assert all(terminations.values())


def test_from_pettingzoo_numpy_integer_seed() -> None:
    env = from_pettingzoo(CoinFlips())
    first = env.reset(seed=np.int64(7)).observations  # gymnasium's seeding rejects np.int64
    assert env._rng_seed == 7 and type(env._rng_seed) is int
    assert env.reset(seed=7).observations == first


def test_from_pettingzoo_generator_seed_is_a_clear_error() -> None:
    env = from_pettingzoo(CoinFlips())
    with pytest.raises(TypeError, match="int seed, not a numpy Generator"):
        env.reset(seed=np.random.default_rng(0))
