"""Objects pickled by sekai 0.2.0 load with this sekai and carry on as they did there.

The fixture is made by `tests/data/make_pickles_0_2_0.py` under sekai 0.2.0 and NumPy 1.x.
Values are compared to rounding, since this NumPy may differ from the one that made it.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import sekai
from sekai.finance import MarketData, MarketPanel

ITEMS: dict[str, tuple[bytes, Any]] = pickle.loads(
    (Path(__file__).parent / "data" / "pickles_0_2_0.pkl").read_bytes()
)["items"]
ENVS = [k for k in ITEMS if k.split(":")[0] in ("env", "wrapped", "vec")]


def same(got: Any, want: Any) -> None:
    if isinstance(want, dict):
        assert set(got) == set(want)
        for key in want:
            same(got[key], want[key])
    elif isinstance(want, (list, tuple)):
        assert len(got) == len(want)
        for a, b in zip(got, want):
            same(a, b)
    elif want is None:
        assert got is None
    else:
        np.testing.assert_allclose(
            np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64), rtol=1e-10
        )


@pytest.mark.parametrize("name", ENVS)
def test_environments_carry_on(name: str) -> None:
    blob, expected = ITEMS[name]
    env = pickle.loads(blob)
    for action, (observation, reward) in zip(expected["actions"], expected["results"]):
        result = env.step(action)
        same(getattr(result, "observations", getattr(result, "observation", None)), observation)
        same(getattr(result, "rewards", getattr(result, "reward", None)), reward)
        if getattr(result, "done", False) is True:
            env.reset()
    if hasattr(env, "snapshot") and not name.startswith("vec"):
        env.restore(env.snapshot())  # and the 0.3 additions work on them


def test_market_data() -> None:
    blob, expected = ITEMS["data"]
    data = pickle.loads(blob)
    same(data.prices, expected["prices"])
    same(data.log_returns(), expected["log_returns"])
    same(data[10:90].prices, expected["slice"])
    assert not data.prices.flags.writeable and not data.has_corporate_actions
    np.testing.assert_allclose(data.simple_returns()[1:], data.prices[1:] / data.prices[:-1] - 1)
    MarketPanel.from_market_data(data)
    env = sekai.make("Market-v1", data=data, lookback=5, episode_length=20)
    env.reset(seed=0)
    env.step(env.action_space.sample())
    split_blob, split_expected = ITEMS["data:split"]
    same(pickle.loads(split_blob).prices, split_expected["prices"])


def test_market_data_saves(tmp_path: Path) -> None:
    data = pickle.loads(ITEMS["data"][0])
    again = MarketData.load(data.save(tmp_path / "saved"))
    np.testing.assert_array_equal(again.prices, data.prices)


@pytest.mark.parametrize("name", ["cost:spread", "cost:composite"])
def test_cost_models(name: str) -> None:
    blob, expected = ITEMS[name]
    model = pickle.loads(blob)
    same(model(expected["trades"], **expected["context"]), expected["cost"])
    repr(model)


def test_stateful_reward_and_model() -> None:
    blob, expected = ITEMS["reward"]
    reward = pickle.loads(blob)
    same([reward(o) for o in expected["outcomes"]], expected["values"])
    repr(reward)
    reward.reset(np.array([True, False]))
    reward.detach_state()
    model_blob, model_expected = ITEMS["model"]
    same(
        pickle.loads(model_blob).simulate(np.random.default_rng(5), 10).prices,
        model_expected["paths"],
    )
