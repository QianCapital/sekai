"""Trading-v1 and Portfolio-v1: orders fill at the next close, not the close just observed."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import sekai
from sekai.finance import SpreadCost

PRICES = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, (400, 3)), axis=0))


def equities(env: sekai.Env, actions: list[Any], start: int) -> list[float]:
    env.reset(seed=0, options={"start": start})
    out = []
    for a in actions:
        result = env.step(a)
        out.append(float(result.info["equity"]))
        if result.done:
            break
    return out


@pytest.mark.parametrize("cost", [1e-3, SpreadCost(half_spread=5e-4)])
def test_trading_next_close_is_close_one_step_later(cost: Any) -> None:
    kw: dict[str, Any] = dict(
        prices=PRICES[:, 0], episode_length=50, lookback=10, transaction_cost=cost
    )
    close, delayed = sekai.make("Trading-v0", **kw), sekai.make("Trading-v1", **kw)
    actions = list(np.random.default_rng(1).integers(0, 3, 50))
    flat = 1  # the index of position 0
    np.testing.assert_allclose(
        equities(delayed, actions, 40),
        equities(close, [flat, *actions[:-1]], 40),
        rtol=0,
        atol=1e-15,
    )


@pytest.mark.parametrize("allow_short", [False, True])
def test_portfolio_next_close_is_close_one_step_later(allow_short: bool) -> None:
    kw: dict[str, Any] = dict(
        prices=PRICES, episode_length=50, lookback=10, allow_short=allow_short
    )
    close, delayed = sekai.make("Portfolio-v0", **kw), sekai.make("Portfolio-v1", **kw)
    delayed.action_space.seed(0)
    actions = [delayed.action_space.sample() for _ in range(50)]
    cash = np.zeros(3, np.float32) if allow_short else np.array([0, 0, 0, 1], np.float32)
    np.testing.assert_allclose(
        equities(delayed, actions, 40),
        equities(close, [cash, *actions[:-1]], 40),
        rtol=0,
        atol=1e-15,
    )


def test_pending_order_is_observed() -> None:
    env = sekai.make("Trading-v1", prices=PRICES[:, 0], episode_length=20, lookback=5)
    obs = env.reset(seed=0).observation
    assert obs.shape == (7,) and obs[-1] == 0.0  # no order pending yet
    result = env.step(2)  # long, filling at the next close
    assert result.observation[-1] == 1.0 and result.info["pending"] == 1.0
    assert result.info["position"] == 0.0 and result.info["turnover"] == 0.0
    result = env.step(2)
    assert result.info["turnover"] == 1.0 and result.info["position"] > 0.0
    portfolio = sekai.make("Portfolio-v1", prices=PRICES, episode_length=20, lookback=5)
    out = portfolio.reset(seed=0).observation
    np.testing.assert_array_equal(out["pending"], [0, 0, 0, 1])
    out = portfolio.step(np.array([1, 0, 0, 0], np.float32)).observation
    np.testing.assert_array_equal(out["pending"], [1, 0, 0, 0])
    with pytest.raises(ValueError, match="execution must be"):
        sekai.make("Trading-v0", execution="next_open")


@pytest.mark.parametrize("env_id", ["Trading-v1", "Portfolio-v1"])
def test_native_vector_env_matches_separate_envs(env_id: str) -> None:
    prices = PRICES[:, 0] if env_id == "Trading-v1" else PRICES
    kw: dict[str, Any] = dict(prices=prices, episode_length=30, lookback=10, sampler="sequential")
    native = sekai.make_vec(env_id, num_envs=3, **kw)
    sync = sekai.make_vec(env_id, num_envs=3, vectorization_mode="sync", **kw)
    assert (
        type(native).__name__.endswith("VecEnv")
        and native.get_attr("execution") == ["next_close"] * 3
    )
    native.reset(seed=0)
    sync.reset(seed=0)
    native.batch_action_space.seed(0)
    for _ in range(70):  # through autoresets
        actions = native.batch_action_space.sample()
        x, y = native.step(actions), sync.step(actions)
        np.testing.assert_allclose(x.rewards, y.rewards, rtol=0, atol=1e-15)
        np.testing.assert_array_equal(x.dones, y.dones)
        for i in range(3):
            assert x.infos[i]["equity"] == pytest.approx(y.infos[i]["equity"], abs=1e-15)
            np.testing.assert_array_equal(x.infos[i]["pending"], y.infos[i]["pending"])
    native.close()
    sync.close()


@pytest.mark.parametrize("env_id", ["Trading-v1", "Portfolio-v1"])
def test_check_env(env_id: str) -> None:
    assert sekai.check_env(sekai.make(env_id), check_determinism=True) == []
    vec = sekai.make_vec(env_id, num_envs=4)
    vec.reset(seed=0)
    vec.step(vec.batch_action_space.sample())
    vec.close()
