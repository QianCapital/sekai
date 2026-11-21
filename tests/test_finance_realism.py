"""Corporate actions, trading sessions and margin in MarketData and Market-v1."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import sekai
from sekai.envs.finance import BatchMarketEnv
from sekai.finance import MarketData, NoCost, ProportionalCost, TargetWeights


def raw_history(seed: int = 0) -> MarketData:
    """Raw prices of 4 assets with splits and dividends."""
    rng = np.random.default_rng(seed)
    total = np.cumprod(1 + rng.normal(0.0003, 0.01, (300, 4)), axis=0) * 50
    split = np.full((300, 4), np.nan)
    split[100, 0], split[200, 1], split[150, 2] = 2.0, 3.0, 0.5  # 2:1, 3:1, 1:2 reverse
    dividend = np.full((300, 4), np.nan)
    dividend[80::60, 3] = 0.4
    shares = np.ones(4)  # raw price = total value / shares held since the start
    raw = np.empty_like(total)
    for t in range(300):
        shares *= np.where(np.isfinite(split[t]), split[t], 1.0)
        raw[t] = total[t] / shares
    # Dividends paid out lower the raw price by the cash paid.
    paid = np.cumsum(np.nan_to_num(dividend), axis=0)
    raw = raw - paid
    return MarketData(raw, fields={"split": split, "dividend": dividend, "open": raw})


def test_total_returns() -> None:
    data = raw_history()
    r = data.simple_returns()
    p = data.prices
    np.testing.assert_allclose(r[100, 0], 2.0 * p[100, 0] / p[99, 0] - 1)
    np.testing.assert_allclose(r[80, 3], (p[80, 3] + 0.4) / p[79, 3] - 1)
    np.testing.assert_allclose(r[50], p[50] / p[49] - 1)
    assert data.has_corporate_actions and not data.adjusted().has_corporate_actions
    adj = data.adjusted()
    np.testing.assert_allclose(adj.prices[1:] / adj.prices[:-1] - 1, r[1:], rtol=1e-12)
    np.testing.assert_array_equal(adj.prices[0], p[0])  # forward adjusted: starts at the raw price
    assert "open" in adj.fields and "split" not in adj.fields
    with pytest.raises(ValueError, match="split field"):
        MarketData(np.ones((3, 1)), fields={"split": np.array([[np.nan], [-2.0], [np.nan]])})
    with pytest.raises(ValueError, match="dividend field"):
        MarketData(np.ones((3, 1)), fields={"dividend": np.array([[np.nan], [-1.0], [np.nan]])})


def test_adjusted_restarts_each_listed_stretch() -> None:
    p = np.array([[10.0], [11.0], [np.nan], [20.0], [22.0]])
    np.testing.assert_allclose(MarketData(p).adjusted().prices[:, 0], [10, 11, np.nan, 20, 22])


@pytest.mark.parametrize("execution", ["close", "next_close"])
def test_raw_prices_with_actions_trade_like_adjusted_prices(execution: str) -> None:
    raw = raw_history()
    kw: dict[str, Any] = dict(
        lookback=5,
        episode_length=250,
        execution=execution,
        transaction_cost=1e-3,
        action=TargetWeights(1.5),
    )
    a = sekai.make("Market-v1", data=raw, **kw)
    b = sekai.make("Market-v1", data=raw.adjusted(), **kw)
    a.reset(seed=0, options={"start": 10})
    b.reset(seed=0, options={"start": 10})
    rng = np.random.default_rng(0)
    for _ in range(250):
        act = rng.uniform(-1, 1, 4).astype(np.float32)
        x, y = a.step(act), b.step(act)
        assert x.info["equity"] == pytest.approx(y.info["equity"], rel=1e-12)
        if x.done:
            break


def test_next_open_overnight_move_includes_corporate_actions() -> None:
    p = np.array([[100.0], [100.0], [50.0], [50.0]])
    split = np.array([[np.nan], [np.nan], [2.0], [np.nan]])
    data = MarketData(p, fields={"open": p, "split": split})
    env = BatchMarketEnv(
        1,
        data=data,
        lookback=1,
        episode_length=2,
        execution="next_open",
        transaction_cost=NoCost(),
        action=TargetWeights(),
    )
    env.reset(seed=0, options={"start": 1})
    env.step(np.array([[1.0]]))  # long from the open of row 2: no change in value over the split
    out = env.step(np.array([[1.0]]))
    assert float(out.info["equity"][0]) == pytest.approx(1.0)


def intraday(days: int = 6, bars: int = 4) -> MarketData:
    stamps = (
        np.datetime64("2024-03-04T10:00")
        + np.arange(days)[:, None] * np.timedelta64(1, "D")
        + np.arange(bars) * np.timedelta64(90, "m")
    ).ravel()
    rng = np.random.default_rng(1)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, (days * bars, 3)), axis=0))
    return MarketData(prices, timestamps=stamps)


def test_sessions() -> None:
    data = intraday()
    np.testing.assert_array_equal(data.sessions(), np.repeat(np.arange(6), 4))
    labelled = MarketData(np.ones((5, 1)), exogenous={"session": [7, 7, 3, 3, 3]})
    np.testing.assert_array_equal(labelled.sessions(), [0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="numeric"):
        MarketData(np.ones((5, 1))).sessions()


def test_flat_overnight() -> None:
    data = intraday()
    env = sekai.make(
        "Market-v1",
        data=data,
        lookback=2,
        episode_length=18,
        execution="close",
        flat_overnight=True,
        transaction_cost=ProportionalCost(1e-4),
    )
    env.reset(seed=0, options={"start": 2})
    sessions = data.sessions()
    for k in range(18):
        out = env.step(np.ones(4, np.float32))
        row = 2 + k
        assert bool(out.info["session_end"]) == (sessions[row + 1] != sessions[row])
        if out.info["session_end"]:
            assert np.all(out.info["weights"] == 0.0)  # nothing held into the next session
        if out.done:
            break
    with pytest.raises(ValueError, match="flat_overnight needs"):
        BatchMarketEnv(1, data=data, execution="next_open", flat_overnight=True)


def test_margin() -> None:
    rng = np.random.default_rng(2)
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.03, (120, 3)), axis=0))
    data = MarketData(p)
    kw: dict[str, Any] = dict(
        data=data,
        lookback=2,
        episode_length=100,
        execution="close",
        action=TargetWeights(3.0),
        transaction_cost=1e-3,
    )
    env = BatchMarketEnv(1, initial_margin=0.5, **kw)
    env.reset(seed=0, options={"start": 5})
    out = env.step(np.array([[1.0, 1.0, 1.0]]))  # asks for 3x gross: capped at 1 / 0.5
    assert float(np.abs(env.weights).sum()) == pytest.approx(2.0, rel=0.1)
    calls = 0
    env = BatchMarketEnv(1, initial_margin=0.5, maintenance_margin=0.48, **kw)
    env.reset(seed=0, options={"start": 5})
    for _ in range(100):
        out = env.step(np.array([[1.0, -1.0, 1.0]]))
        if bool(out.info["margin_call"][0]):
            calls += 1
            assert float(np.abs(env.weights).sum()) == pytest.approx(2.0)
        assert float(np.abs(env.weights).sum()) <= 1 / 0.48 + 1e-9
        if bool(out.dones[0]):
            break
    assert calls > 0
    with pytest.raises(ValueError, match="cannot exceed"):
        BatchMarketEnv(1, data=data, initial_margin=0.3, maintenance_margin=0.4)
    assert (
        BatchMarketEnv(1, data=data, episode_length=50, maintenance_margin=0.25).initial_margin
        == 0.5
    )
