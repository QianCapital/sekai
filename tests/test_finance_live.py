"""Paper and live trading: feeds, brokers, risk limits, and parity with Market-v1."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import sekai
from sekai.envs.finance import LiveMarketEnv
from sekai.finance import (
    GBM,
    Bar,
    BorrowFee,
    Broker,
    DataFeed,
    GuardedBroker,
    ListingProcess,
    MarketData,
    NoCost,
    PaperBroker,
    ReplayFeed,
    RiskLimits,
    SpreadCost,
    SquareRootImpact,
    TargetWeights,
    orders_for,
)


def history(n_steps: int = 400, seed: int = 0) -> MarketData:
    rng = np.random.default_rng(seed)
    sim = ListingProcess(GBM(mu=0.05, sigma=rng.uniform(0.15, 0.4, 6))).simulate(rng, n_steps - 1)
    shape = sim.prices.shape
    return MarketData(
        sim.prices,
        timestamps=np.busday_offset("2020-01-01", np.arange(n_steps), roll="forward"),
        fields={
            "volume": rng.lognormal(12, 1, shape),
            "spread": rng.uniform(1e-4, 1e-3, shape),
            "borrow_rate": rng.uniform(0, 0.2, shape),
        },
    )


DATA = history()


def with_actions(data: MarketData) -> MarketData:
    """`data` with a few splits and dividends, as corporate actions."""
    split = np.full(data.prices.shape, np.nan)
    split[120, 0], split[90, 3] = 2.0, 0.5
    dividend = np.full(data.prices.shape, np.nan)
    dividend[60::40, 1] = 0.3
    fields = {k: v for k, v in data.fields.items() if k != data.price_field}
    return MarketData(
        data.prices,
        fields={**fields, "split": split, "dividend": dividend},
        timestamps=data.timestamps,
        corporate_actions=True,
    )


@pytest.mark.parametrize("actions", [False, True])
@pytest.mark.parametrize("execution", ["next_close", "close"])
def test_paper_trading_reproduces_market_v1(execution: str, actions: bool) -> None:
    data = with_actions(DATA) if actions else DATA
    lookback, start = 10, 50
    cost = SpreadCost() + SquareRootImpact(0.3)
    account: dict[str, Any] = dict(
        cash_rate=0.02, borrow_rate=0.01, capital=5e6, delisting_return=-0.3
    )
    common: dict[str, Any] = dict(
        lookback=lookback,
        episode_length=100,
        execution=execution,
        features=("volume",),
        action=TargetWeights(1.5),
    )
    batch = sekai.make(
        "Market-v1", data=data, transaction_cost=cost, holding_cost=BorrowFee(), **account, **common
    )
    dt = batch.unwrapped.batch_env.dt
    broker = PaperBroker(transaction_cost=cost, holding_cost=BorrowFee(), dt=dt, **account)
    live = LiveMarketEnv(ReplayFeed(data, start=start - lookback), broker, dt=dt, **common)
    a, b = batch.reset(seed=0, options={"start": start}), live.reset()
    np.testing.assert_allclose(
        b.observation["assets"]["values"], a.observation["assets"]["values"], atol=1e-6
    )
    rng = np.random.default_rng(1)
    for _ in range(100):
        action = rng.uniform(-1, 1, 6).astype(np.float32)
        x, y = batch.step(action), live.step(action)
        assert y.reward == pytest.approx(x.reward, abs=1e-14)
        for key in ("equity", "cost", "turnover", "financing", "gross_exposure", "n_assets"):
            assert float(y.info[key]) == pytest.approx(float(x.info[key]), abs=1e-14), key
        np.testing.assert_allclose(y.info["weights"], x.info["weights"], atol=1e-14)
        assert y.info["timestamp"] == x.info["timestamp"]
        for key in x.observation:
            np.testing.assert_allclose(
                np.asarray(y.observation[key] if key != "assets" else y.observation[key]["values"]),
                np.asarray(x.observation[key] if key != "assets" else x.observation[key]["values"]),
                atol=1e-6,
            )
        assert (x.terminated, x.truncated) == (y.terminated, y.truncated)
        if x.done:
            break


def test_feed_end_truncates() -> None:
    env = LiveMarketEnv(ReplayFeed(DATA, start=0, stop=13), PaperBroker(), lookback=10)
    env.reset()
    assert not env.step(env.action_space.sample()).done
    assert not env.step(env.action_space.sample()).done
    last = env.step(env.action_space.sample())
    assert last.truncated and last.info["feed_ended"] and last.reward == 0.0
    with pytest.raises(sekai.ResetRequired):
        env.step(env.action_space.sample())
    with pytest.raises(RuntimeError, match="feed ended before"):
        LiveMarketEnv(ReplayFeed(DATA, start=0, stop=5), PaperBroker(), lookback=10).reset()


def test_protocols_and_replay_feed() -> None:
    feed = ReplayFeed(DATA, start=3, stop=5)
    assert isinstance(feed, DataFeed) and isinstance(PaperBroker(), Broker)
    bar = feed.next_bar()
    assert isinstance(bar, Bar) and bar.timestamp == DATA.timestamps[3]
    np.testing.assert_array_equal(bar.prices, DATA.prices[3])
    assert set(bar.fields) == {"volume", "spread", "borrow_rate"}
    assert feed.next_bar() is not None and feed.next_bar() is None
    with pytest.raises(ValueError):
        ReplayFeed(DATA, start=10, stop=5)


def test_paper_broker_accounting() -> None:
    broker = PaperBroker(transaction_cost=NoCost(), capital=1e6)
    state = broker.reset(("a", "b"))
    assert state.equity == 1.0 and state.value == 1e6
    bar0 = Bar(0, np.array([100.0, 50.0]))
    broker.execute(np.array([0.5, -0.5]), bar0, {})
    state = broker.mark(Bar(1, np.array([110.0, 40.0])))
    assert state.equity == pytest.approx(1 + 0.5 * 0.1 + 0.5 * 0.2)
    with pytest.raises(RuntimeError, match="execute"):
        broker.mark(Bar(2, np.array([1.0, 1.0])))
    split = PaperBroker(transaction_cost=NoCost())
    split.reset(("a",))
    split.execute(np.array([1.0]), Bar(0, np.array([100.0])), {})
    two_for_one = {"split": np.array([2.0])}
    state = split.mark(Bar(1, np.array([50.0]), fields=two_for_one, corporate_actions=True))
    assert state.equity == pytest.approx(1.0)  # a 2-for-1 split is no loss


def test_orders_for() -> None:
    orders = orders_for(
        np.array([0.5, -0.25, 0.1]),
        100_000,
        np.array([100.0, 50.0, np.nan]),
        np.array([100.0, 0.0, 5.0]),
        lot_size=10,
    )
    np.testing.assert_array_equal(orders, [400.0, -500.0, 0.0])


def test_risk_limits() -> None:
    limits = RiskLimits(max_position=0.3, max_gross=1.0, max_net=0.2)
    w, hit = limits.apply(np.array([0.5, 0.4, 0.3, -0.1]), np.zeros(4))
    assert (
        np.abs(w).max() <= 0.3 + 1e-12
        and np.abs(w).sum() <= 1.0 + 1e-12
        and abs(w.sum()) <= 0.2 + 1e-12
    )
    np.testing.assert_allclose(w, [0.1, 0.1, 0.1, -0.1])  # capped, then longs scaled to net 0.2
    assert hit == ["max_position", "max_net"]
    w, hit = RiskLimits(max_turnover=0.2).apply(np.array([1.0, 0.0]), np.array([0.0, 0.0]))
    np.testing.assert_allclose(w, [0.2, 0.0])
    assert hit == ["max_turnover"]
    w, _ = RiskLimits(allow_short=False).apply(np.array([0.5, -0.5]), np.zeros(2))
    np.testing.assert_array_equal(w, [0.5, 0.0])
    with pytest.raises(ValueError):
        RiskLimits(max_drawdown=0.0)


def test_guarded_broker_kill_switch() -> None:
    p = np.array([[100.0]] * 3 + [[80.0]] + [[100.0]] * 6)
    data = MarketData(p)
    broker = GuardedBroker(
        PaperBroker(transaction_cost=NoCost()), RiskLimits(max_gross=1.0, max_drawdown=0.1)
    )
    env = LiveMarketEnv(
        ReplayFeed(data), broker, lookback=2, action=TargetWeights(2.0), execution="close"
    )
    env.reset()
    out = env.step(np.array([1.0], np.float32))  # fully long into the 20% drop
    assert out.info["equity"] == pytest.approx(0.8)
    assert broker.killed and broker.events[-1].kind == "kill_switch"
    out = env.step(np.array([1.0], np.float32))
    assert out.info["gross_exposure"] == 0.0 and out.info["equity"] == pytest.approx(0.8)
    assert broker.events[-1].detail == "account kept flat"
    broker.reset(("asset_0",))
    assert not broker.killed
    assert broker.capital == 1_000_000.0  # the inner broker's attributes show through


def test_guarded_broker_clips_targets() -> None:
    broker = GuardedBroker(PaperBroker(transaction_cost=NoCost()), RiskLimits(max_gross=0.5))
    env = LiveMarketEnv(
        ReplayFeed(DATA), broker, lookback=5, action=TargetWeights(1.0), execution="close"
    )
    env.reset()
    out = env.step(np.where(env.action_mask() > 0, 1.0, 0.0).astype(np.float32))
    assert out.info["gross_exposure"] <= 0.5 * 1.1
    assert any(e.kind == "max_gross" for e in broker.events)
