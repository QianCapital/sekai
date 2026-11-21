"""Paper and live trading: feeds, brokers, risk limits, and parity with Market-v1."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest

import sekai
from sekai.envs.finance import LiveMarketEnv
from sekai.finance import (
    GBM,
    AccountState,
    Bar,
    BorrowFee,
    Broker,
    DataFeed,
    ExecutionReport,
    GuardedBroker,
    ListingProcess,
    MarketData,
    NoCost,
    PaperBroker,
    ReplayFeed,
    RewardFunction,
    RiskLimits,
    SpreadCost,
    SquareRootImpact,
    StepOutcome,
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


class ListFeed:
    """A feed of bars built from rows of prices (and optional per-bar fields)."""

    def __init__(
        self,
        prices: Sequence[Sequence[float]],
        assets: tuple[Any, ...] = ("a", "b"),
        *,
        fields: Sequence[Mapping[str, Any]] | None = None,
        warmup: int = 0,
        **attributes: Any,
    ) -> None:
        self.assets = assets
        self.rows = [np.asarray(p, dtype=np.float64) for p in prices]
        self.fields = list(fields) if fields is not None else [{}] * len(self.rows)
        self.warmup = warmup
        self.position = 0
        for name, value in attributes.items():
            setattr(self, name, value)

    def next_bar(self) -> Bar | None:
        t = self.position
        if t >= len(self.rows):
            return None
        self.position += 1
        return Bar(t, self.rows[t], fields=self.fields[t], warmup=t < self.warmup)


def assert_same_step(x: Any, y: Any) -> None:
    """A Market-v1 step `x` and a LiveMarketEnv step `y` agree."""
    assert y.reward == pytest.approx(x.reward, abs=1e-14)
    for key in ("equity", "cost", "turnover", "financing", "gross_exposure", "n_assets"):
        assert float(y.info[key]) == pytest.approx(float(x.info[key]), abs=1e-14), key
    np.testing.assert_allclose(y.info["weights"], x.info["weights"], atol=1e-14)
    assert y.info["timestamp"] == x.info["timestamp"]
    assert_same_observation(x.observation, y.observation)
    assert (x.terminated, x.truncated) == (y.terminated, y.truncated)


def assert_same_observation(a: dict[str, Any], b: dict[str, Any]) -> None:
    assert set(a) == set(b)
    for key in a:
        x = a[key]["values"] if key == "assets" else a[key]
        y = b[key]["values"] if key == "assets" else b[key]
        np.testing.assert_allclose(np.asarray(y), np.asarray(x), atol=1e-6, err_msg=key)


def with_gaps(data: MarketData) -> MarketData:
    """`data` with assets that stop trading for a while (NaN prices), then trade again."""
    prices = np.array(data.prices)
    prices[70:75, 2] = np.nan
    prices[100, 4] = np.nan
    prices[130:132, :3] = np.nan
    fields = {k: v for k, v in data.fields.items() if k != data.price_field}
    return MarketData(prices, fields=fields, timestamps=data.timestamps)


@pytest.mark.parametrize("variant", ["plain", "actions", "gaps"])
@pytest.mark.parametrize("execution", ["next_close", "close"])
def test_paper_trading_reproduces_market_v1(execution: str, variant: str) -> None:
    data = {"plain": DATA, "actions": with_actions(DATA), "gaps": with_gaps(DATA)}[variant]
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
        assert_same_step(x, y)
        if x.done:
            break


def test_paper_trading_reads_a_volatility_field_as_market_v1_does() -> None:
    # Market-v1 prices impact with the data's volatility field when it has one.
    rng = np.random.default_rng(3)
    data = DATA.with_fields(volatility=rng.uniform(0.01, 0.05, DATA.prices.shape))
    cost = SquareRootImpact(0.5)
    common: dict[str, Any] = dict(
        lookback=10, episode_length=60, execution="close", action=TargetWeights(1.0)
    )
    batch = sekai.make("Market-v1", data=data, transaction_cost=cost, **common)
    broker = PaperBroker(transaction_cost=cost, dt=batch.unwrapped.batch_env.dt)
    live = LiveMarketEnv(ReplayFeed(data, start=40), broker, **common)
    batch.reset(seed=0, options={"start": 50})
    live.reset()
    for _ in range(60):
        action = rng.uniform(-1, 1, 6).astype(np.float32)
        x, y = batch.step(action), live.step(action)
        assert float(y.info["cost"]) == pytest.approx(float(x.info["cost"]), abs=1e-14)
        assert y.reward == pytest.approx(x.reward, abs=1e-14)
        if x.done:
            break


@pytest.mark.parametrize("lookback", [10, 40])  # shorter and longer than the history
@pytest.mark.parametrize("execution", ["next_close", "close"])
def test_paper_trading_a_split_decides_where_market_v1_does(execution: str, lookback: int) -> None:
    _, test = DATA.split(0.8, embargo=5, history=30)
    assert test.warmup == 30
    common: dict[str, Any] = dict(
        lookback=lookback, episode_length=50, execution=execution, action=TargetWeights(1.0)
    )
    cost = SpreadCost()
    batch = sekai.make(
        "Market-v1", data=test, sampler="sequential", transaction_cost=cost, **common
    )
    broker = PaperBroker(transaction_cost=cost, dt=batch.unwrapped.batch_env.dt)
    live = LiveMarketEnv(ReplayFeed(test), broker, **common)
    a, b = batch.reset(seed=0), live.reset()
    first = test.timestamps[max(test.warmup, lookback)]  # never on a history row
    assert b.info["timestamp"] == a.info["timestamp"] == first
    assert_same_observation(a.observation, b.observation)
    rng = np.random.default_rng(2)
    for _ in range(50):
        action = rng.uniform(-1, 1, 6).astype(np.float32)
        x, y = batch.step(action), live.step(action)
        assert_same_step(x, y)
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
    with pytest.raises(RuntimeError, match="feed has ended"):
        env.reset()
    with pytest.raises(RuntimeError, match="feed ended before"):
        LiveMarketEnv(ReplayFeed(DATA, start=0, stop=5), PaperBroker(), lookback=10).reset()
    _, test = DATA.split(0.8, embargo=5, history=30)
    with pytest.raises(RuntimeError, match="feed ended before"):  # only warm-up bars
        LiveMarketEnv(ReplayFeed(test, stop=30), PaperBroker(), lookback=10).reset()


def test_reset_continues_from_the_newest_bar() -> None:
    feed = ReplayFeed(DATA)
    env = LiveMarketEnv(feed, PaperBroker(), lookback=5, episode_length=3)
    r = env.reset()
    assert r.info["timestamp"] == DATA.timestamps[5]
    while not r.done:
        r = env.step(env.action_space.sample())
    position = feed.position
    again = env.reset()  # no bars are read: the window carries over
    assert feed.position == position and again.info["timestamp"] == r.info["timestamp"]
    assert again.info["equity"] == 1.0  # a PaperBroker opens a fresh account
    np.testing.assert_allclose(
        again.observation["assets"]["values"], r.observation["assets"]["values"]
    )


def test_protocols_and_replay_feed() -> None:
    feed = ReplayFeed(DATA, start=3, stop=5)
    assert isinstance(feed, DataFeed) and isinstance(PaperBroker(), Broker)
    bar = feed.next_bar()
    assert isinstance(bar, Bar) and bar.timestamp == DATA.timestamps[3] and not bar.warmup
    np.testing.assert_array_equal(bar.prices, DATA.prices[3])
    assert set(bar.fields) == {"volume", "spread", "borrow_rate"}
    assert feed.next_bar() is not None and feed.next_bar() is None
    assert feed.field_shapes == {"volume": (), "spread": (), "borrow_rate": ()}
    assert feed.exogenous_shapes == {}
    assert PaperBroker.fresh_account and isinstance(
        GuardedBroker(PaperBroker(), RiskLimits()), Broker
    )
    with pytest.raises(ValueError):
        ReplayFeed(DATA, start=10, stop=5)


def test_replay_feed_marks_the_history_rows_of_a_split() -> None:
    _, test = DATA.split(0.8, embargo=5, history=30)
    feed = ReplayFeed(test)
    flags = [bar.warmup for bar in iter(feed.next_bar, None)]
    assert flags == [t < 30 for t in range(test.n_steps)]
    assert not any(bar.warmup for bar in iter(ReplayFeed(test, start=30).next_bar, None))


def test_warm_up_bars_come_first() -> None:
    rows = [[100.0, 100.0]] * 8
    env = LiveMarketEnv(ListFeed(rows, warmup=4), PaperBroker(), lookback=2)
    assert env.reset().info["timestamp"] == 4  # the first bar that is not warm-up
    env = LiveMarketEnv(ListFeed(rows, warmup=8), PaperBroker(), lookback=2)
    with pytest.raises(RuntimeError, match="feed ended before"):  # never a bar to trade on
        env.reset()
    feed = ListFeed(rows)
    env = LiveMarketEnv(feed, PaperBroker(), lookback=2)
    env.reset()
    feed.warmup = 10  # from now on the feed says "warm-up", after bars to trade on
    with pytest.raises(ValueError, match="warm-up bar after bars to trade on"):
        env.step(env.action_space.sample())
    with pytest.raises(sekai.ResetRequired):
        env.step(env.action_space.sample())


def test_observation_space_fits_multi_column_fields() -> None:
    rng = np.random.default_rng(0)
    T, N = 40, 4
    data = MarketData(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, (T, N)), axis=0)),
        fields={"f": rng.normal(size=(T, N, 3)), "g": rng.normal(size=(T, N))},
        exogenous={"e": rng.normal(size=(T, 2, 2)), "rate": rng.normal(size=T)},
    )
    observed: dict[str, Any] = dict(features=("f", "g"), exogenous=("e", "rate"))
    env = LiveMarketEnv(ReplayFeed(data), PaperBroker(), lookback=10, **observed)
    assert env.field_shape("f") == (3,) and env.market_shape("e") == (2, 2)
    batch = sekai.make("Market-v1", data=data, lookback=10, episode_length=20, **observed)
    for key in ("assets", "market", "portfolio", "pending"):
        assert env.observation_space[key].shape == batch.observation_space[key].shape, key
    r = env.reset()
    assert env.observation_space.contains(r.observation)
    for _ in range(5):
        r = env.step(env.action_space.sample())
        assert env.observation_space.contains(r.observation)
    # A feed without the shapes: declare them, or the bars are refused.
    rows = [[100.0, 100.0]] * 6
    fields = [{"f": np.zeros((2, 3))}] * 6
    with pytest.raises(ValueError, match=r"field 'f' of shape \(2, 3\).*expects \(2,\)"):
        LiveMarketEnv(
            ListFeed(rows, fields=fields), PaperBroker(), lookback=2, features=("f",)
        ).reset()
    for env in (
        LiveMarketEnv(
            ListFeed(rows, fields=fields),
            PaperBroker(),
            lookback=2,
            features=("f",),
            field_shapes={"f": (3,)},
        ),
        LiveMarketEnv(
            ListFeed(rows, fields=fields, field_shapes={"f": (3,)}),
            PaperBroker(),
            lookback=2,
            features=("f",),
        ),
    ):
        assert env.observation_space.contains(env.reset().observation)
    changing = ListFeed(rows, fields=[{"f": np.zeros((2, 3))}] * 3 + [{"f": np.zeros((2, 2))}] * 3)
    env = LiveMarketEnv(
        changing, PaperBroker(), lookback=2, features=("f",), field_shapes={"f": (3,)}
    )
    env.reset()
    with pytest.raises(ValueError, match=r"The bar at 3 has a field 'f' of shape \(2, 2\)"):
        env.step(env.action_space.sample())
    with pytest.raises(ValueError, match="no field 'f'"):
        LiveMarketEnv(ListFeed(rows), PaperBroker(), lookback=2, features=("f",)).reset()
    series = ListFeed(rows)
    series.next_bar = lambda: Bar(0, np.ones(2), exogenous={"m": np.ones(3)})  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="market series 'm' of shape"):
        LiveMarketEnv(series, PaperBroker(), lookback=2, exogenous=("m",)).reset()
    infinite = ListFeed(rows, fields=[{"f": np.array([1.0, np.inf])}] * 6)
    with pytest.raises(ValueError, match="infinite values in the field 'f'"):
        LiveMarketEnv(infinite, PaperBroker(), lookback=2, features=("f",)).reset()
    with pytest.raises(TypeError):
        LiveMarketEnv(ListFeed(rows), PaperBroker(), field_shapes={"f": 3})  # type: ignore[dict-item]


@pytest.mark.parametrize("bad", [0.0, -1.0, np.inf, -np.inf])
def test_bad_prices_are_refused(bad: float) -> None:
    rows = [[100.0, 100.0]] * 3 + [[bad, 100.0]] + [[100.0, 100.0]] * 3
    broker = GuardedBroker(PaperBroker(transaction_cost=NoCost()), RiskLimits(max_drawdown=0.2))
    env = LiveMarketEnv(
        ListFeed(rows), broker, lookback=2, action=TargetWeights(1.0), execution="close"
    )
    env.reset()
    message = rf"The bar at 3 gives asset 'a' a price of {bad!r}; a price must be NaN"
    with pytest.raises(ValueError, match=message):
        env.step(np.array([1.0, 0.0], np.float32))
    with pytest.raises(sekai.ResetRequired):  # an order is out and no mark has come
        env.step(np.array([1.0, 0.0], np.float32))
    assert not broker.killed  # the bad price never reached the account
    early = ListFeed([[1.0, 1.0], [1.0, bad], *rows])
    with pytest.raises(ValueError, match="The bar at 1 gives asset 'b'"):
        LiveMarketEnv(early, PaperBroker(), lookback=2).reset()
    paper = PaperBroker()
    paper.reset(("a", "b"))
    with pytest.raises(ValueError, match="The bar at 0 gives asset 'a'"):
        paper.execute(np.array([0.5, 0.5]), Bar(0, np.array([bad, 1.0])), {})
    paper.execute(np.array([0.5, 0.5]), Bar(0, np.array([1.0, 1.0])), {})
    with pytest.raises(ValueError, match="The bar at 1 gives asset 'b'"):
        paper.mark(Bar(1, np.array([1.0, bad])))
    with pytest.raises(ValueError, match="shape"):
        paper.mark(Bar(1, np.array([1.0, 1.0, 1.0])))


def test_bad_corporate_actions_are_refused() -> None:
    prices = np.array([1.0, 1.0])
    split = Bar(5, prices, fields={"split": np.array([np.nan, 0.0])}, corporate_actions=True)
    with pytest.raises(ValueError, match=r"asset 'b' a split of 0\.0"):
        split.validate(("a", "b"))
    dividend = Bar(5, prices, fields={"dividend": np.array([-0.1, 0.0])}, corporate_actions=True)
    with pytest.raises(ValueError, match=r"asset 'a' a dividend of -0\.1"):
        dividend.validate(("a", "b"))
    ordinary = Bar(5, np.array([1.0, np.nan]), fields={"split": np.array([0.0, 0.0])})
    ordinary.validate(("a", "b"))  # without corporate_actions, "split" is an ordinary field


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


def test_paper_broker_does_not_trade_assets_without_a_price() -> None:
    broker = PaperBroker(transaction_cost=0.01)
    broker.reset(("a", "b"))
    report = broker.execute(np.array([0.5, 0.5]), Bar(0, np.array([np.nan, 10.0])), {})
    np.testing.assert_array_equal(report.weights, [0.0, 0.5])  # no price: no trade
    assert report.turnover == 0.5 and report.cost == pytest.approx(0.005)  # nor cost
    state = broker.mark(Bar(1, np.array([20.0, 11.0])))
    assert state.weights[0] == 0.0
    report = broker.execute(np.array([np.nan, 0.5]), Bar(1, np.array([np.nan, 11.0])), {})
    assert report.weights[0] == 0.0  # a NaN target where there is no price is no order either
    with pytest.raises(ValueError, match="target weight of asset 'b' at 2 is nan"):
        broker.execute(np.array([0.0, np.nan]), Bar(2, np.array([1.0, 1.0])), {})
    with pytest.raises(ValueError, match="shape"):
        broker.execute(np.array([0.5]), Bar(2, np.array([1.0, 1.0])), {})


def test_orders_for() -> None:
    orders = orders_for(
        np.array([0.5, -0.25, 0.1]),
        100_000,
        np.array([100.0, 50.0, np.nan]),
        np.array([100.0, 0.0, 5.0]),
        lot_size=10,
    )
    np.testing.assert_array_equal(orders, [400.0, -500.0, 0.0])


def test_orders_for_hits_exact_targets() -> None:
    np.testing.assert_array_equal(orders_for([0.29], 1e5, [10.0], [0.0]), [2900.0])
    rng = np.random.default_rng(0)
    n = 5000
    lots = rng.choice([1.0, 10.0, 100.0, 0.5], n)
    target = rng.integers(-300, 300, n) * lots  # whole lots
    held = rng.integers(-300, 300, n) * lots
    prices = rng.integers(100, 100_000, n) / 100.0
    value = rng.choice([1e4, 1e5, 2.5e5, 1e6, 3.7e7], n)
    weights = target * prices / value
    for i in range(n):
        orders = orders_for(
            weights[i : i + 1], value[i], prices[i : i + 1], held[i : i + 1], lot_size=lots[i]
        )
        assert orders[0] == target[i] - held[i], (weights[i], value[i], prices[i], lots[i])
    zero = orders_for([0.0], 1e5, [10.0], [0.3])  # 0.3 shares: no whole share to sell
    assert zero[0] == 0.0 and not np.signbit(zero[0])


def test_orders_for_refuses_what_would_be_nan() -> None:
    w, p, held = np.array([0.5, 0.0]), np.array([10.0, 20.0]), np.zeros(2)
    for lot in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="lot_size"):
            orders_for(w, 1e5, p, held, lot_size=lot)
    for value in (np.nan, np.inf):
        with pytest.raises(ValueError, match="account_value"):
            orders_for(w, value, p, held)
    with pytest.raises(ValueError, match="weight of asset 1"):
        orders_for(np.array([0.5, np.nan]), 1e5, p, held)
    with pytest.raises(ValueError, match="position of asset 0"):
        orders_for(w, 1e5, p, np.array([np.inf, 0.0]))
    for price in (0.0, -5.0, np.inf):
        with pytest.raises(ValueError, match="price of asset 0"):
            orders_for(w, 1e5, np.array([price, 20.0]), held)  # asset 0 needs an order
        with pytest.raises(ValueError, match="price of asset 1"):
            orders_for(w, 1e5, np.array([10.0, price]), np.array([0.0, 3.0]))  # it holds shares
        out = orders_for(w, 1e5, np.array([10.0, price]), held)  # nothing to trade in asset 1
        np.testing.assert_array_equal(out, [5000.0, 0.0])
    np.testing.assert_array_equal(orders_for(w, 1e5, np.array([np.nan, 20.0]), held), [0.0, 0.0])
    with pytest.raises(ValueError, match="shape"):
        orders_for(w, 1e5, p[:1], held)


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


def test_risk_limits_with_non_finite_targets() -> None:
    limits = RiskLimits(max_position=0.1, max_gross=1.0, max_turnover=0.5)
    w, hit = limits.apply(np.array([np.nan, 0.5]), np.zeros(2))
    np.testing.assert_allclose(w, [0.0, 0.1])  # asset 0 is not traded
    assert hit == ["non_finite_target", "max_position"]
    w, hit = RiskLimits(max_gross=1.0).apply(np.array([np.inf, 0.5]), np.array([0.2, 0.0]))
    np.testing.assert_allclose(w, [0.2, 0.5])
    assert hit == ["non_finite_target"]
    w, hit = RiskLimits(max_gross=0.5).apply(np.array([-np.inf, 0.5]), np.array([0.4, 0.0]))
    assert np.abs(w).sum() == pytest.approx(0.5)  # the kept weight still counts towards the caps
    assert hit == ["non_finite_target", "max_gross"]
    with pytest.raises(ValueError, match="current"):
        RiskLimits().apply(np.zeros(2), np.array([np.nan, 0.0]))
    with pytest.raises(ValueError, match="shape"):
        RiskLimits().apply(np.zeros(2), np.zeros(3))


def test_limits_bind_on_the_assets_that_can_be_traded() -> None:
    limits = RiskLimits(max_gross=1.0, max_net=0.5)
    tradable = np.array([True, True, False])  # asset 2 cannot be traded now
    w, hit = limits.apply(np.array([0.5, 0.5, 0.5]), np.array([0.0, 0.0, 0.3]), tradable)
    np.testing.assert_allclose(w, [0.1, 0.1, 0.3])  # 0.3 stays, so the others make up 0.2
    assert hit == ["max_net"]
    w, hit = limits.apply(np.array([np.nan, 0.2, np.nan]), np.zeros(3), tradable)
    np.testing.assert_array_equal(w, [0.0, 0.2, 0.0])
    assert hit == ["non_finite_target"]  # asset 2 could not be traded anyway
    w, _ = limits.apply(np.array([0.9, 0.0, 0.0]), np.array([0.0, 0.0, 0.8]), tradable)
    np.testing.assert_array_equal(w, [0.0, 0.0, 0.8])  # the untradable position alone breaks it
    with pytest.raises(ValueError, match="tradable"):
        limits.apply(np.zeros(3), np.zeros(3), np.array([True, False]))
    # GuardedBroker: an asset without a price at the bar is not traded (by the broker
    # either), so the limits hold for the book as executed.
    broker = GuardedBroker(PaperBroker(transaction_cost=NoCost()), RiskLimits(max_net=0.4))
    broker.reset(("a", "b", "c"))
    report = broker.execute(np.array([-0.3, -0.3, 0.5]), Bar(0, np.array([10.0, 10.0, np.nan])), {})
    np.testing.assert_allclose(report.weights, [-0.2, -0.2, 0.0])
    assert abs(report.weights.sum()) <= 0.4 + 1e-12


def test_turnover_never_leaves_the_book_over_a_cap() -> None:
    w, hit = RiskLimits(max_position=0.5, max_turnover=0.05).apply(np.array([0.6]), np.array([0.6]))
    np.testing.assert_allclose(w, [0.5])
    assert hit == ["max_position"]
    limits = RiskLimits(max_gross=1.0, max_turnover=0.1)
    w, hit = limits.apply(np.array([0.7, 0.7]), np.array([0.7, 0.7]))
    np.testing.assert_allclose(w, [0.5, 0.5])
    w, hit = limits.apply(np.array([1.0, 0.0]), np.array([0.7, 0.7]))
    np.testing.assert_allclose(w, [0.55, 0.45])  # back to gross 1, then 0.1 towards the target
    assert hit == ["max_gross", "max_turnover"]
    rng = np.random.default_rng(0)
    for _ in range(3000):
        caps = rng.uniform(0.05, 1.0, 4) * (rng.random(4) < 0.6)  # 0: no such cap
        limits = RiskLimits(
            max_position=caps[0] or None,
            max_gross=2 * caps[1] or None,
            max_net=caps[2] or None,
            max_turnover=caps[3] * 0.5 if caps[3] else None,
            allow_short=bool(rng.integers(2)),
        )
        current = rng.normal(0.0, 0.5, 5)
        target = np.where(rng.random(5) < 0.1, np.nan, rng.normal(0.0, 0.5, 5))
        w, hit = limits.apply(target, current)
        assert np.all(np.isfinite(w))
        if limits.max_position is not None:
            assert np.abs(w).max() <= limits.max_position + 1e-12
        if limits.max_gross is not None:
            assert np.abs(w).sum() <= limits.max_gross + 1e-12
        if limits.max_net is not None:
            assert abs(w.sum()) <= limits.max_net + 1e-12
        if not limits.allow_short:
            assert np.all(w >= 0)
        if limits.max_turnover is not None:
            within, _ = limits.apply(current, current)  # the current book within the caps
            budget = limits.max_turnover + np.abs(within - current).sum()
            assert np.abs(w - current).sum() <= budget + 1e-12  # risk reduction is beyond it


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
    assert broker.drawdown == pytest.approx(0.2)
    out = env.step(np.array([1.0], np.float32))
    assert out.info["gross_exposure"] == 0.0 and out.info["equity"] == pytest.approx(0.8)
    assert broker.events[-1].detail == "account kept flat"
    broker.reset(("asset_0",))
    assert broker.killed  # latched: a reset does not clear it
    broker.rearm()
    assert not broker.killed and broker.events[-1].kind == "rearm"
    assert broker.drawdown == 0.0  # measured from the account's value at the rearm
    assert broker.capital == 1_000_000.0  # the inner broker's attributes show through
    with pytest.raises(AttributeError):
        broker.killed = False  # type: ignore[misc]


def test_kill_switch_stays_tripped_across_episodes_until_rearm() -> None:
    p = 100.0 * 0.99 ** np.arange(60)  # a steady decline
    data = MarketData(np.stack([p, p], axis=1))
    broker = GuardedBroker(PaperBroker(transaction_cost=NoCost()), RiskLimits(max_drawdown=0.03))
    env = LiveMarketEnv(
        ReplayFeed(data),
        broker,
        lookback=2,
        episode_length=8,
        action=TargetWeights(1.0),
        execution="close",
    )
    gross, killed = [], []
    for _ in range(3):
        r = env.reset()
        while not r.done:
            r = env.step(np.array([0.5, 0.5], np.float32))
            gross.append(r.info["gross_exposure"])
            killed.append(broker.killed)
    trips = [e for e in broker.events if e.kind == "kill_switch" and "drawdown" in e.detail]
    assert len(trips) == 1  # tripped once, in the first episode, and never cleared
    tripped = killed.index(True)  # at the mark of step 4: 0.99**4 is a 3.9% drawdown
    assert tripped == 3 and all(killed[tripped:])
    assert all(g > 0 for g in gross[: tripped + 1]) and not any(gross[tripped + 1 :])
    broker.rearm()
    r = env.reset()
    r = env.step(np.array([0.5, 0.5], np.float32))
    assert r.info["gross_exposure"] > 0  # trading again


class PersistentAccount(PaperBroker):
    """A paper account that persists across resets, as a live account does.

    Like the adapter of the docs, it reports equity relative to the latest reset.
    """

    fresh_account = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.base = 1.0
        self.calls: list[str] = []

    def reset(self, assets: tuple[Any, ...]) -> AccountState:
        self.calls.append("reset")
        if not self._assets:
            super().reset(assets)
        self.base = self._equity
        return self._rebased(self._state(0.0))

    def execute(self, weights: Any, bar: Bar, context: Mapping[str, Any]) -> ExecutionReport:
        self.calls.append(f"execute {bar.timestamp}")
        return super().execute(weights, bar, context)

    def mark(self, bar: Bar) -> AccountState:
        self.calls.append(f"mark {bar.timestamp}")
        return self._rebased(super().mark(bar))

    def _rebased(self, state: AccountState) -> AccountState:
        return AccountState(state.equity / self.base, state.weights, state.value, state.financing)


@pytest.mark.parametrize("fresh", [False, True])
def test_drawdown_runs_across_resets_of_a_persistent_account(fresh: bool) -> None:
    p = 100.0 * 0.99 ** np.arange(40)  # -1% a bar: about -3% an episode
    data = MarketData(p[:, None])
    account = PersistentAccount(transaction_cost=NoCost())
    account.fresh_account = fresh  # misreported as fresh, the peak restarts every episode
    broker = GuardedBroker(account, RiskLimits(max_drawdown=0.05))
    env = LiveMarketEnv(
        ReplayFeed(data),
        broker,
        lookback=2,
        episode_length=3,
        action=TargetWeights(1.0),
        execution="close",
    )
    episode_drawdowns = []
    for _ in range(3):
        r = env.reset()
        while not r.done:
            r = env.step(np.array([1.0], np.float32))
        episode_drawdowns.append(r.info["drawdown"])
    assert max(episode_drawdowns) < 0.05  # no single episode reaches the limit
    assert broker.killed is not fresh  # the account as a whole does
    # Between episodes the account is attached again at the same bar: no bar goes unmarked.
    steps = [c for c in account.calls if c != "reset"]
    assert all(a.split()[0] != b.split()[0] for a, b in itertools.pairwise(steps))


def test_a_fresh_paper_account_is_not_killed_by_an_earlier_peak() -> None:
    up = 100.0 * 1.05 ** np.arange(10)
    p = np.concatenate([up, np.full(10, up[-1])])  # +5% a bar, then flat
    data = MarketData(p[:, None])
    broker = GuardedBroker(PaperBroker(transaction_cost=NoCost()), RiskLimits(max_drawdown=0.1))
    env = LiveMarketEnv(
        ReplayFeed(data),
        broker,
        lookback=2,
        episode_length=6,
        action=TargetWeights(1.0),
        execution="close",
    )
    r = env.reset()
    while not r.done:
        r = env.step(np.array([1.0], np.float32))
    assert r.info["equity"] > 1.3 and broker.peak == pytest.approx(r.info["equity"] * 1e6)
    r = env.reset()  # a new account of 1.0, far below the last one's peak
    assert broker.peak == 1e6 and broker.drawdown == 0.0
    r = env.step(np.array([1.0], np.float32))
    assert not broker.killed and r.info["gross_exposure"] > 0


def test_reset_marks_and_checks_positions_while_it_reads_bars() -> None:
    account = PersistentAccount(transaction_cost=NoCost())
    broker = GuardedBroker(account, RiskLimits(max_drawdown=0.2))
    first = LiveMarketEnv(
        ListFeed([[100.0]] * 4, ("a",)),
        broker,
        lookback=2,
        action=TargetWeights(1.0),
        execution="close",
    )
    first.reset()
    first.step(np.array([1.0], np.float32))  # the account is now fully long
    account.calls.clear()
    # A new process attaches to the account; its feed has no history, so reset() waits for
    # four bars to trade on, while the account holds its position through a 40% fall.
    later = LiveMarketEnv(
        ListFeed([[100.0], [100.0], [60.0], [60.0], [60.0]], ("a",)),
        broker,
        lookback=3,
        action=TargetWeights(1.0),
        execution="close",
    )
    r = later.reset()
    assert account.calls == [
        "reset",
        "execute 0",
        "mark 1",
        "execute 1",
        "mark 2",
        "execute 2",
        "mark 3",
    ]
    assert broker.killed and r.info["equity"] == pytest.approx(0.6)
    assert r.info["gross_exposure"] == 0.0  # sold at bar 2, as soon as the fall was seen
    assert [e.kind for e in broker.events][-2:] == ["kill_switch", "kill_switch"]


def test_guarded_broker_keeps_an_asset_with_a_non_finite_target() -> None:
    p = 100.0 * 0.7 ** np.arange(6)  # -30% a bar
    broker = GuardedBroker(
        PaperBroker(transaction_cost=NoCost()), RiskLimits(max_gross=1.0, max_drawdown=0.1)
    )
    broker.reset(("a", "b"))
    report = broker.execute(np.array([np.nan, 0.5]), Bar(0, np.array([p[0], p[0]])), {})
    np.testing.assert_array_equal(report.weights, [0.0, 0.5])
    assert broker.events[0].kind == "non_finite_target" and "'a'" in broker.events[0].detail
    state = broker.mark(Bar(1, np.array([p[1], p[1]])))
    assert state.equity == pytest.approx(0.85) and broker.killed  # the kill switch still works
    for t in range(1, 5):
        broker.execute(np.array([np.nan, 0.5]), Bar(t, np.array([p[t], p[t]])), {})
        state = broker.mark(Bar(t + 1, np.array([p[t + 1], p[t + 1]])))
    assert state.equity == pytest.approx(0.85)  # flat since the trip


class FlakyBroker(PaperBroker):
    """A paper broker whose account reports go bad (non-finite) from `bad_from` on."""

    fresh_account = False

    def __init__(self, bad_from: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.bad_from = bad_from
        self.sent: list[np.ndarray] = []

    def execute(self, weights: Any, bar: Bar, context: Mapping[str, Any]) -> ExecutionReport:
        self.sent.append(np.asarray(weights, dtype=np.float64))
        return super().execute(weights, bar, context)

    def mark(self, bar: Bar) -> AccountState:
        state = super().mark(bar)
        if bar.timestamp >= self.bad_from:
            return AccountState(np.nan, state.weights, np.nan)
        return state


def test_guarded_broker_fails_closed_on_a_non_finite_account() -> None:
    inner = FlakyBroker(bad_from=3, transaction_cost=NoCost())
    broker = GuardedBroker(inner, RiskLimits())  # no drawdown limit: it fails closed anyway
    env = LiveMarketEnv(
        ListFeed([[100.0, 100.0]] * 8),
        broker,
        lookback=1,
        action=TargetWeights(1.0),
        execution="close",
    )
    env.reset()
    for _ in range(4):
        env.step(np.array([0.5, 0.5], np.float32))
    assert broker.killed
    trip = next(e for e in broker.events if e.kind == "kill_switch")
    assert trip.timestamp == 3 and "non-finite" in trip.detail
    assert np.any(inner.sent[1] != 0) and not np.any(inner.sent[-1])  # liquidated, then flat
    assert broker.drawdown == 0.0  # measured on the last finite report


class Spy(RewardFunction):
    """Records the weights of every outcome it rewards."""

    def __init__(self) -> None:
        self.seen: list[np.ndarray] = []

    def __call__(self, outcome: StepOutcome) -> Any:
        self.seen.append(np.asarray(outcome.weights))
        return 0.0


@pytest.mark.parametrize("execution", ["next_close", "close"])
def test_rewards_see_the_weights_executed(execution: str) -> None:
    broker = GuardedBroker(PaperBroker(transaction_cost=NoCost()), RiskLimits(max_gross=0.5))
    env = LiveMarketEnv(
        ReplayFeed(DATA),
        broker,
        lookback=5,
        action=TargetWeights(1.0),
        execution=execution,
        reward=Spy(),
    )
    env.reset()
    for _ in range(4):
        env.step(np.where(env.action_mask() > 0, 1.0, 0.0).astype(np.float32))
    seen = env.reward_fn.seen  # type: ignore[attr-defined]
    assert len(seen) == 4 and all(np.abs(w).sum() <= 0.5 + 1e-12 for w in seen)
    assert np.abs(seen[-1]).sum() == pytest.approx(0.5)  # what was held, not the 1.0 asked for


def test_guarded_broker_clips_targets() -> None:
    broker = GuardedBroker(PaperBroker(transaction_cost=NoCost()), RiskLimits(max_gross=0.5))
    env = LiveMarketEnv(
        ReplayFeed(DATA), broker, lookback=5, action=TargetWeights(1.0), execution="close"
    )
    env.reset()
    out = env.step(np.where(env.action_mask() > 0, 1.0, 0.0).astype(np.float32))
    assert out.info["gross_exposure"] <= 0.5 * 1.1
    assert any(e.kind == "max_gross" for e in broker.events)
