"""MarketData, samplers, cost models, accounting, rewards and SlotMap."""

from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pytest

from sekai.envs.finance import (
    MarketEnv,
    PortfolioEnv,
    PortfolioVecEnv,
    TradingEnv,
    TradingVecEnv,
)
from sekai.finance import (
    AccountConfig,
    DifferentialSharpe,
    DrawdownPenalty,
    FixedCost,
    LogReturn,
    MarketData,
    MeanVariance,
    NoCost,
    ProfitAndLoss,
    ProportionalCost,
    SequentialStart,
    SimpleReturn,
    SlotMap,
    SpreadCost,
    SquareRootImpact,
    StepOutcome,
    Time,
    TurnoverPenalty,
)
from sekai.finance.account import project_weights, settle, turnover
from sekai.finance.costs import as_cost_model
from sekai.finance.sampling import (
    FixedStart,
    RandomStart,
    as_sampler,
    first_start,
    resolve_start,
    sample_start,
)


def make_data(n_steps: int = 300, n_assets: int = 3, *, dates: bool = True) -> MarketData:
    rng = np.random.default_rng(0)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_steps, n_assets)), axis=0))
    prices[:50, -1] = np.nan  # the last asset lists at row 50
    times = np.arange("2021-01-01", n_steps, dtype="datetime64[D]") if dates else None
    if dates:
        times = np.datetime64("2021-01-01") + np.arange(n_steps)
    return MarketData(
        prices,
        timestamps=times,
        assets=[f"A{i}" for i in range(n_assets)],
        fields={"volume": np.full((n_steps, n_assets), 1e6)},
        exogenous={"rate": np.full(n_steps, 0.01), "factors": np.zeros((n_steps, 2))},
    )


class TestMarketData:
    def test_shapes_mask_and_immutability(self) -> None:
        data = make_data()
        assert (data.n_steps, data.n_assets) == (300, 3) and len(data) == 300
        assert data.mask[:50, 2].sum() == 0 and data.mask[50:, 2].all()
        assert data.exogenous_width("factors") == 2 and data.exogenous_width("rate") == 1
        with pytest.raises(ValueError):
            data.prices[0, 0] = 1.0
        assert data.periods_per_year == pytest.approx(365.25, rel=0.01)
        assert "T=300" in repr(data) and "2021-01-01" in repr(data)

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            MarketData(np.array([[1.0], [-1.0]]))
        with pytest.raises(ValueError, match="infinite"):
            MarketData(np.array([[1.0], [np.inf]]))
        with pytest.raises(ValueError, match="increasing"):
            MarketData(np.ones((3, 1)), timestamps=[0, 2, 1])
        with pytest.raises(ValueError, match="unique"):
            MarketData(np.ones((3, 2)), assets=["a", "a"])
        with pytest.raises(ValueError, match="shape"):
            MarketData(np.ones((3, 2)), fields={"volume": np.ones((3, 3))})
        with pytest.raises(ValueError):
            MarketData()

    def test_log_returns_respect_listings(self) -> None:
        data = make_data()
        r = data.log_returns()
        assert np.isnan(r[0]).all() and np.isnan(r[50, 2]) and np.isfinite(r[51, 2])

    def test_slicing_and_selection(self) -> None:
        data = make_data()
        part = data[100:200]
        assert len(part) == 100 and part.timestamps[0] == data.timestamps[100]
        june = data.between("2021-06-01", "2021-07-01")
        assert len(june) == 30
        sub = data.select(["A2", "A0"])
        np.testing.assert_array_equal(sub.prices[:, 1], data.prices[:, 0])
        with pytest.raises(KeyError):
            data.select(["ZZZ"])
        assert data.index_of(5) == 5 and data.index_of("2021-01-03") == 2

    def test_split_with_embargo_and_history(self) -> None:
        data = make_data(dates=False)
        train, test = data.split(0.8, embargo=5, history=20)
        assert len(train) == 240 and test.timestamps[0] == 240 + 5 - 20
        assert test.timestamps[-1] == 299
        a, b, c = data.split(100, 200)
        assert (len(a), len(b), len(c)) == (100, 100, 100)
        with pytest.raises(ValueError):
            data.split(0.5, 0.4)

    def test_walk_forward(self) -> None:
        folds = list(make_data(dates=False).walk_forward(100, 50, embargo=3, history=10))
        assert len(folds) == 3  # test periods 103-152, 153-202, 203-252; the next would overrun
        (tr0, te0), (tr1, _), _ = folds
        assert (tr0.timestamps[0], tr0.timestamps[-1]) == (0, 99)
        assert te0.timestamps[0] == 100 + 3 - 10 and te0.timestamps[-1] == 152
        assert tr1.timestamps[0] == 50
        expanding = list(make_data(dates=False).walk_forward(100, 50, expanding=True))
        assert all(train.timestamps[0] == 0 for train, _ in expanding)

    def test_with_fields_and_exogenous(self) -> None:
        data = make_data()
        richer = data.with_fields(momentum=np.zeros((300, 3))).with_exogenous(vix=np.ones(300))
        assert set(richer.fields) == {"close", "volume", "momentum"}
        assert "vix" in richer.exogenous and "momentum" not in data.fields

    def test_pandas_round_trip(self) -> None:
        pd = pytest.importorskip("pandas")
        dates = pd.bdate_range("2022-01-03", periods=60)
        wide = pd.DataFrame(
            np.linspace(10, 20, 120).reshape(60, 2), index=dates, columns=["X", "Y"]
        )
        volume = wide.iloc[:40] * 0 + 7.0  # shorter: aligned with NaN
        data = MarketData.from_pandas(
            wide, fields={"volume": volume}, exogenous=pd.DataFrame({"vix": np.arange(60.0)}, dates)
        )
        assert data.assets == ("X", "Y") and data.is_datetime
        assert np.isnan(data.field("volume")[50]).all() and data.exogenous["vix"][3] == 3.0
        long = wide.stack().rename("close").reset_index()
        long.columns = ["date", "symbol", "close"]
        again = MarketData.from_long(long)
        np.testing.assert_array_equal(again.prices, data.prices)


class TestSamplers:
    def test_sequential_tiles_and_wraps(self) -> None:
        s = SequentialStart()
        rng = np.random.default_rng(0)
        starts = [s.sample(rng, 25, 10) for _ in range(4)]
        assert starts == [0, 10, 20, 0] and s.epoch == 1
        s.reset()
        assert s.sample(rng, 25, 10) == 0 and s.epoch == 0
        assert SequentialStart(start=3, stride=2).sample(rng, 25, 10) == 3

    def test_fixed_random_and_parsing(self) -> None:
        rng = np.random.default_rng(0)
        assert FixedStart(4).sample(rng, 10, 5) == 4
        assert all(0 <= RandomStart().sample(rng, 10, 5) < 10 for _ in range(50))
        assert isinstance(as_sampler("random"), RandomStart)
        assert isinstance(as_sampler(3), FixedStart)
        with pytest.raises(ValueError):
            as_sampler("nope")
        with pytest.raises(ValueError):
            FixedStart(20).sample(rng, 10, 5)

    def test_resolve_start(self) -> None:
        times = np.datetime64("2021-01-01") + np.arange(100)
        assert resolve_start("2021-01-21", timestamps=times, lookback=10, n_starts=50) == 10
        assert resolve_start(15, timestamps=times, lookback=10, n_starts=50) == 5
        with pytest.raises(ValueError, match="valid range"):
            resolve_start(5, timestamps=times, lookback=10, n_starts=50)


class TestCosts:
    trades = np.array([0.5, -0.25, 0.0])

    def test_proportional_and_spread(self) -> None:
        assert ProportionalCost(0.001)(self.trades) == pytest.approx(0.00075)
        per_asset = ProportionalCost([0.001, 0.002, 0.0])(self.trades)
        assert per_asset == pytest.approx(0.0005 + 0.0005)
        assert SpreadCost(1e-3)(self.trades) == pytest.approx(0.00075)
        quoted = SpreadCost()(self.trades, spread=np.array([0.002, 0.004, 0.1]))
        assert quoted == pytest.approx(0.5 * 0.001 + 0.25 * 0.002)
        with pytest.raises(ValueError, match="spread"):
            SpreadCost()(self.trades)

    def test_fixed_and_impact(self) -> None:
        assert FixedCost(10.0)(self.trades, equity=1e4) == pytest.approx(2e-3)
        impact = SquareRootImpact(1.0)(
            np.array([0.1]), equity=1e6, volatility=np.array([0.02]), volume=np.array([1e7])
        )
        assert impact == pytest.approx(0.1 * 0.02 * np.sqrt(0.01))
        bigger = SquareRootImpact(1.0)(
            np.array([0.4]), equity=1e6, volatility=np.array([0.02]), volume=np.array([1e7])
        )
        assert bigger == pytest.approx(impact * 8)  # cost grows as size^1.5
        with pytest.raises(ValueError, match="volume"):
            SquareRootImpact()(self.trades, equity=1.0, volatility=np.ones(3))

    def test_composition_and_batching(self) -> None:
        model = ProportionalCost(1e-3) + SpreadCost(1e-3) + NoCost()
        assert len(model.models) == 3
        batch = np.stack([self.trades, 2 * self.trades])
        np.testing.assert_allclose(model(batch), [0.0015, 0.003])
        assert as_cost_model(0.002)(self.trades) == pytest.approx(0.0015)
        with pytest.raises(ValueError):
            as_cost_model(-1.0)
        assert "ProportionalCost" in repr(model)


class TestAccount:
    def test_project_long_only(self) -> None:
        cfg = AccountConfig()
        w = project_weights([1.0, 1.0, 2.0], [True, True, False], cfg, cash_score=2.0)
        np.testing.assert_allclose(w, [0.25, 0.25, 0.0])
        np.testing.assert_allclose(project_weights([0.0, 0.0], [True, True], cfg), [0.0, 0.0])
        lev = project_weights([1.0, 1.0], [True, True], AccountConfig(max_leverage=2.0))
        np.testing.assert_allclose(lev, [1.0, 1.0])

    def test_project_long_short(self) -> None:
        cfg = AccountConfig(allow_short=True, max_leverage=1.0)
        np.testing.assert_allclose(project_weights([0.8, -0.8], [True, True], cfg), [0.5, -0.5])
        np.testing.assert_allclose(project_weights([0.2, -0.3], [True, True], cfg), [0.2, -0.3])
        batch = project_weights(np.ones((4, 2)), np.ones((4, 2), bool), cfg)
        assert batch.shape == (4, 2)

    def test_settle_financing_and_drift(self) -> None:
        cfg = AccountConfig(cash_rate=0.05, borrow_rate=0.02, allow_short=True)
        s = settle([0.5, -0.2], [0.1, 0.05], cost=0.001, dt=1.0, config=cfg)
        financing = 0.7 * 0.05 - 0.02 * 0.2
        assert s.financing == pytest.approx(financing)
        gross = 1 + 0.05 - 0.01 + financing
        assert s.growth == pytest.approx(0.999 * gross)
        np.testing.assert_allclose(s.drifted, [0.55 / gross, -0.21 / gross])
        margin = settle([1.5], [0.0], cost=0.0, dt=1.0, config=AccountConfig(margin_rate=0.1))
        assert margin.financing == pytest.approx(-0.05)

    def test_wipe_out(self) -> None:
        s = settle([2.0], [-0.6], cost=0.0, dt=0.0, config=AccountConfig(max_leverage=2.0))
        assert s.wiped_out and np.all(s.drifted == 0)
        assert turnover([0.5, 0.5], [0.0, 1.0]) == pytest.approx(1.0)


class TestRewards:
    def outcome(self, growth: float, equity: float) -> StepOutcome:
        return StepOutcome.from_growth(growth, equity=equity, turnover=0.5, cost=0.001)

    def test_basic(self) -> None:
        o = self.outcome(1.1, 1.1)
        assert LogReturn()(o) == pytest.approx(np.log(1.1))
        assert SimpleReturn()(o) == pytest.approx(0.1)
        assert ProfitAndLoss()(o) == pytest.approx(0.1)
        assert MeanVariance(2.0)(o) == pytest.approx(np.log(1.1) - np.log(1.1) ** 2)
        assert TurnoverPenalty()(o) == 0.5
        assert LogReturn()(self.outcome(0.0, 0.0)) == pytest.approx(np.log(1e-12))

    def test_composition(self) -> None:
        reward = LogReturn() - 0.5 * TurnoverPenalty() + 0.01
        assert reward(self.outcome(1.0, 1.0)) == pytest.approx(-0.25 + 0.01)
        assert (-LogReturn())(self.outcome(np.e, 1.0)) == pytest.approx(-1.0)
        assert "LogReturn()" in repr(reward)

    def test_drawdown_penalty(self) -> None:
        dd = DrawdownPenalty()
        values = [dd(self.outcome(1.0, e)) for e in (1.2, 0.9, 1.0, 0.6, 1.3)]
        np.testing.assert_allclose(values, [0.0, 0.25, 0.0, 0.5 - 1 / 6, 0.0], atol=1e-12)
        dd.reset()
        assert dd(self.outcome(1.0, 0.5)) == pytest.approx(0.5)

    def test_differential_sharpe_is_the_sharpe_derivative(self) -> None:
        # D_t is dS_t/d(eta) at eta = 0, where S = A / sqrt(B - A^2) after updating the
        # moving moments A, B with the new return (Moody & Saffell, 2001).
        rng = np.random.default_rng(0)
        ds = DifferentialSharpe(eta=0.05)
        for _ in range(200):
            ds(self.outcome(1 + 0.001 + 0.01 * rng.standard_normal(), 1.0))
        assert ds._state is not None
        a, b = float(ds._state["A"]), float(ds._state["B"])
        r, h = 0.004, 1e-7

        def sharpe(eta: float) -> float:
            a2, b2 = a + eta * (r - a), b + eta * (r * r - b)
            return a2 / np.sqrt(b2 - a2 * a2)

        expected = (sharpe(h) - sharpe(0.0)) / h
        assert float(ds(self.outcome(1 + r, 1.0))) == pytest.approx(expected, rel=1e-4)
        assert float(DifferentialSharpe()(self.outcome(1.01, 1.0))) == 0.0  # no variance yet

    def test_batched_state_and_partial_reset(self) -> None:
        dd = DrawdownPenalty()
        batch = StepOutcome.from_growth(
            np.ones(3), equity=np.array([1.0, 0.5, 0.8]), turnover=np.zeros(3), cost=np.zeros(3)
        )
        np.testing.assert_allclose(dd(batch), [0.0, 0.5, 0.2])
        dd.reset(np.array([False, True, False]))
        again = StepOutcome.from_growth(
            np.ones(3), equity=np.array([1.0, 0.4, 0.8]), turnover=np.zeros(3), cost=np.zeros(3)
        )
        np.testing.assert_allclose(dd(again), [0.0, 0.6, 0.0])


class TestSlotMap:
    def test_stable_assignment(self) -> None:
        slots = SlotMap(3)
        assert slots.update(["a", "b"]).tolist() == [0, 1]
        assert slots.update(["b", "c", "d"]).tolist() == [1, 0, 2]  # "a" left, "c" takes slot 0
        assert slots.changed.tolist() == [True, False, True]
        assert slots.assets == ("c", "b", "d") and slots.mask.tolist() == [1, 1, 1]
        assert slots.slot_of("b") == 1 and slots.slot_of("a") is None and len(slots) == 3

    def test_overflow_and_reset(self) -> None:
        slots = SlotMap(2)
        with pytest.raises(ValueError):
            slots.update(["a", "b", "c"])
        with pytest.raises(ValueError):
            slots.update(["a", "a"])
        slots.update(["x"])
        slots.reset()
        assert len(slots) == 0 and slots.update(["y"]).tolist() == [0]


# ---------------------------------------------------------------------------
# Regressions
# ---------------------------------------------------------------------------

# Flat, +10 %, then a 60 % crash: a 2x levered long is wiped out on its second step.
CRASH = np.array([100.0, 100.0, 100.0, 110.0, 44.0, 44.0, 44.0, 44.0])


def _run(env: Any, action: Any, steps: int) -> tuple[float, Any]:
    env.reset(seed=0)
    total, result = 0.0, None
    for _ in range(steps):
        result = env.step(action)
        total += result.reward
        if result.terminated or result.truncated:
            break
    return total, result


class TestProfitAndLossIsExact:
    """The P&L reward on a wipe-out used to be 0 instead of minus the equity lost."""

    def test_outcome_carries_the_change_in_equity(self) -> None:
        wiped = StepOutcome.from_growth(
            -0.2, equity=0.0, turnover=0.0, cost=0.0, previous_equity=1.2
        )
        assert ProfitAndLoss()(wiped) == pytest.approx(-1.2)
        batch = StepOutcome.from_growth(
            np.array([1.1, 0.0]),
            equity=np.array([1.1, 0.0]),
            turnover=np.zeros(2),
            cost=np.zeros(2),
            previous_equity=np.array([1.0, 0.7]),
        )
        np.testing.assert_allclose(ProfitAndLoss()(batch), [0.1, -0.7])
        # Without the previous equity a wipe-out cannot be priced: refuse, never return 0.
        with pytest.raises(ValueError, match="previous_equity"):
            ProfitAndLoss()(StepOutcome.from_growth(0.0, equity=0.0, turnover=0.0, cost=0.0))
        assert StepOutcome.from_growth(1.1, equity=1.1, turnover=0.0, cost=0.0).pnl is None

    def test_market_env_wipe_out(self) -> None:
        env = MarketEnv(
            data=MarketData(CRASH[:, None]),
            lookback=2,
            episode_length=5,
            max_leverage=2.0,
            sampler=0,
            reward=ProfitAndLoss(),
        )
        total, result = _run(env, np.array([1.0, 0.0], dtype=np.float32), 5)
        assert result.terminated and env.equity == 0.0
        assert total == pytest.approx(-1.0, abs=1e-12)

    def test_trading_env_wipe_out(self) -> None:
        env = TradingEnv(
            CRASH,
            lookback=2,
            episode_length=5,
            positions=(0.0, 2.0),
            sampler=0,
            reward=ProfitAndLoss(),
        )
        total, result = _run(env, 1, 5)
        assert result.terminated and env.equity == 0.0
        assert total == pytest.approx(-1.0, abs=1e-12)

    def test_portfolio_env_sums_to_final_equity(self) -> None:
        prices = make_data(60, 2).prices[:, :1] * np.array([1.0, 0.5])
        env = PortfolioEnv(prices, lookback=3, episode_length=20, reward=ProfitAndLoss())
        env.reset(seed=0)
        rng, total = np.random.default_rng(1), 0.0
        for _ in range(20):
            total += env.step(rng.random(3).astype(np.float32)).reward
        assert total == pytest.approx(env.portfolio_value - 1.0, abs=1e-12)

    def test_native_vector_envs(self) -> None:
        vec = TradingVecEnv(
            2,
            autoreset=False,
            prices=CRASH,
            lookback=2,
            episode_length=2,
            positions=(0.0, 2.0),
            sampler=0,
            reward=ProfitAndLoss(),
        )
        vec.reset(seed=0)
        totals = np.zeros(2)
        for _ in range(2):
            step = vec.step(np.array([1, 0]))  # sub-env 0 levered long, sub-env 1 flat
            totals += step.rewards
        assert step.terminated.tolist() == [True, False]
        equity = np.array([info["equity"] for info in step.infos])
        np.testing.assert_allclose(totals, equity - 1.0, atol=1e-12)
        assert totals[0] == pytest.approx(-1.0)

        prices = make_data(60, 2).prices[:, :1] * np.array([1.0, 0.5])
        pvec = PortfolioVecEnv(
            3,
            autoreset=False,
            prices=prices,
            lookback=3,
            episode_length=10,
            reward=ProfitAndLoss(),
        )
        pvec.reset(seed=0)
        rng, totals = np.random.default_rng(2), np.zeros(3)
        for _ in range(10):
            pstep = pvec.step(rng.random((3, 3)).astype(np.float32))
            totals += pstep.rewards
        value = np.array([info["portfolio_value"] for info in pstep.infos])
        np.testing.assert_allclose(totals, value - 1.0, atol=1e-12)


class TestCostsAreNeverSilentlyFree:
    """With no valid input for any asset, impact and quoted-spread costs used to be 0."""

    trades = np.array([0.3, -0.3])

    def test_raise_without_any_valid_input(self) -> None:
        missing = np.full(2, np.nan)
        with pytest.raises(ValueError, match="fallback_volatility"):
            SquareRootImpact(0.5)(
                self.trades, equity=1e6, volatility=missing, volume=np.full(2, 1e7)
            )
        with pytest.raises(ValueError, match="fallback_spread"):
            SpreadCost()(self.trades, spread=missing)
        with pytest.raises(ValueError, match="no valid 'spread'"):  # only the second row fails
            SpreadCost()(
                np.array([[0.1, 0.0], [0.1, 0.0]]),
                spread=np.array([[0.002, np.nan], [np.nan, np.nan]]),
            )
        # A row without inputs but without trades has nothing to price.
        assert SpreadCost()(np.zeros(2), spread=missing) == 0.0
        # One valid input: the others are charged the cross-sectional median, as before.
        assert SpreadCost()(self.trades, spread=np.array([0.002, np.nan])) == pytest.approx(
            0.5 * 0.002 * 0.6
        )

    def test_explicit_fallbacks(self) -> None:
        missing = np.full(2, np.nan)
        spread = SpreadCost(fallback_spread=0.004)
        assert spread(self.trades, spread=missing) == pytest.approx(0.5 * 0.004 * 0.6)
        impact = SquareRootImpact(0.5, fallback_volatility=0.02)
        cost = impact(self.trades, equity=1e6, volatility=missing, volume=np.full(2, 1e7))
        assert cost == pytest.approx(2 * 0.5 * 0.02 * np.sqrt(0.3 * 1e6 / 1e7) * 0.3)
        assert "fallback_spread" in repr(spread) and "fallback_volatility" in repr(impact)
        assert spread.fallbacks == {"spread"} and impact.fallbacks == {"volatility"}
        assert (impact + ProportionalCost(1e-3)).fallbacks == {"volatility"}
        assert (impact + SquareRootImpact(1.0)).fallbacks == frozenset()
        with pytest.raises(ValueError):
            SpreadCost(1e-4, fallback_spread=1e-3)  # a fixed half-spread never needs one
        with pytest.raises(ValueError):
            SquareRootImpact(fallback_volatility=-1.0)

    @pytest.mark.parametrize("kind", ["market", "trading", "portfolio"])
    def test_envs_refuse_an_impact_model_their_lookback_cannot_feed(self, kind: str) -> None:
        rng = np.random.default_rng(0)
        n = 1 if kind == "trading" else 2
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, (40, n)), axis=0))
        data = MarketData(prices, fields={"volume": np.full((40, n), 1e4)})

        def build(cost: Any, lookback: int) -> Any:
            options: dict[str, Any] = {"lookback": lookback, "episode_length": 5, "sampler": 0}
            if kind == "market":
                return MarketEnv(data=data, transaction_cost=cost, **options)
            env_cls = TradingEnv if kind == "trading" else PortfolioEnv
            return env_cls(data=data, transaction_cost=cost, **options)

        with pytest.raises(ValueError, match="lookback=1"):
            build(SquareRootImpact(0.5), 1)
        # A fallback cannot help: some envs report a one-return window's volatility as 0,
        # a valid (flat) estimate, so the fallback would never apply and impact would be 0.
        with pytest.raises(ValueError, match="lookback=1"):
            build(SquareRootImpact(0.5, fallback_volatility=0.02), 1)
        env = build(SquareRootImpact(0.5), 2)  # two returns give an estimate
        env.reset(seed=0)
        action = {"market": np.ones(n + 1, np.float32), "trading": 2}.get(
            kind, np.array([1.0, 1.0, 0.0], np.float32)
        )
        assert env.step(action).info["cost"] > 0.0


class TestZeroInputsAreValid:
    """A finite zero volatility or spread counted as missing and crashed flat windows."""

    @staticmethod
    def flat_prices() -> np.ndarray:
        rng = np.random.default_rng(0)
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
        prices[100:110] = prices[100]  # ten flat days: zero volatility, a valid estimate
        return prices

    def test_model_level(self) -> None:
        zero_vol = SquareRootImpact(0.5)(
            np.array([0.1, 0.1]),
            equity=1e6,
            volatility=np.zeros(2),
            volume=np.full(2, 1e7),
        )
        assert zero_vol == 0.0
        # A zero spread is charged as zero, not replaced by the other assets' median ...
        trades = np.array([0.1, 0.1])
        assert SpreadCost()(trades, spread=np.array([0.0, 0.002])) == pytest.approx(1e-4)
        assert SpreadCost()(trades, spread=np.zeros(2)) == 0.0
        # ... while NaN (missing) and negative (invalid) spreads still take the median.
        assert SpreadCost()(trades, spread=np.array([np.nan, 0.002])) == pytest.approx(2e-4)
        assert SpreadCost()(trades, spread=np.array([-0.001, 0.002])) == pytest.approx(2e-4)

    def test_flat_window_trades_at_zero_impact(self) -> None:
        prices = self.flat_prices()
        env = TradingEnv(
            data=MarketData(prices, fields={"volume": np.full(200, 1e5)}),
            lookback=5,
            episode_length=150,
            transaction_cost=SquareRootImpact(),
            sampler=0,
        )
        env.reset(seed=0)
        costs = [env.step(t % 3).info["cost"] for t in range(150)]  # trades every step
        assert min(costs) == 0.0 and max(costs) > 0.0

    def test_zero_quoted_spread_is_free(self) -> None:
        spread = np.full(200, 4e-4)
        spread[50] = 0.0
        env = TradingEnv(
            data=MarketData(self.flat_prices(), fields={"spread": spread}),
            lookback=5,
            episode_length=150,
            transaction_cost=SpreadCost(),
            sampler=0,
        )
        env.reset(seed=0)
        costs = [env.step(t % 3).info["cost"] for t in range(150)]
        assert costs[45] == 0.0 and min(costs[:45]) > 0.0


class TestWarmup:
    """Test episodes could decide on the history rows a split prepends (in-sample rows)."""

    def test_split_and_walk_forward_record_the_history(self) -> None:
        data = make_data(dates=False)
        train, test = data.split(0.8, embargo=5, history=20)
        assert train.warmup == 0 and test.warmup == 20 and test.timestamps[20] == 245
        assert data.split(0.8)[1].warmup == 0 and data.warmup == 0
        assert data.split(10, history=50)[1].warmup == 10  # history clipped at row 0
        _, fold = next(data.walk_forward(100, 50, embargo=3, history=10))
        assert fold.warmup == 10 and fold.timestamps[10] == 103

    def test_derived_data_keep_it_correct(self) -> None:
        _, test = make_data().split(0.8, history=20)  # warm-up rows 220..239
        assert test[5:].warmup == 15 and test[25:].warmup == 0 and test[:10].warmup == 10
        assert test[::2].warmup == 10 and test[-30:].warmup == 0
        assert test.between(test.timestamps[12]).warmup == 8
        assert test.select(["A0"]).warmup == 20
        assert test.with_fields(x=np.zeros(test.prices.shape)).warmup == 20
        assert test.with_exogenous(y=np.zeros(len(test))).warmup == 20
        assert test.split(0.5)[0].warmup == 20 and "warmup=20" in repr(test)
        with pytest.raises(ValueError, match="warmup"):
            MarketData(np.ones((3, 1)), warmup=4)
        with pytest.raises(TypeError, match="warmup"):
            MarketData(np.ones((3, 1)), warmup=True)

    def test_sampling_skips_the_warmup(self) -> None:
        rng = np.random.default_rng(0)
        kwargs: dict[str, Any] = {"n_starts": 20, "episode_length": 4, "lookback": 3}
        assert first_start(3, 8) == 5 and first_start(10, 8) == 0
        starts = {sample_start(RandomStart(), rng, warmup=8, **kwargs) for _ in range(300)}
        assert starts == set(range(5, 20))
        assert sample_start(FixedStart(0), rng, warmup=8, **kwargs) == 5
        assert sample_start(FixedStart(0), rng, **kwargs) == 0
        with pytest.raises(ValueError, match="warm-up"):
            sample_start(RandomStart(), rng, n_starts=5, episode_length=4, lookback=3, warmup=8)
        with pytest.raises(ValueError, match=r"15 windows after the 8 warm-up rows"):
            sample_start(FixedStart(15), rng, warmup=8, **kwargs)
        with pytest.raises(ValueError) as plain:
            sample_start(FixedStart(25), rng, **kwargs)
        assert "warm-up" not in str(plain.value)  # no warm-up: the sampler's own message
        times = np.arange(30)
        assert resolve_start(8, timestamps=times, lookback=3, n_starts=20, warmup=8) == 5
        with pytest.raises(ValueError, match="warm-up"):
            resolve_start(6, timestamps=times, lookback=3, n_starts=20, warmup=8)

    def test_no_env_decides_on_a_warmup_row(self) -> None:
        _, test = make_data(dates=False).split(0.8, embargo=3, history=30)
        first = 243  # the first test row (row 30); rows 0..29, stamped 213..242, are warm-up
        complete = test.select(["A0", "A1"])
        envs: list[Any] = [
            MarketEnv(data=test, lookback=5, episode_length=10),
            TradingEnv(data=test.select(["A0"]), lookback=5, episode_length=10),
            PortfolioEnv(data=complete, lookback=5, episode_length=10),
        ]
        for env in envs:
            seen = {env.reset(seed=seed).info["timestamp"] for seed in range(40)}
            assert min(seen) >= first, type(env).__name__
            with pytest.raises(ValueError, match="warm-up"):
                env.reset(options={"start": Time(230)})
            with pytest.raises(ValueError, match="warm-up"):
                env.reset(options={"start": 17})  # the same row, by position
            assert env.reset(options={"start": Time(first)}).info["timestamp"] == first
            assert env.reset(options={"start": 30}).info["timestamp"] == first
        fixed = MarketEnv(data=test, lookback=5, episode_length=10, sampler=0)
        assert fixed.reset(seed=0).info["timestamp"] == first
        for vec in (
            TradingVecEnv(8, data=test.select(["A0"]), lookback=5, episode_length=10),
            PortfolioVecEnv(8, data=complete, lookback=5, episode_length=10),
        ):
            stamps = [info["timestamp"] for info in vec.reset(seed=0).infos]
            assert min(stamps) >= first

    def test_sampler_errors_explain_the_positions(self) -> None:
        # Sampler positions count the windows after the warm-up; the error used to quote
        # them as if they were window starts ("beyond the last window start 179").
        rng = np.random.default_rng(0)
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (400, 2)), axis=0))
        _, test = MarketData(prices).split(0.5, history=60)
        options: dict[str, Any] = {"lookback": 10, "episode_length": 20}
        expected = (
            r"position 200 is beyond the last of the 180 windows after the 60 warm-up rows "
            r"\(positions 0\.\.179: window starts 50\.\.229, first decisions at rows 60\.\.239\)"
        )
        with pytest.raises(ValueError, match=expected):
            PortfolioEnv(data=test, sampler=200, **options).reset(seed=0)
        last = PortfolioEnv(data=test, sampler=179, **options).reset(seed=0)
        assert last.info["timestamp"] == test.timestamps[239]


class TestTimezones:
    """Timezone-aware timestamps used to be shifted to UTC with only a warning."""

    def test_aware_values_are_refused(self) -> None:
        aware = [dt.datetime(2024, 1, d, 16, tzinfo=dt.timezone.utc) for d in (1, 2, 3)]
        with pytest.raises(ValueError, match=r"tz_localize\(None\)"):
            MarketData(np.ones((3, 1)), timestamps=aware)
        offsets = ["2024-01-01T16:00+05:00", "2024-01-02T16:00+05:00", "2024-01-03T16:00+05:00"]
        with pytest.raises(ValueError, match="timezone-aware"):
            MarketData(np.ones((3, 1)), timestamps=offsets)
        data = MarketData(np.ones((3, 1)), timestamps=np.datetime64("2024-01-01") + np.arange(3))
        with pytest.raises(ValueError, match="timezone-aware"):
            data.index_of(aware[1])
        with pytest.raises(ValueError, match="timezone-aware"):
            resolve_start(aware[1], timestamps=data.timestamps, lookback=0, n_starts=2)
        assert data.index_of(dt.datetime(2024, 1, 2)) == 1  # naive values are fine

    def test_aware_pandas_index_is_refused(self) -> None:
        pd = pytest.importorskip("pandas")
        index = pd.date_range("2024-01-01 16:00", periods=4, freq="D", tz="America/New_York")
        frame = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]}, index=index)
        with pytest.raises(ValueError, match=r"tz_convert\('UTC'\)\.tz_localize\(None\)"):
            MarketData.from_pandas(frame)
        with pytest.raises(ValueError, match="timezone-aware"):
            MarketData(frame.to_numpy(), timestamps=index)
        local = MarketData.from_pandas(frame.tz_localize(None))
        utc = MarketData.from_pandas(frame.tz_convert("UTC").tz_localize(None))
        assert local.timestamps[0] == np.datetime64("2024-01-01T16:00")
        assert utc.timestamps[0] == np.datetime64("2024-01-01T21:00")


class TestRowsAndTimestamps:
    """An int was a row position on datetime axes but a timestamp on numeric ones."""

    def test_integers_are_row_positions_on_every_axis(self) -> None:
        prices = 100 + np.arange(1000.0)
        late = MarketData(prices)[100:]  # numeric timestamps 100..999
        a, b = late.split(300)  # used to cut at timestamp 300, i.e. row 200
        assert len(a) == 300 and b.timestamps[0] == 400
        assert late.index_of(300) == 300 and late.index_of(-1) == 899
        assert late.between(100, 200).timestamps[[0, -1]].tolist() == [200, 299]
        _, test = MarketData(prices).split(0.8)  # timestamps 800..999
        first, second = test.split(50)  # used to raise "got rows [0]"
        assert (len(first), second.timestamps[0]) == (50, 850)
        dated = make_data()
        assert dated.index_of(5) == 5 and len(dated.split(100)[0]) == 100

    def test_time_looks_up_timestamps(self) -> None:
        late = MarketData(100 + np.arange(1000.0))[100:]
        assert late.index_of(Time(300)) == 200 and late.index_of(300.0) == 200
        assert late.index_of(Time(300), side="right") == 201
        assert len(late.split(Time(300))[0]) == 200
        with pytest.raises(ValueError, match=r"pass Time\(300\.0\) to split at a timestamp"):
            late.split(300.0)  # a float is a fraction in split()
        assert late.between(Time(150), Time(250)).timestamps[[0, -1]].tolist() == [150, 249]
        dated = make_data()
        assert dated.index_of(Time("2021-01-03")) == dated.index_of("2021-01-03") == 2
        assert repr(Time(5)) == "Time(5)" and Time(5) == Time(5)
        for bad in (None, True, [1, 2]):
            with pytest.raises(TypeError, match="Time takes"):
                Time(bad)

    def test_timestamps_must_suit_the_axis(self) -> None:
        numeric = MarketData(100 + np.arange(1000.0))
        dated = make_data()
        with pytest.raises(ValueError, match="not a number"):
            numeric.index_of("2020-03-02")
        with pytest.raises(ValueError, match="not a number"):
            numeric.split(Time(np.datetime64("2020-03-02")))
        with pytest.raises(ValueError, match="not a date"):
            dated.index_of(1.5)
        with pytest.raises(ValueError, match="not a date"):
            dated.between(Time(5))
        with pytest.raises(ValueError, match="not finite"):
            numeric.index_of(np.nan)
        with pytest.raises(TypeError, match="row position"):
            numeric.index_of(True)

    def test_start_options(self) -> None:
        # Regression: a date on a numeric axis was compared as a string with the integer
        # timestamps and started the episode at an arbitrary row (2021 for "2020-03-02").
        with pytest.raises(ValueError, match=r"start '2020-03-02' is not a number"):
            resolve_start("2020-03-02", timestamps=np.arange(5000), lookback=10, n_starts=100)
        _, test = MarketData(100 + np.arange(1000.0)).split(0.5)  # timestamps 500..999
        kwargs: dict[str, Any] = {"timestamps": test.timestamps, "lookback": 10, "n_starts": 470}
        assert resolve_start(300, **kwargs) == 290  # row 300
        assert resolve_start(Time(800), **kwargs) == resolve_start(800.0, **kwargs) == 290
        with pytest.raises(ValueError, match=r"row position; for the timestamp 800, pass Time"):
            resolve_start(800, **kwargs)
        env = MarketEnv(data=test, lookback=10, episode_length=20)
        stamp = env.reset(seed=3).info["timestamp"]
        assert env.reset(options={"start": Time(stamp)}).info["timestamp"] == stamp


@pytest.mark.parametrize(
    "build",
    [
        lambda: ProportionalCost(np.nan),
        lambda: ProportionalCost([1e-3, -1e-3]),
        lambda: SpreadCost(half_spread=np.nan),
        lambda: SpreadCost(half_spread=np.inf),
        lambda: SpreadCost(half_spread=[1e-4, np.nan]),
        lambda: SpreadCost(half_spread=-1e-4),
        lambda: SpreadCost(fallback_spread=np.nan),
        lambda: FixedCost(np.nan),
        lambda: FixedCost(np.inf),
        lambda: FixedCost(-1.0),
        lambda: FixedCost(1.0, threshold=np.nan),
        lambda: FixedCost(1.0, threshold=-1.0),
        lambda: FixedCost(1.0, threshold=np.inf),
        lambda: SquareRootImpact(np.nan),
        lambda: SquareRootImpact(np.inf),
        lambda: SquareRootImpact(-0.5),
        lambda: SquareRootImpact(fallback_volatility=np.inf),
        lambda: as_cost_model(np.nan),
        lambda: AccountConfig(max_leverage=np.nan),
        lambda: AccountConfig(max_leverage=np.inf),
        lambda: AccountConfig(max_leverage=0.0),
        lambda: AccountConfig(cash_rate=np.nan),
        lambda: AccountConfig(margin_rate=np.inf),
        lambda: AccountConfig(borrow_rate=np.nan),
        lambda: AccountConfig(borrow_rate=-0.01),
    ],
)
def test_invalid_model_parameters_are_refused(build: Any) -> None:
    # Regression: NaN and infinite parameters (and negative thresholds) were accepted and
    # priced trades at NaN, inf or the wrong number of tickets.
    with pytest.raises(ValueError, match=r"finite|positive|non-negative"):
        build()


def test_valid_edge_parameters_are_accepted() -> None:
    trades = np.array([0.1, 0.0, 0.0])
    assert FixedCost(5.0, threshold=0.0)(trades, equity=1e6) == pytest.approx(5e-6)
    assert SpreadCost(half_spread=0.0)(trades) == 0.0
    assert SquareRootImpact(0.0).coefficient == 0.0
    assert AccountConfig(cash_rate=-0.005, margin_rate=-0.001).effective_margin_rate == -0.001


def test_settlement_and_time_are_exported() -> None:
    import sekai.finance as finance
    from sekai.finance import account, data

    assert finance.Settlement is account.Settlement and finance.Time is data.Time
    assert {"Settlement", "Time"} <= set(finance.__all__)
    assert all(hasattr(finance, name) for name in finance.__all__)
