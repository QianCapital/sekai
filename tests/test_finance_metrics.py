"""Tests for sekai.finance.metrics: return-series metrics, summaries and RecordPerformance."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

import sekai
from sekai.finance.metrics import (
    PerformanceSummary,
    PnLSummary,
    RecordPerformance,
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    conditional_value_at_risk,
    drawdown_series,
    equity_curve,
    hit_rate,
    max_drawdown,
    performance_summary,
    pnl_summary,
    returns_from_equity,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    value_at_risk,
)

SERIES_FUNCTIONS = [
    equity_curve,
    total_return,
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
    drawdown_series,
    max_drawdown,
    calmar_ratio,
    value_at_risk,
    conditional_value_at_risk,
    hit_rate,
    performance_summary,
    pnl_summary,
]


# ---------------------------------------------------------------------------
# Return-series metrics
# ---------------------------------------------------------------------------


class TestEquityAndReturns:
    def test_equity_curve(self) -> None:
        np.testing.assert_allclose(equity_curve([0.1, -0.5, 0.2]), [1.0, 1.1, 0.55, 0.66])
        np.testing.assert_allclose(
            equity_curve([0.1, -0.5, 0.2], initial=100.0), [100.0, 110.0, 55.0, 66.0]
        )
        assert equity_curve([0.1]).dtype == np.float64

    def test_equity_curve_wipe_out_is_absorbing(self) -> None:
        np.testing.assert_allclose(equity_curve([0.5, -1.0, 0.3]), [1.0, 1.5, 0.0, 0.0])
        # A loss beyond -100% cannot take equity below zero, nor can it "recover".
        np.testing.assert_allclose(equity_curve([-2.0, -2.0]), [1.0, 0.0, 0.0])

    def test_round_trip(self) -> None:
        r = np.random.default_rng(0).normal(0.0005, 0.01, size=50)
        np.testing.assert_allclose(returns_from_equity(equity_curve(r)), r, rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(returns_from_equity(equity_curve(r, initial=250.0)), r)

    def test_returns_from_equity(self) -> None:
        np.testing.assert_allclose(returns_from_equity([100.0, 110.0, 99.0]), [0.1, -0.1])

    def test_returns_from_equity_wipe_out(self) -> None:
        np.testing.assert_allclose(returns_from_equity([1.0, 0.5, 0.0]), [-0.5, -1.0])
        np.testing.assert_allclose(returns_from_equity([1.0, 2.0, -1.0]), [1.0, -1.0])
        # The wipe-out round-trips through equity_curve.
        np.testing.assert_allclose(equity_curve(returns_from_equity([1.0, 0.5, 0.0])), [1, 0.5, 0])

    def test_returns_from_equity_validation(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            returns_from_equity([1.0, 0.0, 1.0])
        with pytest.raises(ValueError, match="positive"):
            returns_from_equity([-1.0, 1.0])
        with pytest.raises(ValueError, match="two values"):
            returns_from_equity([1.0])

    def test_total_return(self) -> None:
        assert total_return([0.1, -0.5, 0.2]) == pytest.approx(-0.34)
        assert total_return([0.0]) == 0.0
        assert total_return([-1.5]) == -1.0  # limited liability
        assert isinstance(total_return([0.1]), float)


class TestAnnualisation:
    def test_annualized_return_constant_daily(self) -> None:
        returns = np.full(252, 0.001)
        assert annualized_return(returns) == pytest.approx(1.001**252 - 1, rel=1e-12)
        # Half a year of the same returns annualises to the same rate.
        assert annualized_return(returns[:126]) == pytest.approx(1.001**252 - 1, rel=1e-12)

    def test_annualized_return_geometric(self) -> None:
        # Two periods of a four-per-year series: (1.1 * 0.9) ** 2 - 1.
        assert annualized_return([0.1, -0.1], periods_per_year=4) == pytest.approx(0.99**2 - 1)

    def test_annualized_return_wipe_out(self) -> None:
        assert annualized_return([0.1, -1.0, 0.5]) == -1.0
        assert annualized_return([-1.5, 0.2]) == -1.0

    def test_annualized_return_overflow_is_inf(self) -> None:
        assert annualized_return([1.0], periods_per_year=252 * 390) == math.inf

    def test_annualized_volatility(self) -> None:
        assert annualized_volatility([0.01, 0.02, 0.03], periods_per_year=4) == pytest.approx(0.02)
        assert annualized_volatility([0.05]) == 0.0
        assert annualized_volatility(np.full(100, 0.001)) == 0.0


class TestRiskAdjusted:
    def test_sharpe(self) -> None:
        # mean 0.02, sample std 0.01.
        assert sharpe_ratio([0.01, 0.02, 0.03], periods_per_year=1) == pytest.approx(2.0)
        assert sharpe_ratio([0.01, 0.02, 0.03]) == pytest.approx(2.0 * math.sqrt(252))

    def test_sharpe_risk_free_is_annual(self) -> None:
        # 4% a year over 4 periods is 1% a period: excess [0, 0.01, 0.02].
        value = sharpe_ratio([0.01, 0.02, 0.03], periods_per_year=4, risk_free=0.04)
        assert value == pytest.approx(0.01 / 0.01 * 2.0)

    def test_sharpe_degenerate(self) -> None:
        assert sharpe_ratio(np.full(252, 0.001)) == 0.0  # std is 0
        assert sharpe_ratio([0.05]) == 0.0  # n < 2

    def test_sortino(self) -> None:
        returns = [0.02, -0.01, 0.03, -0.02]
        # mean 0.005; downside deviation sqrt((0.01**2 + 0.02**2) / 4) = 0.005 * sqrt(5).
        assert sortino_ratio(returns, periods_per_year=1) == pytest.approx(1 / math.sqrt(5))
        assert sortino_ratio(returns) == pytest.approx(math.sqrt(252) / math.sqrt(5))

    def test_sortino_target(self) -> None:
        # Target 4% a year at 4 periods: excess [0.01, -0.02], dd = sqrt(0.02**2 / 2).
        value = sortino_ratio([0.02, -0.01], periods_per_year=4, target=0.04)
        assert value == pytest.approx(-0.005 / math.sqrt(0.0002) * 2.0)

    def test_sortino_inf_rules(self) -> None:
        assert sortino_ratio([0.01, 0.02]) == math.inf
        assert sortino_ratio([0.0, 0.0]) == 0.0
        assert sortino_ratio([0.01], target=252 * 0.01) == 0.0  # exactly on target
        assert sortino_ratio([-0.01, -0.02]) < 0.0

    def test_calmar(self) -> None:
        # Annualised over 3 periods per year is the total return, -0.34; max drawdown 0.5.
        assert calmar_ratio([0.1, -0.5, 0.2], periods_per_year=3) == pytest.approx(-0.68)

    def test_calmar_inf_rules(self) -> None:
        assert calmar_ratio([0.01, 0.02]) == math.inf
        assert calmar_ratio([0.0, 0.0]) == 0.0

    @pytest.mark.parametrize("rate", [0.01, 0.02, 0.03, 0.04, 0.045, 0.05, 0.0525])
    def test_cash_at_the_risk_free_rate_scores_zero(self, rate: float) -> None:
        # Regression: the excess return of cash over the same rate is zero up to rounding
        # (about +-1e-19), and its sign used to decide the Sortino ratio: -sqrt(252) for
        # some rates, inf for others.
        cash = np.full(252, (1.0 + rate / 252) - 1.0)
        assert sortino_ratio(cash, target=rate) == 0.0
        assert sharpe_ratio(cash, risk_free=rate) == 0.0
        summary = performance_summary(cash, risk_free=rate)
        assert summary.sortino_ratio == 0.0 and summary.sharpe_ratio == 0.0
        assert sortino_ratio(cash) == math.inf  # above a zero target it is genuinely positive

    def test_series_constant_up_to_rounding_have_no_volatility(self) -> None:
        # Returns derived from an equity curve carry rounding of a few ulps of 1.
        noise = np.array([0.0, 1.0, -1.0, 2.0] * 63) * np.finfo(np.float64).eps
        cash = 0.04 / 252 + noise
        assert annualized_volatility(cash) == 0.0 and sharpe_ratio(cash) == 0.0
        assert sortino_ratio(cash, target=0.04) == 0.0
        # Genuine (if small) variation is kept, and P&L in tiny units is not rounding.
        assert sharpe_ratio(0.04 / 252 + 1e-10 * noise / noise.max()) > 0.0
        assert pnl_summary(np.array([1e-20, 3e-20, 2e-20])).pnl_volatility > 0.0

    def test_record_performance_cash_at_the_risk_free_rate(self) -> None:
        from sekai.envs.finance import MarketEnv
        from sekai.finance import GBM

        for rate in (0.01, 0.02, 0.03):
            market = MarketEnv(
                model=GBM(0.05, 0.2, n_assets=2), lookback=3, episode_length=30, cash_rate=rate
            )
            env = RecordPerformance(market, risk_free=rate)
            env.reset(seed=0)
            hold_cash = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            while not (result := env.step(hold_cash)).done:
                pass
            summary = result.info["performance"]
            assert summary.sortino_ratio == 0.0 and summary.sharpe_ratio == 0.0


class TestDrawdown:
    def test_drawdown_series(self) -> None:
        dd = drawdown_series([0.1, -0.5, 0.2])
        np.testing.assert_allclose(dd, [0.0, 0.0, 0.5, 1 - 0.66 / 1.1])
        assert max_drawdown([0.1, -0.5, 0.2]) == pytest.approx(0.5)

    def test_monotonic_has_no_drawdown(self) -> None:
        assert max_drawdown([0.01, 0.02, 0.0]) == 0.0
        assert np.all(drawdown_series([0.01, 0.02]) == 0.0)

    def test_drawdown_from_start(self) -> None:
        np.testing.assert_allclose(drawdown_series([-0.2, -0.5]), [0.0, 0.2, 0.6])

    def test_wipe_out_is_full_drawdown(self) -> None:
        np.testing.assert_allclose(drawdown_series([0.5, -1.2, 0.3]), [0, 0, 1, 1])
        assert max_drawdown([-1.0]) == 1.0


class TestTailAndHitRate:
    SAMPLE = (-0.05, -0.02, 0.0, 0.01, 0.03)

    def test_var_cvar_on_known_sample(self) -> None:
        returns = np.random.default_rng(1).permutation(self.SAMPLE)  # order must not matter
        # quantile 0.25 of 5 points sits exactly on the second order statistic.
        assert value_at_risk(returns, level=0.75) == pytest.approx(0.02)
        assert conditional_value_at_risk(returns, level=0.75) == pytest.approx(0.035)

    def test_var_interpolates(self) -> None:
        returns = np.arange(-10, 10) / 100.0  # -0.10 .. 0.09
        # Position 19 * 0.1 = 1.9 between -0.09 and -0.08.
        assert value_at_risk(returns, level=0.9) == pytest.approx(0.081)
        assert conditional_value_at_risk(returns, level=0.9) == pytest.approx(0.095)
        # Position 0.95: tail is just the worst return.
        assert value_at_risk(returns) == pytest.approx(0.0905)
        assert conditional_value_at_risk(returns) == pytest.approx(0.10)

    def test_cvar_at_least_var(self) -> None:
        returns = np.random.default_rng(2).standard_t(3, size=500) * 0.01
        for level in (0.9, 0.95, 0.99):
            assert conditional_value_at_risk(returns, level) >= value_at_risk(returns, level)

    def test_var_of_gains_is_negative_and_zero_is_positive_zero(self) -> None:
        assert value_at_risk([0.01, 0.02]) < 0.0
        assert math.copysign(1.0, value_at_risk([0.0, 0.0])) == 1.0

    @pytest.mark.parametrize("level", [0.0, 1.0, -0.5, 1.5, math.nan])
    def test_level_validation(self, level: float) -> None:
        with pytest.raises(ValueError, match="level"):
            value_at_risk([0.01, -0.01], level=level)
        with pytest.raises(ValueError, match="level"):
            conditional_value_at_risk([0.01, -0.01], level=level)

    def test_hit_rate(self) -> None:
        assert hit_rate([0.01, 0.0, -0.02, 0.03]) == pytest.approx(2 / 3)
        assert hit_rate([0.0, 0.0]) == 0.0
        assert hit_rate([-0.01]) == 0.0


class TestValidation:
    @pytest.mark.parametrize("fn", SERIES_FUNCTIONS)
    @pytest.mark.parametrize(
        "bad",
        [[], [0.01, math.nan], [math.inf], [[0.01, 0.02]], ["x"]],
        ids=["empty", "nan", "inf", "2d", "non-numeric"],
    )
    def test_bad_series(self, fn: Any, bad: Any) -> None:
        with pytest.raises(ValueError):
            fn(bad)

    @pytest.mark.parametrize(
        "fn", [annualized_return, annualized_volatility, sharpe_ratio, sortino_ratio, calmar_ratio]
    )
    @pytest.mark.parametrize("ppy", [0, -252, math.inf, math.nan])
    def test_bad_periods_per_year(self, fn: Any, ppy: float) -> None:
        with pytest.raises(ValueError, match="periods_per_year"):
            fn([0.01, 0.02], periods_per_year=ppy)

    def test_bad_initial_and_rates(self) -> None:
        with pytest.raises(ValueError, match="initial"):
            equity_curve([0.01], initial=0.0)
        with pytest.raises(ValueError, match="risk_free"):
            sharpe_ratio([0.01, 0.02], risk_free=math.nan)
        with pytest.raises(ValueError, match="target"):
            sortino_ratio([0.01, 0.02], target=math.inf)

    def test_inputs_are_not_mutated(self) -> None:
        returns = np.array([0.1, -0.5, 0.2])
        before = returns.copy()
        performance_summary(returns)
        equity_curve(returns)
        np.testing.assert_array_equal(returns, before)

    def test_return_types_are_python_floats(self) -> None:
        returns = [0.01, -0.02, 0.03]
        for fn in (total_return, sharpe_ratio, max_drawdown, value_at_risk, hit_rate):
            assert type(fn(returns)) is float


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


class TestPerformanceSummary:
    def test_matches_functions(self) -> None:
        r = np.random.default_rng(3).normal(0.0005, 0.01, size=252)
        s = performance_summary(r, 252, risk_free=0.02, level=0.99)
        assert s.n_periods == 252
        assert s.total_return == pytest.approx(total_return(r))
        assert s.annualized_return == pytest.approx(annualized_return(r))
        assert s.annualized_volatility == pytest.approx(annualized_volatility(r))
        assert s.sharpe_ratio == pytest.approx(sharpe_ratio(r, risk_free=0.02))
        assert s.sortino_ratio == pytest.approx(sortino_ratio(r, target=0.02))
        assert s.max_drawdown == pytest.approx(max_drawdown(r))
        assert s.calmar_ratio == pytest.approx(calmar_ratio(r))
        assert s.value_at_risk == pytest.approx(value_at_risk(r, 0.99))
        assert s.conditional_value_at_risk == pytest.approx(conditional_value_at_risk(r, 0.99))
        assert s.hit_rate == pytest.approx(hit_rate(r))
        assert s.mean_turnover is None and s.total_cost is None

    def test_turnover_and_costs(self) -> None:
        s = performance_summary([0.01, 0.02], turnover=[0.5, 0.2, 0.3], costs=[0.001, 0.002])
        assert s.mean_turnover == pytest.approx(1 / 3)
        assert s.total_cost == pytest.approx(0.003)
        with pytest.raises(ValueError, match="turnover"):
            performance_summary([0.01], turnover=[math.nan])

    def test_frozen_and_as_dict(self) -> None:
        s = performance_summary([0.01, -0.02, 0.03])
        with pytest.raises(AttributeError):
            s.total_return = 1.0  # type: ignore[misc]
        d = s.as_dict()
        assert d["n_periods"] == 3 and d["mean_turnover"] is None
        assert list(d)[:2] == ["n_periods", "total_return"]
        assert len(d) == 13

    def test_str_renders(self) -> None:
        s = performance_summary([0.1, -0.5, 0.2], turnover=[0.5], costs=[0.01])
        text = str(s)
        lines = text.splitlines()
        assert len(lines) == 13
        assert any(line.startswith("max drawdown") and line.endswith("50.00%") for line in lines)
        assert any(line.startswith("total return") and line.endswith("-34.00%") for line in lines)
        assert "Sharpe ratio" in text and "mean turnover" in text and "total cost" in text
        assert len({len(line) for line in lines}) == 1  # values right-aligned in one column
        # Optional rows are left out when absent, and inf renders plainly.
        bare = str(performance_summary([0.01, 0.02]))
        assert "turnover" not in bare and "inf%" not in bare
        assert "inf" in bare  # Sortino and Calmar

    def test_pickles(self) -> None:
        import pickle

        s = performance_summary([0.01, -0.02])
        assert pickle.loads(pickle.dumps(s)) == s


class TestPnLSummary:
    def test_values(self) -> None:
        pnl = [1.0, 2.0, -4.0, 1.0, 3.0]  # cumulative 0, 1, 3, -1, 0, 3
        s = pnl_summary(pnl, periods_per_year=1)
        assert isinstance(s, PnLSummary)
        assert s.n_periods == 5
        assert s.total_pnl == pytest.approx(3.0)
        assert s.mean_pnl == pytest.approx(0.6)
        assert s.pnl_volatility == pytest.approx(np.std(pnl, ddof=1))
        assert s.sharpe_ratio == pytest.approx(0.6 / np.std(pnl, ddof=1))
        assert s.max_drawdown == pytest.approx(4.0)  # 3 down to -1, in absolute units
        assert s.hit_rate == pytest.approx(0.8)

    def test_drawdown_peak_starts_at_zero(self) -> None:
        assert pnl_summary([-2.0, 1.0]).max_drawdown == pytest.approx(2.0)
        assert pnl_summary([5.0, 1.0]).max_drawdown == 0.0

    def test_scale_is_absolute(self) -> None:
        small = pnl_summary([1.0, -2.0, 0.5])
        big = pnl_summary([1000.0, -2000.0, 500.0])
        assert big.max_drawdown == pytest.approx(1000 * small.max_drawdown)
        assert big.sharpe_ratio == pytest.approx(small.sharpe_ratio)

    def test_degenerate(self) -> None:
        s = pnl_summary([2.0])
        assert s.pnl_volatility == 0.0 and s.sharpe_ratio == 0.0
        assert pnl_summary([0.5, 0.5, 0.5]).sharpe_ratio == 0.0

    def test_str_and_as_dict(self) -> None:
        s = pnl_summary([1.0, 2.0, -4.0, 1.0, 3.0])
        assert s.as_dict()["max_drawdown"] == pytest.approx(4.0)
        text = str(s)
        assert "total P&L" in text and "max drawdown" in text
        assert len(text.splitlines()) == 7


# ---------------------------------------------------------------------------
# RecordPerformance
# ---------------------------------------------------------------------------


class EquityEnv(sekai.Env[np.ndarray, int]):
    """Equity compounds by a fixed per-episode rate each step; ends after `horizon` steps.

    Episode k grows at `rates[k % len(rates)]` per step and starts from `starts[k % len]`,
    so a wrapper that confuses episodes produces visibly wrong summaries.
    """

    def __init__(
        self,
        horizon: int = 3,
        rates: Sequence[float] = (0.01,),
        starts: Sequence[float] = (1.0,),
        *,
        report_reset_equity: bool = True,
        turnover: Sequence[float] | None = None,
        cost: Sequence[float] | None = None,
        equity_path: Sequence[float] | None = None,
    ) -> None:
        self.observation_space = sekai.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        self.action_space = sekai.Discrete(2)
        self.horizon = horizon
        self.rates = rates
        self.starts = starts
        self.report_reset_equity = report_reset_equity
        self.turnover = turnover
        self.cost = cost
        self.equity_path = equity_path
        self.episode = -1
        self.t = 0
        self.equity = 1.0
        self.infos: list[dict[str, Any]] = []
        self.async_steps = 0

    def _obs(self) -> np.ndarray:
        return np.array([self.equity], dtype=np.float32)

    def reset(
        self, *, seed: Any = None, options: dict[str, Any] | None = None
    ) -> sekai.ResetResult:
        self._set_rng(seed)
        self.episode += 1
        self.t = 0
        self.equity = self.starts[self.episode % len(self.starts)]
        info: dict[str, Any] = {"equity": self.equity} if self.report_reset_equity else {}
        info["episode_index"] = self.episode
        self.infos.append(info)
        return sekai.ResetResult(self._obs(), info)

    def step(self, action: int) -> sekai.StepResult:
        self.t += 1
        if self.equity_path is not None:
            self.equity = self.equity_path[self.t - 1]
        else:
            self.equity *= 1.0 + self.rates[self.episode % len(self.rates)]
        info: dict[str, Any] = {"equity": self.equity}
        if self.turnover is not None and self.t <= len(self.turnover):
            info["turnover"] = self.turnover[self.t - 1]
        if self.cost is not None and self.t <= len(self.cost):
            info["cost"] = self.cost[self.t - 1]
        self.infos.append(info)
        return sekai.StepResult(self._obs(), 0.0, self.t >= self.horizon, False, info)

    async def async_step(self, action: int) -> sekai.StepResult:
        await asyncio.sleep(0)
        self.async_steps += 1
        return self.step(action)


class WealthEnv(sekai.Env[np.ndarray, int]):
    """Cumulative wealth follows a fixed path (P&L mode); terminates at the end of it."""

    def __init__(self, path: Sequence[float], start: float | None = 0.0) -> None:
        self.observation_space = sekai.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        self.action_space = sekai.Discrete(2)
        self.path = path
        self.start = start
        self.t = 0

    def reset(
        self, *, seed: Any = None, options: dict[str, Any] | None = None
    ) -> sekai.ResetResult:
        self._set_rng(seed)
        self.t = 0
        info = {} if self.start is None else {"wealth": self.start}
        return sekai.ResetResult(np.zeros(1, np.float32), info)

    def step(self, action: int) -> sekai.StepResult:
        self.t += 1
        done = self.t >= len(self.path)
        return sekai.StepResult(
            np.zeros(1, np.float32), 0.0, done, False, {"wealth": self.path[self.t - 1]}
        )


def run_episode(env: sekai.Env[Any, Any], seed: int | None = None) -> sekai.StepResult:
    result: Any = env.reset(seed=seed)
    while not result.done:
        result = env.step(0)
    return result


class TestRecordPerformance:
    def test_summary_on_done_step(self) -> None:
        inner = EquityEnv(horizon=3, rates=(0.01,))
        env = RecordPerformance(inner)
        env.reset(seed=0)
        r1, r2 = env.step(0), env.step(0)
        assert "performance" not in r1.info and "performance" not in r2.info
        r3 = env.step(0)
        perf = r3.info["performance"]
        assert isinstance(perf, PerformanceSummary)
        assert perf.n_periods == 3
        assert perf.total_return == pytest.approx(1.01**3 - 1)
        assert perf.annualized_return == pytest.approx(1.01**252 - 1)
        assert perf.max_drawdown == 0.0
        assert perf.sharpe_ratio == 0.0  # constant returns
        assert perf.mean_turnover is None and perf.total_cost is None
        assert env.last_summary is perf and list(env.summaries) == [perf]
        assert r3.info["equity"] == pytest.approx(1.01**3)
        # The inner env's info dict is left untouched.
        assert all("performance" not in info for info in inner.infos)

    def test_returns_relative_to_reset_equity(self) -> None:
        env = RecordPerformance(EquityEnv(horizon=2, rates=(0.1,), starts=(100.0,)))
        perf = run_episode(env, seed=0).info["performance"]
        assert perf.total_return == pytest.approx(0.21)

    def test_missing_reset_equity_assumes_one(self) -> None:
        env = RecordPerformance(
            EquityEnv(horizon=2, report_reset_equity=False, equity_path=(1.05, 1.155))
        )
        perf = run_episode(env, seed=0).info["performance"]
        assert perf.total_return == pytest.approx(0.155)
        assert perf.hit_rate == 1.0

    def test_drawdown_and_wipe_out(self) -> None:
        env = RecordPerformance(EquityEnv(horizon=3, equity_path=(1.1, 0.55, 0.66)))
        perf = run_episode(env, seed=0).info["performance"]
        assert perf.max_drawdown == pytest.approx(0.5)
        assert perf.total_return == pytest.approx(-0.34)

        # Equity goes negative on step 2: that return is -100%, later returns are 0.
        env = RecordPerformance(EquityEnv(horizon=4, equity_path=(0.5, -0.2, 0.3, 0.6)))
        perf = run_episode(env, seed=0).info["performance"]
        assert perf.total_return == -1.0
        assert perf.annualized_return == -1.0
        assert perf.max_drawdown == 1.0
        assert perf.conditional_value_at_risk == pytest.approx(1.0)
        assert perf.hit_rate == 0.0  # active periods: -50%, -100%

    def test_deque_and_summary_aggregation(self) -> None:
        env = RecordPerformance(EquityEnv(horizon=2, rates=(0.01, 0.02, 0.03)), deque_size=2)
        assert env.summary() == {} and env.last_summary is None
        for k in range(3):
            run_episode(env, seed=k)
        assert len(env.summaries) == 2
        totals = [s.total_return for s in env.summaries]
        assert totals == pytest.approx([1.02**2 - 1, 1.03**2 - 1])
        agg = env.summary()
        assert agg["total_return"] == pytest.approx(np.mean(totals))
        assert agg["n_periods"] == 2.0
        assert "mean_turnover" not in agg and "total_cost" not in agg  # None in every summary
        assert all(isinstance(v, float) for v in agg.values())

    def test_turnover_and_cost(self) -> None:
        env = RecordPerformance(
            EquityEnv(horizon=3, turnover=(0.5, 0.2, 0.3), cost=(0.001, 0.002, 0.003))
        )
        perf = run_episode(env, seed=0).info["performance"]
        assert perf.mean_turnover == pytest.approx(1 / 3)
        assert perf.total_cost == pytest.approx(0.006)
        assert env.summary()["total_cost"] == pytest.approx(0.006)

    def test_turnover_only_on_some_steps(self) -> None:
        env = RecordPerformance(EquityEnv(horizon=3, turnover=(0.4,)))
        perf = run_episode(env, seed=0).info["performance"]
        assert perf.mean_turnover == pytest.approx(0.4)
        assert perf.total_cost is None

    def test_custom_and_disabled_keys(self) -> None:
        env = RecordPerformance(EquityEnv(horizon=2, turnover=(0.5, 0.5)), turnover_key=None)
        assert run_episode(env, seed=0).info["performance"].mean_turnover is None

    def test_missing_key_error_names_keys(self) -> None:
        env = RecordPerformance(EquityEnv(horizon=2), equity_key="nav")
        env.reset(seed=0)
        with pytest.raises(KeyError, match=r"'nav'.*available keys: \['equity'\]"):
            env.step(0)

    def test_mid_episode_reset_discards_partial_episode(self) -> None:
        env = RecordPerformance(EquityEnv(horizon=3, rates=(0.5, 0.01)))
        env.reset(seed=0)
        env.step(0)
        perf = run_episode(env).info["performance"]
        assert perf.n_periods == 3
        assert perf.total_return == pytest.approx(1.01**3 - 1)

    def test_pnl_mode(self) -> None:
        env = RecordPerformance(WealthEnv([1.0, 3.0, -1.0, 0.0, 3.0]), pnl_key="wealth")
        perf = run_episode(env, seed=0).info["performance"]
        assert isinstance(perf, PnLSummary)
        assert perf.n_periods == 5
        assert perf.total_pnl == pytest.approx(3.0)
        assert perf.max_drawdown == pytest.approx(4.0)
        assert perf.hit_rate == pytest.approx(0.8)
        assert env.summary()["total_pnl"] == pytest.approx(3.0)

    def test_pnl_mode_start_from_reset_info(self) -> None:
        env = RecordPerformance(WealthEnv([12.0, 9.0], start=10.0), pnl_key="wealth")
        perf = run_episode(env, seed=0).info["performance"]
        assert perf.total_pnl == pytest.approx(-1.0)
        assert perf.max_drawdown == pytest.approx(3.0)
        env = RecordPerformance(WealthEnv([2.0, 1.0], start=None), pnl_key="wealth")
        assert run_episode(env, seed=0).info["performance"].total_pnl == pytest.approx(1.0)

    def test_with_inner_autoreset(self) -> None:
        inner = EquityEnv(horizon=3, rates=(0.01, -0.02), starts=(1.0, 50.0))
        env = RecordPerformance(sekai.AutoReset(inner))
        env.reset(seed=0)
        results = [env.step(0) for _ in range(6)]  # two episodes, no reset() in between
        done = [i for i, r in enumerate(results) if "performance" in r.info]
        assert done == [2, 5]
        first, second = (results[i].info["performance"] for i in done)
        assert first.n_periods == 3 and second.n_periods == 3
        assert first.total_return == pytest.approx(1.01**3 - 1)
        # Measured from the second episode's reset equity (50), not the first's terminal one.
        assert second.total_return == pytest.approx(0.98**3 - 1)
        assert second.max_drawdown == pytest.approx(1 - 0.98**3)
        assert list(env.summaries) == [first, second]
        assert env.summary()["n_periods"] == 3.0
        # AutoReset's own keys survive alongside the summary.
        assert results[2].info["final_info"]["equity"] == pytest.approx(1.01**3)
        assert "reset_info" in results[2].info

    def test_with_inner_autoreset_pnl_mode(self) -> None:
        env = RecordPerformance(sekai.AutoReset(WealthEnv([1.0, 3.0])), pnl_key="wealth")
        env.reset(seed=0)
        results = [env.step(0) for _ in range(4)]
        summaries = [r.info["performance"] for r in results if "performance" in r.info]
        assert [s.total_pnl for s in summaries] == pytest.approx([3.0, 3.0])

    def test_async_path(self) -> None:
        inner = EquityEnv(horizon=2, rates=(0.1,))
        env = RecordPerformance(inner)
        asyncio.run(env.async_reset(seed=0))
        r1 = asyncio.run(env.async_step(0))
        r2 = asyncio.run(env.async_step(0))
        assert "performance" not in r1.info
        assert r2.info["performance"].total_return == pytest.approx(0.21)
        assert inner.async_steps == 2  # the inner env's async path was used

    def test_async_matches_sync_with_autoreset(self) -> None:
        def make() -> RecordPerformance[np.ndarray, int]:
            inner = EquityEnv(horizon=2, rates=(0.01, -0.03), starts=(1.0, 2.0))
            return RecordPerformance(sekai.AutoReset(inner), turnover_key=None)

        sync_env = make()
        sync_env.reset(seed=0)
        for _ in range(4):
            sync_env.step(0)

        async def drive(env: RecordPerformance[np.ndarray, int]) -> None:
            await env.async_reset(seed=0)
            for _ in range(4):
                await env.async_step(0)

        async_env = make()
        asyncio.run(drive(async_env))
        assert list(async_env.summaries) == list(sync_env.summaries)
        assert len(async_env.summaries) == 2

    def test_attributes(self) -> None:
        env = RecordPerformance(EquityEnv(), periods_per_year=52, risk_free=0.03)
        assert env.periods_per_year == 52.0
        assert env.risk_free == 0.03
        assert env.equity_key == "equity" and env.pnl_key is None
        assert env.summaries.maxlen == 100
        assert env.get_attr("periods_per_year") == 52.0

    def test_periods_per_year_used(self) -> None:
        env = RecordPerformance(EquityEnv(horizon=2, rates=(0.01,)), periods_per_year=2)
        perf = run_episode(env, seed=0).info["performance"]
        assert perf.annualized_return == pytest.approx(perf.total_return)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"periods_per_year": 0},
            {"periods_per_year": math.nan},
            {"deque_size": 0},
            {"deque_size": 2.5},
            {"risk_free": math.inf},
        ],
    )
    def test_invalid_arguments(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            RecordPerformance(EquityEnv(), **kwargs)

    def test_invalid_key_types(self) -> None:
        with pytest.raises(TypeError, match="pnl_key"):
            RecordPerformance(EquityEnv(), pnl_key=1)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            RecordPerformance("not an env")  # type: ignore[arg-type]
