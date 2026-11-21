"""Tests for the multiple-testing statistics in sekai.finance.metrics.

The probabilistic and deflated Sharpe ratios, the minimum track-record length, the expected
maximum Sharpe ratio and the probability of backtest overfitting (CSCV) are checked against
direct, slow implementations of the papers' formulas written here with the standard library
(`statistics.NormalDist` for the normal distribution), against the numerical example of
Bailey and López de Prado (2014), and for their documented properties.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import statistics
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest

import sekai
from sekai.finance import metrics
from sekai.finance.metrics import (
    OverfittingReport,
    deflated_sharpe_ratio,
    expected_maximum_sharpe,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)

NORMAL = statistics.NormalDist()
EULER_GAMMA = 0.5772156649015329
EPS = float(np.finfo(np.float64).eps)


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def textbook_moments(returns: Sequence[float]) -> tuple[float, float, float, float]:
    """Mean, sample standard deviation (ddof=1), skewness and kurtosis, with plain sums."""
    x = [float(v) for v in returns]
    n = len(x)
    mean = sum(x) / n
    m2 = sum((v - mean) ** 2 for v in x) / n
    m3 = sum((v - mean) ** 3 for v in x) / n
    m4 = sum((v - mean) ** 4 for v in x) / n
    return mean, math.sqrt(m2 * n / (n - 1)), m3 / m2**1.5, m4 / m2**2


def textbook_psr(returns: Sequence[float], benchmark: float, ppy: float) -> float:
    """Bailey and López de Prado (2012), eq. for the PSR, with an annualised benchmark."""
    mean, sd, skew, kurt = textbook_moments(returns)
    sr, sr_b, n = mean / sd, benchmark / math.sqrt(ppy), len(returns)
    return NORMAL.cdf(
        (sr - sr_b) * math.sqrt(n - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    )


def textbook_min_trl(
    returns: Sequence[float], benchmark: float, confidence: float, ppy: float
) -> float:
    mean, sd, skew, kurt = textbook_moments(returns)
    sr, sr_b = mean / sd, benchmark / math.sqrt(ppy)
    z = NORMAL.inv_cdf(confidence)
    return 1 + (1 - skew * sr + (kurt - 1) / 4 * sr**2) * (z / (sr - sr_b)) ** 2


def textbook_expected_max(n: int, std: float) -> float:
    """Bailey and López de Prado (2014): the expected maximum of n skill-less trials.

    ``Phi^-1(1 - q)`` is written ``-Phi^-1(q)``: ``1 - 1 / n`` loses the digits of a small
    ``1 / n`` to rounding.
    """
    return std * (
        (1 - EULER_GAMMA) * -NORMAL.inv_cdf(1 / n) + EULER_GAMMA * -NORMAL.inv_cdf(1 / (n * math.e))
    )


def column_sharpe(block: np.ndarray) -> list[float]:
    """Per-period Sharpe ratio of each column, with the statistics module."""
    out = []
    for j in range(block.shape[1]):
        column = [float(v) for v in block[:, j]]
        out.append(statistics.fmean(column) / statistics.stdev(column))
    return out


def textbook_cscv(
    returns: np.ndarray,
    n_partitions: int,
    metric: Callable[[np.ndarray], Sequence[float]] = column_sharpe,
) -> dict[str, Any]:
    """Direct CSCV (Bailey, Borwein, López de Prado and Zhu, 2015) with explicit row lists.

    The oldest `T % n_partitions` rows are dropped. Ties for the in-sample best are
    resolved as a uniformly random choice among them (so every per-combination value is
    an average over the tied strategies), and out-of-sample ties share their mid-rank.
    """
    t, n = returns.shape
    size = t // n_partitions
    first = t - size * n_partitions
    blocks = [range(first + b * size, first + (b + 1) * size) for b in range(n_partitions)]
    logits, overfit, loss, is_perf, oos_perf, ties = [], [], [], [], [], 0
    for chosen in itertools.combinations(range(n_partitions), n_partitions // 2):
        is_rows = [r for b in chosen for r in blocks[b]]
        oos_rows = [r for b in range(n_partitions) if b not in chosen for r in blocks[b]]
        ins = list(metric(returns[is_rows]))
        oos = list(metric(returns[oos_rows]))
        best = max(ins)
        tied = [j for j in range(n) if ins[j] == best]
        ties += len(tied) > 1
        lam, below, lost = [], [], []
        for j in tied:
            rank = sum(v < oos[j] for v in oos) + (sum(v == oos[j] for v in oos) + 1) / 2
            w = rank / (n + 1)
            lam.append(math.log(w / (1 - w)))
            below.append(w <= 0.5)
            lost.append(oos[j] < 0)
        logits.append(sum(lam) / len(tied))
        overfit.append(sum(below) / len(tied))
        loss.append(sum(lost) / len(tied))
        is_perf.append(best)
        oos_perf.append(sum(oos[j] for j in tied) / len(tied))
    return {
        "pbo": sum(overfit) / len(overfit),
        "logits": logits,
        "probability_of_loss": sum(loss) / len(loss),
        "is_performance": is_perf,
        "oos_performance": oos_perf,
        "ties": ties,
    }


def skewed_returns(seed: int, n: int = 500, mean: float = 0.0008) -> np.ndarray:
    """Fat-tailed, negatively skewed daily returns: Student-t noise with rare crashes."""
    rng = np.random.default_rng(seed)
    crashes = rng.random(n) < 0.02
    return mean + 0.01 * rng.standard_t(5, size=n) - 0.03 * crashes


# ---------------------------------------------------------------------------
# Normal distribution
# ---------------------------------------------------------------------------


class TestNormalDistribution:
    # Standard normal quantiles to double precision.
    QUANTILES = [
        (0.5, 0.0),
        (0.75, 0.6744897501960817),
        (0.9, 1.2815515655446004),
        (0.95, 1.6448536269514722),
        (0.975, 1.959963984540054),
        (0.99, 2.3263478740408408),
        (0.995, 2.5758293035489004),
        (0.999, 3.090232306167813),
        (1e-5, -4.264890793922825),
        (1e-10, -6.361340902404056),
        (1e-20, -9.262340089798409),
        (1e-100, -21.273453560965322),
        (1e-300, -37.0470962993612),
    ]

    @pytest.mark.parametrize(("p", "quantile"), QUANTILES)
    def test_known_quantiles(self, p: float, quantile: float) -> None:
        assert math.isclose(metrics._norm_ppf(p), quantile, rel_tol=1e-14, abs_tol=1e-15)
        if quantile > 0.0:  # the upper tail through the symmetric lower one
            assert math.isclose(metrics._norm_ppf(1 - p), -quantile, rel_tol=1e-14)

    def test_matches_statistics_normaldist(self) -> None:
        grid = np.concatenate(
            [
                np.logspace(-300, -1, 400),
                np.linspace(0.001, 0.999, 999),
                1 - np.logspace(-15, -1, 200),
            ]
        )
        for p in grid:
            assert math.isclose(
                metrics._norm_ppf(float(p)), NORMAL.inv_cdf(float(p)), rel_tol=1e-13, abs_tol=1e-15
            ), p

    def test_inverts_cdf(self) -> None:
        # In the lower tail the CDF magnifies relative errors in x by about x**2.
        for p in np.logspace(-300, math.log10(0.5), 300):
            x = metrics._norm_ppf(float(p))
            tolerance = 8 * EPS * max(1.0, x * x)
            assert math.isclose(metrics._norm_cdf(x), p, rel_tol=tolerance), p

    def test_cdf(self) -> None:
        assert metrics._norm_cdf(0.0) == 0.5
        assert math.isclose(metrics._norm_cdf(1.96), 0.9750021048517795, rel_tol=1e-15)
        # Relative accuracy far into the lower tail (references to 20 digits), within the
        # x**2 magnification of the rounding of x / sqrt(2).
        for x, reference in [(-10.0, 7.619853024160526e-24), (-30.0, 4.906713927148187e-198)]:
            assert math.isclose(metrics._norm_cdf(x), reference, rel_tol=4 * EPS * x * x)
        # NormalDist.cdf goes through 1 + erf, so it is only absolutely accurate.
        for x in np.linspace(-8.0, 8.0, 161):
            reference = NORMAL.cdf(float(x))
            assert math.isclose(
                metrics._norm_cdf(float(x)), reference, rel_tol=1e-14, abs_tol=2e-16
            )

    def test_ends(self) -> None:
        assert metrics._norm_ppf(0.0) == -math.inf
        assert metrics._norm_ppf(1.0) == math.inf
        assert metrics._norm_cdf(-math.inf) == 0.0
        assert metrics._norm_cdf(math.inf) == 1.0


# ---------------------------------------------------------------------------
# Probabilistic Sharpe ratio and minimum track-record length
# ---------------------------------------------------------------------------


class TestProbabilisticSharpe:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    @pytest.mark.parametrize(("benchmark", "ppy"), [(0.0, 252), (0.5, 252), (-0.3, 12), (1.0, 52)])
    def test_matches_textbook(self, seed: int, benchmark: float, ppy: float) -> None:
        r = skewed_returns(seed)
        expected = textbook_psr(r, benchmark, ppy)
        assert math.isclose(
            probabilistic_sharpe_ratio(r, benchmark, ppy), expected, rel_tol=1e-10, abs_tol=1e-14
        )

    def test_paper_example(self) -> None:
        # Bailey and López de Prado (2014), numerical example: 100 trials whose annualised
        # Sharpe ratios have variance 1/2, 250 trading days a year, and a selected strategy
        # with an annualised Sharpe ratio of 2.5 over 1250 days, skewness -3, kurtosis 10.
        sr0 = expected_maximum_sharpe(100, math.sqrt(0.5)) / math.sqrt(250)
        assert sr0 == pytest.approx(0.1132, abs=5e-5)
        dsr = metrics._probabilistic_sharpe(2.5 / math.sqrt(250), sr0, 1250, -3.0, 10.0)
        assert dsr == pytest.approx(0.9004, abs=5e-5)

    @pytest.mark.parametrize("ppy", [252, 12, 1])
    def test_half_at_the_observed_sharpe(self, ppy: float) -> None:
        r = skewed_returns(3)
        observed = sharpe_ratio(r, periods_per_year=ppy)
        assert probabilistic_sharpe_ratio(r, observed, ppy) == pytest.approx(0.5, abs=1e-12)

    def test_benchmark_is_annualised(self) -> None:
        r = skewed_returns(4)
        # Zero is zero at every frequency; other benchmarks convert with sqrt(ppy).
        assert probabilistic_sharpe_ratio(r, 0.0, 252) == probabilistic_sharpe_ratio(r, 0.0, 12)
        assert probabilistic_sharpe_ratio(r, 1.0, 252) == pytest.approx(
            probabilistic_sharpe_ratio(r, 1.0 * math.sqrt(12 / 252), 12), rel=1e-12
        )

    def test_monotone(self) -> None:
        r = skewed_returns(5, mean=0.001)
        values = [probabilistic_sharpe_ratio(r, b) for b in (-1.0, 0.0, 0.5, 1.0, 2.0, 4.0)]
        assert all(a > b for a, b in itertools.pairwise(values))
        assert all(0.0 <= v <= 1.0 for v in values)
        # The same per-period statistics over a longer record are more conclusive.
        assert probabilistic_sharpe_ratio(np.tile(r, 2)) > probabilistic_sharpe_ratio(r) > 0.5

    def test_negative_skew_lowers_psr(self) -> None:
        r = skewed_returns(6, mean=0.001)
        mirrored = 2 * r.mean() - r  # same mean and volatility, opposite skewness
        assert textbook_moments(r)[2] < 0 < textbook_moments(mirrored)[2]
        assert sharpe_ratio(r) == pytest.approx(sharpe_ratio(mirrored), rel=1e-9)
        assert probabilistic_sharpe_ratio(r) < probabilistic_sharpe_ratio(mirrored)

    def test_constant_returns(self) -> None:
        # Zero volatility: a Sharpe ratio of 0, as sharpe_ratio reports.
        flat = np.full(100, 0.001)
        assert probabilistic_sharpe_ratio(flat) == 0.5
        assert probabilistic_sharpe_ratio(flat, 1.0) == pytest.approx(
            NORMAL.cdf(-1.0 / math.sqrt(252) * math.sqrt(99)), rel=1e-12
        )

    def test_degenerate_variance(self) -> None:
        # 1 - skew * sr + (kurt - 1) / 4 * sr**2 = 0: the estimate has no spread.
        assert metrics._probabilistic_sharpe(0.5, 0.1, 10, 4.0, 17.0) == 1.0
        assert metrics._probabilistic_sharpe(0.5, 0.9, 10, 4.0, 17.0) == 0.0
        assert metrics._probabilistic_sharpe(0.5, 0.5, 10, 4.0, 17.0) == 0.5


class TestMinimumTrackRecordLength:
    @pytest.mark.parametrize("seed", [0, 1])
    @pytest.mark.parametrize(
        ("benchmark", "confidence", "ppy"), [(0.0, 0.95, 252), (0.5, 0.99, 252), (0.2, 0.9, 12)]
    )
    def test_matches_textbook(
        self, seed: int, benchmark: float, confidence: float, ppy: float
    ) -> None:
        r = skewed_returns(seed, mean=0.002)
        expected = textbook_min_trl(r, benchmark, confidence, ppy)
        assert minimum_track_record_length(r, benchmark, confidence, ppy) == pytest.approx(
            expected, rel=1e-10
        )

    @pytest.mark.parametrize("benchmark", [0.0, 0.5, 1.0])
    def test_inverts_psr(self, benchmark: float) -> None:
        # The record's own length is the MinTRL at the confidence its PSR reaches.
        r = skewed_returns(7, n=750, mean=0.0015)
        psr = probabilistic_sharpe_ratio(r, benchmark)
        assert psr > 0.5
        mintrl = minimum_track_record_length(r, benchmark, confidence=psr)
        assert mintrl == pytest.approx(r.size, rel=1e-7)

    def test_monotone(self) -> None:
        r = skewed_returns(8, mean=0.002)
        by_confidence = [minimum_track_record_length(r, 0.0, c) for c in (0.5, 0.9, 0.95, 0.99)]
        assert by_confidence[0] == 1.0
        assert all(a < b for a, b in itertools.pairwise(by_confidence))
        by_benchmark = [minimum_track_record_length(r, b) for b in (-1.0, 0.0, 0.5, 1.0)]
        assert all(a < b for a, b in itertools.pairwise(by_benchmark))

    def test_unreachable_benchmark(self) -> None:
        r = skewed_returns(9, mean=0.001)
        assert minimum_track_record_length(r, sharpe_ratio(r) + 0.1) == math.inf
        assert minimum_track_record_length(-r) == math.inf
        assert minimum_track_record_length(np.zeros(50)) == math.inf


# ---------------------------------------------------------------------------
# Expected maximum and deflated Sharpe ratio
# ---------------------------------------------------------------------------


def exact_expected_max(n: int) -> float:
    """E[max of n iid standard normals]: n x phi(x) Phi(x)^(n-1) integrated on a fine grid."""
    x = np.linspace(-12.0, 12.0, 48_001)
    cdf = np.array([0.5 * math.erfc(-v / math.sqrt(2.0)) for v in x.tolist()])
    pdf = np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    return float(np.sum(x * n * pdf * cdf ** (n - 1)) * (x[1] - x[0]))


class TestExpectedMaximumSharpe:
    @pytest.mark.parametrize("n", [2, 3, 10, 100, 1000, 10**6, 10**12])
    def test_matches_textbook(self, n: int) -> None:
        assert expected_maximum_sharpe(n, 1.0) == pytest.approx(
            textbook_expected_max(n, 1.0), rel=1e-12
        )

    def test_approximates_the_exact_expectation(self) -> None:
        assert exact_expected_max(2) == pytest.approx(1 / math.sqrt(math.pi), rel=1e-4)
        errors = []
        for n in (10, 100, 1000):
            exact = exact_expected_max(n)
            errors.append(abs(expected_maximum_sharpe(n, 1.0) / exact - 1))
        assert all(e < 0.03 for e in errors)
        assert errors[0] > errors[1] > errors[2]  # an asymptotic approximation

    def test_properties(self) -> None:
        assert expected_maximum_sharpe(1, 2.0) == 0.0
        assert expected_maximum_sharpe(50, 0.0) == 0.0
        values = [expected_maximum_sharpe(n, 1.0) for n in (1, 2, 5, 10, 100, 10**4, 10**8)]
        assert all(a < b for a, b in itertools.pairwise(values))
        assert expected_maximum_sharpe(40, 0.7) == pytest.approx(
            0.7 * expected_maximum_sharpe(40, 1.0)
        )
        assert expected_maximum_sharpe(np.int64(40), 1.0) == expected_maximum_sharpe(40, 1.0)


class TestDeflatedSharpe:
    @pytest.mark.parametrize("ppy", [252, 52])
    def test_matches_textbook(self, ppy: float) -> None:
        r = skewed_returns(10, mean=0.0015)
        trials = np.random.default_rng(11).normal(0.3, 0.6, size=40)
        benchmark = textbook_expected_max(40, statistics.stdev(trials.tolist()))
        expected = textbook_psr(r, benchmark, ppy)
        assert deflated_sharpe_ratio(r, trials, ppy) == pytest.approx(expected, rel=1e-10)
        assert deflated_sharpe_ratio(
            r, periods_per_year=ppy, n_trials=40, trial_sharpe_std=float(np.std(trials, ddof=1))
        ) == pytest.approx(expected, rel=1e-10)

    def test_single_trial_is_psr(self) -> None:
        r = skewed_returns(12)
        psr = probabilistic_sharpe_ratio(r)
        assert deflated_sharpe_ratio(r, [sharpe_ratio(r)]) == psr
        assert deflated_sharpe_ratio(r, n_trials=1, trial_sharpe_std=0.8) == psr
        assert deflated_sharpe_ratio(r, [0.7, 0.7, 0.7]) == psr  # no spread, no luck

    def test_decreases_with_trials(self) -> None:
        r = skewed_returns(13, n=1000, mean=0.0015)
        values = [
            deflated_sharpe_ratio(r, n_trials=n, trial_sharpe_std=0.5)
            for n in (1, 2, 5, 10, 100, 1000, 10**6)
        ]
        assert all(a > b for a, b in itertools.pairwise(values))
        by_spread = [
            deflated_sharpe_ratio(r, n_trials=20, trial_sharpe_std=s) for s in (0.0, 0.2, 0.5, 1.0)
        ]
        assert all(a > b for a, b in itertools.pairwise(by_spread))

    def test_selection_bias(self) -> None:
        # The best of 50 skill-less configurations looks skilful until the trials count.
        trials = np.random.default_rng(1).normal(0.0, 0.01, size=(756, 50))
        sharpes = [sharpe_ratio(trials[:, j]) for j in range(50)]
        best = trials[:, int(np.argmax(sharpes))]
        assert probabilistic_sharpe_ratio(best) > 0.95
        assert deflated_sharpe_ratio(best, sharpes) < 0.5


# ---------------------------------------------------------------------------
# Validation of the Sharpe-ratio statistics
# ---------------------------------------------------------------------------


GOOD = np.random.default_rng(0).normal(0.001, 0.01, size=100)


class TestSharpeValidation:
    @pytest.mark.parametrize(
        "fn",
        [
            probabilistic_sharpe_ratio,
            minimum_track_record_length,
            lambda r: deflated_sharpe_ratio(r, [0.5, 1.0]),
        ],
    )
    @pytest.mark.parametrize(
        ("returns", "match"),
        [
            ([], "must not be empty"),
            ([0.01], "at least two periods"),
            ([[0.01, 0.02], [0.0, 0.01]], "must be 1-D"),
            ([0.01, math.nan, 0.02], "must be finite"),
            ([0.01, math.inf], "must be finite"),
            (["a", "b"], "array of numbers"),
        ],
    )
    def test_bad_returns(self, fn: Callable[[Any], float], returns: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            fn(returns)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"periods_per_year": 0}, "periods_per_year"),
            ({"periods_per_year": -252}, "periods_per_year"),
            ({"benchmark_sharpe": math.nan}, "benchmark_sharpe"),
            ({"benchmark_sharpe": math.inf}, "benchmark_sharpe"),
        ],
    )
    def test_bad_psr_parameters(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            probabilistic_sharpe_ratio(GOOD, **kwargs)
        with pytest.raises(ValueError, match=match):
            minimum_track_record_length(GOOD, **kwargs)

    @pytest.mark.parametrize("confidence", [0.0, 0.49, 1.0, 1.5, math.nan])
    def test_bad_confidence(self, confidence: float) -> None:
        with pytest.raises(ValueError, match=r"confidence must lie in \[0.5, 1\)"):
            minimum_track_record_length(GOOD, confidence=confidence)

    @pytest.mark.parametrize("n_trials", [0, -3, 2.5, 3.0, True, np.bool_(True), "10"])
    def test_bad_n_trials(self, n_trials: Any) -> None:
        with pytest.raises(ValueError, match="n_trials must be an int >= 1"):
            expected_maximum_sharpe(n_trials, 1.0)
        with pytest.raises(ValueError, match="n_trials must be an int >= 1"):
            deflated_sharpe_ratio(GOOD, n_trials=n_trials, trial_sharpe_std=1.0)

    @pytest.mark.parametrize("std", [-0.1, math.nan, math.inf])
    def test_bad_sharpe_std(self, std: float) -> None:
        with pytest.raises(ValueError, match="sharpe_std"):
            expected_maximum_sharpe(10, std)
        with pytest.raises(ValueError, match="trial_sharpe_std"):
            deflated_sharpe_ratio(GOOD, n_trials=10, trial_sharpe_std=std)

    def test_deflated_needs_one_description_of_the_trials(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            deflated_sharpe_ratio(GOOD, [0.5, 1.0], n_trials=2)
        with pytest.raises(ValueError, match="not both"):
            deflated_sharpe_ratio(GOOD, [0.5, 1.0], trial_sharpe_std=0.3)
        for kwargs in ({}, {"n_trials": 10}, {"trial_sharpe_std": 0.5}):
            with pytest.raises(ValueError, match="needs trial_sharpes, or both"):
                deflated_sharpe_ratio(GOOD, **kwargs)

    @pytest.mark.parametrize(
        ("trials", "match"),
        [([], "must not be empty"), ([0.5, math.nan], "must be finite"), ([[0.5]], "must be 1-D")],
    )
    def test_bad_trial_sharpes(self, trials: Any, match: str) -> None:
        with pytest.raises(ValueError, match=f"trial_sharpes {match}"):
            deflated_sharpe_ratio(GOOD, trials)


# ---------------------------------------------------------------------------
# Probability of backtest overfitting
# ---------------------------------------------------------------------------


def noise_panel(seed: int, n_periods: int = 400, n_strategies: int = 10) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 0.01, size=(n_periods, n_strategies))


def assert_matches(report: OverfittingReport, expected: dict[str, Any]) -> None:
    assert report.pbo == pytest.approx(expected["pbo"], abs=1e-12)
    assert report.probability_of_loss == pytest.approx(expected["probability_of_loss"], abs=1e-12)
    np.testing.assert_allclose(report.logits, expected["logits"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(report.is_performance, expected["is_performance"], rtol=1e-10)
    np.testing.assert_allclose(report.oos_performance, expected["oos_performance"], rtol=1e-10)


class TestProbabilityOfBacktestOverfitting:
    @pytest.mark.parametrize(
        ("n_periods", "n_strategies", "n_partitions"),
        [(63, 5, 6), (200, 7, 8), (41, 3, 2), (100, 4, 10)],
    )
    def test_matches_textbook(self, n_periods: int, n_strategies: int, n_partitions: int) -> None:
        rng = np.random.default_rng(n_periods)
        # Strategies with different edges, so the selection is not pure noise.
        x = rng.normal(
            np.linspace(-0.001, 0.002, n_strategies), 0.01, size=(n_periods, n_strategies)
        )
        report = probability_of_backtest_overfitting(x, n_partitions)
        assert_matches(report, textbook_cscv(x, n_partitions))
        assert report.n_combinations == math.comb(n_partitions, n_partitions // 2)
        assert report.n_partitions == n_partitions
        assert report.block_size == n_periods // n_partitions
        slope, intercept = np.polyfit(report.is_performance, report.oos_performance, 1)
        assert report.degradation_slope == pytest.approx(slope, rel=1e-8)
        assert report.degradation_intercept == pytest.approx(intercept, rel=1e-8, abs=1e-12)

    def test_custom_metric_matches_textbook(self) -> None:
        x = noise_panel(1, n_periods=90, n_strategies=6)

        def mean_return(block: np.ndarray) -> np.ndarray:
            return block.mean(axis=0)

        expected = textbook_cscv(
            x, 6, lambda block: [statistics.fmean(c) for c in block.T.tolist()]
        )
        assert_matches(probability_of_backtest_overfitting(x, 6, mean_return), expected)

    def test_default_metric_is_the_per_period_sharpe(self) -> None:
        x = noise_panel(2, n_periods=301, n_strategies=8)
        x[:, 5] = 0.0  # a strategy that never trades: Sharpe ratio 0

        def sharpe(block: np.ndarray) -> np.ndarray:
            return np.array([sharpe_ratio(c, periods_per_year=1) for c in block.T])

        fast = probability_of_backtest_overfitting(x, 8)
        slow = probability_of_backtest_overfitting(x, 8, sharpe)
        assert fast.pbo == slow.pbo
        np.testing.assert_allclose(fast.logits, slow.logits, rtol=1e-12)
        np.testing.assert_allclose(fast.is_performance, slow.is_performance, rtol=1e-10)
        np.testing.assert_allclose(
            fast.oos_performance, slow.oos_performance, rtol=1e-10, atol=1e-15
        )

    def test_dominant_strategy(self) -> None:
        # Strategy 3 beats every other strategy in every period.
        x = noise_panel(3, n_periods=600)
        x[:, 3] = np.delete(x, 3, axis=1).max(axis=1) + 0.002
        report = probability_of_backtest_overfitting(x)
        assert report.pbo == 0.0
        assert report.probability_of_loss == 0.0
        np.testing.assert_allclose(report.logits, math.log(10))  # always ranked best of 10

    def test_noise_is_a_coin_flip(self) -> None:
        # For skill-less strategies the in-sample best is as likely to land below the
        # out-of-sample median as above it. One panel's PBO varies a lot (the splits share
        # data), so average over panels.
        pbos = [probability_of_backtest_overfitting(noise_panel(seed), 8).pbo for seed in range(20)]
        assert 0.4 < statistics.fmean(pbos) < 0.6

    @pytest.mark.parametrize("use_metric", [False, True])
    def test_invariant_to_strategy_order(self, use_metric: bool) -> None:
        x = noise_panel(4, n_periods=250, n_strategies=9)
        x[:, 7] = x[:, 2]  # duplicates tie in and out of sample
        x[:, 8] = x[:, 2]
        metric = (lambda b: (b > 0).sum(axis=0).astype(float)) if use_metric else None
        report = probability_of_backtest_overfitting(x, 8, metric)
        for seed in range(3):
            perm = np.random.default_rng(seed).permutation(9)
            shuffled = probability_of_backtest_overfitting(x[:, perm], 8, metric)
            assert shuffled.pbo == report.pbo
            assert shuffled.probability_of_loss == report.probability_of_loss
            np.testing.assert_array_equal(shuffled.logits, report.logits)
            np.testing.assert_array_equal(shuffled.oos_performance, report.oos_performance)
            assert shuffled.degradation_slope == report.degradation_slope

    def test_ties_follow_random_selection(self) -> None:
        # A count metric ties often, in and out of sample.
        x = noise_panel(5, n_periods=96, n_strategies=6)

        def positive_periods(block: np.ndarray) -> np.ndarray:
            return (block > 0).sum(axis=0).astype(float)

        expected = textbook_cscv(x, 6, lambda b: positive_periods(b).tolist())
        assert expected["ties"] > 0
        report = probability_of_backtest_overfitting(x, 6, positive_periods)
        assert_matches(report, expected)

    def test_identical_strategies(self) -> None:
        # Selection among identical strategies is always at the median: overfit by definition.
        x = np.repeat(noise_panel(6, n_strategies=1), 4, axis=1)
        report = probability_of_backtest_overfitting(x, 4)
        assert report.pbo == 1.0
        np.testing.assert_array_equal(report.logits, 0.0)

    def test_remainder_periods_are_dropped_from_the_start(self) -> None:
        x = noise_panel(7, n_periods=103, n_strategies=5)
        seen: set[float] = set()

        def recording(block: np.ndarray) -> np.ndarray:
            assert block.shape == (4 * 12, 5)  # half of 8 blocks of 103 // 8 = 12 periods
            assert np.all(np.diff(block[:, 0]) > 0)  # rows arrive in time order
            seen.update(block[:, 0].tolist())
            return block.mean(axis=0)

        stamped = x.copy()
        stamped[:, 0] = np.arange(103)  # column 0 carries the period index
        report = probability_of_backtest_overfitting(stamped, 8, recording)
        assert report.block_size == 12
        assert seen == set(range(7, 103))  # the 103 - 96 = 7 oldest periods are left out
        trimmed = probability_of_backtest_overfitting(x[7:], 8)
        untrimmed = probability_of_backtest_overfitting(x, 8)
        np.testing.assert_array_equal(untrimmed.logits, trimmed.logits)

    def test_sixteen_partitions(self) -> None:
        report = probability_of_backtest_overfitting(noise_panel(8, n_periods=64, n_strategies=3))
        assert report.n_combinations == 12_870
        assert report.logits.shape == report.is_performance.shape == report.oos_performance.shape
        assert report.block_size == 4

    def test_constant_in_sample_performance(self) -> None:
        def ascending(block: np.ndarray) -> np.ndarray:
            return np.arange(block.shape[1], dtype=float)

        report = probability_of_backtest_overfitting(noise_panel(9, n_strategies=4), 4, ascending)
        assert report.pbo == 0.0
        np.testing.assert_allclose(report.logits, math.log(4))
        assert math.isnan(report.degradation_slope)
        assert math.isnan(report.degradation_intercept)

    def test_report(self) -> None:
        report = probability_of_backtest_overfitting(noise_panel(10), 8)
        assert isinstance(report, OverfittingReport)
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.pbo = 0.0  # type: ignore[misc]
        lines = str(report).splitlines()
        assert lines[0].split() == ["combinations", "70"]
        assert lines[1].split() == ["probability", "of", "overfitting", f"{report.pbo:.2%}"]
        assert 0.0 <= report.pbo <= 1.0
        assert 0.0 <= report.probability_of_loss <= 1.0

    def test_exported(self) -> None:
        for name in (
            "probabilistic_sharpe_ratio",
            "minimum_track_record_length",
            "deflated_sharpe_ratio",
            "probability_of_backtest_overfitting",
            "OverfittingReport",
        ):
            assert getattr(sekai.finance, name) is getattr(metrics, name)
            assert name in sekai.finance.__all__
        assert "expected_maximum_sharpe" in metrics.__all__


class TestOverfittingValidation:
    @pytest.mark.parametrize(
        ("returns", "match"),
        [
            (np.zeros(50), "must be 2-D"),
            (np.zeros((50, 1)), "at least two strategies"),
            (np.zeros((50, 0)), "at least two strategies"),
            (np.full((50, 3), math.nan), "must be finite"),
            ([["a", "b"], ["c", "d"]], "array of numbers"),
            (np.zeros((10, 3)), "returns has 10 periods, fewer than n_partitions=16"),
        ],
    )
    def test_bad_returns(self, returns: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            probability_of_backtest_overfitting(returns)

    @pytest.mark.parametrize(
        ("n_partitions", "match"),
        [
            (0, "n_partitions must be an int >= 2"),
            (1, "n_partitions must be an int >= 2"),
            (4.0, "n_partitions must be an int >= 2"),
            (True, "n_partitions must be an int >= 2"),
            (3, "n_partitions must be even"),
            (15, "n_partitions must be even"),
            (22, "n_partitions=22 gives 705,432 combinations; at most 20"),
            (100, "at most 20 partitions"),
        ],
    )
    def test_bad_partitions(self, n_partitions: Any, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            probability_of_backtest_overfitting(noise_panel(0), n_partitions)

    def test_twenty_partitions_are_allowed(self) -> None:
        report = probability_of_backtest_overfitting(
            noise_panel(0, n_periods=40, n_strategies=2), 20
        )
        assert report.n_combinations == 184_756

    @pytest.mark.parametrize(
        ("metric", "match"),
        [
            (lambda b: b.mean(), r"shape \(10,\), got shape \(\)"),
            (lambda b: b.mean(axis=0)[:5], r"shape \(10,\), got shape \(5,\)"),
            (lambda b: np.full(10, math.nan), "must return finite values"),
            (lambda b: np.full(10, math.inf), "must return finite values"),
            (lambda b: ["x"] * 10, "must return an array of numbers"),
        ],
    )
    def test_bad_metric(self, metric: Callable[[np.ndarray], Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            probability_of_backtest_overfitting(noise_panel(0), 4, metric)

    def test_metric_must_be_callable(self) -> None:
        with pytest.raises(TypeError, match="metric must be callable or None"):
            probability_of_backtest_overfitting(noise_panel(0), 4, "sharpe")  # type: ignore[arg-type]
