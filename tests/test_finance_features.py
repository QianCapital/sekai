from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from sekai.error import LookAheadError
from sekai.finance import features
from sekai.finance.features import (
    asof_join,
    check_causal,
    cross_sectional_rank,
    cross_sectional_zscore,
    ewm_mean,
    ewm_std,
    forward_fill,
    lag,
    log_returns,
    momentum,
    realized_volatility,
    rolling_mean,
    rolling_std,
    rolling_zscore,
    simple_returns,
)

T, N = 80, 5


def _prices(seed: int = 0, n_rows: int = T, n_cols: int = N) -> np.ndarray:
    """Positive random-walk prices with scattered NaNs, a late listing and a delisting."""
    rng = np.random.default_rng(seed)
    p = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, (n_rows, n_cols)), axis=0))
    p[rng.random((n_rows, n_cols)) < 0.15] = np.nan
    p[:9, 1] = np.nan
    p[n_rows // 2 :, 2] = np.nan
    return p


def _values(seed: int = 1, n_rows: int = T, n_cols: int = N) -> np.ndarray:
    """Signed random data with scattered NaNs and an all-NaN stretch."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (n_rows, n_cols))
    x[rng.random((n_rows, n_cols)) < 0.2] = np.nan
    x[10:25, 3] = np.nan
    return x


def _close(a: np.ndarray, b: np.ndarray, rtol: float = 1e-12, atol: float = 1e-12) -> None:
    assert a.shape == b.shape
    assert a.dtype == np.float64
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    np.testing.assert_array_equal(np.isnan(a), np.isnan(b))


# ---------------------------------------------------------------------------
# Straightforward loop references
# ---------------------------------------------------------------------------


def _cols(x: np.ndarray) -> np.ndarray:
    return x.reshape(len(x), -1)


def ref_ratio(p: np.ndarray, k: int) -> np.ndarray:
    p2 = _cols(p)
    out = np.full(p2.shape, np.nan)
    for t in range(k, len(p2)):
        for j in range(p2.shape[1]):
            a, b = p2[t - k, j], p2[t, j]
            if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
                out[t, j] = b / a
    return out.reshape(p.shape)


def ref_ffill(x: np.ndarray, limit: int | None) -> np.ndarray:
    x2 = _cols(x)
    out = np.full(x2.shape, np.nan)
    for j in range(x2.shape[1]):
        last, gap = np.nan, 0
        for t in range(len(x2)):
            if not np.isnan(x2[t, j]):
                last, gap = x2[t, j], 0
                out[t, j] = last
            else:
                gap += 1
                if limit is None or gap <= limit:
                    out[t, j] = last
    return out.reshape(x.shape)


def ref_rolling(x: np.ndarray, w: int, mp: int, kind: str, ddof: int = 1) -> np.ndarray:
    x2 = _cols(x)
    out = np.full(x2.shape, np.nan)
    for t in range(len(x2)):
        for j in range(x2.shape[1]):
            vals = x2[max(0, t - w + 1) : t + 1, j]
            vals = vals[~np.isnan(vals)]
            if kind == "mean" and len(vals) >= mp:
                out[t, j] = vals.mean()
            elif kind == "std" and len(vals) >= max(mp, ddof + 1):
                out[t, j] = vals.std(ddof=ddof)
            elif kind == "z" and len(vals) >= max(mp, 2) and not np.isnan(x2[t, j]):
                s = vals.std(ddof=1)
                out[t, j] = 0.0 if s == 0 else (x2[t, j] - vals.mean()) / s
    return out.reshape(x.shape)


def ref_ewm(x: np.ndarray, halflife: float, mp: int, kind: str) -> np.ndarray:
    a = 1.0 - np.exp(np.log(0.5) / halflife)
    x2 = _cols(x)
    out = np.full(x2.shape, np.nan)
    for j in range(x2.shape[1]):
        m = v = 0.0
        seen = 0
        for t in range(len(x2)):
            xt = x2[t, j]
            if np.isnan(xt):
                continue
            if seen == 0:
                m, v = xt, 0.0
            else:
                d = xt - m
                if kind == "mean":
                    m = (1 - a) * m + a * xt
                else:
                    m = m + a * d
                    v = (1 - a) * (v + a * d * d)
            seen += 1
            if seen >= mp:
                out[t, j] = m if kind == "mean" else np.sqrt(v)
    return out.reshape(x.shape)


def ref_cs_zscore(x: np.ndarray) -> np.ndarray:
    out = np.full(x.shape, np.nan)
    for t in range(len(x)):
        ok = ~np.isnan(x[t])
        vals = x[t, ok]
        if len(vals) < 2:
            continue
        s = vals.std()
        out[t, ok] = 0.0 if s == 0 else (vals - vals.mean()) / s
    return out


def ref_cs_rank(x: np.ndarray) -> np.ndarray:
    out = np.full(x.shape, np.nan)
    for t in range(len(x)):
        vals = x[t, ~np.isnan(x[t])]
        for j in range(x.shape[1]):
            if np.isnan(x[t, j]):
                continue
            if len(vals) == 1:
                out[t, j] = 0.5
                continue
            less = np.sum(vals < x[t, j])
            equal = np.sum(vals == x[t, j])
            rank = less + (equal + 1) / 2.0
            out[t, j] = (rank - 1) / (len(vals) - 1)
    return out


def ref_asof(targets: Any, events: Any, values: np.ndarray, lag_: Any) -> np.ndarray:
    out = np.full((len(targets), *values.shape[1:]), np.nan)
    for i, t in enumerate(targets):
        for e, et in enumerate(events):
            eff = et if lag_ is None else et + lag_
            if eff <= t:
                out[i] = values[e]
    return out


# ---------------------------------------------------------------------------
# Returns and lags
# ---------------------------------------------------------------------------


class TestReturns:
    def test_log_and_simple_returns_match_reference(self) -> None:
        p = _prices()
        p[30, 0], p[31, 4] = 0.0, -5.0  # non-positive prices are invalid
        ratio = ref_ratio(p, 1)
        _close(log_returns(p), np.log(ratio))
        _close(simple_returns(p), ratio - 1.0)
        assert np.all(np.isnan(log_returns(p)[0]))
        assert np.isnan(log_returns(p)[[30, 31], 0]).all()
        _close(log_returns(p[:, 0]), np.log(ratio[:, 0]))

    def test_momentum(self) -> None:
        p = _prices()
        for k in (1, 3, 20):
            _close(momentum(p, k), np.log(ref_ratio(p, k)))
        _close(momentum(p, 1), log_returns(p))
        assert np.isnan(momentum(p, T)).all() and momentum(p, T).shape == p.shape

    def test_accepts_lists_and_other_dtypes(self) -> None:
        out = log_returns([1, 2, 4])
        assert out.dtype == np.float64
        np.testing.assert_allclose(out, [np.nan, np.log(2), np.log(2)])
        out32 = simple_returns(np.array([1.0, 1.5], dtype=np.float32))
        assert out32.dtype == np.float64

    def test_lag(self) -> None:
        x = _values()
        _close(lag(x, 0), x)
        assert lag(x, 0) is not x
        expected = np.full_like(x, np.nan)
        expected[3:] = x[:-3]
        _close(lag(x, 3), expected)
        _close(lag(x[:, 1]), np.concatenate([[np.nan], x[:-1, 1]]))
        assert np.isnan(lag(x, T)).all() and np.isnan(lag(x, T + 5)).all()

    @pytest.mark.parametrize("limit", [None, 0, 1, 2, 5])
    def test_forward_fill_matches_reference(self, limit: int | None) -> None:
        x = _values()
        x[:4, 0] = np.nan  # leading NaNs must stay NaN
        before = x.copy()
        _close(forward_fill(x, limit=limit), ref_ffill(x, limit))
        _close(forward_fill(x[:, 3], limit=limit), ref_ffill(x[:, 3], limit))
        np.testing.assert_array_equal(x, before)  # input untouched

    def test_forward_fill_example(self) -> None:
        x = np.array([np.nan, 1.0, np.nan, np.nan, np.nan, 2.0, np.nan])
        np.testing.assert_array_equal(forward_fill(x, limit=2), [np.nan, 1, 1, 1, np.nan, 2, 2])
        np.testing.assert_array_equal(forward_fill(x), [np.nan, 1, 1, 1, 1, 2, 2])


# ---------------------------------------------------------------------------
# Rolling and exponentially weighted statistics
# ---------------------------------------------------------------------------

ROLLING_PARAMS = [(1, None), (3, None), (5, 2), (10, 1), (10, 10), (T + 10, 3)]


class TestRolling:
    @pytest.mark.parametrize(("window", "min_periods"), ROLLING_PARAMS)
    def test_mean_matches_reference(self, window: int, min_periods: int | None) -> None:
        x = _values()
        mp = window if min_periods is None else min_periods
        _close(rolling_mean(x, window, min_periods=min_periods), ref_rolling(x, window, mp, "mean"))
        _close(
            rolling_mean(x[:, 0], window, min_periods=min_periods),
            ref_rolling(x[:, 0], window, mp, "mean"),
        )

    @pytest.mark.parametrize(("window", "min_periods"), ROLLING_PARAMS)
    @pytest.mark.parametrize("ddof", [0, 1, 2])
    def test_std_matches_reference(self, window: int, min_periods: int | None, ddof: int) -> None:
        x = _values()
        mp = window if min_periods is None else min_periods
        _close(
            rolling_std(x, window, min_periods=min_periods, ddof=ddof),
            ref_rolling(x, window, mp, "std", ddof),
        )

    @pytest.mark.parametrize(("window", "min_periods"), ROLLING_PARAMS)
    def test_zscore_matches_reference(self, window: int, min_periods: int | None) -> None:
        x = _values()
        mp = window if min_periods is None else min_periods
        _close(
            rolling_zscore(x, window, min_periods=min_periods),
            ref_rolling(x, window, mp, "z"),
            rtol=1e-10,
            atol=1e-10,
        )

    def test_chunked_evaluation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        x = _values()
        expected = [rolling_mean(x, 6, min_periods=2), rolling_std(x, 6), rolling_zscore(x, 6)]
        monkeypatch.setattr(features, "_CHUNK_ELEMENTS", 37)  # forces many small row chunks
        got = [rolling_mean(x, 6, min_periods=2), rolling_std(x, 6), rolling_zscore(x, 6)]
        for a, b in zip(got, expected):
            np.testing.assert_array_equal(a, b)
        _close(got[1], ref_rolling(x, 6, 6, "std"))

    def test_constant_window_gives_exact_zero(self) -> None:
        x = np.array([0.1, 0.1, 0.1, np.nan, 0.1, 0.3, 0.3, 0.3])
        std = rolling_std(x, 3, min_periods=2)
        z = rolling_zscore(x, 3, min_periods=2)
        assert std[2] == 0.0 and std[4] == 0.0 and std[7] == 0.0
        assert z[2] == 0.0 and z[4] == 0.0 and z[7] == 0.0
        assert np.isnan(z[3])  # x_t missing
        assert np.isnan(z[0])  # a single value has no std

    def test_realized_volatility(self) -> None:
        r = log_returns(_prices())
        _close(realized_volatility(r, 10), rolling_std(r, 10))
        _close(realized_volatility(r, 10, periods_per_year=252), rolling_std(r, 10) * np.sqrt(252))

    def test_pandas_parity(self) -> None:
        pd = pytest.importorskip("pandas")
        x = _values(seed=7, n_rows=200)
        df = pd.DataFrame(x)
        for window, min_periods in [(1, None), (5, None), (5, 1), (20, 7), (250, 3)]:
            roll = df.rolling(window, min_periods=min_periods)
            _close(
                rolling_mean(x, window, min_periods=min_periods),
                roll.mean().to_numpy(),
                rtol=1e-10,
                atol=1e-12,
            )
            for ddof in (0, 1):
                _close(
                    rolling_std(x, window, min_periods=min_periods, ddof=ddof),
                    roll.std(ddof=ddof).to_numpy(),
                    rtol=1e-10,
                    atol=1e-12,
                )
        series = pd.Series(x[:, 0])
        _close(rolling_mean(x[:, 0], 4), series.rolling(4).mean().to_numpy(), atol=1e-12)

    def test_accurate_after_level_shift(self) -> None:
        # Windows are evaluated independently (two-pass), so a past spike leaves no
        # residue; pandas' online update drifts by ~1e-3 on this data.
        x = _values(seed=7, n_rows=120)
        x[50:60] += 1e6
        for window, min_periods in [(5, None), (20, 7)]:
            mp = window if min_periods is None else min_periods
            _close(
                rolling_std(x, window, min_periods=min_periods),
                ref_rolling(x, window, mp, "std"),
                rtol=1e-9,
                atol=1e-9,
            )
            _close(
                rolling_zscore(x, window, min_periods=min_periods),
                ref_rolling(x, window, mp, "z"),
                rtol=1e-6,
                atol=1e-6,
            )


class TestEwm:
    @pytest.mark.parametrize("halflife", [0.5, 1.0, 3.7, 20.0])
    @pytest.mark.parametrize("min_periods", [1, 2, 5])
    def test_matches_reference(self, halflife: float, min_periods: int) -> None:
        x = _values()
        _close(
            ewm_mean(x, halflife, min_periods=min_periods),
            ref_ewm(x, halflife, min_periods, "mean"),
        )
        _close(
            ewm_std(x, halflife, min_periods=min_periods), ref_ewm(x, halflife, min_periods, "std")
        )
        _close(ewm_mean(x[:, 3], halflife), ref_ewm(x[:, 3], halflife, 1, "mean"))

    def test_nan_leaves_state_unchanged(self) -> None:
        out = ewm_mean([1.0, np.nan, 3.0], 1.0)  # a = 0.5
        np.testing.assert_allclose(out, [1.0, np.nan, 2.0])
        np.testing.assert_allclose(ewm_std([1.0, np.nan, 3.0], 1.0), [np.nan, np.nan, 1.0])
        np.testing.assert_allclose(ewm_std([1.0, 3.0], 1.0, min_periods=1), [0.0, 1.0])
        np.testing.assert_allclose(
            ewm_mean([np.nan, 4.0, 2.0], 1.0, min_periods=2), [np.nan] * 2 + [3]
        )

    def test_pandas_parity(self) -> None:
        pd = pytest.importorskip("pandas")
        rng = np.random.default_rng(3)
        x = rng.normal(0.0, 1.0, (150, 4)).cumsum(axis=0)
        for halflife in (0.5, 2.0, 10.0):
            for min_periods in (1, 4):
                expected = (
                    pd.DataFrame(x)
                    .ewm(halflife=halflife, adjust=False, min_periods=min_periods)
                    .mean()
                    .to_numpy()
                )
                _close(ewm_mean(x, halflife, min_periods=min_periods), expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Cross-sectional transforms
# ---------------------------------------------------------------------------


class TestCrossSectional:
    def test_zscore_matches_reference(self) -> None:
        x = _values(n_cols=7)
        x[5, :] = np.nan  # empty row
        x[6, 1:] = np.nan  # single valid entry
        x[7, :] = 0.3  # constant row
        x[8, [0, 1]] = np.nan
        x[8, 2:] = -1.25  # constant valid entries
        out = cross_sectional_zscore(x)
        _close(out, ref_cs_zscore(x), rtol=1e-10, atol=1e-12)
        assert np.isnan(out[5]).all() and np.isnan(out[6]).all()
        assert np.all(out[7] == 0.0) and np.all(out[8, 2:] == 0.0)
        valid_rows = np.sum(~np.isnan(x), axis=1) >= 2
        np.testing.assert_allclose(np.nanmean(out[valid_rows], axis=1), 0.0, atol=1e-12)

    def test_rank_matches_reference(self) -> None:
        rng = np.random.default_rng(5)
        x = rng.integers(0, 4, (60, 8)).astype(np.float64)  # plenty of ties
        x[rng.random(x.shape) < 0.25] = np.nan
        x[0, :] = np.nan
        x[1, 1:] = np.nan
        x[1, 0] = 2.0
        x[2] = [3, 1, 2, np.nan, 2, 2, 5, -1]
        out = cross_sectional_rank(x)
        _close(out, ref_cs_rank(x))
        assert np.isnan(out[0]).all() and out[1, 0] == 0.5
        np.testing.assert_allclose(out[2], [5 / 6, 1 / 6, 0.5, np.nan, 0.5, 0.5, 1.0, 0.0])
        assert np.nanmin(out) >= 0.0 and np.nanmax(out) <= 1.0

    def test_rank_continuous(self) -> None:
        x = _values(n_cols=9)
        _close(cross_sectional_rank(x), ref_cs_rank(x))

    @pytest.mark.parametrize("fn", [cross_sectional_zscore, cross_sectional_rank])
    def test_requires_2d(self, fn: Callable[[Any], np.ndarray]) -> None:
        with pytest.raises(ValueError, match="2-D"):
            fn(np.arange(5.0))
        assert fn(np.empty((3, 0))).shape == (3, 0)
        assert fn(np.empty((0, 4))).shape == (0, 4)


# ---------------------------------------------------------------------------
# asof_join
# ---------------------------------------------------------------------------


class TestAsofJoin:
    def test_datetime_with_lag(self) -> None:
        bars = np.arange("2024-01-01", "2024-01-11", dtype="datetime64[D]")
        releases = np.array(
            ["2024-01-02T12:00", "2024-01-04T08:30", "2024-01-07T16:00"], dtype="datetime64[m]"
        )
        vals = np.array([1.0, 2.0, 3.0])
        out = asof_join(bars, releases, vals, lag=np.timedelta64(12, "h"))
        # Known at 01-03 00:00, 01-04 20:30 and 01-08 04:00 respectively.
        expected = [np.nan, np.nan, 1, 1, 2, 2, 2, 2, 3, 3]
        np.testing.assert_array_equal(out, expected)
        no_lag = asof_join(bars, releases, vals)
        np.testing.assert_array_equal(no_lag, [np.nan, np.nan, 1, 1, 2, 2, 2, 3, 3, 3])
        py_lag = asof_join(bars, releases, vals, lag=dt.timedelta(hours=12))
        np.testing.assert_array_equal(py_lag, out)
        exact = asof_join(bars, bars[[2, 5]], [7.0, 8.0], lag=np.timedelta64(1, "D"))
        np.testing.assert_array_equal(exact, [np.nan] * 3 + [7.0] * 3 + [8.0] * 4)

    def test_numeric_duplicates_and_bounds(self) -> None:
        targets = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        events = np.array([-3.0, 1.0, 1.0, 1.0, 4.5, 99.0])
        vals = np.array([10.0, 20.0, 21.0, 22.0, 30.0, 40.0])
        np.testing.assert_array_equal(asof_join(targets, events, vals), [10, 22, 22, 22, 22, 30])
        np.testing.assert_array_equal(
            asof_join(targets, events, vals, lag=1), [10, 10, 22, 22, 22, 22]
        )
        after_all = asof_join(targets, np.array([10, 11]), [1.0, 2.0])
        assert after_all.shape == (6,) and np.isnan(after_all).all()
        before_all = asof_join([5, 6], [1, 2, 3], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(before_all, [3.0, 3.0])
        empty = asof_join(targets, np.array([], dtype=np.float64), np.empty((0, 2)))
        assert empty.shape == (6, 2) and np.isnan(empty).all()

    def test_2d_values(self) -> None:
        out = asof_join([0, 1, 2, 3], [1, 2], [[1.0, -1.0], [2.0, -2.0]])
        assert out.shape == (4, 2) and out.dtype == np.float64
        np.testing.assert_array_equal(out, [[np.nan, np.nan], [1, -1], [2, -2], [2, -2]])

    @pytest.mark.parametrize("lag_", [None, 0.0, 2, 3.5])
    def test_matches_reference(self, lag_: float | None) -> None:
        rng = np.random.default_rng(11)
        targets = np.sort(rng.uniform(0, 50, 40))
        events = np.sort(np.round(rng.uniform(-5, 55, 30)))  # rounding creates duplicates
        vals = rng.normal(size=(30, 3))
        _close(asof_join(targets, events, vals, lag=lag_), ref_asof(targets, events, vals, lag_))

    def test_is_point_in_time(self) -> None:
        times = np.arange(30.0)
        x = _values(n_rows=30)
        for lag_ in (None, 0, 2.0, 7):
            check_causal(lambda v, lag_=lag_: asof_join(times, times, v, lag=lag_), x)

    def test_validation(self) -> None:
        days = np.arange("2024-01-01", "2024-01-05", dtype="datetime64[D]")
        with pytest.raises(ValueError, match="event_times must be sorted"):
            asof_join([0, 1], [2, 1], [1.0, 2.0])
        with pytest.raises(ValueError, match="target_times must be sorted"):
            asof_join([1, 0], [0, 1], [1.0, 2.0])
        with pytest.raises(ValueError, match="values must be shaped"):
            asof_join([0, 1], [0, 1], [1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="1-D"):
            asof_join([[0, 1]], [0, 1], [1.0, 2.0])
        with pytest.raises(TypeError, match=r"must both be np\.datetime64 or both numeric"):
            asof_join(days, [0, 1], [1.0, 2.0])
        with pytest.raises(TypeError, match=r"np\.timedelta64"):
            asof_join(days, days, np.ones(4), lag=1.0)
        with pytest.raises(TypeError, match="number"):
            asof_join([0, 1], [0, 1], [1.0, 2.0], lag=np.timedelta64(1, "D"))
        with pytest.raises(TypeError, match="number"):
            asof_join([0, 1], [0, 1], [1.0, 2.0], lag=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="non-negative"):
            asof_join([0, 1], [0, 1], [1.0, 2.0], lag=-1)
        with pytest.raises(ValueError, match="non-negative"):
            asof_join(days, days, np.ones(4), lag=np.timedelta64(-1, "h"))
        with pytest.raises(ValueError, match="NaN"):
            asof_join([0, 1], [0.0, np.nan], [1.0, 2.0])
        with pytest.raises(ValueError, match="NaT"):
            asof_join(days, np.array(["NaT", "2024-01-01"], dtype="datetime64[D]"), [1.0, 2.0])
        with pytest.raises(TypeError, match="numeric"):
            asof_join(np.array(["a", "b"]), np.array(["a", "b"]), [1.0, 2.0])


# ---------------------------------------------------------------------------
# check_causal
# ---------------------------------------------------------------------------

CAUSAL: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "log_returns": log_returns,
    "simple_returns": simple_returns,
    "momentum": lambda x: momentum(x, 5),
    "lag": lambda x: lag(x, 2),
    "forward_fill": forward_fill,
    "forward_fill_limit": lambda x: forward_fill(x, limit=2),
    "rolling_mean": lambda x: rolling_mean(x, 7, min_periods=3),
    "rolling_std": lambda x: rolling_std(x, 7),
    "rolling_zscore": lambda x: rolling_zscore(x, 10, min_periods=4),
    "ewm_mean": lambda x: ewm_mean(x, 3.0),
    "ewm_std": lambda x: ewm_std(x, 3.0),
    "realized_volatility": lambda x: realized_volatility(log_returns(x), 5, periods_per_year=252),
    "cross_sectional_zscore": cross_sectional_zscore,
    "cross_sectional_rank": cross_sectional_rank,
    "zscore_of_momentum": lambda x: cross_sectional_zscore(momentum(forward_fill(x), 3)),
}


def _centred_mean(x: np.ndarray) -> np.ndarray:
    """5-row centred rolling mean: the trailing mean moved two rows earlier."""
    out = np.roll(rolling_mean(x, 5, min_periods=1), -2, axis=0)
    out[-2:] = np.nan
    return out


NON_CAUSAL: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "centred_rolling_mean": _centred_mean,
    "roll_back": lambda x: np.roll(x, -1, axis=0),
    "full_sample_zscore": lambda x: (x - np.nanmean(x, axis=0)) / np.nanstd(x, axis=0),
    "backward_fill": lambda x: forward_fill(x[::-1])[::-1],
}


class TestCheckCausal:
    @pytest.mark.parametrize("name", list(CAUSAL))
    def test_causal_functions_pass(self, name: str) -> None:
        p = _prices()
        check_causal(CAUSAL[name], p, n_checks=8)
        check_causal(CAUSAL[name], p, n_checks=3, rng=np.random.default_rng(42))

    @pytest.mark.parametrize(
        "name", [n for n in CAUSAL if not n.startswith("cross_sectional") and "zscore_of" not in n]
    )
    def test_causal_functions_pass_1d(self, name: str) -> None:
        check_causal(CAUSAL[name], _prices()[:, 0])

    @pytest.mark.parametrize("name", list(NON_CAUSAL))
    def test_non_causal_functions_fail(self, name: str) -> None:
        with pytest.raises(LookAheadError, match="Look-ahead"):
            check_causal(NON_CAUSAL[name], _prices())

    def test_leaks_through_signs_zeros_and_nans_are_detected(self) -> None:
        # Regression: only rescaling the future kept signs, zeros and the NaN pattern, so
        # these one-step look-aheads (and survivorship bias) passed the check.
        rng = np.random.default_rng(0)
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (200, 5)), axis=0))
        prices[150:, 2] = np.nan  # asset 2 delists at row 150
        returns = log_returns(prices)
        volume = np.where(rng.random((200, 5)) < 0.3, 0.0, rng.random((200, 5)))

        def next_row(v: np.ndarray) -> np.ndarray:
            return np.vstack([v[1:], np.full((1, v.shape[1]), np.nan)])

        leaks: list[tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]] = [
            (lambda v: np.sign(next_row(v)), returns),
            (lambda v: (next_row(v) > 0).astype(np.float64), returns),
            (lambda v: (next_row(v) == 0).astype(np.float64), volume),
            (lambda v: np.where(np.isnan(v).any(axis=0), np.nan, v), prices),  # survivorship
            (lambda v: np.where(np.isnan(next_row(v)), 0.0, v), prices),  # "delists tomorrow"
        ]
        for fn, x in leaks:
            with pytest.raises(LookAheadError, match="Look-ahead"):
                check_causal(fn, x, n_checks=20)

    @pytest.mark.parametrize("name", list(CAUSAL))
    def test_causal_functions_pass_on_signed_and_zero_data(self, name: str) -> None:
        values = _values()
        check_causal(CAUSAL[name], values, n_checks=8)
        sparse = np.where(np.random.default_rng(3).random(values.shape) < 0.3, 0.0, values)
        check_causal(CAUSAL[name], np.abs(sparse), n_checks=8)

    def test_error_names_row_and_cut(self) -> None:
        x = np.array([1.0, 2.0, 3.0])  # the only cut point is 1
        with pytest.raises(LookAheadError, match=r"time index 1 changed .* after time index 1"):
            check_causal(lambda v: np.roll(v, -1, axis=0), x)
        with pytest.raises(LookAheadError, match=r"time index 0 changed .* after time index 1"):
            check_causal(lambda v: v - v[-1], x)

    def test_tolerance(self) -> None:
        x = _prices()

        def leaky(v: np.ndarray) -> np.ndarray:
            return v + 1e-9 * np.nansum(v)

        with pytest.raises(LookAheadError):
            check_causal(leaky, x)
        check_causal(leaky, x, atol=1e-3)

    def test_all_cuts_when_n_checks_large(self) -> None:
        cuts: list[int] = []
        x = np.arange(1.0, 8.0)

        def spy(v: np.ndarray) -> np.ndarray:
            changed = np.flatnonzero(v != x)
            cuts.append(int(changed[0]) - 1 if changed.size else -1)
            return v

        check_causal(spy, x, n_checks=100)
        # The baseline, then two perturbations per cut for positive data without NaNs
        # (rescaled, rows reordered); the rescaled copy always differs from row cut + 1 on.
        assert cuts[0] == -1 and len(cuts) == 1 + 2 * 5
        assert cuts[1::2] == [1, 2, 3, 4, 5]

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="at least 3 rows"):
            check_causal(log_returns, np.ones(2))
        with pytest.raises(ValueError, match="at least 3 rows"):
            check_causal(log_returns, 1.0)
        with pytest.raises(ValueError, match="axis 0"):
            check_causal(lambda v: v[1:], np.ones(5))
        with pytest.raises(ValueError, match="n_checks"):
            check_causal(log_returns, np.ones(5), n_checks=0)
        with pytest.raises(ValueError, match="non-negative"):
            check_causal(log_returns, np.ones(5), atol=-1.0)


# ---------------------------------------------------------------------------
# Validation and edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("call", "error", "match"),
    [
        (lambda: rolling_mean(np.ones(5), 0), ValueError, "window"),
        (lambda: rolling_mean(np.ones(5), 2.5), TypeError, "window"),
        (lambda: rolling_mean(np.ones(5), True), TypeError, "window"),
        (lambda: rolling_mean(np.ones(5), 3, min_periods=0), ValueError, "min_periods"),
        (lambda: rolling_mean(np.ones(5), 3, min_periods=4), ValueError, "min_periods"),
        (lambda: rolling_std(np.ones(5), 3, ddof=-1), ValueError, "ddof"),
        (lambda: rolling_zscore(np.ones(5), 3, min_periods=5), ValueError, "min_periods"),
        (lambda: rolling_mean(np.ones((2, 2, 2)), 1), ValueError, "1-D"),
        (lambda: momentum(np.ones(5), 0), ValueError, "window"),
        (lambda: lag(np.ones(5), -1), ValueError, "periods"),
        (lambda: forward_fill(np.ones(5), limit=-1), ValueError, "limit"),
        (lambda: ewm_mean(np.ones(5), 0.0), ValueError, "halflife"),
        (lambda: ewm_std(np.ones(5), -1.0), ValueError, "halflife"),
        (lambda: ewm_mean(np.ones(5), np.inf), ValueError, "halflife"),
        (lambda: ewm_mean(np.ones(5), 1.0, min_periods=0), ValueError, "min_periods"),
        (lambda: realized_volatility(np.ones(5), 2, periods_per_year=0), ValueError, "periods"),
        (lambda: log_returns(np.ones((2, 2, 2))), ValueError, "prices"),
        (lambda: log_returns(1.0), ValueError, "prices"),
    ],
)
def test_validation_errors(call: Callable[[], Any], error: type[Exception], match: str) -> None:
    with pytest.raises(error, match=match):
        call()


@pytest.mark.parametrize(
    "fn",
    [
        log_returns,
        simple_returns,
        lambda x: momentum(x, 2),
        lag,
        forward_fill,
        lambda x: rolling_mean(x, 3),
        lambda x: rolling_std(x, 3),
        lambda x: rolling_zscore(x, 3),
        lambda x: ewm_mean(x, 2.0),
        lambda x: ewm_std(x, 2.0),
    ],
)
def test_empty_and_all_nan_inputs(fn: Callable[[np.ndarray], np.ndarray]) -> None:
    for shape in [(0,), (0, 3)]:
        out = fn(np.empty(shape))
        assert out.shape == shape and out.dtype == np.float64
    out = fn(np.full((6, 2), np.nan))
    assert out.shape == (6, 2) and np.isnan(out).all()


def test_does_not_mutate_inputs() -> None:
    x = _values()
    before = x.copy()
    for fn in (*CAUSAL.values(),):
        fn(np.abs(x) + 1.0)
        fn(x)
    np.testing.assert_array_equal(x, before)
