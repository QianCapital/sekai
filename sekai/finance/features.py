"""Causal (point-in-time) feature transforms for market data, plus a look-ahead detector.

Every transform works along axis 0 (time) of an array shaped (T,) or (T, N) for N assets,
treats NaN as missing data (an asset not yet listed, a halted day, ...) and is causal: the
output at time t depends only on inputs at times <= t. Features can therefore be computed
once over a whole history and sliced inside an environment without leaking the future.
Results are float64 arrays of the input's shape unless stated otherwise.

`check_causal` verifies the causality property empirically for any transform, and
`asof_join` aligns irregularly timed data (macro releases, earnings) to bar timestamps
without look-ahead.
"""

from __future__ import annotations

import datetime as _dt
import operator
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike, NDArray

from sekai.error import LookAheadError

__all__ = [
    # returns and lags
    "log_returns",
    "simple_returns",
    "momentum",
    "lag",
    "forward_fill",
    # rolling and exponentially weighted statistics
    "rolling_mean",
    "rolling_std",
    "rolling_zscore",
    "ewm_mean",
    "ewm_std",
    "realized_volatility",
    # cross-sectional transforms
    "cross_sectional_zscore",
    "cross_sectional_rank",
    # point-in-time alignment and look-ahead detection
    "asof_join",
    "check_causal",
]


# ---------------------------------------------------------------------------
# Validation and array helpers
# ---------------------------------------------------------------------------


def _as_2d(x: ArrayLike, name: str) -> tuple[NDArray[np.float64], bool]:
    """Convert to float64 (T, N); the flag records whether the input was 1-D."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        return arr[:, None], True
    if arr.ndim == 2:
        return arr, False
    raise ValueError(f"{name} must be 1-D (T,) or 2-D (T, N), got shape {arr.shape}.")


def _restore(out: NDArray[np.float64], was_1d: bool) -> NDArray[np.float64]:
    return out[:, 0] if was_1d else out


def _as_cross_section(x: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"x must be 2-D (T, N) for a cross-sectional transform, got {arr.shape}.")
    return arr


def _check_int(name: str, value: int, minimum: int) -> int:
    if isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be an integer, got {value!r}.")
    try:
        ivalue = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {value!r}.") from None
    if ivalue < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {ivalue}.")
    return ivalue


def _check_min_periods(min_periods: int | None, window: int) -> int:
    if min_periods is None:
        return window
    mp = _check_int("min_periods", min_periods, 1)
    if mp > window:
        raise ValueError(f"min_periods ({mp}) must not exceed window ({window}).")
    return mp


def _check_positive(name: str, value: float) -> float:
    fvalue = float(value)
    if not np.isfinite(fvalue) or fvalue <= 0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}.")
    return fvalue


def _trailing_windows(a: NDArray[Any], window: int, fill: Any) -> NDArray[Any]:
    """(T, N) -> read-only (T, N, window) view; row t holds rows t-window+1..t of `a`.

    Rows before the start of the data are filled with `fill`.
    """
    n_rows, n_cols = a.shape
    if n_rows == 0:
        return np.empty((0, n_cols, window), dtype=a.dtype)
    pad = np.full((window - 1, n_cols), fill, dtype=a.dtype)
    return sliding_window_view(np.concatenate([pad, a], axis=0), window, axis=0)


# Cap on the size of (rows, N, window) temporaries in the rolling statistics (~16 MB per
# float64 array); longer histories are processed in chunks of rows.
_CHUNK_ELEMENTS = 1 << 21


def _row_chunks(shape: tuple[int, ...], window: int) -> list[slice]:
    n_rows, n_cols = shape
    step = max(1, _CHUNK_ELEMENTS // max(1, n_cols * window))
    return [slice(start, min(start + step, n_rows)) for start in range(0, n_rows, step)]


def _last_valid_index(valid: NDArray[np.bool_]) -> NDArray[np.intp]:
    """Row of the most recent valid entry at or before each t (-1 if none yet)."""
    rows = np.arange(valid.shape[0])[:, None]
    out: NDArray[np.intp] = np.maximum.accumulate(np.where(valid, rows, -1), axis=0)
    return out


def _last_valid_value(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unlimited forward fill of a (T, N) array, with 0.0 before the first valid entry."""
    last = _last_valid_index(~np.isnan(arr))
    filled = arr[np.maximum(last, 0), np.arange(arr.shape[1])]
    out: NDArray[np.float64] = np.where(last >= 0, filled, 0.0)
    return out


# ---------------------------------------------------------------------------
# Returns and lags
# ---------------------------------------------------------------------------


def _price_ratio(prices: ArrayLike, periods: int) -> NDArray[np.float64]:
    """P_t / P_{t-periods}; NaN where either price is missing or not strictly positive."""
    arr = np.asarray(prices, dtype=np.float64)
    if arr.ndim not in (1, 2):
        raise ValueError(f"prices must be 1-D (T,) or 2-D (T, N), got shape {arr.shape}.")
    ratio = np.full(arr.shape, np.nan)
    if periods < arr.shape[0]:
        prev, cur = arr[:-periods], arr[periods:]
        ok = np.isfinite(prev) & np.isfinite(cur) & (prev > 0) & (cur > 0)
        ratio[periods:] = np.where(ok, cur / np.where(ok, prev, 1.0), np.nan)
    return ratio


def log_returns(prices: ArrayLike) -> NDArray[np.float64]:
    """One-period log returns log(P_t / P_{t-1}).

    Args:
        prices: Prices shaped (T,) or (T, N).

    Returns:
        Log returns of the same shape. Row 0 is NaN, as is any entry where either price
        is missing (non-finite) or not strictly positive.
    """
    out: NDArray[np.float64] = np.log(_price_ratio(prices, 1))
    return out


def simple_returns(prices: ArrayLike) -> NDArray[np.float64]:
    """One-period simple returns P_t / P_{t-1} - 1.

    Args:
        prices: Prices shaped (T,) or (T, N).

    Returns:
        Simple returns of the same shape, with the NaN rules of `log_returns`.
    """
    return _price_ratio(prices, 1) - 1.0


def momentum(prices: ArrayLike, window: int) -> NDArray[np.float64]:
    """Trailing log return over `window` periods, log(P_t / P_{t-window}).

    Args:
        prices: Prices shaped (T,) or (T, N).
        window: Look-back in periods (>= 1); window=1 equals `log_returns`.

    Returns:
        Momentum of the same shape; NaN for t < window or where either price is missing
        or not strictly positive.

    Raises:
        ValueError: If window < 1.
    """
    w = _check_int("window", window, 1)
    out: NDArray[np.float64] = np.log(_price_ratio(prices, w))
    return out


def lag(x: ArrayLike, periods: int = 1) -> NDArray[np.float64]:
    """Shift a series down (later in time) by `periods` rows.

    Args:
        x: Data shaped (T,) or (T, N).
        periods: Number of rows to shift by (>= 0).

    Returns:
        Array of the same shape whose row t is row t - periods of x; the first
        `periods` rows are NaN.

    Raises:
        ValueError: If periods < 0.
    """
    arr, was_1d = _as_2d(x, "x")
    k = _check_int("periods", periods, 0)
    out = np.full(arr.shape, np.nan)
    if k < arr.shape[0]:
        out[k:] = arr[: arr.shape[0] - k]
    return _restore(out, was_1d)


def forward_fill(x: ArrayLike, *, limit: int | None = None) -> NDArray[np.float64]:
    """Carry the last valid (non-NaN) value forward along time.

    Args:
        x: Data shaped (T,) or (T, N).
        limit: Maximum number of consecutive NaNs to fill after a valid value
            (None = unlimited, 0 = no filling).

    Returns:
        Filled array of the same shape. Leading NaNs (before the first valid value)
        stay NaN, as do NaNs more than `limit` rows after the last valid value.

    Raises:
        ValueError: If limit < 0.
    """
    arr, was_1d = _as_2d(x, "x")
    max_fill = None if limit is None else _check_int("limit", limit, 0)
    last = _last_valid_index(~np.isnan(arr))
    fill = last >= 0
    if max_fill is not None:
        fill &= np.arange(arr.shape[0])[:, None] - last <= max_fill
    source = arr[np.maximum(last, 0), np.arange(arr.shape[1])]
    out: NDArray[np.float64] = np.where(fill, source, np.nan)
    return _restore(out, was_1d)


# ---------------------------------------------------------------------------
# Rolling and exponentially weighted statistics
# ---------------------------------------------------------------------------


def _rolling_deviation_stats(
    arr: NDArray[np.float64], window: int, ddof: int, need: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Trailing-window moments computed on data shifted by an in-window value.

    Each window is shifted by the most recent valid value up to t (always inside the
    window when the window has any valid entry). This two-pass scheme is accurate, and a
    constant window yields a standard deviation of exactly 0.

    Returns:
        (shift, mean_dev, std): the shift used at each t, the window mean minus the
        shift, and the window standard deviation (NaN where fewer than `need` valid
        values are in the window).
    """
    shift = _last_valid_value(arr)
    windows = _trailing_windows(arr, window, np.nan)
    mean_dev = np.zeros(arr.shape)
    var = np.full(arr.shape, np.nan)
    for rows in _row_chunks(arr.shape, window):
        win = windows[rows]
        win_valid = ~np.isnan(win)
        count = win_valid.sum(axis=-1)
        dev = np.where(win_valid, win - shift[rows, :, None], 0.0)
        mean_dev[rows] = dev.sum(axis=-1) / np.maximum(count, 1)
        resid = np.where(win_valid, dev - mean_dev[rows, :, None], 0.0)
        sum_sq = np.square(resid).sum(axis=-1)
        # Where count >= need > ddof the divisor is positive; elsewhere the result is NaN.
        var[rows] = np.where(count >= need, sum_sq / np.maximum(count - ddof, 1), np.nan)
    return shift, mean_dev, np.sqrt(var)


def rolling_mean(
    x: ArrayLike, window: int, *, min_periods: int | None = None
) -> NDArray[np.float64]:
    """Trailing rolling mean over rows t-window+1..t, ignoring NaNs.

    Matches pandas `rolling(window, min_periods).mean()`.

    Args:
        x: Data shaped (T,) or (T, N).
        window: Window length in rows (>= 1).
        min_periods: Minimum number of valid values required in the window
            (1 <= min_periods <= window; default: window).

    Returns:
        Rolling mean of the same shape; NaN where the window holds fewer than
        `min_periods` valid values.

    Raises:
        ValueError: If window or min_periods is out of range.
    """
    arr, was_1d = _as_2d(x, "x")
    w = _check_int("window", window, 1)
    mp = _check_min_periods(min_periods, w)
    windows = _trailing_windows(arr, w, np.nan)
    out = np.full(arr.shape, np.nan)
    for rows in _row_chunks(arr.shape, w):
        win = windows[rows]
        win_valid = ~np.isnan(win)
        count = win_valid.sum(axis=-1)
        total = np.where(win_valid, win, 0.0).sum(axis=-1)
        out[rows] = np.where(count >= mp, total / np.maximum(count, 1), np.nan)
    return _restore(out, was_1d)


def rolling_std(
    x: ArrayLike, window: int, *, min_periods: int | None = None, ddof: int = 1
) -> NDArray[np.float64]:
    """Trailing rolling standard deviation over rows t-window+1..t, ignoring NaNs.

    Matches pandas `rolling(window, min_periods).std(ddof)`.

    Args:
        x: Data shaped (T,) or (T, N).
        window: Window length in rows (>= 1).
        min_periods: Minimum number of valid values required in the window
            (1 <= min_periods <= window; default: window).
        ddof: Delta degrees of freedom (>= 0); the divisor is n_valid - ddof.

    Returns:
        Rolling standard deviation of the same shape; NaN where the window holds fewer
        than max(min_periods, ddof + 1) valid values.

    Raises:
        ValueError: If window, min_periods or ddof is out of range.
    """
    arr, was_1d = _as_2d(x, "x")
    w = _check_int("window", window, 1)
    mp = _check_min_periods(min_periods, w)
    d = _check_int("ddof", ddof, 0)
    _, _, std = _rolling_deviation_stats(arr, w, d, max(mp, d + 1))
    return _restore(std, was_1d)


def rolling_zscore(
    x: ArrayLike, window: int, *, min_periods: int | None = None
) -> NDArray[np.float64]:
    """Trailing rolling z-score (x_t - mean) / std over rows t-window+1..t.

    The mean and the (ddof=1) standard deviation are those of `rolling_mean` and
    `rolling_std` over the same window, which includes x_t itself.

    Args:
        x: Data shaped (T,) or (T, N).
        window: Window length in rows (>= 1).
        min_periods: Minimum number of valid values required in the window
            (1 <= min_periods <= window; default: window). At least 2 are always needed.

    Returns:
        Z-scores of the same shape: NaN where x_t is NaN or the window holds too few
        valid values, and 0.0 where the window's standard deviation is 0.

    Raises:
        ValueError: If window or min_periods is out of range.
    """
    arr, was_1d = _as_2d(x, "x")
    w = _check_int("window", window, 1)
    mp = _check_min_periods(min_periods, w)
    shift, mean_dev, std = _rolling_deviation_stats(arr, w, 1, max(mp, 2))
    ok = ~np.isnan(arr) & ~np.isnan(std)
    scale = ok & (std > 0)
    out = np.full(arr.shape, np.nan)
    # x_t - mean == (x_t - shift) - mean_dev, and x_t - shift is exactly 0 where x_t is valid.
    out[scale] = (arr - shift - mean_dev)[scale] / std[scale]
    out[ok & (std == 0)] = 0.0
    return _restore(out, was_1d)


def _ewm(
    arr: NDArray[np.float64], halflife: float, min_periods: int, *, std: bool
) -> NDArray[np.float64]:
    """Recursive EW mean (or standard deviation) along axis 0 of a (T, N) array."""
    alpha = 1.0 - np.exp(np.log(0.5) / halflife)
    n_cols = arr.shape[1]
    mean = np.zeros(n_cols)
    var = np.zeros(n_cols)
    seen = np.zeros(n_cols, dtype=np.int64)
    valid = ~np.isnan(arr)
    out = np.full(arr.shape, np.nan)
    for t in range(arr.shape[0]):
        v = valid[t]
        if not v.any():
            continue
        first = v & (seen == 0)
        update = v & (seen > 0)
        d = np.where(update, arr[t] - mean, 0.0)
        var = np.where(update, (1.0 - alpha) * (var + alpha * d * d), var)
        mean = np.where(first, arr[t], mean + alpha * d)
        seen += v
        emit = v & (seen >= min_periods)
        out[t, emit] = (np.sqrt(var) if std else mean)[emit]
    return out


def ewm_mean(x: ArrayLike, halflife: float, *, min_periods: int = 1) -> NDArray[np.float64]:
    """Exponentially weighted moving average m_t = (1 - a) m_{t-1} + a x_t.

    a = 1 - exp(log(0.5) / halflife) and the recursion starts at the first valid value.
    Without NaNs this matches pandas `ewm(halflife=halflife, adjust=False).mean()`.

    Args:
        x: Data shaped (T,) or (T, N).
        halflife: Half-life in rows (> 0).
        min_periods: Number of valid values that must have been seen before a value is
            emitted (>= 1).

    Returns:
        EWMA of the same shape. A NaN input leaves the state unchanged and gives NaN at
        that t; the output is also NaN until `min_periods` valid values have been seen.

    Raises:
        ValueError: If halflife <= 0 or min_periods < 1.
    """
    arr, was_1d = _as_2d(x, "x")
    h = _check_positive("halflife", halflife)
    mp = _check_int("min_periods", min_periods, 1)
    return _restore(_ewm(arr, h, mp, std=False), was_1d)


def ewm_std(x: ArrayLike, halflife: float, *, min_periods: int = 2) -> NDArray[np.float64]:
    """Exponentially weighted moving standard deviation.

    On each valid x_t, with d = x_t - m_{t-1}: m_t = m_{t-1} + a d and
    v_t = (1 - a) (v_{t-1} + a d^2), starting from m = x, v = 0 at the first valid value;
    the output is sqrt(v_t). a is as in `ewm_mean`.

    Args:
        x: Data shaped (T,) or (T, N).
        halflife: Half-life in rows (> 0).
        min_periods: Number of valid values that must have been seen before a value is
            emitted (>= 1).

    Returns:
        EW standard deviation of the same shape, with the NaN rules of `ewm_mean`.

    Raises:
        ValueError: If halflife <= 0 or min_periods < 1.
    """
    arr, was_1d = _as_2d(x, "x")
    h = _check_positive("halflife", halflife)
    mp = _check_int("min_periods", min_periods, 1)
    return _restore(_ewm(arr, h, mp, std=True), was_1d)


def realized_volatility(
    returns: ArrayLike, window: int, *, periods_per_year: float | None = None
) -> NDArray[np.float64]:
    """Trailing realised volatility: `rolling_std(returns, window)` (ddof=1).

    Args:
        returns: Per-period returns shaped (T,) or (T, N).
        window: Window length in rows (>= 1); a full window of valid returns is required.
        periods_per_year: If given (> 0), annualise by multiplying by
            sqrt(periods_per_year), e.g. 252 for daily returns.

    Returns:
        Volatility of the same shape, per period or annualised.

    Raises:
        ValueError: If window < 1 or periods_per_year <= 0.
    """
    vol = rolling_std(returns, window)
    if periods_per_year is None:
        return vol
    out: NDArray[np.float64] = vol * np.sqrt(_check_positive("periods_per_year", periods_per_year))
    return out


# ---------------------------------------------------------------------------
# Cross-sectional transforms
# ---------------------------------------------------------------------------


def cross_sectional_zscore(x: ArrayLike) -> NDArray[np.float64]:
    """Z-score each row across assets: (x - mean) / std over the row's valid entries.

    Args:
        x: Data shaped (T, N).

    Returns:
        (T, N) z-scores using the population (ddof=0) standard deviation. NaN entries
        stay NaN; rows with fewer than 2 valid entries are all NaN; rows whose valid
        entries are all equal give 0.0.

    Raises:
        ValueError: If x is not 2-D.
    """
    arr = _as_cross_section(x)
    n_rows, n_cols = arr.shape
    if n_cols == 0:
        empty: NDArray[np.float64] = arr.copy()
        return empty
    valid = ~np.isnan(arr)
    count = valid.sum(axis=1)
    # Shift each row by one of its own values so that constant rows give exactly 0.
    first = np.argmax(valid, axis=1)
    shift = np.where(count > 0, arr[np.arange(n_rows), first], 0.0)
    dev = np.where(valid, arr - shift[:, None], 0.0)
    mean_dev = dev.sum(axis=1) / np.maximum(count, 1)
    resid = np.where(valid, dev - mean_dev[:, None], 0.0)
    std = np.sqrt(np.square(resid).sum(axis=1) / np.maximum(count, 1))
    row_ok = count >= 2
    scale = valid & (row_ok & (std > 0))[:, None]
    out = np.full(arr.shape, np.nan)
    out[scale] = (resid / np.where(std > 0, std, 1.0)[:, None])[scale]
    out[valid & (row_ok & (std == 0))[:, None]] = 0.0
    return out


def cross_sectional_rank(x: ArrayLike) -> NDArray[np.float64]:
    """Rank each row across assets, scaled to [0, 1].

    Valid entries get (rank - 1) / (n_valid - 1), where rank is the 1-based average
    rank (ties share the mean of their ranks).

    Args:
        x: Data shaped (T, N).

    Returns:
        (T, N) scaled ranks. NaN entries stay NaN; a row with a single valid entry gives
        0.5 for it.

    Raises:
        ValueError: If x is not 2-D.
    """
    arr = _as_cross_section(x)
    n_rows, n_cols = arr.shape
    if n_cols == 0:
        empty: NDArray[np.float64] = arr.copy()
        return empty
    valid = ~np.isnan(arr)
    count = valid.sum(axis=1)
    order = np.argsort(arr, axis=1, kind="stable")  # NaNs sort last
    ranked = np.take_along_axis(arr, order, axis=1)
    pos = np.broadcast_to(np.arange(n_cols), (n_rows, n_cols))
    starts = np.ones((n_rows, n_cols), dtype=bool)
    starts[:, 1:] = ranked[:, 1:] != ranked[:, :-1]  # NaN != NaN, so NaNs never tie
    ends = np.ones((n_rows, n_cols), dtype=bool)
    ends[:, :-1] = starts[:, 1:]
    group_first = np.maximum.accumulate(np.where(starts, pos, 0), axis=1)
    group_last = np.minimum.accumulate(np.where(ends, pos, n_cols - 1)[:, ::-1], axis=1)[:, ::-1]
    avg_rank0 = 0.5 * (group_first + group_last)
    denom = (count - 1)[:, None]
    scaled = np.where(denom > 0, avg_rank0 / np.maximum(denom, 1), 0.5)
    out = np.empty(arr.shape)
    np.put_along_axis(out, order, scaled, axis=1)
    out[~valid] = np.nan
    return out


# ---------------------------------------------------------------------------
# Point-in-time alignment and look-ahead detection
# ---------------------------------------------------------------------------


def asof_join(
    target_times: ArrayLike,
    event_times: ArrayLike,
    values: ArrayLike,
    *,
    lag: np.timedelta64 | _dt.timedelta | float | None = None,
) -> NDArray[np.float64]:
    """Align irregularly timed data to target timestamps without look-ahead.

    For each target time t the result holds the value of the last event with
    event_time + lag <= t, i.e. the latest information that was public at t. Typical
    uses are macro releases or earnings (with `lag` modelling the publication delay)
    aligned to bar timestamps.

    Args:
        target_times: (T,) target timestamps, sorted ascending.
        event_times: (E,) event timestamps, sorted ascending; among equal times the
            last event wins. Either both time arrays are np.datetime64 or both numeric.
        values: (E,) or (E, k) event values.
        lag: Non-negative delay before an event becomes known: a np.timedelta64 (or
            datetime.timedelta) for datetime64 times, a number for numeric times, or
            None for no delay.

    Returns:
        float64 array shaped (T,) or (T, k); NaN where no event is known yet.

    Raises:
        TypeError: If the time arrays are of mixed or unsupported kinds, or `lag` does
            not match their kind.
        ValueError: If a time array is not 1-D, contains NaN/NaT or is not sorted, if
            `values` does not have E rows, or if `lag` is negative or NaN/NaT.
    """
    targets = np.asarray(target_times)
    events = np.asarray(event_times)
    vals = np.asarray(values, dtype=np.float64)
    if targets.ndim != 1 or events.ndim != 1:
        raise ValueError(
            f"target_times and event_times must be 1-D, got shapes {targets.shape} "
            f"and {events.shape}."
        )
    if vals.ndim not in (1, 2) or vals.shape[0] != events.shape[0]:
        raise ValueError(
            f"values must be shaped (E,) or (E, k) with E = len(event_times) = "
            f"{events.shape[0]}, got {vals.shape}."
        )

    is_datetime = (targets.dtype.kind == "M", events.dtype.kind == "M")
    if is_datetime == (True, True):
        if np.isnat(targets).any() or np.isnat(events).any():
            raise ValueError("target_times and event_times must not contain NaT.")
        effective = events
        if lag is not None:
            if not isinstance(lag, np.timedelta64 | _dt.timedelta):
                raise TypeError(f"lag must be a np.timedelta64 for datetime64 times, got {lag!r}.")
            lag_td = np.timedelta64(lag)
            if np.isnat(lag_td) or lag_td < np.timedelta64(0):
                raise ValueError(f"lag must be a non-negative timedelta, got {lag!r}.")
            effective = events + lag_td
        common = np.promote_types(effective.dtype, targets.dtype)
    elif is_datetime == (False, False):
        for name, arr in (("target_times", targets), ("event_times", events)):
            if arr.dtype.kind not in "iuf":
                raise TypeError(f"{name} must be np.datetime64 or numeric, got dtype {arr.dtype}.")
            if arr.dtype.kind == "f" and np.isnan(arr).any():
                raise ValueError(f"{name} must not contain NaN.")
        effective = events
        if lag is not None:
            if isinstance(lag, bool | np.bool_ | np.timedelta64 | _dt.timedelta) or not (
                isinstance(lag, int | float | np.integer | np.floating)
            ):
                raise TypeError(f"lag must be a number for numeric times, got {lag!r}.")
            if not np.isfinite(lag) or lag < 0:
                raise ValueError(f"lag must be non-negative and finite, got {lag!r}.")
            effective = events + lag
        common = np.result_type(effective, targets)
    else:
        raise TypeError(
            "target_times and event_times must both be np.datetime64 or both numeric, "
            f"got dtypes {targets.dtype} and {events.dtype}."
        )

    if np.any(events[1:] < events[:-1]):
        raise ValueError("event_times must be sorted ascending.")
    if np.any(targets[1:] < targets[:-1]):
        raise ValueError("target_times must be sorted ascending.")

    idx = np.searchsorted(effective.astype(common), targets.astype(common), side="right") - 1
    known = idx >= 0
    out = np.full((targets.shape[0], *vals.shape[1:]), np.nan)
    out[known] = vals[idx[known]]
    return out


def _call_aligned(
    fn: Callable[[NDArray[np.float64]], ArrayLike], x: NDArray[np.float64]
) -> NDArray[np.float64]:
    out = np.asarray(fn(x.copy()), dtype=np.float64)
    if out.ndim == 0 or out.shape[0] != x.shape[0]:
        raise ValueError(
            f"fn must return an array whose axis 0 has length T = {x.shape[0]} (aligned "
            f"with the input's time axis), got shape {out.shape}."
        )
    return out


def check_causal(
    fn: Callable[[NDArray[np.float64]], ArrayLike],
    x: ArrayLike,
    *,
    n_checks: int = 5,
    rng: np.random.Generator | None = None,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    """Empirically check that a transform has no look-ahead bias.

    For each of `n_checks` random cut points c in [1, T-2] (every such point when
    n_checks >= T - 2), x is copied with the rows after c perturbed in several ways, and a
    causal `fn` must give the same output for rows 0..c on every copy:

    - rescaled: valid entries multiplied by exp(Normal(0, 0.1));
    - reordered: the rows after c permuted;
    - values changed so that signs and zeros move, within the data's domain: for data
      with negative values, signs flipped at random plus additive noise, and some
      entries set to 0; for non-negative data with zeros, |x + noise| with some entries
      set to 0 (strictly positive data, such as prices, stay positive);
    - for data with NaNs, the NaN pattern changed: some valid entries become NaN and the
      NaNs are filled with values drawn from x (this catches survivorship bias, e.g.
      dropping assets that delist later).

    Perturbed inputs can be outside what `fn` expects downstream of the cut, so numpy
    floating-point warnings are silenced while `fn` runs on them.

    Args:
        fn: Transform mapping an array shaped like x to an array whose axis 0 is aligned
            with x's time axis. It must be deterministic.
        x: Representative input data shaped (T, ...) with T >= 3.
        n_checks: Number of cut points to test (>= 1).
        rng: Source of randomness for the cut points and perturbations
            (default: np.random.default_rng(0)).
        rtol: Relative tolerance for the comparison (as in np.allclose).
        atol: Absolute tolerance for the comparison (as in np.allclose).

    Raises:
        LookAheadError: If an output at or before a cut point changed, naming the first
            time index that differs and the cut point.
        ValueError: If T < 3, n_checks < 1, a tolerance is negative, or fn's output is
            not aligned with x's time axis.
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0 or arr.shape[0] < 3:
        raise ValueError(f"x needs at least 3 rows along axis 0, got shape {arr.shape}.")
    n = _check_int("n_checks", n_checks, 1)
    if rtol < 0 or atol < 0:
        raise ValueError(f"rtol and atol must be non-negative, got {rtol} and {atol}.")
    gen = np.random.default_rng(0) if rng is None else rng

    candidates = np.arange(1, arr.shape[0] - 1)
    if n < candidates.size:
        candidates = np.sort(gen.choice(candidates, size=n, replace=False))
    baseline = _call_aligned(fn, arr)
    for cut in candidates.tolist():
        for kind, tail in _perturbed_tails(arr, cut, gen):
            perturbed = arr.copy()
            perturbed[cut + 1 :] = tail
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                result = _call_aligned(fn, perturbed)
            if result.shape != baseline.shape:
                raise ValueError(
                    f"fn returned shape {result.shape} for the perturbed input but "
                    f"{baseline.shape} for the original."
                )
            same = np.isclose(
                baseline[: cut + 1], result[: cut + 1], rtol=rtol, atol=atol, equal_nan=True
            )
            if not same.all():
                row = int(np.argmin(same.reshape(cut + 1, -1).all(axis=1)))
                raise LookAheadError(
                    f"Look-ahead bias detected: the output at time index {row} changed when "
                    f"only inputs after time index {cut} (the cut point) were perturbed "
                    f"({kind})."
                )


def _perturbed_tails(
    arr: NDArray[np.float64], cut: int, gen: np.random.Generator
) -> list[tuple[str, NDArray[np.float64]]]:
    """Replacements for the rows of `arr` after `cut`, each changing what a leak could see."""
    tail = arr[cut + 1 :]
    valid = ~np.isnan(tail)
    finite_values = arr[np.isfinite(arr)]
    out = [
        ("rescaled", tail * np.exp(gen.normal(0.0, 0.1, size=tail.shape))),
        ("rows reordered", tail[gen.permutation(tail.shape[0])]),
    ]
    if finite_values.size:
        negative, zero = bool(np.any(finite_values < 0)), bool(np.any(finite_values == 0))
        if negative or zero:  # signs and zeros carry information: move them
            scale = float(np.std(finite_values)) or float(np.max(np.abs(finite_values))) or 1.0
            noisy = tail + gen.normal(0.0, scale, size=tail.shape)
            if negative:
                noisy = noisy * gen.choice([-1.0, 1.0], size=tail.shape)
            else:
                noisy = np.abs(noisy)
            noisy = np.where(gen.random(tail.shape) < 0.1, 0.0, noisy)
            out.append(("signs and zeros changed", np.where(np.isfinite(tail), noisy, tail)))
        if np.isnan(arr).any():  # the NaN pattern (listings, delistings) carries information
            fill = gen.choice(finite_values, size=tail.shape)
            masked = np.where(valid, tail, fill)
            masked = np.where(valid & (gen.random(tail.shape) < 0.2), np.nan, masked)
            out.append(("NaN pattern changed", masked))
    return out
