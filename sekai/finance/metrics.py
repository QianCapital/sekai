"""Trading performance metrics, and a wrapper that reports them per episode.

The metric functions take a 1-D series of *simple* per-period returns (``0.01``
is +1 % over the period), check that it is non-empty and finite (raising
`ValueError` otherwise) and return plain Python floats. Annualisation uses
`periods_per_year`: 252 for daily bars, 52 for weekly, 12 for monthly,
``252 * 390`` for one-minute bars of a US equity session.

Wipe-outs follow a limited-liability convention, as the finance envs do:
equity cannot fall below zero. A return of -100 % or worse wipes the account
out and equity stays at zero from then on, so `total_return` is at least -1,
drawdowns lie in [0, 1] and `annualized_return` is -1. Dispersion and tail
statistics (volatility, Sharpe, Sortino, VaR, CVaR, hit rate) use the returns
exactly as given.

Returns are undefined for P&L processes that start from zero wealth (e.g. a
market maker's mark-to-market P&L); summarise those with `pnl_summary`, which
works on per-period P&L increments in absolute units.

Picking the best of many backtests (seeds, hyperparameters, rules) inflates its
Sharpe ratio. `probabilistic_sharpe_ratio`, `minimum_track_record_length` and
`deflated_sharpe_ratio` account for the length of the track record, non-normal
returns and the number of trials; `probability_of_backtest_overfitting` estimates,
from the return series of every trial, how often the in-sample best does worse
than the median out of sample.

`RecordPerformance` wraps an env whose info reports account equity (or
cumulative P&L) and attaches a `PerformanceSummary` (or `PnLSummary`) to the
info of each episode's final step.
"""

from __future__ import annotations

import itertools
import math
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sekai._types import ActType, ObsType, SeedType
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.core.wrapper import Wrapper

__all__ = [
    # Return-series metrics
    "equity_curve",
    "returns_from_equity",
    "total_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "drawdown_series",
    "max_drawdown",
    "calmar_ratio",
    "value_at_risk",
    "conditional_value_at_risk",
    "hit_rate",
    # Summaries
    "PerformanceSummary",
    "performance_summary",
    "PnLSummary",
    "pnl_summary",
    # Multiple testing and overfitting
    "probabilistic_sharpe_ratio",
    "minimum_track_record_length",
    "expected_maximum_sharpe",
    "deflated_sharpe_ratio",
    "OverfittingReport",
    "probability_of_backtest_overfitting",
    # Wrapper
    "RecordPerformance",
]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _as_series(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """`values` as a non-empty, finite 1-D float64 array."""
    try:
        x = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a 1-D array of numbers: {exc}") from exc
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {x.shape}")
    if x.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be finite (no NaN or inf)")
    return x


def _check_positive(value: float, name: str) -> float:
    v = float(value)
    if not (math.isfinite(v) and v > 0.0):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
    return v


def _check_finite(value: float, name: str) -> float:
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


def _check_non_negative(value: float, name: str) -> float:
    v = float(value)
    if not (math.isfinite(v) and v >= 0.0):
        raise ValueError(f"{name} must be a non-negative finite number, got {value!r}")
    return v


def _check_level(level: float) -> float:
    v = float(level)
    if not 0.0 < v < 1.0:
        raise ValueError(f"level must lie strictly between 0 and 1, got {level!r}")
    return v


def _check_int(value: object, name: str, minimum: int) -> int:
    if (
        isinstance(value, bool | np.bool_)
        or not isinstance(value, int | np.integer)
        or value < minimum
    ):
        raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")
    return int(value)


def _check_confidence(confidence: float) -> float:
    v = float(confidence)
    if not 0.5 <= v < 1.0:
        raise ValueError(f"confidence must lie in [0.5, 1), got {confidence!r}")
    return v


def _as_track_record(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """`values` as a finite 1-D float64 array of at least two returns."""
    x = _as_series(values, name)
    if x.size < 2:
        raise ValueError(f"{name} needs at least two periods, got {x.size}")
    return x


def _as_panel(values: ArrayLike, name: str) -> NDArray[np.float64]:
    """`values` as a finite 2-D float64 array (n_periods, n_strategies), n_strategies >= 2."""
    try:
        x = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a 2-D array of numbers: {exc}") from exc
    if x.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n_periods, n_strategies), got shape {x.shape}")
    if x.shape[1] < 2:
        raise ValueError(f"{name} needs at least two strategies (columns), got shape {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be finite (no NaN or inf)")
    return x


# ---------------------------------------------------------------------------
# Implementations on validated arrays
# ---------------------------------------------------------------------------


def _growth(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Wealth relative to 1, shape (n + 1,), floored at zero after a wipe-out."""
    growth = np.empty(x.size + 1, dtype=np.float64)
    growth[0] = 1.0
    np.cumprod(1.0 + x, out=growth[1:])
    growth[np.logical_or.accumulate(growth <= 0.0)] = 0.0
    return growth


# Relative size of floating-point rounding in a return (a difference of growth factors
# near 1) or in a P&L figure: a few ulps, with a margin.
_ROUNDING = 16.0 * float(np.finfo(np.float64).eps)


def _sample_std(x: NDArray[np.float64], *, offset: float = 1.0) -> float:
    """Sample standard deviation (ddof=1); exactly 0 for 1-element series and for series
    that are constant up to rounding.

    Rounding is measured against `offset + max|x|`: simple returns are differences of
    growth factors near 1, so they carry rounding of order eps whatever their size
    (`offset=1`); P&L increments carry rounding of order eps times the wealth level
    they were differenced from (`offset` = that level), or only their own relative
    rounding if they are exact (`offset=0`).
    """
    if x.size < 2 or bool(np.all(x == x[0])):
        return 0.0
    sd = float(np.std(x, ddof=1))
    if sd <= _ROUNDING * (offset + float(np.max(np.abs(x)))):
        return 0.0
    return sd


def _excess(x: NDArray[np.float64], hurdle: float) -> NDArray[np.float64]:
    """`x - hurdle`, with differences that are zero up to rounding set to exactly 0.

    A return computed as a ratio of equities (`e1 / e0 - 1`) carries rounding of order
    eps, so cash earning exactly the risk-free rate gives excess returns of about
    +-1e-19; treated as genuine they would decide the sign of a Sortino ratio.
    """
    excess = x - hurdle
    tolerance = _ROUNDING * (1.0 + np.abs(x) + abs(hurdle))
    out: NDArray[np.float64] = np.where(np.abs(excess) <= tolerance, 0.0, excess)
    return out


def _total_return(x: NDArray[np.float64]) -> float:
    return float(_growth(x)[-1]) - 1.0


def _annualized_return(x: NDArray[np.float64], ppy: float) -> float:
    if bool(np.any(x <= -1.0)):
        return -1.0
    log_growth = float(np.sum(np.log1p(x)))
    with np.errstate(over="ignore"):
        return float(np.expm1(log_growth * ppy / x.size))


def _annualized_volatility(x: NDArray[np.float64], ppy: float) -> float:
    return _sample_std(x) * math.sqrt(ppy)


def _sharpe_ratio(x: NDArray[np.float64], ppy: float, risk_free: float) -> float:
    excess = _excess(x, risk_free / ppy)
    sd = _sample_std(excess)
    if sd == 0.0:
        return 0.0
    return float(np.mean(excess)) / sd * math.sqrt(ppy)


def _sortino_ratio(x: NDArray[np.float64], ppy: float, target: float) -> float:
    excess = _excess(x, target / ppy)
    downside = np.minimum(excess, 0.0)
    dd = math.sqrt(float(np.mean(downside * downside)))
    mean = float(np.mean(excess))
    if dd == 0.0:
        return math.inf if mean > 0.0 else 0.0
    return mean / dd * math.sqrt(ppy)


def _drawdown_series(x: NDArray[np.float64]) -> NDArray[np.float64]:
    growth = _growth(x)
    peak = np.maximum.accumulate(growth)  # >= 1, so the division is safe
    out: NDArray[np.float64] = np.clip(1.0 - growth / peak, 0.0, 1.0)
    return out


def _max_drawdown(x: NDArray[np.float64]) -> float:
    return float(np.max(_drawdown_series(x)))


def _calmar_ratio(x: NDArray[np.float64], ppy: float) -> float:
    ann = _annualized_return(x, ppy)
    mdd = _max_drawdown(x)
    if mdd == 0.0:
        return math.inf if ann > 0.0 else 0.0
    return ann / mdd


def _value_at_risk(x: NDArray[np.float64], level: float) -> float:
    # `0.0 - q` rather than `-q` so a zero quantile reports 0.0, not -0.0.
    return 0.0 - float(np.quantile(x, 1.0 - level))


def _conditional_value_at_risk(x: NDArray[np.float64], level: float) -> float:
    q = np.quantile(x, 1.0 - level)
    tail = x[x <= q]
    if tail.size == 0:  # only possible through rounding in the interpolation
        tail = x[x == np.min(x)]
    return 0.0 - float(np.mean(tail))


def _hit_rate(x: NDArray[np.float64]) -> float:
    active = x[x != 0.0]
    if active.size == 0:
        return 0.0
    return float(np.mean(active > 0.0))


# ---------------------------------------------------------------------------
# Public return-series metrics
# ---------------------------------------------------------------------------


def equity_curve(returns: ArrayLike, initial: float = 1.0) -> NDArray[np.float64]:
    """Equity path implied by compounding simple returns.

    Args:
        returns: Simple per-period returns, shape (n,).
        initial: Starting equity (positive, any currency unit).

    Returns:
        Equity, shape (n + 1,): ``initial`` followed by ``initial * cumprod(1 + r)``.
        Once equity reaches zero or below (a return of -100 % or worse) it stays
        at zero.

    Raises:
        ValueError: If `returns` is empty, not 1-D or not finite, or `initial`
            is not positive and finite.
    """
    x = _as_series(returns, "returns")
    start = _check_positive(initial, "initial")
    return start * _growth(x)


def returns_from_equity(equity: ArrayLike) -> NDArray[np.float64]:
    """Simple returns between consecutive equity values (the inverse of `equity_curve`).

    Args:
        equity: Equity path, shape (n + 1,) with n >= 1, starting positive. The
            first value at or below zero is a wipe-out, and every value after it
            must be 0 (equity stays at zero, as `equity_curve` reports it).

    Returns:
        Simple returns ``equity[t + 1] / equity[t] - 1``, shape (n,); a wipe-out
        gives a return of exactly -1 (-100 %) and the periods after it 0, as
        `RecordPerformance` records them.

    Raises:
        ValueError: If `equity` has fewer than two values, is not 1-D or finite,
            does not start positive, or is not 0 after a wipe-out.
    """
    e = _as_series(equity, "equity")
    if e.size < 2:
        raise ValueError("equity needs at least two values to define a return")
    down = np.flatnonzero(e <= 0.0)
    wipe = int(down[0]) if down.size else e.size  # index of the wipe-out, if any
    if wipe == 0 or bool(np.any(e[wipe + 1 :] != 0.0)):
        raise ValueError("equity must be positive until a wipe-out (a value <= 0) and 0 after it")
    r = np.zeros(e.size - 1)
    r[: wipe - 1] = e[1:wipe] / e[: wipe - 1] - 1.0
    if wipe < e.size:
        r[wipe - 1] = -1.0
    return r


def total_return(returns: ArrayLike) -> float:
    """Compounded return over the whole series, ``prod(1 + r) - 1``.

    Args:
        returns: Simple per-period returns, shape (n,).

    Returns:
        Total simple return; -1.0 after a wipe-out (never lower).

    Raises:
        ValueError: If `returns` is empty, not 1-D or not finite.
    """
    return _total_return(_as_series(returns, "returns"))


def annualized_return(returns: ArrayLike, periods_per_year: float = 252) -> float:
    """Geometric annualised return, ``prod(1 + r) ** (periods_per_year / n) - 1``.

    Args:
        returns: Simple per-period returns, shape (n,).
        periods_per_year: Number of return periods in a year.

    Returns:
        Compound annual growth rate; -1.0 if equity ever reaches zero or below.
        May be ``inf`` when a short, strongly positive series is annualised.

    Raises:
        ValueError: If `returns` is empty, not 1-D or not finite, or
            `periods_per_year` is not positive.
    """
    x = _as_series(returns, "returns")
    return _annualized_return(x, _check_positive(periods_per_year, "periods_per_year"))


def annualized_volatility(returns: ArrayLike, periods_per_year: float = 252) -> float:
    """Annualised volatility, ``std(r, ddof=1) * sqrt(periods_per_year)``.

    Args:
        returns: Simple per-period returns, shape (n,).
        periods_per_year: Number of return periods in a year.

    Returns:
        Annualised standard deviation of returns; 0.0 when n < 2 or the returns are
        constant up to floating-point rounding.

    Raises:
        ValueError: If `returns` is empty, not 1-D or not finite, or
            `periods_per_year` is not positive.
    """
    x = _as_series(returns, "returns")
    return _annualized_volatility(x, _check_positive(periods_per_year, "periods_per_year"))


def sharpe_ratio(
    returns: ArrayLike, periods_per_year: float = 252, risk_free: float = 0.0
) -> float:
    """Annualised Sharpe ratio of returns in excess of a risk-free rate.

    ``mean(e) / std(e, ddof=1) * sqrt(periods_per_year)`` with
    ``e = r - risk_free / periods_per_year``.

    Args:
        returns: Simple per-period returns, shape (n,).
        periods_per_year: Number of return periods in a year.
        risk_free: Annual simple risk-free rate (e.g. 0.04 for 4 %).

    Returns:
        The Sharpe ratio; 0.0 when n < 2 or the excess returns are constant
        (zero standard deviation, up to floating-point rounding). Excess returns
        within rounding of zero count as exactly zero.

    Raises:
        ValueError: If `returns` is empty, not 1-D or not finite, or a parameter
            is invalid.
    """
    x = _as_series(returns, "returns")
    ppy = _check_positive(periods_per_year, "periods_per_year")
    return _sharpe_ratio(x, ppy, _check_finite(risk_free, "risk_free"))


def sortino_ratio(returns: ArrayLike, periods_per_year: float = 252, target: float = 0.0) -> float:
    """Annualised Sortino ratio: mean excess return over downside deviation.

    With ``e = r - target / periods_per_year``, the downside deviation is
    ``sqrt(mean(min(e, 0) ** 2))`` over all periods and the ratio is
    ``mean(e) / dd * sqrt(periods_per_year)``.

    Args:
        returns: Simple per-period returns, shape (n,).
        periods_per_year: Number of return periods in a year.
        target: Annual minimum acceptable return (simple rate).

    Returns:
        The Sortino ratio. With no period below target, ``math.inf`` if the mean
        excess return is positive, else 0.0. Excess returns within floating-point
        rounding of zero count as exactly zero, so a series that earns exactly the
        target (e.g. cash at the risk-free rate) gives 0.0.

    Raises:
        ValueError: If `returns` is empty, not 1-D or not finite, or a parameter
            is invalid.
    """
    x = _as_series(returns, "returns")
    ppy = _check_positive(periods_per_year, "periods_per_year")
    return _sortino_ratio(x, ppy, _check_finite(target, "target"))


def drawdown_series(returns: ArrayLike) -> NDArray[np.float64]:
    """Drawdown from the running peak after each period.

    Args:
        returns: Simple per-period returns, shape (n,).

    Returns:
        ``1 - equity / running_peak(equity)`` for the equity path starting at 1,
        shape (n + 1,), values in [0, 1] (1 means wiped out). The first value
        is always 0.

    Raises:
        ValueError: If `returns` is empty, not 1-D or not finite.
    """
    return _drawdown_series(_as_series(returns, "returns"))


def max_drawdown(returns: ArrayLike) -> float:
    """Maximum drawdown as a positive fraction of the running peak.

    Args:
        returns: Simple per-period returns, shape (n,).

    Returns:
        The largest value of `drawdown_series`, in [0, 1].

    Raises:
        ValueError: If `returns` is empty, not 1-D or not finite.
    """
    return _max_drawdown(_as_series(returns, "returns"))


def calmar_ratio(returns: ArrayLike, periods_per_year: float = 252) -> float:
    """Calmar ratio: annualised return over maximum drawdown.

    Args:
        returns: Simple per-period returns, shape (n,).
        periods_per_year: Number of return periods in a year.

    Returns:
        ``annualized_return / max_drawdown``. With no drawdown, ``math.inf`` if
        the annualised return is positive, else 0.0.

    Raises:
        ValueError: If `returns` is empty, not 1-D or not finite, or
            `periods_per_year` is not positive.
    """
    x = _as_series(returns, "returns")
    return _calmar_ratio(x, _check_positive(periods_per_year, "periods_per_year"))


def value_at_risk(returns: ArrayLike, level: float = 0.95) -> float:
    """Historical value at risk, reported as a positive loss.

    Args:
        returns: Simple per-period returns, shape (n,).
        level: Confidence level, strictly between 0 and 1.

    Returns:
        ``-quantile(returns, 1 - level)`` (linear interpolation between order
        statistics): the per-period loss not exceeded with probability `level`.
        Negative if even the tail quantile is a gain.

    Raises:
        ValueError: If `returns` is empty, not 1-D or not finite, or `level` is
            outside (0, 1).
    """
    x = _as_series(returns, "returns")
    return _value_at_risk(x, _check_level(level))


def conditional_value_at_risk(returns: ArrayLike, level: float = 0.95) -> float:
    """Historical expected shortfall (CVaR), reported as a positive loss.

    Args:
        returns: Simple per-period returns, shape (n,).
        level: Confidence level, strictly between 0 and 1.

    Returns:
        Minus the mean of the returns at or below ``quantile(returns, 1 - level)``.
        Always at least `value_at_risk` at the same level.

    Raises:
        ValueError: If `returns` is empty, not 1-D or not finite, or `level` is
            outside (0, 1).
    """
    x = _as_series(returns, "returns")
    return _conditional_value_at_risk(x, _check_level(level))


def hit_rate(returns: ArrayLike) -> float:
    """Fraction of active (non-zero) periods with a positive return.

    Args:
        returns: Simple per-period returns (or P&L increments), shape (n,).

    Returns:
        Winning periods over non-zero periods, in [0, 1]; 0.0 if every period
        is zero.

    Raises:
        ValueError: If `returns` is empty, not 1-D or not finite.
    """
    return _hit_rate(_as_series(returns, "returns"))


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _fmt_pct(value: float) -> str:
    return f"{value:.2%}" if math.isfinite(value) else str(value)


def _fmt_num(value: float) -> str:
    return f"{value:.3f}" if math.isfinite(value) else str(value)


def _fmt_amount(value: float) -> str:
    return f"{value:.6g}"


def _table(rows: list[tuple[str, str]]) -> str:
    label_width = max(len(label) for label, _ in rows)
    value_width = max(len(value) for _, value in rows)
    return "\n".join(f"{label:<{label_width}}  {value:>{value_width}}" for label, value in rows)


@dataclass(frozen=True)
class PerformanceSummary:
    """Performance metrics of one return series (e.g. one episode).

    Return, volatility, drawdown and VaR fields are fractions (0.05 is 5 %);
    ratios are annualised. See the module functions of the same names for the
    exact definitions.

    Attributes:
        n_periods: Number of return periods.
        total_return: Compounded simple return over the series.
        annualized_return: Geometric annualised return.
        annualized_volatility: Annualised sample standard deviation of returns.
        sharpe_ratio: Annualised Sharpe ratio.
        sortino_ratio: Annualised Sortino ratio (may be ``inf``).
        max_drawdown: Maximum drawdown, in [0, 1].
        calmar_ratio: Annualised return over maximum drawdown (may be ``inf``).
        value_at_risk: Historical per-period VaR as a positive loss.
        conditional_value_at_risk: Historical per-period expected shortfall.
        hit_rate: Fraction of non-zero periods with a positive return.
        mean_turnover: Mean per-period turnover, if turnover was supplied.
        total_cost: Sum of per-period trading costs, if costs were supplied.
    """

    n_periods: int
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    value_at_risk: float
    conditional_value_at_risk: float
    hit_rate: float
    mean_turnover: float | None = None
    total_cost: float | None = None

    def as_dict(self) -> dict[str, float | int | None]:
        """The fields as a plain dict, e.g. for logging."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __str__(self) -> str:
        rows = [
            ("periods", str(self.n_periods)),
            ("total return", _fmt_pct(self.total_return)),
            ("annualised return", _fmt_pct(self.annualized_return)),
            ("annualised volatility", _fmt_pct(self.annualized_volatility)),
            ("Sharpe ratio", _fmt_num(self.sharpe_ratio)),
            ("Sortino ratio", _fmt_num(self.sortino_ratio)),
            ("max drawdown", _fmt_pct(self.max_drawdown)),
            ("Calmar ratio", _fmt_num(self.calmar_ratio)),
            ("value at risk", _fmt_pct(self.value_at_risk)),
            ("conditional VaR", _fmt_pct(self.conditional_value_at_risk)),
            ("hit rate", _fmt_pct(self.hit_rate)),
        ]
        if self.mean_turnover is not None:
            rows.append(("mean turnover", _fmt_amount(self.mean_turnover)))
        if self.total_cost is not None:
            rows.append(("total cost", _fmt_amount(self.total_cost)))
        return _table(rows)


def performance_summary(
    returns: ArrayLike,
    periods_per_year: float = 252,
    *,
    risk_free: float = 0.0,
    level: float = 0.95,
    turnover: ArrayLike | None = None,
    costs: ArrayLike | None = None,
) -> PerformanceSummary:
    """Compute every return-series metric at once.

    Args:
        returns: Simple per-period returns, shape (n,).
        periods_per_year: Number of return periods in a year.
        risk_free: Annual simple risk-free rate; the hurdle for both the Sharpe
            and the Sortino ratio.
        level: Confidence level for VaR and CVaR, strictly between 0 and 1.
        turnover: Optional per-period turnover (e.g. traded notional as a
            fraction of equity), shape (m,). It need not cover every period.
        costs: Optional per-period trading costs, shape (k,), in the units the
            env reports them (typically a fraction of equity).

    Returns:
        A `PerformanceSummary`; `mean_turnover` / `total_cost` are None when
        `turnover` / `costs` are not given.

    Raises:
        ValueError: If any series is empty, not 1-D or not finite, or a
            parameter is invalid.
    """
    x = _as_series(returns, "returns")
    ppy = _check_positive(periods_per_year, "periods_per_year")
    rf = _check_finite(risk_free, "risk_free")
    lvl = _check_level(level)
    mean_turnover = None if turnover is None else float(np.mean(_as_series(turnover, "turnover")))
    total_cost = None if costs is None else float(np.sum(_as_series(costs, "costs")))
    return PerformanceSummary(
        n_periods=int(x.size),
        total_return=_total_return(x),
        annualized_return=_annualized_return(x, ppy),
        annualized_volatility=_annualized_volatility(x, ppy),
        sharpe_ratio=_sharpe_ratio(x, ppy, rf),
        sortino_ratio=_sortino_ratio(x, ppy, rf),
        max_drawdown=_max_drawdown(x),
        calmar_ratio=_calmar_ratio(x, ppy),
        value_at_risk=_value_at_risk(x, lvl),
        conditional_value_at_risk=_conditional_value_at_risk(x, lvl),
        hit_rate=_hit_rate(x),
        mean_turnover=mean_turnover,
        total_cost=total_cost,
    )


@dataclass(frozen=True)
class PnLSummary:
    """Performance of a P&L process measured in absolute units (no returns).

    Attributes:
        n_periods: Number of periods.
        total_pnl: Sum of per-period P&L.
        mean_pnl: Mean per-period P&L.
        pnl_volatility: Sample standard deviation of per-period P&L (ddof=1);
            0.0 when n < 2.
        sharpe_ratio: ``mean_pnl / pnl_volatility * sqrt(periods_per_year)``;
            0.0 when the standard deviation is zero or n < 2.
        max_drawdown: Largest fall of cumulative P&L from its running peak, in
            absolute units (>= 0); the running peak starts at 0.
        hit_rate: Fraction of non-zero periods with positive P&L.
    """

    n_periods: int
    total_pnl: float
    mean_pnl: float
    pnl_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    hit_rate: float

    def as_dict(self) -> dict[str, float | int]:
        """The fields as a plain dict, e.g. for logging."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __str__(self) -> str:
        return _table(
            [
                ("periods", str(self.n_periods)),
                ("total P&L", _fmt_amount(self.total_pnl)),
                ("mean P&L", _fmt_amount(self.mean_pnl)),
                ("P&L volatility", _fmt_amount(self.pnl_volatility)),
                ("Sharpe ratio", _fmt_num(self.sharpe_ratio)),
                ("max drawdown", _fmt_amount(self.max_drawdown)),
                ("hit rate", _fmt_pct(self.hit_rate)),
            ]
        )


def pnl_summary(pnl: ArrayLike, periods_per_year: float = 252, *, scale: float = 0.0) -> PnLSummary:
    """Summarise a series of per-period P&L increments.

    Args:
        pnl: Per-period P&L increments in absolute units, shape (n,).
        periods_per_year: Number of periods in a year (annualises the Sharpe ratio).
        scale: Size of the cumulative wealth or P&L that the increments are
            differences of, such as its largest absolute value. A difference of
            values near `scale` carries rounding of order ``eps * scale``, so a
            standard deviation within that of zero counts as zero (and the Sharpe
            ratio is 0), as for returns. 0 (the default) treats the increments as
            exact; `RecordPerformance` passes the episode's largest absolute value.

    Returns:
        A `PnLSummary`.

    Raises:
        ValueError: If `pnl` is empty, not 1-D or not finite, `periods_per_year`
            is not positive, or `scale` is negative or not finite.
    """
    x = _as_series(pnl, "pnl")
    ppy = _check_positive(periods_per_year, "periods_per_year")
    sd = _sample_std(x, offset=_check_non_negative(scale, "scale"))
    mean = float(np.mean(x))
    cumulative = np.concatenate(([0.0], np.cumsum(x)))
    drawdown = np.maximum.accumulate(cumulative) - cumulative
    return PnLSummary(
        n_periods=int(x.size),
        total_pnl=float(np.sum(x)),
        mean_pnl=mean,
        pnl_volatility=sd,
        sharpe_ratio=0.0 if sd == 0.0 else mean / sd * math.sqrt(ppy),
        max_drawdown=float(np.max(drawdown)),
        hit_rate=_hit_rate(x),
    )


# ---------------------------------------------------------------------------
# Multiple testing and overfitting
# ---------------------------------------------------------------------------

_EULER_GAMMA = 0.5772156649015329

# Acklam's rational approximation to the inverse normal CDF (relative error below
# 1.15e-9): numerator and denominator coefficients, highest power first, for the
# central region (A, B) and the tails (C, D).
_PPF_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_PPF_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
    1.0,
)
_PPF_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_PPF_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
    1.0,
)
_PPF_TAIL = 0.02425  # below this probability, the tail approximation applies

# CSCV evaluates C(n_partitions, n_partitions / 2) combinations: 12,870 for 16
# partitions, 184,756 for 20, 2,704,156 for 24.
_MAX_PARTITIONS = 20

# Upper bound on the elements of each intermediate array in CSCV (8 MB of float64).
_CSCV_CHUNK = 1 << 20


def _horner(coefficients: tuple[float, ...], x: float) -> float:
    out = 0.0
    for c in coefficients:
        out = out * x + c
    return out


def _norm_cdf(x: float) -> float:
    """Standard normal CDF, with relative accuracy far into the lower tail.

    The rounding of ``x / sqrt(2)`` is magnified about ``x**2`` times there (1e-13 at -30).
    """
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF for 0 <= p <= 1 (``-inf`` at 0, ``inf`` at 1).

    Acklam's approximation (relative error 1.15e-9) refined by two Newton steps on
    `_norm_cdf`, accurate to an ulp or two. The upper half comes from the lower one by
    symmetry, so ``-_norm_ppf(q)`` is an accurate upper quantile even where ``1 - q``
    would round to 1.
    """
    if p > 0.5:
        return -_norm_ppf(1.0 - p)  # 1 - p is exact for p in [0.5, 1]
    if p <= 0.0:
        return -math.inf
    if p < _PPF_TAIL:
        q = math.sqrt(-2.0 * math.log(p))
        x = _horner(_PPF_C, q) / _horner(_PPF_D, q)
    else:
        q = p - 0.5
        r = q * q
        x = _horner(_PPF_A, r) * q / _horner(_PPF_B, r)
    for _ in range(2):  # the second step matters in the far tail, where |x| is large
        density = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        if density > 0.0:
            x -= (_norm_cdf(x) - p) / density
    return x


def _sharpe_moments(x: NDArray[np.float64]) -> tuple[float, float, float]:
    """Per-period Sharpe ratio (as `_sharpe_ratio` with no hurdle), skewness and kurtosis.

    Skewness and (non-excess) kurtosis are the moment estimators ``m3 / m2**1.5`` and
    ``m4 / m2**2``. Returns with zero standard deviation have a Sharpe ratio of 0, which
    makes the moments irrelevant; they are reported as a normal's, 0 and 3.
    """
    excess = _excess(x, 0.0)
    sd = _sample_std(excess)
    if sd == 0.0:
        return 0.0, 0.0, 3.0
    dev = excess - np.mean(excess)
    z = dev / math.sqrt(float(np.mean(dev * dev)))
    return float(np.mean(excess)) / sd, float(np.mean(z**3)), float(np.mean(z**4))


def _probabilistic_sharpe(
    sharpe: float, benchmark: float, n: int, skew: float, kurt: float
) -> float:
    """PSR from per-period Sharpe ratios, the number of returns and their moments."""
    diff = sharpe - benchmark
    if diff == 0.0:
        return 0.5
    # Never negative for sample moments (kurt >= skew**2 + 1); zero only for a
    # two-valued sample, whose estimated Sharpe ratio then has no spread.
    var = 1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe * sharpe
    if var <= 0.0:
        return 1.0 if diff > 0.0 else 0.0
    return _norm_cdf(diff * math.sqrt((n - 1) / var))


def _expected_maximum(n_trials: int, std: float) -> float:
    if n_trials == 1:
        return 0.0
    q = 1.0 / n_trials
    # -ppf(q) is the upper quantile ppf(1 - q), accurate for any number of trials.
    return std * ((1.0 - _EULER_GAMMA) * -_norm_ppf(q) + _EULER_GAMMA * -_norm_ppf(q / math.e))


def probabilistic_sharpe_ratio(
    returns: ArrayLike, benchmark_sharpe: float = 0.0, periods_per_year: float = 252
) -> float:
    """Probability that the true Sharpe ratio exceeds a benchmark.

    The probabilistic Sharpe ratio (PSR) of Bailey and López de Prado (2012) is
    ``Phi((sr - sr_b) * sqrt(n - 1) / sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2))``,
    where `sr` is the observed and `sr_b` the benchmark Sharpe ratio, both per period, `n`
    the number of returns, `skew` and `kurt` their skewness and (non-excess, 3 for a
    normal) kurtosis, and `Phi` the standard normal CDF. A short record, negative
    skewness and fat tails all make a positive Sharpe ratio less conclusive.

    `benchmark_sharpe` is annualised, like the result of `sharpe_ratio`, and is converted
    to the per-period ``benchmark_sharpe / sqrt(periods_per_year)``. The observed ratio is
    the per-period ``sharpe_ratio(returns, periods_per_year) / sqrt(periods_per_year)``,
    with no risk-free rate (pass excess returns for one). Skewness and kurtosis are the
    moment estimators ``m3 / m2**1.5`` and ``m4 / m2**2``, with
    ``m_k = mean((r - mean(r))**k)``.

    Args:
        returns: Simple per-period returns, shape (n,) with n >= 2.
        benchmark_sharpe: Annualised Sharpe ratio to beat; 0 asks whether there is any
            skill at all.
        periods_per_year: Number of return periods in a year (converts
            `benchmark_sharpe` to a per-period ratio).

    Returns:
        The PSR, in [0, 1]; 0.5 when the observed Sharpe ratio equals the benchmark.
        Returns with zero standard deviation have a Sharpe ratio of 0, as in
        `sharpe_ratio`.

    Raises:
        ValueError: If `returns` has fewer than two values, is not 1-D or not finite, or
            a parameter is invalid.

    Example:
        ```python
        rng = np.random.default_rng(0)
        returns = rng.normal(0.001, 0.01, size=756)  # three years of daily returns
        sharpe_ratio(returns)                                      # 1.171
        probabilistic_sharpe_ratio(returns)                        # 0.978: P[SR > 0]
        probabilistic_sharpe_ratio(returns, benchmark_sharpe=0.5)  # 0.876: P[SR > 0.5]
        ```
    """
    x = _as_track_record(returns, "returns")
    ppy = _check_positive(periods_per_year, "periods_per_year")
    benchmark = _check_finite(benchmark_sharpe, "benchmark_sharpe") / math.sqrt(ppy)
    sharpe, skew, kurt = _sharpe_moments(x)
    return _probabilistic_sharpe(sharpe, benchmark, x.size, skew, kurt)


def minimum_track_record_length(
    returns: ArrayLike,
    benchmark_sharpe: float = 0.0,
    confidence: float = 0.95,
    periods_per_year: float = 252,
) -> float:
    """Number of periods needed for the observed Sharpe ratio to beat a benchmark.

    The minimum track-record length (MinTRL) of Bailey and López de Prado (2012) is
    ``1 + (1 - skew * sr + (kurt - 1) / 4 * sr**2) * (Phi^-1(confidence) / (sr - sr_b))**2``,
    with the per-period Sharpe ratios, moments and benchmark conversion of
    `probabilistic_sharpe_ratio`: the number of returns at which a strategy with the
    observed Sharpe ratio, skewness and kurtosis has a PSR of `confidence`. A record is
    long enough if it has at least that many periods.

    Args:
        returns: Simple per-period returns, shape (n,) with n >= 2, from which the Sharpe
            ratio, skewness and kurtosis are estimated.
        benchmark_sharpe: Annualised Sharpe ratio to beat.
        confidence: Required probability that the true Sharpe ratio exceeds the
            benchmark, in [0.5, 1).
        periods_per_year: Number of return periods in a year (converts
            `benchmark_sharpe` to a per-period ratio).

    Returns:
        The MinTRL as a number of periods of `returns` (a float >= 1, not rounded up;
        divide by `periods_per_year` for years). ``math.inf`` if the observed Sharpe
        ratio does not exceed the benchmark.

    Raises:
        ValueError: If `returns` has fewer than two values, is not 1-D or not finite, or
            a parameter is invalid.

    Example:
        ```python
        rng = np.random.default_rng(0)
        returns = rng.normal(0.001, 0.01, size=756)
        minimum_track_record_length(returns) / 252  # 2.0 years for 95 % that SR > 0
        ```
    """
    x = _as_track_record(returns, "returns")
    ppy = _check_positive(periods_per_year, "periods_per_year")
    benchmark = _check_finite(benchmark_sharpe, "benchmark_sharpe") / math.sqrt(ppy)
    z = _norm_ppf(_check_confidence(confidence))
    sharpe, skew, kurt = _sharpe_moments(x)
    diff = sharpe - benchmark
    if diff <= 0.0:
        return math.inf
    var = max(1.0 - skew * sharpe + (kurt - 1.0) / 4.0 * sharpe * sharpe, 0.0)
    return 1.0 + var * (z / diff) ** 2


def expected_maximum_sharpe(n_trials: int, sharpe_std: float) -> float:
    """Expected best Sharpe ratio of `n_trials` independent trials without skill.

    ``sharpe_std * ((1 - g) * Phi^-1(1 - 1 / n_trials) + g * Phi^-1(1 - 1 / (n_trials * e)))``,
    with `g` the Euler-Mascheroni constant and ``Phi^-1`` the inverse standard normal CDF,
    is Bailey and López de Prado's (2014) approximation to the expected maximum of
    `n_trials` independent normal Sharpe-ratio estimates with true value zero and standard
    deviation `sharpe_std`: what the best of that many skill-less trials shows by luck
    alone. It is the benchmark of `deflated_sharpe_ratio`.

    The result is in the units of `sharpe_std`: annualised in, annualised out.

    Args:
        n_trials: Number of independent trials, an int >= 1.
        sharpe_std: Standard deviation of the trials' Sharpe ratios (non-negative).

    Returns:
        The expected maximum Sharpe ratio, increasing in `n_trials` and proportional to
        `sharpe_std`. 0.0 for a single trial, whose expected Sharpe ratio is zero (the
        approximation diverges there).

    Raises:
        ValueError: If `n_trials` is not an int >= 1, or `sharpe_std` is negative or not
            finite.

    Example:
        ```python
        expected_maximum_sharpe(100, 1.0)  # 2.531 (the exact expectation is 2.508)
        ```
    """
    n = _check_int(n_trials, "n_trials", 1)
    return _expected_maximum(n, _check_non_negative(sharpe_std, "sharpe_std"))


def deflated_sharpe_ratio(
    returns: ArrayLike,
    trial_sharpes: ArrayLike | None = None,
    periods_per_year: float = 252,
    *,
    n_trials: int | None = None,
    trial_sharpe_std: float | None = None,
) -> float:
    """Probabilistic Sharpe ratio against the best Sharpe ratio that luck explains.

    When `returns` belong to the best of many backtests (seeds, hyperparameters, rules),
    selection biases its Sharpe ratio upward. The deflated Sharpe ratio (DSR) of Bailey
    and López de Prado (2014) is
    `probabilistic_sharpe_ratio` with the benchmark set to `expected_maximum_sharpe` of
    the trials: the probability that the selected strategy's true Sharpe ratio exceeds
    what the best of that many skill-less trials would show.

    Describe the trials either by `trial_sharpes`, the annualised Sharpe ratios of every
    configuration tried (including the selected one), whose number and sample standard
    deviation (ddof=1) are used; or by `n_trials` and `trial_sharpe_std` directly, e.g.
    an effective number of independent trials when the trials are correlated. The
    expected maximum is annualised like the trial Sharpe ratios and de-annualised with
    `periods_per_year`, as `benchmark_sharpe` is in `probabilistic_sharpe_ratio`.

    Args:
        returns: Simple per-period returns of the selected strategy, shape (n,) with
            n >= 2.
        trial_sharpes: Annualised Sharpe ratios of all trials, shape (n_trials,).
        periods_per_year: Number of return periods in a year.
        n_trials: Number of independent trials, an int >= 1, instead of `trial_sharpes`.
        trial_sharpe_std: Standard deviation of the trials' annualised Sharpe ratios
            (non-negative), given with `n_trials`.

    Returns:
        The DSR, in [0, 1]. It falls as the number of trials or the spread of their
        Sharpe ratios grows; for a single trial (or trials with identical Sharpe ratios)
        it equals ``probabilistic_sharpe_ratio(returns)``.

    Raises:
        ValueError: If `trial_sharpes` is given together with `n_trials` or
            `trial_sharpe_std`, if neither `trial_sharpes` nor both of them are given, or
            if an input is invalid.

    Example:
        ```python
        rng = np.random.default_rng(1)
        trials = rng.normal(0.0, 0.01, size=(756, 50))  # 50 configurations, no skill
        sharpes = [sharpe_ratio(trials[:, j]) for j in range(50)]
        best = trials[:, int(np.argmax(sharpes))]
        probabilistic_sharpe_ratio(best)       # 0.984: looks skilful on its own
        deflated_sharpe_ratio(best, sharpes)   # 0.325: not once the 50 trials count
        ```
    """
    x = _as_track_record(returns, "returns")
    ppy = _check_positive(periods_per_year, "periods_per_year")
    if trial_sharpes is not None:
        if n_trials is not None or trial_sharpe_std is not None:
            raise ValueError("pass either trial_sharpes or n_trials and trial_sharpe_std, not both")
        trials = _as_series(trial_sharpes, "trial_sharpes")
        n, std = trials.size, _sample_std(trials, offset=0.0)
    elif n_trials is None or trial_sharpe_std is None:
        raise ValueError(
            "deflated_sharpe_ratio needs trial_sharpes, or both n_trials and trial_sharpe_std"
        )
    else:
        n = _check_int(n_trials, "n_trials", 1)
        std = _check_non_negative(trial_sharpe_std, "trial_sharpe_std")
    benchmark = _expected_maximum(n, std) / math.sqrt(ppy)
    sharpe, skew, kurt = _sharpe_moments(x)
    return _probabilistic_sharpe(sharpe, benchmark, x.size, skew, kurt)


@dataclass(frozen=True, slots=True, eq=False)
class OverfittingReport:
    """Result of `probability_of_backtest_overfitting` (CSCV).

    Each combination chooses half of the partitions as the in-sample set; the other half
    is its out-of-sample set. The arrays have one entry per combination, in the order of
    ``itertools.combinations(range(n_partitions), n_partitions // 2)`` for the
    in-sample partitions. Performance values are in the units of the `metric` (the
    per-period Sharpe ratio by default). When several strategies tie for the best
    in-sample performance, each counts as selected with equal probability, and the
    per-combination values are averages over them.

    Attributes:
        pbo: Probability of backtest overfitting: the fraction of combinations in which
            the in-sample best is at or below the out-of-sample median (logit <= 0).
        logits: ``log(w / (1 - w))`` for the out-of-sample relative rank
            ``w = rank / (n_strategies + 1)`` of the in-sample best (rank 1 is the
            worst; ties share their average rank), shape (n_combinations,).
        is_performance: In-sample performance of the in-sample best, shape
            (n_combinations,).
        oos_performance: Out-of-sample performance of the in-sample best, shape
            (n_combinations,).
        degradation_slope: Slope of the least-squares line of `oos_performance` on
            `is_performance`. Near or below zero when a better in-sample result does not
            mean a better out-of-sample one. NaN if `is_performance` is constant.
        degradation_intercept: Intercept of that line (NaN with the slope).
        probability_of_loss: Fraction of combinations in which the in-sample best has an
            out-of-sample performance below zero.
        n_partitions: Number of partitions of the periods.
        block_size: Periods per partition. The first
            ``n_periods - n_partitions * block_size`` periods were left out.
    """

    pbo: float
    logits: NDArray[np.float64]
    is_performance: NDArray[np.float64]
    oos_performance: NDArray[np.float64]
    degradation_slope: float
    degradation_intercept: float
    probability_of_loss: float
    n_partitions: int
    block_size: int

    @property
    def n_combinations(self) -> int:
        """Number of in-sample / out-of-sample combinations evaluated."""
        return int(self.logits.size)

    def __str__(self) -> str:
        return _table(
            [
                ("combinations", str(self.n_combinations)),
                ("probability of overfitting", _fmt_pct(self.pbo)),
                ("probability of loss", _fmt_pct(self.probability_of_loss)),
                ("degradation slope", _fmt_num(self.degradation_slope)),
                ("degradation intercept", _fmt_num(self.degradation_intercept)),
                ("partitions", str(self.n_partitions)),
                ("periods per partition", str(self.block_size)),
            ]
        )


def _midranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """1-based ranks along axis 1 of a 2-D array; tied values share their average rank."""
    rows, n = values.shape
    order = np.argsort(values, axis=1, kind="stable")
    ordered = np.take_along_axis(values, order, axis=1)
    position = np.broadcast_to(np.arange(n), (rows, n))
    starts = np.ones((rows, n), dtype=bool)  # first position of each run of equal values
    starts[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    ends = np.ones((rows, n), dtype=bool)  # last position of each run
    ends[:, :-1] = starts[:, 1:]
    first = np.maximum.accumulate(np.where(starts, position, 0), axis=1)
    last = np.minimum.accumulate(np.where(ends, position, n - 1)[:, ::-1], axis=1)[:, ::-1]
    ranks = np.empty((rows, n), dtype=np.float64)
    np.put_along_axis(ranks, order, (first + last) / 2.0 + 1.0, axis=1)
    return ranks


def _sum_over_axis1(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sum over axis 1 by elementwise additions, one slice at a time.

    Each element of the result is the same sequence of IEEE additions wherever its column
    sits. numpy's reductions do not promise that (they group elements differently by
    position), and the CSCV default metric must not depend on the order of the strategies.
    """
    total: NDArray[np.float64] = values[:, 0].copy()
    for i in range(1, values.shape[1]):
        total += values[:, i]
    return total


def _cscv_select(
    ins: NDArray[np.float64], oos: NDArray[np.float64]
) -> tuple[NDArray[np.float64], ...]:
    """The in-sample best strategy of each row and how it does out of sample.

    Returns its logit, overfitting (logit <= 0) and loss (out-of-sample value < 0)
    indicators, and its in-sample and out-of-sample performance, shape (rows,) each. Tied
    best strategies each count as selected with equal probability: the values are
    averages over them. Their ranks and values are summed in sorted order, so the
    floating-point results do not depend on the order of the strategies.
    """
    n = ins.shape[1]
    best = ins.max(axis=1)
    selected = ins == best[:, None]
    count = selected.sum(axis=1)
    midranks = _midranks(oos)  # half-integers, so comparisons and sums are exact
    ranks = np.sort(np.where(selected, midranks, (n + 1.0) / 2.0), axis=1)
    logits = (np.log(ranks) - np.log(n + 1.0 - ranks)).sum(axis=1)  # 0 at (n + 1) / 2
    overfit = (selected & (2.0 * midranks <= n + 1.0)).sum(axis=1)
    loss = (selected & (oos < 0.0)).sum(axis=1)
    oos_best = np.sort(np.where(selected, oos, 0.0), axis=1).sum(axis=1)
    return logits / count, overfit / count, loss / count, best, oos_best / count


def _cscv_sharpe(
    blocks: NDArray[np.float64],
) -> Callable[[NDArray[np.intp]], NDArray[np.float64]]:
    """Per-period Sharpe ratios of sets of blocks, combined from per-block statistics.

    `blocks` has shape (n_partitions, block_size, n_strategies). The returned function
    maps sets of block indices, shape (c, k), to the Sharpe ratio of each strategy over
    the union of each set's blocks, shape (c, n_strategies), as `sharpe_ratio` with
    ``periods_per_year=1`` computes it on the concatenated returns.
    """
    excess = _excess(blocks, 0.0)
    size = blocks.shape[1]
    means = _sum_over_axis1(excess) / size
    m2 = _sum_over_axis1((excess - means[:, None, :]) ** 2)
    scale = np.abs(excess).max(axis=1)

    def sharpe(chosen: NDArray[np.intp]) -> NDArray[np.float64]:
        block_means = means[chosen]  # (c, k, n_strategies)
        mean = _sum_over_axis1(block_means) / chosen.shape[1]
        rows = size * chosen.shape[1]
        if rows < 2:
            return np.zeros_like(mean)
        # Exact pooling of the blocks' sums of squared deviations (no cancellation).
        spread = _sum_over_axis1((block_means - mean[:, None, :]) ** 2)
        sd = np.sqrt((_sum_over_axis1(m2[chosen]) + size * spread) / (rows - 1))
        flat = sd <= _ROUNDING * (1.0 + scale[chosen].max(axis=1))
        out: NDArray[np.float64] = np.divide(
            mean, sd, out=np.zeros_like(mean), where=np.logical_not(flat)
        )
        return out

    return sharpe


def _cscv_metric(
    metric: Callable[[NDArray[np.float64]], ArrayLike], blocks: NDArray[np.float64]
) -> Callable[[NDArray[np.intp]], NDArray[np.float64]]:
    """A user metric evaluated on the rows of each set of blocks, in time order."""
    n = blocks.shape[2]

    def evaluate(chosen: NDArray[np.intp]) -> NDArray[np.float64]:
        out = np.empty((chosen.shape[0], n), dtype=np.float64)
        for row, half in enumerate(chosen):
            value = metric(blocks[half].reshape(-1, n))
            try:
                perf = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"metric must return an array of numbers: {exc}") from exc
            if perf.shape != (n,):
                raise ValueError(
                    f"metric must return one value per strategy, shape ({n},), "
                    f"got shape {perf.shape}"
                )
            if not np.all(np.isfinite(perf)):
                raise ValueError("metric must return finite values (no NaN or inf)")
            out[row] = perf
        return out

    return evaluate


def probability_of_backtest_overfitting(
    returns: ArrayLike,
    n_partitions: int = 16,
    metric: Callable[[NDArray[np.float64]], ArrayLike] | None = None,
) -> OverfittingReport:
    """Probability that the in-sample best of many strategies disappoints out of sample.

    Combinatorially symmetric cross-validation (CSCV; Bailey, Borwein, López de Prado and Zhu,
    "The Probability of Backtest Overfitting", 2015) splits the periods (rows) into
    `n_partitions` consecutive blocks of equal size. For each of the
    ``C(n_partitions, n_partitions / 2)`` ways to choose half the blocks as the in-sample
    set, with the other half as the out-of-sample set, it selects the strategy with the
    best in-sample `metric` and records the out-of-sample relative rank
    ``w = rank / (n_strategies + 1)`` of that strategy as a logit ``log(w / (1 - w))``.
    The probability of backtest overfitting (PBO) is the fraction of combinations whose
    logit is <= 0, i.e. in which the in-sample best is at or below the out-of-sample
    median. Near 0, choosing the backtest winner carries over to new data; near 0.5 or
    above, it is no better than choosing at random.

    In-sample and out-of-sample sets always have the same number of periods. If the
    number of periods is not a multiple of `n_partitions`, the first (oldest)
    ``n_periods % n_partitions`` periods are left out so that the blocks are equal;
    `OverfittingReport.block_size` gives the periods per block.

    The result does not depend on the order of the strategies: when several share the
    best in-sample performance, each counts as selected with equal probability (their
    logits, overfitting indicators and out-of-sample performance are averaged), and
    strategies tied out of sample share their average rank.

    Args:
        returns: Per-period returns of every strategy over the same periods, shape
            (n_periods, n_strategies), with n_periods >= `n_partitions` and
            n_strategies >= 2. Columns are the configurations tried (seeds,
            hyperparameters, rules).
        n_partitions: Number of blocks, an even int from 2 to 20. The default 16 gives
            C(16, 8) = 12,870 combinations; each block should still hold enough periods
            for the metric to be meaningful.
        metric: Performance of every strategy over a set of periods, higher is better: a
            callable that takes the rows of the chosen blocks in time order, shape
            (t, n_strategies), and returns shape (n_strategies,) of finite floats.
            Defaults to the per-period Sharpe ratio of each column,
            ``mean / std(ddof=1)`` (0 for a constant column), as `sharpe_ratio` with
            ``periods_per_year=1`` computes it; this default is evaluated from per-block
            statistics, much faster than an equivalent callable.

    Returns:
        An `OverfittingReport` with the PBO, the logits, the in-sample and out-of-sample
        performance of the selected strategy per combination, the slope of the
        performance degradation and the probability of an out-of-sample loss.

    Raises:
        ValueError: If `returns` is not a finite 2-D array with at least two strategies
            and `n_partitions` periods, if `n_partitions` is not an even int from 2 to
            20, or if `metric` returns the wrong shape or non-finite values.
        TypeError: If `metric` is neither callable nor None.

    Example:
        ```python
        rng = np.random.default_rng(0)
        noise = rng.normal(0.0, 0.01, size=(1000, 20))  # 20 skill-less strategies
        probability_of_backtest_overfitting(noise).pbo  # 0.416: close to a coin flip
        ```
    """
    x = _as_panel(returns, "returns")
    s = _check_int(n_partitions, "n_partitions", 2)
    if s % 2:
        raise ValueError(f"n_partitions must be even, got {s}")
    if s > _MAX_PARTITIONS:
        raise ValueError(
            f"n_partitions={s} gives {math.comb(s, s // 2):,} combinations; at most "
            f"{_MAX_PARTITIONS} partitions ({math.comb(_MAX_PARTITIONS, _MAX_PARTITIONS // 2):,}"
            " combinations) are supported"
        )
    if metric is not None and not callable(metric):
        raise TypeError(f"metric must be callable or None, got {type(metric).__name__}")
    n_periods, n = x.shape
    if n_periods < s:
        raise ValueError(f"returns has {n_periods} periods, fewer than n_partitions={s}")
    size = n_periods // s
    blocks = x[n_periods - s * size :].reshape(s, size, n)

    # In lexicographic order, the complement of combination i is combination
    # total - 1 - i (complementing reverses the order), so pairing i < total / 2 with its
    # complement covers every combination once in sample and once out of sample.
    halves = np.array(list(itertools.combinations(range(s), s // 2)), dtype=np.intp)
    total = halves.shape[0]
    perf = _cscv_sharpe(blocks) if metric is None else _cscv_metric(metric, blocks)

    # Per combination: logit, overfitting and loss indicators, in- and out-of-sample
    # performance of the in-sample best.
    results = np.empty((5, total), dtype=np.float64)
    chunk = max(1, _CSCV_CHUNK // (halves.shape[1] * n))
    for start in range(0, total // 2, chunk):
        first = np.arange(start, min(start + chunk, total // 2))
        perf_first, perf_second = perf(halves[first]), perf(halves[total - 1 - first])
        ins = np.concatenate((perf_first, perf_second))
        oos = np.concatenate((perf_second, perf_first))
        results[:, np.concatenate((first, total - 1 - first))] = _cscv_select(ins, oos)
    logits, overfit, loss, is_perf, oos_perf = results

    if _sample_std(is_perf, offset=0.0) == 0.0:
        slope = intercept = math.nan
    else:
        is_mean, oos_mean = float(np.mean(is_perf)), float(np.mean(oos_perf))
        dev = is_perf - is_mean
        slope = float(dev @ (oos_perf - oos_mean)) / float(dev @ dev)
        intercept = oos_mean - slope * is_mean
    return OverfittingReport(
        pbo=float(np.mean(overfit)),
        logits=logits,
        is_performance=is_perf,
        oos_performance=oos_perf,
        degradation_slope=slope,
        degradation_intercept=intercept,
        probability_of_loss=float(np.mean(loss)),
        n_partitions=s,
        block_size=size,
    )


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class RecordPerformance(Wrapper[ObsType, ActType, ObsType, ActType]):
    """Reports trading performance metrics at the end of every episode.

    **Equity mode** (default, `pnl_key` None) reads account equity from
    ``info[equity_key]`` on every reset and step. Each step's simple return is
    ``equity_t / equity_{t-1} - 1``, where the previous equity is the last
    step's or, on the first step, the reset info's. If the reset info has no
    `equity_key`, the starting equity is assumed to be 1.0. A step that takes
    equity to zero or below is a wipe-out: its return is -1 (-100 %) and the
    series ends there, i.e. every later return in the episode is 0.

    **P&L mode** (`pnl_key` given) reads cumulative wealth or P&L in absolute
    units from ``info[pnl_key]``; it may start at zero or be negative. Each
    step's P&L is the change since the previous value (from the reset info,
    else 0.0), and episodes are summarised with `pnl_summary`, with the
    episode's largest absolute value as the rounding `scale` (a P&L that is
    constant up to the rounding of the wealth level has zero volatility).

    In equity mode, ``info[turnover_key]`` and ``info[cost_key]`` are collected
    from the steps that report them and summarised as the mean turnover and the
    total cost (None if no step reported them). P&L mode ignores them.

    On the final step of an episode, the summary is attached to that step's
    info as ``info["performance"]`` (in a new dict; the inner info is never
    modified), appended to `summaries` and exposed as `last_summary`, and a new
    episode starts. Around an inner `AutoReset`, the terminal value is read
    from ``info["final_info"]`` and the next episode's starting value from
    ``info["reset_info"]``, so back-to-back episodes are summarised correctly
    without `reset()` reaching this wrapper. A `reset()` mid-episode discards
    the unfinished episode.

    Args:
        env: Environment whose info reports equity (or cumulative P&L).
        periods_per_year: Steps per year, for annualisation.
        equity_key: Info key holding account equity (equity mode).
        pnl_key: Info key holding cumulative P&L; selects P&L mode when given.
        turnover_key: Info key holding per-step turnover, or None to ignore.
        cost_key: Info key holding per-step trading cost, or None to ignore.
        risk_free: Annual simple risk-free rate for the Sharpe and Sortino ratios.
        deque_size: Number of recent episode summaries kept in `summaries`.

    Raises:
        ValueError: If a numeric argument is out of range.
        TypeError: If a key argument is not a string (or None where allowed).
    """

    def __init__(
        self,
        env: Env[ObsType, ActType],
        *,
        periods_per_year: float = 252,
        equity_key: str = "equity",
        pnl_key: str | None = None,
        turnover_key: str | None = "turnover",
        cost_key: str | None = "cost",
        risk_free: float = 0.0,
        deque_size: int = 100,
    ) -> None:
        super().__init__(env)
        if not isinstance(equity_key, str):
            raise TypeError(f"equity_key must be a str, got {type(equity_key).__name__}")
        for name, key in (
            ("pnl_key", pnl_key),
            ("turnover_key", turnover_key),
            ("cost_key", cost_key),
        ):
            if key is not None and not isinstance(key, str):
                raise TypeError(f"{name} must be a str or None, got {type(key).__name__}")
        if isinstance(deque_size, bool) or not isinstance(deque_size, int) or deque_size < 1:
            raise ValueError(f"deque_size must be a positive int, got {deque_size!r}")
        self.periods_per_year = _check_positive(periods_per_year, "periods_per_year")
        self.equity_key = equity_key
        self.pnl_key = pnl_key
        self.turnover_key = turnover_key
        self.cost_key = cost_key
        self.risk_free = _check_finite(risk_free, "risk_free")
        self.summaries: deque[PerformanceSummary | PnLSummary] = deque(maxlen=deque_size)
        self.last_summary: PerformanceSummary | PnLSummary | None = None

        self._prev: float = 0.0
        self._level: float = 0.0  # largest |value| of the episode (P&L mode rounding)
        self._wiped: bool = False
        self._flows: list[float] = []
        self._turnover: list[float] = []
        self._costs: list[float] = []
        self._start_episode(None)

    # ------------------------------------------------------------------
    # Episode bookkeeping (shared by the sync and async paths)
    # ------------------------------------------------------------------

    @property
    def _value_key(self) -> str:
        return self.equity_key if self.pnl_key is None else self.pnl_key

    @staticmethod
    def _read(info: Mapping[str, Any], key: str) -> float:
        if key not in info:
            raise KeyError(
                f"RecordPerformance expected info[{key!r}] in every step result; "
                f"available keys: {list(info)}"
            )
        try:
            value = float(info[key])
        except (TypeError, ValueError) as exc:
            raise TypeError(f"info[{key!r}] must be a real number, got {info[key]!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"info[{key!r}] must be finite, got {value}")
        return value

    def _start_episode(self, info: Mapping[str, Any] | None) -> None:
        key = self._value_key
        if info is not None and key in info:
            self._prev = self._read(info, key)
        else:
            self._prev = 0.0 if self.pnl_key is not None else 1.0
        self._level = abs(self._prev)
        self._wiped = self.pnl_key is None and self._prev <= 0.0
        self._flows = []
        self._turnover = []
        self._costs = []

    def _record(self, info: Mapping[str, Any]) -> None:
        value = self._read(info, self._value_key)
        if self.pnl_key is not None:
            self._flows.append(value - self._prev)
            self._level = max(self._level, abs(value))
        elif self._wiped:
            self._flows.append(0.0)
        elif value <= 0.0:
            self._flows.append(-1.0)
            self._wiped = True
        else:
            self._flows.append(value / self._prev - 1.0)
        self._prev = value

        if self.pnl_key is None:
            if self.turnover_key is not None and self.turnover_key in info:
                self._turnover.append(self._read(info, self.turnover_key))
            if self.cost_key is not None and self.cost_key in info:
                self._costs.append(self._read(info, self.cost_key))

    def _summarise(self) -> PerformanceSummary | PnLSummary:
        flows = np.asarray(self._flows, dtype=np.float64)
        if self.pnl_key is not None:
            return pnl_summary(flows, self.periods_per_year, scale=self._level)
        return performance_summary(
            flows,
            self.periods_per_year,
            risk_free=self.risk_free,
            turnover=self._turnover or None,
            costs=self._costs or None,
        )

    def _on_reset(self, result: ResetResult[ObsType]) -> ResetResult[ObsType]:
        self._start_episode(result.info)
        return result

    def _on_step(self, result: StepResult[ObsType]) -> StepResult[ObsType]:
        info = result.info
        final_info = info.get("final_info") if result.done else None
        self._record(final_info if isinstance(final_info, Mapping) else info)
        if not result.done:
            return result

        summary = self._summarise()
        self.summaries.append(summary)
        self.last_summary = summary
        reset_info = info.get("reset_info")
        self._start_episode(reset_info if isinstance(reset_info, Mapping) else None)
        return StepResult(
            observation=result.observation,
            reward=result.reward,
            terminated=result.terminated,
            truncated=result.truncated,
            info={**info, "performance": summary},
        )

    def summary(self) -> dict[str, float]:
        """Mean of each numeric summary field over the stored episodes.

        Returns:
            Field name to mean value across `summaries`; fields that are None in
            every stored summary are left out. Empty if no episode has finished.
        """
        if not self.summaries:
            return {}
        out: dict[str, float] = {}
        for f in fields(self.summaries[0]):
            values = [getattr(s, f.name, None) for s in self.summaries]
            present = [float(v) for v in values if v is not None]
            if present:
                out[f.name] = sum(present) / len(present)
        return out

    # ------------------------------------------------------------------
    # Env API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[ObsType]:
        return self._on_reset(self.env.reset(seed=seed, options=options))

    async def async_reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[ObsType]:
        return self._on_reset(await self.env.async_reset(seed=seed, options=options))

    def step(self, action: ActType) -> StepResult[ObsType]:
        return self._on_step(self.env.step(action))

    async def async_step(self, action: ActType) -> StepResult[ObsType]:
        return self._on_step(await self.env.async_step(action))
