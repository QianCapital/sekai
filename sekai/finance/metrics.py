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

`RecordPerformance` wraps an env whose info reports account equity (or
cumulative P&L) and attaches a `PerformanceSummary` (or `PnLSummary`) to the
info of each episode's final step.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
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


def _check_level(level: float) -> float:
    v = float(level)
    if not 0.0 < v < 1.0:
        raise ValueError(f"level must lie strictly between 0 and 1, got {level!r}")
    return v


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
    (`offset=1`); P&L in absolute units has only relative rounding (`offset=0`).
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
        tail = x[x == x.min()]
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
        equity: Equity path, shape (n + 1,) with n >= 1. Every value but the last
            must be positive; a last value at or below zero is a wipe-out and gives
            a return of exactly -1 (-100 %).

    Returns:
        Simple returns ``equity[t + 1] / equity[t] - 1``, shape (n,).

    Raises:
        ValueError: If `equity` has fewer than two values, is not 1-D or finite,
            or has a non-positive value before the last.
    """
    e = _as_series(equity, "equity")
    if e.size < 2:
        raise ValueError("equity needs at least two values to define a return")
    if bool(np.any(e[:-1] <= 0.0)):
        raise ValueError("equity must be positive; only the last value may be <= 0 (a wipe-out)")
    r = e[1:] / e[:-1] - 1.0
    if e[-1] <= 0.0:
        r[-1] = -1.0
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


def pnl_summary(pnl: ArrayLike, periods_per_year: float = 252) -> PnLSummary:
    """Summarise a series of per-period P&L increments.

    Args:
        pnl: Per-period P&L increments in absolute units, shape (n,).
        periods_per_year: Number of periods in a year (annualises the Sharpe ratio).

    Returns:
        A `PnLSummary`.

    Raises:
        ValueError: If `pnl` is empty, not 1-D or not finite, or
            `periods_per_year` is not positive.
    """
    x = _as_series(pnl, "pnl")
    ppy = _check_positive(periods_per_year, "periods_per_year")
    sd = _sample_std(x, offset=0.0)
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
    else 0.0), and episodes are summarised with `pnl_summary`.

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
        self._wiped = self.pnl_key is None and self._prev <= 0.0
        self._flows = []
        self._turnover = []
        self._costs = []

    def _record(self, info: Mapping[str, Any]) -> None:
        value = self._read(info, self._value_key)
        if self.pnl_key is not None:
            self._flows.append(value - self._prev)
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
            return pnl_summary(flows, self.periods_per_year)
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
