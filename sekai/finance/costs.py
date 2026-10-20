"""Transaction-cost and market-impact models.

A cost model prices a rebalance: given the change in each asset's portfolio
weight (as a fraction of equity), it returns the cost as a fraction of equity.
Models compose with `+`, so a realistic schedule reads like its definition:

```python
costs = SpreadCost(half_spread=2e-4) + SquareRootImpact(0.1) + FixedCost(1.0)
```

Everything is vectorised: trades may carry leading batch dimensions (one row
per sub-env of a native vector env), and the result has those dimensions.
Context that some models need (account equity in currency, per-asset
volatility, traded volume, quoted spread) is passed as keyword arguments;
environments supply what their data has, and a model that is missing an input
raises a clear error. A model never prices a trade at zero because its input is
missing: an asset without a valid value is charged the cross-sectional median, and
when no asset has one the model raises, unless it was given an explicit fallback
(`SpreadCost(fallback_spread=...)`, `SquareRootImpact(fallback_volatility=...)`).
A model's own parameters (rates, spreads, fees, thresholds, coefficients,
fallbacks) must be finite and non-negative; anything else raises `ValueError`
when the model is built.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


class CostModel(abc.ABC):
    """Cost of trading `trades` (weight changes, fraction of equity) as a fraction of equity."""

    #: Context inputs the model reads (environments compute only these).
    requires: frozenset[str] = frozenset()
    #: Inputs for which the model has an explicit fallback value, used when no asset
    #: has a valid value (so an environment that cannot estimate them may still run).
    fallbacks: frozenset[str] = frozenset()

    @abc.abstractmethod
    def __call__(
        self,
        trades: ArrayLike,
        *,
        equity: ArrayLike | None = None,
        volatility: ArrayLike | None = None,
        volume: ArrayLike | None = None,
        spread: ArrayLike | None = None,
    ) -> Array:
        """Cost of a rebalance.

        Args:
            trades: `(..., n)` signed weight changes per asset.
            equity: `(...)` account value in currency (for fixed and impact costs).
            volatility: `(..., n)` per-step return volatility of each asset.
            volume: `(..., n)` traded value per step of each asset, in currency.
            spread: `(..., n)` quoted bid-ask spread as a fraction of the price.

        Returns:
            `(...)` cost as a fraction of equity.
        """

    def __add__(self, other: CostModel) -> CompositeCost:
        if not isinstance(other, CostModel):
            return NotImplemented
        return CompositeCost([self, other])


def _fill_from_median(
    values: Array,
    traded: NDArray[np.bool_],
    *,
    fallback: float | None,
    name: str,
    model: str,
    option: str,
) -> Array:
    """Replace missing (NaN or infinite) and negative entries with the median of the valid ones.

    Missing data for one asset (a just-listed name with no history, a gap in a
    spread series) then costs like a typical asset instead of nothing. A finite zero
    is a valid value (a flat window has zero volatility, a locked market zero spread)
    and is kept. A row in which no asset has a valid value uses `fallback`; without
    one, trading on such a row raises, since the trade would otherwise be free.

    Args:
        values: `(..., n)` input values, broadcast to the shape of the trades.
        traded: `(..., n)` True where an asset is traded (its value is needed).
        fallback: Value for rows without any valid entry, or None to raise.
        name: Name of the input, for the error message.
        model: Name of the model, for the error message.
        option: Name of the model's fallback argument, for the error message.
    """
    ok = np.isfinite(values) & (values >= 0)
    if ok.all():
        return values
    any_ok = np.asarray(ok.any(axis=-1))
    if fallback is None and np.any(~any_ok & traded.any(axis=-1)):
        raise ValueError(
            f"{model} has no valid {name!r} (finite and >= 0) for any asset in a row with "
            f"trades, so it cannot price them (they would be free). Supply the input, or "
            f"set an explicit fallback value with {model}(..., {option}=...)."
        )
    with np.errstate(invalid="ignore"):
        filled = np.where(ok, values, np.nan)
        median = np.nanmedian(np.where(any_ok[..., None], filled, 0.0), axis=-1)
    # Rows without a valid entry take the fallback (or, untraded, any placeholder).
    median = np.where(any_ok, median, 0.0 if fallback is None else fallback)
    return np.where(ok, values, np.asarray(median, dtype=np.float64)[..., None])


def _non_negative(value: ArrayLike, name: str) -> Array:
    """A model parameter as a float array, refusing NaN, infinite and negative values."""
    out = np.asarray(value, dtype=np.float64)
    if not (np.all(np.isfinite(out)) and np.all(out >= 0)):
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}.")
    return out


def _non_negative_scalar(value: float, name: str) -> float:
    if np.ndim(value) != 0:
        raise TypeError(f"{name} must be a single number, got {value!r}.")
    return float(_non_negative(value, name))


def _check_fallback(value: float | None, name: str) -> float | None:
    return None if value is None else _non_negative_scalar(value, name)


def _require(name: str, value: ArrayLike | None, model: str) -> Array:
    if value is None:
        raise ValueError(
            f"{model} needs {name!r}; this environment does not provide it. Add a "
            f"{name!r} field to the market data or choose another cost model."
        )
    return np.asarray(value, dtype=np.float64)


class NoCost(CostModel):
    """Frictionless trading."""

    def __call__(self, trades: ArrayLike, **context: Any) -> Array:
        t = np.asarray(trades, dtype=np.float64)
        return np.zeros(t.shape[:-1])

    def __repr__(self) -> str:
        return "NoCost()"


class ProportionalCost(CostModel):
    """`rate * |trade|` per asset: commissions and fees proportional to traded value.

    Args:
        rate: Cost per unit of traded value; a scalar or one rate per asset (finite,
            >= 0).
    """

    def __init__(self, rate: ArrayLike) -> None:
        self.rate = _non_negative(rate, "rate")

    def __call__(self, trades: ArrayLike, **context: Any) -> Array:
        t = np.abs(np.asarray(trades, dtype=np.float64))
        if self.rate.ndim == 0:
            out: Array = self.rate * t.sum(axis=-1)
            return out
        return np.asarray((self.rate * t).sum(axis=-1), dtype=np.float64)

    def __repr__(self) -> str:
        return f"ProportionalCost({self.rate.tolist()})"


class SpreadCost(CostModel):
    """Crossing half the bid-ask spread on every trade.

    Args:
        half_spread: Half-spread as a fraction of the price (scalar or per asset;
            finite, >= 0). If None, the env's quoted `spread` (full spread, as a
            fraction of the price) is used instead; an asset whose spread is missing
            (NaN) or negative is charged the cross-sectional median spread, while a
            zero spread is valid and costs nothing.
        fallback_spread: Full spread (fraction of the price) charged when no asset
            has a valid quoted spread. Without it, trading on such a row raises
            `ValueError` rather than trading for free. Only used with quoted spreads.
    """

    def __init__(
        self, half_spread: ArrayLike | None = None, *, fallback_spread: float | None = None
    ) -> None:
        self.half_spread = (
            None if half_spread is None else _non_negative(half_spread, "half_spread")
        )
        if self.half_spread is not None and fallback_spread is not None:
            raise ValueError("fallback_spread only applies to quoted spreads (half_spread=None).")
        self.fallback_spread = _check_fallback(fallback_spread, "fallback_spread")
        self.requires = frozenset() if self.half_spread is not None else frozenset({"spread"})
        self.fallbacks = frozenset() if self.fallback_spread is None else frozenset({"spread"})

    def __call__(
        self, trades: ArrayLike, *, spread: ArrayLike | None = None, **context: Any
    ) -> Array:
        t = np.abs(np.asarray(trades, dtype=np.float64))
        if self.half_spread is not None:
            half = self.half_spread
        else:
            quoted = np.broadcast_to(_require("spread", spread, "SpreadCost()"), t.shape)
            half = 0.5 * _fill_from_median(
                quoted.astype(np.float64),
                t > 0,
                fallback=self.fallback_spread,
                name="spread",
                model="SpreadCost",
                option="fallback_spread",
            )
        return np.asarray((half * t).sum(axis=-1), dtype=np.float64)

    def __repr__(self) -> str:
        hs = "from data" if self.half_spread is None else self.half_spread.tolist()
        extra = "" if self.fallback_spread is None else f", fallback_spread={self.fallback_spread}"
        return f"SpreadCost(half_spread={hs}{extra})"


class FixedCost(CostModel):
    """A fixed fee in currency for every asset traded (a ticket charge).

    Args:
        amount: Fee per traded asset, in the same currency as the env's capital
            (finite, >= 0).
        threshold: Weight changes at or below this size are not counted as trades
            (finite, >= 0; the default is above float32 rounding, so echoing back
            observed weights is free).
    """

    def __init__(self, amount: float, *, threshold: float = 1e-6) -> None:
        self.requires = frozenset({"equity"})
        self.amount = _non_negative_scalar(amount, "amount")
        self.threshold = _non_negative_scalar(threshold, "threshold")

    def __call__(
        self, trades: ArrayLike, *, equity: ArrayLike | None = None, **context: Any
    ) -> Array:
        t = np.abs(np.asarray(trades, dtype=np.float64))
        value = _require("equity", equity, "FixedCost")
        n_trades = (t > self.threshold).sum(axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(value > 0, self.amount * n_trades / np.maximum(value, 1e-300), 0.0)
        return np.asarray(out, dtype=np.float64)

    def __repr__(self) -> str:
        return f"FixedCost({self.amount})"


class SquareRootImpact(CostModel):
    """Square-root market impact: `coefficient * vol * sqrt(traded / volume)` per unit traded.

    The empirical "square-root law" of impact (e.g. Almgren et al. 2005; Tóth
    et al. 2011): trading a fraction `q / V` of an asset's volume moves its
    price by about `coefficient * sigma * sqrt(q / V)`, and the trade pays
    that on its whole size, so the cost grows as `|q|^1.5`. Large rebalances in
    illiquid names become expensive, which a proportional cost cannot express.

    Needs `equity` (currency), `volatility` (per-step) and `volume` (traded
    value per step, currency) from the environment. The impact per unit traded
    is capped at 1 (the whole traded value), which is also what trading an asset
    with zero or missing volume costs: there is no liquidity to trade against. A
    missing (NaN) volatility, e.g. a name with no return history yet, is replaced
    by the cross-sectional median of the other assets; a zero volatility (a flat
    window) is a valid estimate and gives zero impact. When no asset has a
    volatility, trading raises `ValueError` unless `fallback_volatility` is given.
    Environments refuse this model with `lookback=1`, which gives no estimate.

    Args:
        coefficient: Impact coefficient (finite, >= 0; order 0.1-1 in empirical studies).
        fallback_volatility: Per-step volatility used when no asset has a valid
            volatility estimate. Without it such trades raise instead of being free.
    """

    def __init__(
        self, coefficient: float = 0.5, *, fallback_volatility: float | None = None
    ) -> None:
        self.requires = frozenset({"equity", "volatility", "volume"})
        self.coefficient = _non_negative_scalar(coefficient, "coefficient")
        self.fallback_volatility = _check_fallback(fallback_volatility, "fallback_volatility")
        self.fallbacks = (
            frozenset() if self.fallback_volatility is None else frozenset({"volatility"})
        )

    def __call__(
        self,
        trades: ArrayLike,
        *,
        equity: ArrayLike | None = None,
        volatility: ArrayLike | None = None,
        volume: ArrayLike | None = None,
        **context: Any,
    ) -> Array:
        t = np.abs(np.asarray(trades, dtype=np.float64))
        value = _require("equity", equity, "SquareRootImpact")
        vol = _require("volatility", volatility, "SquareRootImpact")
        traded = _require("volume", volume, "SquareRootImpact")
        notional = t * np.asarray(value, dtype=np.float64)[..., None]
        sigma = _fill_from_median(
            np.broadcast_to(vol, t.shape).astype(np.float64),
            t > 0,
            fallback=self.fallback_volatility,
            name="volatility",
            model="SquareRootImpact",
            option="fallback_volatility",
        )
        liquid = np.isfinite(traded) & (traded > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            participation = np.where(liquid, notional / np.where(liquid, traded, 1.0), 0.0)
            unit = np.where(liquid, self.coefficient * sigma * np.sqrt(participation), 1.0)
        per_asset = np.where(t > 0, np.minimum(unit, 1.0) * t, 0.0)
        return np.asarray(per_asset.sum(axis=-1), dtype=np.float64)

    def __repr__(self) -> str:
        if self.fallback_volatility is None:
            return f"SquareRootImpact({self.coefficient})"
        fallback = self.fallback_volatility
        return f"SquareRootImpact({self.coefficient}, fallback_volatility={fallback})"


class CompositeCost(CostModel):
    """The sum of several cost models (build it with `+`)."""

    models: list[CostModel]

    def __init__(self, models: Sequence[CostModel]) -> None:
        flat: list[CostModel] = []
        for model in models:
            flat.extend(model.models if isinstance(model, CompositeCost) else [model])
        self.models = flat
        self.requires = frozenset().union(*(m.requires for m in flat))
        # An input has a fallback only if every model that reads it has one.
        self.fallbacks = frozenset(
            name
            for name in self.requires
            if all(name in m.fallbacks for m in flat if name in m.requires)
        )

    def __call__(self, trades: ArrayLike, **context: Any) -> Array:
        total: Array = np.asarray(self.models[0](trades, **context), dtype=np.float64)
        for model in self.models[1:]:
            total = total + model(trades, **context)
        return total

    def __repr__(self) -> str:
        return " + ".join(repr(m) for m in self.models)


def as_cost_model(cost: float | CostModel) -> CostModel:
    """A cost model from a CostModel or a proportional rate."""
    if isinstance(cost, CostModel):
        return cost
    if isinstance(cost, (int, float, np.integer, np.floating)) and not isinstance(cost, bool):
        return ProportionalCost(_non_negative_scalar(cost, "transaction_cost"))
    raise TypeError(f"Expected a CostModel or a non-negative rate, got {cost!r}.")


__all__ = [
    "CostModel",
    "NoCost",
    "ProportionalCost",
    "SpreadCost",
    "FixedCost",
    "SquareRootImpact",
    "CompositeCost",
    "as_cost_model",
]
