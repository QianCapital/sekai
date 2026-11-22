"""Transaction-cost and market-impact models.

A cost model prices a rebalance: given the change in each asset's portfolio
weight (as a fraction of equity), it returns the cost as a fraction of equity.
Models compose with `+`, so a realistic schedule reads like its definition:

```python
costs = SpreadCost(half_spread=2e-4) + SquareRootImpact(0.1) + FixedCost(1.0)
```

Everything is vectorised: trades may carry leading batch dimensions (one row
per sub-env of a vector env), and the result has those dimensions. Models run on
any `sekai.backend` (NumPy arrays, or torch tensors on any device, differentiable in
the trades), so a batched environment trains through them.
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
import functools
import inspect
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sekai.backend import Array, Backend, backend_and_like
from sekai.backend._base import _CHECKS


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
        assets: ArrayLike | None = None,
    ) -> Array:
        """Cost of a rebalance.

        Args:
            trades: `(..., n)` signed weight changes per asset.
            equity: `(...)` account value in currency (for fixed and impact costs).
            volatility: `(..., n)` per-step return volatility of each asset.
            volume: `(..., n)` traded value per step of each asset, in currency.
            spread: `(..., n)` quoted bid-ask spread as a fraction of the price.
            assets: `(..., n)` the asset (data column) behind each entry of `trades`,
                when trades are given per slot of a changing universe rather than per
                asset; models with per-asset parameters look them up with it. None
                means entry i is asset i.

        Returns:
            `(...)` cost as a fraction of equity.
        """

    def __add__(self, other: CostModel) -> CompositeCost:
        if not isinstance(other, CostModel):
            return NotImplemented
        return CompositeCost([self, other])


def _fill_from_median(
    B: Backend,
    values: Array,
    traded: Array,
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
        B: The backend of `values`.
        values: `(..., n)` input values, broadcast to the shape of the trades.
        traded: `(..., n)` True where an asset is traded (its value is needed).
        fallback: Value for rows without any valid entry, or None to raise.
        name: Name of the input, for the error message.
        model: Name of the model, for the error message.
        option: Name of the model's fallback argument, for the error message.
    """
    ok = B.isfinite(values) & (values >= 0)
    checks = _CHECKS.get()
    if checks is None and bool(B.all(ok)):  # (collected checks skip it: the fill is the same)
        return values
    any_ok = B.any(ok, axis=-1)
    if fallback is None:
        unpriced = B.any(~any_ok & B.any(traded, axis=-1))
        message = (
            f"{model} has no valid {name!r} (finite and >= 0) for any asset in a row with "
            f"trades, so it cannot price them (they would be free). Supply the input, or "
            f"set an explicit fallback value with {model}(..., {option}=...)."
        )
        if checks is not None:
            checks.append((unpriced, functools.partial(ValueError, message)))
        elif bool(unpriced):
            raise ValueError(message)
    with B.errstate():
        filled = B.where(ok, values, float("nan"))
        median = B.nanmedian(B.where(any_ok[..., None], filled, 0.0), axis=-1)
    # Rows without a valid entry take the fallback (or, untraded, any placeholder).
    median = B.where(any_ok, median, 0.0 if fallback is None else fallback)
    return B.where(ok, values, B.asfloat(median, values)[..., None])


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


def _require(B: Backend, name: str, value: ArrayLike | None, model: str, like: Array) -> Array:
    if value is None:
        raise ValueError(
            f"{model} needs {name!r}; this environment does not provide it. Add a "
            f"{name!r} field to the market data or choose another cost model."
        )
    return B.asfloat(value, like)


def _per_asset(
    B: Backend, param: NDArray[np.float64], assets: ArrayLike | None, like: Array
) -> Array:
    """A model parameter on the backend of `like`, looked up per slot when `assets` is given."""
    out = B.asfloat(param, like)
    if param.ndim == 0 or assets is None:
        return out
    # Empty slots (index -1) trade nothing, so the value looked up for them is irrelevant.
    return out[B.asint(assets, like)]


def _accepts(fn: Any, name: str) -> bool:
    """Whether calling `fn` with the keyword argument `name` is allowed."""
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name or p.kind is inspect.Parameter.VAR_KEYWORD for p in params)


# Whether a cost model class takes `assets=` (sekai 0.2's models don't), per class.
_TAKES_ASSETS: dict[type, bool] = {}


def _takes_assets(model: Any) -> bool:
    kind = type(model)
    takes = _TAKES_ASSETS.get(kind)
    if takes is None:
        takes = _TAKES_ASSETS[kind] = _accepts(model, "assets")
    return takes


class NoCost(CostModel):
    """Frictionless trading."""

    def __call__(self, trades: ArrayLike, **context: Any) -> Array:
        B, like = backend_and_like(trades)
        t = B.asfloat(trades, like)
        return B.zeros(tuple(t.shape[:-1]), like=t)

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

    def __call__(
        self, trades: ArrayLike, *, assets: ArrayLike | None = None, **context: Any
    ) -> Array:
        B, like = backend_and_like(trades)
        t = B.abs(B.asfloat(trades, like))
        # With NumPy and no slots, the rate is used as it is (as in 0.2).
        rate = self.rate if like is None and assets is None else _per_asset(B, self.rate, assets, t)
        if self.rate.ndim == 0:
            out: Array = rate * B.sum(t, axis=-1)
            return out
        return B.asfloat(B.sum(rate * t, axis=-1), t)

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
        self,
        trades: ArrayLike,
        *,
        spread: ArrayLike | None = None,
        assets: ArrayLike | None = None,
        **context: Any,
    ) -> Array:
        B, like = backend_and_like(trades, spread)
        t = B.abs(B.asfloat(trades, like))
        if self.half_spread is not None:
            half = (
                self.half_spread
                if like is None and assets is None
                else _per_asset(B, self.half_spread, assets, t)
            )
        else:
            quoted = B.broadcast_to(_require(B, "spread", spread, "SpreadCost()", t), t.shape)
            half = 0.5 * _fill_from_median(
                B,
                B.asfloat(quoted, t),
                t > 0,
                fallback=self.fallback_spread,
                name="spread",
                model="SpreadCost",
                option="fallback_spread",
            )
        return B.asfloat(B.sum(half * t, axis=-1), t)

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
        B, like = backend_and_like(trades, equity)
        t = B.abs(B.asfloat(trades, like))
        value = _require(B, "equity", equity, "FixedCost", t)
        n_trades = B.sum(t > self.threshold, axis=-1)
        positive = value > 0
        with B.errstate():
            out = B.where(positive, self.amount * n_trades / B.where(positive, value, 1.0), 0.0)
        return B.asfloat(out, t)

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
        B, like = backend_and_like(trades, equity, volatility, volume)
        t = B.abs(B.asfloat(trades, like))
        value = _require(B, "equity", equity, "SquareRootImpact", t)
        vol = _require(B, "volatility", volatility, "SquareRootImpact", t)
        traded = _require(B, "volume", volume, "SquareRootImpact", t)
        notional = t * value[..., None]
        sigma = _fill_from_median(
            B,
            B.asfloat(B.broadcast_to(vol, t.shape), t),
            t > 0,
            fallback=self.fallback_volatility,
            name="volatility",
            model="SquareRootImpact",
            option="fallback_volatility",
        )
        with B.errstate():
            liquid = B.isfinite(traded) & (traded > 0)
            participation = B.where(liquid, notional / B.where(liquid, traded, 1.0), 0.0)
            # The square root of 0 has an infinite derivative, so with torch the entries
            # that are not used (untraded or illiquid) take a harmless 1 and gradients stay
            # finite. NumPy needs no gradients (and the unused entries don't count).
            if like is not None:
                participation = B.where(liquid & (t > 0), participation, 1.0)
            unit = B.where(liquid, self.coefficient * sigma * B.sqrt(participation), 1.0)
        per_asset = B.where(t > 0, B.minimum(unit, 1.0) * t, 0.0)
        return B.asfloat(B.sum(per_asset, axis=-1), t)

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
        B, like = backend_and_like(trades)
        plain: dict[str, Any] | None = None  # the context without `assets`, if needed
        total: Any = None
        for model in self.models:
            if "assets" in context and not _takes_assets(model):
                if plain is None:
                    plain = {k: v for k, v in context.items() if k != "assets"}
                cost = model(trades, **plain)
            else:
                cost = model(trades, **context)
            total = B.asfloat(cost, like) if total is None else total + cost
        out: Array = total
        return out

    def __repr__(self) -> str:
        return " + ".join(repr(m) for m in self.models)


# ---------------------------------------------------------------------------
# Holding costs
# ---------------------------------------------------------------------------


class HoldingCost(abc.ABC):
    """Cost of holding positions for one period, as a fraction of equity.

    Trading costs (`CostModel`) price a rebalance; holding costs price the positions
    held afterwards: per-name borrow fees on shorts, a financing charge on the gross
    book. An environment charges them each period alongside its account's financing
    (cash interest, margin interest, a uniform borrow rate). They compose with `+`
    and, like cost models, broadcast over batch dimensions and run on any backend.
    """

    #: Context inputs the model reads (environments compute only these).
    requires: frozenset[str] = frozenset()

    @abc.abstractmethod
    def __call__(
        self,
        weights: ArrayLike,
        *,
        dt: float,
        borrow_rate: ArrayLike | None = None,
        assets: ArrayLike | None = None,
    ) -> Array:
        """Cost of holding `weights` for one period.

        Args:
            weights: `(..., n)` weights held over the period.
            dt: Period length in years.
            borrow_rate: `(..., n)` annual borrow fee of each asset (the `borrow_rate`
                data field), for models that read it.
            assets: `(..., n)` the asset behind each entry, as for `CostModel`.

        Returns:
            `(...)` cost as a fraction of equity.
        """

    def __add__(self, other: HoldingCost) -> CompositeHoldingCost:
        if not isinstance(other, HoldingCost):
            return NotImplemented
        return CompositeHoldingCost([self, other])


class BorrowFee(HoldingCost):
    """An annual fee on the value of each short position.

    Args:
        rate: Annual fee as a fraction of the short's value, a scalar or one per asset
            (finite, >= 0). If None, the env's `borrow_rate` data field gives each
            asset's fee at each time (so hard-to-borrow names can cost more as they
            become crowded); a missing (NaN) or negative fee is charged the
            cross-sectional median, as cost models treat missing inputs.
        fallback_rate: Fee used when no asset has a valid `borrow_rate`; without it,
            holding a short on such a row raises `ValueError`.
    """

    def __init__(
        self, rate: ArrayLike | None = None, *, fallback_rate: float | None = None
    ) -> None:
        self.rate = None if rate is None else _non_negative(rate, "rate")
        if self.rate is not None and fallback_rate is not None:
            raise ValueError("fallback_rate only applies to data rates (rate=None).")
        self.fallback_rate = _check_fallback(fallback_rate, "fallback_rate")
        self.requires = frozenset() if self.rate is not None else frozenset({"borrow_rate"})

    def __call__(
        self,
        weights: ArrayLike,
        *,
        dt: float,
        borrow_rate: ArrayLike | None = None,
        assets: ArrayLike | None = None,
    ) -> Array:
        B, like = backend_and_like(weights, borrow_rate)
        w = B.asfloat(weights, like)
        short = B.clip(-w, 0.0, None)
        if self.rate is not None:
            rate = _per_asset(B, self.rate, assets, w)
        else:
            quoted = B.broadcast_to(
                _require(B, "borrow_rate", borrow_rate, "BorrowFee()", w), w.shape
            )
            rate = _fill_from_median(
                B,
                B.asfloat(quoted, w),
                short > 0,
                fallback=self.fallback_rate,
                name="borrow_rate",
                model="BorrowFee",
                option="fallback_rate",
            )
        return B.asfloat(dt * B.sum(rate * short, axis=-1), w)

    def __repr__(self) -> str:
        rate = "from data" if self.rate is None else self.rate.tolist()
        return f"BorrowFee(rate={rate})"


class GrossFinancing(HoldingCost):
    """An annual financing charge on gross exposure, `rate * dt * sum(|w|)`.

    A prime-brokerage style charge on the whole book, longs and shorts alike, as
    long/short equity simulations often model the cost of running leverage.

    Args:
        rate: Annual charge as a fraction of gross exposure (finite, >= 0).
    """

    def __init__(self, rate: float) -> None:
        self.rate = _non_negative_scalar(rate, "rate")

    def __call__(
        self,
        weights: ArrayLike,
        *,
        dt: float,
        borrow_rate: ArrayLike | None = None,
        assets: ArrayLike | None = None,
    ) -> Array:
        B, like = backend_and_like(weights)
        w = B.asfloat(weights, like)
        return self.rate * dt * B.sum(B.abs(w), axis=-1)

    def __repr__(self) -> str:
        return f"GrossFinancing({self.rate})"


class CompositeHoldingCost(HoldingCost):
    """The sum of several holding costs (build it with `+`)."""

    models: list[HoldingCost]

    def __init__(self, models: Sequence[HoldingCost]) -> None:
        flat: list[HoldingCost] = []
        for model in models:
            flat.extend(model.models if isinstance(model, CompositeHoldingCost) else [model])
        self.models = flat
        self.requires = frozenset().union(*(m.requires for m in flat))

    def __call__(
        self,
        weights: ArrayLike,
        *,
        dt: float,
        borrow_rate: ArrayLike | None = None,
        assets: ArrayLike | None = None,
    ) -> Array:
        total: Array = 0.0
        for model in self.models:
            total = total + model(weights, dt=dt, borrow_rate=borrow_rate, assets=assets)
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
    "HoldingCost",
    "BorrowFee",
    "GrossFinancing",
    "CompositeHoldingCost",
]
