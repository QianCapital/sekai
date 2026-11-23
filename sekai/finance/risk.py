"""Factor risk models: ex-ante portfolio variance and factor exposures.

A factor model writes the covariance of asset returns as `Σ = L F L' + diag(d)`:
`L` holds each asset's loadings on `k` factors, `F` is the factor covariance and
`d` the idiosyncratic variances. The variance of a portfolio `w` is then
`x' F x + Σ_i w_i² d_i` with exposures `x = L' w`, which costs `O(n k + k²)` instead
of the `O(n²)` of a full covariance and stays meaningful for universes far larger
than the history is long.

Environments read the model's inputs from the data at each decision (per-asset
fields for the loadings and idiosyncratic variances, a market-wide series for the
factor covariance), so the same model serves historical data and generated
scenarios, and report the variance (`StepOutcome.variance`, for rewards such as
`PortfolioVariance`) and the factor exposures (in `info`). Everything runs on any
`sekai.backend` and is differentiable in the weights.
"""

from __future__ import annotations

from dataclasses import dataclass

from sekai.backend import Array, backend_and_like


def factor_exposures(weights: Array, loadings: Array) -> Array:
    """`L' w`: `(..., k)` exposures of `(..., n)` weights to `(..., n, k)` loadings.

    A missing (NaN) loading of an asset that isn't held is ignored; one of a held asset
    makes its exposures NaN.
    """
    B, like = backend_and_like(weights, loadings)
    w = B.asfloat(weights, like)
    loads = B.asfloat(loadings, like)
    loads = B.where(B.isnan(loads) & (w == 0)[..., None], 0.0, loads)
    return B.einsum("...n,...nk->...k", w, loads)


def portfolio_variance(
    weights: Array,
    *,
    loadings: Array | None = None,
    factor_cov: Array | None = None,
    idio_var: Array | None = None,
) -> Array:
    """The variance `x' F x + Σ w_i² d_i` of `(..., n)` weights under a factor model.

    Args:
        weights: `(..., n)` portfolio weights.
        loadings: `(..., n, k)` factor loadings (with `factor_cov`).
        factor_cov: `(..., k, k)` factor covariance (with `loadings`).
        idio_var: `(..., n)` idiosyncratic variances.

    Returns:
        `(...)` variances, in the units of the inputs (per period for per-period
        covariances). Missing (NaN) inputs of an asset that isn't held are ignored; one
        of a held asset makes the variance NaN rather than understating it.

    Raises:
        ValueError: If only one of `loadings` and `factor_cov` is given, or neither
            part of the model is.
    """
    return _variance_and_exposures(weights, loadings, factor_cov, idio_var)[0]


def _variance_and_exposures(
    weights: Array, loadings: Array | None, factor_cov: Array | None, idio_var: Array | None
) -> tuple[Array, Array | None]:
    """`portfolio_variance(...)`, and the factor exposures it computes on the way (None
    without loadings)."""
    B, like = backend_and_like(weights, loadings, factor_cov, idio_var)
    w = B.asfloat(weights, like)
    if (loadings is None) != (factor_cov is None):
        raise ValueError("loadings and factor_cov go together: give both or neither.")
    if loadings is None and idio_var is None:
        raise ValueError("A factor model needs loadings and factor_cov, idio_var, or both.")
    total: Array = 0.0
    x = None
    if loadings is not None and factor_cov is not None:
        x = factor_exposures(w, loadings)
        total = B.einsum("...k,...kj,...j->...", x, B.asfloat(factor_cov, like), x)
    if idio_var is not None:
        d = B.asfloat(idio_var, like)
        d = B.where(B.isnan(d) & (w == 0), 0.0, d)
        total = total + B.sum(w * w * d, axis=-1)
    return total, x


@dataclass(frozen=True)
class FactorRiskModel:
    """Where an environment finds a factor model in its data, by field name.

    Args:
        loadings: Per-asset field `(T, N, k)` of factor loadings, or None for a
            purely idiosyncratic (diagonal) model.
        factor_cov: Market-wide series `(T, k, k)` of factor covariances (with
            `loadings`).
        idio_var: Per-asset field `(T, N)` of idiosyncratic variances, or None.

    Raises:
        ValueError: If only one of `loadings` and `factor_cov` is named, or nothing is.

    Example:
        ```python
        # Fields "loadings" (T, N, k) and "idio_var" (T, N), series "factor_cov" (T, k, k):
        env = BatchMarketEnv(
            8, data=data, risk=FactorRiskModel(),
            reward=SimpleReturn() - 2.0 * PortfolioVariance(),
        )
        ```
    """

    loadings: str | None = "loadings"
    factor_cov: str | None = "factor_cov"
    idio_var: str | None = "idio_var"

    def __post_init__(self) -> None:
        if (self.loadings is None) != (self.factor_cov is None):
            raise ValueError("loadings and factor_cov go together: name both or neither.")
        if self.loadings is None and self.idio_var is None:
            raise ValueError("A FactorRiskModel needs loadings and factor_cov, idio_var, or both.")

    @property
    def asset_fields(self) -> tuple[str, ...]:
        """The per-asset fields the model reads."""
        return tuple(name for name in (self.loadings, self.idio_var) if name is not None)

    @property
    def market_fields(self) -> tuple[str, ...]:
        """The market-wide series the model reads."""
        return () if self.factor_cov is None else (self.factor_cov,)


__all__ = ["FactorRiskModel", "factor_exposures", "portfolio_variance"]
