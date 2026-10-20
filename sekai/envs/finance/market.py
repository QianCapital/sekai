"""Price-path simulators and historical-data windowing shared by the finance envs."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def simulate_gbm(
    rng: np.random.Generator,
    n_steps: int,
    *,
    s0: ArrayLike = 100.0,
    mu: ArrayLike = 0.0,
    sigma: ArrayLike = 0.2,
    dt: float = 1 / 252,
    corr: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Simulate geometric Brownian motion price paths.

    Uses the exact log-normal transition, so prices stay strictly positive.

    Args:
        rng: Source of randomness.
        n_steps: Number of steps; the result has n_steps + 1 rows (row 0 is s0).
        s0: Initial price. Scalars for s0, mu and sigma give a single path of
            shape (n_steps + 1,); length-n arrays give n assets and shape
            (n_steps + 1, n).
        mu: Annualised drift (scalar or length-n array).
        sigma: Annualised volatility (scalar or length-n array).
        dt: Step length in years (1/252 is one trading day).
        corr: Optional (n, n) correlation matrix between the assets' shocks.
    """
    s0_arr = np.asarray(s0, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    shape = np.broadcast_shapes(s0_arr.shape, mu_arr.shape, sigma_arr.shape)
    if len(shape) > 1:
        raise ValueError("s0, mu and sigma must be scalars or 1-D arrays.")
    if np.any(sigma_arr < 0):
        raise ValueError("sigma must be non-negative.")
    n_assets = shape[0] if shape else 1

    shocks = rng.standard_normal((n_steps, n_assets))
    if corr is not None:
        corr_arr = np.asarray(corr, dtype=np.float64)
        if corr_arr.shape != (n_assets, n_assets):
            raise ValueError(
                f"corr must have shape ({n_assets}, {n_assets}), got {corr_arr.shape}."
            )
        shocks = shocks @ np.linalg.cholesky(corr_arr).T

    drift = (mu_arr - 0.5 * sigma_arr**2) * dt
    log_increments = drift + sigma_arr * np.sqrt(dt) * shocks
    log_paths = np.vstack([np.zeros((1, n_assets)), np.cumsum(log_increments, axis=0)])
    paths: NDArray[np.float64] = np.broadcast_to(s0_arr, (n_assets,)) * np.exp(log_paths)
    return paths if shape else paths[:, 0]


def random_correlation(
    rng: np.random.Generator, n: int, strength: float = 0.5
) -> NDArray[np.float64]:
    """A random valid correlation matrix with typical off-diagonal size ~strength."""
    factors = rng.standard_normal((n, max(1, n // 2))) * strength
    cov = factors @ factors.T + np.eye(n) * (1 - strength**2)
    d = np.sqrt(np.diag(cov))
    out: NDArray[np.float64] = cov / np.outer(d, d)
    return out


def validate_prices(prices: ArrayLike, *, ndim: int, min_length: int) -> NDArray[np.float64]:
    """Check user-supplied historical prices and return them as float64.

    For `ndim=1` a single column `(T, 1)` (a one-column DataFrame) is accepted as `(T,)`.
    """
    arr = np.asarray(prices, dtype=np.float64)
    if ndim == 2 and arr.ndim == 1:
        arr = arr[:, None]
    if ndim == 1 and arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != ndim:
        raise ValueError(f"prices must be {ndim}-D, got shape {arr.shape}.")
    if arr.shape[0] < min_length:
        raise ValueError(
            f"prices has {arr.shape[0]} rows but at least {min_length} are needed "
            "(lookback + episode_length + 1)."
        )
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise ValueError("prices must be finite and strictly positive.")
    return arr


def sample_window(
    rng: np.random.Generator, prices: NDArray[Any], length: int
) -> NDArray[np.float64]:
    """A random contiguous window of `length` rows from historical prices."""
    start = int(rng.integers(0, prices.shape[0] - length + 1))
    out: NDArray[np.float64] = prices[start : start + length].astype(np.float64)
    return out
