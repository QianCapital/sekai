"""Batch-first market price simulators ("market models") for synthetic financial markets.

A `MarketModel` turns a random generator into price paths for a fixed universe of assets.
Every model is vectorised over paths and assets: ask for one path with
`model.simulate(rng, n_steps)` (prices of shape (n_steps + 1, n_assets)) or for a batch with
`model.simulate(rng, n_steps, n_paths=k)` (shape (k, n_steps + 1, n_assets)). Models may also
report latent series (market regime, stochastic variance, jump counts, ...) in
`Simulation.state`, and a model can describe a dynamic universe in which assets list and delist
over time through `Simulation.mask`.

Parameters are annualised unless stated otherwise and `dt` is the step length in years
(1/252 is one trading day). Per-asset parameters accept a scalar or a length-n array.

```python
rng = np.random.default_rng(0)
market = ListingProcess(Heston(n_assets=5), min_listed=2)
sim = market.simulate(rng, 252, n_paths=64)
sim.prices.shape, sim.mask.shape, sim.state["variance"].shape  # (64, 253, 5) each
```
"""

from __future__ import annotations

import abc
import math
import operator
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "Simulation",
    "MarketModel",
    "GBM",
    "Heston",
    "MertonJumpDiffusion",
    "RegimeSwitching",
    "GARCH",
    "MeanReverting",
    "CointegratedPair",
    "BlockBootstrap",
    "Randomized",
    "ListingProcess",
]

_DEFAULT_DT = 1 / 252


# --------------------------------------------------------------------------------------------
# Result type and base class
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class Simulation:
    """Simulated market paths returned by `MarketModel.simulate`.

    Attributes:
        prices: Prices of shape (n_steps + 1, n_assets), or (n_paths, n_steps + 1, n_assets)
            when simulated with `n_paths`. Row 0 (along the time axis) is the initial price.
            NaN where an asset is not listed.
        mask: Boolean array shaped like `prices`, True where the asset is listed and tradable,
            or None when every entry is valid.
        state: Latent or auxiliary series keyed by name (see `MarketModel.state_spec`). Each
            array's leading dimensions equal `prices.shape[:-1]`, i.e. (n_steps + 1,) or
            (n_paths, n_steps + 1), followed by the model's per-step shape for that key.

    Raises:
        ValueError: If the shapes of `prices`, `mask` and `state` are inconsistent.
    """

    prices: NDArray[np.float64]
    mask: NDArray[np.bool_] | None = None
    state: Mapping[str, NDArray[np.float64]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.prices, np.ndarray) or self.prices.ndim not in (2, 3):
            raise ValueError(
                "prices must be a 2-D (n_steps + 1, n_assets) or 3-D "
                "(n_paths, n_steps + 1, n_assets) array."
            )
        if self.mask is not None and (
            not isinstance(self.mask, np.ndarray)
            or self.mask.dtype != np.bool_
            or self.mask.shape != self.prices.shape
        ):
            raise ValueError(f"mask must be a boolean array of shape {self.prices.shape}.")
        lead = self.prices.shape[:-1]
        for key, value in self.state.items():
            if np.shape(value)[: len(lead)] != lead:
                raise ValueError(
                    f"state[{key!r}] has shape {np.shape(value)}; its leading dimensions "
                    f"must be {lead}."
                )


class MarketModel(abc.ABC):
    """A generator of synthetic price paths for a fixed universe of `n_assets` assets.

    Subclasses implement `simulate`, which must be vectorised over paths and assets, return
    strictly positive prices wherever an asset is listed and honour the shapes documented on
    `Simulation`.
    """

    @property
    @abc.abstractmethod
    def n_assets(self) -> int:
        """Number of assets (the size of the last axis of `Simulation.prices`)."""

    @property
    @abc.abstractmethod
    def dt(self) -> float:
        """Step length in years."""

    def state_spec(self) -> dict[str, tuple[int, ...]]:
        """Keys of `Simulation.state` mapped to their per-step trailing shape.

        For example `{"regime": (), "variance": (n_assets,)}`. Empty by default.
        """
        return {}

    @abc.abstractmethod
    def simulate(
        self, rng: np.random.Generator, n_steps: int, *, n_paths: int | None = None
    ) -> Simulation:
        """Simulate price paths.

        Args:
            rng: Source of randomness.
            n_steps: Number of steps (>= 1); the time axis has n_steps + 1 entries.
            n_paths: Number of independent paths (>= 1). None gives a single path without a
                leading paths axis; an int (even 1) adds a leading axis of that size.

        Returns:
            A `Simulation` with prices of shape (n_steps + 1, n_assets) or
            (n_paths, n_steps + 1, n_assets).

        Raises:
            ValueError: If n_steps or n_paths is smaller than 1.
            TypeError: If n_steps or n_paths is not an integer.
        """


class _Model(MarketModel):
    """Shared plumbing for the built-in models: read-only attributes and a compact repr."""

    _n_assets: int
    _dt: float

    @property
    def n_assets(self) -> int:
        return self._n_assets

    @property
    def dt(self) -> float:
        return self._dt

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable; build a new model instead.")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable; build a new model instead.")

    def _set(self, **attributes: Any) -> None:
        for name, value in attributes.items():
            object.__setattr__(self, name, value)

    def _repr_fields(self) -> dict[str, str]:
        return {}

    def __repr__(self) -> str:
        body = ", ".join(f"{key}={value}" for key, value in self._repr_fields().items())
        return f"{type(self).__name__}({body})"


# --------------------------------------------------------------------------------------------
# Validation and numeric helpers
# --------------------------------------------------------------------------------------------


def _require(condition: bool | np.bool_, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _as_int(name: str, value: Any) -> int:
    if isinstance(value, bool | np.bool_):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.") from None


def _sim_shape(n_steps: Any, n_paths: Any) -> tuple[int, tuple[int, ...]]:
    """Validate simulate() arguments; return n_steps and the leading paths shape."""
    steps = _as_int("n_steps", n_steps)
    _require(steps >= 1, f"n_steps must be >= 1, got {steps}.")
    if n_paths is None:
        return steps, ()
    paths = _as_int("n_paths", n_paths)
    _require(paths >= 1, f"n_paths must be >= 1, got {paths}.")
    return steps, (paths,)


def _scalar(name: str, value: Any) -> float:
    arr = np.asarray(value, dtype=np.float64)
    _require(arr.ndim == 0, f"{name} must be a scalar, got shape {arr.shape}.")
    out = float(arr)
    _require(math.isfinite(out), f"{name} must be finite, got {out}.")
    return out


def _positive_dt(dt: Any) -> float:
    out = _scalar("dt", dt)
    _require(out > 0, f"dt must be positive, got {out}.")
    return out


def _readonly(values: ArrayLike) -> NDArray[np.float64]:
    out = np.array(values, dtype=np.float64)
    out.setflags(write=False)
    return out


def _infer_n_assets(n_assets: int | None, corr: ArrayLike | None, **params: Any) -> int:
    """Number of assets implied by an explicit count, per-asset parameters and `corr`."""
    candidates: list[tuple[str, int]] = []
    if n_assets is not None:
        count = _as_int("n_assets", n_assets)
        _require(count >= 1, f"n_assets must be >= 1, got {count}.")
        candidates.append(("n_assets", count))
    for name, value in params.items():
        if value is None:
            continue
        shape = np.shape(value)
        _require(len(shape) <= 1, f"{name} must be a scalar or a 1-D array, got shape {shape}.")
        if len(shape) == 1 and shape[0] != 1:
            _require(shape[0] >= 1, f"{name} must not be empty.")
            candidates.append((name, shape[0]))
    if corr is not None and np.ndim(corr) == 2:
        candidates.append(("corr", np.shape(corr)[0]))
    if not candidates:
        return 1
    first, n = candidates[0]
    for name, m in candidates[1:]:
        _require(m == n, f"{name} implies {m} assets but {first} implies {n}.")
    return n


def _per_asset(name: str, value: ArrayLike, n: int) -> NDArray[np.float64]:
    """Broadcast a scalar or length-n parameter to a read-only float64 array of shape (n,)."""
    arr = np.asarray(value, dtype=np.float64)
    _require(
        arr.ndim == 0 or (arr.ndim == 1 and arr.shape[0] in (1, n)),
        f"{name} must be a scalar or an array of length {n}, got shape {arr.shape}.",
    )
    _require(bool(np.all(np.isfinite(arr))), f"{name} must be finite.")
    return _readonly(np.broadcast_to(arr, (n,)))


def _positive_prices(s0: ArrayLike, n: int) -> NDArray[np.float64]:
    out = _per_asset("s0", s0, n)
    _require(bool(np.all(out > 0)), "s0 must be strictly positive.")
    return out


def _correlation(
    corr: ArrayLike | None, n: int
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """Validate a correlation matrix; return it (read-only) and its lower Cholesky factor."""
    if corr is None:
        return None, None
    arr = np.asarray(corr, dtype=np.float64)
    _require(arr.shape == (n, n), f"corr must have shape ({n}, {n}), got {arr.shape}.")
    _require(bool(np.all(np.isfinite(arr))), "corr must be finite.")
    _require(bool(np.allclose(arr, arr.T, rtol=0.0, atol=1e-8)), "corr must be symmetric.")
    _require(
        bool(np.allclose(np.diag(arr), 1.0, rtol=0.0, atol=1e-8)),
        "corr must have a unit diagonal.",
    )
    try:
        chol = np.linalg.cholesky(arr)
    except np.linalg.LinAlgError:
        raise ValueError("corr must be positive definite.") from None
    return _readonly(arr), _readonly(chol)


def _normals(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    n: int,
    chol: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """Standard normals of shape (*shape, n), correlated across the last axis by `chol`."""
    z: NDArray[np.float64] = rng.standard_normal((*shape, n))
    if chol is not None:
        z = z @ chol.T
    return z


def _prices(s0: NDArray[np.float64], log_increments: NDArray[np.float64]) -> NDArray[np.float64]:
    """Prices from per-step log increments of shape (..., n_steps, n_assets)."""
    lead, n = log_increments.shape[:-2], log_increments.shape[-1]
    log_paths = np.concatenate(
        [np.zeros((*lead, 1, n)), np.cumsum(log_increments, axis=-2)], axis=-2
    )
    out: NDArray[np.float64] = s0 * np.exp(log_paths)
    return out


def _ou_coefficients(theta: Any, sigma: Any, dt: float) -> tuple[Any, Any]:
    """Exact OU transition: x' = x * decay + scale * z (for zero mean)."""
    decay = np.exp(-theta * dt)
    scale = sigma * np.sqrt(-np.expm1(-2.0 * theta * dt) / (2.0 * theta))
    return decay, scale


def _format(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, np.ndarray):
        if value.ndim == 1 and value.size and bool(np.all(value == value[0])):
            return _format(float(value[0]))
        if value.size > 16:
            return f"<array of shape {value.shape}>"
        return str(np.round(value, 6).tolist())
    if isinstance(value, float):
        text = f"{value:.6g}"
        return text if any(c in text for c in ".ein") else f"{text}.0"
    return repr(value)


def _format_dt(dt: float) -> str:
    steps = 1.0 / dt
    if abs(steps - round(steps)) < 1e-9 and round(steps) > 1:
        return f"1/{round(steps)}"
    return _format(dt)


# --------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------


class GBM(_Model):
    """Geometric Brownian motion with the exact log-normal transition.

    log S_{t+1} = log S_t + (mu - sigma^2 / 2) dt + sigma sqrt(dt) z, with z correlated across
    assets by `corr`. Without `n_paths` it draws random numbers in the same order as
    `sekai.envs.finance.market.simulate_gbm`, so both give identical paths for the same seed.

    Args:
        mu: Annualised drift, scalar or shape (n_assets,).
        sigma: Annualised volatility (>= 0), scalar or shape (n_assets,).
        s0: Initial price (> 0), scalar or shape (n_assets,).
        corr: Optional (n_assets, n_assets) correlation matrix of the assets' shocks.
        dt: Step length in years.
        n_assets: Number of assets; needed only to broadcast all-scalar parameters to more
            than one asset (otherwise inferred from the parameters or `corr`).

    Raises:
        ValueError: On inconsistent shapes or out-of-range parameters.
    """

    mu: NDArray[np.float64]
    sigma: NDArray[np.float64]
    s0: NDArray[np.float64]
    corr: NDArray[np.float64] | None
    _chol: NDArray[np.float64] | None

    def __init__(
        self,
        mu: ArrayLike = 0.0,
        sigma: ArrayLike = 0.2,
        *,
        s0: ArrayLike = 100.0,
        corr: ArrayLike | None = None,
        dt: float = _DEFAULT_DT,
        n_assets: int | None = None,
    ) -> None:
        n = _infer_n_assets(n_assets, corr, mu=mu, sigma=sigma, s0=s0)
        sigma_arr = _per_asset("sigma", sigma, n)
        _require(bool(np.all(sigma_arr >= 0)), "sigma must be non-negative.")
        corr_arr, chol = _correlation(corr, n)
        self._set(
            _n_assets=n,
            _dt=_positive_dt(dt),
            mu=_per_asset("mu", mu, n),
            sigma=sigma_arr,
            s0=_positive_prices(s0, n),
            corr=corr_arr,
            _chol=chol,
        )

    def simulate(
        self, rng: np.random.Generator, n_steps: int, *, n_paths: int | None = None
    ) -> Simulation:
        """Simulate GBM paths; see `MarketModel.simulate`. The state is empty."""
        steps, lead = _sim_shape(n_steps, n_paths)
        shocks = _normals(rng, (*lead, steps), self._n_assets, self._chol)
        drift = (self.mu - 0.5 * self.sigma**2) * self._dt
        log_increments = drift + self.sigma * np.sqrt(self._dt) * shocks
        return Simulation(prices=_prices(self.s0, log_increments))

    def _repr_fields(self) -> dict[str, str]:
        return {
            "n_assets": str(self._n_assets),
            "mu": _format(self.mu),
            "sigma": _format(self.sigma),
            "s0": _format(self.s0),
            "corr": _format(self.corr),
            "dt": _format_dt(self._dt),
        }


class Heston(_Model):
    """Heston stochastic-volatility model, simulated with full-truncation Euler.

    With v+ = max(v, 0) at each step:
    log S += (mu - v+ / 2) dt + sqrt(v+ dt) z1 and
    v += kappa (theta - v+) dt + xi sqrt(v+ dt) z2, where corr(z1, z2) = rho per asset and z1
    is correlated across assets by `corr`.

    Args:
        mu: Annualised drift, scalar or shape (n_assets,).
        kappa: Mean-reversion speed of the variance (> 0), per year.
        theta: Long-run variance (> 0), annualised.
        xi: Volatility of variance (> 0).
        rho: Correlation between price and variance shocks, in [-1, 1].
        v0: Initial variance (>= 0); defaults to theta.
        s0: Initial price (> 0).
        corr: Optional (n_assets, n_assets) correlation matrix of the price shocks.
        dt: Step length in years.
        n_assets: Number of assets when every parameter is a scalar.

    State:
        "variance": per-step shape (n_assets,), the truncated variance v+ at each time.

    Raises:
        ValueError: On inconsistent shapes or out-of-range parameters.
    """

    mu: NDArray[np.float64]
    kappa: NDArray[np.float64]
    theta: NDArray[np.float64]
    xi: NDArray[np.float64]
    rho: NDArray[np.float64]
    v0: NDArray[np.float64]
    s0: NDArray[np.float64]
    corr: NDArray[np.float64] | None
    _chol: NDArray[np.float64] | None

    def __init__(
        self,
        mu: ArrayLike = 0.05,
        kappa: ArrayLike = 2.0,
        theta: ArrayLike = 0.04,
        xi: ArrayLike = 0.3,
        rho: ArrayLike = -0.7,
        *,
        v0: ArrayLike | None = None,
        s0: ArrayLike = 100.0,
        corr: ArrayLike | None = None,
        dt: float = _DEFAULT_DT,
        n_assets: int | None = None,
    ) -> None:
        n = _infer_n_assets(
            n_assets, corr, mu=mu, kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0, s0=s0
        )
        kappa_arr = _per_asset("kappa", kappa, n)
        theta_arr = _per_asset("theta", theta, n)
        xi_arr = _per_asset("xi", xi, n)
        rho_arr = _per_asset("rho", rho, n)
        v0_arr = theta_arr if v0 is None else _per_asset("v0", v0, n)
        _require(bool(np.all(kappa_arr > 0)), "kappa must be positive.")
        _require(bool(np.all(theta_arr > 0)), "theta must be positive.")
        _require(bool(np.all(xi_arr > 0)), "xi must be positive.")
        _require(bool(np.all(np.abs(rho_arr) <= 1)), "rho must lie in [-1, 1].")
        _require(bool(np.all(v0_arr >= 0)), "v0 must be non-negative.")
        corr_arr, chol = _correlation(corr, n)
        self._set(
            _n_assets=n,
            _dt=_positive_dt(dt),
            mu=_per_asset("mu", mu, n),
            kappa=kappa_arr,
            theta=theta_arr,
            xi=xi_arr,
            rho=rho_arr,
            v0=v0_arr,
            s0=_positive_prices(s0, n),
            corr=corr_arr,
            _chol=chol,
        )

    def state_spec(self) -> dict[str, tuple[int, ...]]:
        return {"variance": (self._n_assets,)}

    def simulate(
        self, rng: np.random.Generator, n_steps: int, *, n_paths: int | None = None
    ) -> Simulation:
        """Simulate Heston paths; see `MarketModel.simulate`."""
        steps, lead = _sim_shape(n_steps, n_paths)
        n, dt = self._n_assets, self._dt
        eps = rng.standard_normal((*lead, steps, 2, n))
        z1 = eps[..., 0, :]
        if self._chol is not None:
            z1 = z1 @ self._chol.T
        z2 = self.rho * z1 + np.sqrt(1.0 - self.rho**2) * eps[..., 1, :]

        log_rel = np.zeros((*lead, steps + 1, n))
        variance = np.empty((*lead, steps + 1, n))
        x = np.zeros((*lead, n))
        v = np.zeros((*lead, n)) + self.v0
        for t in range(steps):
            v_plus = np.maximum(v, 0.0)
            variance[..., t, :] = v_plus
            root = np.sqrt(v_plus * dt)
            x = x + (self.mu - 0.5 * v_plus) * dt + root * z1[..., t, :]
            v = v + self.kappa * (self.theta - v_plus) * dt + self.xi * root * z2[..., t, :]
            log_rel[..., t + 1, :] = x
        variance[..., steps, :] = np.maximum(v, 0.0)
        prices: NDArray[np.float64] = self.s0 * np.exp(log_rel)
        return Simulation(prices=prices, state={"variance": variance})

    def _repr_fields(self) -> dict[str, str]:
        return {
            "n_assets": str(self._n_assets),
            "mu": _format(self.mu),
            "kappa": _format(self.kappa),
            "theta": _format(self.theta),
            "xi": _format(self.xi),
            "rho": _format(self.rho),
            "v0": _format(self.v0),
            "s0": _format(self.s0),
            "corr": _format(self.corr),
            "dt": _format_dt(self._dt),
        }


class MertonJumpDiffusion(_Model):
    """Merton jump-diffusion: GBM plus compound-Poisson log-normal jumps.

    Each step's log increment is (mu - sigma^2 / 2 - lambda k) dt + sigma sqrt(dt) z plus the
    sum of N ~ Poisson(lambda dt) iid Normal(jump_mean, jump_std^2) log jumps, where
    k = exp(jump_mean + jump_std^2 / 2) - 1 compensates the jumps so that
    E[S_t] = s0 exp(mu t). Diffusive shocks are correlated across assets by `corr`; jumps are
    independent across assets.

    Args:
        mu: Annualised drift (including the jump compensator).
        sigma: Annualised diffusive volatility (>= 0).
        jump_intensity: Expected number of jumps per year (>= 0).
        jump_mean: Mean of a single log jump.
        jump_std: Standard deviation of a single log jump (>= 0).
        s0: Initial price (> 0).
        corr: Optional (n_assets, n_assets) correlation matrix of the diffusive shocks.
        dt: Step length in years.
        n_assets: Number of assets when every parameter is a scalar.

    State:
        "jumps": per-step shape (n_assets,), the number of jumps in the step ending at t
        (0 at t = 0), as float64.

    Raises:
        ValueError: On inconsistent shapes or out-of-range parameters.
    """

    mu: NDArray[np.float64]
    sigma: NDArray[np.float64]
    jump_intensity: NDArray[np.float64]
    jump_mean: NDArray[np.float64]
    jump_std: NDArray[np.float64]
    s0: NDArray[np.float64]
    corr: NDArray[np.float64] | None
    _chol: NDArray[np.float64] | None

    def __init__(
        self,
        mu: ArrayLike = 0.05,
        sigma: ArrayLike = 0.2,
        jump_intensity: ArrayLike = 1.0,
        jump_mean: ArrayLike = -0.05,
        jump_std: ArrayLike = 0.1,
        *,
        s0: ArrayLike = 100.0,
        corr: ArrayLike | None = None,
        dt: float = _DEFAULT_DT,
        n_assets: int | None = None,
    ) -> None:
        n = _infer_n_assets(
            n_assets,
            corr,
            mu=mu,
            sigma=sigma,
            jump_intensity=jump_intensity,
            jump_mean=jump_mean,
            jump_std=jump_std,
            s0=s0,
        )
        sigma_arr = _per_asset("sigma", sigma, n)
        intensity = _per_asset("jump_intensity", jump_intensity, n)
        jump_std_arr = _per_asset("jump_std", jump_std, n)
        _require(bool(np.all(sigma_arr >= 0)), "sigma must be non-negative.")
        _require(bool(np.all(intensity >= 0)), "jump_intensity must be non-negative.")
        _require(bool(np.all(jump_std_arr >= 0)), "jump_std must be non-negative.")
        corr_arr, chol = _correlation(corr, n)
        self._set(
            _n_assets=n,
            _dt=_positive_dt(dt),
            mu=_per_asset("mu", mu, n),
            sigma=sigma_arr,
            jump_intensity=intensity,
            jump_mean=_per_asset("jump_mean", jump_mean, n),
            jump_std=jump_std_arr,
            s0=_positive_prices(s0, n),
            corr=corr_arr,
            _chol=chol,
        )

    def state_spec(self) -> dict[str, tuple[int, ...]]:
        return {"jumps": (self._n_assets,)}

    def simulate(
        self, rng: np.random.Generator, n_steps: int, *, n_paths: int | None = None
    ) -> Simulation:
        """Simulate jump-diffusion paths; see `MarketModel.simulate`."""
        steps, lead = _sim_shape(n_steps, n_paths)
        n, dt = self._n_assets, self._dt
        shape = (*lead, steps, n)
        z = _normals(rng, (*lead, steps), n, self._chol)
        counts = rng.poisson(self.jump_intensity * dt, size=shape).astype(np.float64)
        w = rng.standard_normal(shape)

        k = np.expm1(self.jump_mean + 0.5 * self.jump_std**2)
        drift = (self.mu - 0.5 * self.sigma**2 - self.jump_intensity * k) * dt
        log_jumps = counts * self.jump_mean + np.sqrt(counts) * self.jump_std * w
        log_increments = drift + self.sigma * np.sqrt(dt) * z + log_jumps
        jumps = np.concatenate([np.zeros((*lead, 1, n)), counts], axis=-2)
        return Simulation(prices=_prices(self.s0, log_increments), state={"jumps": jumps})

    def _repr_fields(self) -> dict[str, str]:
        return {
            "n_assets": str(self._n_assets),
            "mu": _format(self.mu),
            "sigma": _format(self.sigma),
            "jump_intensity": _format(self.jump_intensity),
            "jump_mean": _format(self.jump_mean),
            "jump_std": _format(self.jump_std),
            "s0": _format(self.s0),
            "corr": _format(self.corr),
            "dt": _format_dt(self._dt),
        }


def _stationary_distribution(transition: NDArray[np.float64]) -> NDArray[np.float64]:
    """A stationary distribution pi = pi P (the minimum-norm one if it is not unique)."""
    k = transition.shape[0]
    system = np.vstack([transition.T - np.eye(k), np.ones((1, k))])
    target = np.zeros(k + 1)
    target[-1] = 1.0
    pi = np.clip(np.linalg.lstsq(system, target, rcond=None)[0], 0.0, None)
    out: NDArray[np.float64] = pi / pi.sum()
    return out


def _sample_categorical(
    cumulative: NDArray[np.float64], u: NDArray[np.float64]
) -> NDArray[np.intp]:
    """Category index for uniforms `u` given cumulative probabilities along the last axis."""
    k = cumulative.shape[-1]
    index = (u[..., None] >= cumulative).sum(axis=-1)
    out: NDArray[np.intp] = np.minimum(index, k - 1).astype(np.intp)
    return out


class RegimeSwitching(_Model):
    """GBM whose drift, volatility and correlation follow a hidden Markov chain of regimes.

    The regime at time t governs the increment from t to t + 1. The chain starts from
    `initial` at t = 0 and moves according to `transition` after every step.

    Args:
        regimes: K >= 1 `GBM` models giving (mu, sigma, corr) for each regime. They must share
            n_assets, dt and s0.
        transition: (K, K) row-stochastic matrix; entry (i, j) is the per-step probability of
            moving from regime i to regime j.
        initial: Distribution of the regime at t = 0, shape (K,). Defaults to the stationary
            distribution of `transition`.

    State:
        "regime": per-step shape (), the regime index at each time as float64.

    Raises:
        TypeError: If a regime is not a `GBM`.
        ValueError: On inconsistent regimes or an invalid transition matrix or distribution.
    """

    regimes: tuple[GBM, ...]
    transition: NDArray[np.float64]
    initial: NDArray[np.float64]

    def __init__(
        self,
        regimes: Sequence[GBM],
        transition: ArrayLike,
        *,
        initial: ArrayLike | None = None,
    ) -> None:
        models = tuple(regimes)
        _require(len(models) >= 1, "regimes must contain at least one GBM.")
        for model in models:
            if not isinstance(model, GBM):
                raise TypeError(f"regimes must be GBM instances, got {type(model).__name__}.")
        first = models[0]
        for model in models[1:]:
            _require(model.n_assets == first.n_assets, "all regimes must share n_assets.")
            _require(model.dt == first.dt, "all regimes must share dt.")
            _require(bool(np.array_equal(model.s0, first.s0)), "all regimes must share s0.")

        k = len(models)
        matrix = np.asarray(transition, dtype=np.float64)
        _require(matrix.shape == (k, k), f"transition must have shape ({k}, {k}).")
        _require(bool(np.all(np.isfinite(matrix))), "transition must be finite.")
        _require(bool(np.all(matrix >= 0)), "transition must be non-negative.")
        _require(
            bool(np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=1e-8)),
            "each row of transition must sum to 1.",
        )
        if initial is None:
            start = _stationary_distribution(matrix)
        else:
            start = np.asarray(initial, dtype=np.float64)
            _require(start.shape == (k,), f"initial must have shape ({k},).")
            _require(
                bool(np.all(np.isfinite(start)) and np.all(start >= 0)),
                "initial must be finite and non-negative.",
            )
            _require(abs(start.sum() - 1.0) <= 1e-8, "initial must sum to 1.")
        self._set(
            _n_assets=first.n_assets,
            _dt=first.dt,
            regimes=models,
            transition=_readonly(matrix),
            initial=_readonly(start),
        )

    @property
    def s0(self) -> NDArray[np.float64]:
        """Initial prices, shape (n_assets,) (shared by every regime)."""
        return self.regimes[0].s0

    def state_spec(self) -> dict[str, tuple[int, ...]]:
        return {"regime": ()}

    def simulate(
        self, rng: np.random.Generator, n_steps: int, *, n_paths: int | None = None
    ) -> Simulation:
        """Simulate regime-switching paths; see `MarketModel.simulate`."""
        steps, lead = _sim_shape(n_steps, n_paths)
        n, dt = self._n_assets, self._dt
        u0 = rng.random(lead)
        u = rng.random((*lead, steps))
        regime = np.empty((*lead, steps + 1), dtype=np.intp)
        regime[..., 0] = _sample_categorical(np.cumsum(self.initial), u0)
        cumulative = np.cumsum(self.transition, axis=1)
        for t in range(steps):
            regime[..., t + 1] = _sample_categorical(cumulative[regime[..., t]], u[..., t])

        eps = rng.standard_normal((*lead, steps, n))
        log_increments = np.empty((*lead, steps, n))
        active = regime[..., :-1]
        for index, model in enumerate(self.regimes):
            selected = active == index
            if not np.any(selected):
                continue
            z = eps[selected]
            if model._chol is not None:
                z = z @ model._chol.T
            drift = (model.mu - 0.5 * model.sigma**2) * dt
            log_increments[selected] = drift + model.sigma * np.sqrt(dt) * z
        return Simulation(
            prices=_prices(self.s0, log_increments),
            state={"regime": regime.astype(np.float64)},
        )

    def _repr_fields(self) -> dict[str, str]:
        return {
            "regimes": "[" + ", ".join(repr(model) for model in self.regimes) + "]",
            "transition": _format(self.transition),
            "initial": str(np.round(self.initial, 6).tolist()),
        }


class GARCH(_Model):
    """GARCH(1,1) volatility clustering on per-step log returns.

    eps_t = sqrt(h_t) z_t, log return_t = mu dt + eps_t and
    h_{t+1} = omega + alpha eps_t^2 + beta h_t, starting from the unconditional variance
    h_0 = omega / (1 - alpha - beta). The z_t are correlated across assets by `corr`.

    Args:
        omega: Per-step (not annualised) variance intercept (> 0).
        alpha: ARCH coefficient (>= 0).
        beta: GARCH coefficient (>= 0); alpha + beta must be < 1.
        mu: Annualised drift of the log price.
        s0: Initial price (> 0).
        corr: Optional (n_assets, n_assets) correlation matrix of the shocks z.
        dt: Step length in years.
        n_assets: Number of assets when every parameter is a scalar.

    State:
        "variance": per-step shape (n_assets,), the conditional variance h_t of the increment
        starting at t.

    Raises:
        ValueError: On inconsistent shapes or out-of-range parameters.
    """

    omega: NDArray[np.float64]
    alpha: NDArray[np.float64]
    beta: NDArray[np.float64]
    mu: NDArray[np.float64]
    s0: NDArray[np.float64]
    corr: NDArray[np.float64] | None
    _chol: NDArray[np.float64] | None

    def __init__(
        self,
        omega: ArrayLike = 1e-6,
        alpha: ArrayLike = 0.08,
        beta: ArrayLike = 0.9,
        *,
        mu: ArrayLike = 0.0,
        s0: ArrayLike = 100.0,
        corr: ArrayLike | None = None,
        dt: float = _DEFAULT_DT,
        n_assets: int | None = None,
    ) -> None:
        n = _infer_n_assets(n_assets, corr, omega=omega, alpha=alpha, beta=beta, mu=mu, s0=s0)
        omega_arr = _per_asset("omega", omega, n)
        alpha_arr = _per_asset("alpha", alpha, n)
        beta_arr = _per_asset("beta", beta, n)
        _require(bool(np.all(omega_arr > 0)), "omega must be positive.")
        _require(bool(np.all(alpha_arr >= 0)), "alpha must be non-negative.")
        _require(bool(np.all(beta_arr >= 0)), "beta must be non-negative.")
        _require(bool(np.all(alpha_arr + beta_arr < 1)), "alpha + beta must be < 1.")
        corr_arr, chol = _correlation(corr, n)
        self._set(
            _n_assets=n,
            _dt=_positive_dt(dt),
            omega=omega_arr,
            alpha=alpha_arr,
            beta=beta_arr,
            mu=_per_asset("mu", mu, n),
            s0=_positive_prices(s0, n),
            corr=corr_arr,
            _chol=chol,
        )

    @property
    def unconditional_variance(self) -> NDArray[np.float64]:
        """Per-step unconditional variance omega / (1 - alpha - beta), shape (n_assets,)."""
        out: NDArray[np.float64] = self.omega / (1.0 - self.alpha - self.beta)
        return out

    def state_spec(self) -> dict[str, tuple[int, ...]]:
        return {"variance": (self._n_assets,)}

    def simulate(
        self, rng: np.random.Generator, n_steps: int, *, n_paths: int | None = None
    ) -> Simulation:
        """Simulate GARCH(1,1) paths; see `MarketModel.simulate`."""
        steps, lead = _sim_shape(n_steps, n_paths)
        n = self._n_assets
        z = _normals(rng, (*lead, steps), n, self._chol)
        variance = np.empty((*lead, steps + 1, n))
        shocks = np.empty((*lead, steps, n))
        h = np.zeros((*lead, n)) + self.unconditional_variance
        for t in range(steps):
            variance[..., t, :] = h
            eps = np.sqrt(h) * z[..., t, :]
            shocks[..., t, :] = eps
            h = self.omega + self.alpha * eps**2 + self.beta * h
        variance[..., steps, :] = h
        log_increments = self.mu * self._dt + shocks
        return Simulation(prices=_prices(self.s0, log_increments), state={"variance": variance})

    def _repr_fields(self) -> dict[str, str]:
        return {
            "n_assets": str(self._n_assets),
            "omega": _format(self.omega),
            "alpha": _format(self.alpha),
            "beta": _format(self.beta),
            "mu": _format(self.mu),
            "s0": _format(self.s0),
            "corr": _format(self.corr),
            "dt": _format_dt(self._dt),
        }


class MeanReverting(_Model):
    """Exponential Ornstein-Uhlenbeck prices: the log price reverts to log(mean).

    With x = log S and m = log(mean), the exact discretisation is
    x_{t+1} = m + (x_t - m) e^{-theta dt} + sigma sqrt((1 - e^{-2 theta dt}) / (2 theta)) z,
    with z correlated across assets by `corr`.

    Args:
        theta: Mean-reversion speed (> 0), per year.
        mean: Price level the log price reverts to (> 0).
        sigma: Annualised volatility of the log price (>= 0).
        s0: Initial price (> 0).
        corr: Optional (n_assets, n_assets) correlation matrix of the shocks.
        dt: Step length in years.
        n_assets: Number of assets when every parameter is a scalar.

    Raises:
        ValueError: On inconsistent shapes or out-of-range parameters.
    """

    theta: NDArray[np.float64]
    mean: NDArray[np.float64]
    sigma: NDArray[np.float64]
    s0: NDArray[np.float64]
    corr: NDArray[np.float64] | None
    _chol: NDArray[np.float64] | None

    def __init__(
        self,
        theta: ArrayLike = 5.0,
        mean: ArrayLike = 100.0,
        sigma: ArrayLike = 0.2,
        *,
        s0: ArrayLike = 100.0,
        corr: ArrayLike | None = None,
        dt: float = _DEFAULT_DT,
        n_assets: int | None = None,
    ) -> None:
        n = _infer_n_assets(n_assets, corr, theta=theta, mean=mean, sigma=sigma, s0=s0)
        theta_arr = _per_asset("theta", theta, n)
        mean_arr = _per_asset("mean", mean, n)
        sigma_arr = _per_asset("sigma", sigma, n)
        _require(bool(np.all(theta_arr > 0)), "theta must be positive.")
        _require(bool(np.all(mean_arr > 0)), "mean must be positive.")
        _require(bool(np.all(sigma_arr >= 0)), "sigma must be non-negative.")
        corr_arr, chol = _correlation(corr, n)
        self._set(
            _n_assets=n,
            _dt=_positive_dt(dt),
            theta=theta_arr,
            mean=mean_arr,
            sigma=sigma_arr,
            s0=_positive_prices(s0, n),
            corr=corr_arr,
            _chol=chol,
        )

    def simulate(
        self, rng: np.random.Generator, n_steps: int, *, n_paths: int | None = None
    ) -> Simulation:
        """Simulate exponential OU paths; see `MarketModel.simulate`. The state is empty."""
        steps, lead = _sim_shape(n_steps, n_paths)
        n = self._n_assets
        z = _normals(rng, (*lead, steps), n, self._chol)
        decay, scale = _ou_coefficients(self.theta, self.sigma, self._dt)
        target = np.log(self.mean) - np.log(self.s0)  # log(mean) relative to log(s0)
        log_rel = np.zeros((*lead, steps + 1, n))
        x = np.zeros((*lead, n))
        for t in range(steps):
            x = target + (x - target) * decay + scale * z[..., t, :]
            log_rel[..., t + 1, :] = x
        prices: NDArray[np.float64] = self.s0 * np.exp(log_rel)
        return Simulation(prices=prices)

    def _repr_fields(self) -> dict[str, str]:
        return {
            "n_assets": str(self._n_assets),
            "theta": _format(self.theta),
            "mean": _format(self.mean),
            "sigma": _format(self.sigma),
            "s0": _format(self.s0),
            "corr": _format(self.corr),
            "dt": _format_dt(self._dt),
        }


class CointegratedPair(_Model):
    """Two cointegrated assets: a shared GBM trend plus a stationary OU spread.

    The common log trend x (x_0 = 0) is a Brownian motion with drift (mu - sigma^2 / 2) and
    volatility sigma; the spread s (s_0 = 0) is a zero-mean OU process discretised exactly.
    log P1 = log s0[0] + x_t + s_t / 2 and log P2 = log s0[1] + x_t - s_t / 2, so each price
    wanders while log(P1 / P2) - log(s0[0] / s0[1]) = s_t mean-reverts.

    Args:
        mu: Annualised drift of the common trend (scalar).
        sigma: Annualised volatility of the common trend (scalar, >= 0).
        theta: Mean-reversion speed of the spread (scalar, > 0), per year.
        spread_sigma: Annualised volatility of the spread (scalar, >= 0).
        s0: Initial prices, a scalar or a length-2 sequence (> 0).
        dt: Step length in years.

    State:
        "spread": per-step shape (), the spread s_t.

    Raises:
        ValueError: On non-scalar or out-of-range parameters.
    """

    mu: float
    sigma: float
    theta: float
    spread_sigma: float
    s0: NDArray[np.float64]

    def __init__(
        self,
        mu: float = 0.05,
        sigma: float = 0.2,
        theta: float = 5.0,
        spread_sigma: float = 0.1,
        *,
        s0: ArrayLike = (100.0, 100.0),
        dt: float = _DEFAULT_DT,
    ) -> None:
        sigma_f = _scalar("sigma", sigma)
        theta_f = _scalar("theta", theta)
        spread_sigma_f = _scalar("spread_sigma", spread_sigma)
        _require(sigma_f >= 0, "sigma must be non-negative.")
        _require(theta_f > 0, "theta must be positive.")
        _require(spread_sigma_f >= 0, "spread_sigma must be non-negative.")
        self._set(
            _n_assets=2,
            _dt=_positive_dt(dt),
            mu=_scalar("mu", mu),
            sigma=sigma_f,
            theta=theta_f,
            spread_sigma=spread_sigma_f,
            s0=_positive_prices(s0, 2),
        )

    def state_spec(self) -> dict[str, tuple[int, ...]]:
        return {"spread": ()}

    def simulate(
        self, rng: np.random.Generator, n_steps: int, *, n_paths: int | None = None
    ) -> Simulation:
        """Simulate the pair; see `MarketModel.simulate`."""
        steps, lead = _sim_shape(n_steps, n_paths)
        dt = self._dt
        z = rng.standard_normal((*lead, steps, 2))
        trend_drift = (self.mu - 0.5 * self.sigma**2) * dt
        trend_increments = trend_drift + self.sigma * np.sqrt(dt) * z[..., 0]
        trend = np.concatenate([np.zeros((*lead, 1)), np.cumsum(trend_increments, axis=-1)], -1)

        decay, scale = _ou_coefficients(self.theta, self.spread_sigma, dt)
        spread = np.zeros((*lead, steps + 1))
        s = np.zeros(lead)
        for t in range(steps):
            s = s * decay + scale * z[..., t, 1]
            spread[..., t + 1] = s
        log_rel = np.stack([trend + 0.5 * spread, trend - 0.5 * spread], axis=-1)
        prices: NDArray[np.float64] = self.s0 * np.exp(log_rel)
        return Simulation(prices=prices, state={"spread": spread})

    def _repr_fields(self) -> dict[str, str]:
        return {
            "mu": _format(self.mu),
            "sigma": _format(self.sigma),
            "theta": _format(self.theta),
            "spread_sigma": _format(self.spread_sigma),
            "s0": _format(self.s0),
            "dt": _format_dt(self._dt),
        }


class BlockBootstrap(_Model):
    """Circular block bootstrap of historical log returns.

    Each path concatenates blocks of `block_size` consecutive rows of `returns`, each starting
    at a uniformly random row and wrapping around circularly, truncated to n_steps rows. Whole
    rows are resampled, so cross-asset correlation and short-range dependence are preserved.

    Args:
        returns: Historical per-step log returns, shape (T, n_assets) or (T,) for one asset.
        block_size: Rows per block, 1 <= block_size <= T.
        s0: Initial price (> 0), scalar or shape (n_assets,).
        dt: Step length in years of the historical data.

    Raises:
        ValueError: If returns are empty or non-finite, or block_size is out of range.
    """

    returns: NDArray[np.float64]
    block_size: int
    s0: NDArray[np.float64]

    def __init__(
        self,
        returns: ArrayLike,
        *,
        block_size: int = 20,
        s0: ArrayLike = 100.0,
        dt: float = _DEFAULT_DT,
    ) -> None:
        data = np.asarray(returns, dtype=np.float64)
        if data.ndim == 1:
            data = data[:, None]
        _require(data.ndim == 2, f"returns must be 1-D or 2-D, got shape {data.shape}.")
        _require(data.shape[0] >= 1 and data.shape[1] >= 1, "returns must not be empty.")
        _require(bool(np.all(np.isfinite(data))), "returns must be finite.")
        size = _as_int("block_size", block_size)
        _require(
            1 <= size <= data.shape[0],
            f"block_size must be in [1, {data.shape[0]}], got {size}.",
        )
        n = data.shape[1]
        self._set(
            _n_assets=n,
            _dt=_positive_dt(dt),
            returns=_readonly(data),
            block_size=size,
            s0=_positive_prices(s0, n),
        )

    def simulate(
        self, rng: np.random.Generator, n_steps: int, *, n_paths: int | None = None
    ) -> Simulation:
        """Resample paths; see `MarketModel.simulate`. The state is empty."""
        steps, lead = _sim_shape(n_steps, n_paths)
        n_rows, size = self.returns.shape[0], self.block_size
        n_blocks = -(-steps // size)
        starts = rng.integers(0, n_rows, size=(*lead, n_blocks))
        rows = (starts[..., None] + np.arange(size)) % n_rows
        rows = rows.reshape((*lead, n_blocks * size))[..., :steps]
        return Simulation(prices=_prices(self.s0, self.returns[rows]))

    def _repr_fields(self) -> dict[str, str]:
        return {
            "returns": f"<{self.returns.shape[0]}x{self.returns.shape[1]} array>",
            "block_size": str(self.block_size),
            "s0": _format(self.s0),
            "dt": _format_dt(self._dt),
        }


class Randomized(_Model):
    """Domain randomisation: a freshly parameterised model for every simulated path.

    `factory(rng)` builds a model with randomly drawn parameters. The model returned for
    `np.random.default_rng(0)` is kept as `reference` and fixes n_assets, dt and state_spec;
    every later model must match them.

    Args:
        factory: Callable mapping a generator to a `MarketModel`.

    Raises:
        TypeError: If factory is not callable or does not return a `MarketModel`.
        ValueError: (from `simulate`) if a built model's n_assets, dt or state_spec differ
            from the reference.
    """

    factory: Callable[[np.random.Generator], MarketModel]
    reference: MarketModel
    _spec: dict[str, tuple[int, ...]]

    def __init__(self, factory: Callable[[np.random.Generator], MarketModel]) -> None:
        if not callable(factory):
            raise TypeError("factory must be callable.")
        reference = factory(np.random.default_rng(0))
        if not isinstance(reference, MarketModel):
            raise TypeError(f"factory must return a MarketModel, got {type(reference).__name__}.")
        self._set(
            _n_assets=reference.n_assets,
            _dt=float(reference.dt),
            factory=factory,
            reference=reference,
            _spec={key: tuple(shape) for key, shape in reference.state_spec().items()},
        )

    def state_spec(self) -> dict[str, tuple[int, ...]]:
        return dict(self._spec)

    def _build(self, rng: np.random.Generator) -> MarketModel:
        model = self.factory(rng)
        if not isinstance(model, MarketModel):
            raise TypeError(f"factory must return a MarketModel, got {type(model).__name__}.")
        spec = {key: tuple(shape) for key, shape in model.state_spec().items()}
        if model.n_assets != self._n_assets or model.dt != self._dt or spec != self._spec:
            raise ValueError(
                "factory built a model whose n_assets, dt or state_spec differ from the "
                f"reference: {model!r} vs {self.reference!r}."
            )
        return model

    def simulate(
        self, rng: np.random.Generator, n_steps: int, *, n_paths: int | None = None
    ) -> Simulation:
        """Simulate one freshly built model per path; see `MarketModel.simulate`."""
        steps, lead = _sim_shape(n_steps, n_paths)
        if not lead:
            return self._build(rng).simulate(rng, steps)
        sims = [self._build(rng).simulate(rng, steps) for _ in range(lead[0])]
        mask: NDArray[np.bool_] | None = None
        if any(sim.mask is not None for sim in sims):
            mask = np.stack(
                [
                    np.ones(sim.prices.shape, dtype=bool) if sim.mask is None else sim.mask
                    for sim in sims
                ]
            )
        state = {key: np.stack([sim.state[key] for sim in sims]) for key in self._spec}
        return Simulation(prices=np.stack([sim.prices for sim in sims]), mask=mask, state=state)

    def _repr_fields(self) -> dict[str, str]:
        name = getattr(self.factory, "__qualname__", type(self.factory).__name__)
        return {"factory": name, "reference": repr(self.reference)}


class ListingProcess(_Model):
    """A dynamic universe: assets of an inner model list and delist over time.

    Each asset follows an independent two-state Markov chain. At t = 0 it is listed with
    probability `initial_listed`; after every step an unlisted asset lists with probability
    1 - exp(-listing_rate dt) and a listed asset delists with probability
    1 - exp(-delisting_rate dt). Whenever fewer than `min_listed` assets are listed (also at
    t = 0), the lowest-index unlisted assets are listed until `min_listed` are. In the long run
    an asset is listed a fraction p_list / (p_list + p_delist) of the steps, where p_list and
    p_delist are the two step probabilities above; this tends to
    listing_rate / (listing_rate + delisting_rate) as dt -> 0.

    Prices come from the inner model's path; where unlisted they are NaN and `mask` is False.
    If the inner simulation has its own mask, the two are combined with a logical AND
    (`min_listed` applies only to this process's own listing flags). The inner model's state
    passes through unchanged.

    Args:
        model: The inner market model.
        initial_listed: Probability that an asset is listed at t = 0, in [0, 1] (scalar or
            shape (n_assets,)).
        listing_rate: Listing intensity per year for an unlisted asset (>= 0).
        delisting_rate: Delisting intensity per year for a listed asset (>= 0).
        min_listed: Minimum number of listed assets at every time, 0 <= min_listed <= n_assets.

    Raises:
        TypeError: If model is not a `MarketModel`.
        ValueError: On out-of-range parameters.
    """

    model: MarketModel
    initial_listed: NDArray[np.float64]
    listing_rate: NDArray[np.float64]
    delisting_rate: NDArray[np.float64]
    min_listed: int

    def __init__(
        self,
        model: MarketModel,
        *,
        initial_listed: ArrayLike = 0.8,
        listing_rate: ArrayLike = 1.0,
        delisting_rate: ArrayLike = 0.25,
        min_listed: int = 1,
    ) -> None:
        if not isinstance(model, MarketModel):
            raise TypeError(f"model must be a MarketModel, got {type(model).__name__}.")
        n = model.n_assets
        initial = _per_asset("initial_listed", initial_listed, n)
        listing = _per_asset("listing_rate", listing_rate, n)
        delisting = _per_asset("delisting_rate", delisting_rate, n)
        floor = _as_int("min_listed", min_listed)
        _require(bool(np.all((initial >= 0) & (initial <= 1))), "initial_listed must be in [0, 1].")
        _require(bool(np.all(listing >= 0)), "listing_rate must be non-negative.")
        _require(bool(np.all(delisting >= 0)), "delisting_rate must be non-negative.")
        _require(0 <= floor <= n, f"min_listed must be in [0, {n}], got {floor}.")
        self._set(
            _n_assets=n,
            _dt=float(model.dt),
            model=model,
            initial_listed=initial,
            listing_rate=listing,
            delisting_rate=delisting,
            min_listed=floor,
        )

    def state_spec(self) -> dict[str, tuple[int, ...]]:
        return self.model.state_spec()

    def _enforce_minimum(self, listed: NDArray[np.bool_]) -> NDArray[np.bool_]:
        if self.min_listed == 0:
            return listed
        deficit = self.min_listed - listed.sum(axis=-1, keepdims=True)
        rank = np.cumsum(~listed, axis=-1)  # 1-based rank of each unlisted asset by index
        out: NDArray[np.bool_] = listed | (~listed & (rank <= deficit))
        return out

    def simulate(
        self, rng: np.random.Generator, n_steps: int, *, n_paths: int | None = None
    ) -> Simulation:
        """Simulate the inner model, then the listing flags; see `MarketModel.simulate`."""
        steps, lead = _sim_shape(n_steps, n_paths)
        inner = self.model.simulate(rng, steps, n_paths=n_paths)
        n, dt = self._n_assets, self._dt
        p_list = -np.expm1(-self.listing_rate * dt)
        p_delist = -np.expm1(-self.delisting_rate * dt)

        listed = np.empty((*lead, steps + 1, n), dtype=bool)
        current = self._enforce_minimum(rng.random((*lead, n)) < self.initial_listed)
        listed[..., 0, :] = current
        u = rng.random((*lead, steps, n))
        for t in range(steps):
            u_t = u[..., t, :]
            current = self._enforce_minimum(np.where(current, u_t >= p_delist, u_t < p_list))
            listed[..., t + 1, :] = current

        mask = listed if inner.mask is None else listed & inner.mask
        prices = np.where(mask, inner.prices, np.nan)
        return Simulation(prices=prices, mask=mask, state=inner.state)

    def _repr_fields(self) -> dict[str, str]:
        return {
            "model": repr(self.model),
            "initial_listed": _format(self.initial_listed),
            "listing_rate": _format(self.listing_rate),
            "delisting_rate": _format(self.delisting_rate),
            "min_listed": str(self.min_listed),
        }
