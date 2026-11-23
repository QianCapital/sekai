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

The built-in models also simulate with torch, on any device: `model.simulate_torch(generator,
n_steps, n_paths=k)` draws from a `torch.Generator` and returns tensors on its device, so a
batched environment on a GPU gets fresh paths without copying them from the host. It is the
same model (distributions, parameters, shapes, mask and state) as `simulate`, not the same
numbers. torch is imported only then.
"""

from __future__ import annotations

import abc
import math
import operator
import weakref
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeGuard

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sekai.backend import Array, is_tensor

if TYPE_CHECKING:
    import torch

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
    """Simulated market paths returned by `MarketModel.simulate`, as NumPy arrays, or by
    `MarketModel.simulate_torch`, as torch tensors on one device.

    Attributes:
        prices: Prices of shape (n_steps + 1, n_assets), or (n_paths, n_steps + 1, n_assets)
            when simulated with `n_paths`. Row 0 (along the time axis) is the initial price.
            NaN where an asset is not listed.
        mask: Boolean array shaped like `prices`, True where the asset is listed and tradable,
            or None when every entry is valid.
        state: Latent or auxiliary series keyed by name (see `MarketModel.state_spec`). Each
            array's leading dimensions equal `prices.shape[:-1]`, i.e. (n_steps + 1,) or
            (n_paths, n_steps + 1), followed by the model's per-step shape for that key.

    With tensor prices, the mask (a bool tensor) and the state are tensors on their device.

    Raises:
        ValueError: If the shapes of `prices`, `mask` and `state` are inconsistent, or tensor
            prices come with a mask or state that are not tensors on their device.
    """

    prices: NDArray[np.float64] | Array
    mask: NDArray[np.bool_] | Array | None = None
    state: Mapping[str, NDArray[np.float64] | Array] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if is_tensor(self.prices):
            self._check_tensors()
            return
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

    def _check_tensors(self) -> None:
        """The checks of `__post_init__` for tensor prices (read from shapes, not values)."""
        import torch

        prices = self.prices
        if prices.ndim not in (2, 3):
            raise ValueError(
                "prices must be a 2-D (n_steps + 1, n_assets) or 3-D "
                "(n_paths, n_steps + 1, n_assets) array."
            )
        shape, device = tuple(prices.shape), prices.device
        mask = self.mask
        if mask is not None and not (
            isinstance(mask, torch.Tensor)
            and mask.dtype == torch.bool
            and tuple(mask.shape) == shape
            and mask.device == device
        ):
            raise ValueError(f"mask must be a boolean tensor of shape {shape} on {device}.")
        lead = shape[:-1]
        for key, value in self.state.items():
            if not isinstance(value, torch.Tensor) or value.device != device:
                raise ValueError(f"state[{key!r}] must be a tensor on {device}, as prices are.")
            if tuple(value.shape)[: len(lead)] != lead:
                raise ValueError(
                    f"state[{key!r}] has shape {tuple(value.shape)}; its leading dimensions "
                    f"must be {lead}."
                )


class MarketModel(abc.ABC):
    """A generator of synthetic price paths for a fixed universe of `n_assets` assets.

    Subclasses implement `simulate`, which must be vectorised over paths and assets, return
    strictly positive prices wherever an asset is listed and honour the shapes documented on
    `Simulation`. They may also implement `simulate_torch`, to simulate on a device.
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

    def simulate_torch(
        self,
        generator: torch.Generator,
        n_steps: int,
        *,
        n_paths: int | None = None,
        dtype: torch.dtype | None = None,
    ) -> Simulation:
        """Simulate price paths with torch, on the generator's device.

        The same model as `simulate` (the same distributions and parameters, and a
        `Simulation` of the same shapes, mask and state keys), computed with torch on
        `generator.device`: the tensors are made there and nothing is read back, so a
        batched environment on a GPU gets fresh paths without copying them from the host.
        The numbers differ from those of `simulate`. Every built-in model implements it,
        copying its parameters to a device on its first simulation there and keeping them
        for later ones (`Randomized` copies those it draws, on each call), so a model's
        arrays must not be edited in place (they are read-only, though an unpickled
        model's are not). A custom model may implement it; by default it raises
        `NotImplementedError`, and such a model simulates on the host only (`simulate`).

        Args:
            generator: Source of randomness; the paths are simulated on its device.
            n_steps: Number of steps (>= 1); the time axis has n_steps + 1 entries.
            n_paths: Number of independent paths (>= 1). None gives a single path without a
                leading paths axis; an int (even 1) adds a leading axis of that size.
            dtype: `torch.float32` or `torch.float64`, the dtype of the prices and state
                (default: torch's default dtype, usually float32; MPS has no float64). In
                half precision a simulation would not be the same model.

        Returns:
            A `Simulation` of tensors, shaped as `simulate` shapes its arrays.

        Raises:
            NotImplementedError: If the model has no torch simulation.
            ValueError: If n_steps or n_paths is smaller than 1.
            TypeError: If n_steps or n_paths is not an integer, `generator` is not a
                `torch.Generator`, or `dtype` is not `torch.float32` or `torch.float64`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has no torch simulation: simulate it on the host with "
            "simulate() (ModelScenarios without device=), or implement simulate_torch()."
        )


class _Model(MarketModel):
    """Shared plumbing for the built-in models: read-only attributes, a compact repr, and the
    torch simulation (see "Simulation with torch" below)."""

    _n_assets: int
    _dt: float

    @property
    def n_assets(self) -> int:
        return self._n_assets

    @property
    def dt(self) -> float:
        return self._dt

    def simulate_torch(
        self,
        generator: torch.Generator,
        n_steps: int,
        *,
        n_paths: int | None = None,
        dtype: torch.dtype | None = None,
    ) -> Simulation:
        """Simulate with torch on the generator's device; see `MarketModel.simulate_torch`."""
        steps, lead = _sim_shape(n_steps, n_paths)
        if not _has_torch_simulation(self):
            raise NotImplementedError(
                f"{type(self).__name__} overrides simulate() but not simulate_torch(), so it has "
                "no torch simulation of its own: simulate it on the host with simulate(), or "
                "implement simulate_torch()."
            )
        ctx = _TorchContext(generator, dtype)
        sim = self._torch_simulate(ctx, _device_params(self, ctx), lead[0] if lead else 1, steps)
        return sim if lead else _first_path(sim)

    def _torch_key(self) -> Hashable | None:
        """What the torch simulation depends on besides `_torch_params()` (the class, whether
        there is a correlation, ...): models with equal keys simulate together, their
        parameters stacked. None for a model that draws its parameters on each call."""
        return (type(self),)

    def _torch_params(self) -> dict[str, Any]:
        """The fixed parameters the torch simulation reads, as float64 arrays (computed in
        float64 on the host; an inner model's as a nested dict)."""
        return {}

    def _torch_simulate(
        self, ctx: _TorchContext, params: dict[str, Any], n_paths: int, n_steps: int
    ) -> Simulation:
        """The torch simulation, with a leading paths axis, from `params`: the tensors of
        `_torch_params()` with a leading axis of 1 (this model for every path) or n_paths
        (one model per path, of equal keys). It reads its numbers from `params` only."""
        raise NotImplementedError

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
# Simulation with torch
# --------------------------------------------------------------------------------------------
#
# A built-in model simulates with torch from tensors of the parameters it reads,
# `_torch_params()`: float64 arrays computed on the host, copied to a device on the model's
# first simulation there and kept for its lifetime (`_device_params`), so that later
# simulations copy nothing to the device. `_torch_simulate` always simulates with a leading
# paths axis, and takes parameters with a leading axis of 1 (one model for every path) or
# n_paths (a model per path: `Randomized` stacks the parameters of models with equal
# `_torch_key()` and simulates them together). Nothing is read back from the device, and no
# operation has a shape that depends on the values (no boolean indexing), so a simulation on
# a GPU is queued without waiting for it. Recursions (variance, regimes, listings) are loops
# over time of operations batched over paths and assets: they draw their shocks time-major,
# (step, path, asset), so that each step reads contiguous ones (and parameters of shape
# (L, n) broadcast against them as they are), and stack the steps' values once at the end.


class _TorchContext:
    """Where a torch simulation runs: its generator, and the device and dtype of its tensors."""

    __slots__ = ("device", "dtype", "generator")

    def __init__(self, generator: Any, dtype: Any) -> None:
        import torch

        if not isinstance(generator, torch.Generator):
            raise TypeError(f"generator must be a torch.Generator, got {type(generator).__name__}.")
        dtype = torch.get_default_dtype() if dtype is None else dtype
        if dtype not in (torch.float32, torch.float64):
            raise TypeError(f"dtype must be torch.float32 or torch.float64, got {dtype!r}.")
        device = generator.device
        if device.type != "cpu" and device.index is None:
            # A generator of torch.Generator(device="cuda") has no index: tensors made for
            # it go to the current device, which keys the parameters copied there.
            device = torch.empty(0, device=device).device
        self.generator = generator
        self.device = device
        self.dtype = dtype

    def normal(self, *shape: int) -> torch.Tensor:
        import torch

        return torch.randn(shape, generator=self.generator, dtype=self.dtype, device=self.device)

    def uniform(self, *shape: int) -> torch.Tensor:
        import torch

        return torch.rand(shape, generator=self.generator, dtype=self.dtype, device=self.device)

    def integers(self, high: int, *shape: int) -> torch.Tensor:
        """int64 draws in [0, high)."""
        import torch

        return torch.randint(high, shape, generator=self.generator, device=self.device)


def _owner(model: MarketModel, name: str) -> type:
    """The class whose method `name` the model has."""
    return next(c for c in type(model).__mro__ if name in vars(c))


def _has_torch_simulation(model: _Model) -> bool:
    """Whether the model's torch simulation is that of its `simulate`: a subclass of a
    built-in model that overrides `simulate` alone must not inherit its parent's."""
    return issubclass(_owner(model, "_torch_simulate"), _owner(model, "simulate"))


def _simulates_on_torch(model: MarketModel) -> TypeGuard[_Model]:
    """Whether a model in another one (`ListingProcess`, `Randomized`) can be simulated with
    its parameters, `_torch_simulate`: a built-in model with its torch simulation (and not
    a subclass with its own `simulate_torch`, which is called instead)."""
    return (
        isinstance(model, _Model)
        and _owner(model, "simulate_torch") is _Model
        and _has_torch_simulation(model)
    )


# The parameters of each model simulated on a device, per (device, dtype); freed with it.
_DEVICE_PARAMS: weakref.WeakKeyDictionary[MarketModel, dict[tuple[Any, Any], dict[str, Any]]] = (
    weakref.WeakKeyDictionary()
)


def _device_params(model: _Model, ctx: _TorchContext) -> dict[str, Any]:
    """`model._torch_params()` as tensors on the context's device, in its dtype, with a
    leading axis of 1: copied there on the model's first simulation with that device and
    dtype, and reused by later ones."""
    key = (ctx.device, ctx.dtype)
    try:
        cache = _DEVICE_PARAMS.setdefault(model, {})
    except TypeError:  # a subclass that is not hashable: copy them for every simulation
        return _to_device([model._torch_params()], ctx)
    params = cache.get(key)
    if params is None:
        params = cache[key] = _to_device([model._torch_params()], ctx)
    return params


def _to_device(params: Sequence[dict[str, Any]], ctx: _TorchContext) -> dict[str, Any]:
    """The `_torch_params()` of models with equal keys as tensors on the context's device, in
    its dtype, stacked along a new leading axis, all copied in one transfer (and converted to
    the dtype on the host: MPS has no float64)."""
    import torch

    arrays: list[NDArray[np.float64]] = []

    def stack(trees: list[Any]) -> Any:  # the layout of the parameters: indices into arrays
        if trees[0] is None:  # an inner model without parameters of this kind
            return None
        if isinstance(trees[0], dict):
            return {key: stack([tree[key] for tree in trees]) for key in trees[0]}
        arrays.append(np.stack([np.asarray(tree, dtype=np.float64) for tree in trees]))
        return len(arrays) - 1

    layout = stack(list(params))
    tensors: list[torch.Tensor] = []
    if arrays:
        host = torch.from_numpy(np.concatenate([a.ravel() for a in arrays])).to(ctx.dtype)
        pieces = torch.split(host.to(ctx.device), [a.size for a in arrays])
        tensors = [piece.view(a.shape) for piece, a in zip(pieces, arrays)]

    def build(node: Any) -> Any:
        if node is None:
            return None
        if isinstance(node, dict):
            return {key: build(value) for key, value in node.items()}
        return tensors[node]

    out: dict[str, Any] = build(layout)
    return out


def _first_path(sim: Simulation) -> Simulation:
    """A simulation of one path without its paths axis (`n_paths=None`)."""
    return Simulation(
        prices=sim.prices[0],
        mask=None if sim.mask is None else sim.mask[0],
        state={key: value[0] for key, value in sim.state.items()},
    )


def _correlated(
    z: torch.Tensor, chol_t: torch.Tensor | None, *, time_major: bool = False
) -> torch.Tensor:
    """Shocks `z` (P, S, n), or (S, P, n) when `time_major`, correlated across the last axis,
    `z @ chol.T`, given the transposed Cholesky factors `chol_t`: (1, n, n) for every path, or
    (P, n, n) for each."""
    import torch

    if chol_t is None:
        return z
    if chol_t.shape[0] == 1:
        out: torch.Tensor = z @ chol_t[0]
        return out
    return torch.einsum("spi,pij->spj" if time_major else "psi,pij->psj", z, chol_t)


def _torch_prices(s0: torch.Tensor, log_increments: torch.Tensor) -> torch.Tensor:
    """Prices (P, n_steps + 1, n) from log increments (P, n_steps, n); `s0` is (L, n)."""
    import torch

    first = torch.zeros_like(log_increments[:, :1])
    log_paths = torch.cat([first, torch.cumsum(log_increments, dim=1)], dim=1)
    return s0[:, None] * torch.exp(log_paths)


def _host_rng(generator: torch.Generator) -> np.random.Generator:
    """A NumPy generator for randomness drawn on the host, seeded from `generator`'s state,
    which then moves on: a given state gives the same numbers, and the next call others.
    Every device's generator keeps its state on the host, so nothing is read from the
    device."""
    import torch

    rng = np.random.default_rng(generator.get_state().numpy())
    # Move the state on, by a float32 draw: every device has float32 (MPS has no float64),
    # and the state then moves on alike whatever torch's default dtype.
    torch.empty(1, device=generator.device, dtype=torch.float32).uniform_(generator=generator)
    return rng


def _poisson_table(rate: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Inverse-CDF tables of Poisson(rate), for each entry of `rate` (n,): the first count
    `low` (n,) and the CDF from it, `cdf` (n, K), so that `low + #{k : cdf[:, k] <= u}` for
    u uniform in [0, 1) is a Poisson draw. The counts outside a table (more than 9 standard
    deviations and 10 from the mean) have a total probability below 1e-16. K, a multiple of
    16, is the same over wide ranges of rates."""
    mean = np.floor(rate)
    width = np.ceil(9.0 * np.sqrt(rate)) + 10.0
    low = np.maximum(mean - width, 0.0)
    size = 16 * math.ceil((float(np.max(mean + width - low)) + 1.0) / 16.0)
    counts = low[:, None] + np.arange(size)
    # log(k!) for the counts of each table, summed up from log(low!)
    log_factorial = np.array([math.lgamma(x + 1.0) for x in low])[:, None] + np.concatenate(
        [np.zeros((len(low), 1)), np.cumsum(np.log(counts[:, 1:]), axis=1)], axis=1
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        log_pmf = counts * np.log(rate)[:, None] - rate[:, None] - log_factorial
    pmf = np.where(rate[:, None] > 0, np.exp(log_pmf), (counts == 0).astype(np.float64))
    return low, np.minimum(np.cumsum(pmf, axis=1), 1.0)


def _poisson_counts(
    ctx: _TorchContext, low: torch.Tensor, cdf: torch.Tensor, n_paths: int, n_steps: int
) -> torch.Tensor:
    """Poisson counts (P, n_steps, n), as floats, by inverting the tables of `_poisson_table`
    (`low` (L, n) and `cdf` (L, n, K)), with operations every device has (torch's Poisson
    sampler is not on MPS).

    A uniform of exactly 0 (one float32 draw in 2**24) would invert to a table's lowest
    count, far below the mean of a large rate; it stands for the half step above 0 instead
    (torch's uniforms are multiples of eps / 2)."""
    import torch

    n = cdf.shape[1]
    half_step = torch.finfo(ctx.dtype).eps / 4
    if cdf.shape[0] == 1:  # the same tables for every path
        u = ctx.uniform(n, n_paths * n_steps)
        u.masked_fill_(u == 0, half_step)
        index = torch.searchsorted(cdf[0], u, right=True).view(n, n_paths, n_steps)
        index = index.permute(1, 2, 0)
    else:
        u = ctx.uniform(n_paths, n, n_steps)
        u.masked_fill_(u == 0, half_step)
        index = torch.searchsorted(cdf, u, right=True)
        index = index.transpose(1, 2)
    return low[:, None] + index.to(ctx.dtype)


def _torch_categorical(cumulative: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """`_sample_categorical` on tensors: categories (P,) for uniforms `u` (P,) given cumulative
    probabilities (P, K) or (1, K)."""
    index: torch.Tensor = (u[:, None] >= cumulative).sum(dim=-1)
    return index.clamp_(max=cumulative.shape[-1] - 1)


# The most comparisons a regime chain makes at once ((steps, P, K, K)): it works out its
# steps' moves a chunk of steps at a time, so that its memory does not grow with n_steps.
_MOVES_ELEMENTS = 2**22


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

    def _torch_key(self) -> Hashable:
        return (type(self), self._chol is not None)

    def _torch_params(self) -> dict[str, Any]:
        params = {
            "drift": (self.mu - 0.5 * self.sigma**2) * self._dt,
            "vol": self.sigma * np.sqrt(self._dt),
            "s0": self.s0,
        }
        if self._chol is not None:
            params["chol_t"] = self._chol.T
        return params

    def _torch_simulate(
        self, ctx: _TorchContext, params: dict[str, Any], n_paths: int, n_steps: int
    ) -> Simulation:
        z = _correlated(ctx.normal(n_paths, n_steps, self._n_assets), params.get("chol_t"))
        log_increments = params["drift"][:, None] + params["vol"][:, None] * z
        return Simulation(prices=_torch_prices(params["s0"], log_increments))

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

    def _torch_key(self) -> Hashable:
        return (type(self), self._chol is not None)

    def _torch_params(self) -> dict[str, Any]:
        n, dt = self._n_assets, self._dt
        params = {
            "mu_dt": self.mu * dt,
            "half_dt": np.full(n, 0.5 * dt),
            "kappa_dt": self.kappa * dt,
            "theta": self.theta,
            "sqrt_dt": np.full(n, np.sqrt(dt)),
            "xi_sqrt_dt": self.xi * np.sqrt(dt),
            "rho": self.rho,
            "rho_c": np.sqrt(1.0 - self.rho**2),
            "v0": self.v0,
            "s0": self.s0,
        }
        if self._chol is not None:
            params["chol_t"] = self._chol.T
        return params

    def _torch_simulate(
        self, ctx: _TorchContext, params: dict[str, Any], n_paths: int, n_steps: int
    ) -> Simulation:
        import torch

        p, n = params, self._n_assets
        z1 = _correlated(ctx.normal(n_steps, n_paths, n), p.get("chol_t"), time_major=True)
        z2 = p["rho"] * z1 + p["rho_c"] * ctx.normal(n_steps, n_paths, n)
        dw1 = p["sqrt_dt"] * z1  # sqrt(v+ dt) z1 = sqrt(v+) dw1
        dw2 = p["xi_sqrt_dt"] * z2
        mu_dt, half_dt, kappa_dt, theta = p["mu_dt"], p["half_dt"], p["kappa_dt"], p["theta"]
        x = torch.zeros((n_paths, n), dtype=ctx.dtype, device=ctx.device)
        v = p["v0"].expand(n_paths, n)
        log_rel, variance = [x], []
        for t in range(n_steps):
            v_plus = torch.clamp(v, min=0.0)
            root = torch.sqrt(v_plus)
            x = torch.addcmul(x + mu_dt, half_dt, v_plus, value=-1.0).addcmul_(root, dw1[t])
            v = torch.addcmul(v, kappa_dt, theta - v_plus).addcmul_(root, dw2[t])
            log_rel.append(x)
            variance.append(v_plus)
        variance.append(torch.clamp(v, min=0.0))
        prices = p["s0"][:, None] * torch.exp(torch.stack(log_rel, dim=1))
        return Simulation(prices=prices, state={"variance": torch.stack(variance, dim=1)})

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

    def _torch_key(self) -> Hashable:
        table = _poisson_table(self.jump_intensity * self._dt)[1]
        return (type(self), self._chol is not None, table.shape[1])

    def _torch_params(self) -> dict[str, Any]:
        dt = self._dt
        k = np.expm1(self.jump_mean + 0.5 * self.jump_std**2)
        low, cdf = _poisson_table(self.jump_intensity * dt)
        params = {
            "drift": (self.mu - 0.5 * self.sigma**2 - self.jump_intensity * k) * dt,
            "vol": self.sigma * np.sqrt(dt),
            "jump_mean": self.jump_mean,
            "jump_std": self.jump_std,
            "jump_low": low,
            "jump_cdf": cdf,
            "s0": self.s0,
        }
        if self._chol is not None:
            params["chol_t"] = self._chol.T
        return params

    def _torch_simulate(
        self, ctx: _TorchContext, params: dict[str, Any], n_paths: int, n_steps: int
    ) -> Simulation:
        import torch

        p, n = params, self._n_assets
        z = _correlated(ctx.normal(n_paths, n_steps, n), p.get("chol_t"))
        counts = _poisson_counts(ctx, p["jump_low"], p["jump_cdf"], n_paths, n_steps)
        w = ctx.normal(n_paths, n_steps, n)
        log_jumps = (
            counts * p["jump_mean"][:, None] + torch.sqrt(counts) * p["jump_std"][:, None] * w
        )
        log_increments = p["drift"][:, None] + p["vol"][:, None] * z + log_jumps
        jumps = torch.cat([torch.zeros_like(counts[:, :1]), counts], dim=1)
        return Simulation(prices=_torch_prices(p["s0"], log_increments), state={"jumps": jumps})

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

    def _torch_key(self) -> Hashable:
        return (type(self), tuple(model._chol is not None for model in self.regimes))

    def _torch_params(self) -> dict[str, Any]:
        dt = self._dt
        params = {
            "initial": np.cumsum(self.initial),
            "transition": np.cumsum(self.transition, axis=1),
            "drift": np.stack([(m.mu - 0.5 * m.sigma**2) * dt for m in self.regimes]),
            "vol": np.stack([m.sigma * np.sqrt(dt) for m in self.regimes]),
            "s0": self.s0,
        }
        if any(m._chol is not None for m in self.regimes):  # (identity for those without)
            eye = np.eye(self._n_assets)
            params["chol_t"] = np.stack(
                [eye if m._chol is None else m._chol.T for m in self.regimes]
            )
        return params

    def _torch_simulate(
        self, ctx: _TorchContext, params: dict[str, Any], n_paths: int, n_steps: int
    ) -> Simulation:
        import torch

        p, n = params, self._n_assets
        current = _torch_categorical(p["initial"], ctx.uniform(n_paths))
        # The regime each one would move to at each step of a chunk (steps, P, K), so that a
        # step of the chain is a lookup of the current one's.
        u = ctx.uniform(n_steps, n_paths)
        cumulative = p["transition"]  # (L, K, K), cumulative along the rows
        k = cumulative.shape[-1]
        chunk = max(1, _MOVES_ELEMENTS // (n_paths * k * k))
        chain = [current]
        for start in range(0, n_steps, chunk):
            moves = (u[start : start + chunk, :, None, None] >= cumulative).sum(dim=-1)
            for step in moves.clamp_(max=k - 1):
                current = torch.gather(step, 1, current[:, None])[:, 0]
                chain.append(current)
        regime = torch.stack(chain, dim=1)

        eps = ctx.normal(n_paths, n_steps, n)
        active = regime[:, :-1, None]
        log_increments = eps
        for index, model in enumerate(self.regimes):
            z = eps if model._chol is None else _correlated(eps, p["chol_t"][:, index])
            increment = p["drift"][:, index, None] + p["vol"][:, index, None] * z
            log_increments = (
                increment if index == 0 else torch.where(active == index, increment, log_increments)
            )
        return Simulation(
            prices=_torch_prices(p["s0"], log_increments),
            state={"regime": regime.to(ctx.dtype)},
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

    def _torch_key(self) -> Hashable:
        return (type(self), self._chol is not None)

    def _torch_params(self) -> dict[str, Any]:
        params = {
            "omega": self.omega,
            "alpha": self.alpha,
            "beta": self.beta,
            "h0": self.unconditional_variance,
            "mu_dt": self.mu * self._dt,
            "s0": self.s0,
        }
        if self._chol is not None:
            params["chol_t"] = self._chol.T
        return params

    def _torch_simulate(
        self, ctx: _TorchContext, params: dict[str, Any], n_paths: int, n_steps: int
    ) -> Simulation:
        import torch

        p, n = params, self._n_assets
        z = _correlated(ctx.normal(n_steps, n_paths, n), p.get("chol_t"), time_major=True)
        # h' = omega + alpha (sqrt(h) z)^2 + beta h = omega + h (alpha z^2 + beta)
        growth = torch.addcmul(p["beta"], p["alpha"], z * z)
        h, omega = p["h0"].expand(n_paths, n), p["omega"]
        steps = [h]
        for t in range(n_steps):
            h = torch.addcmul(omega, h, growth[t])
            steps.append(h)
        variance = torch.stack(steps, dim=1)
        log_increments = torch.sqrt(variance[:, :-1]) * z.transpose(0, 1) + p["mu_dt"][:, None]
        return Simulation(
            prices=_torch_prices(p["s0"], log_increments), state={"variance": variance}
        )

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

    def _torch_key(self) -> Hashable:
        return (type(self), self._chol is not None)

    def _torch_params(self) -> dict[str, Any]:
        decay, scale = _ou_coefficients(self.theta, self.sigma, self._dt)
        params = {
            "target": np.log(self.mean) - np.log(self.s0),  # log(mean) relative to log(s0)
            "decay": decay,
            "scale": scale,
            "s0": self.s0,
        }
        if self._chol is not None:
            params["chol_t"] = self._chol.T
        return params

    def _torch_simulate(
        self, ctx: _TorchContext, params: dict[str, Any], n_paths: int, n_steps: int
    ) -> Simulation:
        import torch

        p, n = params, self._n_assets
        z = _correlated(ctx.normal(n_steps, n_paths, n), p.get("chol_t"), time_major=True)
        shocks = p["scale"] * z
        target, decay = p["target"], p["decay"]
        # The deviation y = x - target follows y' = y decay + scale z, from x = 0.
        y = (-target).expand(n_paths, n)
        deviation = [y]
        for t in range(n_steps):
            y = torch.addcmul(shocks[t], y, decay)
            deviation.append(y)
        log_rel = target[:, None] + torch.stack(deviation, dim=1)  # exactly 0 at t = 0
        return Simulation(prices=p["s0"][:, None] * torch.exp(log_rel))

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

    def _torch_params(self) -> dict[str, Any]:
        dt = self._dt
        decay, scale = _ou_coefficients(self.theta, self.spread_sigma, dt)
        return {
            "trend_drift": (self.mu - 0.5 * self.sigma**2) * dt,
            "trend_vol": self.sigma * np.sqrt(dt),
            "decay": decay,
            "scale": scale,
            "s0": self.s0,
        }

    def _torch_simulate(
        self, ctx: _TorchContext, params: dict[str, Any], n_paths: int, n_steps: int
    ) -> Simulation:
        import torch

        p = params
        trend_shocks = ctx.normal(n_paths, n_steps)
        trend_increments = p["trend_drift"][:, None] + p["trend_vol"][:, None] * trend_shocks
        trend = torch.cat(
            [torch.zeros_like(trend_increments[:, :1]), torch.cumsum(trend_increments, dim=1)], 1
        )
        shocks = p["scale"] * ctx.normal(n_steps, n_paths)  # time-major, for the recursion
        s, decay = torch.zeros(n_paths, dtype=ctx.dtype, device=ctx.device), p["decay"]
        steps = [s]
        for t in range(n_steps):
            s = torch.addcmul(shocks[t], s, decay)
            steps.append(s)
        spread = torch.stack(steps, dim=1)
        log_rel = torch.stack([trend + 0.5 * spread, trend - 0.5 * spread], dim=-1)
        prices = p["s0"][:, None] * torch.exp(log_rel)
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

    def _torch_key(self) -> Hashable:
        return (type(self), self.returns.shape, self.block_size)

    def _torch_params(self) -> dict[str, Any]:
        return {"returns": self.returns, "s0": self.s0}  # the history is copied to a device once

    def _torch_simulate(
        self, ctx: _TorchContext, params: dict[str, Any], n_paths: int, n_steps: int
    ) -> Simulation:
        import torch

        n_rows, size = self.returns.shape[0], self.block_size
        n_blocks = -(-n_steps // size)
        starts = ctx.integers(n_rows, n_paths, n_blocks)
        rows = (starts[..., None] + torch.arange(size, device=ctx.device)) % n_rows
        rows = rows.reshape(n_paths, n_blocks * size)[:, :n_steps]
        returns = params["returns"]
        if returns.shape[0] == 1:  # one history for every path
            increments = returns[0][rows]
        else:
            increments = returns[torch.arange(n_paths, device=ctx.device)[:, None], rows]
        return Simulation(prices=_torch_prices(params["s0"], increments))

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

    `simulate_torch` builds the models on the host too, with a NumPy generator seeded from
    the torch generator's state (which it then moves on, so a given state builds the same
    models and the next call others), and nothing is read back from the device. The built-in
    models it builds then simulate together, those of a kind (the same class and structure,
    such as having a correlation) at once, from their parameters stacked and copied to the
    device in one transfer per kind and call. Other models simulate one path at a time with
    their own `simulate_torch`.

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

    def _torch_key(self) -> Hashable | None:
        return None  # its parameters are drawn on each call

    def _torch_simulate(
        self, ctx: _TorchContext, params: dict[str, Any], n_paths: int, n_steps: int
    ) -> Simulation:
        import torch

        rng = _host_rng(ctx.generator)
        models = [self._build(rng) for _ in range(n_paths)]
        sims: list[Simulation] = []
        order: list[int] = []  # the path of each simulated path, in the order of `sims`
        kinds: dict[Hashable, list[tuple[int, _Model]]] = {}  # built-in models, by kind
        for path, model in enumerate(models):
            if _simulates_on_torch(model) and (key := model._torch_key()) is not None:
                kinds.setdefault(key, []).append((path, model))
            else:  # simulated on its own
                sims.append(
                    model.simulate_torch(ctx.generator, n_steps, n_paths=1, dtype=ctx.dtype)
                )
                order.append(path)
        for members in kinds.values():
            stacked = _to_device([model._torch_params() for _, model in members], ctx)
            sims.append(members[0][1]._torch_simulate(ctx, stacked, len(members), n_steps))
            order.extend(path for path, _ in members)
        if len(sims) == 1:  # one kind of model: the paths are in order
            return sims[0]
        # Back to the order of the paths (the one copy to the device of models of many kinds).
        inverse = torch.as_tensor(np.argsort(order), device=ctx.device)

        def joined(tensors: list[Any]) -> torch.Tensor:
            return torch.cat(tensors)[inverse]

        prices: list[Any] = [sim.prices for sim in sims]
        masks: list[Any] = [sim.mask for sim in sims]
        states: list[Mapping[str, Any]] = [sim.state for sim in sims]
        mask = None
        if any(m is not None for m in masks):
            listed = [
                torch.ones_like(p, dtype=torch.bool) if m is None else m
                for p, m in zip(prices, masks)
            ]
            mask = joined(listed)
        return Simulation(
            prices=joined(prices),
            mask=mask,
            state={key: joined([state[key] for state in states]) for key in self._spec},
        )

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

    def _torch_key(self) -> Hashable | None:
        inner = self.model._torch_key() if _simulates_on_torch(self.model) else None
        return None if inner is None else (type(self), self.min_listed, inner)

    def _torch_params(self) -> dict[str, Any]:
        dt = self._dt
        return {
            "initial_listed": self.initial_listed,
            "p_list": -np.expm1(-self.listing_rate * dt),
            "p_delist": -np.expm1(-self.delisting_rate * dt),
            "model": self.model._torch_params() if _simulates_on_torch(self.model) else None,
        }

    @staticmethod
    def _torch_minimum(listed: torch.Tensor, floor: torch.Tensor | None) -> torch.Tensor:
        """`_enforce_minimum` on tensors. The k-th unlisted asset (by index) is listed when
        k <= min_listed - (listed assets), that is when at most `floor` = min_listed - 1 -
        (its index) assets after it are listed. (Counted in floats: MPS may lack an integer
        cumsum.)"""
        if floor is None:  # min_listed = 0
            return listed
        import torch

        count = torch.cumsum(listed, dim=-1, dtype=torch.float32)  # listed up to each asset
        return listed | (count[:, -1:] - count <= floor)

    def _torch_simulate(
        self, ctx: _TorchContext, params: dict[str, Any], n_paths: int, n_steps: int
    ) -> Simulation:
        import torch

        p, n = params, self._n_assets
        if _simulates_on_torch(self.model):
            inner = self.model._torch_simulate(ctx, p["model"], n_paths, n_steps)
        else:
            inner = self.model.simulate_torch(
                ctx.generator, n_steps, n_paths=n_paths, dtype=ctx.dtype
            )
        floor = None
        if self.min_listed > 0:
            index = torch.arange(n, dtype=torch.float32, device=ctx.device)
            floor = (self.min_listed - 1) - index
        current = self._torch_minimum(ctx.uniform(n_paths, n) < p["initial_listed"], floor)
        u = ctx.uniform(n_steps, n_paths, n)  # time-major, for the recursion
        stays = u >= p["p_delist"]  # a listed asset stays listed
        lists = u < p["p_list"]  # an unlisted one lists
        steps = [current]
        for t in range(n_steps):
            current = self._torch_minimum(torch.where(current, stays[t], lists[t]), floor)
            steps.append(current)
        listed = torch.stack(steps, dim=1)
        inner_mask: Any = inner.mask
        mask = listed if inner_mask is None else listed & inner_mask
        inner_prices: Any = inner.prices
        return Simulation(
            prices=torch.where(mask, inner_prices, torch.nan), mask=mask, state=inner.state
        )

    def _repr_fields(self) -> dict[str, str]:
        return {
            "model": repr(self.model),
            "initial_listed": _format(self.initial_listed),
            "listing_rate": _format(self.listing_rate),
            "delisting_rate": _format(self.delisting_rate),
            "min_listed": str(self.min_listed),
        }
