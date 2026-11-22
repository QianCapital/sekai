"""Batched market data on an array backend, and generators of market scenarios.

`MarketPanel` holds `P` market histories ("paths") of `T` times and `N` assets on one
`sekai.backend`: returns, a validity mask, per-asset fields and market-wide series,
as NumPy arrays or torch tensors on any device. Batched environments
(`sekai.envs.finance.BatchMarketEnv`) read their episodes from a panel: windows of
one historical path built from a `MarketData`, or a fresh path per episode drawn from
a `ScenarioGenerator`.

A `ScenarioGenerator` is anything with `sample(n_paths, *, seed) -> MarketPanel`:
a generative model of markets (a GAN, a diffusion model, a factor simulator), a
bootstrap of history, or `ModelScenarios`, which wraps a `sekai.finance.MarketModel`.
sekai never provides market data itself; generators and `MarketData` are where
yours comes in.

Example:
    ```python
    class MyGAN:
        def sample(self, n_paths, *, seed):
            g = torch.Generator().manual_seed(seed)
            returns, volume = self.model.generate(n_paths, generator=g)  # (P, T, N) each
            return MarketPanel.from_returns(returns, fields={"volume": volume})

    env = BatchMarketEnv(64, scenarios=MyGAN(), backend="torch", device="cuda")
    ```
"""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from sekai.backend import Array, Backend, backend_of, get_backend, is_tensor, to_numpy
from sekai.finance.data import MarketData
from sekai.finance.simulators import MarketModel, Simulation


def _convert(B: Backend, x: Any, *, device: Any, dtype: Any) -> Array:
    """`x` on backend B: floats in `dtype` (torch) or float64 (NumPy); bools stay bools."""
    if B.name == "numpy":
        arr = B.to_numpy(x) if is_tensor(x) else np.asarray(x)
        return arr if arr.dtype == np.bool_ else np.asarray(arr, dtype=np.float64)
    import torch

    if isinstance(x, torch.Tensor):
        if x.dtype == torch.bool:
            return x if device is None else x.to(device=device)
        target = dtype if dtype is not None else (x.dtype if x.is_floating_point() else None)
        return x.to(device=device if device is not None else x.device, dtype=target)
    arr = np.asarray(x)
    if not arr.flags.writeable:  # e.g. MarketData's read-only arrays (torch would warn)
        arr = arr.copy()
    if arr.dtype == np.bool_:
        return torch.as_tensor(arr, device=device)
    return torch.as_tensor(arr, dtype=dtype or torch.get_default_dtype(), device=device)


@dataclass(frozen=True, eq=False)
class MarketPanel:
    """`P` paths of `T` times and `N` assets of market data, on one array backend.

    Attributes:
        returns: `(P, T, N)` simple return of each asset over the period ending at each
            time, `(t - 1, t]`. Row 0 is never used as a return (it may be NaN), and
            neither is a NaN.
        valid: `(P, T, N)` bool, True where the asset is listed and tradable.
        fields: Per-asset data, `(P, T, N, ...)` each (volume, spread, alphas, factor
            loadings, borrow rates, open prices...).
        exogenous: Market-wide data, `(P, T, ...)` each (rates, macro series, factor
            covariances...).
        timestamps: NumPy timestamps `(T,)` (shared by every path) or `(P, T)`, or None.
        assets: The `N` asset identifiers, or None.
        state: Latent series of a simulation, `(P, T, ...)` each (e.g. the regime), for
            privileged information; empty for historical data.
        corporate_actions: Whether the fields `split` and `dividend` are corporate
            actions (see `MarketData.simple_returns`), which `returns` already include
            and environments apply when they fill at a price other than the close.
            Otherwise fields with those names are ordinary fields.

    Raises:
        ValueError: If the shapes disagree or the arrays mix backends.
    """

    returns: Array
    valid: Array
    fields: Mapping[str, Array] = field(default_factory=dict)
    exogenous: Mapping[str, Array] = field(default_factory=dict)
    timestamps: NDArray[Any] | None = None
    assets: tuple[Any, ...] | None = None
    state: Mapping[str, Array] = field(default_factory=dict)
    corporate_actions: bool = False

    def __post_init__(self) -> None:
        if self.returns.ndim != 3:
            raise ValueError(f"returns must be (P, T, N), got shape {tuple(self.returns.shape)}.")
        shape = tuple(self.returns.shape)
        if tuple(self.valid.shape) != shape:
            raise ValueError(f"valid has shape {tuple(self.valid.shape)}; expected {shape}.")
        B = backend_of(self.returns)
        arrays = {"valid": self.valid}
        arrays.update({f"fields[{k!r}]": v for k, v in self.fields.items()})
        arrays.update({f"exogenous[{k!r}]": v for k, v in self.exogenous.items()})
        arrays.update({f"state[{k!r}]": v for k, v in self.state.items()})
        for name, arr in arrays.items():
            if backend_of(arr).name != B.name:
                raise ValueError(f"{name} is on another backend than returns ({B.name}).")
        for name, arr in self.fields.items():
            if tuple(arr.shape[:3]) != shape:
                raise ValueError(
                    f"field {name!r} has shape {tuple(arr.shape)}; its first three "
                    f"dimensions must be (P, T, N) = {shape}."
                )
        for kind, series in (("exogenous", self.exogenous), ("state", self.state)):
            for name, arr in series.items():
                if tuple(arr.shape[:2]) != shape[:2]:
                    raise ValueError(
                        f"{kind} {name!r} has shape {tuple(arr.shape)}; its first two "
                        f"dimensions must be (P, T) = {shape[:2]}."
                    )
        if self.timestamps is not None:
            ts = np.asarray(self.timestamps)
            if ts.shape not in ((shape[1],), shape[:2]):
                raise ValueError(
                    f"timestamps has shape {ts.shape}; expected ({shape[1]},) or {shape[:2]}."
                )
        if self.assets is not None and len(self.assets) != shape[2]:
            raise ValueError(f"Got {len(self.assets)} asset ids for {shape[2]} assets.")

    # ------------------------------------------------------------------
    # Shape and backend
    # ------------------------------------------------------------------

    @property
    def backend(self) -> Backend:
        return backend_of(self.returns)

    @property
    def device(self) -> Any:
        """The torch device of the arrays (None for NumPy)."""
        return getattr(self.returns, "device", None)

    @property
    def n_paths(self) -> int:
        return int(self.returns.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.returns.shape[1])

    @property
    def n_assets(self) -> int:
        return int(self.returns.shape[2])

    def to(self, backend: str | Backend, *, device: Any = None, dtype: Any = None) -> MarketPanel:
        """The panel on another backend (or device, or float dtype for torch)."""
        B = get_backend(backend)

        def conv(x: Any) -> Array:
            return _convert(B, x, device=device, dtype=dtype)

        return MarketPanel(
            returns=conv(self.returns),
            valid=conv(self.valid),
            fields={k: conv(v) for k, v in self.fields.items()},
            exogenous={k: conv(v) for k, v in self.exogenous.items()},
            timestamps=self.timestamps,
            assets=self.assets,
            state={k: conv(v) for k, v in self.state.items()},
            corporate_actions=self.corporate_actions,
        )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_returns(
        cls,
        returns: Any,
        *,
        valid: Any = None,
        fields: Mapping[str, Any] | None = None,
        exogenous: Mapping[str, Any] | None = None,
        timestamps: Any = None,
        assets: Sequence[Any] | None = None,
        state: Mapping[str, Any] | None = None,
        corporate_actions: bool = False,
    ) -> MarketPanel:
        """A panel from arrays already on their backend (the usual way generators build one).

        Args:
            returns: `(P, T, N)` simple returns over `(t - 1, t]`; row 0 is unused.
            valid: `(P, T, N)` bool tradability mask (default: every asset is always
                tradable).
            fields: Per-asset data `(P, T, N, ...)`.
            exogenous: Market-wide data `(P, T, ...)`.
            timestamps: NumPy timestamps `(T,)` or `(P, T)`.
            assets: `N` asset identifiers.
            state: Latent series `(P, T, ...)`.
            corporate_actions: The fields `split` and `dividend` are corporate actions
                (which `returns` must already include).
        """
        B = backend_of(returns)
        r = B.asfloat(returns)
        ok = B.bools(tuple(r.shape), True, like=r) if valid is None else B.asbool(valid, r)
        return cls(
            returns=r,
            valid=ok,
            fields={k: B.asfloat(v, r) for k, v in (fields or {}).items()},
            exogenous={k: B.asfloat(v, r) for k, v in (exogenous or {}).items()},
            timestamps=None if timestamps is None else np.asarray(timestamps),
            assets=None if assets is None else tuple(assets),
            state={k: B.asfloat(v, r) for k, v in (state or {}).items()},
            corporate_actions=corporate_actions,
        )

    @classmethod
    def from_market_data(
        cls,
        data: MarketData,
        *,
        fields: Sequence[str] | None = None,
        exogenous: Sequence[str] | None = None,
        backend: str | Backend = "numpy",
        device: Any = None,
        dtype: Any = None,
    ) -> MarketPanel:
        """A one-path panel of a `MarketData` history.

        Returns are `data.simple_returns()`: close-to-close (`prices[t] / prices[t - 1] -
        1`, NaN where either price is missing), with the data's corporate actions if it
        has them. `valid` is where a price exists. On the NumPy backend the
        panel shares the data's arrays (memory maps included) instead of copying them.

        Args:
            data: The history.
            fields: Data fields to include (default: all, prices included).
            exogenous: Exogenous series to include (default: all).
            backend: "numpy" or "torch".
            device: torch device.
            dtype: torch float dtype (default: torch's default dtype, usually float32).
        """
        B = get_backend(backend)
        returns = data.simple_returns()  # shared with the data (memory-mapped if it is)
        names = list(data.fields) if fields is None else list(fields)
        series = list(data.exogenous) if exogenous is None else list(exogenous)

        def conv(x: Any) -> Array:
            return _convert(B, np.asarray(x)[None], device=device, dtype=dtype)

        return cls(
            returns=conv(returns),
            valid=conv(data.mask),
            fields={name: conv(data.field(name)) for name in names},
            exogenous={name: conv(data.exogenous_series(name)) for name in series},
            timestamps=data.timestamps,
            assets=data.assets,
            corporate_actions=data.has_corporate_actions,
        )

    @classmethod
    def from_simulation(
        cls,
        sim: Simulation,
        *,
        backend: str | Backend = "numpy",
        device: Any = None,
        dtype: Any = None,
    ) -> MarketPanel:
        """A panel of simulated paths (`MarketModel.simulate(..., n_paths=P)`).

        The simulation's prices become returns and its mask (or the finite prices) the
        validity mask; its state series are both `exogenous` (so environments can
        observe them) and `state` (privileged information). The prices are kept as the
        `"close"` field.

        With `backend="torch"`, a simulation of tensors (`MarketModel.simulate_torch`)
        stays where it is (or moves to `device`): the panel is computed there, in `dtype`
        (default: the simulation's), without a copy to or from the host. With the NumPy
        backend, its arrays are copied to the host first.
        """
        B = get_backend(backend)
        if is_tensor(sim.prices):
            if B.name == "torch":
                return cls._from_tensor_simulation(sim, device=device, dtype=dtype)
            sim = Simulation(
                prices=to_numpy(sim.prices),
                mask=None if sim.mask is None else to_numpy(sim.mask),
                state={k: to_numpy(v) for k, v in sim.state.items()},
            )
        single = np.ndim(sim.prices) == 2  # one path without a paths axis: add it
        prices = np.asarray(sim.prices, dtype=np.float64)
        prices = prices[None] if single else prices
        valid = np.isfinite(prices)
        if sim.mask is not None:
            valid &= sim.mask[None] if single else sim.mask
        returns = np.full(prices.shape, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            returns[:, 1:] = prices[:, 1:] / prices[:, :-1] - 1.0
        state = {
            k: np.asarray(v, dtype=np.float64)[None] if single else np.asarray(v, dtype=np.float64)
            for k, v in sim.state.items()
        }

        def conv(x: Any) -> Array:
            return _convert(B, x, device=device, dtype=dtype)

        return cls(
            returns=conv(returns),
            valid=conv(valid),
            fields={"close": conv(prices)},
            exogenous={k: conv(v) for k, v in state.items()},
            state={k: conv(v) for k, v in state.items()},
        )

    @classmethod
    def _from_tensor_simulation(cls, sim: Simulation, *, device: Any, dtype: Any) -> MarketPanel:
        """`from_simulation` of tensors, on the torch backend: computed on the device, in the
        wider of the simulation's dtype and `dtype`, then kept in `dtype`."""
        import torch

        prices: Any = sim.prices  # a tensor
        single = prices.ndim == 2  # one path without a paths axis: add it

        def paths(x: Any) -> Any:
            return x[None] if single else x

        floating = prices.is_floating_point()
        if dtype is not None:
            target = dtype
        else:
            target = prices.dtype if floating else torch.get_default_dtype()
        work = torch.promote_types(prices.dtype, target) if floating else target
        where = prices.device if device is None else torch.device(device)
        p = paths(prices).to(device=where, dtype=work)
        valid = torch.isfinite(p)
        if sim.mask is not None:
            valid &= paths(sim.mask).to(device=where)
        returns = torch.cat([torch.full_like(p[:, :1], torch.nan), p[:, 1:] / p[:, :-1] - 1.0], 1)
        state = {k: paths(v).to(device=where, dtype=target) for k, v in sim.state.items()}
        return cls(
            returns=returns.to(target),
            valid=valid,
            fields={"close": p.to(target)},
            exogenous=dict(state),
            state=state,
        )


@runtime_checkable
class ScenarioGenerator(Protocol):
    """A source of market scenarios: `sample(n_paths, *, seed)` returns a `MarketPanel`.

    The panel must have `n_paths` paths, enough times for an episode (environments
    say how many they need), and the fields the environment reads, on the backend
    the environment runs on (or NumPy, which environments convert). Sampling must be
    deterministic given the seed, so seeded environments are reproducible.
    """

    def sample(self, n_paths: int, *, seed: int) -> MarketPanel: ...


class ModelScenarios:
    """A `ScenarioGenerator` of a `sekai.finance.MarketModel`: fresh simulated paths.

    By default the paths are simulated on the host, with NumPy (`MarketModel.simulate`),
    and the panels are NumPy arrays. With a torch `device`, they are simulated there
    (`MarketModel.simulate_torch`, drawing from a `torch.Generator` on the device seeded
    with the seed) and the panels are tensors made on it, not copied from the host.

    Args:
        model: The market model.
        n_steps: Steps per path (the panel has `n_steps + 1` times).
        device: A torch device to simulate on, or None to simulate on the host.
        dtype: The torch float dtype of panels simulated on a device (default: torch's
            default dtype).

    Raises:
        TypeError: If model is not a `MarketModel`, or dtype not a floating torch dtype.
        ValueError: If n_steps is smaller than 1, or dtype is given without a device.
    """

    # The defaults (simulate on the host). Instances set them only when given, so one that
    # simulates on the host has the attributes, and pickles, it had before.
    device: Any = None
    dtype: Any = None

    def __init__(
        self, model: MarketModel, n_steps: int, *, device: Any = None, dtype: Any = None
    ) -> None:
        if not isinstance(model, MarketModel):
            raise TypeError(f"Expected a MarketModel, got {model!r}.")
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}.")
        self.model = model
        self.n_steps = int(n_steps)
        if device is None:
            if dtype is not None:
                raise ValueError("dtype= is for panels simulated on a device: give device= too.")
            return
        import torch

        self.device = torch.device(device)
        if dtype is not None:
            if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
                raise TypeError(f"dtype must be a floating-point torch dtype, got {dtype!r}.")
            self.dtype = dtype

    def sample(self, n_paths: int, *, seed: int) -> MarketPanel:
        if self.device is None:
            sim = self.model.simulate(np.random.default_rng(seed), self.n_steps, n_paths=n_paths)
            return MarketPanel.from_simulation(sim)
        import torch

        generator = torch.Generator(device=self.device).manual_seed(operator.index(seed))
        sim = self.model.simulate_torch(generator, self.n_steps, n_paths=n_paths, dtype=self.dtype)
        return MarketPanel.from_simulation(
            sim, backend="torch", device=self.device, dtype=self.dtype
        )

    def __repr__(self) -> str:
        on = "" if self.device is None else f", device={str(self.device)!r}"
        if self.dtype is not None:
            on += f", dtype={self.dtype}"
        return f"ModelScenarios({self.model!r}, n_steps={self.n_steps}{on})"


__all__ = ["MarketPanel", "ScenarioGenerator", "ModelScenarios"]
