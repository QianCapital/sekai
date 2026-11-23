"""Observation builders for batched market environments.

A batched market environment (`sekai.envs.finance.BatchMarketEnv`) delegates what an
agent observes to an `ObservationBuilder`. At every decision the environment hands
the builder a `MarketView` of all its lanes: windows of returns and data fields per
asset slot, market-wide series, and the state of each account. The builder returns
the observation (a dict of `(num_envs, ...)` arrays on the env's backend) and
declares its space.

`MarketObservation` is the default, the observation of `Market-v0` (a padded set of
assets with a window of returns and features each, the portfolio, the account and
market series), plus the pending orders when execution is delayed. Write your own
builder to observe anything else (cross-sectionally normalised features, alpha
signals, risk-model exposures...), as differentiable code with the torch backend.

Example:
    ```python
    class ZScored(ObservationBuilder):
        fields = ("alpha",)

        def space(self, env):
            return sekai.Box(-np.inf, np.inf, shape=(env.n_slots, 2), dtype=np.float32)

        def __call__(self, view):
            B = view.backend
            alpha = view.field("alpha")                                 # (B, S)
            mu = B.sum(alpha, axis=-1, keepdims=True) / view.n_tradable[:, None]
            return B.stack([alpha - mu, view.weights], axis=-1)       # (B, S, 2)
    ```
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

import numpy as np

from sekai.backend import Array, Backend
from sekai.spaces.box import Box
from sekai.spaces.dict import Dict
from sekai.spaces.masked_box import MaskedBox
from sekai.spaces.space import Space

if TYPE_CHECKING:
    from sekai.finance.panel import MarketPanel


class MarketView:
    """What an observation builder sees: every lane's data and account at its decision.

    Lane `b` decides at time `rows[b]` of path `paths[b]` of the panel; slot `s` of
    lane `b` holds asset `slots[b, s]` (-1 for an empty slot). Windows end at the
    decision time and run back `length` rows (at most `lookback + 1`).

    Attributes:
        backend: The arrays' backend.
        lookback: The environment's lookback.
        episode_length: Decisions per episode.
        elapsed: NumPy `(B,)` decisions taken so far in each lane's episode.
        slots: `(B, S)` int, the asset (panel column) in each slot, -1 if empty.
        occupied: `(B, S)` bool, slots that hold an asset.
        tradable: `(B, S)` bool, slots whose asset can be traded at the decision.
        weights: `(B, S)` current weights.
        equity: `(B,)` equity relative to the start of the episode.
        peak: `(B,)` highest equity of the episode so far.
        pending: `(B, S)` target weights of orders not yet executed (delayed
            execution), or None.
    """

    def __init__(
        self,
        *,
        backend: Backend,
        panel: MarketPanel,
        paths: Array,
        rows: Array,
        slots: Array,
        lookback: int,
        episode_length: int,
        elapsed: Any,
        weights: Array,
        equity: Array,
        peak: Array,
        pending: Array | None,
    ) -> None:
        B = backend
        self.backend = backend
        self.panel = panel
        self.paths = paths
        self.rows = rows
        self.slots = slots
        self.lookback = lookback
        self.episode_length = episode_length
        self.elapsed = elapsed
        self.weights = weights
        self.equity = equity
        self.peak = peak
        self.pending = pending
        self.occupied = slots >= 0
        self._cols = B.where(self.occupied, slots, 0)
        valid = panel.valid[paths[:, None], rows[:, None], self._cols]
        self.tradable = self.occupied & valid
        # Window rows by length, and panel arrays gathered per slot by (array, length):
        # builders and cost models ask for the same ones several times (a panel never
        # changes, and these never leave the view).
        self._rows_of: dict[int, Array] = {}
        self._gathered: dict[tuple[int, int | None], tuple[Array, Array]] = {}

    @property
    def num_envs(self) -> int:
        return int(self.slots.shape[0])

    @property
    def n_slots(self) -> int:
        return int(self.slots.shape[1])

    @property
    def n_tradable(self) -> Array:
        """`(B,)` number of tradable slots, as floats (at least 1, for averages)."""
        B = self.backend
        return B.maximum(B.sum(B.asfloat(self.tradable, self.weights), axis=-1), 1.0)

    def _window_rows(self, length: int) -> Array:
        if not 1 <= length <= self.lookback + 1:
            raise ValueError(f"window length must be in [1, lookback + 1 = {self.lookback + 1}].")
        rows = self._rows_of.get(length)
        if rows is None:
            offsets = self.backend.arange(length, like=self.rows) - (length - 1)
            rows = self._rows_of[length] = self.rows[:, None] + offsets[None, :]
        return rows

    def _assets(self, arr: Array, length: int | None) -> Array:
        cached = self._gathered.get((id(arr), length))
        if cached is not None and cached[0] is arr:
            return cached[1]
        if length is None:
            out = arr[self.paths[:, None], self.rows[:, None], self._cols]
        else:
            r = self._window_rows(length)
            out = arr[self.paths[:, None, None], r[:, None, :], self._cols[:, :, None]]
        self._gathered[(id(arr), length)] = (arr, out)
        return out

    def _mask_like(self, mask: Array, values: Array) -> Array:
        """`mask` (`(B, S)` or `(B, S, L)`) broadcast over the trailing dims of `values`."""
        return mask.reshape(tuple(mask.shape) + (1,) * (values.ndim - mask.ndim))

    def returns(self, length: int | None = None, *, fill: float | None = 0.0) -> Array:
        """Simple returns per slot: `(B, S)` at the decision, or `(B, S, L)` windows.

        Unavailable returns (empty slot, asset not listed at either end of the
        period) are `fill` (NaN with `fill=None`).
        """
        B = self.backend
        r = self._assets(self.panel.returns, length)
        if fill is None:
            return B.where(self._available(length), r, float("nan"))
        return B.where(self._available(length), r, fill)

    def available(self, length: int | None = None) -> Array:
        """Where `returns(length)` is defined: `(B, S)` or `(B, S, L)` bool."""
        B = self.backend
        r = self._assets(self.panel.returns, length)
        occ = self.occupied if length is None else self.occupied[:, :, None]
        return occ & B.isfinite(r)

    def _available(self, length: int | None) -> Array:
        """`available(length)`, made once per view for sekai's own use (never handed out:
        a caller could change it in place)."""
        key = (-1, length)  # (no panel array has the id -1)
        cached = self._gathered.get(key)
        if cached is None:
            cached = self._gathered[key] = (None, self.available(length))
        return cached[1]

    def field(self, name: str, length: int | None = None, *, fill: float | None = 0.0) -> Array:
        """A per-asset field per slot: `(B, S, ...)` at the decision, `(B, S, L, ...)`
        windows. Missing values (NaN, empty slots) are `fill` (kept NaN with None)."""
        B = self.backend
        values = self._assets(self._field(name), length)
        occ = self.occupied if length is None else self.occupied[:, :, None]
        ok = self._mask_like(occ, values) & B.isfinite(values)
        return B.where(ok, values, float("nan") if fill is None else fill)

    def market(self, name: str, length: int | None = None, *, fill: float | None = 0.0) -> Array:
        """A market-wide series: `(B, ...)` at the decision or `(B, L, ...)` windows."""
        B = self.backend
        arr = self._series(name)
        if length is None:
            values = arr[self.paths, self.rows]
        else:
            values = arr[self.paths[:, None], self._window_rows(length)]
        if fill is None:
            return values
        return B.where(B.isfinite(values), values, fill)

    def _field(self, name: str) -> Array:
        try:
            return self.panel.fields[name]
        except KeyError:
            raise KeyError(
                f"The panel has no field {name!r} (it has {sorted(self.panel.fields)}); list "
                "it in the builder's `fields`."
            ) from None

    def _series(self, name: str) -> Array:
        try:
            return self.panel.exogenous[name]
        except KeyError:
            raise KeyError(
                f"The panel has no market series {name!r} (it has {sorted(self.panel.exogenous)}); "
                "list it in the builder's `market_fields`."
            ) from None


class ObservationBuilder(abc.ABC):
    """Builds the observation of every lane of a batched market environment.

    Declare the per-asset data fields (`fields`) and market-wide series
    (`market_fields`) the builder reads, so the environment loads them; implement
    `space(env)` (the observation space of one lane) and `__call__(view)` (the
    batched observation, arrays on the view's backend; NumPy observations should
    match the space's dtypes).
    """

    fields: tuple[str, ...] = ()
    market_fields: tuple[str, ...] = ()

    @abc.abstractmethod
    def space(self, env: Any) -> Space[Any]:
        """The observation space of one lane of `env` (a `BatchMarketEnv`)."""

    @abc.abstractmethod
    def __call__(self, view: MarketView) -> Any:
        """The observation of every lane."""


def _float32(B: Backend, x: Array) -> Array:
    """NumPy arrays as float32 (the spaces' dtype); torch tensors keep the env's dtype."""
    return x.astype(np.float32) if B.name == "numpy" else x


class MarketObservation(ObservationBuilder):
    """The observation of `Market-v0`, plus pending orders when execution is delayed.

    A `Dict` with:

    - `"assets"`: `MaskedBox` of `S` slots, each a `(lookback, F)` window of the log
      return, an availability flag (1 where the return is defined), then each data
      field in `features` (a field with several values per asset adds that many
      channels). Missing values are 0, and slots that cannot be traded are all 0
      with mask 0.
    - `"portfolio"`: `Box (S + 1,)`, current weights per slot, cash last.
    - `"account"`: `Box (2,)`, log equity and gross exposure.
    - `"pending"` (only with `execution="next_close"`): `Box (S + 1,)`, the target
      weights of the order that executes at the next close, cash last (so the
      observation is Markov).
    - `"market"` (only with `exogenous`): `Box (lookback, E)`, a window of the
      market-wide series named in `exogenous`, flattened per row.

    Args:
        features: Per-asset data fields to observe.
        exogenous: Market-wide series to observe.
    """

    def __init__(self, features: tuple[str, ...] = (), exogenous: tuple[str, ...] = ()) -> None:
        self.fields = tuple(features)
        self.market_fields = tuple(exogenous)

    def space(self, env: Any) -> Space[Any]:
        S, L = env.n_slots, env.lookback
        width = 2 + sum(int(np.prod(env.field_shape(f), dtype=np.int64)) for f in self.fields)
        spaces: dict[str, Space[Any]] = {
            "assets": MaskedBox(-np.inf, np.inf, shape=(L, width), max_size=S, dtype=np.float32),
            "portfolio": Box(-np.inf, np.inf, shape=(S + 1,), dtype=np.float32),
            "account": Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
        }
        if env.execution == "next_close":
            spaces["pending"] = Box(-np.inf, np.inf, shape=(S + 1,), dtype=np.float32)
        if self.market_fields:
            e = sum(
                int(np.prod(env.market_shape(name), dtype=np.int64)) for name in self.market_fields
            )
            spaces["market"] = Box(-np.inf, np.inf, shape=(L, e), dtype=np.float32)
        self._lookback = L
        return Dict(spaces)

    def __call__(self, view: MarketView) -> dict[str, Any]:
        B = view.backend
        L = view.lookback
        like = view.weights
        avail = view._available(L)
        log_ret = B.where(avail, B.log1p(view.returns(L, fill=0.0)), 0.0)
        channels = [log_ret[..., None], B.asfloat(avail, like)[..., None]]
        for name in self.fields:
            values = view.field(name, L)
            channels.append(values.reshape((*tuple(values.shape[:3]), -1)))
        values = B.concat(channels, axis=-1)
        values = B.where(view.tradable[:, :, None, None], values, 0.0)
        weights = view.weights
        cash = 1.0 - B.sum(weights, axis=-1, keepdims=True)
        mask = view.tradable
        obs: dict[str, Any] = {
            "assets": {
                "values": _float32(B, values),
                "mask": B.asint8(mask),
            },
            "portfolio": _float32(B, B.concat([weights, cash], axis=-1)),
            "account": _float32(
                B,
                B.stack(
                    [B.log(B.maximum(view.equity, 1e-12)), B.sum(B.abs(weights), axis=-1)], axis=-1
                ),
            ),
        }
        if view.pending is not None:
            pending_cash = 1.0 - B.sum(view.pending, axis=-1, keepdims=True)
            obs["pending"] = _float32(B, B.concat([view.pending, pending_cash], axis=-1))
        if self.market_fields:
            series = [view.market(name, L) for name in self.market_fields]
            flat = [s.reshape((*tuple(s.shape[:2]), -1)) for s in series]
            obs["market"] = _float32(B, B.concat(flat, axis=-1))
        return obs

    def __repr__(self) -> str:
        return f"MarketObservation(features={self.fields}, exogenous={self.market_fields})"


__all__ = ["MarketView", "ObservationBuilder", "MarketObservation"]
