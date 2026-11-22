"""Point-in-time market data for historical environments.

`MarketData` is an immutable, time-aligned panel: `T` timestamps x `N` assets x
named fields (close prices, volumes, features...), plus market-wide
"exogenous" series (rates, an index level, macro releases joined as-of). A NaN
price means the asset is not listed (or not tradable) at that time, so a
universe whose members come and go is represented directly, without
forward-filling or dropping assets.

Splits are chronological and support an embargo (a gap of unused rows between
a training and a test period) and history (rows before a test period that an
environment may observe as lookback, but never trades on), which together
prevent the usual leaks between training and evaluation. A split records its
history rows as `MarketData.warmup`, and the finance environments never make a
decision on a warm-up row, whatever their lookback.

A point in time (`index_of`, `between`, `split`, and an environment's
`reset(options={"start": ...})`) is either a row position or a timestamp. An
`int` is always a row position, counted within the object at hand, whatever its
time axis. Timestamps are looked up by value: dates and times on datetime axes,
floats on numeric axes, and `Time(t)` for any timestamp `t`, integers included
(epoch seconds, or the numeric timestamps of a split, which no longer start at 0).

Timestamps are timezone-naive. Timezone-aware input is refused rather than converted
silently; convert it first (`index.tz_convert("UTC").tz_localize(None)` for UTC, or
`index.tz_localize(None)` to keep local wall-clock times).
"""

from __future__ import annotations

import datetime as _dt
import json
import mmap
import operator
import os
import shutil
import tempfile
import uuid
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Timestamp values: numbers on numeric time axes, these on datetime ones (`datetime`
# and pandas `Timestamp` are `date` subclasses).
_NUMBERS = (int, float, np.integer, np.floating)
_DATETIMES = (str, _dt.date, np.datetime64)


@dataclass(frozen=True, repr=False)
class Time:
    """A timestamp to look up by value, where a plain `int` would be a row position.

    `MarketData.index_of`, `between` and `split`, and `reset(options={"start": ...})`
    in the finance environments, read an `int` as a row position on every time axis.
    `Time(t)` is always a timestamp: the first row at or after `t` is used. It is how
    to address numeric timestamps that are integers, such as epoch seconds or the
    timestamps of a split (which no longer start at 0). Since `info["timestamp"]`
    reports an episode's decision times, `options={"start": Time(info["timestamp"])}`
    replays an episode on any data.

    Args:
        value: A number on numeric timestamps; on datetime timestamps anything
            `numpy.datetime64` accepts (a `datetime64`, `datetime`, `date`, pandas
            `Timestamp` or date string).

    Raises:
        TypeError: If `value` is neither a number nor a date or time.

    Example:
        ```python
        data = MarketData(prices, timestamps=epoch_seconds)
        before, after = data.split(Time(1_700_000_000))
        ```
    """

    value: Any

    def __post_init__(self) -> None:
        if isinstance(self.value, bool | np.bool_) or not isinstance(
            self.value, _NUMBERS + _DATETIMES
        ):
            raise TypeError(
                "Time takes a number or a date or time (datetime64, datetime, date, pandas "
                f"Timestamp, date string), got {self.value!r}."
            )

    def __repr__(self) -> str:
        return f"Time({self.value!r})"


TimeLike = int | np.integer[Any] | float | np.floating[Any] | np.datetime64 | str | _dt.date | Time


def _readonly(arr: NDArray[Any]) -> NDArray[Any]:
    arr.setflags(write=False)
    return arr


def _float_array(value: ArrayLike, *, copy: bool) -> NDArray[np.float64]:
    """`value` as a float64 array the MarketData can keep: a copy of it, or with
    `copy=False` a float64 array itself (through a view, so making it read-only leaves the
    caller's array as it was)."""
    if not copy and isinstance(value, np.ndarray) and value.dtype == np.float64:
        return value.view() if value.flags.writeable else value
    return np.array(value, dtype=np.float64)


def _is_mapped(arr: Any) -> bool:
    """Whether `arr` is (a view of) a memory map."""
    while arr is not None:
        if isinstance(arr, np.memmap | mmap.mmap):
            return True
        arr = getattr(arr, "base", None)
    return False


def _as_field(name: str, value: ArrayLike, *, copy: bool, validate: bool) -> NDArray[np.float64]:
    """A per-asset field as a `(T, N, ...)` float64 array, checked when `validate`."""
    arr = _float_array(value, copy=copy)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim < 2:
        raise ValueError(f"field {name!r} must be (T, N) or (T, N, ...), got {arr.shape}.")
    if validate and np.isinf(arr).any():
        raise ValueError(f"field {name!r} contains infinite values; use NaN for missing.")
    return arr


def _as_exogenous(
    name: str, value: ArrayLike, n_steps: int, *, copy: bool, validate: bool
) -> NDArray[np.float64]:
    """A market-wide series as a `(T, ...)` float64 array, checked when `validate`."""
    arr = _float_array(value, copy=copy)
    if arr.ndim < 1 or arr.shape[0] != n_steps:
        raise ValueError(
            f"exogenous {name!r} must have shape ({n_steps},) or ({n_steps}, ...), got {arr.shape}."
        )
    if validate and np.isinf(arr).any():
        raise ValueError(f"exogenous {name!r} contains infinite values.")
    return arr


def _check_corporate_actions(arrays: Mapping[str, NDArray[np.float64]], validate: bool) -> None:
    if validate and "split" in arrays:
        ratios = arrays["split"]
        if ratios.ndim != 2 or np.any(ratios[np.isfinite(ratios)] <= 0):
            raise ValueError("the split field must be (T, N) and positive (NaN = no split).")
    if validate and "dividend" in arrays:
        cash = arrays["dividend"]
        if cash.ndim != 2 or np.any(cash[np.isfinite(cash)] < 0):
            raise ValueError("the dividend field must be (T, N) and >= 0 (NaN = none).")


_FORMAT = "sekai.MarketData"
_FORMAT_VERSION = 1


_TZ_HINT = (
    "sekai uses timezone-naive timestamps; convert explicitly first, e.g. "
    "`index.tz_convert('UTC').tz_localize(None)` for UTC times or "
    "`index.tz_localize(None)` to keep the local wall-clock times."
)


def naive_datetime64(value: Any, what: str, *, unit: str | None = None) -> Any:
    """`value` as numpy datetime64 (an array of `unit` if given), refusing timezone-aware input.

    numpy converts timezone-aware values (pandas Timestamps, datetimes with a tzinfo,
    strings with a UTC offset) to UTC with only a warning, which would silently shift
    every timestamp by the offset; this raises instead.

    Raises:
        ValueError: If `value` is timezone-aware.
    """
    if getattr(getattr(value, "dtype", None), "tz", None) is not None:
        raise ValueError(f"{what} is timezone-aware. {_TZ_HINT}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = np.asarray(value).astype(f"datetime64[{unit}]") if unit else np.datetime64(value)
    for warning in caught:
        if "timezone" in str(warning.message):
            raise ValueError(f"{what} is timezone-aware. {_TZ_HINT}")
    for warning in caught:  # pass on anything unrelated
        warnings.warn_explicit(warning.message, warning.category, warning.filename, warning.lineno)
    return out


def row_of(
    timestamps: NDArray[Any], when: Any, *, side: str = "left", what: str = "the time"
) -> int:
    """The row that `when` names on the time axis `timestamps` (see `MarketData.index_of`).

    An `int` is a row position (negative counts from the end), returned without a
    range check. Anything else is a timestamp: the first row at or after it (after
    it with `side="right"`), or `len(timestamps)` if there is none.

    Args:
        timestamps: The `(T,)` strictly increasing time axis, datetime64 or numeric.
        when: A row position, a timestamp of the axis' kind, or a `Time`.
        side: "left" or "right", as in `numpy.searchsorted`.
        what: Name of `when` in error messages.

    Raises:
        TypeError: If `when` is a bool.
        ValueError: If a timestamp does not suit the axis (a date on numeric timestamps,
            a number on datetime ones), is NaN or NaT, or is timezone-aware.
    """
    if isinstance(when, bool | np.bool_):
        raise TypeError(f"{what} must be a row position or a timestamp, got {when!r}.")
    if isinstance(when, int | np.integer):
        return int(when) if when >= 0 else int(timestamps.shape[0]) + int(when)
    value = when.value if isinstance(when, Time) else when
    key: Any
    if timestamps.dtype.kind == "M":
        if not isinstance(value, _DATETIMES):
            raise ValueError(
                f"{what} {when!r} is not a date or time, but the timestamps are datetimes. "
                "Pass anything numpy.datetime64 accepts (a datetime64, datetime, pandas "
                "Timestamp or date string), or an int row position."
            )
        key = naive_datetime64(value, what)
        if np.isnat(key):
            raise ValueError(f"{what} {when!r} is NaT.")
    else:
        if not isinstance(value, _NUMBERS) or isinstance(value, bool | np.bool_):
            raise ValueError(
                f"{what} {when!r} is not a number, but the timestamps are numeric "
                f"({timestamps[0]} .. {timestamps[-1]}). Pass an int row position or "
                "Time(t) for the timestamp t; to address dates, give the data datetime "
                "timestamps."
            )
        if not np.isfinite(value):
            raise ValueError(f"{what} {when!r} is not finite.")
        key = value
    position: Any = np.searchsorted(timestamps, key, side=side)  # type: ignore[call-overload,unused-ignore]
    return int(position)


class MarketData:
    """An immutable panel of market data aligned on one time axis.

    Args:
        prices: Prices shaped `(T, N)` (or `(T,)` for one asset); stored as the
            field `price_field`. NaN marks an asset that is not listed at that
            time. Finite prices must be positive.
        fields: More per-asset data, each `(T, N)` or `(T, N, ...)`: volumes, spreads,
            features, or several values per asset and time such as a risk model's
            factor loadings `(T, N, k)`. NaN is allowed; infinities are not.
        timestamps: `(T,)` strictly increasing `numpy.datetime64` values (or
            anything numpy converts to them, such as date strings) or numbers.
            Defaults to `0, 1, ..., T-1`. Timezone-aware values are refused (see
            the module docstring for the explicit conversion).
        assets: `N` unique asset identifiers. Defaults to `"asset_0", ...`.
        exogenous: Market-wide series, each `(T,)` or `(T, ...)`: interest rates,
            an index, macro data aligned with `sekai.finance.features.asof_join`, or
            matrices such as a factor covariance `(T, k, k)`.
        price_field: Name of the field that holds the prices.
        warmup: Number of leading rows that are lookback-only history (`split` and
            `walk_forward` set it): environments may observe them, but never make a
            decision on them. Between 0 and T.
        validate: Check the values (no infinities, positive prices, increasing
            timestamps). False skips the checks that read every value, for data already
            validated, such as a `load()` of data that `save()` wrote.
        corporate_actions: Read the fields `split` and `dividend` as corporate actions
            (see `simple_returns`): they are checked, returns include them, and
            environments and brokers apply them. Off by default, so fields with those
            names are ordinary fields, as they were in 0.2.
        copy: Copy the arrays given (the default, as in 0.2), so the data never changes
            whatever happens to them later. With `copy=False`, float64 arrays are kept as
            they are (read-only views of them), sharing their memory: for memory maps you
            opened yourself, say. Those arrays must then not change.

    A MarketData never changes once built. Data derived from it shares its arrays rather
    than copying them (`with_fields`, `adjusted`), as do memory-mapped data
    (`MarketData.load`) and its row slices; slices and splits of data held in memory are
    copies, so they don't keep the whole history alive.

    Example:
        ```python
        data = MarketData(closes, timestamps=dates, assets=tickers, fields={"volume": volumes})
        train, test = data.split(0.8, embargo=5, history=60)

        # Raw (unadjusted) prices with their splits and dividends:
        data = MarketData(
            raw_closes, fields={"split": splits, "dividend": dividends}, corporate_actions=True
        )
        ```
    """

    def __init__(
        self,
        prices: ArrayLike | None = None,
        *,
        fields: Mapping[str, ArrayLike] | None = None,
        timestamps: ArrayLike | None = None,
        assets: Sequence[Any] | None = None,
        exogenous: Mapping[str, ArrayLike] | None = None,
        price_field: str = "close",
        warmup: int = 0,
        validate: bool = True,
        corporate_actions: bool = False,
        copy: bool = True,
    ) -> None:
        raw: dict[str, ArrayLike] = dict(fields or {})
        if prices is not None:
            if price_field in raw:
                raise ValueError(f"prices given twice: as prices= and as fields[{price_field!r}].")
            raw[price_field] = prices
        if price_field not in raw:
            raise ValueError(f"MarketData needs prices (or a {price_field!r} field).")

        arrays = {
            name: _as_field(name, value, copy=copy, validate=validate)
            for name, value in raw.items()
        }
        shape = arrays[price_field].shape
        if len(shape) != 2:
            raise ValueError(f"prices ({price_field!r}) must be 2-D (T, N), got shape {shape}.")
        for name, arr in arrays.items():
            if arr.shape[:2] != shape:
                raise ValueError(
                    f"field {name!r} has shape {arr.shape} but {price_field!r} has {shape}; "
                    "a field's first two dimensions are (T, N)."
                )
        n_steps, n_assets = shape
        if n_steps < 2 or n_assets < 1:
            raise ValueError(f"MarketData needs at least 2 rows and 1 asset, got shape {shape}.")
        price = arrays[price_field]
        if validate and np.any(price[np.isfinite(price)] <= 0):
            raise ValueError("prices must be positive where they are defined (NaN = not listed).")
        if corporate_actions:
            if "split" not in arrays and "dividend" not in arrays:
                raise ValueError(
                    "corporate_actions=True needs a 'split' or a 'dividend' field (or both)."
                )
            _check_corporate_actions(arrays, validate)

        if timestamps is None:
            times: NDArray[Any] = np.arange(n_steps, dtype=np.int64)
        else:
            times = np.array(timestamps)
            if times.dtype.kind in "OUS":  # date strings, datetime objects, pandas Timestamps
                times = naive_datetime64(times, "timestamps", unit="ns")
            if times.dtype.kind not in "iufM":
                raise TypeError(f"timestamps must be datetime64 or numeric, got {times.dtype}.")
        if times.shape != (n_steps,):
            raise ValueError(f"timestamps has shape {times.shape}; expected ({n_steps},).")
        if validate:
            if times.dtype.kind == "M" and np.isnat(times).any():
                raise ValueError("timestamps contain NaT.")
            if times.dtype.kind == "f" and not np.all(np.isfinite(times)):
                raise ValueError("timestamps must be finite.")
            if not np.all(times[1:] > times[:-1]):
                raise ValueError("timestamps must be strictly increasing.")

        ids = tuple(f"asset_{i}" for i in range(n_assets)) if assets is None else tuple(assets)
        if len(ids) != n_assets:
            raise ValueError(f"Got {len(ids)} asset ids for {n_assets} assets.")
        if len(set(ids)) != n_assets:
            raise ValueError("asset ids must be unique.")

        exo = {
            name: _readonly(_as_exogenous(name, value, n_steps, copy=copy, validate=validate))
            for name, value in (exogenous or {}).items()
        }

        if isinstance(warmup, bool | np.bool_):
            raise TypeError(f"warmup must be an integer, got {warmup!r}.")
        try:
            n_warmup = operator.index(warmup)
        except TypeError:
            raise TypeError(f"warmup must be an integer, got {warmup!r}.") from None
        if not 0 <= n_warmup <= n_steps:
            raise ValueError(f"warmup must be in [0, {n_steps}], got {n_warmup}.")

        self._fields = {name: _readonly(arr) for name, arr in arrays.items()}
        self._timestamps = _readonly(times)
        self._assets = ids
        self._exogenous = exo
        self.price_field = price_field
        self._warmup = n_warmup
        self._corporate_actions = bool(corporate_actions)
        self._mask: NDArray[np.bool_] = _readonly(np.isfinite(price))
        self._returns: NDArray[np.float64] | None = None
        # (directory, first row, end row, save id) when the arrays are memory-mapped from a
        # saved MarketData: pickling then sends that, not the arrays (see __reduce_ex__).
        self._origin: tuple[str, int, int, str | None] | None = None

    @classmethod
    def _from_arrays(
        cls,
        *,
        fields: dict[str, NDArray[np.float64]],
        timestamps: NDArray[Any],
        assets: tuple[Any, ...],
        exogenous: dict[str, NDArray[np.float64]],
        price_field: str,
        warmup: int,
        corporate_actions: bool,
        mask: NDArray[np.bool_],
        returns: NDArray[np.float64] | None = None,
        origin: tuple[str, int, int, str | None] | None = None,
    ) -> MarketData:
        """A MarketData of arrays that are already checked and read-only: no checks, no
        copies, nothing read (how derived and loaded data are built)."""
        out = cls.__new__(cls)
        out._fields = fields
        out._timestamps = timestamps
        out._assets = assets
        out._exogenous = exogenous
        out.price_field = price_field
        out._warmup = warmup
        out._corporate_actions = corporate_actions
        out._mask = mask
        out._returns = returns
        out._origin = origin
        return out

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Data pickled by sekai 0.2 has none of the attributes added since.
        state.setdefault("_corporate_actions", False)
        state.setdefault("_returns", None)
        state.setdefault("_origin", None)
        self.__dict__.update(state)
        # NumPy 1.x pickles datetime dtypes with (empty) metadata, which np.save warns about.
        times = self._timestamps
        if times.dtype.metadata is not None:
            self._timestamps = times.view(np.dtype(times.dtype.str))
        # Unpickled arrays come back writeable; a MarketData's are read-only.
        arrays = [*self._fields.values(), *self._exogenous.values(), self._timestamps, self._mask]
        for arr in [*arrays, self._returns]:
            if arr is not None:
                arr.setflags(write=False)

    # ------------------------------------------------------------------
    # Construction from pandas
    # ------------------------------------------------------------------

    @classmethod
    def from_pandas(
        cls,
        prices: Any,
        *,
        fields: Mapping[str, Any] | None = None,
        exogenous: Any = None,
        price_field: str = "close",
        corporate_actions: bool = False,
    ) -> MarketData:
        """Build from wide pandas DataFrames (index = timestamps, columns = assets).

        Every field is aligned to the prices' index and columns (missing → NaN).
        `exogenous` is a DataFrame (one series per column) or a mapping of name
        to Series/DataFrame, aligned to the same index. `corporate_actions` is as in
        the constructor.
        """
        index = prices.index
        columns = prices.columns
        aligned = {
            name: frame.reindex(index=index, columns=columns)
            for name, frame in (fields or {}).items()
        }
        exo: dict[str, Any] = {}
        if exogenous is not None:  # a DataFrame or a mapping: both have .items()
            for name, series in exogenous.items():
                exo[str(name)] = series.reindex(index).to_numpy(dtype=np.float64)
        return cls(
            prices.to_numpy(dtype=np.float64),
            fields={name: frame.to_numpy(dtype=np.float64) for name, frame in aligned.items()},
            timestamps=index.to_numpy(),
            assets=[str(c) for c in columns],
            exogenous=exo,
            price_field=price_field,
            corporate_actions=corporate_actions,
        )

    @classmethod
    def from_long(
        cls,
        frame: Any,
        *,
        time: str = "date",
        asset: str = "symbol",
        price: str = "close",
        fields: Sequence[str] = (),
        corporate_actions: bool = False,
    ) -> MarketData:
        """Build from a long ("tidy") DataFrame with one row per (time, asset).

        Columns `time` and `asset` identify the row, `price` holds the price, and
        each column in `fields` becomes a field of the same name. `corporate_actions`
        is as in the constructor.
        """
        wide = frame.pivot_table(index=time, columns=asset, values=[price, *fields], aggfunc="last")
        wide = wide.sort_index()
        prices = wide[price]
        return cls.from_pandas(
            prices,
            fields={name: wide[name] for name in fields},
            price_field=price,
            corporate_actions=corporate_actions,
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def prices(self) -> NDArray[np.float64]:
        """Prices `(T, N)`, NaN where an asset is not listed (read-only)."""
        return self._fields[self.price_field]

    @property
    def mask(self) -> NDArray[np.bool_]:
        """`(T, N)` True where the asset has a price (is listed and tradable)."""
        return self._mask

    @property
    def fields(self) -> Mapping[str, NDArray[np.float64]]:
        """Every per-asset field, including prices: `(T, N)` or `(T, N, ...)` each."""
        return dict(self._fields)

    @property
    def exogenous(self) -> Mapping[str, NDArray[np.float64]]:
        """Market-wide series, each `(T,)` or `(T, ...)`."""
        return dict(self._exogenous)

    @property
    def timestamps(self) -> NDArray[Any]:
        return self._timestamps

    @property
    def assets(self) -> tuple[Any, ...]:
        return self._assets

    @property
    def n_steps(self) -> int:
        return int(self._timestamps.shape[0])

    @property
    def n_assets(self) -> int:
        return len(self._assets)

    @property
    def is_datetime(self) -> bool:
        return self._timestamps.dtype.kind == "M"

    @property
    def warmup(self) -> int:
        """Leading rows that are lookback-only history: observable, but never a decision row.

        `split` and `walk_forward` set it to the `history` rows they prepend to a test
        period; slicing keeps it correct (a slice starting inside the warm-up keeps the
        remaining warm-up rows), and it is 0 for data built directly.
        """
        return self._warmup

    def field(self, name: str) -> NDArray[np.float64]:
        """One per-asset field, `(T, N)` or `(T, N, ...)`."""
        try:
            return self._fields[name]
        except KeyError:
            raise KeyError(f"No field {name!r}; available: {sorted(self._fields)}.") from None

    def exogenous_series(self, name: str) -> NDArray[np.float64]:
        """One market-wide series."""
        try:
            return self._exogenous[name]
        except KeyError:
            raise KeyError(
                f"No exogenous series {name!r}; available: {sorted(self._exogenous)}."
            ) from None

    def exogenous_width(self, name: str) -> int:
        """Number of values per row of an exogenous series (1 for a 1-D series; the
        product of the trailing dimensions otherwise)."""
        arr = self.exogenous_series(name)
        return int(np.prod(arr.shape[1:], dtype=np.int64))

    @property
    def periods_per_year(self) -> float | None:
        """Rows per year implied by datetime timestamps (about 252 for trading days), else None."""
        if not self.is_datetime:
            return None
        span = (self._timestamps[-1] - self._timestamps[0]) / np.timedelta64(1, "D")
        return float((self.n_steps - 1) / (span / 365.25)) if span > 0 else None

    def simple_returns(self) -> NDArray[np.float64]:
        """`(T, N)` simple returns of holding each asset over `(t - 1, t]`; NaN in row 0 and
        wherever either price is missing (read-only, computed once).

        Without corporate actions this is `prices[t] / prices[t - 1] - 1`. Data built
        with `corporate_actions=True` may have a `split` field (new shares per old share,
        effective at row t: 2.0 for a 2-for-1 split; NaN means none) and a `dividend`
        field (cash per share, after any split, going ex at row t; NaN means none); the
        return is then the total return of a holder:
        `split[t] * (prices[t] + dividend[t]) / prices[t - 1] - 1`, so raw (unadjusted)
        prices give the right returns.
        """
        if self._returns is None:
            price = self.prices
            out = np.full(price.shape, np.nan)
            now = price[1:]
            if self._corporate_actions and "dividend" in self._fields:
                now = now + np.nan_to_num(self._fields["dividend"][1:], nan=0.0)
            if self._corporate_actions and "split" in self._fields:
                ratio = self._fields["split"][1:]
                now = now * np.where(np.isfinite(ratio), ratio, 1.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[1:] = now / price[:-1] - 1.0
            self._returns = _readonly(out)
        return self._returns

    @property
    def has_corporate_actions(self) -> bool:
        """Whether the `split` and `dividend` fields are corporate actions (the data was
        built with `corporate_actions=True`; see `simple_returns`)."""
        return self._corporate_actions

    def adjusted(self) -> MarketData:
        """A copy whose prices are a total-return index, with the corporate-action fields removed.

        Data without corporate actions is returned as it is.

        Each listed stretch of an asset starts at its raw price and then moves with
        `simple_returns()`, so splits no longer look like crashes and dividends are
        reinvested. The index is built forwards from the first price, using only
        corporate actions up to each row: unlike the usual back-adjustment, which rescales
        history with later splits and dividends, it never puts future information in a
        past price. Use it for environments that compute returns from prices
        (`Trading-v0/v1`, `Portfolio-v0/v1`, `Market-v0`); `Market-v1` reads
        `simple_returns()` and takes raw prices directly. Other fields are unchanged:
        derive a share-count field such as `volume` into a traded value first
        (`dollar_volume`, which `Market-v1`'s cost models prefer) if a split affects it.
        """
        if not self._corporate_actions:
            return self
        price = self.prices
        r = self.simple_returns()
        index = np.array(price, copy=True)
        for t in range(1, self.n_steps):
            ok = np.isfinite(r[t]) & np.isfinite(index[t - 1])
            index[t] = np.where(ok, index[t - 1] * (1.0 + r[t]), price[t])
        fields = {k: v for k, v in self._fields.items() if k not in ("split", "dividend")}
        # Positive and listed where the raw prices are (total returns are above -100%).
        fields[self.price_field] = _readonly(index)
        return MarketData._from_arrays(
            fields=fields,
            timestamps=self._timestamps,
            assets=self._assets,
            exogenous=self._exogenous,
            price_field=self.price_field,
            warmup=self._warmup,
            corporate_actions=False,
            mask=self._mask,
        )

    def sessions(self) -> NDArray[np.int64]:
        """`(T,)` trading-session number of each row (0, 1, ...), for intraday data.

        From the `session` exogenous series if there is one (any labels; a new session
        starts wherever it changes), else from datetime timestamps (one session per
        calendar date).

        Raises:
            ValueError: For numeric timestamps without a `session` series.
        """
        if "session" in self._exogenous:
            labels = np.asarray(self._exogenous["session"]).reshape(self.n_steps, -1)[:, 0]
        elif self.is_datetime:
            labels = self._timestamps.astype("datetime64[D]").astype(np.int64)
        else:
            raise ValueError(
                "Sessions need datetime timestamps or a 'session' exogenous series; these "
                "timestamps are numeric."
            )
        change = np.concatenate([[False], labels[1:] != labels[:-1]])
        out: NDArray[np.int64] = np.cumsum(change).astype(np.int64)
        return out

    def log_returns(self) -> NDArray[np.float64]:
        """`(T, N)` log returns of the prices; NaN in row 0 and wherever either price is missing."""
        out = np.full(self.prices.shape, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[1:] = np.log(self.prices[1:] / self.prices[:-1])
        return out

    def index_of(self, when: TimeLike, *, side: str = "left") -> int:
        """Row position of a point in time: a row position or a timestamp.

        - An `int` is a row position on every time axis (negative counts from the
          end), returned as is.
        - A timestamp gives the first row at or after it (after it with
          `side="right"`), or `n_steps` if there is none. On datetime timestamps it is
          anything `numpy.datetime64` accepts (a `datetime64`, `datetime`, `date`,
          pandas `Timestamp` or date string); on numeric timestamps, a float.
        - `Time(t)` is a timestamp on either kind of axis, so it also looks up integer
          timestamps (epoch seconds, or the timestamps of a split).

        Raises:
            TypeError: If `when` is a bool.
            ValueError: If a timestamp does not suit the axis (a date on numeric
                timestamps, a number on datetime ones), is NaN or NaT, or is
                timezone-aware.
        """
        return row_of(self._timestamps, when, side=side)

    # ------------------------------------------------------------------
    # Derived data
    # ------------------------------------------------------------------

    def _replace(
        self,
        *,
        rows: slice | NDArray[np.intp] | None = None,
        cols: NDArray[np.intp] | None = None,
        warmup: int = 0,
    ) -> MarketData:
        """The rows `rows` (a slice with a positive step) and the assets `cols`.

        Memory-mapped arrays stay mapped (the subset is a view of the files); arrays held
        in memory are copied, so a small slice doesn't keep the whole history alive. The
        subset's warm-up is this data's warm-up rows that `rows` keeps (they stay
        leading, since timestamps increase), or `warmup` leading rows if more.
        """
        r: Any = slice(None) if rows is None else rows
        if isinstance(r, slice) and r.step is not None and r.step < 1:
            raise ValueError(
                f"MarketData slices must keep time increasing (a positive step), got {r.step}."
            )
        c: Any = slice(None) if cols is None else cols
        assets = self._assets if cols is None else tuple(self._assets[i] for i in cols)
        kept = np.count_nonzero(np.arange(self.n_steps)[r] < self._warmup)

        def part(arr: NDArray[Any], *, by_asset: bool) -> NDArray[Any]:
            out = arr[r][:, c] if by_asset and cols is not None else arr[r]
            if not _is_mapped(arr) and np.may_share_memory(out, arr):
                out = out.copy()
            return _readonly(out)

        n_rows = len(range(*r.indices(self.n_steps))) if isinstance(r, slice) else len(r)
        shape = (n_rows, self.n_assets if cols is None else len(cols))
        if shape[0] < 2 or shape[1] < 1:
            raise ValueError(f"MarketData needs at least 2 rows and 1 asset, got shape {shape}.")
        start, stop, step = r.indices(self.n_steps) if isinstance(r, slice) else (0, 0, 0)
        contiguous = isinstance(r, slice) and step == 1
        # Returns carry over only from row 0: a later first row has no price before it,
        # so its return is NaN in the subset (computed when first needed).
        returns = self._returns if contiguous and start == 0 else None
        origin = None
        if contiguous and cols is None and self._origin is not None:
            directory, first, _, save_id = self._origin
            origin = (directory, first + start, first + max(start, stop), save_id)
        return MarketData._from_arrays(
            fields={name: part(arr, by_asset=True) for name, arr in self._fields.items()},
            timestamps=part(self._timestamps, by_asset=False),
            assets=assets,
            exogenous={name: part(arr, by_asset=False) for name, arr in self._exogenous.items()},
            price_field=self.price_field,
            warmup=max(int(kept), warmup),
            corporate_actions=self._corporate_actions,
            mask=part(self._mask, by_asset=True),
            returns=None if returns is None else part(returns, by_asset=True),
            origin=origin,
        )

    def __getitem__(self, rows: slice) -> MarketData:
        """Positional time slicing: `data[100:500]`."""
        if not isinstance(rows, slice):
            raise TypeError("MarketData supports slicing by row position, e.g. data[100:500].")
        return self._replace(rows=rows)

    def between(self, start: TimeLike | None = None, end: TimeLike | None = None) -> MarketData:
        """Rows from `start` up to, not including, `end` (either bound optional).

        Each bound is read as by `index_of`: timestamps select `start <= timestamp <
        end` (`between("2020-01-01", "2021-01-01")` is the rows of 2020, and on numeric
        timestamps `between(Time(100), Time(200))` the rows stamped 100 to 199), while
        an `int` is a row position (`between(100, 200)` is `data[100:200]`).
        """
        lo = 0 if start is None else self.index_of(start)
        hi = self.n_steps if end is None else self.index_of(end)
        return self[lo:hi]

    def select(self, assets: Sequence[Any]) -> MarketData:
        """A subset of assets, by identifier."""
        lookup = {a: i for i, a in enumerate(self._assets)}
        missing = [a for a in assets if a not in lookup]
        if missing:
            raise KeyError(f"Unknown assets: {missing}.")
        if len(set(assets)) != len(assets):
            raise ValueError("asset ids must be unique.")
        return self._replace(cols=np.array([lookup[a] for a in assets], dtype=np.intp))

    def with_fields(self, **arrays: ArrayLike) -> MarketData:
        """A copy with fields added or replaced (each `(T, N)` or `(T, N, ...)`).

        The new arrays are copied and checked; the other fields are shared, not copied
        (memory-mapped ones stay mapped).
        """
        if self.price_field in arrays:  # new prices: check everything again
            fresh = {k: np.array(v, dtype=np.float64) for k, v in arrays.items()}
            return MarketData(
                fields={**self._fields, **fresh},
                timestamps=self._timestamps,
                assets=self._assets,
                exogenous=self._exogenous,
                price_field=self.price_field,
                warmup=self._warmup,
                corporate_actions=self._corporate_actions,
                copy=False,  # the rest is this data's own, read-only
            )
        shape = self.prices.shape
        new = {}
        for name, value in arrays.items():
            arr = _as_field(name, value, copy=True, validate=True)
            if arr.shape[:2] != shape:
                raise ValueError(
                    f"field {name!r} has shape {arr.shape} but {self.price_field!r} has {shape}; "
                    "a field's first two dimensions are (T, N)."
                )
            new[name] = _readonly(arr)
        actions = self._corporate_actions and bool({"split", "dividend"} & set(new))
        if actions:
            _check_corporate_actions(new, validate=True)
        return MarketData._from_arrays(
            fields={**self._fields, **new},
            timestamps=self._timestamps,
            assets=self._assets,
            exogenous=self._exogenous,
            price_field=self.price_field,
            warmup=self._warmup,
            corporate_actions=self._corporate_actions,
            mask=self._mask,
            returns=None if actions else self._returns,
        )

    def with_exogenous(self, **arrays: ArrayLike) -> MarketData:
        """A copy with exogenous series added or replaced (each `(T,)` or `(T, ...)`).

        The new series are copied and checked; everything else is shared, not copied.
        """
        new = {
            name: _readonly(_as_exogenous(name, value, self.n_steps, copy=True, validate=True))
            for name, value in arrays.items()
        }
        return MarketData._from_arrays(
            fields=self._fields,
            timestamps=self._timestamps,
            assets=self._assets,
            exogenous={**self._exogenous, **new},
            price_field=self.price_field,
            warmup=self._warmup,
            corporate_actions=self._corporate_actions,
            mask=self._mask,
            returns=self._returns,
        )

    # ------------------------------------------------------------------
    # Saving, memory mapping and pickling
    # ------------------------------------------------------------------

    def save(self, path: str | os.PathLike[str], *, overwrite: bool = False) -> Path:
        """Write the data to a directory of `.npy` files, for `MarketData.load()`.

        The directory holds one file per field and exogenous series, the timestamps, the
        validity mask, the simple returns and a `metadata.json` (names, asset ids, price
        field, warm-up, corporate actions). Asset ids must be JSON values (strings or
        numbers, NumPy ones included).

        The files are written to a new directory next to `path` that then takes its
        place, so `path` is never left half-written, and data memory-mapped from an
        earlier save there (even this data itself: `load(p).save(p, overwrite=True)`)
        is not destroyed: it keeps reading the old files.

        Args:
            path: The directory (created if needed).
            overwrite: Replace a MarketData already saved there.

        Returns:
            The directory.

        Other files in the directory are kept, a symbolic link to a directory is saved
        through, and with a read-only parent (no room for the new copy) the files are
        replaced in place, data first and metadata last.

        Raises:
            FileExistsError: If the directory is not empty and `overwrite` is False, or
                it holds files that are not a saved MarketData (which are never deleted).
            TypeError: If an asset id is not a JSON value.
            OSError: If the directory can't be replaced (on Windows, while its files are
                memory-mapped).
        """
        root = Path(path)
        if root.exists() and not root.is_dir():
            raise FileExistsError(f"{root} is a file, not a directory.")
        if root.exists() and any(root.iterdir()):
            if not overwrite:
                raise FileExistsError(f"{root} is not empty; pass overwrite=True to replace it.")
            if not _holds_saved_data(root):
                raise FileExistsError(
                    f"{root} holds files that are not a saved MarketData; not replacing them."
                )
        assets = [a.item() if isinstance(a, np.generic) else a for a in self._assets]
        for a in assets:
            if isinstance(a, bool) or not isinstance(a, (str, int, float)):
                raise TypeError(f"save() needs asset ids that are strings or numbers, got {a!r}.")
        target = root.resolve() if root.is_symlink() else root  # where a link points
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            tmp = Path(
                tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
            )
        except PermissionError:  # the parent is read-only: replace the files one by one
            target.mkdir(exist_ok=True)
            self._save_in_place(target, assets)
            return root
        try:
            self._write(tmp, assets)
            _replace_dir(tmp, target)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return root

    def _save_in_place(self, root: Path, assets: list[Any]) -> None:
        """`save` into `root` file by file (data first, metadata last), for a directory
        whose parent can't hold the new copy that `save` otherwise swaps in."""
        tmp = Path(tempfile.mkdtemp(prefix=".save.", dir=root))
        try:
            self._write(tmp, assets)
            for sub in ("fields", "exogenous"):
                (root / sub).mkdir(exist_ok=True)
                for f in (tmp / sub).iterdir():
                    os.replace(f, root / sub / f.name)
                kept = {f.name for f in (tmp / sub).iterdir()} | {
                    f"{i}.npy"
                    for i in range(len(self._fields if sub == "fields" else self._exogenous))
                }
                for f in (root / sub).glob("*.npy"):  # a field of the previous save only
                    if f.name not in kept:
                        f.unlink()
            for name in ("timestamps.npy", "mask.npy", "returns.npy", "metadata.json"):
                os.replace(tmp / name, root / name)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def _write(self, root: Path, assets: list[Any]) -> None:
        (root / "fields").mkdir()
        (root / "exogenous").mkdir()
        fields = list(self._fields)
        exogenous = list(self._exogenous)
        for i, name in enumerate(fields):
            np.save(root / "fields" / f"{i}.npy", np.ascontiguousarray(self._fields[name]))
        for i, name in enumerate(exogenous):
            np.save(root / "exogenous" / f"{i}.npy", np.ascontiguousarray(self._exogenous[name]))
        np.save(root / "timestamps.npy", np.ascontiguousarray(self._timestamps))
        np.save(root / "mask.npy", np.ascontiguousarray(self._mask))
        np.save(root / "returns.npy", np.ascontiguousarray(self.simple_returns()))
        meta = {
            "format": _FORMAT,
            "version": _FORMAT_VERSION,
            "fields": fields,
            "exogenous": exogenous,
            "assets": assets,
            "price_field": self.price_field,
            "warmup": self._warmup,
            "corporate_actions": self._corporate_actions,
            # Lets a pickled slice tell this save from a later one at the same path.
            "save_id": uuid.uuid4().hex,
        }
        (root / "metadata.json").write_text(json.dumps(meta, indent=1))

    @classmethod
    def load(cls, path: str | os.PathLike[str], *, mmap: bool = True) -> MarketData:
        """Read data written by `save()`.

        With `mmap=True` (the default) the arrays are memory-mapped read-only: nothing is
        read until it is used, the operating system shares the pages between processes,
        and pickling the data (as `ProcVecEnv` does for its workers) sends the directory
        and row range, not the arrays, so every worker maps the same files instead of
        holding a copy. Slicing rows (`data[a:b]`, `split`, `walk_forward`, `between`)
        keeps that. `select` copies the assets it keeps into memory, and `with_fields`
        shares the mapped fields but pickles by value, since its new fields are not in
        the files.

        Args:
            path: The directory `save()` wrote.
            mmap: Memory-map the arrays (else read them into memory).

        Raises:
            ValueError: If the directory does not hold a saved MarketData, or its files
                don't agree with each other.
        """
        root = Path(path)
        try:
            meta = json.loads((root / "metadata.json").read_text())
        except FileNotFoundError:
            raise ValueError(
                f"{root} does not hold a saved MarketData (no metadata.json)."
            ) from None
        if meta.get("format") != _FORMAT or meta.get("version") != _FORMAT_VERSION:
            raise ValueError(
                f"{root} holds {meta.get('format')!r} version {meta.get('version')!r}."
            )
        mode: Literal["r"] | None = "r" if mmap else None

        def read(name: str) -> NDArray[Any]:
            arr = np.load(root / name, mmap_mode=mode, allow_pickle=False)
            return arr if mmap else _readonly(arr)

        fields = {name: read(f"fields/{i}.npy") for i, name in enumerate(meta["fields"])}
        exogenous = {name: read(f"exogenous/{i}.npy") for i, name in enumerate(meta["exogenous"])}
        timestamps = read("timestamps.npy")
        mask = read("mask.npy")
        returns = read("returns.npy")
        price_field = meta["price_field"]
        # The shapes come from the file headers: checking them reads no data.
        shape = (len(timestamps), len(meta["assets"]))
        bad = [n for n, a in fields.items() if a.shape[:2] != shape]
        bad += [n for n, a in exogenous.items() if a.shape[:1] != shape[:1]]
        if price_field not in fields or bad or mask.shape != shape or returns.shape != shape:
            raise ValueError(f"The files in {root} don't agree with each other: {bad or shape}.")
        save_id = meta.get("save_id")
        return cls._from_arrays(
            fields=fields,
            timestamps=timestamps,
            assets=tuple(meta["assets"]),
            exogenous=exogenous,
            price_field=price_field,
            warmup=int(meta["warmup"]),
            corporate_actions=bool(meta.get("corporate_actions", False)),
            mask=mask,
            returns=returns,
            origin=(str(root.resolve()), 0, shape[0], save_id) if mmap else None,
        )

    @property
    def memory_mapped(self) -> bool:
        """Whether the arrays are memory-mapped from a saved MarketData (see `load`)."""
        return self._origin is not None

    def __reduce_ex__(self, protocol: Any) -> Any:
        if self._origin is not None:
            directory, start, stop, save_id = self._origin
            if _saved_id(directory) == save_id:  # the files are still this data's
                return (_load_rows, (directory, start, stop, self._warmup, save_id))
            # Saved over since (this data reads the old files): pickle the values instead.
            state = dict(self.__dict__, _origin=None)
            return (_new_market_data, (), state)
        return super().__reduce_ex__(protocol)

    # ------------------------------------------------------------------
    # Chronological splits
    # ------------------------------------------------------------------

    def _boundary(self, b: TimeLike) -> int:
        if isinstance(b, float | np.floating):
            if not 0.0 < b < 1.0:
                raise ValueError(
                    f"Fractional split points must be in (0, 1), got {b}. A float is a "
                    f"fraction of the rows here: pass Time({b}) to split at a timestamp."
                )
            return round(float(b) * self.n_steps)
        return self.index_of(b)

    def split(self, *boundaries: TimeLike, embargo: int = 0, history: int = 0) -> list[MarketData]:
        """Split chronologically at the given points.

        A point is a fraction of the rows (a float in (0, 1), such as `0.8`), a row
        position (an `int`, counted within this data on every time axis, so
        `test.split(50)` cuts `test` at its own row 50; negative counts from the
        end), or a timestamp: on datetime timestamps anything `numpy.datetime64`
        accepts (`"2021-01-01"`), on any axis `Time(t)` (as in `index_of`; a float is
        a fraction here, so numeric timestamps need `Time`).
        Every part after the first starts `embargo` rows after its split point (a
        gap that keeps information from straddling the boundary) and then
        prepends `history` rows before that start, which an environment may
        observe as lookback without trading on them. The part records them as its
        `warmup`, and the finance environments never make a decision on a warm-up
        row, whatever their lookback (a shorter lookback just observes fewer of
        them). Pass at least the env's lookback as `history` so the first decision
        of a test episode can fall right at the start of the test period.

        Example:
            ```python
            train, val, test = data.split(0.6, 0.8, embargo=5, history=lookback)
            ```
        """
        if embargo < 0 or history < 0:
            raise ValueError("embargo and history must be >= 0.")
        if not boundaries:
            raise ValueError("split() needs at least one split point.")
        cuts = [self._boundary(b) for b in boundaries]
        if any(b <= a for a, b in zip([0, *cuts], [*cuts, self.n_steps])):
            raise ValueError(
                f"Split points must be increasing and inside the data, got rows {cuts}."
            )
        parts = [self[0 : cuts[0]]]
        for i, cut in enumerate(cuts):
            end = cuts[i + 1] if i + 1 < len(cuts) else self.n_steps
            start = cut + embargo
            if end - start < 2:
                raise ValueError(f"Part {i + 1} has fewer than 2 rows after the embargo.")
            lo = max(start - history, 0)
            parts.append(self._replace(rows=slice(lo, end), warmup=start - lo))
        return parts

    def walk_forward(
        self,
        train_size: int,
        test_size: int,
        *,
        step: int | None = None,
        embargo: int = 0,
        history: int = 0,
        expanding: bool = False,
    ) -> Iterator[tuple[MarketData, MarketData]]:
        """Rolling (or expanding) train/test folds for walk-forward evaluation.

        Fold k trains on `train_size` rows (all rows so far with `expanding=True`)
        and tests on the next `test_size` rows after an `embargo` gap, with
        `history` extra rows before the test period available as lookback (recorded
        as the test part's `warmup`, so no decision falls on them). Folds advance by
        `step` rows (default: `test_size`, so test periods tile the data without
        overlap).
        """
        if train_size < 2 or test_size < 2:
            raise ValueError("train_size and test_size must be >= 2.")
        if embargo < 0 or history < 0:
            raise ValueError("embargo and history must be >= 0.")
        step = test_size if step is None else step
        if step < 1:
            raise ValueError("step must be >= 1.")
        start = 0
        while True:
            train_end = start + train_size
            test_start = train_end + embargo
            test_end = test_start + test_size
            if test_end > self.n_steps:
                return
            train = self[0 if expanding else start : train_end]
            lo = max(test_start - history, 0)
            test = self._replace(rows=slice(lo, test_end), warmup=test_start - lo)
            yield train, test
            start += step

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.n_steps

    def __repr__(self) -> str:
        first, last = self._timestamps[0], self._timestamps[-1]
        if self.is_datetime:
            first, last = (np.datetime_as_string(t, unit="auto") for t in (first, last))
        extra = f", exogenous={sorted(self._exogenous)}" if self._exogenous else ""
        if self._warmup:
            extra += f", warmup={self._warmup}"
        if self._corporate_actions:
            extra += ", corporate_actions=True"
        return (
            f"MarketData(T={self.n_steps}, N={self.n_assets}, fields={sorted(self._fields)}"
            f"{extra}, {first} .. {last})"
        )


def _load_rows(
    directory: str, start: int, stop: int, warmup: int, save_id: str | None = None
) -> MarketData:
    """Rows `start:stop` of a saved MarketData, memory-mapped (how memory-mapped data unpickles).

    Raises:
        ValueError: If the data saved at `directory` was replaced by another save since.
    """
    data = MarketData.load(directory)
    assert data._origin is not None
    if save_id is not None and data._origin[3] != save_id:
        raise ValueError(
            f"The MarketData saved at {directory} was replaced by another save after this "
            "data was pickled from it."
        )
    if (start, stop) == (0, data.n_steps) and warmup == data.warmup:
        return data
    return data._replace(rows=slice(start, stop), warmup=warmup)


def _new_market_data() -> MarketData:
    return MarketData.__new__(MarketData)


def _saved_id(directory: str) -> str | None:
    """The save id in `directory`'s metadata (None if it has none or can't be read)."""
    try:
        meta = json.loads((Path(directory) / "metadata.json").read_text())
    except (OSError, ValueError):
        return "<unreadable>"
    return meta.get("save_id") if isinstance(meta, dict) else "<unreadable>"


# What a save consists of; anything else in its directory is the user's.
_SAVED_NAMES = frozenset(
    {"fields", "exogenous", "timestamps.npy", "mask.npy", "returns.npy", "metadata.json"}
)


def _holds_saved_data(root: Path) -> bool:
    try:
        meta = json.loads((root / "metadata.json").read_text())
    except (OSError, ValueError):
        return False
    return isinstance(meta, dict) and meta.get("format") == _FORMAT


def _replace_dir(new: Path, root: Path) -> None:
    """Put the directory `new` at `root`, replacing any directory there.

    Two renames, so `root` is never half-written. On POSIX systems arrays memory-mapped
    from the old files keep working (the files live on until they are unmapped); Windows
    refuses to rename a directory whose files are in use.
    """
    if not root.exists():
        os.replace(new, root)
        return
    old = root.with_name(f".{root.name}.{uuid.uuid4().hex}.old")
    try:
        os.replace(root, old)
    except OSError as e:
        raise OSError(
            f"Can't replace {root}: {e}. On Windows, a saved MarketData can't be replaced "
            "while it is memory-mapped; save to another directory."
        ) from e
    try:
        os.replace(new, root)
    except BaseException:
        os.replace(old, root)
        raise
    # Files that aren't part of the save (the user's) move across to the new directory.
    for entry in old.iterdir():
        if entry.name not in _SAVED_NAMES:
            try:
                os.replace(entry, root / entry.name)
            except OSError as e:
                raise OSError(
                    f"Saved to {root}, but {entry.name!r} from the previous directory could "
                    f"not be moved there; it is in {old}."
                ) from e
    shutil.rmtree(old, ignore_errors=True)


__all__ = ["MarketData", "Time"]
