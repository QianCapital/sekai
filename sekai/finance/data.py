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

Timestamps are timezone-naive. Timezone-aware input is refused rather than converted
silently; convert it first (`index.tz_convert("UTC").tz_localize(None)` for UTC, or
`index.tz_localize(None)` to keep local wall-clock times).
"""

from __future__ import annotations

import datetime as _dt
import operator
import warnings
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

TimeLike = int | np.integer[Any] | np.datetime64 | str | _dt.date | _dt.datetime


def _readonly(arr: NDArray[Any]) -> NDArray[Any]:
    arr.setflags(write=False)
    return arr


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


class MarketData:
    """An immutable panel of market data aligned on one time axis.

    Args:
        prices: Prices shaped `(T, N)` (or `(T,)` for one asset); stored as the
            field `price_field`. NaN marks an asset that is not listed at that
            time. Finite prices must be positive.
        fields: More per-asset data, each `(T, N)`: volumes, spreads, features.
            NaN is allowed; infinities are not.
        timestamps: `(T,)` strictly increasing `numpy.datetime64` values (or
            anything numpy converts to them, such as date strings) or numbers.
            Defaults to `0, 1, ..., T-1`. Timezone-aware values are refused (see
            the module docstring for the explicit conversion).
        assets: `N` unique asset identifiers. Defaults to `"asset_0", ...`.
        exogenous: Market-wide series, each `(T,)` or `(T, k)`: interest rates,
            an index, macro data aligned with `sekai.finance.features.asof_join`.
        price_field: Name of the field that holds the prices.
        warmup: Number of leading rows that are lookback-only history (`split` and
            `walk_forward` set it): environments may observe them, but never make a
            decision on them. Between 0 and T.

    Example:
        ```python
        data = MarketData(closes, timestamps=dates, assets=tickers, fields={"volume": volumes})
        train, test = data.split(0.8, embargo=5, history=60)
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
    ) -> None:
        raw: dict[str, ArrayLike] = dict(fields or {})
        if prices is not None:
            if price_field in raw:
                raise ValueError(f"prices given twice: as prices= and as fields[{price_field!r}].")
            raw[price_field] = prices
        if price_field not in raw:
            raise ValueError(f"MarketData needs prices (or a {price_field!r} field).")

        arrays: dict[str, NDArray[np.float64]] = {}
        for name, value in raw.items():
            arr = np.array(value, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr[:, None]
            if arr.ndim != 2:
                raise ValueError(f"field {name!r} must be 2-D (T, N), got shape {arr.shape}.")
            if np.isinf(arr).any():
                raise ValueError(f"field {name!r} contains infinite values; use NaN for missing.")
            arrays[name] = arr
        shape = arrays[price_field].shape
        for name, arr in arrays.items():
            if arr.shape != shape:
                raise ValueError(
                    f"field {name!r} has shape {arr.shape} but {price_field!r} has {shape}."
                )
        n_steps, n_assets = shape
        if n_steps < 2 or n_assets < 1:
            raise ValueError(f"MarketData needs at least 2 rows and 1 asset, got shape {shape}.")
        price = arrays[price_field]
        if np.any(price[np.isfinite(price)] <= 0):
            raise ValueError("prices must be positive where they are defined (NaN = not listed).")

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

        exo: dict[str, NDArray[np.float64]] = {}
        for name, value in (exogenous or {}).items():
            arr = np.array(value, dtype=np.float64)
            if arr.ndim not in (1, 2) or arr.shape[0] != n_steps:
                raise ValueError(
                    f"exogenous {name!r} must have shape ({n_steps},) or ({n_steps}, k), "
                    f"got {arr.shape}."
                )
            if np.isinf(arr).any():
                raise ValueError(f"exogenous {name!r} contains infinite values.")
            exo[name] = _readonly(arr)

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
        self._mask: NDArray[np.bool_] = _readonly(np.isfinite(price))

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
    ) -> MarketData:
        """Build from wide pandas DataFrames (index = timestamps, columns = assets).

        Every field is aligned to the prices' index and columns (missing → NaN).
        `exogenous` is a DataFrame (one series per column) or a mapping of name
        to Series/DataFrame, aligned to the same index.
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
    ) -> MarketData:
        """Build from a long ("tidy") DataFrame with one row per (time, asset).

        Columns `time` and `asset` identify the row, `price` holds the price, and
        each column in `fields` becomes a field of the same name.
        """
        wide = frame.pivot_table(index=time, columns=asset, values=[price, *fields], aggfunc="last")
        wide = wide.sort_index()
        prices = wide[price]
        return cls.from_pandas(
            prices,
            fields={name: wide[name] for name in fields},
            price_field=price,
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
        """Every per-asset field, including prices."""
        return dict(self._fields)

    @property
    def exogenous(self) -> Mapping[str, NDArray[np.float64]]:
        """Market-wide series, each `(T,)` or `(T, k)`."""
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
        """One per-asset field `(T, N)`."""
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
        """Number of columns of an exogenous series (1 for a 1-D series)."""
        arr = self.exogenous_series(name)
        return 1 if arr.ndim == 1 else int(arr.shape[1])

    @property
    def periods_per_year(self) -> float | None:
        """Rows per year implied by datetime timestamps (about 252 for trading days), else None."""
        if not self.is_datetime:
            return None
        span = (self._timestamps[-1] - self._timestamps[0]) / np.timedelta64(1, "D")
        return float((self.n_steps - 1) / (span / 365.25)) if span > 0 else None

    def log_returns(self) -> NDArray[np.float64]:
        """`(T, N)` log returns of the prices; NaN in row 0 and wherever either price is missing."""
        out = np.full(self.prices.shape, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[1:] = np.log(self.prices[1:] / self.prices[:-1])
        return out

    def index_of(self, when: TimeLike, *, side: str = "left") -> int:
        """Row position of a timestamp (the first row at or after it, by default).

        Integers are taken as row positions when the timestamps are datetimes.
        """
        if self.is_datetime and isinstance(when, (int, np.integer)):
            return int(when) if when >= 0 else self.n_steps + int(when)
        key: Any = naive_datetime64(when, "the time") if self.is_datetime else when
        position: Any = np.searchsorted(self._timestamps, key, side=side)  # type: ignore[call-overload,unused-ignore]
        return int(position)

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
        """A copy restricted to `rows` and `cols`.

        The copy's warm-up is this data's warm-up rows that `rows` keeps (they stay
        leading, since timestamps must increase), or `warmup` leading rows if more.
        """
        r: Any = slice(None) if rows is None else rows
        c: Any = slice(None) if cols is None else cols
        assets = self._assets if cols is None else tuple(self._assets[i] for i in cols)
        kept = np.count_nonzero(np.arange(self.n_steps)[r] < self._warmup)
        return MarketData(
            fields={name: arr[r][:, c] for name, arr in self._fields.items()},
            timestamps=self._timestamps[r],
            assets=assets,
            exogenous={name: arr[r] for name, arr in self._exogenous.items()},
            price_field=self.price_field,
            warmup=max(int(kept), warmup),
        )

    def __getitem__(self, rows: slice) -> MarketData:
        """Positional time slicing: `data[100:500]`."""
        if not isinstance(rows, slice):
            raise TypeError("MarketData supports slicing by row position, e.g. data[100:500].")
        return self._replace(rows=rows)

    def between(self, start: TimeLike | None = None, end: TimeLike | None = None) -> MarketData:
        """Rows with `start <= timestamp < end` (either bound optional)."""
        lo = 0 if start is None else self.index_of(start)
        hi = self.n_steps if end is None else self.index_of(end)
        return self[lo:hi]

    def select(self, assets: Sequence[Any]) -> MarketData:
        """A subset of assets, by identifier."""
        lookup = {a: i for i, a in enumerate(self._assets)}
        missing = [a for a in assets if a not in lookup]
        if missing:
            raise KeyError(f"Unknown assets: {missing}.")
        return self._replace(cols=np.array([lookup[a] for a in assets], dtype=np.intp))

    def with_fields(self, **arrays: ArrayLike) -> MarketData:
        """A copy with fields added or replaced (each `(T, N)`)."""
        return MarketData(
            fields={**self._fields, **arrays},
            timestamps=self._timestamps,
            assets=self._assets,
            exogenous=self._exogenous,
            price_field=self.price_field,
            warmup=self._warmup,
        )

    def with_exogenous(self, **arrays: ArrayLike) -> MarketData:
        """A copy with exogenous series added or replaced (each `(T,)` or `(T, k)`)."""
        return MarketData(
            fields=self._fields,
            timestamps=self._timestamps,
            assets=self._assets,
            exogenous={**self._exogenous, **arrays},
            price_field=self.price_field,
            warmup=self._warmup,
        )

    # ------------------------------------------------------------------
    # Chronological splits
    # ------------------------------------------------------------------

    def _boundary(self, b: float | TimeLike) -> int:
        if isinstance(b, float) and not isinstance(b, np.datetime64):
            if not 0.0 < b < 1.0:
                raise ValueError(f"Fractional split points must be in (0, 1), got {b}.")
            return round(b * self.n_steps)
        return self.index_of(b)

    def split(
        self, *boundaries: float | TimeLike, embargo: int = 0, history: int = 0
    ) -> list[MarketData]:
        """Split chronologically at the given points.

        Points are fractions of the rows (`0.8`), row positions, or timestamps.
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
        return (
            f"MarketData(T={self.n_steps}, N={self.n_assets}, fields={sorted(self._fields)}"
            f"{extra}, {first} .. {last})"
        )


__all__ = ["MarketData"]
