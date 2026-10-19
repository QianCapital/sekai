from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from sekai._types import SeedType
from sekai.spaces.space import Space


def float_dtype(dtype: Any) -> np.dtype[Any]:
    """dtype itself if it is floating point, else float32 (for wrappers that output fractions)."""
    dt = np.dtype(dtype)
    return dt if np.issubdtype(dt, np.floating) else np.dtype(np.float32)


def _exact_ints(value: Any) -> NDArray[np.object_] | None:
    """value (not an ndarray) as an object array of exact ints, or None if it holds others.

    np.asarray() would turn a list mixing ints below and at or above 2**63 into a lossy
    float64 array.
    """
    try:
        arr: NDArray[np.object_] = np.asarray(value, dtype=object)
    except Exception:
        return None
    if all(isinstance(v, (int, np.integer)) for v in arr.flat):
        return arr
    return None


def _as_bound(value: ArrayLike | float, dtype: np.dtype[Any], name: str) -> NDArray[Any]:
    """A Box bound as an array of dtype.

    NaN is rejected. For integer dtypes, -inf/inf (unbounded) and other values outside the
    dtype's range map to its limits, as in gymnasium, instead of wrapping around on the cast.
    """
    integer = np.issubdtype(dtype, np.integer)
    ints = _exact_ints(value) if integer and not isinstance(value, np.ndarray) else None
    if ints is not None:
        arr: NDArray[Any] = ints
    else:
        arr = np.asarray(value)
        if arr.dtype.kind in "fc" and np.any(np.isnan(arr)):
            raise ValueError(
                f"Box {name} contains NaN; use -np.inf/np.inf for unbounded dimensions."
            )
        if not integer or np.can_cast(arr.dtype, dtype):
            return arr.astype(dtype)
    info = np.iinfo(dtype)
    if arr.dtype.kind == "f":
        # float(info.max) + 1 is the first float past the max (2**63 for int64, where
        # float(info.max) itself already rounds up to 2**63).
        below, above = arr < info.min, arr >= float(info.max) + 1
        clipped = (below | above) & ~np.isinf(arr)
    else:
        exact = arr.astype(object)  # Python ints compare exactly, whatever the input dtype
        below = np.asarray(exact < info.min, dtype=bool)
        above = np.asarray(exact > info.max, dtype=bool)
        clipped = below | above
    if np.any(clipped):
        warnings.warn(
            f"Box {name} has values outside the range of {dtype}; "
            f"they were clipped to [{info.min}, {info.max}].",
            UserWarning,
            stacklevel=3,
        )
    out = np.empty(arr.shape, dtype=dtype)
    inside = ~(below | above)
    out[inside] = arr[inside]
    out[below] = info.min
    out[above] = info.max
    return out


class Box(Space[NDArray[Any]]):
    """Continuous n-dimensional box space.

    Supports bounded, semi-bounded, and unbounded dimensions.
    Unbounded dimensions are sampled from a shifted/scaled normal distribution.

    Bounds must not be NaN. For integer dtypes, infinite bounds (and any others outside
    the dtype's range) become the dtype's limits, e.g. `Box(-np.inf, np.inf, dtype=np.int64)`
    spans `np.iinfo(np.int64)`.
    """

    low: NDArray[Any]
    high: NDArray[Any]

    def __init__(
        self,
        low: ArrayLike | float,
        high: ArrayLike | float,
        shape: tuple[int, ...] | None = None,
        dtype: DTypeLike = np.float32,
        seed: SeedType = None,
    ) -> None:
        dtype_ = np.dtype(dtype)
        low_arr = _as_bound(low, dtype_, "low")
        high_arr = _as_bound(high, dtype_, "high")

        if shape is not None:
            shape = tuple(shape)
            low_arr = np.full(shape, low_arr, dtype=dtype_)
            high_arr = np.full(shape, high_arr, dtype=dtype_)
        else:
            if low_arr.shape != high_arr.shape:
                raise ValueError(
                    f"low.shape {low_arr.shape} != high.shape {high_arr.shape}. "
                    "Provide an explicit shape= if broadcasting is intended."
                )
            shape = low_arr.shape

        if np.any(low_arr > high_arr):
            raise ValueError("low must be <= high for all dimensions.")

        super().__init__(shape=shape, dtype=dtype_, seed=seed)
        self.low = low_arr
        self.high = high_arr

    @property
    def shape(self) -> tuple[int, ...]:
        assert self._shape is not None
        return self._shape

    @property
    def dtype(self) -> np.dtype[Any]:
        assert self._dtype is not None
        return self._dtype

    def zero(self) -> NDArray[Any]:
        """The zero element, clipped into the bounds (the neutral value for padding)."""
        out: NDArray[Any] = np.clip(np.zeros(self.shape, dtype=self.dtype), self.low, self.high)
        return out

    def sample(self, mask: NDArray[Any] | None = None) -> NDArray[Any]:
        """Sample uniformly (bounded), exponentially (half-bounded) or normally (unbounded).

        mask: optional array broadcastable to `shape`; entries where it is 0 are
        not sampled but set to zero clipped into the bounds (`zero()`), e.g. the
        weight of an asset that cannot be traded.
        """
        sample = self._sample()
        if mask is None:
            return sample
        keep = np.broadcast_to(np.asarray(mask).astype(bool), self.shape)
        out: NDArray[Any] = np.where(keep, sample, self.zero()).astype(self.dtype)
        return out

    def _sample(self) -> NDArray[Any]:
        dtype = self.dtype

        sample = np.empty(self.shape, dtype=dtype)

        if np.issubdtype(dtype, np.floating):
            bounded_lo = np.isfinite(self.low)
            bounded_hi = np.isfinite(self.high)

            both = bounded_lo & bounded_hi
            lo_only = bounded_lo & ~bounded_hi
            hi_only = ~bounded_lo & bounded_hi
            neither = ~bounded_lo & ~bounded_hi

            if both.any():
                sample[both] = self.rng.uniform(self.low[both], self.high[both])
            if lo_only.any():
                sample[lo_only] = self.low[lo_only] + self.rng.exponential(size=lo_only.sum())
            if hi_only.any():
                sample[hi_only] = self.high[hi_only] - self.rng.exponential(size=hi_only.sum())
            if neither.any():
                sample[neither] = self.rng.standard_normal(size=neither.sum())

        elif np.issubdtype(dtype, np.integer):
            # endpoint=True instead of high + 1, which overflows at the dtype's max (uint8 255).
            sample = self.rng.integers(
                self.low, self.high, size=self.shape, dtype=dtype, endpoint=True
            )
        else:
            raise TypeError(f"Unsupported dtype for Box sampling: {dtype}")

        return sample.astype(dtype)

    def contains(self, x: Any) -> bool:
        if isinstance(x, np.ndarray):
            castable = np.can_cast(x.dtype, self.dtype, casting="same_kind")
        else:
            # Python ints would arrive as int64, which is not same_kind as e.g. uint8 (or
            # as lossy float64 past 2**63): for an integer Box they are judged by value.
            ints = _exact_ints(x) if np.issubdtype(self.dtype, np.integer) else None
            if ints is not None:
                return self._contains_ints(ints)
            try:
                x = np.asarray(x)
            except Exception:
                return False
            castable = np.can_cast(x.dtype, self.dtype, casting="same_kind")
        return bool(
            x.shape == self.shape and castable and np.all(x >= self.low) and np.all(x <= self.high)
        )

    def _contains_ints(self, ints: NDArray[np.object_]) -> bool:
        """Membership of exact ints, by value. The range check comes before the cast because
        NumPy < 2 compares int64 with uint64 bounds as float64."""
        if ints.shape != self.shape:
            return False
        if ints.size:
            info = np.iinfo(self.dtype)
            if int(ints.min()) < info.min or int(ints.max()) > info.max:
                return False
        x = ints.astype(self.dtype)
        return bool(np.all(x >= self.low) and np.all(x <= self.high))

    @property
    def is_flattenable(self) -> bool:
        return True

    @property
    def is_bounded(self) -> tuple[bool, bool]:
        """Returns (lower_bounded, upper_bounded) for the entire space."""
        return bool(np.all(np.isfinite(self.low))), bool(np.all(np.isfinite(self.high)))

    def to_jsonable(self) -> dict[str, Any]:
        assert self.dtype is not None
        return {
            "type": "Box",
            "low": self.low.tolist(),
            "high": self.high.tolist(),
            "shape": list(self.shape or []),
            "dtype": str(self.dtype),
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> Box:
        return cls(
            low=data["low"],
            high=data["high"],
            shape=tuple(data["shape"]),
            dtype=np.dtype(data["dtype"]),
        )

    def _format_bound(self, bound: NDArray[Any]) -> str:
        """A scalar when every entry is equal, else the flattened values (summarised if long)."""

        def fmt(value: Any) -> str:
            if np.issubdtype(bound.dtype, np.floating):
                return np.format_float_positional(value, trim="0")  # shortest, no float32 noise
            return str(value)

        flat = bound.reshape(-1)
        if flat.size and np.all(flat == flat[0]):
            return fmt(flat[0])
        values = (
            [fmt(v) for v in flat]
            if flat.size <= 8
            else [
                *(fmt(v) for v in flat[:3]),
                "...",
                *(fmt(v) for v in flat[-3:]),
            ]
        )
        return "[" + ", ".join(values) + "]"

    def __repr__(self) -> str:
        low, high = self._format_bound(self.low), self._format_bound(self.high)
        return f"Box({low}, {high}, {self.shape}, {self.dtype})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Box):
            return False
        return (
            self.shape == other.shape
            and self.dtype == other.dtype
            and bool(np.array_equal(self.low, other.low))
            and bool(np.array_equal(self.high, other.high))
        )
