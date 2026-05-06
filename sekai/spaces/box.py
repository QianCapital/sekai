from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from sekai._types import SeedType
from sekai.spaces.space import Space


class Box(Space[NDArray[Any]]):
    """Continuous n-dimensional box space.

    Supports bounded, semi-bounded, and unbounded dimensions.
    Unbounded dimensions are sampled from a shifted/scaled normal distribution.
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

        if shape is not None:
            shape = tuple(shape)
            low_arr = np.full(shape, low, dtype=dtype_)
            high_arr = np.full(shape, high, dtype=dtype_)
        else:
            low_arr = np.asarray(low, dtype=dtype_)
            high_arr = np.asarray(high, dtype=dtype_)
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

    def sample(self, mask: None = None) -> NDArray[Any]:
        dtype = self.dtype
        assert dtype is not None

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
            sample = self.rng.integers(
                self.low, self.high + 1, size=self.shape, dtype=dtype
            )
        else:
            raise TypeError(f"Unsupported dtype for Box sampling: {dtype}")

        return sample.astype(dtype)

    def contains(self, x: Any) -> bool:
        if not isinstance(x, np.ndarray):
            try:
                x = np.asarray(x)
            except Exception:
                return False
        return bool(
            x.shape == self.shape
            and np.can_cast(x.dtype, self.dtype, casting="same_kind")
            and np.all(x >= self.low)
            and np.all(x <= self.high)
        )

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

    def __repr__(self) -> str:
        dtype = self.dtype
        lo = float(self.low.min()) if self.low.ndim > 0 else float(self.low)
        hi = float(self.high.max()) if self.high.ndim > 0 else float(self.high)
        return f"Box({lo}, {hi}, {self.shape}, {dtype})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Box):
            return False
        return (
            self.shape == other.shape
            and self.dtype == other.dtype
            and bool(np.array_equal(self.low, other.low))
            and bool(np.array_equal(self.high, other.high))
        )
