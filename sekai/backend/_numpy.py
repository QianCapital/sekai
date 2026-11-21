"""The NumPy backend: float64 arrays, exactly the NumPy calls sekai has always made."""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Iterator, Sequence
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from sekai.backend._base import Array, Backend


class NumpyBackend(Backend):
    """NumPy arrays; floats are float64."""

    name: ClassVar[str] = "numpy"

    def is_array(self, x: Any) -> bool:
        return isinstance(x, np.ndarray)

    def asfloat(self, x: Any, like: Array | None = None) -> Array:
        return np.asarray(x, dtype=np.float64)

    def asbool(self, x: Any, like: Array | None = None) -> Array:
        return np.asarray(x, dtype=bool)

    def asint(self, x: Any, like: Array | None = None) -> Array:
        return np.asarray(x, dtype=np.int64)

    def cast(self, x: Any, like: Array) -> Array:
        return np.asarray(x, dtype=np.asarray(like).dtype)

    def asint8(self, x: Any) -> Array:
        return np.asarray(x, dtype=np.int8)

    def to_numpy(self, x: Any) -> NDArray[Any]:
        return np.asarray(x)

    def zeros(self, shape: Sequence[int], like: Array | None = None) -> Array:
        return np.zeros(tuple(shape))

    def full(self, shape: Sequence[int], value: float, like: Array | None = None) -> Array:
        return np.full(tuple(shape), float(value))

    def bools(self, shape: Sequence[int], value: bool, like: Array | None = None) -> Array:
        return np.full(tuple(shape), bool(value))

    def ints(self, shape: Sequence[int], value: int, like: Array | None = None) -> Array:
        return np.full(tuple(shape), int(value), dtype=np.int64)

    def arange(self, n: int, like: Array | None = None) -> Array:
        return np.arange(n, dtype=np.int64)

    def copy(self, x: Array) -> Array:
        return np.array(x, copy=True)

    def detach(self, x: Array) -> Array:
        return x

    def item(self, x: Array) -> Any:
        return np.asarray(x).item()

    def where(self, condition: Array, x: Any, y: Any) -> Array:
        return np.where(condition, x, y)

    def abs(self, x: Array) -> Array:
        return np.abs(x)

    def sign(self, x: Array) -> Array:
        return np.sign(x)

    def sqrt(self, x: Array) -> Array:
        return np.sqrt(x)

    def exp(self, x: Array) -> Array:
        return np.exp(x)

    def log(self, x: Array) -> Array:
        return np.log(x)

    def log1p(self, x: Array) -> Array:
        return np.log1p(x)

    def tanh(self, x: Array) -> Array:
        return np.tanh(x)

    def sigmoid(self, x: Array) -> Array:
        # Numerically stable in both tails.
        x = np.asarray(x, dtype=np.float64)
        z = np.exp(-np.abs(x))
        return np.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))

    def clip(self, x: Array, lo: Any, hi: Any) -> Array:
        return np.clip(x, lo, hi)

    def maximum(self, x: Any, y: Any) -> Array:
        return np.maximum(x, y)

    def minimum(self, x: Any, y: Any) -> Array:
        return np.minimum(x, y)

    def isfinite(self, x: Array) -> Array:
        return np.isfinite(x)

    def isnan(self, x: Array) -> Array:
        return np.isnan(x)

    def nan_to_num(self, x: Array, nan: float = 0.0) -> Array:
        return np.nan_to_num(x, nan=nan)

    def sum(
        self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array:
        return np.sum(x, axis=axis, keepdims=keepdims)

    def mean(self, x: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        return np.mean(x, axis=axis, keepdims=keepdims)

    def max(self, x: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        return np.max(x, axis=axis, keepdims=keepdims)

    def any(self, x: Array, axis: int | None = None) -> Array:
        return np.any(x, axis=axis)

    def all(self, x: Array, axis: int | None = None) -> Array:
        return np.all(x, axis=axis)

    def cumsum(self, x: Array, axis: int) -> Array:
        return np.cumsum(x, axis=axis)

    def nanmedian(self, x: Array, axis: int = -1) -> Array:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices give NaN
            return np.nanmedian(x, axis=axis)

    def stack(self, xs: Sequence[Array], axis: int = 0) -> Array:
        return np.stack(list(xs), axis=axis)

    def concat(self, xs: Sequence[Array], axis: int = 0) -> Array:
        return np.concatenate(list(xs), axis=axis)

    def broadcast_to(self, x: Array, shape: Sequence[int]) -> Array:
        return np.broadcast_to(x, tuple(shape))

    def moveaxis(self, x: Array, source: int, destination: int) -> Array:
        return np.moveaxis(x, source, destination)

    def take_along_axis(self, x: Array, indices: Array, axis: int) -> Array:
        return np.take_along_axis(x, indices, axis=axis)

    def put_along_axis(self, x: Array, indices: Array, values: Array, axis: int) -> Array:
        out = np.array(x, copy=True)
        np.put_along_axis(out, indices, values, axis=axis)
        return out

    def argsort(self, x: Array, axis: int = -1) -> Array:
        return np.argsort(x, axis=axis, kind="stable")

    def einsum(self, subscripts: str, *operands: Array) -> Array:
        return np.einsum(subscripts, *operands)

    def solve(self, a: Array, b: Array) -> Array:
        return np.linalg.solve(a, b)

    def eye(self, n: int, like: Array | None = None) -> Array:
        return np.eye(n)

    def tiny(self, like: Array | None = None) -> float:
        return float(np.finfo(np.float64).tiny)

    @contextlib.contextmanager
    def errstate(self) -> Iterator[None]:
        with np.errstate(all="ignore"):
            yield


__all__ = ["NumpyBackend"]
