"""The NumPy backend: float64 arrays, exactly the NumPy calls sekai has always made.

Single environments call these methods many times a step on small arrays, where a
Python call costs as much as the arithmetic. So the methods that only forward to a
NumPy function are that function, and for an `ndarray` the reductions and `clip`
call the ufunc NumPy's own wrappers (`np.sum`, `np.clip`, ...) end up calling: the
same results, without the wrappers.
"""

from __future__ import annotations

import contextlib
import warnings
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from sekai.backend._base import Array, Backend

_ndarray = np.ndarray
# NumPy, untyped: its stubs do not match the `Backend` signatures the aliases below fill.
_np: Any = np


def _legacy_dtype(like: Any, dtype: Any) -> Any:
    """The dtype of a `zeros` call: `dtype`, or a dtype given where `like` goes (as sekai
    0.2's backend protocol took it, `zeros(shape, dtype)`), else None (float64)."""
    if dtype is not None:
        return dtype
    return like if isinstance(like, type | np.dtype | str) else None


class NumpyBackend(Backend):
    """NumPy arrays; floats are float64.

    It also keeps the methods of sekai 0.2's (deprecated) `BackendOps` protocol, so code
    written against it still works: `ones`, `concatenate`, `astype`, `random_uniform`,
    `random_integers`, `array_equal`, and `zeros(shape, dtype)`.
    """

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

    def zeros(self, shape: Sequence[int], like: Array | None = None, *, dtype: Any = None) -> Array:
        dt = _legacy_dtype(like, dtype)
        return np.zeros(shape, dtype=np.float64 if dt is None else dt)

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

    # The NumPy functions themselves, with no wrapper call in between.
    where = staticmethod(_np.where)
    abs = staticmethod(_np.abs)
    sign = staticmethod(_np.sign)
    sqrt = staticmethod(_np.sqrt)
    exp = staticmethod(_np.exp)
    log = staticmethod(_np.log)
    log1p = staticmethod(_np.log1p)
    tanh = staticmethod(_np.tanh)
    maximum = staticmethod(_np.maximum)
    minimum = staticmethod(_np.minimum)
    isfinite = staticmethod(_np.isfinite)
    isnan = staticmethod(_np.isnan)
    cumsum = staticmethod(_np.cumsum)
    moveaxis = staticmethod(_np.moveaxis)
    take_along_axis = staticmethod(_np.take_along_axis)
    einsum = staticmethod(_np.einsum)
    solve = staticmethod(_np.linalg.solve)

    def sigmoid(self, x: Array) -> Array:
        # Numerically stable in both tails.
        x = np.asarray(x, dtype=np.float64)
        z = np.exp(-np.abs(x))
        return np.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))

    def clip(self, x: Array, lo: Any, hi: Any) -> Array:
        if type(x) is _ndarray:
            return x.clip(lo, hi)
        return np.clip(x, lo, hi)

    def nan_to_num(self, x: Array, nan: float = 0.0) -> Array:
        return np.nan_to_num(x, nan=nan)

    def sum(
        self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array:
        if type(x) is _ndarray:
            return np.add.reduce(x, axis=axis, keepdims=keepdims)
        return np.sum(x, axis=axis, keepdims=keepdims)

    def mean(self, x: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        if type(x) is _ndarray:
            return x.mean(axis=axis, keepdims=keepdims)
        return np.mean(x, axis=axis, keepdims=keepdims)

    def max(self, x: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        if type(x) is _ndarray:
            return np.maximum.reduce(x, axis=axis, keepdims=keepdims)
        return np.max(x, axis=axis, keepdims=keepdims)

    def any(self, x: Array, axis: int | None = None) -> Array:
        if type(x) is _ndarray:
            return x.any(axis=axis)
        return np.any(x, axis=axis)

    def all(self, x: Array, axis: int | None = None) -> Array:
        if type(x) is _ndarray:
            return x.all(axis=axis)
        return np.all(x, axis=axis)

    def nanmedian(self, x: Array, axis: int = -1) -> Array:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices give NaN
            return np.nanmedian(x, axis=axis)

    def stack(self, arrays: Sequence[Array], axis: int = 0) -> Array:
        return np.stack(list(arrays), axis=axis)

    def concat(self, arrays: Sequence[Array], axis: int = 0) -> Array:
        return np.concatenate(list(arrays), axis=axis)

    def broadcast_to(self, x: Array, shape: Sequence[int]) -> Array:
        return np.broadcast_to(x, tuple(shape))

    def put_along_axis(self, x: Array, indices: Array, values: Array, axis: int) -> Array:
        out = np.array(x, copy=True)
        np.put_along_axis(out, indices, values, axis=axis)
        return out

    def argsort(self, x: Array, axis: int = -1) -> Array:
        return np.argsort(x, axis=axis, kind="stable")

    def eye(self, n: int, like: Array | None = None) -> Array:
        return np.eye(n)

    def tiny(self, like: Array | None = None) -> float:
        return float(np.finfo(np.float64).tiny)

    def errstate(self) -> contextlib.AbstractContextManager[Any]:
        return np.errstate(all="ignore")

    # sekai 0.2's BackendOps methods ------------------------------------------

    def ones(self, shape: Sequence[int], dtype: Any = np.float64) -> Array:
        return np.ones(shape, dtype=dtype)

    def concatenate(self, arrays: Sequence[Array], axis: int = 0) -> Array:
        return np.concatenate(arrays, axis=axis)

    def astype(self, x: Any, dtype: Any) -> Array:
        return (x if isinstance(x, np.ndarray) else np.asarray(x)).astype(dtype)

    def random_uniform(
        self, rng: np.random.Generator, low: Any, high: Any, shape: Sequence[int]
    ) -> Array:
        return rng.uniform(low, high, size=shape)

    def random_integers(
        self, rng: np.random.Generator, low: int, high: int, size: Sequence[int]
    ) -> Array:
        return rng.integers(low, high, size=size)

    def array_equal(self, a: Any, b: Any) -> bool:
        return bool(np.array_equal(a, b))


__all__ = ["NumpyBackend"]
