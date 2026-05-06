from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


class NumpyBackend:
    """Default numpy-based backend implementation."""

    def zeros(self, shape: tuple[int, ...], dtype: Any) -> NDArray[Any]:
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape: tuple[int, ...], dtype: Any) -> NDArray[Any]:
        return np.ones(shape, dtype=dtype)

    def concatenate(self, arrays: Sequence[NDArray[Any]], axis: int = 0) -> NDArray[Any]:
        return np.concatenate(arrays, axis=axis)

    def stack(self, arrays: Sequence[NDArray[Any]], axis: int = 0) -> NDArray[Any]:
        return np.stack(arrays, axis=axis)

    def clip(self, x: NDArray[Any], lo: NDArray[Any], hi: NDArray[Any]) -> NDArray[Any]:
        return np.clip(x, lo, hi)

    def astype(self, x: NDArray[Any], dtype: Any) -> NDArray[Any]:
        return x.astype(dtype)

    def random_uniform(
        self,
        rng: np.random.Generator,
        low: NDArray[Any],
        high: NDArray[Any],
        shape: tuple[int, ...],
    ) -> NDArray[Any]:
        return rng.uniform(low, high, size=shape)

    def random_integers(
        self,
        rng: np.random.Generator,
        low: int,
        high: int,
        size: tuple[int, ...],
    ) -> NDArray[Any]:
        return rng.integers(low, high, size=size)

    def is_array(self, x: Any) -> bool:
        return isinstance(x, np.ndarray)

    def array_equal(self, a: Any, b: Any) -> bool:
        return bool(np.array_equal(a, b))
