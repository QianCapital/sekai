"""MultiDiscrete space — product of discrete spaces."""

from __future__ import annotations

from typing import Any

import numpy as np

from .space import Space


class MultiDiscrete(Space[np.ndarray]):
    """A space representing the Cartesian product of discrete spaces.

    Each element of *nvec* specifies the number of elements for the
    corresponding dimension.

    Parameters
    ----------
    nvec:
        Array-like of positive integers giving the number of choices for
        each dimension.
    start:
        Starting values for each dimension (default 0 for all).
    seed:
        Optional seed.

    Examples
    --------
    >>> md = MultiDiscrete([3, 4, 2])
    >>> s = md.sample()
    >>> s.shape
    (3,)
    >>> md.contains(s)
    True
    """

    def __init__(
        self,
        nvec: list[int] | np.ndarray,
        *,
        start: list[int] | np.ndarray | None = None,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        self.nvec: np.ndarray = np.asarray(nvec, dtype=np.int64)
        if np.any(self.nvec <= 0):
            raise ValueError("All elements of nvec must be positive")
        self.start: np.ndarray = (
            np.zeros_like(self.nvec) if start is None else np.asarray(start, dtype=np.int64)
        )
        if self.start.shape != self.nvec.shape:
            raise ValueError("start must have the same shape as nvec")
        super().__init__(shape=self.nvec.shape, dtype=np.int64, seed=seed)

    def sample(self) -> np.ndarray:
        return self.start + self._rng.integers(self.nvec, dtype=np.int64)

    def contains(self, x: Any) -> bool:
        if not isinstance(x, np.ndarray):
            try:
                x = np.asarray(x, dtype=np.int64)
            except (TypeError, ValueError):
                return False
        return bool(
            x.shape == self.shape
            and np.all(x >= self.start)
            and np.all(x < self.start + self.nvec)
        )

    def __repr__(self) -> str:
        if np.all(self.start == 0):
            return f"MultiDiscrete({self.nvec.tolist()})"
        return f"MultiDiscrete({self.nvec.tolist()}, start={self.start.tolist()})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, MultiDiscrete)
            and np.array_equal(self.nvec, other.nvec)
            and np.array_equal(self.start, other.start)
        )
