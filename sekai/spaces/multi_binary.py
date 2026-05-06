"""MultiBinary space — binary vectors of fixed length."""

from __future__ import annotations

from typing import Any

import numpy as np

from .space import Space


class MultiBinary(Space[np.ndarray]):
    """A space of binary vectors of shape *n*.

    Parameters
    ----------
    n:
        Number of binary variables (positive integer or tuple of ints).
    seed:
        Optional seed.

    Examples
    --------
    >>> mb = MultiBinary(5)
    >>> mb.sample().shape
    (5,)
    """

    def __init__(
        self,
        n: int | tuple[int, ...],
        seed: int | np.random.Generator | None = None,
    ) -> None:
        if isinstance(n, int):
            shape: tuple[int, ...] = (n,)
        else:
            shape = tuple(n)
        if any(s <= 0 for s in shape):
            raise ValueError(f"All dimensions must be positive, got {shape}")
        super().__init__(shape=shape, dtype=np.int8, seed=seed)

    def sample(self) -> np.ndarray:
        return self._rng.integers(2, size=self.shape, dtype=np.int8)

    def contains(self, x: Any) -> bool:
        if not isinstance(x, np.ndarray):
            try:
                x = np.asarray(x)
            except (TypeError, ValueError):
                return False
        return bool(x.shape == self.shape and np.all((x == 0) | (x == 1)))

    def __repr__(self) -> str:
        return f"MultiBinary({self.shape if len(self.shape) > 1 else self.shape[0]})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MultiBinary) and self.shape == other.shape
