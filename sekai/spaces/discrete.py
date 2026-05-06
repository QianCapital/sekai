"""Discrete action/observation space."""

from __future__ import annotations

from typing import Any, SupportsInt

import numpy as np

from .space import Space


class Discrete(Space[np.intp]):
    """A discrete space with *n* elements: ``{start, start+1, ..., start+n-1}``.

    Parameters
    ----------
    n:
        Number of elements (must be positive).
    start:
        First element of the space (default 0).
    seed:
        Optional seed for the random number generator.

    Examples
    --------
    >>> space = Discrete(4)
    >>> space.sample() in space
    True
    >>> Discrete(3, start=1).contains(2)
    True
    """

    def __init__(
        self,
        n: SupportsInt,
        *,
        start: SupportsInt = 0,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        self.n: np.intp = np.intp(n)
        if self.n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        self.start: np.intp = np.intp(start)
        super().__init__(shape=(), dtype=np.int64, seed=seed)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def sample(self) -> np.intp:
        return self.start + self._rng.integers(self.n, dtype=np.intp)

    def contains(self, x: Any) -> bool:
        if isinstance(x, (int, np.integer)):
            return bool(self.start <= x < self.start + self.n)
        return False

    def __repr__(self) -> str:
        if self.start == 0:
            return f"Discrete({self.n})"
        return f"Discrete({self.n}, start={self.start})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Discrete)
            and self.n == other.n
            and self.start == other.start
        )
