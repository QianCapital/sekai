"""Continuous (box) observation/action space."""

from __future__ import annotations

from typing import Any

import numpy as np

from .space import Space


class Box(Space[np.ndarray]):
    """A (possibly unbounded) box in :math:`\\mathbb{R}^n`.

    Generalises the concept of a Cartesian product of closed intervals.
    Every element of the space is an ``ndarray`` with shape *shape* and
    values clipped to ``[low, high]``.

    Parameters
    ----------
    low:
        Lower bounds (scalar or array-like).
    high:
        Upper bounds (scalar or array-like).
    shape:
        Shape of a single sample.  Inferred from *low*/*high* when they
        are array-like.
    dtype:
        Element type (default ``np.float32``).
    seed:
        Optional seed for the random number generator.

    Examples
    --------
    >>> import numpy as np
    >>> box = Box(low=-1.0, high=1.0, shape=(3,))
    >>> box.sample().shape
    (3,)
    >>> box.contains(np.zeros(3, dtype=np.float32))
    True
    """

    def __init__(
        self,
        low: float | np.ndarray,
        high: float | np.ndarray,
        shape: tuple[int, ...] | None = None,
        dtype: np.dtype | str | type = np.float32,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        dtype = np.dtype(dtype)

        # Resolve shape
        if shape is None:
            if isinstance(low, np.ndarray):
                shape = low.shape
            elif isinstance(high, np.ndarray):
                shape = high.shape
            else:
                shape = ()

        self.low: np.ndarray = np.full(shape, low, dtype=dtype)
        self.high: np.ndarray = np.full(shape, high, dtype=dtype)

        if np.any(self.low > self.high):
            raise ValueError("low must be <= high for all dimensions")

        super().__init__(shape=shape, dtype=dtype, seed=seed)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def sample(self) -> np.ndarray:
        """Sample uniformly within bounds, handling unbounded dimensions."""
        dtype = self.dtype

        sample = np.empty(self.shape, dtype=dtype)

        unbounded = ~np.isfinite(self.low) & ~np.isfinite(self.high)
        upperbounded = ~np.isfinite(self.low) & np.isfinite(self.high)
        lowerbounded = np.isfinite(self.low) & ~np.isfinite(self.high)
        bounded = np.isfinite(self.low) & np.isfinite(self.high)

        # Fully unbounded — standard normal
        sample[unbounded] = self._rng.standard_normal(np.sum(unbounded))

        # Only upper bound — exponential reflected
        if np.any(upperbounded):
            sample[upperbounded] = (
                -self._rng.exponential(size=np.sum(upperbounded))
                + self.high[upperbounded]
            )

        # Only lower bound — exponential
        if np.any(lowerbounded):
            sample[lowerbounded] = (
                self._rng.exponential(size=np.sum(lowerbounded))
                + self.low[lowerbounded]
            )

        # Fully bounded — uniform
        if np.any(bounded):
            sample[bounded] = self._rng.uniform(
                self.low[bounded], self.high[bounded], size=np.sum(bounded)
            )

        return sample.astype(dtype)

    def contains(self, x: Any) -> bool:
        if not isinstance(x, np.ndarray):
            try:
                x = np.asarray(x, dtype=self.dtype)
            except (TypeError, ValueError):
                return False
        return bool(
            x.shape == self.shape
            and x.dtype == self.dtype
            and np.all(x >= self.low)
            and np.all(x <= self.high)
        )

    def __repr__(self) -> str:
        return (
            f"Box({self.low.tolist()}, {self.high.tolist()}, "
            f"shape={self.shape}, dtype={self.dtype})"
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Box)
            and self.shape == other.shape
            and np.array_equal(self.low, other.low)
            and np.array_equal(self.high, other.high)
            and self.dtype == other.dtype
        )
