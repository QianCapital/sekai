from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sekai._types import SeedType
from sekai.spaces.space import Space


class MultiBinary(Space[NDArray[np.int8]]):
    """Space of n-dimensional binary vectors with values in {0, 1}."""

    def __init__(
        self,
        n: int | np.integer[Any] | tuple[int, ...],
        seed: SeedType = None,
    ) -> None:
        if isinstance(n, (int, np.integer)):
            self.n: tuple[int, ...] = (int(n),)
        else:
            self.n = tuple(int(d) for d in n)
        super().__init__(shape=self.n, dtype=np.dtype(np.int8), seed=seed)

    def sample(self, mask: ArrayLike | None = None) -> NDArray[np.int8]:
        """Sample uniformly; mask (shape n): 1 forces a 1, 0 forces a 0, -1 samples freely."""
        if mask is not None:
            m = np.asarray(mask)
            if m.shape != self.n:
                raise ValueError(f"mask.shape must be {self.n}, got {m.shape}.")
            sample = np.zeros(self.n, dtype=np.int8)
            free = m == -1
            sample[free] = self.rng.integers(0, 2, size=free.sum(), dtype=np.int8)
            sample[m == 1] = 1
            return sample
        return self.rng.integers(0, 2, size=self.n, dtype=np.int8)

    def contains(self, x: Any) -> bool:
        if isinstance(x, (list, tuple)):
            try:
                x = np.asarray(x)
            except ValueError:  # ragged nesting
                return False
        if not isinstance(x, np.ndarray):
            return False
        return bool(
            x.shape == self.shape
            and (np.issubdtype(x.dtype, np.integer) or x.dtype == np.bool_)
            and np.all((x == 0) | (x == 1))
        )

    @property
    def is_flattenable(self) -> bool:
        return True

    def to_jsonable(self) -> dict[str, Any]:
        return {"type": "MultiBinary", "n": list(self.n)}

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> MultiBinary:
        return cls(n=tuple(data["n"]))

    def __repr__(self) -> str:
        n = self.n[0] if len(self.n) == 1 else self.n
        return f"MultiBinary({n})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MultiBinary):
            return False
        return self.n == other.n
