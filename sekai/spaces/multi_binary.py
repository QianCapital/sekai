from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from sekai._types import SeedType
from sekai.spaces.space import Space


class MultiBinary(Space[NDArray[np.int8]]):
    """Space of n-dimensional binary vectors with values in {0, 1}."""

    def __init__(
        self,
        n: int | tuple[int, ...],
        seed: SeedType = None,
    ) -> None:
        if isinstance(n, int):
            self.n: tuple[int, ...] = (n,)
        else:
            self.n = tuple(n)
        super().__init__(shape=self.n, dtype=np.dtype(np.int8), seed=seed)

    def sample(self, mask: NDArray[np.int8] | None = None) -> NDArray[np.int8]:
        if mask is not None:
            if mask.shape != self.n:
                raise ValueError(f"mask.shape must be {self.n}, got {mask.shape}.")
            sample = np.zeros(self.n, dtype=np.int8)
            free = mask == -1
            sample[free] = self.rng.integers(0, 2, size=free.sum(), dtype=np.int8)
            sample[mask == 1] = 1
            return sample
        return self.rng.integers(0, 2, size=self.n, dtype=np.int8)

    def contains(self, x: Any) -> bool:
        if not isinstance(x, np.ndarray):
            return False
        return bool(
            x.shape == self.shape
            and x.dtype == np.int8
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
