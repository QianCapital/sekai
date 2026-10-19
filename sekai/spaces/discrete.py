from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from sekai._types import SeedType
from sekai.spaces.space import Space


class Discrete(Space[np.intp]):
    """Discrete space with n values: {start, start+1, ..., start+n-1}.

    Equivalent to gymnasium's Discrete, with an optional start offset.
    """

    def __init__(
        self,
        n: int | np.integer[Any],
        seed: SeedType = None,
        start: int | np.integer[Any] = 0,
    ) -> None:
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}.")
        self.n = int(n)
        self.start = int(start)
        super().__init__(shape=(), dtype=np.dtype(np.int64), seed=seed)

    def sample(self, mask: ArrayLike | None = None) -> np.intp:
        """Sample uniformly; mask (shape (n,), e.g. int8 or a list) marks allowed values with 1."""
        if mask is not None:
            m = np.asarray(mask)
            if m.shape != (self.n,):
                raise ValueError(f"mask.shape must be ({self.n},), got {m.shape}.")
            valid = np.where(m == 1)[0]
            if len(valid) == 0:
                raise ValueError("All actions masked — at least one must be valid.")
            value = int(self.rng.choice(valid)) + self.start
        else:
            value = int(self.rng.integers(self.start, self.start + self.n))
        out: np.intp = np.intp(value)
        return out

    def contains(self, x: Any) -> bool:
        if isinstance(x, np.ndarray) and x.shape == () and np.issubdtype(x.dtype, np.integer):
            x = x.item()
        if isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_)):
            return bool(self.start <= x < self.start + self.n)
        return False

    @property
    def is_flattenable(self) -> bool:
        return True

    def to_jsonable(self) -> dict[str, Any]:
        return {"type": "Discrete", "n": self.n, "start": self.start}

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> Discrete:
        return cls(n=data["n"], start=data.get("start", 0))

    def __repr__(self) -> str:
        if self.start == 0:
            return f"Discrete({self.n})"
        return f"Discrete({self.n}, start={self.start})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Discrete):
            return False
        return self.n == other.n and self.start == other.start
