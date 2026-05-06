from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sekai._types import SeedType
from sekai.spaces.space import Space


class MultiDiscrete(Space[NDArray[np.intp]]):
    """Space for multi-dimensional discrete actions.

    Each dimension i takes values in {start[i], ..., start[i] + nvec[i] - 1}.
    """

    def __init__(
        self,
        nvec: ArrayLike,
        dtype: type = np.int64,
        seed: SeedType = None,
        start: ArrayLike | None = None,
    ) -> None:
        self.nvec = np.asarray(nvec, dtype=np.int64)
        if np.any(self.nvec <= 0):
            raise ValueError("All entries in nvec must be positive.")
        self.start = (
            np.zeros_like(self.nvec) if start is None else np.asarray(start, dtype=np.int64)
        )
        if self.start.shape != self.nvec.shape:
            raise ValueError("start.shape must match nvec.shape.")
        super().__init__(shape=self.nvec.shape, dtype=np.dtype(dtype), seed=seed)

    def sample(self, mask: NDArray[np.int8] | None = None) -> NDArray[np.intp]:
        if mask is not None:
            result = np.zeros(self.nvec.shape, dtype=self.dtype)
            for idx in np.ndindex(self.nvec.shape):
                m = mask[idx]
                valid = np.where(m == 1)[0]
                if len(valid) == 0:
                    raise ValueError(f"All actions masked in dimension {idx}.")
                result[idx] = self.rng.choice(valid) + self.start[idx]
            return result.astype(np.intp)

        return (
            self.rng.integers(self.start, self.start + self.nvec, size=self.nvec.shape)
            .astype(np.intp)
        )

    def contains(self, x: Any) -> bool:
        if not isinstance(x, np.ndarray):
            return False
        return bool(
            x.shape == self.shape
            and np.issubdtype(x.dtype, np.integer)
            and np.all(x >= self.start)
            and np.all(x < self.start + self.nvec)
        )

    @property
    def is_flattenable(self) -> bool:
        return True

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "type": "MultiDiscrete",
            "nvec": self.nvec.tolist(),
            "start": self.start.tolist(),
            "dtype": str(self.dtype),
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> MultiDiscrete:
        return cls(
            nvec=data["nvec"],
            start=data.get("start"),
            dtype=np.dtype(data.get("dtype", np.int64)),
        )

    def __repr__(self) -> str:
        return f"MultiDiscrete({self.nvec.tolist()})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MultiDiscrete):
            return False
        return bool(
            np.array_equal(self.nvec, other.nvec) and np.array_equal(self.start, other.start)
        )
