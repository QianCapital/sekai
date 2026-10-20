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

    @property
    def dtype(self) -> np.dtype[Any]:
        assert self._dtype is not None
        return self._dtype

    def sample(self, mask: NDArray[np.int8] | tuple[Any, ...] | None = None) -> NDArray[np.intp]:
        """Sample uniformly, optionally restricted by a mask.

        mask: per entry, an int8 array of length nvec[i] with 1 marking allowed
        values. Give either a tuple of such arrays (nested to match nvec's shape,
        as in gymnasium) or one array whose last axis is padded to max(nvec).
        A mask of any other shape raises ValueError.
        """
        if mask is not None:
            result = np.zeros(self.nvec.shape, dtype=self.dtype)
            for idx, m in zip(np.ndindex(self.nvec.shape), self._mask_entries(mask)):
                valid = np.flatnonzero(m == 1)
                if len(valid) == 0:
                    raise ValueError(f"All actions masked in dimension {idx}.")
                result[idx] = self.rng.choice(valid) + self.start[idx]
            return result

        return self.rng.integers(self.start, self.start + self.nvec, size=self.nvec.shape).astype(
            self.dtype
        )

    def _mask_entries(self, mask: Any) -> list[NDArray[Any]]:
        """The flags of a mask (see sample()) for each entry, in np.ndindex order.

        Each has length nvec[idx]. Raises ValueError for a mask of the wrong structure or
        with an entry of the wrong length (flags would otherwise be silently dropped or
        missing).
        """
        if isinstance(mask, np.ndarray):
            width = int(np.max(self.nvec)) if self.nvec.size else 0
            if (
                mask.ndim != self.nvec.ndim + 1
                or mask.shape[:-1] != self.nvec.shape
                or mask.shape[-1] < width
            ):
                raise ValueError(
                    f"A mask array for {self!r} must have shape {(*self.nvec.shape, width)} "
                    f"(one row per entry, padded to max(nvec)), got {mask.shape}."
                )
            return [mask[idx][: int(self.nvec[idx])] for idx in np.ndindex(self.nvec.shape)]

        entries: list[NDArray[Any]] = []

        def walk(node: Any, idx: tuple[int, ...]) -> None:
            if len(idx) == self.nvec.ndim:
                n = int(self.nvec[idx])
                flags = np.asarray(node)
                if flags.shape != (n,):
                    raise ValueError(
                        f"The mask for dimension {idx} must have shape ({n},) "
                        f"(nvec{list(idx)} = {n}), got {flags.shape}."
                    )
                entries.append(flags)
                return
            size = self.nvec.shape[len(idx)]
            if not hasattr(node, "__len__") or len(node) != size:
                found = len(node) if hasattr(node, "__len__") else type(node).__name__
                raise ValueError(
                    f"A mask for {self!r} needs {size} sub-masks at dimension {idx}, got {found}."
                )
            for i in range(size):
                walk(node[i], (*idx, i))

        walk(mask, ())
        return entries

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
        if np.any(self.start != 0):
            return f"MultiDiscrete({self.nvec.tolist()}, start={self.start.tolist()})"
        return f"MultiDiscrete({self.nvec.tolist()})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MultiDiscrete):
            return False
        return bool(
            np.array_equal(self.nvec, other.nvec)
            and np.array_equal(self.start, other.start)
            and self.dtype == other.dtype
        )
