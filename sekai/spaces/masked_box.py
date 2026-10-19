from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, DTypeLike, NDArray

from sekai._types import SeedType
from sekai.spaces.box import Box
from sekai.spaces.dict import Dict
from sekai.spaces.multi_binary import MultiBinary


class MaskedBox(Dict):
    """A variable-size set of up to `max_size` Box elements, padded to a fixed shape.

    Samples are dicts with two arrays:

    - `"values"`: shape `(max_size, *shape)`, one row per slot;
    - `"mask"`: int8 of shape `(max_size,)`, 1 where the slot holds an element.

    Empty slots hold `fill_value` (by default zero, clipped into the bounds). This
    is the standard way to give a policy an observation whose size changes, such
    as the assets in a trading universe that gains and loses listings: shapes stay
    fixed, so samples batch across vector sub-envs, flatten, and convert to
    gymnasium, while the mask tells a set encoder, transformer or masked mean which
    rows are real.

    MaskedBox is a `Dict` subclass, so every Dict utility (flatten, batch_space,
    concatenate, FilterObservation, gymnasium conversion) applies unchanged.

    Example:
        ```python
        space = sekai.MaskedBox(-np.inf, np.inf, shape=(3,), max_size=100)
        obs = space.pack(features, slots=[0, 4, 7])  # 3 assets in slots 0, 4 and 7
        values, slots = space.unpack(obs)
        ```

    Args:
        low: Lower bound of each element (scalar or array of `shape`).
        high: Upper bound of each element.
        shape: Shape of one element; inferred from low/high when omitted.
        max_size: Number of slots.
        dtype: Element dtype.
        fill_value: Value of empty slots; must lie within the bounds.
        seed: Seed for sampling.
    """

    def __init__(
        self,
        low: ArrayLike | float,
        high: ArrayLike | float,
        shape: tuple[int, ...] | None = None,
        *,
        max_size: int,
        dtype: DTypeLike = np.float32,
        fill_value: ArrayLike | None = None,
        seed: SeedType = None,
    ) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}.")
        element = Box(low, high, shape=shape, dtype=dtype)
        if fill_value is None:
            fill = np.clip(np.zeros(element.shape, dtype=element.dtype), element.low, element.high)
        else:
            fill = np.broadcast_to(np.asarray(fill_value, dtype=element.dtype), element.shape)
            if not element.contains(fill):
                raise ValueError("fill_value must lie within [low, high].")
        values = Box(
            low=np.broadcast_to(element.low, (max_size, *element.shape)),
            high=np.broadcast_to(element.high, (max_size, *element.shape)),
            dtype=element.dtype,
        )
        self.element_space = element
        self.max_size = int(max_size)
        self.fill_value: NDArray[Any] = np.array(fill, dtype=element.dtype)
        super().__init__({"values": values, "mask": MultiBinary(max_size)}, seed=seed)

    @property
    def element_shape(self) -> tuple[int, ...]:
        """Shape of one element."""
        return self.element_space.shape

    @property
    def value_dtype(self) -> np.dtype[Any]:
        """dtype of the values array."""
        return self.element_space.dtype

    # ------------------------------------------------------------------
    # Building and reading samples
    # ------------------------------------------------------------------

    def empty(self) -> dict[str, NDArray[Any]]:
        """A sample with every slot empty."""
        values = np.empty((self.max_size, *self.element_shape), dtype=self.value_dtype)
        values[...] = self.fill_value
        return {"values": values, "mask": np.zeros(self.max_size, dtype=np.int8)}

    def pack(self, values: ArrayLike, slots: ArrayLike | None = None) -> dict[str, NDArray[Any]]:
        """Place `k` elements into slots (default: the first `k`) and pad the rest.

        Args:
            values: Array of shape `(k, *shape)`.
            slots: `k` distinct slot indices in `[0, max_size)`.
        """
        rows: NDArray[Any] = np.asarray(values, dtype=self.value_dtype)
        rows = rows.reshape(-1, *self.element_shape)
        k = len(rows)
        idx = np.arange(k) if slots is None else np.asarray(slots, dtype=np.intp).reshape(-1)
        if idx.shape[0] != k:
            raise ValueError(f"Got {k} values but {idx.shape[0]} slots.")
        if k > self.max_size:
            raise ValueError(f"Got {k} values but the space has only {self.max_size} slots.")
        if k and (idx.min() < 0 or idx.max() >= self.max_size or np.unique(idx).size != k):
            raise ValueError(f"slots must be distinct indices in [0, {self.max_size}).")
        out = self.empty()
        out["values"][idx] = rows
        out["mask"][idx] = 1
        return out

    def unpack(self, x: dict[str, Any]) -> tuple[NDArray[Any], NDArray[np.intp]]:
        """The occupied rows and their slot indices: `(values[slots], slots)`."""
        slots = np.flatnonzero(np.asarray(x["mask"]))
        return np.asarray(x["values"])[slots], slots

    # ------------------------------------------------------------------
    # Space API
    # ------------------------------------------------------------------

    def sample(self, mask: Any = None) -> dict[str, NDArray[Any]]:
        """Sample a padded set.

        mask: optional int8/bool array of shape `(max_size,)` fixing which slots are
        occupied; by default each slot is occupied with probability 1/2.
        """
        if mask is None:
            occupied = self.rng.integers(0, 2, size=self.max_size, dtype=np.int8)
        else:
            occupied = np.asarray(mask).astype(np.int8)
            if occupied.shape != (self.max_size,):
                raise ValueError(f"mask.shape must be ({self.max_size},), got {occupied.shape}.")
        values = self.spaces["values"].sample()
        values[occupied == 0] = self.fill_value
        return {"values": values, "mask": occupied}

    def contains(self, x: Any) -> bool:
        if not isinstance(x, dict) or set(x.keys()) != {"values", "mask"}:
            return False
        if not self.spaces["mask"].contains(x["mask"]):
            return False
        values = np.asarray(x["values"])
        if values.shape != (self.max_size, *self.element_shape) or not np.can_cast(
            values.dtype, self.value_dtype, casting="same_kind"
        ):
            return False
        occupied = np.asarray(x["mask"]).astype(bool)
        rows = values[occupied]
        in_bounds = bool(
            np.all(rows >= self.element_space.low) and np.all(rows <= self.element_space.high)
        )
        return in_bounds and bool(np.all(values[~occupied] == self.fill_value))

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "type": "MaskedBox",
            "low": self.element_space.low.tolist(),
            "high": self.element_space.high.tolist(),
            "shape": list(self.element_shape),
            "max_size": self.max_size,
            "dtype": str(self.value_dtype),
            "fill_value": self.fill_value.tolist(),
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> MaskedBox:
        return cls(
            low=data["low"],
            high=data["high"],
            shape=tuple(data["shape"]),
            max_size=int(data["max_size"]),
            dtype=np.dtype(data["dtype"]),
            fill_value=data.get("fill_value"),
        )

    def __repr__(self) -> str:
        box = self.element_space
        low, high = box._format_bound(box.low), box._format_bound(box.high)
        fill = ""
        if np.any(self.fill_value != 0):
            fill = f", fill_value={box._format_bound(self.fill_value)}"
        return f"MaskedBox({low}, {high}, {box.shape}, max_size={self.max_size}, {box.dtype}{fill})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, MaskedBox)
            and self.element_space == other.element_space
            and self.max_size == other.max_size
            and bool(np.array_equal(self.fill_value, other.fill_value))
        )
