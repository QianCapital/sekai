"""Stable slot assignment for a changing universe of instruments.

A policy that sees assets as rows of a fixed-size array (a `MaskedBox`) needs
row i to keep meaning the same instrument while that instrument is alive;
otherwise per-row features, normalisation statistics and recurrent state get
silently mixed between instruments. `SlotMap` keeps each instrument in the
slot it was first given until it leaves the universe, gives newcomers the
lowest free slot, and reports which slots changed hands.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray


class SlotMap:
    """Assigns up to `max_slots` instruments to stable slot indices.

    Example:
        ```python
        slots = SlotMap(3)
        slots.update(["AAPL", "MSFT"])          # -> [0, 1]
        slots.update(["MSFT", "NVDA", "AMZN"])  # AAPL left: -> [1, 0, 2]
        ```
    """

    def __init__(self, max_slots: int) -> None:
        if max_slots < 1:
            raise ValueError(f"max_slots must be >= 1, got {max_slots}.")
        self.max_slots = int(max_slots)
        self._slot_of: dict[Hashable, int] = {}
        self._owner: list[Hashable | None] = [None] * self.max_slots
        self.changed: NDArray[np.bool_] = np.zeros(self.max_slots, dtype=bool)

    def reset(self) -> None:
        """Forget every assignment."""
        self._slot_of.clear()
        self._owner = [None] * self.max_slots
        self.changed = np.zeros(self.max_slots, dtype=bool)

    def update(self, present: Iterable[Hashable]) -> NDArray[np.intp]:
        """Set the current universe; return each present instrument's slot, in order.

        Instruments that are no longer present free their slots; new ones take
        the lowest free slots. `changed` marks slots whose owner differs from
        before the call (freed or newly assigned), e.g. to reset per-slot state.

        Raises:
            ValueError: If more instruments are present than there are slots.
        """
        ids = list(present)
        wanted = set(ids)
        if len(wanted) != len(ids):
            raise ValueError("present instruments must be unique.")
        if len(ids) > self.max_slots:
            raise ValueError(
                f"{len(ids)} instruments are present but there are only {self.max_slots} slots."
            )
        before = list(self._owner)
        for asset in [a for a in self._slot_of if a not in wanted]:
            self._owner[self._slot_of.pop(asset)] = None
        free = iter([i for i, owner in enumerate(self._owner) if owner is None])
        for asset in ids:
            if asset not in self._slot_of:
                slot = next(free)
                self._slot_of[asset] = slot
                self._owner[slot] = asset
        self.changed = np.array([a != b for a, b in zip(before, self._owner)], dtype=bool)
        return np.array([self._slot_of[a] for a in ids], dtype=np.intp)

    def slot_of(self, asset: Hashable) -> int | None:
        """The slot an instrument occupies, or None."""
        return self._slot_of.get(asset)

    @property
    def assets(self) -> tuple[Any, ...]:
        """The instrument in each slot (None for empty slots)."""
        return tuple(self._owner)

    @property
    def mask(self) -> NDArray[np.int8]:
        """1 for occupied slots."""
        return np.array([owner is not None for owner in self._owner], dtype=np.int8)

    def __len__(self) -> int:
        return len(self._slot_of)

    def __repr__(self) -> str:
        return f"SlotMap({len(self)}/{self.max_slots} slots used)"


__all__ = ["SlotMap"]
