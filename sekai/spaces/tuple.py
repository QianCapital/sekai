from __future__ import annotations

from typing import Any

import numpy as np

from sekai._types import SeedType
from sekai.spaces.space import Space


class Tuple(Space[tuple[Any, ...]]):
    """Space of ordered tuples of sub-spaces.

    Unlike Dict, Tuple sub-spaces are accessed by index rather than key.
    Useful when the number and type of components are fixed and heterogeneous.
    """

    spaces: tuple[Space[Any], ...]

    def __init__(
        self,
        spaces: tuple[Space[Any], ...] | list[Space[Any]],
        seed: SeedType | list[SeedType] | None = None,
    ) -> None:
        self.spaces = tuple(spaces)
        super().__init__(shape=None, dtype=None)

        if isinstance(seed, list):
            for space, s in zip(self.spaces, seed):
                if s is not None:
                    space.seed(s)
        elif seed is not None:
            from sekai.utils.seeding import make_seed_sequence
            seeds = make_seed_sequence(int(seed), len(self.spaces))
            for space, s in zip(self.spaces, seeds):
                space.seed(s)

    def seed(self, seed: SeedType = None) -> int:
        from sekai.utils.seeding import np_random, make_seed_sequence
        self._rng, actual_seed = np_random(seed)
        if actual_seed != -1:
            seeds = make_seed_sequence(actual_seed, len(self.spaces))
            for space, s in zip(self.spaces, seeds):
                space.seed(s)
        return actual_seed

    def sample(self, mask: tuple[Any, ...] | None = None) -> tuple[Any, ...]:
        if mask is None:
            return tuple(space.sample() for space in self.spaces)
        return tuple(
            space.sample(m) for space, m in zip(self.spaces, mask)
        )

    def contains(self, x: Any) -> bool:
        if not isinstance(x, (tuple, list)):
            return False
        if len(x) != len(self.spaces):
            return False
        return all(space.contains(val) for space, val in zip(self.spaces, x))

    @property
    def is_flattenable(self) -> bool:
        return all(space.is_flattenable for space in self.spaces)

    def __getitem__(self, idx: int) -> Space[Any]:
        return self.spaces[idx]

    def __len__(self) -> int:
        return len(self.spaces)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "type": "Tuple",
            "spaces": [space.to_jsonable() for space in self.spaces],
        }

    def __repr__(self) -> str:
        return f"Tuple({', '.join(repr(s) for s in self.spaces)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tuple):
            return False
        return self.spaces == other.spaces
