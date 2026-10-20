from __future__ import annotations

from typing import Any

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

        if seed is not None:
            self.seed(seed)

    def seed(self, seed: SeedType | list[SeedType] | tuple[SeedType, ...] = None) -> int:
        """Seed this space and, deterministically, every sub-space.

        A list of seeds (`[1, None, 3]`) seeds the sub-spaces individually instead, in
        order (`None` leaves one as it is), and returns -1 (there is no single seed).
        More seeds than sub-spaces raise ValueError.
        """
        from sekai.utils.seeding import child_seeds, np_random

        if isinstance(seed, (list, tuple)):
            if len(seed) > len(self.spaces):
                raise ValueError(
                    f"Got {len(seed)} seeds for a Tuple of {len(self.spaces)} sub-spaces."
                )
            for space, s in zip(self.spaces, seed):
                if s is not None:
                    space.seed(s)
            return -1
        self._rng, actual_seed = np_random(seed)
        for space, s in zip(self.spaces, child_seeds(self._rng, actual_seed, len(self))):
            space.seed(s)
        return actual_seed

    def sample(self, mask: tuple[Any, ...] | None = None) -> tuple[Any, ...]:
        """Sample a tuple; mask, if given, has one sub-mask (or None) per sub-space."""
        if mask is None:
            return tuple(space.sample() for space in self.spaces)
        if len(mask) != len(self.spaces):
            raise ValueError(
                f"Tuple mask needs one mask per sub-space ({len(self.spaces)}), got {len(mask)}."
            )
        return tuple(space.sample(m) for space, m in zip(self.spaces, mask))

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

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> Tuple:
        from sekai.spaces.utils import space_from_jsonable

        return cls([space_from_jsonable(sub) for sub in data["spaces"]])

    def __repr__(self) -> str:
        return f"Tuple({', '.join(repr(s) for s in self.spaces)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tuple):
            return False
        return self.spaces == other.spaces
