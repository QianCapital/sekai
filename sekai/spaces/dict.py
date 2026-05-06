from __future__ import annotations

from collections import OrderedDict
from typing import Any, Iterator, Mapping

import numpy as np

from sekai._types import SeedType
from sekai.spaces.space import Space


class Dict(Space[dict[str, Any]]):
    """Space of ordered dicts of sub-spaces.

    Useful for heterogeneous observations, e.g. combining price arrays,
    portfolio state, and order book snapshots into a single observation.
    """

    spaces: OrderedDict[str, Space[Any]]

    def __init__(
        self,
        spaces: Mapping[str, Space[Any]] | None = None,
        seed: SeedType | dict[str, SeedType] | None = None,
        **spaces_kwargs: Space[Any],
    ) -> None:
        if spaces is None:
            spaces = {}
        merged: dict[str, Space[Any]] = {**spaces, **spaces_kwargs}
        self.spaces = OrderedDict(merged)

        super().__init__(shape=None, dtype=None)

        if isinstance(seed, dict):
            for key, s in seed.items():
                if key in self.spaces:
                    self.spaces[key].seed(s)
        elif seed is not None:
            from sekai.utils.seeding import make_seed_sequence
            seeds = make_seed_sequence(int(seed), len(self.spaces))
            for space, s in zip(self.spaces.values(), seeds):
                space.seed(s)

    def seed(self, seed: SeedType = None) -> int:
        from sekai.utils.seeding import np_random, make_seed_sequence
        self._rng, actual_seed = np_random(seed)
        if actual_seed != -1:
            seeds = make_seed_sequence(actual_seed, len(self.spaces))
            for space, s in zip(self.spaces.values(), seeds):
                space.seed(s)
        return actual_seed

    def sample(self, mask: dict[str, Any] | None = None) -> dict[str, Any]:
        if mask is None:
            return {key: space.sample() for key, space in self.spaces.items()}
        return {
            key: space.sample(mask.get(key))
            for key, space in self.spaces.items()
        }

    def contains(self, x: Any) -> bool:
        if not isinstance(x, dict):
            return False
        if set(x.keys()) != set(self.spaces.keys()):
            return False
        return all(space.contains(x[key]) for key, space in self.spaces.items())

    @property
    def is_flattenable(self) -> bool:
        return all(space.is_flattenable for space in self.spaces.values())

    def keys(self) -> Iterator[str]:
        return iter(self.spaces.keys())

    def values(self) -> Iterator[Space[Any]]:
        return iter(self.spaces.values())

    def items(self) -> Iterator[tuple[str, Space[Any]]]:
        return iter(self.spaces.items())

    def __getitem__(self, key: str) -> Space[Any]:
        return self.spaces[key]

    def __setitem__(self, key: str, space: Space[Any]) -> None:
        self.spaces[key] = space

    def __len__(self) -> int:
        return len(self.spaces)

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "type": "Dict",
            "spaces": {key: space.to_jsonable() for key, space in self.spaces.items()},
        }

    def __repr__(self) -> str:
        spaces_repr = ", ".join(f"{k}: {v}" for k, v in self.spaces.items())
        return f"Dict({{{spaces_repr}}})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dict):
            return False
        return self.spaces == other.spaces
