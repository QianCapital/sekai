from __future__ import annotations

import abc
from typing import Any, Generic

import numpy as np

from sekai._types import SeedType, T


class Space(abc.ABC, Generic[T]):
    """Abstract base class for all observation and action spaces.

    Spaces define the valid range of values for observations and actions.
    Unlike gymnasium, T is invariant so contains() is correctly typed.
    """

    def __init__(
        self,
        shape: tuple[int, ...] | None = None,
        dtype: np.dtype | None = None,
        seed: SeedType = None,
    ) -> None:
        self._shape = shape
        self._dtype = np.dtype(dtype) if dtype is not None else None
        self._rng: np.random.Generator | None = None
        if seed is not None:
            self.seed(seed)

    @property
    def shape(self) -> tuple[int, ...] | None:
        return self._shape

    @property
    def dtype(self) -> np.dtype | None:
        return self._dtype

    @property
    def rng(self) -> np.random.Generator:
        if self._rng is None:
            from sekai.utils.seeding import np_random
            self._rng, _ = np_random()
        return self._rng

    def seed(self, seed: SeedType = None) -> int:
        """Seed the space's RNG. Returns the actual seed used."""
        from sekai.utils.seeding import np_random
        self._rng, actual_seed = np_random(seed)
        return actual_seed

    @abc.abstractmethod
    def sample(self, mask: Any = None) -> T:
        """Sample a random element from this space."""
        ...

    @abc.abstractmethod
    def contains(self, x: Any) -> bool:
        """Return True if x is a valid element of this space."""
        ...

    def __contains__(self, x: object) -> bool:
        return self.contains(x)

    @property
    @abc.abstractmethod
    def is_flattenable(self) -> bool:
        """Whether this space can be flattened to a Box."""
        ...

    def to_jsonable(self) -> dict[str, Any]:
        raise NotImplementedError(f"{type(self).__name__} does not support JSON serialization")

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> Space[Any]:
        raise NotImplementedError(f"{cls.__name__} does not support JSON deserialization")

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other):
            return False
        return True
