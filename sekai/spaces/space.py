"""Abstract base class for all observation and action spaces."""

from __future__ import annotations

import abc
from typing import Any, Generic, TypeVar

import numpy as np
from numpy.random import Generator

T = TypeVar("T")


class Space(abc.ABC, Generic[T]):
    """Abstract base class for spaces.

    A space defines the valid set of values for observations or actions.
    Every space has a shape, a dtype, and a random number generator that can
    be used for sampling.

    Subclasses must implement :meth:`sample`, :meth:`contains`, and
    :meth:`__repr__`.
    """

    def __init__(
        self,
        shape: tuple[int, ...] | None = None,
        dtype: np.dtype | str | type | None = None,
        seed: int | Generator | None = None,
    ) -> None:
        self._shape: tuple[int, ...] = tuple(shape) if shape is not None else ()
        self.dtype: np.dtype = np.dtype(dtype) if dtype is not None else np.dtype(np.float32)
        self._rng: Generator = self._make_rng(seed)

    # ------------------------------------------------------------------
    # Random number generator helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_rng(seed: int | Generator | None) -> Generator:
        if isinstance(seed, Generator):
            return seed
        return np.random.default_rng(seed)

    def seed(self, seed: int | None = None) -> list[int]:
        """Re-seed the space's random number generator.

        Parameters
        ----------
        seed:
            Integer seed or ``None`` for a random seed.

        Returns
        -------
        list[int]
            The seeds used to initialise the RNG (for reproducibility).
        """
        ss = np.random.SeedSequence(seed)
        self._rng = np.random.default_rng(ss)
        return [int(s) for s in ss.generate_state(1)]

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of a single sample from this space."""
        return self._shape

    @abc.abstractmethod
    def sample(self) -> T:
        """Draw a uniform random sample from the space."""

    @abc.abstractmethod
    def contains(self, x: Any) -> bool:
        """Return ``True`` if *x* is a valid element of this space."""

    def __contains__(self, x: Any) -> bool:  # noqa: D105
        return self.contains(x)

    @abc.abstractmethod
    def __repr__(self) -> str: ...

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self._shape == other._shape
