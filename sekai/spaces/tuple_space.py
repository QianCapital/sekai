"""Tuple space — ordered product of heterogeneous spaces."""

from __future__ import annotations

from typing import Any

import numpy as np

from .space import Space


class Tuple(Space[tuple[Any, ...]]):
    """An ordered product of possibly heterogeneous spaces.

    Parameters
    ----------
    spaces:
        Sequence of :class:`Space` objects.
    seed:
        Optional seed.  If an integer, seeds are derived sequentially for
        each sub-space.

    Examples
    --------
    >>> from sekai.spaces import Discrete, Box
    >>> t = Tuple((Discrete(3), Box(0, 1, shape=(2,))))
    >>> obs = t.sample()
    >>> isinstance(obs, tuple)
    True
    """

    def __init__(
        self,
        spaces: tuple[Space[Any], ...] | list[Space[Any]],
        seed: int | list[int] | None = None,
    ) -> None:
        self.spaces: tuple[Space[Any], ...] = tuple(spaces)
        if seed is not None:
            seeds: list[int | None]
            if isinstance(seed, int):
                seeds = list(
                    np.random.SeedSequence(seed).generate_state(len(self.spaces))
                )
            else:
                seeds = list(seed)
            for space, s in zip(self.spaces, seeds, strict=False):
                space.seed(s)
        # Shape/dtype not meaningful for a composite space
        super().__init__(shape=None, dtype=None, seed=None)

    def sample(self) -> tuple[Any, ...]:
        return tuple(s.sample() for s in self.spaces)

    def contains(self, x: Any) -> bool:
        if not isinstance(x, (tuple, list)) or len(x) != len(self.spaces):
            return False
        return all(space.contains(xi) for space, xi in zip(self.spaces, x, strict=True))

    def seed(self, seed: int | None = None) -> list[int]:  # type: ignore[override]
        seeds_out: list[int] = []
        ss = np.random.SeedSequence(seed)
        for space, child_seed in zip(
            self.spaces, ss.generate_state(len(self.spaces)), strict=True
        ):
            seeds_out.extend(space.seed(int(child_seed)))
        return seeds_out

    def __repr__(self) -> str:
        inner = ", ".join(repr(s) for s in self.spaces)
        return f"Tuple({inner})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Tuple) and self.spaces == other.spaces
