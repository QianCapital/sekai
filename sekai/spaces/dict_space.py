"""Dict space — named product of heterogeneous spaces."""

from __future__ import annotations

from typing import Any

import numpy as np

from .space import Space


class Dict(Space[dict[str, Any]]):
    """A dictionary of named sub-spaces.

    Parameters
    ----------
    spaces:
        A mapping of names to :class:`Space` objects.  An ordered
        ``dict`` is recommended to ensure deterministic ordering.
    seed:
        Optional seed.

    Examples
    --------
    >>> from sekai.spaces import Discrete, Box
    >>> d = Dict({"obs": Box(0, 1, shape=(4,)), "action_mask": Discrete(2)})
    >>> sample = d.sample()
    >>> list(sample.keys())
    ['obs', 'action_mask']
    """

    def __init__(
        self,
        spaces: dict[str, Space[Any]],
        seed: int | dict[str, int] | None = None,
    ) -> None:
        self.spaces: dict[str, Space[Any]] = dict(spaces)
        if seed is not None:
            if isinstance(seed, int):
                ss = np.random.SeedSequence(seed)
                child_seeds = ss.generate_state(len(self.spaces))
                for space, s in zip(self.spaces.values(), child_seeds, strict=True):
                    space.seed(int(s))
            else:
                for key, s in seed.items():
                    self.spaces[key].seed(s)
        super().__init__(shape=None, dtype=None, seed=None)

    def sample(self) -> dict[str, Any]:
        return {key: space.sample() for key, space in self.spaces.items()}

    def contains(self, x: Any) -> bool:
        if not isinstance(x, dict) or set(x.keys()) != set(self.spaces.keys()):
            return False
        return all(self.spaces[key].contains(x[key]) for key in self.spaces)

    def seed(self, seed: int | None = None) -> list[int]:  # type: ignore[override]
        seeds_out: list[int] = []
        ss = np.random.SeedSequence(seed)
        for space, child_seed in zip(
            self.spaces.values(), ss.generate_state(len(self.spaces)), strict=True
        ):
            seeds_out.extend(space.seed(int(child_seed)))
        return seeds_out

    def __repr__(self) -> str:
        inner = ", ".join(f"{k!r}: {v!r}" for k, v in self.spaces.items())
        return f"Dict({{{inner}}})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Dict) and self.spaces == other.spaces
