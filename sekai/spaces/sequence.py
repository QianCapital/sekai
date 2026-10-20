from __future__ import annotations

from typing import Any

import numpy as np

from sekai._types import SeedType
from sekai.spaces.space import Space


class Sequence(Space[Any]):
    """Variable-length sequences of elements from a feature space.

    For observations whose size changes from step to step: the trades printed
    since the last decision, the news items in a window, the orders resting in
    a book. With `stack=False` (the default) a sample is a tuple of elements;
    with `stack=True` the elements are stacked into one batched sample (an
    array of shape `(length, *feature_shape)` for a Box, a dict of arrays for
    a Dict, and so on), which is what most models consume.

    Sequences cannot be flattened or stacked across vector sub-envs into one
    array; convert them to a fixed-size padded form with
    `sekai.PadSequenceObservation` (which yields a `MaskedBox`) when a policy
    needs fixed shapes.

    Args:
        feature_space: The space of each element.
        max_length: Optional upper bound on the length. Samples are uniform in
            `[0, max_length]`; without a bound the length is geometric (mean 3).
        stack: Represent samples as one stacked batch instead of a tuple.
        seed: Seed for sampling.
    """

    def __init__(
        self,
        feature_space: Space[Any],
        *,
        max_length: int | None = None,
        stack: bool = False,
        seed: SeedType = None,
    ) -> None:
        if not isinstance(feature_space, Space):
            raise TypeError(f"feature_space must be a sekai.Space, got {feature_space!r}.")
        if max_length is not None and max_length < 0:
            raise ValueError(f"max_length must be >= 0, got {max_length}.")
        if stack:
            _empty_batch(feature_space)  # raises TypeError for spaces that cannot be stacked
            from sekai.spaces.utils import _has_data

            if not _has_data(feature_space):
                raise TypeError(
                    f"Sequence(stack=True) cannot stack elements of {feature_space!r}: it has no "
                    "sub-space holding data, so a stacked sample could not record its length."
                )
        self.feature_space = feature_space
        self.max_length = None if max_length is None else int(max_length)
        self.stack = stack
        super().__init__(shape=None, dtype=None, seed=seed)

    def seed(self, seed: SeedType = None) -> int:
        """Seed this space and, deterministically, its feature space."""
        from sekai.utils.seeding import child_seeds, np_random

        self._rng, actual_seed = np_random(seed)
        self.feature_space.seed(child_seeds(self._rng, actual_seed, 1)[0])
        return actual_seed

    def sample(self, mask: tuple[int | None, Any] | None = None) -> Any:
        """Sample a sequence.

        mask: optional `(length, feature_mask)`; a given length fixes the
        sequence length, and feature_mask is passed to every element's sample().
        """
        length_mask, feature_mask = (None, None) if mask is None else mask
        if length_mask is None:
            if self.max_length is not None:
                length = int(self.rng.integers(0, self.max_length + 1))
            else:
                length = int(self.rng.geometric(0.25)) - 1
        else:
            length = int(length_mask)
            if length < 0 or (self.max_length is not None and length > self.max_length):
                raise ValueError(f"length {length} is outside [0, {self.max_length}].")
        items = [self.feature_space.sample(feature_mask) for _ in range(length)]
        return self.pack(items)

    def pack(self, items: list[Any] | tuple[Any, ...]) -> Any:
        """Build a sample of this space from a list of elements."""
        if not self.stack:
            return tuple(items)
        if not items:
            return _empty_batch(self.feature_space)
        from sekai.spaces.utils import concatenate

        return concatenate(self.feature_space, items)

    def elements(self, x: Any) -> list[Any]:
        """The elements of a sample, as a list (inverse of pack())."""
        if not self.stack:
            return list(x)
        from sekai.spaces.utils import iterate

        return list(iterate(self.feature_space, x))

    def _length_of(self, x: Any) -> int | None:
        if not self.stack:
            return len(x) if isinstance(x, (tuple, list)) else None
        from sekai.spaces.dict import Dict
        from sekai.spaces.tuple import Tuple
        from sekai.spaces.utils import _has_data

        # Follow the first leaf of the structure, skipping empty sub-Dicts/Tuples (whose
        # stacked form, {} or (), has no length); __init__ checks that a leaf exists.
        space, leaf = self.feature_space, x
        while isinstance(space, (Dict, Tuple)):
            if isinstance(space, Dict):
                if not isinstance(leaf, dict):
                    return None
                key = next(k for k, sub in space.items() if _has_data(sub))
                space, leaf = space[key], leaf.get(key)
            else:
                if not isinstance(leaf, tuple):
                    return None
                i = next(i for i, sub in enumerate(space.spaces) if _has_data(sub))
                space, leaf = space[i], leaf[i] if len(leaf) > i else None
        if not isinstance(leaf, np.ndarray) or leaf.ndim == 0:
            return None
        return int(leaf.shape[0])

    def contains(self, x: Any) -> bool:
        if self.stack and isinstance(x, list):
            return False
        length = self._length_of(x)
        if length is None:
            return False
        if self.max_length is not None and length > self.max_length:
            return False
        try:
            items = self.elements(x)
        except Exception:
            return False
        return len(items) == length and all(self.feature_space.contains(item) for item in items)

    @property
    def is_flattenable(self) -> bool:
        return False

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "type": "Sequence",
            "feature_space": self.feature_space.to_jsonable(),
            "max_length": self.max_length,
            "stack": self.stack,
        }

    @classmethod
    def from_jsonable(cls, data: dict[str, Any]) -> Sequence:
        from sekai.spaces.utils import space_from_jsonable

        return cls(
            space_from_jsonable(data["feature_space"]),
            max_length=data.get("max_length"),
            stack=bool(data.get("stack", False)),
        )

    def __repr__(self) -> str:
        extras = ""
        if self.max_length is not None:
            extras += f", max_length={self.max_length}"
        if self.stack:
            extras += ", stack=True"
        return f"Sequence({self.feature_space!r}{extras})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Sequence)
            and self.feature_space == other.feature_space
            and self.max_length == other.max_length
            and self.stack == other.stack
        )


def _empty_batch(space: Space[Any]) -> Any:
    """A stacked batch of zero elements of space (raises TypeError if space cannot stack)."""
    from sekai.spaces.box import Box
    from sekai.spaces.dict import Dict
    from sekai.spaces.discrete import Discrete
    from sekai.spaces.multi_binary import MultiBinary
    from sekai.spaces.multi_discrete import MultiDiscrete
    from sekai.spaces.tuple import Tuple

    if isinstance(space, Dict):
        return {key: _empty_batch(sub) for key, sub in space.items()}
    if isinstance(space, Tuple):
        return tuple(_empty_batch(sub) for sub in space.spaces)
    if isinstance(space, Box):
        return np.zeros((0, *space.shape), dtype=space.dtype)
    if isinstance(space, Discrete):
        return np.zeros((0,), dtype=np.int64)
    if isinstance(space, MultiDiscrete):
        return np.zeros((0, *space.nvec.shape), dtype=space.dtype)
    if isinstance(space, MultiBinary):
        return np.zeros((0, *space.n), dtype=np.int8)
    raise TypeError(f"Sequence(stack=True) cannot stack elements of {type(space).__name__}.")
