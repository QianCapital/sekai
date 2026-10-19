from __future__ import annotations

from typing import Any, Literal

import numpy as np

from sekai._types import ActType
from sekai.core.env import Env
from sekai.core.wrapper import ObservationWrapper
from sekai.spaces.box import Box
from sekai.spaces.dict import Dict
from sekai.spaces.masked_box import MaskedBox
from sekai.spaces.sequence import Sequence
from sekai.spaces.space import Space
from sekai.spaces.tuple import Tuple

Truncate = Literal["tail", "head", "error"]


class PadSequenceObservation(ObservationWrapper[Any, ActType, Any]):
    """Turns variable-length Sequence observations into fixed-size MaskedBox ones.

    Every `Sequence` of Box elements in the observation space (at the top level
    or inside a Dict/Tuple) becomes a `MaskedBox` of `max_length` slots, with
    the elements in order in the first slots. The result has fixed shapes, so it
    can be vectorised, flattened, normalised and fed to a standard network.

    Args:
        env: The environment to wrap.
        max_length: Slots per sequence. Defaults to each Sequence's own max_length.
        truncate: What to do with a longer sequence: keep the most recent
            `max_length` elements ("tail", the default, right for event streams),
            keep the first ones ("head"), or raise ValueError ("error").
    """

    def __init__(
        self,
        env: Env[Any, ActType],
        max_length: int | None = None,
        *,
        truncate: Truncate = "tail",
    ) -> None:
        if truncate not in ("tail", "head", "error"):
            raise ValueError(f"truncate must be 'tail', 'head' or 'error', got {truncate!r}.")
        if max_length is not None and max_length < 1:
            raise ValueError(f"max_length must be >= 1, got {max_length}.")
        self.max_length = max_length
        self.truncate: Truncate = truncate
        self._found = False
        new_space = self._convert(env.observation_space)
        if not self._found:
            raise TypeError(f"No Sequence in the observation space {env.observation_space!r}.")
        super().__init__(env)
        self._inner_space = env.observation_space
        self.observation_space = new_space

    def _convert(self, space: Space[Any]) -> Space[Any]:
        if isinstance(space, Sequence):
            feature = space.feature_space
            if not isinstance(feature, Box):
                raise TypeError(
                    "PadSequenceObservation pads sequences of Box elements, got "
                    f"{type(feature).__name__} elements."
                )
            length = self.max_length or space.max_length
            if not length:
                raise ValueError(
                    "PadSequenceObservation needs max_length (the Sequence has no max_length)."
                )
            self._found = True
            return MaskedBox(
                feature.low, feature.high, shape=feature.shape, max_size=length, dtype=feature.dtype
            )
        if isinstance(space, MaskedBox):
            return space
        if isinstance(space, Dict):
            return Dict({key: self._convert(sub) for key, sub in space.items()})
        if isinstance(space, Tuple):
            return Tuple([self._convert(sub) for sub in space.spaces])
        return space

    def _pad(self, inner: Space[Any], outer: Space[Any], x: Any) -> Any:
        if isinstance(inner, Sequence):
            assert isinstance(outer, MaskedBox)
            items = inner.elements(x)
            if len(items) > outer.max_size:
                if self.truncate == "error":
                    raise ValueError(
                        f"Sequence of length {len(items)} exceeds max_length {outer.max_size}."
                    )
                items = (
                    items[-outer.max_size :] if self.truncate == "tail" else items[: outer.max_size]
                )
            rows = (
                np.stack([np.asarray(item) for item in items])
                if items
                else np.zeros((0, *outer.element_shape), dtype=outer.value_dtype)
            )
            return outer.pack(rows)
        if isinstance(inner, MaskedBox):
            return x
        if isinstance(inner, Dict):
            assert isinstance(outer, Dict)
            return {key: self._pad(sub, outer[key], x[key]) for key, sub in inner.items()}
        if isinstance(inner, Tuple):
            assert isinstance(outer, Tuple)
            return tuple(
                self._pad(sub, out, v) for sub, out, v in zip(inner.spaces, outer.spaces, x)
            )
        return x

    def observation(self, obs: Any) -> Any:
        return self._pad(self._inner_space, self.observation_space, obs)
