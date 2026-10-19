"""Space utilities: flattening, batching, and (de)serialisation.

Every function here is a `functools.singledispatch` generic, so custom spaces
can opt in by registering an implementation:

    @sekai.spaces.flatdim.register(MySpace)
    def _(space: MySpace) -> int: ...
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping, Sequence
from functools import singledispatch
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sekai.spaces.box import Box
from sekai.spaces.dict import Dict
from sekai.spaces.discrete import Discrete
from sekai.spaces.masked_box import MaskedBox
from sekai.spaces.multi_binary import MultiBinary
from sekai.spaces.multi_discrete import MultiDiscrete
from sekai.spaces.sequence import Sequence as SequenceSpace
from sekai.spaces.space import Space
from sekai.spaces.tuple import Tuple

# ---------------------------------------------------------------------------
# Flattening
# ---------------------------------------------------------------------------


@singledispatch
def flatdim(space: Space[Any]) -> int:
    """Return the number of dimensions when this space is flattened."""
    raise NotImplementedError(f"flatdim not implemented for {type(space).__name__}")


@singledispatch
def flatten(space: Space[Any], x: Any) -> NDArray[np.float32]:
    """Flatten a sample from space into a 1D float32 array."""
    raise NotImplementedError(f"flatten not implemented for {type(space).__name__}")


@singledispatch
def unflatten(space: Space[Any], x: NDArray[Any]) -> Any:
    """Unflatten a 1D array back into a valid sample from space."""
    raise NotImplementedError(f"unflatten not implemented for {type(space).__name__}")


@singledispatch
def flatten_space(space: Space[Any]) -> Box:
    """Return the float32 Box that flatten() maps samples of space into."""
    raise NotImplementedError(f"flatten_space not implemented for {type(space).__name__}")


def _onehot_box(n: int) -> Box:
    return Box(low=0.0, high=1.0, shape=(n,), dtype=np.float32)


def _concat_boxes(boxes: Sequence[Box]) -> Box:
    if not boxes:
        return Box(low=np.zeros(0), high=np.zeros(0), dtype=np.float32)
    return Box(
        low=np.concatenate([b.low for b in boxes]),
        high=np.concatenate([b.high for b in boxes]),
        dtype=np.float32,
    )


# Box ----------------------------------------------------------------------


@flatdim.register(Box)
def _flatdim_box(space: Box) -> int:
    return int(np.prod(space.shape or (1,)))


@flatten.register(Box)
def _flatten_box(space: Box, x: NDArray[Any]) -> NDArray[np.float32]:
    return np.asarray(x).flatten().astype(np.float32)


@unflatten.register(Box)
def _unflatten_box(space: Box, x: NDArray[Any]) -> NDArray[Any]:
    assert space.dtype is not None
    return np.asarray(x).reshape(space.shape).astype(space.dtype)


@flatten_space.register(Box)
def _flatten_space_box(space: Box) -> Box:
    return Box(low=space.low.flatten(), high=space.high.flatten(), dtype=np.float32)


# Discrete -----------------------------------------------------------------


@flatdim.register(Discrete)
def _flatdim_discrete(space: Discrete) -> int:
    return space.n


@flatten.register(Discrete)
def _flatten_discrete(space: Discrete, x: Any) -> NDArray[np.float32]:
    onehot = np.zeros(space.n, dtype=np.float32)
    onehot[int(x) - space.start] = 1.0
    return onehot


@unflatten.register(Discrete)
def _unflatten_discrete(space: Discrete, x: NDArray[Any]) -> np.intp:
    out: np.intp = np.intp(int(np.argmax(x)) + space.start)
    return out


@flatten_space.register(Discrete)
def _flatten_space_discrete(space: Discrete) -> Box:
    return _onehot_box(space.n)


# MultiDiscrete ------------------------------------------------------------


@flatdim.register(MultiDiscrete)
def _flatdim_md(space: MultiDiscrete) -> int:
    return int(space.nvec.sum())


@flatten.register(MultiDiscrete)
def _flatten_md(space: MultiDiscrete, x: NDArray[Any]) -> NDArray[np.float32]:
    x = np.asarray(x)
    out = np.zeros(int(space.nvec.sum()), dtype=np.float32)
    offset = 0
    for i, n in enumerate(space.nvec.flat):
        out[offset + int(x.flat[i]) - int(space.start.flat[i])] = 1.0
        offset += int(n)
    return out


@unflatten.register(MultiDiscrete)
def _unflatten_md(space: MultiDiscrete, x: NDArray[Any]) -> NDArray[Any]:
    result = np.zeros(space.nvec.shape, dtype=space.dtype)
    offset = 0
    for i, (n, s) in enumerate(zip(space.nvec.flat, space.start.flat)):
        result.flat[i] = np.argmax(x[offset : offset + n]) + s
        offset += int(n)
    return result


@flatten_space.register(MultiDiscrete)
def _flatten_space_md(space: MultiDiscrete) -> Box:
    return _onehot_box(int(space.nvec.sum()))


# MultiBinary --------------------------------------------------------------


@flatdim.register(MultiBinary)
def _flatdim_mb(space: MultiBinary) -> int:
    return int(np.prod(space.n))


@flatten.register(MultiBinary)
def _flatten_mb(space: MultiBinary, x: NDArray[Any]) -> NDArray[np.float32]:
    return np.asarray(x).flatten().astype(np.float32)


@unflatten.register(MultiBinary)
def _unflatten_mb(space: MultiBinary, x: NDArray[Any]) -> NDArray[np.int8]:
    return np.asarray(x).reshape(space.n).astype(np.int8)


@flatten_space.register(MultiBinary)
def _flatten_space_mb(space: MultiBinary) -> Box:
    return _onehot_box(int(np.prod(space.n)))


# Dict ---------------------------------------------------------------------


@flatdim.register(Dict)
def _flatdim_dict(space: Dict) -> int:
    return sum(flatdim(s) for s in space.spaces.values())


@flatten.register(Dict)
def _flatten_dict(space: Dict, x: Mapping[str, Any]) -> NDArray[np.float32]:
    parts = [flatten(s, x[k]) for k, s in space.spaces.items()]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


@unflatten.register(Dict)
def _unflatten_dict(space: Dict, x: NDArray[Any]) -> dict[str, Any]:
    result = {}
    offset = 0
    for key, s in space.spaces.items():
        dim = flatdim(s)
        result[key] = unflatten(s, x[offset : offset + dim])
        offset += dim
    return result


@flatten_space.register(Dict)
def _flatten_space_dict(space: Dict) -> Box:
    return _concat_boxes([flatten_space(s) for s in space.spaces.values()])


# Tuple --------------------------------------------------------------------


@flatdim.register(Tuple)
def _flatdim_tuple(space: Tuple) -> int:
    return sum(flatdim(s) for s in space.spaces)


@flatten.register(Tuple)
def _flatten_tuple(space: Tuple, x: Sequence[Any]) -> NDArray[np.float32]:
    parts = [flatten(s, v) for s, v in zip(space.spaces, x)]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


@unflatten.register(Tuple)
def _unflatten_tuple(space: Tuple, x: NDArray[Any]) -> tuple[Any, ...]:
    result = []
    offset = 0
    for s in space.spaces:
        dim = flatdim(s)
        result.append(unflatten(s, x[offset : offset + dim]))
        offset += dim
    return tuple(result)


@flatten_space.register(Tuple)
def _flatten_space_tuple(space: Tuple) -> Box:
    return _concat_boxes([flatten_space(s) for s in space.spaces])


# ---------------------------------------------------------------------------
# Batching — used by vectorised environments
# ---------------------------------------------------------------------------


@singledispatch
def batch_space(space: Space[Any], n: int) -> Space[Any]:
    """Return the space of n stacked samples from space (leading axis of size n)."""
    raise NotImplementedError(f"batch_space not implemented for {type(space).__name__}")


@batch_space.register(Box)
def _batch_box(space: Box, n: int) -> Box:
    return Box(low=np.stack([space.low] * n), high=np.stack([space.high] * n), dtype=space.dtype)


@batch_space.register(Discrete)
def _batch_discrete(space: Discrete, n: int) -> MultiDiscrete:
    return MultiDiscrete(
        nvec=np.full(n, space.n, dtype=np.int64),
        start=np.full(n, space.start, dtype=np.int64),
        dtype=space.dtype.type if space.dtype is not None else np.int64,
    )


@batch_space.register(MultiDiscrete)
def _batch_md(space: MultiDiscrete, n: int) -> MultiDiscrete:
    reps = (n,) + (1,) * space.nvec.ndim
    return MultiDiscrete(
        nvec=np.tile(space.nvec, reps),
        start=np.tile(space.start, reps),
        dtype=space.dtype.type if space.dtype is not None else np.int64,
    )


@batch_space.register(MultiBinary)
def _batch_mb(space: MultiBinary, n: int) -> MultiBinary:
    return MultiBinary((n, *space.n))


@batch_space.register(Dict)
def _batch_dict(space: Dict, n: int) -> Dict:
    return Dict({key: batch_space(sub, n) for key, sub in space.spaces.items()})


@batch_space.register(Tuple)
def _batch_tuple(space: Tuple, n: int) -> Tuple:
    return Tuple([batch_space(sub, n) for sub in space.spaces])


@batch_space.register(SequenceSpace)
def _batch_sequence(space: SequenceSpace, n: int) -> Tuple:
    # Sequences are ragged, so a batch is a tuple of per-env samples, one space each,
    # seeded independently (a plain deep copy would repeat the same draws n times).
    from sekai.utils.seeding import make_seed_sequence

    copies: list[Space[Any]] = [copy.deepcopy(space) for _ in range(n)]
    for sub, seed in zip(copies, make_seed_sequence(int(space.rng.integers(2**63)), n)):
        sub.seed(seed)
    return Tuple(copies)


@singledispatch
def concatenate(space: Space[Any], items: Sequence[Any]) -> Any:
    """Stack per-env samples of space into one batched sample.

    Array spaces stack along a new leading axis; Dict and Tuple recurse into
    their sub-spaces. Spaces without a registered rule return a plain list.
    """
    return list(items)


@concatenate.register(Box)
@concatenate.register(Discrete)
@concatenate.register(MultiDiscrete)
@concatenate.register(MultiBinary)
def _concatenate_array(space: Space[Any], items: Sequence[Any]) -> NDArray[Any]:
    return np.stack([np.asarray(item) for item in items])


@concatenate.register(Dict)
def _concatenate_dict(space: Dict, items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {key: concatenate(sub, [item[key] for item in items]) for key, sub in space.items()}


@concatenate.register(Tuple)
def _concatenate_tuple(space: Tuple, items: Sequence[Sequence[Any]]) -> tuple[Any, ...]:
    return tuple(
        concatenate(sub, [item[i] for item in items]) for i, sub in enumerate(space.spaces)
    )


@concatenate.register(SequenceSpace)
def _concatenate_sequence(space: SequenceSpace, items: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(items)


@singledispatch
def iterate(space: Space[Any], batch: Any) -> Iterator[Any]:
    """Split a batched sample of space back into per-env samples (inverse of concatenate).

    A list of per-env samples is also accepted and iterated as-is.
    """
    return iter(batch)


@iterate.register(Dict)
def _iterate_dict(space: Dict, batch: Any) -> Iterator[Any]:
    if not isinstance(batch, Mapping):
        return iter(batch)
    keys = list(space.keys())
    columns = [list(iterate(space[key], batch[key])) for key in keys]
    return (dict(zip(keys, row)) for row in zip(*columns))


@iterate.register(Tuple)
def _iterate_tuple(space: Tuple, batch: Any) -> Iterator[Any]:
    # A tuple is the batched form (one entry per sub-space); a list is per-env samples.
    if not isinstance(batch, tuple):
        return iter(batch)
    columns = [list(iterate(sub, part)) for sub, part in zip(space.spaces, batch)]
    return (tuple(row) for row in zip(*columns))


# ---------------------------------------------------------------------------
# Zero (padding) samples
# ---------------------------------------------------------------------------


@singledispatch
def zero_sample(space: Space[Any]) -> Any:
    """The neutral element of space: zeros clipped into bounds, the lowest discrete value,
    empty sequences and empty MaskedBox slots. Used for padding (frame stacks, delays)."""
    raise TypeError(f"zero_sample is not implemented for {type(space).__name__}.")


@zero_sample.register(Box)
def _zero_box(space: Box) -> NDArray[Any]:
    return space.zero()


@zero_sample.register(Discrete)
def _zero_discrete(space: Discrete) -> np.intp:
    out: np.intp = np.intp(space.start)
    return out


@zero_sample.register(MultiDiscrete)
def _zero_md(space: MultiDiscrete) -> NDArray[Any]:
    return space.start.astype(space.dtype)


@zero_sample.register(MultiBinary)
def _zero_mb(space: MultiBinary) -> NDArray[np.int8]:
    return np.zeros(space.n, dtype=np.int8)


@zero_sample.register(Dict)
def _zero_dict(space: Dict) -> dict[str, Any]:
    return {key: zero_sample(sub) for key, sub in space.items()}


@zero_sample.register(MaskedBox)
def _zero_masked(space: MaskedBox) -> dict[str, NDArray[Any]]:
    return space.empty()


@zero_sample.register(Tuple)
def _zero_tuple(space: Tuple) -> tuple[Any, ...]:
    return tuple(zero_sample(sub) for sub in space.spaces)


@zero_sample.register(SequenceSpace)
def _zero_sequence(space: SequenceSpace) -> Any:
    return space.pack([])


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

_SPACE_TYPES: dict[str, type[Space[Any]]] = {
    "Box": Box,
    "Discrete": Discrete,
    "MultiDiscrete": MultiDiscrete,
    "MultiBinary": MultiBinary,
    "Dict": Dict,
    "Tuple": Tuple,
    "Sequence": SequenceSpace,
    "MaskedBox": MaskedBox,
}


def register_space_type(name: str, cls: type[Space[Any]]) -> None:
    """Make a custom space reconstructable by space_from_jsonable()."""
    _SPACE_TYPES[name] = cls


def space_from_jsonable(data: Mapping[str, Any]) -> Space[Any]:
    """Rebuild a space from the dict produced by its to_jsonable()."""
    type_name = data.get("type")
    if type_name not in _SPACE_TYPES:
        raise ValueError(
            f"Unknown space type {type_name!r}. Known types: {sorted(_SPACE_TYPES)}. "
            "Register custom spaces with sekai.spaces.register_space_type()."
        )
    return _SPACE_TYPES[type_name].from_jsonable(dict(data))
