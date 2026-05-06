from __future__ import annotations

from functools import singledispatch
from typing import Any, Iterator

import numpy as np
from numpy.typing import NDArray

from sekai.spaces.space import Space


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
def flatten_space(space: Space[Any]) -> Any:
    """Return a Box space equivalent to flattening all samples from space."""
    raise NotImplementedError(f"flatten_space not implemented for {type(space).__name__}")


def _register_box() -> None:
    from sekai.spaces.box import Box

    @flatdim.register(Box)
    def _flatdim_box(space: Box) -> int:
        return int(np.prod(space.shape or (1,)))

    @flatten.register(Box)
    def _flatten_box(space: Box, x: NDArray[Any]) -> NDArray[np.float32]:
        return x.flatten().astype(np.float32)

    @unflatten.register(Box)
    def _unflatten_box(space: Box, x: NDArray[Any]) -> NDArray[Any]:
        assert space.dtype is not None
        return x.reshape(space.shape).astype(space.dtype)

    @flatten_space.register(Box)
    def _flatten_space_box(space: Box) -> Box:
        dim = flatdim(space)
        assert space.dtype is not None
        return Box(
            low=space.low.flatten(),
            high=space.high.flatten(),
            dtype=space.dtype,
        )


def _register_discrete() -> None:
    from sekai.spaces.discrete import Discrete

    @flatdim.register(Discrete)
    def _flatdim_discrete(space: Discrete) -> int:
        return space.n

    @flatten.register(Discrete)
    def _flatten_discrete(space: Discrete, x: np.intp) -> NDArray[np.float32]:
        onehot = np.zeros(space.n, dtype=np.float32)
        onehot[int(x) - space.start] = 1.0
        return onehot

    @unflatten.register(Discrete)
    def _unflatten_discrete(space: Discrete, x: NDArray[Any]) -> np.intp:
        return np.intp(np.argmax(x) + space.start)

    @flatten_space.register(Discrete)
    def _flatten_space_discrete(space: Discrete) -> Any:
        from sekai.spaces.box import Box
        return Box(low=0.0, high=1.0, shape=(space.n,), dtype=np.float32)


def _register_multi_discrete() -> None:
    from sekai.spaces.multi_discrete import MultiDiscrete

    @flatdim.register(MultiDiscrete)
    def _flatdim_md(space: MultiDiscrete) -> int:
        return int(space.nvec.sum())

    @flatten.register(MultiDiscrete)
    def _flatten_md(space: MultiDiscrete, x: NDArray[Any]) -> NDArray[np.float32]:
        result = []
        for i, n in enumerate(space.nvec.flat):
            onehot = np.zeros(n, dtype=np.float32)
            val = int(x.flat[i]) - int(space.start.flat[i])
            onehot[val] = 1.0
            result.append(onehot)
        return np.concatenate(result)

    @unflatten.register(MultiDiscrete)
    def _unflatten_md(space: MultiDiscrete, x: NDArray[Any]) -> NDArray[np.intp]:
        result = np.zeros(space.nvec.shape, dtype=np.intp)
        idx = 0
        for i, (n, s) in enumerate(zip(space.nvec.flat, space.start.flat)):
            result.flat[i] = np.argmax(x[idx:idx + n]) + s
            idx += n
        return result

    @flatten_space.register(MultiDiscrete)
    def _flatten_space_md(space: MultiDiscrete) -> Any:
        from sekai.spaces.box import Box
        return Box(low=0.0, high=1.0, shape=(int(space.nvec.sum()),), dtype=np.float32)


def _register_multi_binary() -> None:
    from sekai.spaces.multi_binary import MultiBinary

    @flatdim.register(MultiBinary)
    def _flatdim_mb(space: MultiBinary) -> int:
        return int(np.prod(space.n))

    @flatten.register(MultiBinary)
    def _flatten_mb(space: MultiBinary, x: NDArray[Any]) -> NDArray[np.float32]:
        return x.flatten().astype(np.float32)

    @unflatten.register(MultiBinary)
    def _unflatten_mb(space: MultiBinary, x: NDArray[Any]) -> NDArray[np.int8]:
        return x.reshape(space.n).astype(np.int8)

    @flatten_space.register(MultiBinary)
    def _flatten_space_mb(space: MultiBinary) -> Any:
        from sekai.spaces.box import Box
        return Box(low=0.0, high=1.0, shape=(int(np.prod(space.n)),), dtype=np.float32)


def _register_dict() -> None:
    from sekai.spaces.dict import Dict

    @flatdim.register(Dict)
    def _flatdim_dict(space: Dict) -> int:
        return sum(flatdim(s) for s in space.spaces.values())

    @flatten.register(Dict)
    def _flatten_dict(space: Dict, x: dict[str, Any]) -> NDArray[np.float32]:
        return np.concatenate([flatten(s, x[k]) for k, s in space.spaces.items()])

    @unflatten.register(Dict)
    def _unflatten_dict(space: Dict, x: NDArray[Any]) -> dict[str, Any]:
        result = {}
        idx = 0
        for key, s in space.spaces.items():
            dim = flatdim(s)
            result[key] = unflatten(s, x[idx:idx + dim])
            idx += dim
        return result

    @flatten_space.register(Dict)
    def _flatten_space_dict(space: Dict) -> Any:
        from sekai.spaces.box import Box
        dim = flatdim(space)
        return Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def _register_tuple() -> None:
    from sekai.spaces.tuple import Tuple

    @flatdim.register(Tuple)
    def _flatdim_tuple(space: Tuple) -> int:
        return sum(flatdim(s) for s in space.spaces)

    @flatten.register(Tuple)
    def _flatten_tuple(space: Tuple, x: tuple[Any, ...]) -> NDArray[np.float32]:
        return np.concatenate([flatten(s, v) for s, v in zip(space.spaces, x)])

    @unflatten.register(Tuple)
    def _unflatten_tuple(space: Tuple, x: NDArray[Any]) -> tuple[Any, ...]:
        result = []
        idx = 0
        for s in space.spaces:
            dim = flatdim(s)
            result.append(unflatten(s, x[idx:idx + dim]))
            idx += dim
        return tuple(result)

    @flatten_space.register(Tuple)
    def _flatten_space_tuple(space: Tuple) -> Any:
        from sekai.spaces.box import Box
        dim = flatdim(space)
        return Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


_register_box()
_register_discrete()
_register_multi_discrete()
_register_multi_binary()
_register_dict()
_register_tuple()


def batch_space(space: Space[Any], n: int) -> Any:
    """Create a batched version of space for use in vectorised environments."""
    from sekai.spaces.box import Box
    from sekai.spaces.discrete import Discrete
    from sekai.spaces.multi_discrete import MultiDiscrete
    from sekai.spaces.multi_binary import MultiBinary

    if isinstance(space, Box):
        low = np.stack([space.low] * n)
        high = np.stack([space.high] * n)
        return Box(low=low, high=high, dtype=space.dtype)
    if isinstance(space, Discrete):
        return MultiDiscrete(nvec=np.full(n, space.n, dtype=np.int64), start=np.full(n, space.start, dtype=np.int64))
    raise NotImplementedError(f"batch_space not implemented for {type(space).__name__}")
