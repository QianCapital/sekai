"""Snapshots of an environment's state, for branching rollouts and resuming training.

`env.snapshot()` copies the state of the environment and of every wrapper around it
(random generators included); `env.restore(snapshot)` puts it back, in place, so
references to the environment and its wrappers stay valid. Stepping a restored env
with the same actions reproduces the same results. Uses: tree search and other
planning from a state, "what if" analysis of a decision, and checkpoints of a training
run (a snapshot pickles whenever the environment's state does: a dashboard renderer
recording frames doesn't).

Data that does not change is shared between the environment and its snapshots rather
than copied: spaces, specs, `MarketData` (memory-mapped data pickles as its location),
market models, `MarketPanel`s, read-only NumPy arrays that nothing else can write
through, and whatever a class lists in `_snapshot_shared` or registers with
`register_immutable`. Attributes a class lists in `_snapshot_exclude` (resources such as
an event loop) are neither copied nor restored.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

_IMMUTABLE: list[type] = []


def register_immutable(cls: type) -> type:
    """Mark a class whose instances never change: snapshots share them, not copy them.

    Usable as a decorator.
    """
    if cls not in _IMMUTABLE:
        _IMMUTABLE.append(cls)
    return cls


def _immutable_types() -> tuple[type, ...]:
    from sekai.finance.data import MarketData
    from sekai.finance.panel import MarketPanel
    from sekai.finance.simulators import MarketModel
    from sekai.registry.spec import EnvSpec
    from sekai.spaces.space import Space

    return (Space, EnvSpec, MarketData, MarketModel, MarketPanel, *_IMMUTABLE)


@dataclass(frozen=True)
class Snapshot:
    """The state of an environment (and its wrappers) at one moment; see `Env.snapshot()`.

    Attributes:
        kinds: The class of each environment in the snapshot: the object snapshotted and
            the envs it wraps, outermost first, then the environments they hold (a vector
            env's sub-envs, the batch env behind an adapter, ...).
        layers: The state of each, in the same order.
    """

    kinds: tuple[type, ...]
    layers: tuple[dict[str, Any], ...]

    @property
    def kind(self) -> type:
        """The class of the outermost object snapshotted."""
        return self.kinds[0]


_LINKS = ("env", "venv")  # where wrappers and vector wrappers keep what they wrap
_TARGETS = "_sekai_snapshot_targets"  # memo key: the environments a restore writes to


class _Ref:
    """Stands, in a snapshot's state, for the snapshot's environment number `index`.

    References between the environments of a snapshot (a wrapper's handle on the base
    env, a vector env's sub-envs, an adapter's batch env) are kept as references: on
    restore each becomes the environment restored in place, so outside handles to any
    of them stay valid.
    """

    __slots__ = ("index",)

    def __init__(self, index: int) -> None:
        self.index = index

    def __deepcopy__(self, memo: dict[Any, Any]) -> Any:
        targets = memo.get(_TARGETS)
        return self if targets is None else targets[self.index]

    def __reduce__(self) -> tuple[Any, ...]:
        return (_Ref, (self.index,))

    def __repr__(self) -> str:
        return f"<environment {self.index} of the snapshot>"


def _inner(obj: Any) -> tuple[str, Any] | None:
    for name in _LINKS:
        inner = obj.__dict__.get(name)
        if hasattr(inner, "__dict__") and hasattr(inner, "step"):
            return name, inner
    return None


def _layers(obj: Any) -> Iterator[Any]:
    """The object, then the env it wraps, and so on."""
    seen: set[int] = set()
    while obj is not None and id(obj) not in seen:
        seen.add(id(obj))
        yield obj
        link = _inner(obj)
        obj = None if link is None else link[1]


def _is_env(value: Any) -> bool:
    return (
        hasattr(type(value), "step")
        and hasattr(value, "__dict__")
        and not isinstance(value, type)
        and not isinstance(value, _immutable_types())
    )


def _environments(obj: Any) -> list[Any]:
    """`obj`, the envs it wraps, then every environment they hold, in a fixed order.

    The wrapper chain comes first (outermost first), then, breadth first, the
    environments found in their attributes: a vector env's sub-envs, the batch env
    behind an adapter, a wrapped gymnasium env.
    """
    order: list[Any] = []
    seen: set[int] = set()

    def add(value: Any) -> None:
        if id(value) not in seen:
            seen.add(id(value))
            order.append(value)

    def find(value: Any, depth: int) -> None:
        if depth > 4:
            return
        if _is_env(value):
            add(value)
        elif isinstance(value, dict):
            for v in value.values():
                find(v, depth + 1)
        elif isinstance(value, (list, tuple)):
            for v in value:
                find(v, depth + 1)

    for layer in _layers(obj):
        add(layer)
    i = 0
    while i < len(order):
        for value in order[i].__dict__.values():
            find(value, 0)
        i += 1
    return order


def _frozen(arr: Any) -> bool:
    """Whether no one can change `arr`: it and every array it views are read-only."""
    while isinstance(arr, np.ndarray):
        if arr.flags.writeable:
            return False
        arr = arr.base
    return True


def _shared_memo(kinds: Sequence[type], states: Sequence[dict[str, Any]]) -> dict[Any, Any]:
    """deepcopy memo mapping every immutable object reachable from the states to itself."""
    types = _immutable_types()
    memo: dict[Any, Any] = {}

    def visit(value: Any, depth: int) -> None:
        if id(value) in memo or depth > 6:
            return
        if isinstance(value, types) or (isinstance(value, np.ndarray) and _frozen(value)):
            memo[id(value)] = value
            return
        if isinstance(value, dict):
            for v in value.values():
                visit(v, depth + 1)
        elif isinstance(value, (list, tuple)):
            for v in value:
                visit(v, depth + 1)
        elif hasattr(value, "__dict__") and not isinstance(value, type):
            for v in vars(value).values():
                visit(v, depth + 1)

    for kind, state in zip(kinds, states):
        for name in getattr(kind, "_snapshot_shared", ()):
            if name in state:
                memo[id(state[name])] = state[name]
        for value in state.values():
            visit(value, 0)
    return memo


def _copy(state: Any, memo: dict[Any, Any]) -> Any:
    try:
        return copy.deepcopy(state, memo)
    except RuntimeError as e:  # torch: a tensor inside an autograd graph
        if "deepcopy" in str(e):
            raise RuntimeError(
                "The environment's state holds tensors that are part of an autograd graph "
                "(a differentiable rollout in progress), which cannot be copied. Snapshot "
                "between rollouts, or call detach_state() first."
            ) from e
        raise


def _state(env: Any) -> dict[str, Any]:
    excluded = getattr(type(env), "_snapshot_exclude", ())
    return {k: v for k, v in env.__dict__.items() if k not in excluded}


def take_snapshot(obj: Any) -> Snapshot:
    """A `Snapshot` of `obj` (an env, a wrapper stack or an in-process vector env).

    Raises:
        NotImplementedError: If `obj` or any environment it holds can't be snapshotted
            (e.g. a `ProcVecEnv`, whose sub-envs live in other processes); nothing is
            copied then, so `obj` is left as it was.
    """
    envs = _environments(obj)
    for env in envs:
        if getattr(env, "_snapshot_unsupported", ""):
            raise NotImplementedError(env._snapshot_unsupported)
    states = [_state(env) for env in envs]
    kinds = [type(env) for env in envs]
    memo = _shared_memo(kinds, states)
    memo.update({id(env): _Ref(i) for i, env in enumerate(envs)})
    return Snapshot(kinds=tuple(kinds), layers=tuple(_copy(states, memo)))


def restore_snapshot(obj: Any, snapshot: Snapshot) -> None:
    """Put `obj` back in the state of `snapshot` (in place; the snapshot can be reused).

    Every environment of the snapshot is restored in place, so references to `obj`, its
    wrappers, its sub-envs and the objects that refer to them stay valid.
    """
    if not isinstance(snapshot, Snapshot):
        raise TypeError(f"Expected a Snapshot, got {type(snapshot).__name__}.")
    envs = _environments(obj)
    kinds = [type(env) for env in envs]
    if tuple(kinds) != snapshot.kinds:
        names = " > ".join(k.__name__ for k in snapshot.kinds)
        raise TypeError(f"The snapshot is of {names}, not {' > '.join(k.__name__ for k in kinds)}.")
    memo = _shared_memo(kinds, list(snapshot.layers))
    memo[_TARGETS] = envs
    states = _copy(list(snapshot.layers), memo)
    for env, state in zip(envs, states):
        _put_back(env.__dict__, state)


def _put_back(current: dict[str, Any], state: dict[str, Any]) -> None:
    """Set the attributes of `state`, NumPy arrays in place where they can be.

    An array restored in place keeps every view of it (an observation buffer handed out
    read-only, an alias held by another attribute) a view of the restored values.
    """
    for name, value in state.items():
        old = current.get(name)
        if isinstance(old, np.ndarray) and isinstance(value, np.ndarray) and old is not value:
            if old.flags.writeable and old.shape == value.shape and old.dtype == value.dtype:
                np.copyto(old, value)
                continue
            if not old.flags.writeable and not _frozen(old):
                continue  # a read-only view of live state, which is restored in place
        current[name] = value


__all__ = ["Snapshot", "register_immutable", "take_snapshot", "restore_snapshot"]
