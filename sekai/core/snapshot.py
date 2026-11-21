"""Snapshots of an environment's state, for branching rollouts and resuming training.

`env.snapshot()` copies the state of the environment and of every wrapper around it
(random generators included); `env.restore(snapshot)` puts it back, in place, so
references to the environment and its wrappers stay valid. Stepping a restored env
with the same actions reproduces the same results. Uses: tree search and other
planning from a state, "what if" analysis of a decision, and checkpoints of a training
run (a snapshot pickles).

Data that does not change is shared between the environment and its snapshots rather
than copied: spaces, specs, `MarketData` (memory-mapped data pickles as its location),
market models, `MarketPanel`s, read-only NumPy arrays, and whatever a class lists in
`_snapshot_shared` or registers with `register_immutable`.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
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
        kinds: The class of each layer, outermost first (wrappers, then the env).
        layers: The state of each layer, in the same order.
    """

    kinds: tuple[type, ...]
    layers: tuple[dict[str, Any], ...]

    @property
    def kind(self) -> type:
        """The class of the outermost object snapshotted."""
        return self.kinds[0]


_LINKS = ("env", "venv")  # where wrappers and vector wrappers keep what they wrap


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


def _shared_memo(kinds: list[type], states: list[dict[str, Any]]) -> dict[int, Any]:
    """deepcopy memo mapping every immutable object reachable from the states to itself."""
    types = _immutable_types()
    memo: dict[int, Any] = {}

    def visit(value: Any, depth: int) -> None:
        if id(value) in memo or depth > 6:
            return
        if isinstance(value, types) or (
            isinstance(value, np.ndarray) and not value.flags.writeable
        ):
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


def _copy(state: Any, memo: dict[int, Any]) -> Any:
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


def take_snapshot(obj: Any) -> Snapshot:
    """A `Snapshot` of `obj` (an env, a wrapper stack or an in-process vector env)."""
    if getattr(obj, "_snapshot_unsupported", ""):
        raise NotImplementedError(obj._snapshot_unsupported)
    layers = list(_layers(obj))
    states = []
    for layer in layers:
        link = _inner(layer)
        states.append({k: v for k, v in layer.__dict__.items() if link is None or k != link[0]})
    kinds = [type(layer) for layer in layers]
    return Snapshot(kinds=tuple(kinds), layers=tuple(_copy(states, _shared_memo(kinds, states))))


def restore_snapshot(obj: Any, snapshot: Snapshot) -> None:
    """Put `obj` back in the state of `snapshot` (in place; the snapshot can be reused)."""
    if not isinstance(snapshot, Snapshot):
        raise TypeError(f"Expected a Snapshot, got {type(snapshot).__name__}.")
    layers = list(_layers(obj))
    kinds = [type(layer) for layer in layers]
    if tuple(kinds) != snapshot.kinds:
        names = " > ".join(k.__name__ for k in snapshot.kinds)
        raise TypeError(f"The snapshot is of {names}, not {' > '.join(k.__name__ for k in kinds)}.")
    states = _copy(list(snapshot.layers), _shared_memo(kinds, list(snapshot.layers)))
    for layer, state in zip(layers, states):
        layer.__dict__.update(state)


__all__ = ["Snapshot", "register_immutable", "take_snapshot", "restore_snapshot"]
