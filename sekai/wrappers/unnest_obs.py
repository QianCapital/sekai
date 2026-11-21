"""Flat-keyed observations: one `Dict` of leaves instead of nested `Dict`s and `Tuple`s.

`unnest_space`, `unnest` and `nest` are the structural maps; `UnnestObservation` applies
them to an env (and `sekai.VecUnnestObservation` to a vector env).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sekai._types import ActType
from sekai.core.env import Env
from sekai.core.wrapper import ObservationWrapper
from sekai.spaces.dict import Dict
from sekai.spaces.space import Space
from sekai.spaces.tuple import Tuple


def _check(space: Space[Any], sep: str) -> None:
    if not isinstance(sep, str) or not sep:
        raise ValueError(f"sep must be a non-empty string, got {sep!r}.")
    if not isinstance(space, (Dict, Tuple)):
        raise TypeError(
            f"Unnesting needs a Dict or Tuple observation space, got {type(space).__name__}."
        )


def _children(space: Space[Any]) -> list[tuple[str, str | int, Space[Any]]]:
    """(key part, index into a sample, sub-space) for each child of a Dict or Tuple."""
    if isinstance(space, Dict):
        return [(key, key, sub) for key, sub in space.items()]
    assert isinstance(space, Tuple)
    return [(str(i), i, sub) for i, sub in enumerate(space.spaces)]


def _leaves(
    space: Space[Any], prefix: str | None, sep: str, out: list[tuple[str, Space[Any]]]
) -> None:
    for part, _, sub in _children(space):
        key = part if prefix is None else f"{prefix}{sep}{part}"
        if isinstance(sub, (Dict, Tuple)):
            _leaves(sub, key, sep, out)
        else:
            out.append((key, sub))


def unnest_space(space: Space[Any], sep: str = "/") -> Dict:
    """The flat-keyed `Dict` of every leaf of a nested `Dict`/`Tuple` space.

    A leaf is any space other than a `Dict` or `Tuple` (a `MaskedBox` is a `Dict`, so it
    becomes its `values` and `mask` leaves). Its key joins the path from the top with
    `sep`: `Dict` keys as they are, `Tuple` positions as `"0"`, `"1"`, ... Leaves keep the
    space's order, and a space that is already flat keeps its keys.

    Example:
        ```python
        space = sekai.Dict({"assets": sekai.MaskedBox(0, 1, (2,), max_size=3),
                            "cash": sekai.Box(0, 1, (1,))})
        list(unnest_space(space).keys())  # ['assets/values', 'assets/mask', 'cash']
        ```

    Args:
        space: A `Dict` or `Tuple` space, nested to any depth.
        sep: Separator between the parts of a key; a non-empty string. The default `"/"`
            suits PyTorch libraries: a module name (Stable-Baselines3 builds one per key)
            cannot contain `"."`.

    Raises:
        TypeError: `space` is not a `Dict` or `Tuple`.
        ValueError: `sep` is empty, or two leaves get the same key (a `Dict` key that
            contains `sep`, such as `"a/b"` next to `{"a": {"b": ...}}`).
    """
    _check(space, sep)
    leaves: list[tuple[str, Space[Any]]] = []
    _leaves(space, None, sep, leaves)
    keys = [key for key, _ in leaves]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise ValueError(
            f"Leaves of the observation space share the keys {duplicates} with sep={sep!r}; "
            "pass another separator."
        )
    return Dict(dict(leaves))


def _unnest(space: Space[Any], x: Any, prefix: str | None, sep: str, out: dict[str, Any]) -> None:
    for part, index, sub in _children(space):
        key = part if prefix is None else f"{prefix}{sep}{part}"
        if isinstance(sub, (Dict, Tuple)):
            _unnest(sub, x[index], key, sep, out)
        else:
            out[key] = x[index]


def unnest(space: Space[Any], x: Any, sep: str = "/") -> dict[str, Any]:
    """Flatten the nesting of a sample of `space` into the keys of `unnest_space(space)`.

    Only the containers change: the leaves (arrays, tensors, ints) are the same objects,
    not copies. A batch of samples, as a vector env or a policy holds them (dicts and
    tuples of stacked arrays), has the same nesting and unnests the same way.

    Args:
        space: The nested `Dict` or `Tuple` space.
        x: A sample of `space`, or a batch of samples.
        sep: Separator between the parts of a key.
    """
    _check(space, sep)
    out: dict[str, Any] = {}
    _unnest(space, x, None, sep, out)
    return out


def _nest(space: Space[Any], flat: Mapping[str, Any], prefix: str | None, sep: str) -> Any:
    parts: list[Any] = []
    for part, _, sub in _children(space):
        key = part if prefix is None else f"{prefix}{sep}{part}"
        parts.append(_nest(sub, flat, key, sep) if isinstance(sub, (Dict, Tuple)) else flat[key])
    if isinstance(space, Dict):
        return dict(zip(space.keys(), parts))
    return tuple(parts)


def nest(space: Space[Any], flat: Mapping[str, Any], sep: str = "/") -> Any:
    """The inverse of `unnest`: rebuild a sample of `space` from its flat-keyed leaves.

    `nest(space, unnest(space, x)) == x` for every sample (or batch) `x` of `space`:
    dicts come back with `space`'s key order, tuples as tuples, and empty sub-spaces
    (`Dict()`, `Tuple([])`) as `{}` / `()`. Keys of `flat` that are not leaves of
    `space` are ignored. Use it in a policy that wants the nested structure back, on
    tensors as well as arrays.

    Args:
        space: The nested `Dict` or `Tuple` space.
        flat: A mapping with a value for every key of `unnest_space(space, sep)`.
        sep: Separator between the parts of a key.

    Raises:
        KeyError: A leaf of `space` has no value in `flat`.
    """
    _check(space, sep)
    return _nest(space, flat, None, sep)


class UnnestObservation(ObservationWrapper[dict[str, Any], ActType, Any]):
    """Turns a nested `Dict`/`Tuple` observation into one flat-keyed `Dict`.

    Libraries that take `Dict` observations but not nested ones need this, most of all
    Stable-Baselines3's `MultiInputPolicy` (which raises `NotImplementedError` on the
    `MaskedBox` inside `Market-v0`'s and `LimitOrderBook-v0`'s observation): wrap the
    sekai env before `sekai.compat.to_gymnasium()`. Unlike `FlattenObservation`, every
    leaf keeps its own shape, so a `MaskedBox` entry `"assets"` arrives as
    `"assets/values"` `(max_size, *shape)` and `"assets/mask"` `(max_size,)`, which a
    set encoder such as `sekai.nn.MaskedSetEncoder` reads.

    Keys join the path to each leaf with `sep` (`Tuple` positions as `"0"`, `"1"`, ...);
    see `unnest_space`. `nest(obs)` (or `sekai.wrappers.unnest_obs.nest(space, obs)` with
    `nested_space`) is the exact inverse, for single observations and batches.

    Example:
        ```python
        env = sekai.UnnestObservation(sekai.make("Market-v0"))
        print(list(env.observation_space.keys()))
        # ['assets/values', 'assets/mask', 'portfolio', 'account']
        model = PPO("MultiInputPolicy", to_gymnasium(env))
        ```

    Args:
        env: The environment to wrap; its observation space must be a `Dict` or `Tuple`.
        sep: Separator between the parts of a key; a non-empty string. Keep the default
            `"/"` for Stable-Baselines3: PyTorch module names cannot contain `"."`.

    Raises:
        TypeError: The observation space is not a `Dict` or `Tuple`.
        ValueError: `sep` is empty, or two leaves would get the same key.
    """

    def __init__(self, env: Env[Any, ActType], sep: str = "/") -> None:
        super().__init__(env)
        self.sep = sep
        self.nested_space: Space[Any] = env.observation_space
        self.observation_space = unnest_space(self.nested_space, sep)

    def observation(self, obs: Any) -> dict[str, Any]:
        return unnest(self.nested_space, obs, self.sep)

    def nest(self, obs: Mapping[str, Any]) -> Any:
        """The wrapped env's (nested) observation from a flat one, or a batch of them."""
        return nest(self.nested_space, obs, self.sep)
