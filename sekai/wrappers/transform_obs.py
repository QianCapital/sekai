from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, TypeVar

from sekai._types import ActType
from sekai.core.env import Env
from sekai.core.wrapper import ObservationWrapper
from sekai.spaces.space import Space

F = TypeVar("F", bound=Callable[..., Any])


def _func_argument(func: F | None, fn: F | None, wrapper: str) -> F:
    """The callable given as `func` or, deprecated (sekai 0.1's name), as `fn=`."""
    if fn is not None:
        if func is not None:
            raise TypeError(f"{wrapper}() got both func and fn; pass only func.")
        warnings.warn(
            f"{wrapper}(fn=...) is deprecated and will be removed in sekai 0.3; "
            f"use {wrapper}(func=...), as in gymnasium.",
            DeprecationWarning,
            stacklevel=3,
        )
        return fn
    if func is None:
        raise TypeError(f"{wrapper}() missing required argument: 'func'")
    return func


class TransformObservation(ObservationWrapper[Any, ActType, Any]):
    """Applies an arbitrary callable transformation to every observation.

    `func` was called `fn` before sekai 0.2; `fn=` still works but is deprecated.

    Example:
        ```python
        env = TransformObservation(env, lambda obs: obs / 255.0)
        ```
    """

    def __init__(
        self,
        env: Env[Any, ActType],
        func: Callable[[Any], Any] | None = None,
        new_observation_space: Space[Any] | None = None,
        *,
        fn: Callable[[Any], Any] | None = None,
    ) -> None:
        self.func = _func_argument(func, fn, "TransformObservation")
        super().__init__(env)
        if new_observation_space is not None:
            self.observation_space = new_observation_space

    def observation(self, obs: Any) -> Any:
        return self.func(obs)
