from __future__ import annotations

import numbers
from typing import Any

import numpy as np

from sekai.error import InvalidAction
from sekai.spaces.space import Space


def parse_reset_bounds(
    options: dict[str, Any] | None,
    default_low: float,
    default_high: float,
) -> tuple[float, float]:
    """Read optional {"low": ..., "high": ...} initial-state bounds from reset() options."""
    if options is None:
        return default_low, default_high
    low = options.get("low", default_low)
    high = options.get("high", default_high)
    for name, value in (("low", low), ("high", high)):
        if not isinstance(value, numbers.Real):
            raise TypeError(f"reset option {name!r} must be a real number, got {value!r}.")
    if low > high:
        raise ValueError(f"reset option low ({low}) must be <= high ({high}).")
    return float(low), float(high)


def check_action(space: Space[Any], action: Any) -> None:
    if not space.contains(action):
        raise InvalidAction(action, space)


def flat_action(action: Any) -> np.ndarray:
    """A continuous action as a 1-D array, keeping its dtype.

    The dtype is kept (not cast to float64) because, under NumPy 2 promotion
    rules, gymnasium's dynamics do part of their arithmetic in the action's
    float32; matching that keeps trajectories bit-identical.
    """
    arr = np.asarray(action)
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"Expected a numeric action, got {action!r}.")
    return arr.reshape(-1)
