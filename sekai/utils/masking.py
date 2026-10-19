"""Action masking: which actions are valid in the current state.

Convention: an environment that restricts its actions in some states (a halted
stock cannot be traded, a short is not available, an inventory limit is hit)
implements a method `action_mask()` returning a mask in the format that its
`action_space.sample(mask=...)` accepts:

| Action space | Mask |
|---|---|
| `Discrete(n)` | int8 array `(n,)`, 1 = allowed |
| `MultiDiscrete(nvec)` | tuple of int8 arrays (one per entry), or one array padded to `max(nvec)` |
| `Box` | array broadcastable to the shape; 0 = this component must stay at zero |

Vector envs expose it with `vec.call("action_mask")`, `sekai.ActionMaskObservation`
adds it to the observation, and `sekai.compat.to_gymnasium()` forwards it as
`action_masks()` for learners such as sb3-contrib's MaskablePPO.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from sekai.spaces.box import Box
from sekai.spaces.discrete import Discrete
from sekai.spaces.multi_binary import MultiBinary
from sekai.spaces.multi_discrete import MultiDiscrete
from sekai.spaces.space import Space


def supports_action_mask(env: Any) -> bool:
    """Whether env (or any layer of its wrapper stack) defines action_mask()."""
    has = env.has_attr("action_mask") if hasattr(env, "has_attr") else hasattr(env, "action_mask")
    return bool(has)


def get_action_mask(env: Any) -> Any:
    """The env's current action mask, or None if it does not provide one."""
    if not supports_action_mask(env):
        return None
    attr = env.get_attr("action_mask") if hasattr(env, "get_attr") else env.action_mask
    return attr() if callable(attr) else attr


def action_mask_space(space: Space[Any]) -> MultiBinary:
    """The MultiBinary space of flattened masks for an action space (see flatten_action_mask)."""
    if isinstance(space, Discrete):
        return MultiBinary(space.n)
    if isinstance(space, MultiDiscrete):
        return MultiBinary(int(space.nvec.sum()))
    if isinstance(space, Box):
        return MultiBinary(int(np.prod(space.shape)))
    raise TypeError(f"Action masks are not supported for {type(space).__name__} action spaces.")


def flatten_action_mask(space: Space[Any], mask: Any) -> NDArray[np.bool_]:
    """Convert a mask to the flat boolean layout used by masked-policy learners.

    Discrete: `(n,)`. MultiDiscrete: the per-entry masks concatenated, length
    `sum(nvec)`. Box: the mask broadcast to the shape and flattened.
    """
    if isinstance(space, Discrete):
        flat = np.asarray(mask).astype(bool).reshape(-1)
        if flat.shape != (space.n,):
            raise ValueError(f"Discrete({space.n}) mask must have shape ({space.n},).")
        return flat
    if isinstance(space, MultiDiscrete):
        parts: list[NDArray[np.bool_]] = []
        for idx in np.ndindex(space.nvec.shape):
            if isinstance(mask, np.ndarray):
                entry = mask[idx]
            else:
                entry = mask
                for i in idx:
                    entry = entry[i]
            parts.append(np.asarray(entry).astype(bool).reshape(-1)[: int(space.nvec[idx])])
        return np.concatenate(parts) if parts else np.zeros(0, dtype=bool)
    if isinstance(space, Box):
        return np.broadcast_to(np.asarray(mask).astype(bool), space.shape).reshape(-1)
    raise TypeError(f"Action masks are not supported for {type(space).__name__} action spaces.")


__all__ = [
    "supports_action_mask",
    "get_action_mask",
    "action_mask_space",
    "flatten_action_mask",
]
