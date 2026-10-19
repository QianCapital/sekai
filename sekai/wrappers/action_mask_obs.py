from __future__ import annotations

from typing import Any

import numpy as np

from sekai._types import ActType
from sekai.core.env import Env
from sekai.core.wrapper import ObservationWrapper
from sekai.spaces.dict import Dict
from sekai.spaces.masked_box import MaskedBox
from sekai.utils.masking import (
    action_mask_space,
    flatten_action_mask,
    get_action_mask,
    supports_action_mask,
)


class ActionMaskObservation(ObservationWrapper[dict[str, Any], ActType, Any]):
    """Adds the env's current action mask to the observation.

    For policies that apply the mask themselves (masked logits in PPO, RLlib's
    action-masking models). The mask is flattened to an int8 MultiBinary
    (see `sekai.utils.flatten_action_mask`). A Dict observation gains a `key`
    entry; any other observation becomes `{"observation": obs, key: mask}`.

    The wrapped env must implement `action_mask()` (see `sekai.utils.masking`).
    """

    def __init__(self, env: Env[Any, ActType], *, key: str = "action_mask") -> None:
        if not supports_action_mask(env):
            raise TypeError(f"{env} has no action_mask() method to read the mask from.")
        mask_space = action_mask_space(env.action_space)
        space = env.observation_space
        if isinstance(space, Dict) and not isinstance(space, MaskedBox):
            if key in space.spaces:
                raise KeyError(f"Observation space already has a {key!r} key.")
            new_space = Dict({**space.spaces, key: mask_space})
            self._is_dict = True
        else:
            new_space = Dict({"observation": space, key: mask_space})
            self._is_dict = False
        super().__init__(env)
        self.key = key
        self.observation_space = new_space

    def observation(self, obs: Any) -> dict[str, Any]:
        mask = flatten_action_mask(self.env.action_space, get_action_mask(self.env))
        mask8 = mask.astype(np.int8)
        if self._is_dict:
            return {**obs, self.key: mask8}
        return {"observation": obs, self.key: mask8}
