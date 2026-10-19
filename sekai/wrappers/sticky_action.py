from __future__ import annotations

import copy
from typing import Any

import numpy as np

from sekai._types import ActType, ObsType, SeedType
from sekai.core.env import Env
from sekai.core.result import ResetResult
from sekai.core.wrapper import ActionWrapper
from sekai.utils.seeding import make_seed_sequence, np_random


def _child_generator(gen: np.random.Generator) -> np.random.Generator:
    """A generator independent of gen, spawned from its SeedSequence without drawing from it."""
    bit_gen = gen.bit_generator
    # `seed_seq` is public from NumPy 1.25; older versions keep it in `_seed_seq`.
    seed_seq = getattr(bit_gen, "seed_seq", None) or getattr(bit_gen, "_seed_seq", None)
    if isinstance(seed_seq, np.random.SeedSequence):
        return np.random.default_rng(seed_seq.spawn(1)[0])
    # No SeedSequence to spawn from: draw a seed instead (this advances gen's stream).
    return np_random(int(gen.integers(2**63)))[0]


class StickyAction(ActionWrapper[ObsType, ActType, ActType]):
    """With probability repeat_action_probability, repeats the previous action.

    Models execution that sometimes lags the agent's decision (Machado et al.,
    2018, for Atari; order-routing delay in trading). The wrapper keeps its own
    RNG, derived from the reset seed (an int, or a Generator's SeedSequence), so
    it does not perturb the env's stream.
    """

    def __init__(self, env: Env[ObsType, ActType], repeat_action_probability: float) -> None:
        if not 0.0 <= repeat_action_probability < 1.0:
            raise ValueError(
                f"repeat_action_probability must be in [0, 1), got {repeat_action_probability}."
            )
        super().__init__(env)
        self.repeat_action_probability = repeat_action_probability
        self._last_action: Any = None
        self._sticky_rng: np.random.Generator = np_random()[0]

    def _reseed(self, seed: SeedType) -> None:
        self._last_action = None
        if isinstance(seed, (int, np.integer)):
            self._sticky_rng = np_random(make_seed_sequence(seed, 2)[1])[0]
        elif isinstance(seed, np.random.Generator):
            self._sticky_rng = _child_generator(seed)

    def action(self, action: ActType) -> ActType:
        if self._last_action is not None and (
            self._sticky_rng.random() < self.repeat_action_probability
        ):
            action = self._last_action
        # Copy: callers (and VecEnvs, which pass views of one batch array) may
        # reuse the action buffer, which would silently change the stored action.
        self._last_action = copy.deepcopy(action)
        return action

    def reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[ObsType]:
        self._reseed(seed)
        return self.env.reset(seed=seed, options=options)

    async def async_reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[ObsType]:
        self._reseed(seed)
        return await self.env.async_reset(seed=seed, options=options)
