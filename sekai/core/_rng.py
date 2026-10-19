from __future__ import annotations

import numpy as np

from sekai._types import SeedType


class SeededRNG:
    """The `rng` / `_set_rng(seed)` seeding API shared by Env and MultiAgentEnv."""

    _rng: np.random.Generator | None = None
    _rng_seed: int | None = None

    @property
    def rng(self) -> np.random.Generator:
        """The environment's RNG, lazily initialised from OS entropy."""
        if self._rng is None:
            from sekai.utils.seeding import np_random

            self._rng, self._rng_seed = np_random()
        return self._rng

    @rng.setter
    def rng(self, generator: np.random.Generator) -> None:
        # Assigning a generator directly is supported (MultiAgentEnv subclasses
        # written against 0.1.x managed their own self.rng); the seed is unknown.
        self._rng = generator
        self._rng_seed = -1

    def _set_rng(self, seed: SeedType) -> int:
        """Seed the environment's RNG and return the seed in use.

        With seed=None the existing generator is kept (one is created from OS
        entropy if there is none yet), so `reset(seed=0)` followed by plain
        `reset()` calls gives one reproducible stream, as in gymnasium.
        """
        if seed is None and self._rng is not None:
            return self._rng_seed if self._rng_seed is not None else -1
        from sekai.utils.seeding import np_random

        self._rng, actual_seed = np_random(seed)
        self._rng_seed = actual_seed
        return actual_seed
