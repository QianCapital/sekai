from __future__ import annotations

import numpy as np

from sekai._types import SeedType


def np_random(seed: SeedType = None) -> tuple[np.random.Generator, int]:
    """Create a numpy Generator and return (rng, seed_used).

    If seed is already a Generator, returns it with -1 as the seed (unknown).
    If seed is None, generates a random seed from OS entropy.
    """
    if isinstance(seed, np.random.Generator):
        return seed, -1
    if seed is None:
        seed = int.from_bytes(np.random.bytes(4), byteorder="little")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    return rng, seed


def make_seed_sequence(base_seed: int, n: int) -> list[int]:
    """Deterministically derive n independent seeds from a base seed."""
    ss = np.random.SeedSequence(base_seed)
    children = ss.spawn(n)
    return [int(child.generate_state(1)[0]) for child in children]
