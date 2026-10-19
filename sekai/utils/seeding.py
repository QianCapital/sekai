from __future__ import annotations

import numpy as np

from sekai._types import SeedType


def np_random(seed: SeedType = None) -> tuple[np.random.Generator, int]:
    """Create a numpy Generator and return (rng, seed_used).

    If seed is already a Generator, returns it with -1 as the seed (unknown).
    If seed is None, generates a random 128-bit seed from OS entropy.
    """
    if isinstance(seed, np.random.Generator):
        return seed, -1
    if seed is None:
        # SeedSequence() draws 128 bits from the OS. (np.random.bytes(4) gave only 32 bits from
        # the legacy global RandomState: few enough to brute-force a card shuffle.)
        entropy = np.random.SeedSequence().entropy
        assert isinstance(entropy, int)
        seed = entropy
    seed = int(seed)
    rng = np.random.default_rng(seed)
    return rng, seed


def child_seeds(rng: np.random.Generator, seed_used: int, n: int) -> list[int]:
    """Seeds for n sub-components (e.g. the sub-spaces of a Dict space).

    Derived from the seed when one is known, otherwise drawn from the
    generator itself, so seeding a composite with a Generator also works.
    """
    base = seed_used if seed_used != -1 else int(rng.integers(2**63))
    return make_seed_sequence(base, n)


def make_seed_sequence(base_seed: int, n: int) -> list[int]:
    """Deterministically derive n independent seeds from a base seed."""
    ss = np.random.SeedSequence(base_seed)
    children = ss.spawn(n)
    return [int(child.generate_state(1)[0]) for child in children]
