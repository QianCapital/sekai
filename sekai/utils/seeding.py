"""Seeding utilities for reproducible environments."""

from __future__ import annotations

import numpy as np


def np_random(seed: int | None = None) -> tuple[np.random.Generator, int]:
    """Create a :class:`numpy.random.Generator` from an optional integer seed.

    Parameters
    ----------
    seed:
        Optional integer seed.  When ``None`` a random seed is chosen.

    Returns
    -------
    tuple[Generator, int]
        The new RNG and the integer seed that was used to initialise it.
    """
    if seed is not None and not (0 <= seed <= 2**31 - 1):
        raise ValueError(f"Seed must be in [0, 2**31 - 1], got {seed!r}")

    seed_seq = np.random.SeedSequence(seed)
    rng = np.random.default_rng(seed_seq)
    actual_seed = int(seed_seq.entropy) if seed is None else seed  # type: ignore[arg-type]
    return rng, actual_seed
