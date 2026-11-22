"""Policies for the strategy games: the protocol and a random baseline.

A policy maps an observation and its `info` to an action: `act(observation, info)`.
Every game env puts the legal moves in `info["legal_action_mask"]`, so a policy never
needs the game's rules to stay legal.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from sekai._types import SeedType
from sekai.utils.seeding import np_random


@runtime_checkable
class Policy(Protocol):
    """Anything that chooses an action: `act(observation, info) -> int`."""

    def act(self, observation: Any, info: dict[str, Any]) -> int: ...


class RandomPolicy:
    """Plays a uniformly random legal action (from `info["legal_action_mask"]`).

    Args:
        seed: Seed or `numpy.random.Generator` for its choices.
    """

    def __init__(self, seed: SeedType = None) -> None:
        self._rng, _ = np_random(seed)

    def seed(self, seed: SeedType = None) -> int:
        """Re-seed it; returns the seed used."""
        self._rng, used = np_random(seed)
        return used

    def act(self, observation: Any, info: dict[str, Any]) -> int:
        legal = np.flatnonzero(info["legal_action_mask"])
        if legal.size == 0:
            raise ValueError("No legal actions available: the game is over.")
        return int(self._rng.choice(legal))

    def __repr__(self) -> str:
        return "RandomPolicy()"


__all__ = ["Policy", "RandomPolicy"]
