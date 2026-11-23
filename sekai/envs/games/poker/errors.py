"""Errors raised by heads-up hold'em.

All of them derive from `PokerError`, itself a `sekai.envs.games.GameError` (and so a
`sekai.SekaiError`), so callers can catch one game's errors or every game's alike.
`IllegalAction` also derives from the games' `IllegalMove` (a `sekai.InvalidAction`),
`HandComplete` from `GameOver`, and `InvalidCard` from `ValueError`.
"""

from __future__ import annotations

from typing import Any

from sekai.envs.games.errors import GameError, GameOver, IllegalMove
from sekai.error import SekaiError


class PokerError(GameError):
    """Base class of the errors raised by `sekai.envs.games.poker`."""


class InvalidCard(PokerError, ValueError):
    """A card (or list of cards) could not be parsed, is out of range, or is duplicated."""


class IllegalAction(IllegalMove, PokerError):
    """The requested action is not legal for the player to act.

    Raised by `HoldemHand.act`, `parse_action`, the action abstraction and
    `HoldemEnv.step`.

    Attributes:
        action: Whatever was requested.
        legal: The `LegalActions` in force (`None` when nobody may act).
    """

    def __init__(self, message: str, *, action: Any = None, legal: Any = None) -> None:
        self.action = action
        self.legal = legal
        SekaiError.__init__(self, message)


class OutOfTurn(IllegalAction):
    """A seat tried to act while it is not that seat's turn."""


class HandComplete(GameOver, PokerError):
    """An action was requested after the hand finished. Start a new `HoldemHand`."""

    def __init__(self, message: str = "The hand is complete; start a new hand.") -> None:
        super().__init__(message)


class HiddenInformation(PokerError):
    """Private information was requested before it may be revealed (e.g. `reveal()`
    before the hand is complete)."""


__all__ = [
    "PokerError",
    "InvalidCard",
    "IllegalAction",
    "OutOfTurn",
    "HandComplete",
    "HiddenInformation",
]
