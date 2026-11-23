"""Errors shared by the strategy games.

Each game raises its own subclasses (`sekai.envs.games.chess.IllegalMove`, ...), so a
caller can catch one game's errors, or every game's through these bases.
"""

from __future__ import annotations

from sekai.error import InvalidAction, SekaiError


class GameError(SekaiError):
    """Base class of the errors raised by the strategy games."""


class IllegalMove(InvalidAction, GameError):
    """A move (or the action of one) that is not legal in the current position.

    Use `info["legal_action_mask"]` (or `env.action_mask()`) to choose only legal moves.
    """


class GameOver(GameError):
    """`step()` was called after the game (or hand, or round) had ended: call `reset()`."""


__all__ = ["GameError", "GameOver", "IllegalMove"]
