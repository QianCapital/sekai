"""Exceptions raised by the Go environment and engine.

All of them derive from `GoEnvError`, a `sekai.envs.games.GameError` (and so a
`sekai.SekaiError`), so callers can catch one game's errors or every game's.
`IllegalMove` is also the games' `IllegalMove` (a `sekai.InvalidAction`), `GameOver`
the games' `GameOver`, and `InvalidVertex` a `ValueError`.
"""

from __future__ import annotations

from sekai.envs.games import errors as _games
from sekai.error import SekaiError

ILLEGAL_REASONS: dict[str, str] = {
    "occupied": "the point is already occupied",
    "suicide": "it would leave its own group without liberties (suicide is not allowed)",
    "ko": "it retakes a ko immediately (simple ko)",
    "superko": "it would recreate an earlier board position (positional superko)",
    "off-board": "it is not a point on the board",
}
"""Machine-readable reason codes carried by `IllegalMove.reason`, with explanations."""


class GoEnvError(_games.GameError):
    """Base class of the errors raised by the Go environment and engine."""


class IllegalMove(_games.IllegalMove, GoEnvError):
    """Raised when an action is not a legal move in the current position.

    `reason` is one of the keys of `ILLEGAL_REASONS`: `"occupied"`, `"suicide"`, `"ko"`,
    `"superko"` or `"off-board"`. Use `info["legal_action_mask"]` to avoid it.

    Attributes:
        action: The action index.
        vertex: Its GTP vertex (`"D4"`).
        reason: Why it is illegal.
        board: The board string of the position, if known.
    """

    def __init__(self, action: int, vertex: str, reason: str, board: str | None = None) -> None:
        self.action = action
        self.vertex = vertex
        self.reason = reason
        self.board = board
        explanation = ILLEGAL_REASONS.get(reason, reason)
        SekaiError.__init__(
            self,
            f"Move {vertex} (action {action}) is illegal: {explanation}. "
            "Mask your policy with info['legal_action_mask'].",
        )


class GameOver(_games.GameOver, GoEnvError):
    """Raised when `step()` is called after the game has already ended.

    Attributes:
        result: The game's result (`"B+3.5"`, `"W+0.5"`, `"Draw"`).
    """

    def __init__(self, result: str) -> None:
        self.result = result
        super().__init__(f"The game is over ({result}). Call reset() to start a new game.")


class InvalidVertex(GoEnvError, ValueError):
    """Raised for a malformed or off-board GTP vertex, SGF point or action index."""


__all__ = ["ILLEGAL_REASONS", "GoEnvError", "IllegalMove", "GameOver", "InvalidVertex"]
