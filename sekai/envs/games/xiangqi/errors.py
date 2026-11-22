"""Exceptions raised by the xiangqi engine and environment.

All of them derive from `XiangqiError`, a `sekai.envs.games.GameError` (and so a
`sekai.SekaiError`), so callers can catch framework errors uniformly. `IllegalMove` is
also the games' `IllegalMove` (a `sekai.InvalidAction`), `GameOver` the games'
`GameOver`, and the two parse errors (`InvalidFen`, `InvalidMoveFormat`) are `ValueError`s.
"""

from __future__ import annotations

from sekai.envs.games import errors as _games
from sekai.error import SekaiError


class XiangqiError(_games.GameError):
    """Base class for errors raised by the xiangqi engine and environment."""


class InvalidFen(XiangqiError, ValueError):
    """Raised when a FEN string is malformed or describes an impossible position."""

    def __init__(self, fen: object, reason: str) -> None:
        self.fen = fen
        self.reason = reason
        super().__init__(f"Invalid Xiangqi FEN {fen!r}: {reason}.")


class InvalidMoveFormat(XiangqiError, ValueError):
    """Raised when a string is not a well-formed UCCI move such as `"h2e2"`."""

    def __init__(self, text: object) -> None:
        self.text = text
        super().__init__(
            f"{text!r} is not a UCCI move: expected four characters <file><rank><file><rank> "
            "with files a-i and ranks 0-9, e.g. 'h2e2'."
        )


class IllegalMove(_games.IllegalMove, XiangqiError):
    """Raised when a move (or the move an action decodes to) is illegal in the position.

    `move` is the UCCI string, `action` the action index when the move came from
    `XiangqiEnv.step`. Use `info["legal_action_mask"]` to avoid it.
    """

    def __init__(
        self,
        move: str | None,
        fen: str,
        *,
        action: int | None = None,
        reason: str | None = None,
    ) -> None:
        self.move = move
        self.fen = fen
        self.action = action
        self.reason = reason
        if action is None:
            message = f"Move {move} is illegal in position '{fen}'"
        else:
            message = (
                f"Action {action} decodes to move {move}, which is illegal in position '{fen}'"
            )
        if reason:
            message += f" ({reason})"
        message += "."
        if action is not None:
            message += (
                " Mask your policy with info['legal_action_mask'], or construct the env with "
                "illegal_action='lose' to forfeit instead of raising."
            )
        SekaiError.__init__(self, message)


class GameOver(_games.GameOver, XiangqiError):
    """Raised when a move is played after the game has already ended."""

    def __init__(self, result: str, fen: str) -> None:
        self.result = result
        self.fen = fen
        super().__init__(
            f"The game is over ({result}) in position '{fen}'. Call reset() to start a new game."
        )


__all__ = ["XiangqiError", "InvalidFen", "InvalidMoveFormat", "IllegalMove", "GameOver"]
