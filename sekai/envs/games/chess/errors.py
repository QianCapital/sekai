"""Exceptions raised by the chess environment.

All of them derive from `sekai.envs.games.GameError` (a `sekai.SekaiError`), so callers
can catch framework errors uniformly. `IllegalMove` additionally derives from
`sekai.InvalidAction`.
"""

from __future__ import annotations

from sekai.envs.games.chess.board import Move
from sekai.envs.games.errors import GameError
from sekai.envs.games.errors import GameOver as _GameOver
from sekai.envs.games.errors import IllegalMove as _IllegalMove
from sekai.error import SekaiError


class ChessEnvError(GameError):
    """Base class for errors raised by the chess environment."""


class IllegalMove(_IllegalMove, ChessEnvError):
    """Raised when an action decodes to a move that is not legal in the current position.

    Only raised when the environment was created with `illegal_action="raise"` (the
    default). Use `info["legal_action_mask"]` to avoid it.
    """

    def __init__(self, action: int, move: Move | None, fen: str) -> None:
        self.action = action
        self.move = move
        self.fen = fen
        uci = move.uci() if move is not None else "<off-board>"
        SekaiError.__init__(
            self,
            f"Action {action} decodes to move {uci}, which is illegal in position '{fen}'. "
            "Mask your policy with info['legal_action_mask'], or construct the env with "
            "illegal_action='lose' to forfeit instead of raising.",
        )


class GameOver(_GameOver, ChessEnvError):
    """Raised when step() is called after the game has already ended."""

    def __init__(self, result: str, fen: str) -> None:
        self.result = result
        self.fen = fen
        super().__init__(
            f"The game is over ({result}) in position '{fen}'. Call reset() to start a new game."
        )


__all__ = ["ChessEnvError", "GameOver", "IllegalMove"]
