"""Exceptions raised by the blackjack table and environment.

All of them derive from `BlackjackError`, a `sekai.envs.games.GameError` (so a
`sekai.SekaiError`), so callers can catch them per game or for every game.
`IllegalAction` is also an `IllegalMove` (and so a `sekai.InvalidAction`), and `RoundOver`
a `GameOver`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from sekai.envs.games.errors import GameError, GameOver, IllegalMove
from sekai.error import SekaiError


class BlackjackError(GameError):
    """Base class for errors raised by the blackjack table and environment."""


class IllegalAction(IllegalMove, BlackjackError):
    """Raised when an action is not legal for the seat or hand to act.

    Table moves (`"hit"`, `"bet:40"`) and environment action indices both raise it. Use
    `BlackjackTable.legal_actions()` or `info["legal_action_mask"]` to avoid it.

    Attributes:
        action: The action or move refused.
        reason: Why, in words.
        legal: What was legal instead (a list of moves or indices, or the betting range).
    """

    def __init__(
        self, action: Any, reason: str, legal: Sequence[Any] | dict[str, Any] = ()
    ) -> None:
        self.action = action
        self.reason = reason
        self.legal = legal
        detail = f" Legal: {legal!r}." if legal else ""
        SekaiError.__init__(self, f"Illegal blackjack action {action!r}: {reason}.{detail}")


class NotYourTurn(BlackjackError):
    """Raised when a seat tries to act while another seat (or nobody) is to act.

    Attributes:
        seat: The seat that tried to act.
        to_act: The seat to act, or `None`.
    """

    def __init__(self, seat: int, to_act: int | None) -> None:
        self.seat = seat
        self.to_act = to_act
        who = "nobody can act" if to_act is None else f"seat {to_act} is to act"
        super().__init__(f"Seat {seat} cannot act now: {who}.")


class RoundOver(GameOver, BlackjackError):
    """Raised when `BlackjackEnv.step()` is called after the episode's round has settled.

    Attributes:
        round_no: The number of the round that is over.
    """

    def __init__(self, round_no: int) -> None:
        self.round_no = round_no
        super().__init__(
            f"Round {round_no} is over and has been settled. Call reset() to play the next round."
        )


__all__ = ["BlackjackError", "IllegalAction", "NotYourTurn", "RoundOver"]
