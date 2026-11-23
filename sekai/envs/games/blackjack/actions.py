"""Action encoding for the blackjack environment.

The action space is `Discrete(5 + len(bet_levels))`:

| index | action |
|---|---|
| 0 | stand |
| 1 | hit |
| 2 | double |
| 3 | split |
| 4 | surrender |
| 5 + k | bet `bet_levels[k] * min_bet` (betting phase only) |

With the default `bet_levels = (1, 2, 4, 8)` and a 10-chip minimum, actions 5-8 bet 10,
20, 40 and 80. Table moves are the text the `BlackjackTable` accepts: `"stand"`, `"hit"`,
..., `"bet:40"`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sekai.envs.games.blackjack.errors import IllegalAction

STAND, HIT, DOUBLE, SPLIT, SURRENDER = 0, 1, 2, 3, 4
NUM_PLAY_ACTIONS = 5
BET_OFFSET = NUM_PLAY_ACTIONS
PLAY_ACTION_NAMES: tuple[str, ...] = ("stand", "hit", "double", "split", "surrender")
"""Playing action names in **env index** order (`0` = stand)."""

PLAY_ACTION_INDEX: dict[str, int] = {name: i for i, name in enumerate(PLAY_ACTION_NAMES)}

ACTIONS: tuple[str, ...] = ("hit", "stand", "double", "split", "surrender")
"""Playing moves in **table** order, the order of `BlackjackTable.legal_actions()` lists."""

DEFAULT_BET_LEVELS: tuple[int, ...] = (1, 2, 4, 8)
"""Bet sizes in units of the table minimum."""


def validate_bet_levels(bet_levels: Sequence[int]) -> tuple[int, ...]:
    """`bet_levels` as a tuple, checked to be strictly increasing positive ints."""
    levels = tuple(bet_levels)
    if not levels or any(not isinstance(b, int) or isinstance(b, bool) or b < 1 for b in levels):
        raise ValueError(f"bet_levels must be positive ints, got {bet_levels!r}.")
    if list(levels) != sorted(set(levels)):
        raise ValueError(f"bet_levels must be strictly increasing, got {bet_levels!r}.")
    return levels


def num_actions(bet_levels: Sequence[int] = DEFAULT_BET_LEVELS) -> int:
    """Size of the action space: the five plays and one action per bet level."""
    return NUM_PLAY_ACTIONS + len(bet_levels)


def bet_action(level_index: int) -> int:
    """Action index of bet level `level_index`."""
    return BET_OFFSET + level_index


def action_to_move(
    action: int, *, min_bet: int, bet_levels: Sequence[int] = DEFAULT_BET_LEVELS
) -> str:
    """Table move for an action index: `0` -> `"stand"`, `5 + k` -> `"bet:<amount>"`."""
    a = int(action)
    if 0 <= a < NUM_PLAY_ACTIONS:
        return PLAY_ACTION_NAMES[a]
    k = a - BET_OFFSET
    if 0 <= k < len(bet_levels):
        return f"bet:{bet_levels[k] * min_bet}"
    raise IllegalAction(action, f"action index out of range [0, {num_actions(bet_levels)})")


def move_to_action(
    move: str, *, min_bet: int, bet_levels: Sequence[int] = DEFAULT_BET_LEVELS
) -> int:
    """Inverse of `action_to_move`. Bets that are not a bet level raise."""
    m = move.strip().lower()
    if m in PLAY_ACTION_INDEX:
        return PLAY_ACTION_INDEX[m]
    if m.startswith("bet:"):
        try:
            amount = float(m[4:])
        except ValueError:
            raise IllegalAction(move, "not a bet amount") from None
        for k, level in enumerate(bet_levels):
            if level * min_bet == amount:
                return BET_OFFSET + k
        levels = [lv * min_bet for lv in bet_levels]
        raise IllegalAction(move, f"bet amount is not one of the bet levels {levels}")
    raise IllegalAction(move, "unknown move")


def action_label(
    action: int, *, min_bet: int, bet_levels: Sequence[int] = DEFAULT_BET_LEVELS
) -> str:
    """Human-readable name: `"stand"`, `"bet 40"`."""
    return action_to_move(action, min_bet=min_bet, bet_levels=bet_levels).replace(":", " ")


def legal_action_mask(
    legal: dict[str, Any] | Sequence[str],
    *,
    min_bet: int,
    bet_levels: Sequence[int] = DEFAULT_BET_LEVELS,
) -> NDArray[np.int8]:
    """`int8` mask over the action space from a table's `legal_actions()`.

    A bet level is legal when its amount lies within the betting range (and on its step).
    """
    mask = np.zeros(num_actions(bet_levels), dtype=np.int8)
    if isinstance(legal, dict):
        lo, hi, step = legal["min"], legal["max"], legal["step"]
        for k, level in enumerate(bet_levels):
            amount = level * min_bet
            if lo <= amount <= hi and amount % step == 0:
                mask[BET_OFFSET + k] = 1
    else:
        for name in legal:
            mask[PLAY_ACTION_INDEX[name]] = 1
    return mask


__all__ = [
    "STAND",
    "HIT",
    "DOUBLE",
    "SPLIT",
    "SURRENDER",
    "NUM_PLAY_ACTIONS",
    "BET_OFFSET",
    "PLAY_ACTION_NAMES",
    "PLAY_ACTION_INDEX",
    "ACTIONS",
    "DEFAULT_BET_LEVELS",
    "validate_bet_levels",
    "num_actions",
    "bet_action",
    "action_to_move",
    "move_to_action",
    "action_label",
    "legal_action_mask",
]
