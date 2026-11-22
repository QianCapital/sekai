"""The seven-action betting abstraction used by agents and `HoldemEnv`.

| index | name | concrete action |
|---|---|---|
| 0 | fold | fold (illegal when checking is free) |
| 1 | check/call | check if free, else call (calling all-in if short) |
| 2 | min-raise | bet/raise to `min_raise_to` (the minimum bet when nothing is bet yet) |
| 3 | raise half pot | raise to `current_bet + 1/2 * pot_after_call` |
| 4 | raise pot | raise to `current_bet + 1 * pot_after_call` |
| 5 | raise two pots | raise to `current_bet + 2 * pot_after_call` |
| 6 | all-in | bet/raise to `max_raise_to` |

`current_bet` is the largest street bet, `to_call` what the player needs to add to call,
and `pot_after_call = pot + to_call` with `pot` all chips committed by both players
(including this street's bets). Pot-relative amounts are rounded to the nearest chip (half
up). The raise sizes 2-5 are legal only when `min_raise_to <= to < max_raise_to`
(anything at or above `max_raise_to` is the all-in, action 6), and when two sizes land on
the same amount only the smaller index stays legal. Actions 2-6 are all illegal when
raising is impossible.

Each function takes either a `HoldemHand` (acting for its `to_act`) or a `SeatView`
(acting for the viewer; nothing is legal unless it is the viewer's turn).
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from sekai.envs.games.poker.errors import IllegalAction
from sekai.envs.games.poker.holdem import Action, HoldemHand, LegalActions, SeatView

FOLD = 0
CHECK_CALL = 1
MIN_RAISE = 2
RAISE_HALF_POT = 3
RAISE_POT = 4
RAISE_TWO_POTS = 5
ALL_IN = 6
NUM_ABSTRACT_ACTIONS = 7

ACTION_NAMES: tuple[str, ...] = (
    "fold",
    "check/call",
    "min-raise",
    "raise half pot",
    "raise pot",
    "raise two pots",
    "all-in",
)
POT_FRACTIONS: dict[int, tuple[int, int]] = {
    RAISE_HALF_POT: (1, 2),
    RAISE_POT: (1, 1),
    RAISE_TWO_POTS: (2, 1),
}
"""Pot multiple `X` of each pot-relative size as a fraction `(numerator, denominator)`."""

State: TypeAlias = HoldemHand | SeatView


def _context(state: State) -> tuple[LegalActions | None, tuple[int, int], int, int | None]:
    if isinstance(state, HoldemHand):
        legal = None if state.is_complete else state.legal()
        return legal, state.street_bets, state.pot, state.to_act
    if isinstance(state, SeatView):
        return state.legal, state.street_bets, state.pot, state.seat
    raise TypeError(f"Expected a HoldemHand or SeatView, got {type(state).__name__}.")


_FOLD = Action("fold")
_CHECK = Action("check")
_CALL = Action("call")


def _sized(kind: str, amount: int) -> Action:
    """`Action(kind, amount)`, skipping the validation a positive `int` amount passes."""
    if type(amount) is not int or amount <= 0:
        return Action(kind, amount)
    action = object.__new__(Action)
    object.__setattr__(action, "kind", kind)
    object.__setattr__(action, "amount", amount)
    return action


def abstract_actions_for(
    legal: LegalActions | None, street_bets: tuple[int, int], pot: int
) -> list[Action | None]:
    """The concrete `Action` of each abstract index, `None` where illegal."""
    out: list[Action | None] = [None] * NUM_ABSTRACT_ACTIONS
    if legal is None:
        return out
    if legal.can_fold:
        out[FOLD] = _FOLD
    out[CHECK_CALL] = _CHECK if legal.can_check else _CALL
    lo, hi = legal.min_raise_to, legal.max_raise_to
    if lo is None or hi is None:
        return out
    current = max(street_bets)
    kind = "bet" if current == 0 else "raise"
    pot_after_call = pot + legal.call_amount
    taken: set[int] = set()
    if lo < hi:
        out[MIN_RAISE] = _sized(kind, lo)
        taken.add(lo)
    for index, (num, den) in POT_FRACTIONS.items():
        to = current + (2 * num * pot_after_call + den) // (2 * den)  # round half up
        if lo <= to < hi and to not in taken:
            out[index] = _sized(kind, to)
            taken.add(to)
    out[ALL_IN] = _sized(kind, hi)
    return out


def abstract_actions(state: State) -> list[Action | None]:
    """The concrete `Action` of each of the seven abstract indices, `None` where illegal."""
    legal, street_bets, pot, _ = _context(state)
    return abstract_actions_for(legal, street_bets, pot)


def legal_abstract_mask(state: State) -> NDArray[np.int8]:
    """`int8` mask over the seven abstract actions (all zeros when nobody may act)."""
    return np.array([a is not None for a in abstract_actions(state)], dtype=np.int8)


def legal_abstract_actions(state: State) -> NDArray[np.int64]:
    """Indices of the legal abstract actions."""
    return np.flatnonzero(legal_abstract_mask(state)).astype(np.int64)


def abstract_to_action(index: int, state: State) -> Action:
    """The concrete `Action` for abstract `index`.

    Raises:
        IllegalAction: If `index` is out of range or not legal in `state`.
    """
    if not 0 <= int(index) < NUM_ABSTRACT_ACTIONS:
        raise IllegalAction(f"Abstract action {index!r} out of range [0, 7).", action=index)
    actions = abstract_actions(state)
    action = actions[int(index)]
    if action is None:
        legal = [ACTION_NAMES[i] for i, a in enumerate(actions) if a is not None]
        raise IllegalAction(
            f"Abstract action {index} ({ACTION_NAMES[int(index)]}) is not legal here; legal: "
            f"{', '.join(legal) or 'none'}.",
            action=index,
        )
    return action


def action_to_abstract(action: Action, state: State) -> int:
    """The abstract index closest to a concrete legal `action` (e.g. a human's raise).

    Fold, check and call map exactly; a bet or raise maps to the legal raise size nearest
    in amount.
    """
    actions = abstract_actions(state)
    if action.kind == "fold":
        return FOLD
    if action.kind in ("check", "call"):
        return CHECK_CALL
    assert action.amount is not None
    best, best_gap = CHECK_CALL, None
    for index in range(MIN_RAISE, NUM_ABSTRACT_ACTIONS):
        candidate = actions[index]
        if candidate is None or candidate.amount is None:
            continue
        gap = abs(candidate.amount - action.amount)
        if best_gap is None or gap < best_gap:
            best, best_gap = index, gap
    return best


__all__ = [
    "FOLD",
    "CHECK_CALL",
    "MIN_RAISE",
    "RAISE_HALF_POT",
    "RAISE_POT",
    "RAISE_TWO_POTS",
    "ALL_IN",
    "NUM_ABSTRACT_ACTIONS",
    "ACTION_NAMES",
    "POT_FRACTIONS",
    "abstract_actions",
    "abstract_actions_for",
    "legal_abstract_mask",
    "legal_abstract_actions",
    "abstract_to_action",
    "action_to_abstract",
]
