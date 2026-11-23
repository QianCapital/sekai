"""The observation vector of heads-up hold'em, built from a `SeatView` only.

Because the encoder only ever sees one seat's information set, an observation can never
leak the opponent's cards or the deck. The vector is `float32` with
`125 + 6 * history_length` entries, from the viewer's perspective. Chip amounts are
divided by `scale = max(starting_stacks)`.

| slot | meaning |
|---|---|
| 0-51 | own hole cards, one-hot by card integer (`rank * 4 + suit`) |
| 52-103 | board cards, one-hot by card integer |
| 104-107 | street one-hot: preflop, flop, turn, river (all zero once the hand is over) |
| 108 | 1.0 if the viewer is on the button (small blind; acts first preflop only) |
| 109 | pot (both players' committed chips) / scale, in `[0, 2]` |
| 110 | own stack (chips behind) / scale (above 1 only after winning the hand) |
| 111 | opponent stack / scale (likewise) |
| 112 | own street bet / scale |
| 113 | opponent street bet / scale |
| 114 | to_call: chips the viewer must add to call (capped at its stack) / scale |
| 115 | min_raise_to / scale; 0 when the viewer cannot raise (or is not to act) |
| 116 | big blind / scale (stack depth: `1 / slot` is the deeper stack in big blinds) |
| 117-124 | voluntary bets + raises this hand per street and seat, `min(count, 4) / 4` |
| 125+ | optional: the last `history_length` actions, most recent first, 6 slots each |

The bet counts are in the order [preflop own, preflop opp, flop own, flop opp, turn own,
turn opp, river own, river opp]; blinds do not count. Each history entry is [by viewer, by
opponent, fold, check/call, bet/raise, chips added / scale], all zero where fewer actions
have happened.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from sekai.envs.games.poker.holdem import BETTING_STREETS, SeatView
from sekai.spaces.box import Box

NUM_CARD_SLOTS = 52
HOLE_OFFSET = 0
BOARD_OFFSET = 52
STREET_OFFSET = 104
BUTTON_SLOT = 108
POT_SLOT = 109
OWN_STACK_SLOT = 110
OPP_STACK_SLOT = 111
OWN_BET_SLOT = 112
OPP_BET_SLOT = 113
TO_CALL_SLOT = 114
MIN_RAISE_SLOT = 115
BIG_BLIND_SLOT = 116
BET_COUNT_OFFSET = 117
HISTORY_OFFSET = 125
BASE_SIZE = 125
HISTORY_ENTRY_SIZE = 6
MAX_BET_COUNT = 4

Observation: TypeAlias = NDArray[np.float32]


def observation_size(history_length: int = 0) -> int:
    """The length of the observation vector."""
    return BASE_SIZE + HISTORY_ENTRY_SIZE * history_length


def observation_space(history_length: int = 0) -> Box:
    """`Box` with per-slot bounds: `[0, 2]` for the pot and stacks, `[0, 1]` elsewhere."""
    n = observation_size(history_length)
    high = np.ones(n, dtype=np.float32)
    high[[POT_SLOT, OWN_STACK_SLOT, OPP_STACK_SLOT]] = 2.0
    return Box(np.zeros(n, dtype=np.float32), high, dtype=np.float32)


def encode_observation(view: SeatView, history_length: int = 0) -> Observation:
    """Encode a seat's view (`view.seat` must be a seat, not a public view)."""
    if not isinstance(view, SeatView):
        raise TypeError(f"Expected a SeatView, got {type(view).__name__}.")
    if view.seat is None:
        raise ValueError("encode_observation needs a seat view, not a public view.")
    me, opp = view.seat, 1 - view.seat
    obs = np.zeros(observation_size(history_length), dtype=np.float32)
    for c in view.hole:
        obs[HOLE_OFFSET + c] = 1.0
    for c in view.board:
        obs[BOARD_OFFSET + c] = 1.0
    if view.street in BETTING_STREETS:
        obs[STREET_OFFSET + BETTING_STREETS.index(view.street)] = 1.0
    obs[BUTTON_SLOT] = 1.0 if view.button == me else 0.0

    scale = float(max(view.starting_stacks))
    obs[POT_SLOT] = view.pot / scale
    obs[OWN_STACK_SLOT] = view.stacks[me] / scale
    obs[OPP_STACK_SLOT] = view.stacks[opp] / scale
    obs[OWN_BET_SLOT] = view.street_bets[me] / scale
    obs[OPP_BET_SLOT] = view.street_bets[opp] / scale
    if not view.is_complete:
        obs[TO_CALL_SLOT] = view.to_call / scale
    if view.legal is not None and view.legal.min_raise_to is not None:
        obs[MIN_RAISE_SLOT] = view.legal.min_raise_to / scale
    obs[BIG_BLIND_SLOT] = min(view.big_blind / scale, 1.0)

    counts = [0] * 8
    actions = [e for e in view.events if e["type"] == "action"]
    for e in actions:
        if e["kind"] in ("bet", "raise"):
            counts[2 * BETTING_STREETS.index(e["street"]) + (0 if e["seat"] == me else 1)] += 1
    for i, n in enumerate(counts):
        obs[BET_COUNT_OFFSET + i] = min(n, MAX_BET_COUNT) / MAX_BET_COUNT

    for k, e in enumerate(reversed(actions[-history_length:] if history_length else [])):
        base = HISTORY_OFFSET + HISTORY_ENTRY_SIZE * k
        obs[base + (0 if e["seat"] == me else 1)] = 1.0
        kind = e["kind"]
        obs[base + (2 if kind == "fold" else 3 if kind in ("check", "call") else 4)] = 1.0
        obs[base + 5] = min(e["added"] / scale, 1.0)
    return obs


__all__ = [
    "BASE_SIZE",
    "HISTORY_ENTRY_SIZE",
    "HOLE_OFFSET",
    "BOARD_OFFSET",
    "STREET_OFFSET",
    "BUTTON_SLOT",
    "POT_SLOT",
    "OWN_STACK_SLOT",
    "OPP_STACK_SLOT",
    "OWN_BET_SLOT",
    "OPP_BET_SLOT",
    "TO_CALL_SLOT",
    "MIN_RAISE_SLOT",
    "BIG_BLIND_SLOT",
    "BET_COUNT_OFFSET",
    "HISTORY_OFFSET",
    "observation_size",
    "observation_space",
    "encode_observation",
]
