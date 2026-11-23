"""Observation encoding for the blackjack environment.

An observation is a `float32` vector of length `OBS_SIZE` (43), built only from the public
table view (`BlackjackTable.view`), so it can never leak the shoe order or the dealer's
hole card. "The hand" is the seat's active hand while playing.

| slot | meaning |
|---|---|
| 0 | 1.0 in the betting phase |
| 1 | 1.0 in the playing phase |
| 2 | the hand's total / 21 (clipped to 1) |
| 3 | 1.0 if the hand's total is soft |
| 4-13 | the hand's pair value one-hot (A, 2, ..., 9, T) when it is a two-card pair, else 0 |
| 14 | number of cards in the hand / 10 (clipped to 1) |
| 15 | 1.0 if doubling is legal |
| 16 | 1.0 if splitting is legal |
| 17 | 1.0 if surrender is legal |
| 18-27 | dealer up card one-hot (A, 2, ..., 9, T) |
| 28 | index of the hand among the seat's hands / `max_hands` |
| 29 | number of hands the seat holds / `max_hands` |
| 30 | Hi-Lo true count / 10, clipped to [-1, 1] |
| 31 | fraction of the shoe dealt |
| 32-41 | unseen-card composition: count of each value (A, 2, ..., 9, T) / unseen total |
| 42 | the seat's available bankroll (chips not on the table) / the initial bankroll |

Slots 2-29 are zero during betting. Every slot is in `[0, 1]` except slot 30 (`[-1, 1]`)
and slot 42 (`[0, inf)`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from sekai.envs.games.blackjack.cards import NUM_VALUES, card_value
from sekai.spaces.box import Box

OBS_SIZE = 43
PHASE_BETTING, PHASE_PLAYING = 0, 1
SLOT_TOTAL = 2
SLOT_SOFT = 3
SLOT_PAIR = 4  # 10 slots
SLOT_NUM_CARDS = 14
SLOT_CAN_DOUBLE, SLOT_CAN_SPLIT, SLOT_CAN_SURRENDER = 15, 16, 17
SLOT_DEALER_UP = 18  # 10 slots
SLOT_HAND_INDEX = 28
SLOT_NUM_HANDS = 29
SLOT_TRUE_COUNT = 30
SLOT_DEALT_FRACTION = 31
SLOT_COMPOSITION = 32  # 10 slots
SLOT_BANKROLL = 42

TRUE_COUNT_SCALE = 10.0
CARD_COUNT_SCALE = 10.0

Observation = NDArray[np.float32]


def observation_space() -> Box:
    """The observation space: `Box` of shape `(43,)` with the bounds above."""
    low = np.zeros(OBS_SIZE, dtype=np.float32)
    high = np.ones(OBS_SIZE, dtype=np.float32)
    low[SLOT_TRUE_COUNT] = -1.0
    high[SLOT_BANKROLL] = np.inf
    return Box(low=low, high=high, dtype=np.float32)


def encode_view(view: dict[str, Any], seat: int, initial_bankroll: float) -> Observation:
    """Encode a table view (see the module docstring) for `seat`."""
    obs = [0.0] * OBS_SIZE
    s = view["seats"][seat]
    phase_playing = view["phase"] == "playing"
    obs[PHASE_PLAYING if phase_playing else PHASE_BETTING] = 1.0
    if phase_playing and view["to_act"] == seat:
        h = view["active_hand"]
        hand = s["hands"][h]
        cards = hand["cards"]
        obs[SLOT_TOTAL] = min(hand["total"] / 21.0, 1.0)
        obs[SLOT_SOFT] = 1.0 if hand["soft"] else 0.0
        if len(cards) == 2:
            a, b = card_value(cards[0]), card_value(cards[1])
            if a == b:
                obs[SLOT_PAIR + a - 1] = 1.0
        obs[SLOT_NUM_CARDS] = min(len(cards) / CARD_COUNT_SCALE, 1.0)
        legal = view["legal_actions"]
        obs[SLOT_CAN_DOUBLE] = 1.0 if "double" in legal else 0.0
        obs[SLOT_CAN_SPLIT] = 1.0 if "split" in legal else 0.0
        obs[SLOT_CAN_SURRENDER] = 1.0 if "surrender" in legal else 0.0
        obs[SLOT_DEALER_UP + card_value(view["dealer"]["cards"][0]) - 1] = 1.0
        max_hands = float(view["rules"]["max_hands"])
        obs[SLOT_HAND_INDEX] = h / max_hands
        obs[SLOT_NUM_HANDS] = len(s["hands"]) / max_hands
    # np.clip of a finite float, without the NumPy call.
    obs[SLOT_TRUE_COUNT] = min(max(view["true_count"] / TRUE_COUNT_SCALE, -1.0), 1.0)
    obs[SLOT_DEALT_FRACTION] = view["shoe"]["dealt_fraction"]
    counts = view["remaining_counts"]
    unseen = float(sum(counts))
    if unseen > 0:
        for i in range(NUM_VALUES):
            obs[SLOT_COMPOSITION + i] = counts[i] / unseen
    obs[SLOT_BANKROLL] = max(float(s["bankroll"]), 0.0) / float(initial_bankroll)
    return np.array(obs, dtype=np.float32)


__all__ = [
    "OBS_SIZE",
    "Observation",
    "observation_space",
    "encode_view",
    "PHASE_BETTING",
    "PHASE_PLAYING",
    "SLOT_TOTAL",
    "SLOT_SOFT",
    "SLOT_PAIR",
    "SLOT_NUM_CARDS",
    "SLOT_CAN_DOUBLE",
    "SLOT_CAN_SPLIT",
    "SLOT_CAN_SURRENDER",
    "SLOT_DEALER_UP",
    "SLOT_HAND_INDEX",
    "SLOT_NUM_HANDS",
    "SLOT_TRUE_COUNT",
    "SLOT_DEALT_FRACTION",
    "SLOT_COMPOSITION",
    "SLOT_BANKROLL",
]
