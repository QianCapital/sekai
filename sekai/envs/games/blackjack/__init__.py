"""Blackjack: a casino-accurate multi-seat table, and an environment for one seat.

    env = sekai.make("Blackjack-v0")  # BlackjackEnv: one round per episode
    result = env.reset(seed=0)
    while not result.done:
        result = env.step(env.action_space.sample(mask=env.action_mask()))

`BlackjackEnv` seats one agent against the dealer at a `BlackjackTable`: it bets, then
plays its hands; the reward is the round's net chips in units of the table minimum. The
shoe persists from one episode to the next, so the count carries over and can be
learned. The table itself (`BlackjackTable`, with `Rules` and a `Shoe`) deals any number
of seats and is driven by text moves (`"bet:20"`, `"hit"`, `"stand"`, `"double"`,
`"split"`, `"surrender"`); its `view()` is JSON-friendly and never contains the shoe order
or an unrevealed hole card.

`BlackjackRenderer` draws the felt table (RGB frames); `BlackjackTextRenderer` describes it
in text.
"""

from __future__ import annotations

from sekai.envs.games.blackjack.actions import (
    ACTIONS,
    DEFAULT_BET_LEVELS,
    PLAY_ACTION_NAMES,
    action_to_move,
    legal_action_mask,
    move_to_action,
    num_actions,
)
from sekai.envs.games.blackjack.cards import (
    DECK,
    RANKS,
    SUITS,
    card_value,
    hand_value,
    is_natural,
    parse_card,
)
from sekai.envs.games.blackjack.env import BlackjackEnv
from sekai.envs.games.blackjack.errors import (
    BlackjackError,
    IllegalAction,
    NotYourTurn,
    RoundOver,
)
from sekai.envs.games.blackjack.observations import OBS_SIZE, encode_view, observation_space
from sekai.envs.games.blackjack.rendering import BlackjackRenderer, BlackjackTextRenderer
from sekai.envs.games.blackjack.rules import Rules
from sekai.envs.games.blackjack.shoe import Shoe
from sekai.envs.games.blackjack.table import BlackjackTable

__all__ = [
    # Rules, cards, shoe, table
    "Rules",
    "RANKS",
    "SUITS",
    "DECK",
    "parse_card",
    "card_value",
    "hand_value",
    "is_natural",
    "Shoe",
    "BlackjackTable",
    # Environment
    "BlackjackEnv",
    "OBS_SIZE",
    "encode_view",
    "observation_space",
    "ACTIONS",
    "DEFAULT_BET_LEVELS",
    "PLAY_ACTION_NAMES",
    "num_actions",
    "action_to_move",
    "move_to_action",
    "legal_action_mask",
    # Renderers
    "BlackjackRenderer",
    "BlackjackTextRenderer",
    # Errors
    "BlackjackError",
    "IllegalAction",
    "NotYourTurn",
    "RoundOver",
]
