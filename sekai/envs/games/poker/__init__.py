"""Heads-up no-limit Texas hold'em: engine, evaluator, environment and renderers.

```python
import sekai
from sekai.envs.games import poker

# The environment: one hand per episode, the caller plays both seats.
env = sekai.make("HeadsUpHoldem-v0")
result = env.reset(seed=0)
while not result.done:
    action = env.action_space.sample(mask=result.info["legal_action_mask"])
    result = env.step(action)
print(result.info["payoffs"])  # both seats' net, in big blinds

# The engine: a trusted dealer. Give agents views, never the hand.
hand = poker.HoldemHand(stacks=(10000, 10000), button=0, rng=7)
while not hand.is_complete:
    view = hand.view(hand.to_act)  # SeatView: that seat's information set only
    hand.act(poker.abstract_to_action(poker.CHECK_CALL, view))
print(hand.result, poker.format_hand_history(hand, hand_no=1, reveal=True))
```

Modules: `cards`, `evaluator`, `holdem` (rules, views, text moves), `actions` (the
seven-action abstraction), `observations`, `env`, `history` (hand histories) and
`rendering`.
"""

from __future__ import annotations

from sekai.envs.games.poker.actions import (
    ACTION_NAMES,
    ALL_IN,
    CHECK_CALL,
    FOLD,
    MIN_RAISE,
    NUM_ABSTRACT_ACTIONS,
    RAISE_HALF_POT,
    RAISE_POT,
    RAISE_TWO_POTS,
    abstract_actions,
    abstract_to_action,
    action_to_abstract,
    legal_abstract_actions,
    legal_abstract_mask,
)
from sekai.envs.games.poker.cards import (
    RANKS,
    SUITS,
    Deck,
    card_from_str,
    card_to_str,
    cards_to_str,
    format_cards,
    parse_cards,
    shuffled_deck,
)
from sekai.envs.games.poker.env import HoldemEnv
from sekai.envs.games.poker.errors import (
    HandComplete,
    HiddenInformation,
    IllegalAction,
    InvalidCard,
    OutOfTurn,
    PokerError,
)
from sekai.envs.games.poker.evaluator import (
    best_five,
    best_five_str,
    category,
    category_index,
    describe,
    evaluate,
)
from sekai.envs.games.poker.history import format_hand_history
from sekai.envs.games.poker.holdem import (
    Action,
    HandResult,
    HoldemHand,
    LegalActions,
    SeatView,
    format_action,
    parse_action,
)
from sekai.envs.games.poker.observations import encode_observation, observation_size
from sekai.envs.games.poker.rendering import HoldemRenderer, HoldemTextRenderer

__all__ = [
    # Environment and renderers
    "HoldemEnv",
    "HoldemRenderer",
    "HoldemTextRenderer",
    "encode_observation",
    "observation_size",
    # Engine
    "HoldemHand",
    "Action",
    "LegalActions",
    "HandResult",
    "SeatView",
    "parse_action",
    "format_action",
    "format_hand_history",
    # Abstraction
    "ACTION_NAMES",
    "NUM_ABSTRACT_ACTIONS",
    "FOLD",
    "CHECK_CALL",
    "MIN_RAISE",
    "RAISE_HALF_POT",
    "RAISE_POT",
    "RAISE_TWO_POTS",
    "ALL_IN",
    "abstract_actions",
    "abstract_to_action",
    "action_to_abstract",
    "legal_abstract_mask",
    "legal_abstract_actions",
    # Cards and evaluation
    "RANKS",
    "SUITS",
    "Deck",
    "card_from_str",
    "card_to_str",
    "cards_to_str",
    "format_cards",
    "parse_cards",
    "shuffled_deck",
    "evaluate",
    "best_five",
    "best_five_str",
    "category",
    "category_index",
    "describe",
    # Errors
    "PokerError",
    "IllegalAction",
    "OutOfTurn",
    "HandComplete",
    "HiddenInformation",
    "InvalidCard",
]
