"""Heads-up hold'em: the seven-action betting abstraction."""

from __future__ import annotations

import numpy as np
import pytest

from sekai.envs.games.poker import (
    ACTION_NAMES,
    ALL_IN,
    CHECK_CALL,
    FOLD,
    MIN_RAISE,
    RAISE_HALF_POT,
    RAISE_POT,
    RAISE_TWO_POTS,
    Action,
    HoldemHand,
    IllegalAction,
    LegalActions,
    abstract_actions,
    abstract_to_action,
    action_to_abstract,
    legal_abstract_actions,
    legal_abstract_mask,
)
from sekai.envs.games.poker.actions import abstract_actions_for

DECK = "AsKdQhQc2c7d9hJs3s"


def amounts(state: HoldemHand) -> list[int | str | None]:
    out: list[int | str | None] = []
    for a in abstract_actions(state):
        out.append(None if a is None else a.amount if a.amount is not None else a.kind)
    return out


def test_names() -> None:
    assert len(ACTION_NAMES) == 7
    indices = (FOLD, CHECK_CALL, MIN_RAISE, RAISE_HALF_POT, RAISE_POT, RAISE_TWO_POTS, ALL_IN)
    assert indices == tuple(range(7))


def test_preflop_small_blind() -> None:
    h = HoldemHand(deck=DECK)
    # pot 150, to_call 50 -> pot_after_call 200; half pot = 100 + 100 = 200 = min-raise (dup).
    assert amounts(h) == ["fold", "call", 200, None, 300, 500, 10000]
    assert legal_abstract_mask(h).tolist() == [1, 1, 1, 0, 1, 1, 1]
    assert legal_abstract_mask(h).dtype == np.int8
    assert legal_abstract_actions(h).tolist() == [0, 1, 2, 4, 5, 6]
    assert abstract_to_action(RAISE_POT, h) == Action("raise", 300)


def test_big_blind_option_cannot_fold() -> None:
    h = HoldemHand(deck=DECK)
    h.act("call")
    # pot 200, nothing to call: half pot = 100 + 100 = 200 (dup of min-raise), pot = 300.
    assert amounts(h) == [None, "check", 200, None, 300, 500, 10000]


def test_postflop_bet_and_raise_sizes() -> None:
    h = HoldemHand(deck=DECK)
    h.act("raise:300")
    h.act("call")
    # Flop, pot 600, no bet: min bet 100, half pot 300, pot 600, two pots 1200.
    assert amounts(h) == [None, "check", 100, 300, 600, 1200, 9700]
    assert abstract_to_action(RAISE_HALF_POT, h) == Action("bet", 300)
    h.act("bet:300")
    # Facing 300: pot_after_call = 900 + 300 = 1200 -> 300 + 600, 300 + 1200, 300 + 2400.
    assert amounts(h) == ["fold", "call", 600, 900, 1500, 2700, 9700]


def test_sizes_at_or_above_the_stack_collapse_into_all_in() -> None:
    h = HoldemHand(deck=DECK, stacks=(1000, 1000))
    h.act("raise:300")
    h.act("call")
    # Flop pot 600, stacks 700: pot (600) is legal, two pots (1200) >= max is not.
    assert amounts(h) == [None, "check", 100, 300, 600, None, 700]
    h.act("bet:600")
    # Facing 600 with 700 behind: only fold, call and all-in to 700.
    assert amounts(h) == ["fold", "call", None, None, None, None, 700]


def test_no_raises_when_opponent_all_in_or_call_is_all_in() -> None:
    h = HoldemHand(deck=DECK, stacks=(10000, 60))
    assert amounts(h) == ["fold", "call", None, None, None, None, None]
    h = HoldemHand(deck=DECK, stacks=(10000, 10000))
    h.act("allin")
    assert amounts(h) == ["fold", "call", None, None, None, None, None]


def test_rounding_half_up() -> None:
    h = HoldemHand(deck=DECK, small_blind=1, big_blind=3, stacks=(1000, 1000))
    # pot 4, to_call 2 -> pot_after_call 6: half 3 + 3 = 6 = min-raise (dup), pot 9, 2x 15.
    assert amounts(h) == ["fold", "call", 6, None, 9, 15, 1000]
    h.act("call")
    # pot 6: half = 3 + 3 = 6 (dup), pot 3 + 6 = 9
    h.act("check")
    # flop pot 6: min bet 3, half pot 3 (dup of min), pot 6, two pots 12
    assert amounts(h) == [None, "check", 3, None, 6, 12, 997]
    h.act("bet:3")
    # facing 3: pot_after_call = 9 + 3 = 12: min 6, half 3 + 6 = 9, pot 15, 2x 27
    assert amounts(h) == ["fold", "call", 6, 9, 15, 27, 997]


def test_half_pot_rounds_half_up() -> None:
    legal = LegalActions(
        can_fold=False, can_check=True, call_amount=0, min_raise_to=2, max_raise_to=1000
    )
    acts = abstract_actions_for(legal, (0, 0), 7)  # half of 7 is 3.5 -> 4
    assert [None if a is None else a.amount for a in acts] == [None, None, 2, 4, 7, 14, 1000]
    assert acts[CHECK_CALL] == Action("check")


def test_illegal_abstract_action_raises() -> None:
    h = HoldemHand(deck=DECK)
    with pytest.raises(IllegalAction):
        abstract_to_action(RAISE_HALF_POT, h)
    with pytest.raises(IllegalAction):
        abstract_to_action(7, h)
    h.act("call")
    with pytest.raises(IllegalAction):
        abstract_to_action(FOLD, h)


def test_views_and_hands_agree() -> None:
    h = HoldemHand(deck=DECK)
    h.act("raise:300")
    assert abstract_actions(h.view(1)) == abstract_actions(h)
    assert legal_abstract_mask(h.view(0)).tolist() == [0] * 7  # not seat 0's turn
    h.act("fold")
    assert legal_abstract_mask(h).tolist() == [0] * 7


def test_action_to_abstract() -> None:
    h = HoldemHand(deck=DECK)
    assert action_to_abstract(Action("fold"), h) == FOLD
    assert action_to_abstract(Action("call"), h) == CHECK_CALL
    assert action_to_abstract(Action("raise", 250), h) == MIN_RAISE
    assert action_to_abstract(Action("raise", 320), h) == RAISE_POT
    assert action_to_abstract(Action("raise", 9000), h) == ALL_IN


def test_random_states_mask_consistent_with_engine() -> None:
    rng = np.random.default_rng(3)
    for _ in range(300):
        h = HoldemHand(
            stacks=(int(rng.integers(150, 20000)), int(rng.integers(150, 20000))),
            button=int(rng.integers(2)),
            rng=rng,
        )
        while not h.is_complete:
            actions = abstract_actions(h)
            lg = h.legal()
            assert (actions[FOLD] is not None) == lg.can_fold
            assert actions[CHECK_CALL] is not None
            raises = [a.amount for a in actions[MIN_RAISE:] if a is not None]
            assert len(raises) == len(set(raises))
            if lg.can_raise:
                assert actions[ALL_IN] == Action(actions[ALL_IN].kind, lg.max_raise_to)
                assert all(lg.min_raise_to <= x <= lg.max_raise_to for x in raises)
                assert all(x < lg.max_raise_to for x in raises[:-1])
            else:
                assert raises == []
            legal = [i for i, a in enumerate(actions) if a is not None]
            h.act(actions[int(rng.choice(legal))])
