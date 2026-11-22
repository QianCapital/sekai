"""Blackjack: cards, hand totals, rules and the shoe."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from sekai.envs.games.blackjack import Rules, Shoe
from sekai.envs.games.blackjack.cards import (
    DECK,
    card_value,
    hand_value,
    is_natural,
    parse_card,
)


def test_deck_composition() -> None:
    assert len(DECK) == 52 and len(set(DECK)) == 52
    values = Counter(card_value(c) for c in DECK)
    assert values == {1: 4, 2: 4, 3: 4, 4: 4, 5: 4, 6: 4, 7: 4, 8: 4, 9: 4, 10: 16}


@pytest.mark.parametrize(
    ("text", "card"), [("Ah", "Ah"), ("ah", "Ah"), ("10d", "Td"), (" Qs ", "Qs"), ("tc", "Tc")]
)
def test_parse_card(text: str, card: str) -> None:
    assert parse_card(text) == card


@pytest.mark.parametrize("bad", ["", "A", "Ax", "1h", "Zz", "AhK"])
def test_parse_card_rejects(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_card(bad)


def test_card_value_forms() -> None:
    assert [card_value(c) for c in ("Ah", "2c", "9d", "Th", "Js", "Qc", "Kd")] == [
        1,
        2,
        9,
        10,
        10,
        10,
        10,
    ]
    assert card_value("A") == 1 and card_value("K") == 10 and card_value("10") == 10
    assert card_value(1) == 1 and card_value(11) == 1 and card_value(10) == 10
    for bad in (0, 12, "X", "Ax", True):
        with pytest.raises(ValueError):
            card_value(bad)


@pytest.mark.parametrize(
    ("cards", "expected"),
    [
        ([], (0, False)),
        (["Ah", "6d"], (17, True)),
        (["Ah", "6d", "Tc"], (17, False)),
        (["Ah", "Ad"], (12, True)),
        (["Ah", "Ad", "Tc"], (12, False)),
        (["Ah", "Ad", "9c"], (21, True)),
        (["Kh", "Qd", "5c"], (25, False)),
        (["Ah", "Kd"], (21, True)),
        (["5h", "6d"], (11, False)),
    ],
)
def test_hand_value(cards: list[str], expected: tuple[int, bool]) -> None:
    assert hand_value(cards) == expected


def test_is_natural() -> None:
    assert is_natural(["Ah", "Kd"]) and is_natural(["Tc", "As"])
    assert not is_natural(["Ah", "5d", "5c"]) and not is_natural(["Th", "Kd"])


def test_rules_defaults_and_validation() -> None:
    r = Rules()
    assert (r.decks, r.dealer_hits_soft_17, r.blackjack_payout, r.max_hands) == (6, False, 1.5, 4)
    assert r.surrender == "late" and r.dealer_peeks and r.double_after_split
    assert not r.resplit_aces and not r.hit_split_aces
    assert (r.min_bet, r.max_bet, r.bet_increment, r.penetration) == (10, 500, 5, 0.75)
    assert r.label == "6D S17 DAS LS 3:2"
    assert Rules.from_dict(r.to_dict()) == r
    assert Rules(blackjack_payout=1.2).blackjack_ratio.as_integer_ratio() == (6, 5)
    for kwargs in (
        {"decks": 0},
        {"surrender": "early"},
        {"min_bet": 12},
        {"max_bet": 5},
        {"penetration": 1.0},
    ):
        with pytest.raises(ValueError):
            Rules(**kwargs)  # type: ignore[arg-type]


def test_rules_label() -> None:
    rules = Rules(
        decks=2,
        dealer_hits_soft_17=True,
        double_after_split=False,
        surrender=None,
        resplit_aces=True,
        dealer_peeks=False,
        blackjack_payout=1.2,
    )
    assert rules.label == "2D H17 NDAS RSA ENHC 6:5"
    assert rules.total_cards == 104


def test_shoe_composition_and_seeding() -> None:
    a, b = Shoe(6, seed=3), Shoe(6, seed=3)
    assert a.total == a.remaining == 312
    assert a.full_counts() == [24] * 9 + [96]
    assert [a.draw() for _ in range(50)] == [b.draw() for _ in range(50)]
    assert Counter(Shoe(2, seed=1)._cards) == Counter(DECK * 2)


def test_shoe_seen_cards_and_counts() -> None:
    shoe = Shoe(1, seed=0)
    shoe.stack_top(["5h", "Kd", "Ac", "2s"])
    assert shoe.draw() == "5h"
    hole = shoe.draw(face_up=False)
    assert hole == "Kd"
    assert shoe.seen_total == 1 and shoe.running_count == 1
    # The face-down card still counts as remaining.
    counts = shoe.public_remaining_counts()
    assert counts[9] == 16 and counts[4] == 3 and sum(counts) == 51
    shoe.reveal("Kd")
    assert shoe.public_remaining_counts()[9] == 15 and shoe.running_count == 0
    with pytest.raises(ValueError):
        shoe.reveal("Kd")
    shoe.draw()
    shoe.draw()
    assert shoe.running_count == 0  # +1 (5) -1 (K) -1 (A) +1 (2)
    assert shoe.remaining == 48
    assert shoe.true_count == pytest.approx(0.0)


def test_shoe_running_count_matches_the_seen_cards() -> None:
    shoe = Shoe(2, seed=4)
    tags = (-1, 1, 1, 1, 1, 1, 0, 0, 0, -1)
    for i in range(90):
        shoe.draw(face_up=i % 7 != 3)
        assert shoe.running_count == sum(t * s for t, s in zip(tags, shoe.seen_counts()))
    shoe.shuffle(in_play=["As", "5c"], visible=["5c"])
    assert shoe.running_count == 1 and shoe.seen_total == 1


def test_shoe_true_count_and_penetration() -> None:
    shoe = Shoe(6, seed=0, penetration=0.5)
    while not shoe.needs_shuffle:
        shoe.draw()
    assert shoe.dealt_fraction >= 0.5 and shoe.remaining == 156
    expected = shoe.running_count / (156 / 52)
    assert shoe.true_count == pytest.approx(expected)
    shoe.shuffle()
    assert shoe.remaining == 312 and shoe.running_count == 0 and shoe.seen_total == 0


def test_shoe_shuffle_keeps_cards_in_play_out() -> None:
    shoe = Shoe(1, seed=5)
    in_play = ["Ah", "Kd", "7c"]
    shoe.shuffle(in_play=in_play, visible=["Ah", "7c"])
    assert shoe.remaining == 49
    assert not set(in_play) & set(shoe._cards)
    counts = shoe.public_remaining_counts()
    assert counts[0] == 3 and counts[6] == 3 and counts[9] == 16  # Kd face down: unseen
    shoe.reveal("Kd")
    assert shoe.public_remaining_counts()[9] == 15


def test_shoe_stack_top_validates() -> None:
    shoe = Shoe(1, seed=0)
    shoe.stack_top(["As"])
    assert shoe.draw() == "As"
    with pytest.raises(ValueError):
        shoe.stack_top(["As"])  # the only ace of spades has been dealt


def test_shoe_repr_hides_order() -> None:
    shoe = Shoe(1, seed=0)
    text = repr(shoe)
    assert "remaining=52/52" in text
    assert not any(card in text for card in DECK)


def test_shoe_accepts_generator() -> None:
    rng = np.random.default_rng(0)
    shoe = Shoe(2, rng)
    assert shoe.remaining == 104


def test_empty_shoe_raises() -> None:
    shoe = Shoe(1, seed=0)
    for _ in range(52):
        shoe.draw()
    with pytest.raises(RuntimeError):
        shoe.draw()
