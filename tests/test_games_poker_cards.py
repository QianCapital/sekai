"""Heads-up hold'em: cards, the deck and the hand evaluator."""

from __future__ import annotations

import math
import time
from collections import Counter
from itertools import combinations, combinations_with_replacement, pairwise

import numpy as np
import pytest

from sekai.envs.games.poker import (
    Deck,
    InvalidCard,
    best_five_str,
    card_from_str,
    card_to_str,
    cards_to_str,
    category,
    category_index,
    describe,
    evaluate,
    format_cards,
    parse_cards,
    shuffled_deck,
)
from sekai.envs.games.poker.cards import CARD_STRINGS, card_rank, card_suit, make_card, to_card
from sekai.envs.games.poker.evaluator import (
    CATEGORY_NAMES,
    best_five,
    best_five_unchecked,
    evaluate_unchecked,
)

# ----------------------------------------------------------------------
# Cards
# ----------------------------------------------------------------------


def test_integer_encoding() -> None:
    assert card_from_str("2c") == 0
    assert card_from_str("2d") == 1
    assert card_from_str("3c") == 4
    assert card_from_str("As") == 51
    assert card_from_str("Td") == 8 * 4 + 1
    assert card_to_str(0) == "2c" and card_to_str(51) == "As"
    assert all(card_from_str(card_to_str(c)) == c for c in range(52))
    assert len(set(CARD_STRINGS)) == 52
    assert card_rank(card_from_str("Kh")) == 11 and card_suit(card_from_str("Kh")) == 2
    assert make_card(12, 3) == 51


def test_lenient_text_forms() -> None:
    assert card_from_str("as") == 51
    assert card_from_str(" AS ") == 51
    assert card_from_str("10h") == card_from_str("Th")
    assert parse_cards("AsKd") == [51, 45]
    assert parse_cards("As Kd") == parse_cards("As,Kd") == [51, 45]
    assert parse_cards("10s9s") == [card_from_str("Ts"), card_from_str("9s")]
    assert parse_cards(["As", 45, np.int64(0)]) == [51, 45, 0]
    assert cards_to_str([51, 45]) == ["As", "Kd"]
    assert format_cards([51, 45]) == "As Kd"


@pytest.mark.parametrize("bad", ["", "A", "Ax", "1s", "AsK", "Zz", "AAs"])
def test_invalid_strings(bad: str) -> None:
    with pytest.raises(InvalidCard):
        card_from_str(bad)


def test_invalid_card_lists() -> None:
    for bad in ("AsK", "AsXx", "As Kd Q"):
        with pytest.raises(InvalidCard):
            parse_cards(bad)


def test_invalid_values() -> None:
    for bad in (-1, 52, 3.0, True, None):
        with pytest.raises(InvalidCard):
            to_card(bad)  # type: ignore[arg-type]
    with pytest.raises(InvalidCard):
        parse_cards("AsAs")
    assert parse_cards("AsAs", unique=False) == [51, 51]
    assert issubclass(InvalidCard, ValueError)
    with pytest.raises(InvalidCard):
        make_card(13, 0)


def test_deck() -> None:
    d1, d2 = Deck(rng=3), Deck(rng=np.random.default_rng(3))
    assert d1.remaining() == d2.remaining()
    assert sorted(d1.remaining()) == list(range(52))
    first = d1.deal(2)
    assert len(first) == 2 and len(d1) == 50 and first == d2.deal(2)
    with pytest.raises(InvalidCard):
        d1.deal(51)
    fixed = Deck(cards="AsKd2c")
    assert fixed.deal(3) == [51, 45, 0] and len(fixed) == 0
    assert shuffled_deck(5) == shuffled_deck(5) != shuffled_deck(6)
    assert all(type(c) is int for c in shuffled_deck(5))


def test_shuffle_is_a_numpy_permutation() -> None:
    # The deck is exactly Generator.permutation(52): seeds deal the same cards everywhere.
    assert shuffled_deck(11) == np.random.default_rng(11).permutation(52).tolist()


# ----------------------------------------------------------------------
# The evaluator
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cards", "name", "text"),
    [
        ("AsKsQsJsTs", "Royal Flush", "Royal Flush"),
        ("9h8h7h6h5h", "Straight Flush", "Straight Flush, Nine high"),
        ("5c4c3c2cAc", "Straight Flush", "Straight Flush, Five high"),
        ("QsQhQdQc4h", "Four of a Kind", "Four of a Kind, Queens"),
        ("QsQhQd4c4h", "Full House", "Full House, Queens full of Fours"),
        ("Ah9h7h4h2h", "Flush", "Flush, Ace high"),
        ("5d4c3h2sAd", "Straight", "Straight, Five high"),
        ("AdKcQhJsTs", "Straight", "Straight, Ace high"),
        ("9s9h9d4c2h", "Three of a Kind", "Three of a Kind, Nines"),
        ("KsKh7d7c2h", "Two Pair", "Two Pair, Kings and Sevens"),
        ("JsJh9d4c2h", "One Pair", "One Pair, Jacks"),
        ("AsJh9d4c2h", "High Card", "High Card, Ace high"),
    ],
)
def test_categories_and_descriptions(cards: str, name: str, text: str) -> None:
    value = evaluate(cards)
    assert category(value) == name
    assert describe(value) == text == describe(cards)


def _cmp(a: str, b: str) -> int:
    va, vb = evaluate(a), evaluate(b)
    return (va > vb) - (va < vb)


def test_category_order() -> None:
    ladder = [
        "AsJh9d4c2h",
        "2s2h9d4c3h",
        "3s3h2d2c4h",
        "2s2h2d4c3h",
        "5d4c3h2sAd",
        "7h5h4h3h2h",
        "2s2h2d3c3h",
        "2s2h2d2c3h",
        "5c4c3c2cAc",
        "AsKsQsJsTs",
    ]
    for lo, hi in pairwise(ladder):
        assert _cmp(hi, lo) == 1, (hi, lo)
    assert [category_index(evaluate(h)) for h in ladder] == [*range(9), 8]


def test_kickers() -> None:
    assert _cmp("AsKh9d4c2h", "AsQh9d4c2h") == 1  # high card, second card
    assert _cmp("AsKh9d4c3h", "AsKh9d4c2h") == 1  # high card, fifth card
    assert _cmp("JsJhAd4c2h", "JsJhKd4c2h") == 1  # pair kicker
    assert _cmp("JsJhAd5c2h", "JsJhAd4c3h") == 1  # pair, third kicker
    assert _cmp("KsKh7d7cAh", "KsKh7d7cQh") == 1  # two pair kicker
    assert _cmp("KsKh8d8c2h", "KsKh7d7cAh") == 1  # two pair, second pair
    assert _cmp("9s9h9dAcKh", "9s9h9dAcQh") == 1  # trips kickers
    assert _cmp("3s3h3dAcAh", "2s2h2dKcKh") == 1  # full house: trips first
    assert _cmp("QsQhQdAcAh", "QsQhQdKcKh") == 1
    assert _cmp("7s7h7d7cAh", "7s7h7d7cKh") == 1  # quads kicker
    assert _cmp("6d5c4h3s2d", "5d4c3h2sAd") == 1  # six-high straight beats the wheel
    assert _cmp("Ah9h7h4h3h", "Ah9h7h4h2h") == 1  # flush, fifth card


def test_ties_across_suits_and_sizes() -> None:
    assert evaluate("AsKh9d4c2h") == evaluate("AdKc9s4h2c")
    assert evaluate("AsKsQsJsTs") == evaluate("AhKhQhJhTh")
    # The board plays: both seven-card hands use the same five.
    board = "AsKdQhJcTs"
    assert evaluate(board + "2c3d") == evaluate(board + "4h5h")


def test_seven_card_best_five() -> None:
    # Flush beats the straight also available among the seven cards.
    cards = "9h8h7h6s5h" + "2hKd"
    assert category(evaluate(cards)) == "Flush"
    assert best_five_str(cards) == ["9h", "8h", "7h", "5h", "2h"]
    # Two pair from three pairs keeps the best kicker.
    assert describe("KsKh7d7c2h2dAc") == "Two Pair, Kings and Sevens"
    assert best_five_str("KsKh7d7c2h2dAc") == ["Ks", "Kh", "7d", "7c", "Ac"]
    # Two sets make a full house of the higher set.
    assert describe("9s9h9d4c4h4sAc") == "Full House, Nines full of Fours"
    # Straight flush with a higher card of the suit outside it.
    assert describe("Ah9h8h7h6h5h2c") == "Straight Flush, Nine high"
    # The wheel is ordered 5 4 3 2 A.
    assert best_five_str("Ad2c3h4s5dKcKh") == ["5d", "4s", "3h", "2c", "Ad"]
    assert describe("Ad2c3h4s5dKcKh") == "Straight, Five high"
    # Six cards.
    assert describe("AsAhAdKcKh2s") == "Full House, Aces full of Kings"
    assert best_five_str("AsAhAdKcKh2s") == ["As", "Ah", "Ad", "Kh", "Kc"]


def test_best_five_is_the_first_best_combination() -> None:
    # Of the five-card combinations worth the most, the first in combinations order.
    rng = np.random.default_rng(4)
    for _ in range(300):
        cards = rng.permutation(52)[:7].tolist()
        best_value, first = -1, None
        for combo in combinations(cards, 5):
            v = evaluate_unchecked(combo)
            if v > best_value:
                best_value, first = v, set(combo)
        assert set(best_five(cards)) == first
        assert best_five_unchecked(cards, evaluate_unchecked(cards)) == best_five(cards)


def test_invalid_inputs() -> None:
    for bad in ("AsKs", "AsKsQsJsTs9s8s7s", "AsAsKsQsJs"):
        with pytest.raises(InvalidCard):
            evaluate(bad)
    with pytest.raises(ValueError):
        category(9 << 20)


def test_unchecked_matches_checked() -> None:
    rng = np.random.default_rng(0)
    for n in (5, 6, 7):
        for _ in range(300):
            cards = [int(c) for c in rng.permutation(52)[:n]]
            v = evaluate_unchecked(cards)
            assert v == evaluate(cards)
            assert v == max(evaluate_unchecked(c) for c in combinations(cards, 5))


def test_speed_at_least_30k_seven_card_evaluations_per_second() -> None:
    rng = np.random.default_rng(1)
    hands = [[int(c) for c in rng.permutation(52)[:7]] for _ in range(20000)]
    start = time.perf_counter()
    for h in hands:
        evaluate(h)
    rate = len(hands) / (time.perf_counter() - start)
    assert rate >= 30_000, f"{rate:,.0f} evaluations/s"


def test_every_five_card_rank_pattern() -> None:
    # A five-card hand is worth what its ranks make it, and whether it is a flush: the
    # 6175 rank patterns without a flush and 1287 flushes are the 7462 distinct values.
    hands_worth: dict[int, int] = {}
    for ranks in combinations_with_replacement(range(13), 5):
        counts = Counter(ranks)
        if max(counts.values()) > 4:
            continue
        hands = math.prod(math.comb(4, m) for m in counts.values())
        if len(counts) == 5:  # five ranks: 4 flushes, and a representative with two suits
            flush = evaluate_unchecked([r * 4 + 3 for r in ranks])
            hands_worth[flush] = 4
            cards = [r * 4 + (1 if i == 0 else 0) for i, r in enumerate(ranks)]
            hands -= 4
        else:  # a rank repeats, so the suits differ: no flush
            seen: Counter[int] = Counter()
            cards = []
            for r in ranks:
                cards.append(r * 4 + seen[r])
                seen[r] += 1
        value = evaluate_unchecked(cards)
        assert value not in hands_worth
        hands_worth[value] = hands
        # Any other suits for the same ranks (still no flush) are worth the same.
        shuffled = [(c & ~3) | ((c + 1) & 3) for c in cards]
        assert evaluate_unchecked(shuffled) == value
    assert len(hands_worth) == 7462
    by_category: Counter[str] = Counter()
    for value, hands in hands_worth.items():
        by_category[category(value)] += hands
    assert by_category == {
        "Royal Flush": 4,
        "Straight Flush": 36,
        "Four of a Kind": 624,
        "Full House": 3744,
        "Flush": 5108,
        "Straight": 10200,
        "Three of a Kind": 54912,
        "Two Pair": 123552,
        "One Pair": 1098240,
        "High Card": 1302540,
    }
    assert sum(by_category.values()) == math.comb(52, 5)
    assert sorted(CATEGORY_NAMES) == sorted(set(by_category) - {"Royal Flush"})
