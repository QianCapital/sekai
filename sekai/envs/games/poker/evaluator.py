"""Poker hand evaluator for 5, 6 or 7 cards.

`evaluate` returns an integer **hand value**: higher is better, values compare across any
two hands (of any size from 5 to 7 cards, the best five cards counting), and equal values
tie. The value packs the category and the deciding ranks into nibbles:

    value = category << 20 | r1 << 16 | r2 << 12 | r3 << 8 | r4 << 4 | r5

with rank indices `0` (deuce) .. `12` (ace), the most significant first and unused
nibbles zero:

| cat | name | r1, r2, ... |
|---|---|---|
| 0 | High Card | the five ranks, highest first |
| 1 | One Pair | pair rank, then three kickers |
| 2 | Two Pair | high pair, low pair, kicker |
| 3 | Three of a Kind | trips rank, then two kickers |
| 4 | Straight | top rank (the wheel A-2-3-4-5 is five-high: `3`) |
| 5 | Flush | the five ranks of the flush suit, highest first |
| 6 | Full House | trips rank, pair rank |
| 7 | Four of a Kind | quads rank, kicker |
| 8 | Straight Flush | top rank (ace-high is the Royal Flush) |

There are exactly 7,462 distinct values over all five-card hands.

Speed: the evaluator is pure Python but uses 13-bit rank masks with precomputed tables
(built at import in tens of milliseconds), so a 7-card evaluation is a single pass over
the cards plus a handful of table lookups: on the order of a million evaluations per
second on one core with `evaluate_unchecked` (no input validation, for hot loops) and a
few hundred thousand with `evaluate`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import combinations
from typing import cast

from sekai.envs.games.poker.cards import CardLike, card_to_str, parse_cards
from sekai.envs.games.poker.errors import InvalidCard

HIGH_CARD = 0
ONE_PAIR = 1
TWO_PAIR = 2
THREE_OF_A_KIND = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
FOUR_OF_A_KIND = 7
STRAIGHT_FLUSH = 8

CATEGORY_NAMES: tuple[str, ...] = (
    "High Card",
    "One Pair",
    "Two Pair",
    "Three of a Kind",
    "Straight",
    "Flush",
    "Full House",
    "Four of a Kind",
    "Straight Flush",
)
ROYAL_FLUSH_NAME = "Royal Flush"

RANK_NAMES: tuple[str, ...] = (
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
    "Ten",
    "Jack",
    "Queen",
    "King",
    "Ace",
)
RANK_PLURALS: tuple[str, ...] = (
    "Twos",
    "Threes",
    "Fours",
    "Fives",
    "Sixes",
    "Sevens",
    "Eights",
    "Nines",
    "Tens",
    "Jacks",
    "Queens",
    "Kings",
    "Aces",
)

_ACE = 12
_WHEEL_MASK = (1 << _ACE) | 0b1111  # A 2 3 4 5


def _build_tables() -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    n = 1 << 13
    high = [-1] * n
    straight = [-1] * n
    top2 = [0] * n
    top3 = [0] * n
    top5 = [0] * n
    straights = [(top, 0b11111 << (top - 4)) for top in range(12, 3, -1)]
    straights.append((3, _WHEEL_MASK))
    for mask in range(1, n):
        ranks = [r for r in range(12, -1, -1) if mask >> r & 1]
        high[mask] = ranks[0]
        for k, table in ((2, top2), (3, top3), (5, top5)):
            packed = 0
            for r in ranks[:k]:
                packed = packed << 4 | r
            packed <<= 4 * (k - min(k, len(ranks)))
            table[mask] = packed
        for top, smask in straights:
            if mask & smask == smask:
                straight[mask] = top
                break
    return high, straight, top2, top3, top5


_HIGH, _STRAIGHT, _TOP2, _TOP3, _TOP5 = _build_tables()
_BIT = [1 << (c >> 2) for c in range(52)]
_SUIT_COUNT = [1 << (4 * (c & 3)) for c in range(52)]  # one nibble per suit

_C1 = ONE_PAIR << 20
_C2 = TWO_PAIR << 20
_C3 = THREE_OF_A_KIND << 20
_C4 = STRAIGHT << 20
_C5 = FLUSH << 20
_C6 = FULL_HOUSE << 20
_C7 = FOUR_OF_A_KIND << 20
_C8 = STRAIGHT_FLUSH << 20


def evaluate_unchecked(cards: Sequence[int]) -> int:
    """`evaluate` for 5 to 7 distinct card integers, without any validation."""
    m1 = m2 = m3 = m4 = 0  # ranks held at least once / twice / three / four times
    sc = 0
    for c in cards:
        b = _BIT[c]
        sc += _SUIT_COUNT[c]
        if m1 & b:
            if m2 & b:
                if m3 & b:
                    m4 |= b
                else:
                    m3 |= b
            else:
                m2 |= b
        else:
            m1 |= b

    flush = (sc + 0x3333) & 0x8888  # nibble bit 3 set where a suit has >= 5 cards
    if flush:
        # With at most 7 cards a flush excludes quads and full houses.
        suit = 0 if flush & 0x8 else 1 if flush & 0x80 else 2 if flush & 0x800 else 3
        sm = 0
        for c in cards:
            if c & 3 == suit:
                sm |= _BIT[c]
        top = _STRAIGHT[sm]
        if top >= 0:
            return _C8 | top << 16
        return _C5 | _TOP5[sm]

    if m4:
        q = _HIGH[m4]
        return _C7 | q << 16 | _HIGH[m1 ^ (1 << q)] << 12
    if m3:
        t = _HIGH[m3]
        tb = 1 << t
        rest = m2 ^ tb
        if rest:
            return _C6 | t << 16 | _HIGH[rest] << 12
    top = _STRAIGHT[m1]
    if top >= 0:
        return _C4 | top << 16
    if m3:
        return _C3 | t << 16 | _TOP2[m1 ^ tb] << 8
    if m2:
        p1 = _HIGH[m2]
        rest = m2 ^ (1 << p1)
        if rest:
            p2 = _HIGH[rest]
            return _C2 | p1 << 16 | p2 << 12 | _HIGH[m1 ^ (1 << p1) ^ (1 << p2)] << 8
        return _C1 | p1 << 16 | _TOP3[m1 ^ (1 << p1)] << 4
    return _TOP5[m1]


def rank_mask(cards: Iterable[int]) -> int:
    """13-bit mask of the ranks present among card integers (bit `r` for rank index `r`)."""
    mask = 0
    for c in cards:
        mask |= _BIT[c]
    return mask


def straight_top(mask: int) -> int:
    """Top rank index of the best straight in a 13-bit rank mask (`3` for the wheel), or
    `-1` if there is none."""
    return _STRAIGHT[mask & 0x1FFF]


def _coerce(cards: str | Iterable[CardLike]) -> list[int]:
    out = parse_cards(cards)
    if not 5 <= len(out) <= 7:
        raise InvalidCard(f"Expected 5, 6 or 7 cards, got {len(out)}.")
    return out


def evaluate(cards: str | Iterable[CardLike]) -> int:
    """Hand value of the best five of `cards` (5, 6 or 7 distinct cards). Higher is better."""
    return evaluate_unchecked(_coerce(cards))


def category_index(value: int) -> int:
    """Category `0..8` (`HIGH_CARD` .. `STRAIGHT_FLUSH`) of a hand value."""
    cat = value >> 20
    if not 0 <= cat <= STRAIGHT_FLUSH:
        raise ValueError(f"{value} is not a hand value.")
    return cat


def category(value: int) -> str:
    """Category name of a hand value, e.g. `"Full House"`; the ace-high straight flush is
    `"Royal Flush"`."""
    cat = category_index(value)
    if cat == STRAIGHT_FLUSH and (value >> 16 & 0xF) == _ACE:
        return ROYAL_FLUSH_NAME
    return CATEGORY_NAMES[cat]


def describe(value_or_cards: int | str | Iterable[CardLike]) -> str:
    """Human description of a hand value or of 5-7 cards.

    Examples: `"High Card, Ace high"`, `"One Pair, Jacks"`, `"Two Pair, Kings and
    Sevens"`, `"Three of a Kind, Nines"`, `"Straight, Five high"` (the wheel),
    `"Flush, Ace high"`, `"Full House, Queens full of Fours"`, `"Four of a Kind, Aces"`,
    `"Straight Flush, Nine high"`, `"Royal Flush"`.
    """
    if isinstance(value_or_cards, int) and not isinstance(value_or_cards, bool):
        value = value_or_cards
    else:
        value = evaluate(cast("str | Iterable[CardLike]", value_or_cards))
    cat = category_index(value)
    r1, r2 = value >> 16 & 0xF, value >> 12 & 0xF
    if cat == HIGH_CARD:
        return f"High Card, {RANK_NAMES[r1]} high"
    if cat == ONE_PAIR:
        return f"One Pair, {RANK_PLURALS[r1]}"
    if cat == TWO_PAIR:
        return f"Two Pair, {RANK_PLURALS[r1]} and {RANK_PLURALS[r2]}"
    if cat == THREE_OF_A_KIND:
        return f"Three of a Kind, {RANK_PLURALS[r1]}"
    if cat == STRAIGHT:
        return f"Straight, {RANK_NAMES[r1]} high"
    if cat == FLUSH:
        return f"Flush, {RANK_NAMES[r1]} high"
    if cat == FULL_HOUSE:
        return f"Full House, {RANK_PLURALS[r1]} full of {RANK_PLURALS[r2]}"
    if cat == FOUR_OF_A_KIND:
        return f"Four of a Kind, {RANK_PLURALS[r1]}"
    if r1 == _ACE:
        return ROYAL_FLUSH_NAME
    return f"Straight Flush, {RANK_NAMES[r1]} high"


def _display_order(five: list[int], value: int) -> list[int]:
    """Order five cards for display: groups by size then rank; straights high to low with
    the wheel's ace last."""
    cat = value >> 20
    if cat in (STRAIGHT, STRAIGHT_FLUSH):
        ordered = sorted(five, key=lambda c: (c >> 2, c & 3), reverse=True)
        if (value >> 16 & 0xF) == 3:  # wheel: 5 4 3 2 A
            ordered = ordered[1:] + ordered[:1]
        return ordered
    counts: dict[int, int] = {}
    for c in five:
        counts[c >> 2] = counts.get(c >> 2, 0) + 1
    return sorted(five, key=lambda c: (counts[c >> 2], c >> 2, c & 3), reverse=True)


def best_five(cards: str | Iterable[CardLike]) -> list[int]:
    """The best five of 5-7 cards, in display order (e.g. pairs before kickers)."""
    cs = _coerce(cards)
    return best_five_unchecked(cs, evaluate_unchecked(cs))


def best_five_unchecked(cards: Sequence[int], value: int) -> list[int]:
    """`best_five` of 5 to 7 distinct card integers worth `value` (their `evaluate`).

    The best five are the first five cards, in `itertools.combinations` order, worth the
    most; no combination is worth more than `value`, so the search stops at the first
    one worth that much.
    """
    best_value, best = -1, list(cards)
    for combo in combinations(cards, 5):
        v = evaluate_unchecked(combo)
        if v > best_value:
            best_value, best = v, list(combo)
            if v == value:
                break
    return _display_order(best, best_value)


def best_five_str(cards: str | Iterable[CardLike]) -> list[str]:
    """`best_five` as card strings."""
    return [card_to_str(c) for c in best_five(cards)]


__all__ = [
    "HIGH_CARD",
    "ONE_PAIR",
    "TWO_PAIR",
    "THREE_OF_A_KIND",
    "STRAIGHT",
    "FLUSH",
    "FULL_HOUSE",
    "FOUR_OF_A_KIND",
    "STRAIGHT_FLUSH",
    "CATEGORY_NAMES",
    "ROYAL_FLUSH_NAME",
    "RANK_NAMES",
    "RANK_PLURALS",
    "evaluate",
    "evaluate_unchecked",
    "category",
    "category_index",
    "describe",
    "best_five",
    "best_five_unchecked",
    "best_five_str",
    "rank_mask",
    "straight_top",
]
