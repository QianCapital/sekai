"""Cards, card values and hand totals.

A card is a two-character string, rank then suit: `"Ah"`, `"Td"`, `"Qs"`. Ranks are
`A23456789TJQK` and suits `cdhs` (clubs, diamonds, hearts, spades). Everything public in
this package (table views, events) uses this text form, so it serialises to JSON as is.

For counting only a card's **value** matters: ace `1` (counted as 11 when that does not
bust the hand), `2`-`9` face value, `T J Q K` all `10`. Composition vectors are indexed by
`value - 1`, so index `0` is aces and index `9` is all ten-valued cards (four ranks per
deck, sixteen cards).

`card_value` and `hand_value` accept a looser *card-like*: a card string (`"Ah"`), a bare
rank (`"A"`, `"T"`, `"K"`, also `"10"`) or an `int` value `1`-`10` (`11` is accepted as an
ace).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

RANKS = "A23456789TJQK"
SUITS = "cdhs"
NUM_VALUES = 10
"""Distinct card values, ace (1) through ten-valued (10)."""

CARDS_PER_DECK = 52

RANK_VALUE: dict[str, int] = {r: min(i + 1, 10) for i, r in enumerate(RANKS)}
"""Blackjack value of each rank, ace as 1."""

DECK: tuple[str, ...] = tuple(r + s for r in RANKS for s in SUITS)
"""One 52-card deck in a fixed (unshuffled) order."""

VALUE_COUNTS_PER_DECK: tuple[int, ...] = (4, 4, 4, 4, 4, 4, 4, 4, 4, 16)
"""Cards of each value (ace..ten-valued) in one deck."""

CardLike = str | int


def is_card(text: str) -> bool:
    """True if `text` is a valid two-character card such as `"Ah"`."""
    return len(text) == 2 and text[0] in RANK_VALUE and text[1] in SUITS


def parse_card(text: str) -> str:
    """Validate and normalise a card string (`"ah"` -> `"Ah"`, `"10h"` -> `"Th"`)."""
    if not isinstance(text, str):
        raise ValueError(f"A card must be a string like 'Ah', got {text!r}.")
    t = text.strip()
    if len(t) == 3 and t[:2] == "10":
        t = "T" + t[2]
    if len(t) != 2:
        raise ValueError(f"Invalid card {text!r}: expected rank + suit, e.g. 'Ah', 'Td'.")
    card = t[0].upper() + t[1].lower()
    if not is_card(card):
        raise ValueError(f"Invalid card {text!r}: ranks are {RANKS}, suits are {SUITS}.")
    return card


_FAST_VALUE: dict[CardLike, int] = {c: RANK_VALUE[c[0]] for c in DECK}
_FAST_VALUE.update(RANK_VALUE.items())
_FAST_VALUE.update((v, v) for v in range(1, 11))
_FAST_VALUE[11] = 1


def card_value(card: CardLike) -> int:
    """Blackjack value `1`-`10` of a card-like (ace is `1`)."""
    if card.__class__ is not bool:
        fast = _FAST_VALUE.get(card)
        if fast is not None:
            return fast
    if isinstance(card, bool):
        raise ValueError(f"Invalid card {card!r}.")
    if isinstance(card, int):
        if card == 11:
            return 1
        if 1 <= card <= 10:
            return card
        raise ValueError(f"Card value must be 1-10 (or 11 for an ace), got {card}.")
    if isinstance(card, str):
        t = card.strip()
        if t[:2] == "10":
            return 10
        if t:
            v = RANK_VALUE.get(t[0].upper())
            if v is not None and (len(t) == 1 or (len(t) == 2 and t[1].lower() in SUITS)):
                return v
    raise ValueError(f"Invalid card {card!r}: use 'Ah', a rank like 'T', or a value 1-10.")


def card_values(cards: Iterable[CardLike]) -> list[int]:
    """`card_value` of each card."""
    return [card_value(c) for c in cards]


def total_of_values(values: Iterable[int]) -> tuple[int, bool]:
    """`(total, soft)` for card values (ace = 1). `soft` means an ace counts as 11."""
    total = 0
    has_ace = False
    for v in values:
        total += v
        if v == 1:
            has_ace = True
    if has_ace and total + 10 <= 21:
        return total + 10, True
    return total, False


def hand_value(cards: Iterable[CardLike]) -> tuple[int, bool]:
    """Best blackjack total of `cards` and whether it is soft.

    `hand_value(["Ah", "6d"]) == (17, True)`, `hand_value(["Ah", "6d", "Tc"]) == (17, False)`,
    `hand_value([]) == (0, False)`. Totals above 21 are returned as is (a bust).
    """
    return total_of_values(card_value(c) for c in cards)


def is_natural(cards: Sequence[CardLike]) -> bool:
    """Two cards totalling 21 (an ace and a ten-valued card). Split hands never qualify."""
    if len(cards) != 2:
        return False
    a, b = card_value(cards[0]), card_value(cards[1])
    return {a, b} == {1, 10}


def value_label(value: int) -> str:
    """Display label of a value: `"A"`, `"2"` ... `"9"`, `"T"`."""
    return "A" if value == 1 else ("T" if value == 10 else str(value))


# Table internals: cards that are known to be valid two-character strings.


def _total(cards: Iterable[str]) -> tuple[int, bool]:
    """`hand_value` of valid card strings."""
    total = 0
    has_ace = False
    for c in cards:
        v = RANK_VALUE[c[0]]
        total += v
        if v == 1:
            has_ace = True
    if has_ace and total + 10 <= 21:
        return total + 10, True
    return total, False


def _natural(cards: Sequence[str]) -> bool:
    """`is_natural` of valid card strings."""
    if len(cards) != 2:
        return False
    a, b = RANK_VALUE[cards[0][0]], RANK_VALUE[cards[1][0]]
    return a + b == 11 and (a == 1 or b == 1)


__all__ = [
    "RANKS",
    "SUITS",
    "NUM_VALUES",
    "CARDS_PER_DECK",
    "RANK_VALUE",
    "DECK",
    "VALUE_COUNTS_PER_DECK",
    "CardLike",
    "is_card",
    "parse_card",
    "card_value",
    "card_values",
    "total_of_values",
    "hand_value",
    "is_natural",
    "value_label",
]
