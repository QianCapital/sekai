"""Playing cards for Texas hold'em.

- **Text form**: rank then suit, e.g. `"As"`, `"Td"`, `"7c"`. Ranks are `23456789TJQKA`
  (`"10"` is accepted as an alias of `"T"` and ranks are case-insensitive) and suits
  `cdhs` (clubs, diamonds, hearts, spades).
- **Integer form**: `card = rank_index * 4 + suit_index` in `[0, 52)`, with `rank_index`
  0 for a deuce up to 12 for an ace and `suit_index` the position in `"cdhs"`. So
  `2c = 0`, `2d = 1`, `2h = 2`, `2s = 3`, `3c = 4`, ..., `Ah = 50`, `As = 51`. Hence
  `card >> 2` is the rank and `card & 3` the suit.

Every function that takes cards accepts either form (and `parse_cards` also accepts
strings such as `"AsKd"`, `"As Kd"` or `"As,Kd"`); everything returns integers.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TypeAlias

import numpy as np

from sekai._types import SeedType
from sekai.envs.games.poker.errors import InvalidCard
from sekai.utils.seeding import np_random

CardLike: TypeAlias = int | str

RANKS = "23456789TJQKA"
SUITS = "cdhs"
NUM_RANKS = 13
NUM_SUITS = 4
NUM_CARDS = 52

CARD_STRINGS: tuple[str, ...] = tuple(r + s for r in RANKS for s in SUITS)
"""`CARD_STRINGS[card]` is the text form of `card`."""

_STR_TO_CARD = {text: i for i, text in enumerate(CARD_STRINGS)}


def make_card(rank: int, suit: int) -> int:
    """Card integer from a rank index (0 = deuce .. 12 = ace) and a suit index (0..3 = c d h s)."""
    if not (0 <= rank < NUM_RANKS and 0 <= suit < NUM_SUITS):
        raise InvalidCard(f"Rank {rank} / suit {suit} out of range.")
    return rank * 4 + suit


def card_rank(card: int) -> int:
    """Rank index of `card`: 0 for a deuce up to 12 for an ace."""
    return card >> 2


def card_suit(card: int) -> int:
    """Suit index of `card`: 0 clubs, 1 diamonds, 2 hearts, 3 spades."""
    return card & 3


def card_from_str(text: str) -> int:
    """`"As" -> 51`. Raises `InvalidCard` for anything else."""
    if not isinstance(text, str):
        raise InvalidCard(f"Expected a card string like 'As', got {text!r}.")
    t = text.strip()
    if len(t) == 3 and t[:2] == "10":
        t = "T" + t[2]
    if len(t) != 2:
        raise InvalidCard(f"Invalid card {text!r}: expected rank + suit, like 'As' or 'Td'.")
    card = _STR_TO_CARD.get(t[0].upper() + t[1].lower())
    if card is None:
        raise InvalidCard(f"Invalid card {text!r}: ranks are {RANKS!r} and suits are {SUITS!r}.")
    return card


def card_to_str(card: int) -> str:
    """`51 -> "As"`."""
    return CARD_STRINGS[to_card(card)]


def to_card(card: CardLike) -> int:
    """Coerce a card integer (any integer type) or card string into a validated card integer."""
    if isinstance(card, str):
        return card_from_str(card)
    if isinstance(card, (bool, np.bool_)) or not isinstance(card, (int, np.integer)):
        raise InvalidCard(f"Expected a card (int 0..51 or a string like 'As'), got {card!r}.")
    c = int(card)
    if not 0 <= c < NUM_CARDS:
        raise InvalidCard(f"Card integer {c} out of range [0, 52).")
    return c


def parse_cards(cards: str | Iterable[CardLike], *, unique: bool = True) -> list[int]:
    """Parse `"AsKd"`, `"As Kd"`, `"As,Kd"` or an iterable of cards into card integers.

    With `unique=True` (the default) duplicates raise `InvalidCard`.
    """
    if isinstance(cards, str):
        compact = "".join(ch for ch in cards if not ch.isspace() and ch not in ",;-")
        compact = compact.replace("10", "T")
        if len(compact) % 2:
            raise InvalidCard(f"Cannot split {cards!r} into two-character cards.")
        out = [card_from_str(compact[i : i + 2]) for i in range(0, len(compact), 2)]
    else:
        out = [to_card(c) for c in cards]
    if unique and len(set(out)) != len(out):
        raise InvalidCard(f"Duplicate cards in {format_cards(out)!r}.")
    return out


def cards_to_str(cards: Iterable[CardLike]) -> list[str]:
    """`[51, 46] -> ["As", "Kd"]`."""
    return [CARD_STRINGS[to_card(c)] for c in cards]


def format_cards(cards: Iterable[CardLike], sep: str = " ") -> str:
    """`[51, 46] -> "As Kd"`."""
    return sep.join(cards_to_str(cards))


def full_deck() -> list[int]:
    """All 52 cards in integer order (`2c` first, `As` last)."""
    return list(range(NUM_CARDS))


def shuffled_deck(rng: SeedType = None) -> list[int]:
    """A uniformly shuffled 52-card deck drawn with a NumPy `Generator` (or a seed)."""
    gen, _ = np_random(rng)
    deck: list[int] = gen.permutation(NUM_CARDS).tolist()
    return deck


class Deck:
    """A shuffled deck that deals from the top.

    `Deck(rng)` shuffles all 52 cards with `rng` (a seed or `numpy.random.Generator`);
    `Deck(cards=[...])` uses the given order without shuffling (index 0 is dealt first).
    """

    def __init__(self, rng: SeedType = None, *, cards: Sequence[CardLike] | None = None) -> None:
        if cards is None:
            self._cards = shuffled_deck(rng)
        else:
            self._cards = parse_cards(cards)
        self._pos = 0

    def deal(self, n: int = 1) -> list[int]:
        """Deal `n` cards off the top."""
        if n < 0 or self._pos + n > len(self._cards):
            raise InvalidCard(f"Cannot deal {n} cards: only {len(self)} remain.")
        out = self._cards[self._pos : self._pos + n]
        self._pos += n
        return out

    def remaining(self) -> list[int]:
        """The undealt cards, top first."""
        return self._cards[self._pos :]

    def __len__(self) -> int:
        return len(self._cards) - self._pos

    def __repr__(self) -> str:
        return f"Deck({len(self)} cards remaining)"


__all__ = [
    "RANKS",
    "SUITS",
    "NUM_RANKS",
    "NUM_SUITS",
    "NUM_CARDS",
    "CARD_STRINGS",
    "CardLike",
    "make_card",
    "card_rank",
    "card_suit",
    "card_from_str",
    "card_to_str",
    "to_card",
    "parse_cards",
    "cards_to_str",
    "format_cards",
    "full_deck",
    "shuffled_deck",
    "Deck",
]
