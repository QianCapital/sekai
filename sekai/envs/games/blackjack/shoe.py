"""The dealing shoe, and the public card-counting state derived from it.

`Shoe` holds `decks` x 52 cards shuffled with a NumPy `Generator`. Besides dealing it
tracks which cards have been **seen** since the last shuffle: every card dealt face up, and
a face-down card (the dealer's hole card) once `Shoe.reveal` is called. Everything a
counter could know follows from that:

- `Shoe.public_remaining_counts` is the full shoe composition minus the seen cards, as
  counts per value (index `0` = aces ... index `9` = ten-valued). An unrevealed hole card
  is *not* seen, so it still counts as remaining.
- `Shoe.running_count` / `Shoe.true_count` are the Hi-Lo counts of the seen cards.

The order of the undealt cards is private. Nothing public in this package (views, events)
ever includes it, and `repr(shoe)` only shows counts.

If the shoe runs dry in the middle of a round (only possible with few decks and deep
penetration), the table calls `Shoe.shuffle` with the cards still on the table, which are
left out of the new shoe, as a casino reshuffles its discards.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import numpy as np

from sekai._types import SeedType
from sekai.envs.games.blackjack.cards import (
    CARDS_PER_DECK,
    DECK,
    NUM_VALUES,
    RANK_VALUE,
    VALUE_COUNTS_PER_DECK,
    parse_card,
)
from sekai.utils.seeding import np_random

HILO_TAGS: tuple[int, ...] = (-1, 1, 1, 1, 1, 1, 0, 0, 0, -1)
"""Hi-Lo tag per value index: aces and tens -1, 2-6 +1, 7-9 0."""

MIN_DECKS_REMAINING = 0.5
"""True-count divisor floor (in decks), so the count stays finite at the end of a shoe."""

# Hi-Lo tag of each rank.
_RANK_TAG = {rank: HILO_TAGS[value - 1] for rank, value in RANK_VALUE.items()}


def true_count_from(running: float, cards_remaining: float) -> float:
    """Running count divided by the decks remaining (floored at half a deck)."""
    decks_left = max(cards_remaining / CARDS_PER_DECK, MIN_DECKS_REMAINING)
    return float(running) / decks_left


class Shoe:
    """A multi-deck shoe.

    Args:
        decks: Number of 52-card decks.
        rng: A NumPy `Generator` (shared with the caller) or an int seed.
        penetration: `needs_shuffle` turns true once this fraction has been dealt.
        seed: Alternative to `rng`.
        rng_factory: If given, called for a fresh `Generator` before every shuffle, so no
            two shoes share a random stream (a public table can then not be predicted from
            the shoes it has already dealt). `rng` is then ignored.
    """

    def __init__(
        self,
        decks: int = 6,
        rng: SeedType = None,
        *,
        penetration: float = 0.75,
        seed: int | None = None,
        rng_factory: Callable[[], np.random.Generator] | None = None,
    ) -> None:
        if not isinstance(decks, int) or decks < 1:
            raise ValueError(f"decks must be a positive int, got {decks!r}.")
        self.decks = decks
        self.penetration = float(penetration)
        self._rng_factory = rng_factory
        self._rng, _ = np_random(rng if rng is not None else seed)
        self._full: tuple[str, ...] = DECK * decks
        self._full_counts: tuple[int, ...] = tuple(n * decks for n in VALUE_COUNTS_PER_DECK)
        self._cards: list[str] = []  # undealt cards; the next card is the *last* element
        self._seen = [0] * NUM_VALUES
        self._seen_total = 0
        self._running = 0
        self._face_down: list[str] = []
        self.shuffles = 0
        self.shuffle()

    # ------------------------------------------------------------------
    # Dealing
    # ------------------------------------------------------------------

    def shuffle(self, in_play: Iterable[str] = (), visible: Iterable[str] = ()) -> None:
        """Gather every card and shuffle, forgetting what has been seen.

        `in_play` are cards still on the table, which stay out of the new shoe; `visible`
        are those of them that are face up, which stay seen. `in_play` cards that are not
        `visible` stay face down (they can be revealed later).
        """
        pool = list(self._full)
        hidden = list(in_play)
        for card in hidden:
            pool.remove(card)
        if self._rng_factory is not None:
            self._rng = self._rng_factory()
        order = self._rng.permutation(len(pool))
        self._cards = [pool[i] for i in order.tolist()]
        self._seen = [0] * NUM_VALUES
        self._seen_total = 0
        self._running = 0
        shown = list(visible)
        for card in shown:
            self._mark_seen(card)
        for card in shown:
            hidden.remove(card)
        self._face_down = hidden
        self.shuffles += 1

    def draw(self, face_up: bool = True) -> str:
        """Deal the next card. Face-down cards are not seen until `reveal`."""
        if not self._cards:
            raise RuntimeError("The shoe is empty; shuffle() before drawing.")
        card = self._cards.pop()
        if face_up:
            self._mark_seen(card)
        else:
            self._face_down.append(card)
        return card

    def reveal(self, card: str) -> None:
        """Turn a face-down card up, making it seen."""
        try:
            self._face_down.remove(card)
        except ValueError:
            raise ValueError(f"{card!r} is not a face-down card dealt from this shoe.") from None
        self._mark_seen(card)

    def stack_top(self, cards: Sequence[str]) -> None:
        """Put `cards` on top of the shoe, `cards[0]` dealt first (tests and demos only).

        Each card is taken out of the undealt cards first, so the shoe's composition stays a
        real one; raises `ValueError` if a card is no longer available.
        """
        parsed = [parse_card(c) for c in cards]
        for card in parsed:
            try:
                self._cards.remove(card)
            except ValueError:
                raise ValueError(f"No undealt {card!r} left to stack.") from None
        self._cards.extend(reversed(parsed))

    def _mark_seen(self, card: str) -> None:
        self._seen[RANK_VALUE[card[0]] - 1] += 1
        self._seen_total += 1
        self._running += _RANK_TAG[card[0]]

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------

    @property
    def total(self) -> int:
        """Cards in a full shoe."""
        return len(self._full)

    @property
    def remaining(self) -> int:
        """Undealt cards left in the shoe."""
        return len(self._cards)

    @property
    def dealt_fraction(self) -> float:
        """Fraction of the shoe dealt since the last shuffle."""
        return (len(self._full) - len(self._cards)) / len(self._full)

    @property
    def needs_shuffle(self) -> bool:
        """True once the dealt fraction has reached the penetration."""
        return self.dealt_fraction >= self.penetration

    @property
    def seen_total(self) -> int:
        """Cards seen since the last shuffle."""
        return self._seen_total

    def seen_counts(self) -> list[int]:
        """Cards seen since the last shuffle, per value (index 0 = aces)."""
        return list(self._seen)

    def full_counts(self) -> list[int]:
        """Composition of a full shoe, per value."""
        return list(self._full_counts)

    def public_remaining_counts(self) -> list[int]:
        """Full composition minus the seen cards, per value (an unseen hole card counts)."""
        return [f - s for f, s in zip(self._full_counts, self._seen)]

    @property
    def unseen_total(self) -> int:
        """Cards not seen since the last shuffle (an unrevealed hole card included)."""
        return len(self._full) - self._seen_total

    @property
    def running_count(self) -> int:
        """Hi-Lo running count of the seen cards."""
        return self._running

    @property
    def true_count(self) -> float:
        """Running count per deck of unseen cards (divisor floored at half a deck)."""
        return true_count_from(self._running, len(self._full) - self._seen_total)

    def __repr__(self) -> str:
        return (
            f"Shoe(decks={self.decks}, remaining={self.remaining}/{self.total}, "
            f"running_count={self.running_count})"
        )


__all__ = ["Shoe", "HILO_TAGS", "MIN_DECKS_REMAINING", "true_count_from"]
