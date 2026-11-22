"""Table rules.

`Rules` defaults to a common US shoe game: six decks, dealer stands on soft 17, blackjack
pays 3:2, double on any two cards including after a split, resplit to four hands (but not
aces), split aces get one card each, late surrender, and the dealer peeks for blackjack
under an ace or ten-valued up card.

Details that are fixed rather than configurable:

- **No insurance and no even money.** The dealer peeks silently; a player blackjack
  against a dealer ace is simply paid 3:2 once the peek shows no dealer blackjack.
- Pairs split **by value**, so `K` + `T` is a pair. A two-card 21 after a split is an
  ordinary 21, not a blackjack.
- **Doubling** is allowed on any first two cards (after a split only with
  `double_after_split`) and costs exactly the hand's bet: no doubling for less.
- **Surrender** (`"late"`) is only available as the first decision on the original
  two-card hand, never after a split, and forfeits half the bet.
- `dealer_peeks=False` is European no-hole-card play: the dealer still gets a second card
  face down but does not check it until the players are done, and a dealer blackjack then
  takes **everything** wagered, doubles and splits included. A player blackjack against an
  ace or ten-valued up card is settled only at the end (push against a dealer blackjack).
- Doubling and splitting need the extra chips in the seat's bankroll.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from fractions import Fraction
from typing import Any, Literal

from sekai.envs.games.blackjack.cards import CARDS_PER_DECK

SurrenderRule = Literal["late"] | None


@dataclass(frozen=True, slots=True)
class Rules:
    """The rules of a blackjack table (see the module docstring for the fixed ones).

    Attributes:
        decks: Decks in the shoe, 1 to 16.
        dealer_hits_soft_17: `False`: the dealer stands on all 17s (S17). `True`: the
            dealer hits soft 17 (H17).
        blackjack_payout: Net win on a natural per unit bet (`1.5` is 3:2, `1.2` is 6:5).
        double_after_split: Doubling is allowed on split hands.
        max_hands: Most hands a seat may hold after splitting (`4` = resplit to four hands).
        resplit_aces: Split aces may be split again.
        hit_split_aces: `False`: each split ace receives exactly one card and the hand is
            finished.
        surrender: `"late"` or `None` (no surrender).
        dealer_peeks: US hole-card game: the dealer checks for blackjack under an ace or
            ten-valued card.
        penetration: Reshuffle before a round once this fraction of the shoe has been dealt.
        min_bet: The table minimum.
        max_bet: The table maximum.
        bet_increment: Bets are multiples of this.
    """

    decks: int = 6
    dealer_hits_soft_17: bool = False
    blackjack_payout: float = 1.5
    double_after_split: bool = True
    max_hands: int = 4
    resplit_aces: bool = False
    hit_split_aces: bool = False
    surrender: SurrenderRule = "late"
    dealer_peeks: bool = True
    penetration: float = 0.75
    min_bet: int = 10
    max_bet: int = 500
    bet_increment: int = 5

    def __post_init__(self) -> None:
        if not isinstance(self.decks, int) or not 1 <= self.decks <= 16:
            raise ValueError(f"decks must be an int in [1, 16], got {self.decks!r}.")
        if not 1.0 <= float(self.blackjack_payout) <= 3.0:
            raise ValueError(f"blackjack_payout must be in [1, 3], got {self.blackjack_payout}.")
        if not isinstance(self.max_hands, int) or not 1 <= self.max_hands <= 8:
            raise ValueError(f"max_hands must be an int in [1, 8], got {self.max_hands!r}.")
        if self.surrender not in ("late", None):
            raise ValueError(f"surrender must be 'late' or None, got {self.surrender!r}.")
        if not 0.1 <= float(self.penetration) <= 0.95:
            raise ValueError(f"penetration must be in [0.1, 0.95], got {self.penetration}.")
        for name in ("min_bet", "max_bet", "bet_increment"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise ValueError(f"{name} must be a positive int, got {v!r}.")
        if self.max_bet < self.min_bet:
            raise ValueError(f"max_bet ({self.max_bet}) is below min_bet ({self.min_bet}).")
        if self.min_bet % self.bet_increment or self.max_bet % self.bet_increment:
            raise ValueError(
                f"min_bet ({self.min_bet}) and max_bet ({self.max_bet}) must be multiples of "
                f"bet_increment ({self.bet_increment})."
            )

    @property
    def total_cards(self) -> int:
        """Cards in a full shoe."""
        return self.decks * CARDS_PER_DECK

    @property
    def blackjack_ratio(self) -> Fraction:
        """`blackjack_payout` as an exact fraction (`1.5` -> `3/2`, `1.2` -> `6/5`)."""
        return Fraction(str(self.blackjack_payout)).limit_denominator(1000)

    @property
    def label(self) -> str:
        """Short rules summary, e.g. `"6D S17 DAS LS 3:2"`."""
        ratio = self.blackjack_ratio
        parts = [
            f"{self.decks}D",
            "H17" if self.dealer_hits_soft_17 else "S17",
            "DAS" if self.double_after_split else "NDAS",
        ]
        if self.surrender == "late":
            parts.append("LS")
        if self.resplit_aces:
            parts.append("RSA")
        if not self.dealer_peeks:
            parts.append("ENHC")
        parts.append(f"{ratio.numerator}:{ratio.denominator}")
        return " ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict of every field (round-trips through `from_dict`)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Rules:
        """Inverse of `to_dict`; unknown keys are ignored."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


__all__ = ["Rules", "SurrenderRule"]
