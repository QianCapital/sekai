"""The blackjack table: seats, betting, dealing, play, the dealer and settlement.

`BlackjackTable` runs a multi-seat table against one dealer from one shoe, exactly as a
casino deals it. It is driven by text moves for whichever seat is to act:

- `"bet:<amount>"` during the betting phase (a multiple of `bet_increment` between
  `min_bet` and `min(max_bet, bankroll)`). Bets come off the bankroll when placed.
- `"hit"`, `"stand"`, `"double"`, `"split"`, `"surrender"` while playing.

Round flow:

1. **Betting.** Seats whose bankroll covers `min_bet` are active; the others are *out*
   and skipped. Active seats bet in seat order.
2. **Deal.** When the last bet is in, the shoe is reshuffled if the dealt fraction has
   reached the penetration (a `shuffle` event), then cards go out in casino order: one to
   each seat, the dealer's up card, a second to each seat, the dealer's hole card face down.
3. **Peek.** With `dealer_peeks` and an ace or ten-valued up card the dealer checks for
   blackjack (a `peek` event saying only whether it is there). A dealer blackjack is
   revealed and settled at once: player blackjacks push, every other hand loses, nobody
   acts. Otherwise player naturals are paid 3:2 immediately and skip play.
4. **Play.** Seats act in order, hand by hand. A split hand gets its second card when
   play reaches it; split aces get one card each and are finished (`done`); a hand
   reaching 21 stands automatically; a double receives one card and is `done`. Busts and
   surrenders settle immediately, except a surrender against an ace or ten-valued up card
   without the peek, which waits for the dealer's hole card: a dealer blackjack takes the
   whole bet.
5. **Dealer.** If any hand still needs the dealer, the hole card is revealed and the
   dealer draws to 17 (hitting soft 17 under H17); otherwise the dealer only reveals.
6. **Settle.** Remaining hands are settled and paid, a round summary is appended to
   `history`, and the table returns to betting. The finished hands stay in the view until
   the next deal; `last_round` keeps the summary.

Hidden information: the shoe order and the unrevealed hole card never appear in `view()`,
`events`, `legal_actions()` or any other public accessor: the hole card is `None` until it
is turned over, and the dealer's `total` counts visible cards only. Counting state
(`running_count`, `true_count`, `public_remaining_counts()`) is derived from seen cards
only.

Accounting: chips are conserved: `sum(bankrolls) + chips_on_table + house_net == chips_in`
at every moment, where `chips_in` is the total bankroll ever given to the seats (initial
bankrolls plus `set_bankroll` adjustments). Each settled hand records its net `payout`:
`+bet` for a win (the doubled bet if doubled), `-bet` for a loss, `0` for a push,
`+bet * blackjack_payout` for a natural and `-bet / 2` for a surrender.

Events: `events` is a JSON-friendly ordered log. Every event has `seq` (increasing),
`round` and `type`; the other keys by type:

| type | fields |
|---|---|
| `shuffle` | `reason`: `"new_shoe"`, `"penetration"` or `"empty"` (mid-round) |
| `bankroll` | `seat`, `amount` (bankroll set by `set_bankroll`) |
| `bet` | `seat`, `amount` |
| `deal` | `to` (a seat or `"dealer"`), `hand` (seats only), `card` (`None`: the hole card) |
| `peek` | `blackjack` (whether the dealer's peek found one) |
| `action` | `seat`, `hand`, `action` |
| `dealer_reveal` | `card`, `cards`, `total`, `blackjack` |
| `dealer_draw` | `card`, `total` |
| `settle` | `seat`, `hand`, `outcome`, `payout` |
| `round_end` | `net` (per seat, `None` for seats that sat out), `dealer_total` |

The log keeps the most recent `max_events` entries; `events_since` returns the ones after a
given `seq`, for clients that poll.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Literal

import numpy as np

from sekai._types import SeedType
from sekai.envs.games.blackjack.actions import ACTIONS
from sekai.envs.games.blackjack.cards import RANK_VALUE, _natural, _total
from sekai.envs.games.blackjack.errors import BlackjackError, IllegalAction, NotYourTurn
from sekai.envs.games.blackjack.rules import Rules
from sekai.envs.games.blackjack.shoe import Shoe
from sekai.utils.seeding import np_random

Phase = Literal["betting", "playing"]
HandStatus = Literal["playing", "stood", "bust", "blackjack", "surrendered", "done"]
Outcome = Literal["win", "lose", "push", "blackjack", "surrender"]

PLAYING_ACTIONS: tuple[str, ...] = ACTIONS
"""`("hit", "stand", "double", "split", "surrender")`, the order of legal-action lists."""


def _num(x: float) -> int | float:
    """Chip amounts as `int` when whole, so JSON shows `1000` rather than `1000.0`."""
    f = float(x)
    return int(f) if f.is_integer() else f


@dataclass(slots=True)
class Hand:
    """One player hand. Internal state; `BlackjackTable.view` exposes it as a dict."""

    cards: list[str]
    bet: int
    doubled: bool = False
    from_split: bool = False
    split_aces: bool = False
    natural: bool = False
    status: str = "playing"
    outcome: str | None = None
    payout: float | None = None

    def value(self) -> tuple[int, bool]:
        """The hand's `(total, soft)`."""
        return _total(self.cards)

    def to_dict(self) -> dict[str, Any]:
        """The hand as the table's view shows it."""
        total, soft = _total(self.cards)
        return {
            "cards": list(self.cards),
            "total": total,
            "soft": soft,
            "blackjack": self.natural,
            "bust": total > 21,
            "doubled": self.doubled,
            "from_split": self.from_split,
            "bet": self.bet,
            "status": self.status,
            "outcome": self.outcome,
            "payout": None if self.payout is None else _num(self.payout),
        }


@dataclass(slots=True)
class _Seat:
    bankroll: float
    bet: int | None = None
    """Bet placed for the current round (`None` until placed)."""
    in_round: bool = False
    hands: list[Hand] = field(default_factory=list)


def _copy_round(summary: dict[str, Any]) -> dict[str, Any]:
    """A deep copy of a round summary (the structure `_finish_round` builds).

    Every dict and list is new; the other values are scalars and strings.
    """
    dealer = summary["dealer"]
    return {
        "round": summary["round"],
        "shuffled": summary["shuffled"],
        "dealer": {**dealer, "cards": list(dealer["cards"])},
        "seats": [
            {**s, "hands": [{**h, "cards": list(h["cards"])} for h in s["hands"]]}
            for s in summary["seats"]
        ],
    }


class BlackjackTable:
    """A blackjack table with `seats` players against the dealer.

    Args:
        rules: Table rules.
        seats: Number of player seats.
        bankroll: Starting bankroll of every seat, or one value per seat.
        rng: NumPy `Generator` or int seed for the shoe; `seed` is an alternative.
        seed: Int seed for the shoe, if `rng` is not given.
        max_events: How many recent events `events` keeps.
        max_history: How many round summaries `history` keeps.
        shoe: An existing `Shoe` to deal from (tests and demos); it must have `rules.decks`
            decks.
        rng_factory: Passed to the `Shoe`: a fresh `Generator` for every shuffle.
    """

    def __init__(
        self,
        rules: Rules = Rules(),  # noqa: B008 - a frozen dataclass
        seats: int = 2,
        bankroll: float | Sequence[float] = 1000,
        rng: SeedType = None,
        seed: int | None = None,
        *,
        max_events: int | None = 5000,
        max_history: int | None = 1000,
        shoe: Shoe | None = None,
        rng_factory: Callable[[], np.random.Generator] | None = None,
    ) -> None:
        if not isinstance(seats, int) or seats < 1:
            raise ValueError(f"seats must be a positive int, got {seats!r}.")
        if isinstance(bankroll, (int, float)):
            bankrolls = [float(bankroll)] * seats
        else:
            bankrolls = [float(b) for b in bankroll]
            if len(bankrolls) != seats:
                raise ValueError(f"Got {len(bankrolls)} bankrolls for {seats} seats.")
        if any(b < 0 for b in bankrolls):
            raise ValueError("Bankrolls must be non-negative.")
        self.rules = rules
        self._rng, self.seed = np_random(rng if rng is not None else seed)
        if shoe is not None and shoe.decks != rules.decks:
            raise ValueError(f"The shoe has {shoe.decks} decks but the rules say {rules.decks}.")
        self._shoe = (
            shoe
            if shoe is not None
            else Shoe(
                rules.decks, self._rng, penetration=rules.penetration, rng_factory=rng_factory
            )
        )
        self._shoe.penetration = rules.penetration
        self._ratio = rules.blackjack_ratio
        self._rules_dict = rules.to_dict()
        self._seats = [_Seat(b) for b in bankrolls]
        self._dealer: list[str] = []
        self._hole_revealed = False
        self._phase: Phase = "betting"
        self._round_no = 1
        self._rounds_played = 0
        self._to_act: int | None = None
        self._active_hand: int | None = None
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._seq = 0
        self._history: deque[dict[str, Any]] = deque(maxlen=max_history)
        self._last_round: dict[str, Any] | None = None
        self._shuffled_this_round = False
        self.chips_in = float(sum(bankrolls))
        self.house_net = 0.0
        self._on_table = 0.0
        self._log("shuffle", reason="new_shoe")
        self._start_betting()

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def num_seats(self) -> int:
        """Number of player seats."""
        return len(self._seats)

    @property
    def phase(self) -> Phase:
        """`"betting"` or `"playing"`."""
        return self._phase

    @property
    def round_no(self) -> int:
        """Number of the current round (betting included), starting at 1."""
        return self._round_no

    @property
    def rounds_played(self) -> int:
        """Rounds finished so far."""
        return self._rounds_played

    @property
    def to_act(self) -> int | None:
        """Seat whose decision the table is waiting for, or `None` (every seat is out)."""
        return self._to_act

    @property
    def active_hand(self) -> int | None:
        """Index into the acting seat's hands while playing, else `None`."""
        return self._active_hand

    @property
    def game_over(self) -> bool:
        """True when no seat can cover the minimum bet."""
        return self._phase == "betting" and self._to_act is None

    @property
    def bankrolls(self) -> list[int | float]:
        """Every seat's bankroll (chips not on the table)."""
        return [_num(s.bankroll) for s in self._seats]

    @property
    def chips_on_table(self) -> int | float:
        """Chips currently wagered and not yet settled."""
        return _num(self._on_table)

    @property
    def events(self) -> list[dict[str, Any]]:
        """The event log, oldest first (see the module docstring)."""
        return [dict(e) for e in self._events]

    def events_since(self, seq: int) -> list[dict[str, Any]]:
        """Events with `seq` greater than `seq` (`-1` for all that are kept)."""
        return [dict(e) for e in self._events if e["seq"] > seq]

    @property
    def history(self) -> list[dict[str, Any]]:
        """Summaries of the most recent finished rounds, oldest first."""
        return [_copy_round(r) for r in self._history]

    @property
    def last_round(self) -> dict[str, Any] | None:
        """Summary of the most recently finished round."""
        return None if self._last_round is None else _copy_round(self._last_round)

    @property
    def running_count(self) -> int:
        """Hi-Lo running count of the cards seen since the last shuffle."""
        return self._shoe.running_count

    @property
    def true_count(self) -> float:
        """Running count per deck of unseen cards (divisor floored at half a deck)."""
        return self._shoe.true_count

    def public_remaining_counts(self) -> list[int]:
        """Unseen cards per value (index 0 = aces ... 9 = tens); the hole card counts."""
        return self._shoe.public_remaining_counts()

    def is_out(self, seat: int) -> bool:
        """True if `seat` is not part of the current round (or cannot cover the minimum)."""
        s = self._seats[seat]
        if self._phase == "playing":
            return not s.in_round
        return s.bet is None and s.bankroll < self.rules.min_bet

    def bet_limits(self, seat: int) -> tuple[int, int] | None:
        """`(min, max)` bet `seat` may place now, or `None` if it cannot cover the minimum."""
        r = self.rules
        cap = min(r.max_bet, int(self._seats[seat].bankroll // r.bet_increment) * r.bet_increment)
        if cap < r.min_bet:
            return None
        return r.min_bet, cap

    def legal_actions(self) -> dict[str, Any] | list[str]:
        """What the seat to act may do.

        Betting: `{"kind": "bet", "min": .., "max": .., "step": ..}`. Playing: a list in the
        order `hit stand double split surrender`. `[]` when nobody can act.
        """
        seat = self._to_act
        if seat is None:
            return []
        if self._phase == "betting":
            limits = self.bet_limits(seat)
            assert limits is not None
            return {
                "kind": "bet",
                "min": limits[0],
                "max": limits[1],
                "step": self.rules.bet_increment,
            }
        assert self._active_hand is not None
        return self._hand_actions(self._seats[seat], self._active_hand)

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def act(self, action: str, seat: int | None = None) -> None:
        """Apply `action` for the seat to act (`seat`, if given, must be that seat)."""
        if seat is not None and seat != self._to_act:
            raise NotYourTurn(seat, self._to_act)
        if self._to_act is None:
            raise IllegalAction(action, "no seat can act (every seat is out of chips)")
        if not isinstance(action, str):
            raise IllegalAction(action, "actions are strings such as 'hit' or 'bet:20'")
        move = action.strip().lower()
        if self._phase == "betting":
            self._place_bet(action, move)
        else:
            self._play(action, move)

    def set_bankroll(self, seat: int, amount: float) -> None:
        """Set a seat's bankroll between rounds (a rebuy or cash-out).

        Only during betting, before the seat has bet. The change is added to `chips_in`, so
        the accounting invariant still holds.
        """
        if self._phase != "betting":
            raise IllegalAction("set_bankroll", "bankrolls can only change between rounds")
        s = self._seats[seat]
        if s.bet is not None:
            raise IllegalAction("set_bankroll", f"seat {seat} has already bet this round")
        if amount < 0:
            raise ValueError("A bankroll cannot be negative.")
        self.chips_in += float(amount) - s.bankroll
        s.bankroll = float(amount)
        self._log("bankroll", seat=seat, amount=_num(amount))
        self._to_act = self._next_bettor()

    # ------------------------------------------------------------------
    # The public view
    # ------------------------------------------------------------------

    def view(self, seat: int | None = None) -> dict[str, Any]:
        """JSON-friendly public state of the table, as any spectator would see it.

        `seat` marks the viewer (`"viewer"`); blackjack has no per-seat secrets, so the
        content is otherwise the same for everyone. Never contains the shoe order or an
        unrevealed hole card.
        """
        shoe = self._shoe
        return {
            "phase": self._phase,
            "round_no": self._round_no,
            "rounds_played": self._rounds_played,
            "to_act": self._to_act,
            "active_hand": self._active_hand,
            "viewer": seat,
            "legal_actions": self.legal_actions(),
            "dealer": self._dealer_view(),
            "seats": [self._seat_view(i) for i in range(len(self._seats))],
            "shoe": {
                "decks": shoe.decks,
                "remaining": shoe.remaining,
                "total": shoe.total,
                "penetration": shoe.penetration,
                "dealt_fraction": shoe.dealt_fraction,
                "reshuffle_next": shoe.needs_shuffle,
            },
            "running_count": shoe.running_count,
            "true_count": shoe.true_count,
            "remaining_counts": shoe.public_remaining_counts(),
            "last_round": None if self._last_round is None else _copy_round(self._last_round),
            "game_over": self.game_over,
            "rules": dict(self._rules_dict),
        }

    def _dealer_view(self) -> dict[str, Any]:
        if self._hole_revealed or len(self._dealer) < 2:
            visible: list[str | None] = list(self._dealer)
            shown = self._dealer
        else:
            visible = [self._dealer[0], None, *self._dealer[2:]]
            shown = self._dealer[:1]
        total, soft = _total(shown)
        revealed = self._hole_revealed
        return {
            "cards": visible,
            "total": total,
            "soft": soft,
            "blackjack": revealed and _natural(self._dealer),
            "bust": total > 21,
            "revealed": revealed,
        }

    def _seat_view(self, i: int) -> dict[str, Any]:
        s = self._seats[i]
        return {
            "seat": i,
            "bankroll": _num(s.bankroll),
            "bet": s.bet,
            "out": self.is_out(i),
            "hands": [h.to_dict() for h in s.hands],
        }

    # ------------------------------------------------------------------
    # Betting and dealing
    # ------------------------------------------------------------------

    def _log(self, type_: str, **fields: Any) -> None:
        self._events.append({"seq": self._seq, "round": self._round_no, "type": type_, **fields})
        self._seq += 1

    def _next_bettor(self) -> int | None:
        for i, s in enumerate(self._seats):
            if s.bet is None and s.bankroll >= self.rules.min_bet:
                return i
        return None

    def _start_betting(self) -> None:
        self._phase = "betting"
        self._active_hand = None
        for s in self._seats:
            s.bet = None
        self._to_act = self._next_bettor()

    def _place_bet(self, raw: str, move: str) -> None:
        seat = self._to_act
        assert seat is not None
        legal = self.legal_actions()
        assert isinstance(legal, dict)
        if not move.startswith("bet:"):
            raise IllegalAction(raw, "the table is taking bets; send 'bet:<amount>'", legal)
        text = move[4:].strip()
        try:
            amount_f = float(text)
        except ValueError:
            raise IllegalAction(raw, f"{text!r} is not a number", legal) from None
        if not amount_f.is_integer():
            raise IllegalAction(raw, "bets are whole chips", legal)
        amount = int(amount_f)
        r = self.rules
        if amount % r.bet_increment:
            raise IllegalAction(raw, f"bets must be multiples of {r.bet_increment}", legal)
        if amount < legal["min"]:
            raise IllegalAction(raw, f"the minimum bet is {r.min_bet}", legal)
        if amount > legal["max"]:
            why = "the table maximum" if amount > r.max_bet else "the seat's bankroll"
            raise IllegalAction(raw, f"{amount} is above {why}", legal)
        s = self._seats[seat]
        s.bankroll -= amount
        s.bet = amount
        self._on_table += amount
        self._log("bet", seat=seat, amount=amount)
        self._to_act = self._next_bettor()
        if self._to_act is None:
            self._deal_round()

    def _draw(self, face_up: bool = True) -> str:
        if self._shoe.remaining == 0:
            in_play = list(self._dealer)
            for s in self._seats:
                for h in s.hands:
                    in_play.extend(h.cards)
            visible = list(in_play)
            if len(self._dealer) >= 2 and not self._hole_revealed:
                visible.remove(self._dealer[1])
            self._shoe.shuffle(in_play=in_play, visible=visible)
            self._shuffled_this_round = True
            self._log("shuffle", reason="empty")
            if self._shoe.remaining == 0:
                raise BlackjackError(
                    f"Every card of the {self.rules.decks}-deck shoe is on the table, so none "
                    "is left to deal: play with more decks, or fewer seats or hands."
                )
        return self._shoe.draw(face_up=face_up)

    def _deal_to(self, seat: int, hand: int) -> str:
        card = self._draw()
        self._seats[seat].hands[hand].cards.append(card)
        self._log("deal", to=seat, hand=hand, card=card)
        return card

    def _deal_dealer(self, face_up: bool) -> str:
        card = self._draw(face_up=face_up)
        self._dealer.append(card)
        self._log("deal", to="dealer", card=card if face_up else None)
        return card

    def _deal_round(self) -> None:
        self._shuffled_this_round = False
        if self._shoe.needs_shuffle:
            self._shoe.shuffle()
            self._shuffled_this_round = True
            self._log("shuffle", reason="penetration")
        self._dealer = []
        self._hole_revealed = False
        players: list[int] = []
        for i, s in enumerate(self._seats):
            s.in_round = s.bet is not None
            s.hands = [Hand(cards=[], bet=s.bet)] if s.bet is not None else []
            if s.in_round:
                players.append(i)
        self._phase = "playing"
        self._to_act = None
        for i in players:
            self._deal_to(i, 0)
        self._deal_dealer(face_up=True)
        for i in players:
            self._deal_to(i, 0)
        self._deal_dealer(face_up=False)

        up = RANK_VALUE[self._dealer[0][0]]
        dealer_bj = _natural(self._dealer)
        could_have_bj = up in (1, 10)
        if self.rules.dealer_peeks and could_have_bj:
            self._log("peek", blackjack=dealer_bj)
            if dealer_bj:
                self._reveal_hole()
                for i in players:
                    hand = self._seats[i].hands[0]
                    if _natural(hand.cards):
                        hand.natural = True
                        hand.status = "blackjack"
                        self._settle(i, 0, "push")
                    else:
                        hand.status = "done"
                        self._settle(i, 0, "lose")
                self._finish_round()
                return
        pay_now = self.rules.dealer_peeks or not could_have_bj
        for i in players:
            hand = self._seats[i].hands[0]
            if _natural(hand.cards):
                hand.natural = True
                hand.status = "blackjack"
                if pay_now:
                    self._settle(i, 0, "blackjack")
        self._advance()

    # ------------------------------------------------------------------
    # Play
    # ------------------------------------------------------------------

    def _can_split(self, s: _Seat, hand: Hand) -> bool:
        r = self.rules
        if len(hand.cards) != 2 or len(s.hands) >= r.max_hands or s.bankroll < hand.bet:
            return False
        a, b = RANK_VALUE[hand.cards[0][0]], RANK_VALUE[hand.cards[1][0]]
        if a != b:
            return False
        return not (a == 1 and hand.split_aces and not r.resplit_aces)

    def _hand_actions(self, s: _Seat, h: int) -> list[str]:
        r = self.rules
        hand = s.hands[h]
        locked = hand.split_aces and not r.hit_split_aces
        legal: list[str] = []
        if not locked:
            legal.append("hit")
        legal.append("stand")
        two = len(hand.cards) == 2
        if (
            two
            and not locked
            and (not hand.from_split or r.double_after_split)
            and s.bankroll >= hand.bet
        ):
            legal.append("double")
        if self._can_split(s, hand):
            legal.append("split")
        if r.surrender == "late" and two and len(s.hands) == 1 and not hand.from_split:
            legal.append("surrender")
        return legal

    def _play(self, raw: str, move: str) -> None:
        seat, h = self._to_act, self._active_hand
        assert seat is not None and h is not None
        s = self._seats[seat]
        legal = self._hand_actions(s, h)
        if move not in legal:
            if move.startswith("bet:"):
                reason = "bets are only taken between rounds"
            elif move in PLAYING_ACTIONS:
                reason = f"{move} is not allowed on this hand"
            else:
                reason = f"unknown action; use one of {', '.join(PLAYING_ACTIONS)}"
            raise IllegalAction(raw, reason, legal)
        hand = s.hands[h]
        self._log("action", seat=seat, hand=h, action=move)
        if move == "hit":
            self._deal_to(seat, h)
            total, _ = hand.value()
            if total > 21:
                hand.status = "bust"
                self._settle(seat, h, "lose")
            elif total == 21:
                hand.status = "stood"
        elif move == "stand":
            hand.status = "stood"
        elif move == "double":
            s.bankroll -= hand.bet
            self._on_table += hand.bet
            hand.bet *= 2
            hand.doubled = True
            self._deal_to(seat, h)
            if hand.value()[0] > 21:
                hand.status = "bust"
                self._settle(seat, h, "lose")
            else:
                hand.status = "done"
        elif move == "split":
            s.bankroll -= hand.bet
            self._on_table += hand.bet
            second = hand.cards.pop()
            aces = RANK_VALUE[second[0]] == 1
            hand.from_split = True
            hand.split_aces = aces
            s.hands.insert(
                h + 1, Hand(cards=[second], bet=hand.bet, from_split=True, split_aces=aces)
            )
        else:  # surrender
            hand.status = "surrendered"
            if self.rules.dealer_peeks or RANK_VALUE[self._dealer[0][0]] not in (1, 10):
                self._settle(seat, h, "surrender")
            # (else no peek has ruled out a dealer blackjack, which takes the whole bet: the
            # dealer's turn settles the hand)
        self._advance()

    def _next_playing_hand(self) -> tuple[int, int] | None:
        for i, s in enumerate(self._seats):
            if not s.in_round:
                continue
            for h, hand in enumerate(s.hands):
                if hand.status == "playing":
                    return i, h
        return None

    def _advance(self) -> None:
        """Move play to the next hand needing a decision, or finish the round."""
        while True:
            found = self._next_playing_hand()
            if found is None:
                self._to_act = None
                self._active_hand = None
                self._dealer_turn()
                return
            i, h = found
            s = self._seats[i]
            hand = s.hands[h]
            if len(hand.cards) == 1:  # a split hand receives its second card
                self._deal_to(i, h)
            if hand.split_aces and not self.rules.hit_split_aces:
                if not self._can_split(s, hand):
                    hand.status = "done"
                    continue
            elif hand.value()[0] >= 21:
                hand.status = "stood"
                continue
            self._to_act = i
            self._active_hand = h
            return

    # ------------------------------------------------------------------
    # Dealer and settlement
    # ------------------------------------------------------------------

    def _reveal_hole(self) -> None:
        if self._hole_revealed:
            return
        hole = self._dealer[1]
        self._shoe.reveal(hole)
        self._hole_revealed = True
        total, _ = _total(self._dealer)
        self._log(
            "dealer_reveal",
            card=hole,
            cards=list(self._dealer),
            total=total,
            blackjack=_natural(self._dealer),
        )

    def _dealer_should_hit(self) -> bool:
        total, soft = _total(self._dealer)
        return total < 17 or (total == 17 and soft and self.rules.dealer_hits_soft_17)

    def _dealer_turn(self) -> None:
        self._reveal_hole()
        hands = [
            (i, h, hand)
            for i, s in enumerate(self._seats)
            if s.in_round
            for h, hand in enumerate(s.hands)
            if hand.outcome is None
        ]
        dealer_bj = _natural(self._dealer)
        needs_dealer = not dealer_bj and any(
            not hand.natural and hand.status != "surrendered" for _, _, hand in hands
        )
        if needs_dealer:
            while self._dealer_should_hit():
                card = self._draw()
                self._dealer.append(card)
                self._log("dealer_draw", card=card, total=_total(self._dealer)[0])
        dealer_total, _ = _total(self._dealer)
        for i, h, hand in hands:
            if hand.status == "surrendered":  # (without the peek, against an ace or ten)
                outcome = "lose" if dealer_bj else "surrender"
            elif hand.natural:
                outcome = "push" if dealer_bj else "blackjack"
            elif dealer_bj:
                outcome = "lose"
            else:
                total, _ = hand.value()
                if dealer_total > 21 or total > dealer_total:
                    outcome = "win"
                elif total == dealer_total:
                    outcome = "push"
                else:
                    outcome = "lose"
            self._settle(i, h, outcome)
        self._finish_round()

    def _settle(self, seat: int, h: int, outcome: str) -> None:
        s = self._seats[seat]
        hand = s.hands[h]
        bet = hand.bet
        if outcome == "win":
            payout: float = float(bet)
        elif outcome == "lose":
            payout = -float(bet)
        elif outcome == "push":
            payout = 0.0
        elif outcome == "blackjack":
            payout = float(Fraction(bet) * self._ratio)
        elif outcome == "surrender":
            payout = -bet / 2
        else:  # pragma: no cover - internal misuse
            raise ValueError(f"Unknown outcome {outcome!r}.")
        hand.outcome = outcome
        hand.payout = payout
        s.bankroll += bet + payout
        self._on_table -= bet
        self.house_net -= payout
        self._log("settle", seat=seat, hand=h, outcome=outcome, payout=_num(payout))

    def _finish_round(self) -> None:
        self._to_act = None
        self._active_hand = None
        dealer_total, _ = _total(self._dealer)
        seats: list[dict[str, Any]] = []
        nets: list[int | float | None] = []
        for i, s in enumerate(self._seats):
            if not s.in_round:
                seats.append(
                    {
                        "seat": i,
                        "played": False,
                        "bet": None,
                        "wagered": 0,
                        "net": None,
                        "bankroll": _num(s.bankroll),
                        "hands": [],
                    }
                )
                nets.append(None)
                continue
            net = sum(h.payout or 0.0 for h in s.hands)
            nets.append(_num(net))
            seats.append(
                {
                    "seat": i,
                    "played": True,
                    "bet": s.bet,
                    "wagered": sum(h.bet for h in s.hands),
                    "net": _num(net),
                    "bankroll": _num(s.bankroll),
                    "hands": [
                        {
                            "cards": list(h.cards),
                            "total": h.value()[0],
                            "bet": h.bet,
                            "doubled": h.doubled,
                            "from_split": h.from_split,
                            "status": h.status,
                            "outcome": h.outcome,
                            "payout": _num(h.payout or 0.0),
                        }
                        for h in s.hands
                    ],
                }
            )
        summary = {
            "round": self._round_no,
            "shuffled": self._shuffled_this_round,
            "dealer": {
                "cards": list(self._dealer),
                "total": dealer_total,
                "blackjack": _natural(self._dealer),
                "bust": dealer_total > 21,
            },
            "seats": seats,
        }
        self._history.append(summary)
        self._last_round = summary
        self._log("round_end", net=nets, dealer_total=dealer_total)
        self._rounds_played += 1
        self._round_no += 1
        self._start_betting()

    def __repr__(self) -> str:
        return (
            f"BlackjackTable({self.rules.label}, seats={len(self._seats)}, "
            f"round={self._round_no}, phase={self._phase!r}, bankrolls={self.bankrolls})"
        )


__all__ = ["BlackjackTable", "Hand", "PLAYING_ACTIONS", "Phase", "HandStatus", "Outcome"]
