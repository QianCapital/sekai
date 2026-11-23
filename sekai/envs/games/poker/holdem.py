"""Heads-up no-limit Texas hold'em: one hand, with exact betting rules and pot accounting.

`HoldemHand` is the trusted dealer. It holds the deck and both players' hole cards, so it
must stay with whoever runs the game: an agent (or a browser) only ever receives a
`SeatView` from `HoldemHand.view`, a frozen snapshot of exactly what that seat may know.
Views hold plain tuples, ints and copied dicts, never a reference back to the hand, so
nothing reachable from a view reveals the deck or the opponent's cards before a showdown.

**Rules.**

- Seats `0` and `1`. The **button** posts the small blind and acts first preflop; the
  other seat (the **big blind**) acts first on the flop, turn and river. A stack that
  cannot cover a blind posts all-in.
- Actions: `fold`, `check`, `call`, `bet` / `raise` **to** an amount (the player's total
  for the street). `bet` and `raise` are synonyms in `HoldemHand.act`; the event log
  records `bet` when nothing had been bet on the street and `raise` otherwise (preflop
  is always a raise: the big blind is a bet). Folding is illegal while checking is free.
- The minimum opening bet is the big blind; the minimum raise increment is the largest
  full bet/raise increment so far on the street, and at least the big blind. A player may
  always go all-in for less. An all-in for less than a full raise does not reopen the
  betting for a player who has already acted on the street: they may only call or fold.
- Bets are capped at what the opponent can still match (`max_raise_to` is the smaller of
  the two players' street totals if all-in), so nobody ever puts in chips that cannot be
  called; "all-in" for the bigger stack means putting the opponent all-in. Uncalled chips
  (the excess of a short blind, or a bet that is folded to) are returned and logged as a
  `return` event.
- When a player is all-in and the betting is closed, the remaining board is dealt out and
  the hand goes to showdown. Ties split the pot; an odd chip goes to the big blind (the
  out-of-position seat). Heads-up there are never side pots: after uncalled chips are
  returned both players have always committed the same amount.

Card order with a fixed `deck`: `deck[0:2]` are seat 0's hole cards, `deck[2:4]` seat 1's,
`deck[4:7]` the flop, `deck[7]` the turn and `deck[8]` the river (no burn cards). Any
further cards are ignored.

**Streets.** `street` is `"preflop"`, `"flop"`, `"turn"` or `"river"` while the hand is in
progress, then `"showdown"` if it ended with hands shown or `"complete"` if it ended with a
fold. `HoldemHand.is_complete` is true for both.

**Events.** `HoldemHand.events` (and `SeatView.events`) is a JSON-friendly list of
everything public, in order. Cards are strings:

```python
{"type": "post", "seat": 0, "blind": "small", "amount": 50, "all_in": False}
{"type": "deal", "street": "flop", "cards": ["Ah", "7d", "2c"]}  # turn/river: 1 card
{"type": "action", "seat": 1, "street": "preflop", "kind": "raise",
 "to": 300, "added": 200, "all_in": False}
{"type": "return", "seat": 0, "amount": 400}  # uncalled chips
{"type": "showdown", "seat": 0, "cards": ["As", "Kd"],
 "rank": "Two Pair, Aces and Kings", "best": ["As", "Ah", "Kd", "Kc", "7d"]}
{"type": "win", "seat": 0, "amount": 600}
```

For an `action` event, `to` is the player's street total after the action (unchanged for
fold and check) and `added` the chips it put in. Hole cards never appear in events except
in `showdown` events. At showdown the last aggressor of the final betting round shows
first, otherwise the big blind.
"""

from __future__ import annotations

import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sekai._types import SeedType
from sekai.envs.games.poker.cards import (
    CARD_STRINGS,
    CardLike,
    format_cards,
    parse_cards,
    shuffled_deck,
)
from sekai.envs.games.poker.errors import (
    HandComplete,
    HiddenInformation,
    IllegalAction,
    OutOfTurn,
)
from sekai.envs.games.poker.evaluator import best_five_unchecked, describe, evaluate_unchecked
from sekai.utils.seeding import np_random

BETTING_STREETS: tuple[str, ...] = ("preflop", "flop", "turn", "river")
SHOWDOWN = "showdown"
COMPLETE = "complete"
STREETS: tuple[str, ...] = (*BETTING_STREETS, SHOWDOWN, COMPLETE)

ACTION_KINDS: tuple[str, ...] = ("fold", "check", "call", "bet", "raise")
_BOARD_SLICES = {"flop": (0, 3), "turn": (3, 4), "river": (4, 5)}


# ----------------------------------------------------------------------
# Value types
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Action:
    """A concrete betting action.

    Attributes:
        kind: One of `"fold"`, `"check"`, `"call"`, `"bet"`, `"raise"`.
        amount: The **to** amount (the player's street total) of a `bet` / `raise`;
            `None` for the other kinds.
    """

    kind: str
    amount: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ACTION_KINDS:
            raise IllegalAction(
                f"Unknown action kind {self.kind!r}; expected one of {ACTION_KINDS}.",
                action=self,
            )
        if self.kind in ("bet", "raise"):
            object.__setattr__(self, "amount", _as_chips(self.amount, self))
        elif self.amount is not None:
            raise IllegalAction(
                f"{self.kind!r} takes no amount (got {self.amount!r}).", action=self
            )

    @classmethod
    def fold(cls) -> Action:
        return cls("fold")

    @classmethod
    def check(cls) -> Action:
        return cls("check")

    @classmethod
    def call(cls) -> Action:
        return cls("call")

    @classmethod
    def bet(cls, to: int) -> Action:
        return cls("bet", to)

    @classmethod
    def raise_to(cls, to: int) -> Action:
        return cls("raise", to)

    def __str__(self) -> str:
        """Canonical text form understood by `parse_action`: `"raise:300"`."""
        return self.kind if self.amount is None else f"{self.kind}:{self.amount}"


@dataclass(frozen=True, slots=True)
class LegalActions:
    """What the player to act may do.

    Attributes:
        can_fold: Folding is legal (there is a bet to call).
        can_check: Checking is legal (nothing to call).
        call_amount: The chips a call adds (`0` when checking is free; when the stack is
            short it is the whole stack, i.e. calling all-in).
        min_raise_to: The smallest **to** amount of a bet or raise, `None` when raising
            is impossible.
        max_raise_to: The largest; `min_raise_to == max_raise_to` when the only possible
            raise is all-in.
    """

    can_fold: bool
    can_check: bool
    call_amount: int
    min_raise_to: int | None
    max_raise_to: int | None

    @property
    def can_call(self) -> bool:
        return self.call_amount > 0

    @property
    def can_raise(self) -> bool:
        return self.min_raise_to is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "can_fold": self.can_fold,
            "can_check": self.can_check,
            "can_call": self.can_call,
            "call_amount": self.call_amount,
            "can_raise": self.can_raise,
            "min_raise_to": self.min_raise_to,
            "max_raise_to": self.max_raise_to,
        }


@dataclass(frozen=True, slots=True)
class HandResult:
    """How a finished hand ended.

    Attributes:
        deltas: `deltas[s]` is seat `s`'s net chip change (they sum to zero).
        winners: The winning seats (both on a split pot).
        showdown: Whether the hand ended with hands shown.
        pot: The final pot.
        hands: Filled only at showdown: `{seat: {"cards": [...], "rank": "Flush, Ace
            high", "best": [...], "value": int}}`.
    """

    deltas: tuple[int, int]
    winners: tuple[int, ...]
    showdown: bool
    pot: int
    hands: dict[int, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "deltas": list(self.deltas),
            "winners": list(self.winners),
            "showdown": self.showdown,
            "pot": self.pot,
            "hands": {str(s): dict(h) for s, h in self.hands.items()},
        }

    def _copy(self) -> HandResult:
        hands = {s: _copy_event(h) for s, h in self.hands.items()}
        return HandResult(self.deltas, self.winners, self.showdown, self.pot, hands)


@dataclass(frozen=True, slots=True)
class SeatView:
    """Everything one seat may know about a hand: its information set.

    Built by `HoldemHand.view` (`seat` is 0 or 1), `HoldemHand.public_view` (`seat is
    None`, no hole cards except shown ones) or `HoldemHand.reveal` (`seat is None`, both
    hands; only once the hand is complete). Cards are integers (see `cards`); `to_dict`
    gives a JSON-friendly form with card strings.

    Attributes:
        seat: The viewer's seat, or `None` for a public or revealed view.
        holes: `holes[s]` is seat `s`'s hole cards if this viewer may see them, else
            `None`: the viewer's own cards always, the opponent's only after a showdown.
        board: The community cards dealt so far.
        stacks: Chips behind (not yet committed). After the hand: final stacks, winnings
            included.
        street_bets: Each seat's bets on the current street.
        committed: Chips each seat has put in the pot this hand (uncalled chips excluded
            once returned).
        pot: Both seats' committed chips.
        button: The seat on the button.
        street: The street (see `holdem`).
        to_act: The seat to act, `None` once the hand is over.
        legal: The `LegalActions`, set only when `seat` is the seat to act.
        small_blind: The small blind.
        big_blind: The big blind.
        starting_stacks: Each seat's stack before the blinds.
        events: The public event log (see `holdem`).
        result: The `HandResult` once the hand is over.
    """

    seat: int | None
    holes: tuple[tuple[int, ...] | None, tuple[int, ...] | None]
    board: tuple[int, ...]
    stacks: tuple[int, int]
    street_bets: tuple[int, int]
    committed: tuple[int, int]
    pot: int
    button: int
    street: str
    to_act: int | None
    legal: LegalActions | None
    small_blind: int
    big_blind: int
    starting_stacks: tuple[int, int]
    events: tuple[dict[str, Any], ...]
    result: HandResult | None

    @property
    def hole(self) -> tuple[int, ...]:
        """The viewer's own hole cards (empty for a public or revealed view)."""
        if self.seat is None:
            return ()
        cards = self.holes[self.seat]
        return () if cards is None else cards

    @property
    def opponent(self) -> int | None:
        return None if self.seat is None else 1 - self.seat

    @property
    def opponent_hole(self) -> tuple[int, ...] | None:
        """The opponent's hole cards once shown at showdown, else `None`."""
        return None if self.seat is None else self.holes[1 - self.seat]

    @property
    def is_complete(self) -> bool:
        return self.street in (SHOWDOWN, COMPLETE)

    @property
    def is_my_turn(self) -> bool:
        return self.seat is not None and self.to_act == self.seat

    @property
    def current_bet(self) -> int:
        """The largest street bet: what a call matches."""
        return max(self.street_bets)

    @property
    def to_call(self) -> int:
        """Chips the viewer needs to add to call (capped at its stack); 0 for public views."""
        if self.seat is None:
            return 0
        return min(self.current_bet - self.street_bets[self.seat], self.stacks[self.seat])

    @property
    def big_blind_seat(self) -> int:
        return 1 - self.button

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly snapshot (cards as strings), e.g. for a web page."""
        return {
            "seat": self.seat,
            "hole": [CARD_STRINGS[c] for c in self.hole],
            "holes": [None if h is None else [CARD_STRINGS[c] for c in h] for h in self.holes],
            "board": [CARD_STRINGS[c] for c in self.board],
            "stacks": list(self.stacks),
            "street_bets": list(self.street_bets),
            "committed": list(self.committed),
            "pot": self.pot,
            "button": self.button,
            "street": self.street,
            "to_act": self.to_act,
            "legal": None if self.legal is None else self.legal.to_dict(),
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "starting_stacks": list(self.starting_stacks),
            "events": [_copy_event(e) for e in self.events],
            "result": None if self.result is None else self.result.to_dict(),
        }


# ----------------------------------------------------------------------
# The hand
# ----------------------------------------------------------------------


class HoldemHand:
    """One heads-up no-limit hold'em hand, from the blinds to the payout.

    The blinds are posted on construction. Then, while `not is_complete`, the seat
    `to_act` chooses one of `legal()` and calls `act`.

    Args:
        stacks: The chips each seat starts the hand with (positive integers).
        button: The seat on the button (small blind, first to act preflop).
        small_blind: The small blind, `0 < small_blind <= big_blind`.
        big_blind: The big blind.
        rng: Seed or `numpy.random.Generator` used to shuffle when `deck` is not given.
        deck: Optional fixed card order (ints or strings, or a string like `"AsKd..."`);
            see the module docstring for which positions go where. At least 9 cards.

    Raises:
        ValueError: For stacks or blinds that are not positive whole chips, a button
            other than 0 or 1, a small blind above the big blind or a short fixed deck.
        InvalidCard: For a fixed deck with invalid or duplicated cards.
    """

    def __init__(
        self,
        stacks: Sequence[int] = (10000, 10000),
        button: int = 0,
        small_blind: int = 50,
        big_blind: int = 100,
        rng: SeedType = None,
        deck: str | Sequence[CardLike] | None = None,
    ) -> None:
        if len(stacks) != 2:
            raise ValueError(f"Heads-up needs exactly two stacks, got {len(stacks)}.")
        starting = (_as_chips(stacks[0], "stack"), _as_chips(stacks[1], "stack"))
        if button not in (0, 1):
            raise ValueError(f"button must be 0 or 1, got {button!r}.")
        sb, bb = _as_chips(small_blind, "small_blind"), _as_chips(big_blind, "big_blind")
        if sb > bb:
            raise ValueError(f"small_blind ({sb}) must not exceed big_blind ({bb}).")

        if deck is None:
            cards = shuffled_deck(rng)
        else:
            cards = parse_cards(deck)
            if len(cards) < 9:
                raise ValueError(f"A fixed deck needs at least 9 cards, got {len(cards)}.")

        self._starting: tuple[int, int] = starting
        self._button = int(button)
        self._sb = sb
        self._bb = bb
        self._holes: tuple[tuple[int, int], tuple[int, int]] = (
            (cards[0], cards[1]),
            (cards[2], cards[3]),
        )
        self._runout: tuple[int, ...] = tuple(cards[4:9])  # hidden until dealt

        self._stacks = list(starting)
        self._street_bets = [0, 0]
        self._committed = [0, 0]
        self._board: list[int] = []
        self._street = "preflop"
        self._to_act: int | None = None
        self._legal: LegalActions | None = None
        self._acted = [False, False]
        self._min_increment = bb
        self._aggressor: int | None = None
        self._shown = [False, False]
        self._events: list[dict[str, Any]] = []
        self._result: HandResult | None = None

        self._post_blinds()
        self._settle((self._button, 1 - self._button))

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------

    @property
    def button(self) -> int:
        return self._button

    @property
    def big_blind_seat(self) -> int:
        return 1 - self._button

    @property
    def small_blind(self) -> int:
        return self._sb

    @property
    def big_blind(self) -> int:
        return self._bb

    @property
    def starting_stacks(self) -> tuple[int, int]:
        return self._starting

    @property
    def stacks(self) -> tuple[int, int]:
        """Chips behind. Once the hand is complete these include the winnings."""
        return (self._stacks[0], self._stacks[1])

    @property
    def street_bets(self) -> tuple[int, int]:
        return (self._street_bets[0], self._street_bets[1])

    @property
    def committed(self) -> tuple[int, int]:
        return (self._committed[0], self._committed[1])

    @property
    def pot(self) -> int:
        """Total committed by both seats this hand (the final pot once complete)."""
        return self._committed[0] + self._committed[1]

    @property
    def current_bet(self) -> int:
        return max(self._street_bets)

    @property
    def to_call(self) -> int:
        """Chips the seat to act needs to add to call (0 when nobody is to act)."""
        return 0 if self._legal is None else self._legal.call_amount

    @property
    def street(self) -> str:
        return self._street

    @property
    def board(self) -> list[int]:
        """Community cards dealt so far."""
        return list(self._board)

    @property
    def to_act(self) -> int | None:
        return self._to_act

    @property
    def is_complete(self) -> bool:
        return self._result is not None

    @property
    def result(self) -> HandResult | None:
        return None if self._result is None else self._result._copy()

    @property
    def events(self) -> list[dict[str, Any]]:
        """A copy of the public event log (see the module docstring)."""
        return list(map(_copy_hand_event, self._events))

    def hole(self, seat: int) -> list[int]:
        """Seat `seat`'s hole cards. Trusted callers only: give agents `view`."""
        return list(self._holes[_check_seat(seat)])

    def legal(self) -> LegalActions:
        """Legal actions for `to_act`. Raises `HandComplete` once the hand is over."""
        if self._legal is None:
            raise HandComplete()
        return self._legal

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------

    def view(self, seat: int) -> SeatView:
        """The information set of `seat`: its own cards, public state and shown cards."""
        seat = _check_seat(seat)
        holes = [h if self._shown[s] else None for s, h in enumerate(self._holes)]
        holes[seat] = self._holes[seat]
        return self._make_view(seat, (holes[0], holes[1]))

    def public_view(self) -> SeatView:
        """What a spectator may see: no hole cards except those shown at showdown."""
        holes = [h if self._shown[s] else None for s, h in enumerate(self._holes)]
        return self._make_view(None, (holes[0], holes[1]))

    def reveal(self) -> SeatView:
        """Both hands, for display once the hand is complete.

        Undealt board cards are never revealed.

        Raises:
            HiddenInformation: While the hand is in progress.
        """
        if not self.is_complete:
            raise HiddenInformation("Hole cards can only be revealed once the hand is complete.")
        return self._make_view(None, (self._holes[0], self._holes[1]))

    def _make_view(
        self, seat: int | None, holes: tuple[tuple[int, ...] | None, tuple[int, ...] | None]
    ) -> SeatView:
        return SeatView(
            seat=seat,
            holes=holes,
            board=tuple(self._board),
            stacks=self.stacks,
            street_bets=self.street_bets,
            committed=self.committed,
            pot=self.pot,
            button=self._button,
            street=self._street,
            to_act=self._to_act,
            legal=self._legal if seat is not None and seat == self._to_act else None,
            small_blind=self._sb,
            big_blind=self._bb,
            starting_stacks=self._starting,
            events=tuple(map(_copy_hand_event, self._events)),
            result=self.result,
        )

    @classmethod
    def from_view(cls, view: SeatView, *, rng: SeedType = None) -> HoldemHand:
        """A complete hand consistent with `view`, e.g. for determinized search.

        Unknown cards (the opponent's hole cards if not shown, undealt board cards) are
        drawn uniformly from the unseen cards, then the public actions are replayed. The
        result is a hypothesis, not the real hand.
        """
        if view.seat is None and any(h is None for h in view.holes):
            raise ValueError("from_view needs a seat view or a revealed view.")
        gen, _ = np_random(rng)
        known = {c for h in view.holes if h is not None for c in h} | set(view.board)
        unseen = [c for c in range(52) if c not in known]
        draw = [unseen[int(i)] for i in gen.permutation(len(unseen))]
        holes = []
        for h in view.holes:
            if h is None:
                h, draw = tuple(draw[:2]), draw[2:]
            holes.append(tuple(h))
        board = list(view.board) + draw[: 5 - len(view.board)]
        hand = cls(
            stacks=view.starting_stacks,
            button=view.button,
            small_blind=view.small_blind,
            big_blind=view.big_blind,
            deck=[*holes[0], *holes[1], *board],
        )
        for ev in view.events:
            if ev["type"] == "action":
                kind = ev["kind"]
                hand.act(Action(kind, ev["to"] if kind in ("bet", "raise") else None))
        return hand

    def copy(self) -> HoldemHand:
        """An independent copy (including the hidden cards), e.g. for search."""
        new = object.__new__(HoldemHand)
        for name in _STATE_ATTRS:
            value = getattr(self, name)
            if isinstance(value, list):
                value = [_copy_event(v) if isinstance(v, dict) else v for v in value]
            setattr(new, name, value)
        return new

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def act(self, action: Action | str, *, seat: int | None = None) -> None:
        """Apply `action` for the seat to act.

        Args:
            action: An `Action`, or text for `parse_action`.
            seat: If given, the seat expected to act.

        Raises:
            HandComplete: If the hand is over.
            OutOfTurn: If `seat` is given and it is not that seat's turn.
            IllegalAction: If the action is not legal.
        """
        if self._result is not None or self._to_act is None or self._legal is None:
            raise HandComplete()
        actor = self._to_act
        if seat is not None and seat != actor:
            raise OutOfTurn(
                f"It is seat {actor}'s turn, not seat {seat}'s.", action=action, legal=self._legal
            )
        if isinstance(action, str):
            action = parse_action(action, self)
        if not isinstance(action, Action):
            raise TypeError(f"Expected an Action or action text, got {type(action).__name__}.")
        legal = self._legal
        kind = action.kind
        opp = 1 - actor

        if kind == "fold":
            if not legal.can_fold:
                raise IllegalAction(
                    "Cannot fold when checking is free; check instead.", action=action, legal=legal
                )
            self._log_action(actor, "fold", 0)
            self._finish_fold(winner=opp)
            return

        if kind == "check":
            if not legal.can_check:
                raise IllegalAction(
                    f"Cannot check facing a bet; call {legal.call_amount} or fold.",
                    action=action,
                    legal=legal,
                )
            self._acted[actor] = True
            self._log_action(actor, "check", 0)
        elif kind == "call":
            if not legal.can_call:
                raise IllegalAction("Nothing to call; check instead.", action=action, legal=legal)
            self._commit(actor, legal.call_amount)
            self._acted[actor] = True
            self._log_action(actor, "call", legal.call_amount)
        else:  # bet / raise to action.amount
            to = action.amount
            assert to is not None
            if legal.min_raise_to is None or legal.max_raise_to is None:
                raise IllegalAction(
                    "Raising is not possible here (the opponent is all-in, the stack only "
                    "covers a call, or a short all-in did not reopen the betting).",
                    action=action,
                    legal=legal,
                )
            if not legal.min_raise_to <= to <= legal.max_raise_to:
                verb = "bet" if self.current_bet == 0 else "raise"
                if legal.min_raise_to == legal.max_raise_to:
                    msg = f"the only possible {verb} is all-in to {legal.max_raise_to}"
                else:
                    msg = (
                        f"the {verb} must be to between {legal.min_raise_to} and "
                        f"{legal.max_raise_to}"
                    )
                raise IllegalAction(f"Cannot {verb} to {to}: {msg}.", action=action, legal=legal)
            current = self.current_bet
            increment = to - current
            added = to - self._street_bets[actor]
            self._commit(actor, added)
            if increment >= self._min_increment:  # a full bet/raise reopens the betting
                self._min_increment = increment
                self._acted[opp] = False
            self._acted[actor] = True
            self._aggressor = actor
            self._log_action(actor, "bet" if current == 0 else "raise", added)

        self._settle((opp, actor))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _commit(self, seat: int, chips: int) -> None:
        assert 0 <= chips <= self._stacks[seat], (seat, chips, self._stacks)
        self._stacks[seat] -= chips
        self._street_bets[seat] += chips
        self._committed[seat] += chips

    def _post_blinds(self) -> None:
        for seat, blind, size in (
            (self._button, "small", self._sb),
            (1 - self._button, "big", self._bb),
        ):
            amount = min(size, self._stacks[seat])
            self._commit(seat, amount)
            self._events.append(
                {
                    "type": "post",
                    "seat": seat,
                    "blind": blind,
                    "amount": amount,
                    "all_in": self._stacks[seat] == 0,
                }
            )

    def _log_action(self, seat: int, kind: str, added: int) -> None:
        self._events.append(
            {
                "type": "action",
                "seat": seat,
                "street": self._street,
                "kind": kind,
                "to": self._street_bets[seat],
                "added": added,
                "all_in": self._stacks[seat] == 0,
            }
        )

    def _needs_to_act(self, seat: int) -> bool:
        if self._stacks[seat] == 0:
            return False
        if self._street_bets[seat] < max(self._street_bets):
            return True
        return not self._acted[seat] and self._stacks[1 - seat] > 0

    def _compute_legal(self, seat: int) -> LegalActions:
        opp = 1 - seat
        current = max(self._street_bets)
        to_call = min(current - self._street_bets[seat], self._stacks[seat])
        cap = min(
            self._street_bets[seat] + self._stacks[seat],
            self._street_bets[opp] + self._stacks[opp],
        )
        min_to = max_to = None
        if not self._acted[seat] and self._stacks[seat] > to_call and cap > current:
            max_to = cap
            min_to = min(current + self._min_increment, cap)
        return LegalActions(
            can_fold=to_call > 0,
            can_check=to_call == 0,
            call_amount=to_call,
            min_raise_to=min_to,
            max_raise_to=max_to,
        )

    def _settle(self, order: tuple[int, int]) -> None:
        """Hand the turn to the first seat in `order` that must act, else close the round."""
        while True:
            for seat in order:
                if self._needs_to_act(seat):
                    self._to_act = seat
                    self._legal = self._compute_legal(seat)
                    return
            self._to_act = None
            self._legal = None
            self._return_uncalled()
            if self._street == "river" or min(self._stacks) == 0:
                while self._street != "river":  # all-in: run out the board
                    self._deal_next_street(runout=True)
                self._showdown()
                return
            self._deal_next_street()
            bb = 1 - self._button
            order = (bb, 1 - bb)

    def _deal_next_street(self, *, runout: bool = False) -> None:
        street = BETTING_STREETS[BETTING_STREETS.index(self._street) + 1]
        lo, hi = _BOARD_SLICES[street]
        cards = self._runout[lo:hi]
        self._board.extend(cards)
        self._street = street
        self._street_bets = [0, 0]
        self._acted = [False, False]
        self._min_increment = self._bb
        if not runout:  # the last betting round's aggressor shows first at showdown
            self._aggressor = None
        self._events.append(
            {"type": "deal", "street": street, "cards": [CARD_STRINGS[c] for c in cards]}
        )

    def _return_uncalled(self) -> None:
        a, b = self._street_bets
        if a == b:
            return
        seat = 0 if a > b else 1
        amount = abs(a - b)
        self._street_bets[seat] -= amount
        self._committed[seat] -= amount
        self._stacks[seat] += amount
        self._events.append({"type": "return", "seat": seat, "amount": amount})

    def forfeit(self, *, seat: int | None = None) -> None:
        """The seat to act gives the hand up: a fold, even where checking is free (when
        `act` refuses a fold). `HoldemEnv(illegal_action="lose")` plays it for an illegal
        action.

        Raises:
            HandComplete: If the hand is over.
            OutOfTurn: If `seat` is given and it is not that seat's turn.
        """
        if self._result is not None or self._to_act is None or self._legal is None:
            raise HandComplete()
        actor = self._to_act
        if seat is not None and seat != actor:
            raise OutOfTurn(
                f"It is seat {actor}'s turn, not seat {seat}'s.", action="fold", legal=self._legal
            )
        self._log_action(actor, "fold", 0)
        self._finish_fold(winner=1 - actor)

    def _finish_fold(self, winner: int) -> None:
        self._return_uncalled()
        pot = self.pot
        self._stacks[winner] += pot
        self._events.append({"type": "win", "seat": winner, "amount": pot})
        self._close(winners=(winner,), showdown=False, hands={}, street=COMPLETE)

    def _showdown(self) -> None:
        board = self._board
        values = [evaluate_unchecked([*self._holes[s], *board]) for s in (0, 1)]
        bb = 1 - self._button
        first = self._aggressor if self._aggressor is not None else bb
        hands: dict[int, dict[str, Any]] = {}
        for seat in (first, 1 - first):
            cards = [CARD_STRINGS[c] for c in self._holes[seat]]
            five = best_five_unchecked([*self._holes[seat], *board], values[seat])
            best = [CARD_STRINGS[c] for c in five]
            rank = describe(values[seat])
            self._shown[seat] = True
            hands[seat] = {"cards": cards, "rank": rank, "best": best, "value": values[seat]}
            self._events.append(
                {
                    "type": "showdown",
                    "seat": seat,
                    "cards": list(cards),
                    "rank": rank,
                    "best": list(best),
                }
            )
        pot = self.pot
        if values[0] != values[1]:
            winner = 0 if values[0] > values[1] else 1
            payouts = {winner: pot}
        else:
            half, odd = divmod(pot, 2)
            payouts = {bb: half + odd, 1 - bb: half}
        for seat in sorted(payouts, key=lambda s: s != bb):
            self._stacks[seat] += payouts[seat]
            self._events.append({"type": "win", "seat": seat, "amount": payouts[seat]})
        winners = tuple(sorted(payouts))
        self._close(
            winners=winners, showdown=True, hands=dict(sorted(hands.items())), street=SHOWDOWN
        )

    def _close(
        self,
        *,
        winners: tuple[int, ...],
        showdown: bool,
        hands: dict[int, dict[str, Any]],
        street: str,
    ) -> None:
        self._street = street
        self._to_act = None
        self._legal = None
        deltas = (self._stacks[0] - self._starting[0], self._stacks[1] - self._starting[1])
        assert deltas[0] + deltas[1] == 0, deltas
        self._result = HandResult(
            deltas=deltas, winners=winners, showdown=showdown, pot=self.pot, hands=hands
        )

    def __repr__(self) -> str:
        board = format_cards(self._board) or "-"
        return (
            f"<HoldemHand street={self._street} board=[{board}] pot={self.pot} "
            f"stacks={self.stacks} to_act={self._to_act}>"
        )


_STATE_ATTRS = (
    "_starting",
    "_button",
    "_sb",
    "_bb",
    "_holes",
    "_runout",
    "_stacks",
    "_street_bets",
    "_committed",
    "_board",
    "_street",
    "_to_act",
    "_legal",
    "_acted",
    "_min_increment",
    "_aggressor",
    "_shown",
    "_events",
    "_result",
)


# ----------------------------------------------------------------------
# Text moves
# ----------------------------------------------------------------------


def _legal_of(
    state: HoldemHand | SeatView,
) -> tuple[LegalActions, int, tuple[int, int], tuple[int, int]]:
    """(legal, seat, street_bets, stacks) for the seat to act in `state`."""
    if isinstance(state, HoldemHand):
        legal = state.legal()
        seat = state.to_act
    elif isinstance(state, SeatView):
        if state.legal is None:
            raise IllegalAction("It is not this seat's turn to act.")
        legal, seat = state.legal, state.seat
    else:
        raise TypeError(f"Expected a HoldemHand or SeatView, got {type(state).__name__}.")
    assert seat is not None
    return legal, seat, state.street_bets, state.stacks


_ALLIN_WORDS = frozenset({"allin", "all-in", "all_in", "all in", "shove", "jam"})


def parse_action(text: str, state: HoldemHand | SeatView) -> Action:
    """Parse a text move for the seat to act in `state` and check that it is legal.

    Accepted: `fold`, `check`, `call`, `bet:<to>`, `raise:<to>` (`bet` and `raise` are
    interchangeable; a space works instead of the colon and amounts may contain `,` or
    `_`) and `allin` (also `all-in`): the largest bet/raise possible, or a call when
    raising is impossible.

    Raises:
        IllegalAction: For unknown or illegal moves.
    """
    if not isinstance(text, str):
        raise IllegalAction(f"Expected action text, got {text!r}.", action=text)
    legal, _, street_bets, _ = _legal_of(state)
    t = " ".join(text.strip().lower().replace(":", " ").split())
    if t in _ALLIN_WORDS:
        if legal.max_raise_to is not None:
            return Action("bet" if max(street_bets) == 0 else "raise", legal.max_raise_to)
        if legal.can_call:
            return Action("call")
        raise IllegalAction(
            "Cannot go all-in here: nothing to call and raising is impossible.",
            action=text,
            legal=legal,
        )
    word, _, rest = t.partition(" ")
    if word in ("fold", "check", "call"):
        if rest:
            raise IllegalAction(f"{word!r} takes no amount: {text!r}.", action=text, legal=legal)
        action = Action(word)
        _validate(action, legal)
        return action
    if word in ("bet", "raise"):
        try:
            to = int(rest.replace(",", "").replace("_", ""))
        except ValueError:
            raise IllegalAction(
                f"Cannot read an amount in {text!r}; use e.g. 'raise:300' (the total to raise to).",
                action=text,
                legal=legal,
            ) from None
        action = Action("bet" if max(street_bets) == 0 else "raise", to)
        _validate(action, legal)
        return action
    raise IllegalAction(
        f"Unknown action {text!r}; use fold, check, call, bet:<to>, raise:<to> or allin.",
        action=text,
        legal=legal,
    )


def _validate(action: Action, legal: LegalActions) -> None:
    kind = action.kind
    if kind == "fold" and not legal.can_fold:
        raise IllegalAction(
            "Cannot fold when checking is free; check instead.", action=action, legal=legal
        )
    if kind == "check" and not legal.can_check:
        raise IllegalAction(
            f"Cannot check facing a bet; call {legal.call_amount} or fold.",
            action=action,
            legal=legal,
        )
    if kind == "call" and not legal.can_call:
        raise IllegalAction("Nothing to call; check instead.", action=action, legal=legal)
    if kind in ("bet", "raise"):
        lo, hi = legal.min_raise_to, legal.max_raise_to
        if lo is None or hi is None:
            raise IllegalAction("Raising is not possible here.", action=action, legal=legal)
        assert action.amount is not None
        if not lo <= action.amount <= hi:
            raise IllegalAction(
                f"Cannot {kind} to {action.amount}: the amount must be between {lo} and {hi}.",
                action=action,
                legal=legal,
            )


def format_action(
    action: Action | Mapping[str, Any], context: HoldemHand | SeatView | None = None
) -> str:
    """Display text for an action.

    `"Folds"`, `"Checks"`, `"Calls 200"`, `"Bets 300"`, `"Raises to 900"`,
    `"All-in 10,000"` (a bet/raise that puts the actor all-in) or
    `"Calls 9,900 (all-in)"`.

    Args:
        action: An `Action`, with `context` the hand or seat view *before* it is applied
            (needed for call amounts and all-in detection), or an `action` event dict
            from `HoldemHand.events` (no context needed).
        context: See `action`.
    """
    if isinstance(action, Mapping):
        if action.get("type") != "action":
            raise ValueError(f"Not an action event: {action!r}.")
        return _format(action["kind"], action["to"], action["added"], bool(action["all_in"]))
    if not isinstance(action, Action):
        raise TypeError(f"Expected an Action or an action event, got {type(action).__name__}.")
    if context is None:
        added = 0 if action.kind in ("fold", "check") else None
        return _format(action.kind, action.amount, added, False)
    legal, seat, street_bets, stacks = _legal_of(context)
    if action.kind == "call":
        added = legal.call_amount
        return _format("call", street_bets[seat] + added, added, added == stacks[seat])
    if action.kind in ("bet", "raise"):
        assert action.amount is not None
        kind = "bet" if max(street_bets) == 0 else "raise"
        all_in = action.amount == street_bets[seat] + stacks[seat]
        return _format(kind, action.amount, action.amount - street_bets[seat], all_in)
    return _format(action.kind, street_bets[seat], 0, False)


def _format(kind: str, to: int | None, added: int | None, all_in: bool) -> str:
    if kind == "fold":
        return "Folds"
    if kind == "check":
        return "Checks"
    if kind == "call":
        if added is None:
            return "Calls"
        return f"Calls {added:,}" + (" (all-in)" if all_in else "")
    if all_in:
        return f"All-in {to:,}"
    return f"Bets {to:,}" if kind == "bet" else f"Raises to {to:,}"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _as_chips(value: Any, what: Any) -> int:
    """Validate a positive integral chip amount (ints, NumPy ints or integral floats)."""
    if isinstance(value, (bool, np.bool_)):
        raise _chip_error(value, what)
    if isinstance(value, (float, np.floating)):
        if not float(value).is_integer():
            raise _chip_error(value, what)
        value = int(value)
    try:
        chips = operator.index(value)
    except TypeError:
        raise _chip_error(value, what) from None
    if chips <= 0:
        raise _chip_error(value, what)
    return int(chips)


def _chip_error(value: Any, what: Any) -> Exception:
    if isinstance(what, Action):
        return IllegalAction(
            f"{what.kind} needs a positive whole-chip amount, got {value!r}.", action=what
        )
    return ValueError(f"{what} must be a positive whole number of chips, got {value!r}.")


def _check_seat(seat: int) -> int:
    if seat not in (0, 1):
        raise ValueError(f"seat must be 0 or 1, got {seat!r}.")
    return int(seat)


def _copy_event(event: Mapping[str, Any]) -> dict[str, Any]:
    return {k: list(v) if isinstance(v, list) else v for k, v in event.items()}


# The fields that hold lists, by event type (the others hold only scalars).
_LIST_FIELDS: dict[str, tuple[str, ...]] = {"deal": ("cards",), "showdown": ("cards", "best")}


def _copy_hand_event(event: dict[str, Any]) -> dict[str, Any]:
    """`_copy_event` for the events a hand logs, whose list fields are known by type."""
    copy = event.copy()
    fields = _LIST_FIELDS.get(copy["type"])
    if fields is not None:
        for name in fields:
            copy[name] = list(copy[name])
    return copy


__all__ = [
    "BETTING_STREETS",
    "STREETS",
    "SHOWDOWN",
    "COMPLETE",
    "ACTION_KINDS",
    "Action",
    "LegalActions",
    "HandResult",
    "SeatView",
    "HoldemHand",
    "parse_action",
    "format_action",
]
