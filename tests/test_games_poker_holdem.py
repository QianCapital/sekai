"""Heads-up hold'em: rules, pot accounting, text moves, views and hidden information."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import numpy as np
import pytest

from sekai.envs.games.poker import (
    Action,
    HandComplete,
    HiddenInformation,
    HoldemHand,
    IllegalAction,
    OutOfTurn,
    SeatView,
    evaluate,
    format_action,
    parse_action,
    parse_cards,
)
from sekai.envs.games.poker.cards import CARD_STRINGS

# Seat 0: As Kd, seat 1: Qh Qc, flop 2c 7d 9h, turn Js, river 3s (seat 1 wins at showdown).
DECK = "AsKdQhQc2c7d9hJs3s"


def hand(**kwargs: Any) -> HoldemHand:
    kwargs.setdefault("deck", DECK)
    return HoldemHand(**kwargs)


# ----------------------------------------------------------------------
# Blinds, order of play, streets
# ----------------------------------------------------------------------


@pytest.mark.parametrize("button", [0, 1])
def test_blinds_and_order_of_play(button: int) -> None:
    h = hand(button=button)
    bb = 1 - button
    assert h.street == "preflop" and h.to_act == button
    assert h.street_bets[button] == 50 and h.street_bets[bb] == 100
    assert h.pot == 150 and h.stacks[button] == 9950 and h.stacks[bb] == 9900
    assert h.events[:2] == [
        {"type": "post", "seat": button, "blind": "small", "amount": 50, "all_in": False},
        {"type": "post", "seat": bb, "blind": "big", "amount": 100, "all_in": False},
    ]
    h.act("call")
    assert h.to_act == bb  # the big blind has the option
    assert h.legal().can_check and not h.legal().can_fold
    h.act("check")
    assert h.street == "flop" and h.board == parse_cards("2c7d9h")
    assert h.to_act == bb  # big blind first postflop
    h.act("check")
    assert h.to_act == button
    h.act("check")
    assert h.street == "turn" and h.to_act == bb and len(h.board) == 4
    h.act("check")
    h.act("check")
    assert h.street == "river" and len(h.board) == 5
    h.act("check")
    h.act("check")
    assert h.street == "showdown" and h.is_complete and h.to_act is None


def test_fixed_deck_positions() -> None:
    h = hand()
    assert h.hole(0) == parse_cards("AsKd")
    assert h.hole(1) == parse_cards("QhQc")
    assert h.board == []
    with pytest.raises(ValueError):
        HoldemHand(deck="AsKdQhQc")  # fewer than 9 cards


def test_full_hand_event_log_and_result() -> None:
    h = hand(button=0)
    h.act(Action.raise_to(300))
    h.act(Action.call())
    h.act(Action.check())
    h.act(Action.bet(400))
    h.act(Action.raise_to(1200))
    h.act(Action.call())
    h.act("check")
    h.act("allin")
    h.act("call")
    ev = h.events
    assert ev[2] == {
        "type": "action",
        "seat": 0,
        "street": "preflop",
        "kind": "raise",
        "to": 300,
        "added": 250,
        "all_in": False,
    }
    assert ev[4] == {"type": "deal", "street": "flop", "cards": ["2c", "7d", "9h"]}
    assert ev[7] == {
        "type": "action",
        "seat": 1,
        "street": "flop",
        "kind": "raise",
        "to": 1200,
        "added": 1200,
        "all_in": False,
    }
    assert {"type": "deal", "street": "turn", "cards": ["Js"]} in ev
    assert ev[-1] == {"type": "win", "seat": 1, "amount": 20000}
    showdowns = [e for e in ev if e["type"] == "showdown"]
    assert showdowns[0]["seat"] == 0  # the turn all-in aggressor shows first
    assert showdowns[1] == {
        "type": "showdown",
        "seat": 1,
        "cards": ["Qh", "Qc"],
        "rank": "One Pair, Queens",
        "best": ["Qh", "Qc", "Js", "9h", "7d"],
    }
    r = h.result
    assert r is not None
    assert r.deltas == (-10000, 10000) and r.winners == (1,) and r.showdown and r.pot == 20000
    assert r.hands[0]["rank"] == "High Card, Ace high"
    assert h.stacks == (0, 20000)
    json.dumps(ev)
    json.dumps(r.to_dict())


# ----------------------------------------------------------------------
# Bet sizing rules
# ----------------------------------------------------------------------


def test_min_raise_preflop_and_postflop() -> None:
    h = hand()
    lg = h.legal()
    assert (lg.call_amount, lg.min_raise_to, lg.max_raise_to) == (50, 200, 10000)
    h.act("raise:300")  # increment 200
    assert h.legal().min_raise_to == 500
    h.act("raise:800")  # increment 500
    assert h.legal().min_raise_to == 1300
    h.act("call")
    lg = h.legal()  # flop: minimum bet is the big blind
    assert lg.can_check and lg.min_raise_to == 100 and lg.max_raise_to == 9200
    h.act("bet:250")
    assert h.legal().min_raise_to == 500
    with pytest.raises(IllegalAction):
        h.act("raise:499")
    h.act("raise:500")
    assert h.legal().min_raise_to == 750


def test_bet_and_raise_are_synonyms_and_canonicalised() -> None:
    h = hand()
    h.act(Action.bet(300))  # a preflop "bet" is a raise
    assert h.events[-1]["kind"] == "raise"
    h.act("call")
    h.act(Action.raise_to(200))  # nothing bet on the flop yet: a bet
    assert h.events[-1]["kind"] == "bet"


def test_illegal_actions_raise() -> None:
    h = hand()
    with pytest.raises(IllegalAction):
        h.act("check")  # facing the big blind
    with pytest.raises(IllegalAction):
        h.act("raise:150")  # below the minimum raise
    with pytest.raises(IllegalAction):
        h.act("raise:10001")  # above the stack
    with pytest.raises(OutOfTurn):
        h.act("call", seat=1)
    h.act("call", seat=0)
    with pytest.raises(IllegalAction):
        h.act("fold")  # checking is free for the big blind
    with pytest.raises(IllegalAction):
        h.act("call")  # nothing to call
    with pytest.raises(IllegalAction):
        Action("raise", None)
    with pytest.raises(IllegalAction):
        Action("raise", 150.5)
    with pytest.raises(IllegalAction):
        Action("check", 100)
    with pytest.raises(IllegalAction):
        Action("limp")
    h.act("check")
    h.act("bet:100")
    h.act("fold")
    with pytest.raises(HandComplete):
        h.act("check")
    with pytest.raises(HandComplete):
        h.legal()


def test_uncalled_bet_returned_on_fold() -> None:
    h = hand()
    h.act("raise:300")
    h.act("call")
    h.act("check")
    h.act("bet:400")
    h.act("fold")
    ev = h.events
    assert ev[-2] == {"type": "return", "seat": 0, "amount": 400}
    assert ev[-1] == {"type": "win", "seat": 0, "amount": 600}
    assert h.street == "complete"
    r = h.result
    assert r is not None and r.deltas == (300, -300) and not r.showdown and r.hands == {}


def test_preflop_fold_returns_big_blind_excess() -> None:
    h = hand(button=1)
    h.act("fold")
    assert h.events[-2:] == [
        {"type": "return", "seat": 0, "amount": 50},
        {"type": "win", "seat": 0, "amount": 100},
    ]
    assert h.result is not None and h.result.deltas == (50, -50)


def test_bets_capped_at_what_the_opponent_can_match() -> None:
    h = hand(stacks=(10000, 4000))
    lg = h.legal()
    assert lg.max_raise_to == 4000
    h.act("allin")
    assert h.street_bets == (4000, 100) and h.stacks == (6000, 3900)
    assert format_action(h.events[-1]) == "Raises to 4,000"  # seat 0 is not all-in
    lg = h.legal()
    assert lg.call_amount == 3900 and not lg.can_raise
    h.act("call")
    assert h.is_complete and h.result is not None
    assert sum(h.stacks) == 14000 and h.result.pot == 8000


def test_short_all_in_does_not_reopen_betting() -> None:
    # Seat 1 (big blind) has 450 in total: a re-shove over a raise to 300 is short (+150 < 200).
    h = hand(stacks=(10000, 450))
    h.act("raise:300")
    lg = h.legal()
    assert lg.min_raise_to == 450 and lg.max_raise_to == 450  # all-in for less allowed
    h.act("allin")
    ev = h.events[-1]
    assert ev["kind"] == "raise" and ev["to"] == 450 and ev["all_in"]
    lg = h.legal()
    assert h.to_act == 0 and lg.call_amount == 150
    assert lg.can_fold and not lg.can_raise and lg.min_raise_to is None
    with pytest.raises(IllegalAction):
        h.act("raise:600")
    h.act("call")
    assert h.is_complete and h.result is not None and h.result.pot == 900


def test_all_in_runout_deals_every_street() -> None:
    h = hand()
    h.act("allin")
    h.act("call")
    types = [(e["type"], e.get("street")) for e in h.events]
    assert [t for t in types if t[0] == "deal"] == [
        ("deal", "flop"),
        ("deal", "turn"),
        ("deal", "river"),
    ]
    assert h.street == "showdown" and len(h.board) == 5
    assert h.result is not None and h.result.deltas == (-10000, 10000)


def test_call_all_in_for_less() -> None:
    h = hand(stacks=(250, 10000))  # seat 0 (button) can only call or shove for less
    h.act("call")
    h.act("check")
    lg = h.legal()  # flop: big blind (seat 1) to act, can bet up to seat 0's 150
    assert lg.max_raise_to == 150
    h.act("bet:150")
    lg = h.legal()
    assert lg.call_amount == 150 and not lg.can_raise
    h.act("call")
    assert h.is_complete


# ----------------------------------------------------------------------
# Short stacks and blinds
# ----------------------------------------------------------------------


def test_small_blind_all_in_from_posting_ends_hand() -> None:
    h = hand(stacks=(30, 10000))
    assert h.events[0] == {
        "type": "post",
        "seat": 0,
        "blind": "small",
        "amount": 30,
        "all_in": True,
    }
    assert h.is_complete and h.street == "showdown"
    assert {"type": "return", "seat": 1, "amount": 70} in h.events
    assert h.result is not None and h.result.pot == 60 and sum(h.stacks) == 10030


def test_short_big_blind_can_only_be_called_or_folded() -> None:
    h = hand(stacks=(10000, 60))
    assert h.events[1]["amount"] == 60 and h.events[1]["all_in"]
    lg = h.legal()
    assert h.to_act == 0 and lg.call_amount == 10 and not lg.can_raise
    h.act("call")
    assert h.is_complete and h.result is not None and h.result.pot == 120


def test_both_blinds_all_in() -> None:
    h = hand(stacks=(50, 80))
    assert h.is_complete and {"type": "return", "seat": 1, "amount": 30} in h.events
    assert h.result is not None and h.result.pot == 100 and sum(h.stacks) == 130


def test_small_blind_short_but_can_act() -> None:
    h = hand(stacks=(120, 10000))
    lg = h.legal()
    assert lg.call_amount == 50 and lg.min_raise_to == 120 and lg.max_raise_to == 120
    h.act("allin")  # short raise to 120
    lg = h.legal()
    assert h.to_act == 1 and lg.call_amount == 20 and not lg.can_raise
    h.act("call")
    assert h.is_complete


def test_invalid_construction() -> None:
    with pytest.raises(ValueError):
        HoldemHand(stacks=(0, 100))
    with pytest.raises(ValueError):
        HoldemHand(button=2)
    with pytest.raises(ValueError):
        HoldemHand(small_blind=200, big_blind=100)
    with pytest.raises(ValueError):
        HoldemHand(stacks=(100,))


# ----------------------------------------------------------------------
# Showdown and split pots
# ----------------------------------------------------------------------


def test_split_pot_when_board_plays() -> None:
    h = HoldemHand(deck="2c3d2h3sAsKsQsJsTs", button=0)
    h.act("raise:300")
    h.act("call")
    for _ in range(6):
        h.act("check")
    r = h.result
    assert r is not None and r.winners == (0, 1) and r.deltas == (0, 0) and r.showdown
    wins = [e for e in h.events if e["type"] == "win"]
    assert [w["seat"] for w in wins] == [1, 0]  # big blind first
    assert all(w["amount"] == 300 for w in wins)
    assert r.hands[0]["rank"] == "Royal Flush"


def test_showdown_winner_has_best_hand() -> None:
    h = hand()
    h.act("call")
    for _ in range(7):
        h.act("check")
    r = h.result
    assert r is not None and r.winners == (1,)
    assert evaluate([*h.hole(1), *h.board]) > evaluate([*h.hole(0), *h.board])
    first = next(e for e in h.events if e["type"] == "showdown")
    assert first["seat"] == 1  # no river bet: the big blind shows first


# ----------------------------------------------------------------------
# Text moves
# ----------------------------------------------------------------------


def test_parse_action() -> None:
    h = hand()
    assert parse_action("fold", h) == Action("fold")
    assert parse_action("CALL", h) == Action("call")
    assert parse_action("raise:300", h) == Action("raise", 300)
    assert parse_action("bet 1,000", h) == Action("raise", 1000)
    assert parse_action("all-in", h) == Action("raise", 10000)
    assert parse_action("allin", h) == Action("raise", 10000)
    for bad in ("check", "raise:100", "raise:abc", "dance", "raise:20000", "call 5"):
        with pytest.raises(IllegalAction):
            parse_action(bad, h)
    h.act("raise:300")
    view = h.view(1)
    assert parse_action("raise:600", view) == Action("raise", 600)
    with pytest.raises(IllegalAction):
        parse_action("call", h.view(0))  # not seat 0's turn


def test_allin_maps_to_call_when_raising_is_impossible() -> None:
    h = hand(stacks=(10000, 60))
    assert parse_action("allin", h) == Action("call")


def test_format_action() -> None:
    h = hand()
    assert format_action(Action("fold"), h) == "Folds"
    assert format_action(Action("call"), h) == "Calls 50"
    assert format_action(Action("raise", 300), h) == "Raises to 300"
    assert format_action(Action("raise", 10000), h) == "All-in 10,000"
    h.act("call")
    assert format_action(Action("check"), h) == "Checks"
    h.act("check")
    assert format_action(Action("bet", 300), h) == "Bets 300"
    h.act("bet:9900")
    assert format_action(Action("call"), h) == "Calls 9,900 (all-in)"
    assert format_action(h.events[-1]) == "All-in 9,900"
    assert format_action(Action("raise", 900)) == "Raises to 900"
    assert str(Action("raise", 900)) == "raise:900"


# ----------------------------------------------------------------------
# Views and hidden information
# ----------------------------------------------------------------------


def test_views_hide_opponent_cards_until_showdown() -> None:
    h = hand()
    v0, v1, pub = h.view(0), h.view(1), h.public_view()
    assert v0.hole == tuple(parse_cards("AsKd")) and v0.opponent_hole is None
    assert v1.hole == tuple(parse_cards("QhQc")) and v1.opponent_hole is None
    assert pub.holes == (None, None) and pub.hole == () and pub.legal is None
    assert v0.legal is not None and v1.legal is None  # only the seat to act gets legal
    with pytest.raises(HiddenInformation):
        h.reveal()
    with pytest.raises(dataclasses.FrozenInstanceError):
        v0.board = ()  # type: ignore[misc]
    h.act("allin")
    h.act("call")
    v0 = h.view(0)
    assert v0.opponent_hole == tuple(parse_cards("QhQc"))
    assert h.public_view().holes == (tuple(parse_cards("AsKd")), tuple(parse_cards("QhQc")))
    assert h.reveal().holes == h.public_view().holes


def test_reveal_after_fold_shows_both_hands_but_views_do_not() -> None:
    h = hand()
    h.act("fold")
    assert h.view(1).opponent_hole is None and h.public_view().holes == (None, None)
    assert h.reveal().holes == (tuple(parse_cards("AsKd")), tuple(parse_cards("QhQc")))


def test_view_is_a_detached_snapshot() -> None:
    h = hand()
    v = h.view(0)
    v.events[0]["amount"] = 999_999  # mutating a view's copy cannot touch the hand
    assert h.events[0]["amount"] == 50
    h.events[0]["amount"] = 1
    assert h.events[0]["amount"] == 50
    assert not any(isinstance(getattr(v, f), HoldemHand) for f in v.__dataclass_fields__)
    d = v.to_dict()
    json.dumps(d)
    assert d["hole"] == ["As", "Kd"] and d["holes"][1] is None


def test_from_view_reconstructs_public_state() -> None:
    h = hand()
    h.act("raise:300")
    h.act("call")
    h.act("bet:200")
    view = h.view(0)
    for seed in range(5):
        sample = HoldemHand.from_view(view, rng=seed)
        sv = sample.view(0)
        assert sv.hole == view.hole and sv.board == view.board
        assert sv.stacks == view.stacks and sv.street_bets == view.street_bets
        assert sv.legal == view.legal and sv.to_act == view.to_act
        assert [e for e in sv.events] == [e for e in view.events]
        opp = sample.hole(1)
        assert not set(opp) & set(view.hole + view.board)


def test_copy_is_independent() -> None:
    h = hand()
    c = h.copy()
    c.act("fold")
    assert c.is_complete and not h.is_complete and h.to_act == 0
    assert len(h.events) == 2


# ----------------------------------------------------------------------
# Property-style random play
# ----------------------------------------------------------------------


def _reference_min_increment(events: list[dict], street: str, bb: int) -> int:
    """Largest full bet/raise increment on ``street`` per the rules, recomputed from events."""
    inc = bb
    bets = {0: 0, 1: 0}
    in_street = street == "preflop"
    for e in events:
        if e["type"] == "deal":
            in_street = e["street"] == street
            bets = {0: 0, 1: 0}
            continue
        if not in_street:
            continue
        if e["type"] == "post":
            bets[e["seat"]] = e["amount"]
        elif e["type"] == "action":
            current = max(bets.values())
            if e["kind"] in ("bet", "raise") and e["to"] - current >= inc:
                inc = e["to"] - current
            bets[e["seat"]] = e["to"]
    return inc


def _allowed_card_strings(view: SeatView) -> set[str]:
    allowed = {CARD_STRINGS[c] for c in view.board}
    for cards in view.holes:
        if cards is not None:
            allowed |= {CARD_STRINGS[c] for c in cards}
    return allowed


def _card_strings_in(obj: object) -> set[str]:
    found: set[str] = set()
    if isinstance(obj, str):
        if obj in CARD_STRINGS:
            found.add(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            found |= _card_strings_in(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            found |= _card_strings_in(v)
    return found


def _check_invariants(h: HoldemHand, total: int) -> None:
    stacks, committed, bets = h.stacks, h.committed, h.street_bets
    assert min(stacks) >= 0 and min(committed) >= 0
    assert all(bets[s] <= committed[s] for s in (0, 1))
    if not h.is_complete:
        assert sum(stacks) + h.pot == total
        seat = h.to_act
        assert seat in (0, 1)
        opp = 1 - seat
        lg = h.legal()
        current = max(bets)
        assert lg.can_check == (bets[seat] == current)
        assert lg.can_fold == (not lg.can_check)
        assert lg.call_amount == min(current - bets[seat], stacks[seat])
        assert stacks[seat] > 0
        if lg.can_raise:
            assert lg.max_raise_to is not None and lg.min_raise_to is not None
            assert current < lg.min_raise_to <= lg.max_raise_to
            assert lg.max_raise_to == min(bets[seat] + stacks[seat], bets[opp] + stacks[opp])
            ref = _reference_min_increment(h.events, h.street, h.big_blind)
            assert lg.min_raise_to == min(current + ref, lg.max_raise_to)
        else:
            # Raising is impossible only for a reason the rules give.
            last = [e for e in h.events if e["type"] == "action" and e["street"] == h.street]
            acted = any(e["seat"] == seat for e in last)
            short = bool(last) and last[-1]["seat"] == opp and last[-1]["all_in"]
            assert stacks[seat] <= lg.call_amount or stacks[opp] == 0 or (acted and short)
    else:
        r = h.result
        assert r is not None
        assert sum(stacks) == total
        assert r.deltas[0] + r.deltas[1] == 0
        assert all(stacks[s] == h.starting_stacks[s] + r.deltas[s] for s in (0, 1))
        assert r.pot == h.pot
        wins = sum(e["amount"] for e in h.events if e["type"] == "win")
        assert wins == r.pot
        if r.showdown:
            assert committed[0] == committed[1] and len(h.board) == 5
            v = [evaluate([*h.hole(s), *h.board]) for s in (0, 1)]
            assert all(v[w] == max(v) for w in r.winners)
        else:
            folds = [e for e in h.events if e["type"] == "action" and e["kind"] == "fold"]
            assert len(folds) == 1 and r.winners == (1 - folds[0]["seat"],)
    # Per-seat accounting from the public log matches the committed chips.
    for s in (0, 1):
        put = sum(e["amount"] for e in h.events if e["type"] == "post" and e["seat"] == s)
        put += sum(e["added"] for e in h.events if e["type"] == "action" and e["seat"] == s)
        put -= sum(e["amount"] for e in h.events if e["type"] == "return" and e["seat"] == s)
        assert put == committed[s]
    # Hidden information: every view shows only what that viewer may know.
    for viewer in (0, 1, None):
        view = h.public_view() if viewer is None else h.view(viewer)
        if viewer is not None and not (h.is_complete and h.result and h.result.showdown):
            assert view.opponent_hole is None
        assert _card_strings_in(view.to_dict()) <= _allowed_card_strings(view)
        assert view.board == tuple(h.board)


def _random_action(h: HoldemHand, rng: np.random.Generator) -> Action:
    lg = h.legal()
    options = []
    if lg.can_fold:
        options.append(Action("fold"))
    options.append(Action("check") if lg.can_check else Action("call"))
    if lg.can_raise:
        assert lg.min_raise_to is not None and lg.max_raise_to is not None
        for to in {
            lg.min_raise_to,
            lg.max_raise_to,
            int(rng.integers(lg.min_raise_to, lg.max_raise_to + 1)),
        }:
            options.append(Action("raise", to))
    weights = np.array([0.15 if a.kind == "fold" else 1.0 for a in options])
    return options[int(rng.choice(len(options), p=weights / weights.sum()))]


@pytest.mark.parametrize("block", range(4))
def test_random_hands_respect_invariants(block: int) -> None:
    rng = np.random.default_rng(1000 + block)
    for _ in range(500):
        sb = int(rng.integers(1, 101))
        bb = int(rng.integers(sb, 2 * sb + 1))
        short = rng.random() < 0.3
        hi = 5 * bb if short else 30000
        stacks = (int(rng.integers(1, hi + 1)), int(rng.integers(1, 30001)))
        if rng.random() < 0.5:
            stacks = stacks[::-1]
        h = HoldemHand(
            stacks=stacks, button=int(rng.integers(2)), small_blind=sb, big_blind=bb, rng=rng
        )
        total = sum(stacks)
        _check_invariants(h, total)
        steps = 0
        while not h.is_complete:
            h.act(_random_action(h, rng))
            _check_invariants(h, total)
            steps += 1
            assert steps < 500
        assert h.street in ("showdown", "complete")
