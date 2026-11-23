"""Blackjack: the table (round flow, payouts, rule details, hidden information, accounting)."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence

import numpy as np
import pytest

from sekai.envs.games.blackjack import BlackjackTable, Rules, Shoe
from sekai.envs.games.blackjack.errors import BlackjackError, IllegalAction, NotYourTurn

DEFAULT_RULES = Rules()


def stacked(
    cards: Sequence[str],
    *,
    seats: int = 1,
    rules: Rules = DEFAULT_RULES,
    bankroll: float | Sequence[float] = 1000,
    seed: int = 0,
) -> BlackjackTable:
    """A table whose shoe deals ``cards`` first (seat cards, dealer up, ..., hole, then draws)."""
    shoe = Shoe(rules.decks, seed=seed, penetration=rules.penetration)
    shoe.stack_top(cards)
    return BlackjackTable(rules, seats=seats, bankroll=bankroll, shoe=shoe)


def types(table: BlackjackTable) -> list[str]:
    return [e["type"] for e in table.events]


def assert_conserved(table: BlackjackTable) -> None:
    total = sum(table.bankrolls) + table.chips_on_table + table.house_net
    assert total == pytest.approx(table.chips_in, abs=1e-9)


# ----------------------------------------------------------------------------------------
# Payouts
# ----------------------------------------------------------------------------------------


def test_natural_pays_three_to_two() -> None:
    t = stacked(["Ah", "9c", "Kd", "7s"])
    t.act("bet:10")
    assert t.phase == "betting" and t.rounds_played == 1
    assert t.bankrolls == [1015]
    hand = t.view()["seats"][0]["hands"][0]
    assert hand["blackjack"] and hand["status"] == "blackjack"
    assert hand["outcome"] == "blackjack" and hand["payout"] == 15
    assert "dealer_draw" not in types(t)  # nobody needed the dealer: reveal only
    assert t.view()["dealer"]["revealed"]
    assert t.house_net == -15
    assert_conserved(t)


def test_natural_on_odd_bet_pays_half_chips() -> None:
    t = stacked(["Ah", "9c", "Kd", "7s"])
    t.act("bet:15")
    assert t.bankrolls == [1022.5]
    assert t.last_round["seats"][0]["net"] == 22.5


def test_six_to_five_payout() -> None:
    t = stacked(["Ah", "9c", "Kd", "7s"], rules=Rules(blackjack_payout=1.2))
    t.act("bet:25")
    assert t.bankrolls == [1030]


def test_double_win_pays_twice() -> None:
    t = stacked(["6h", "6c", "5d", "Ts", "Th", "Tc"])
    t.act("bet:10")
    assert t.legal_actions() == ["hit", "stand", "double", "surrender"]
    t.act("double")
    hand = t.last_round["seats"][0]["hands"][0]
    assert hand["doubled"] and hand["bet"] == 20 and hand["status"] == "done"
    assert hand["outcome"] == "win" and hand["payout"] == 20
    assert t.last_round["dealer"]["bust"]
    assert t.bankrolls == [1020]


def test_surrender_forfeits_half() -> None:
    t = stacked(["Th", "Tc", "6d", "7s"])
    t.act("bet:10")
    t.act("surrender")
    hand = t.last_round["seats"][0]["hands"][0]
    assert hand["status"] == "surrendered" and hand["outcome"] == "surrender"
    assert hand["payout"] == -5
    assert t.bankrolls == [995]
    assert "dealer_draw" not in types(t)


def test_plain_win_loss_and_push() -> None:
    # Seat 0: 20 v dealer 19 wins; seat 1: 19 pushes; seat 2: 17 loses.
    t = stacked(["Th", "Tc", "Td", "9s", "Kh", "9h", "8c", "Ts"], seats=3, bankroll=500)
    for _ in range(3):
        t.act("bet:10")
    view = t.view()
    assert view["dealer"]["cards"] == ["9s", None] and view["dealer"]["total"] == 9
    for _ in range(3):
        t.act("stand")
    nets = [s["net"] for s in t.last_round["seats"]]
    assert nets == [10, 0, -10]
    assert t.last_round["dealer"]["total"] == 19
    assert t.bankrolls == [510, 500, 490]


# ----------------------------------------------------------------------------------------
# Dealer peek and blackjacks
# ----------------------------------------------------------------------------------------


def test_dealer_blackjack_after_peek_settles_at_once() -> None:
    # Seat 0 holds 19, seat 1 a natural; dealer A up, K in the hole.
    t = stacked(["Th", "Ac", "As", "9d", "Kd", "Ks"], seats=2)
    t.act("bet:10")
    t.act("bet:20")
    assert t.phase == "betting" and t.rounds_played == 1
    assert "action" not in types(t)
    peek = [e for e in t.events if e["type"] == "peek"]
    assert peek and peek[0]["blackjack"] is True
    outcomes = [s["hands"][0]["outcome"] for s in t.last_round["seats"]]
    assert outcomes == ["lose", "push"]
    assert t.bankrolls == [990, 1000]
    assert t.last_round["dealer"]["blackjack"]


def test_peek_without_blackjack_pays_naturals_then_plays() -> None:
    t = stacked(["Ah", "9c", "Td", "Kd", "7s", "8s"], seats=2)
    t.act("bet:10")
    t.act("bet:10")
    peek = [e for e in t.events if e["type"] == "peek"]
    assert peek == [peek[0]] and peek[0]["blackjack"] is False
    assert t.phase == "playing" and t.to_act == 1  # seat 0's natural is paid and skips play
    assert t.view()["seats"][0]["hands"][0]["outcome"] == "blackjack"
    assert t.bankrolls[0] == 1015
    assert t.view()["dealer"]["cards"] == ["Td", None]


def test_no_peek_for_low_up_card() -> None:
    t = stacked(["Th", "5c", "9d", "Ts"])
    t.act("bet:10")
    assert "peek" not in types(t)


def test_no_peek_rules_dealer_blackjack_takes_doubles() -> None:
    rules = Rules(dealer_peeks=False)
    t = stacked(["5h", "Tc", "6d", "As", "9h"], rules=rules)
    t.act("bet:10")
    assert "peek" not in types(t) and t.phase == "playing"
    t.act("double")
    assert t.last_round["dealer"]["blackjack"]
    hand = t.last_round["seats"][0]["hands"][0]
    assert hand["outcome"] == "lose" and hand["payout"] == -20
    assert t.bankrolls == [980]


def test_no_peek_rules_natural_waits_for_the_reveal() -> None:
    rules = Rules(dealer_peeks=False)
    t = stacked(["Ah", "Ac", "Kd", "6s"], rules=rules)
    t.act("bet:10")
    ev = types(t)
    assert ev.index("dealer_reveal") < ev.index("settle")
    assert t.bankrolls == [1015]
    assert "dealer_draw" not in ev


@pytest.mark.parametrize(
    ("hole", "outcome", "payout"),
    [("Kc", "lose", -10), ("6c", "surrender", -5)],
)
def test_no_peek_rules_surrender_waits_for_the_reveal(hole: str, outcome: str, payout: int) -> None:
    # Late surrender: half the bet back once the dealer turns out to have no blackjack;
    # without the peek, a dealer blackjack takes the whole bet.
    # Seat 0: 9 and 7; seat 1: 5 and 6; the dealer: an ace up, and the hole card.
    t = stacked(["9h", "5c", "As", "7d", "6h", hole], seats=2, rules=Rules(dealer_peeks=False))
    t.act("bet:10")
    t.act("bet:10")
    t.act("surrender")
    view = t.view()
    assert view["seats"][0]["hands"][0]["status"] == "surrendered"
    assert view["seats"][0]["hands"][0]["outcome"] is None  # (seat 1 still to act)
    t.act("stand")
    hand = t.last_round["seats"][0]["hands"][0]
    assert hand["outcome"] == outcome and hand["payout"] == payout
    assert "dealer_draw" not in types(t) or hole == "6c"
    assert_conserved(t)
    # Against a low up card, or with the peek, it settles at once, as before.
    settled_at_once = [
        (["9h", "6s", "7d", "Kc"], Rules(dealer_peeks=False)),
        (["9h", "As", "7d", "6c"], Rules()),
    ]
    for cards, rules in settled_at_once:
        t = stacked(cards, rules=rules)
        t.act("bet:10")
        t.act("surrender")
        assert t.last_round["seats"][0]["hands"][0]["payout"] == -5


def test_forfeit_loses_the_seats_hands_and_play_moves_on() -> None:
    # Seat 0: 8 and 8; seat 1: 10 and 9; the dealer: 9 up, 7 in the hole.
    t = stacked(["8h", "Tc", "9c", "8d", "9h", "7s", "3c"], seats=2)
    with pytest.raises(IllegalAction, match="only a seat playing"):
        t.forfeit()  # betting: nothing to give up yet
    t.act("bet:10")
    t.act("bet:20")
    t.act("split")  # seat 0's first eight gets a three
    with pytest.raises(NotYourTurn):
        t.forfeit(1)
    t.forfeit(0)
    assert t.to_act == 1 and "forfeit" in types(t)
    hands = t.view()["seats"][0]["hands"]
    assert [h["outcome"] for h in hands] == ["lose", "lose"] and t.bankrolls[0] == 980
    t.act("stand")  # seat 1's 19 against the dealer's 16, who draws
    assert t.phase == "betting" and t.last_round["seats"][0]["net"] == -20
    assert_conserved(t)


def test_a_round_that_needs_more_cards_than_the_shoe_raises_a_blackjack_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Possible only with a small shoe and many hands (one deck, seven seats splitting to
    # eight hands each); here the shoe is emptied by hand, as if every card were dealt.
    t = stacked(["9h", "6s", "7d", "Kc"])
    t.act("bet:10")
    monkeypatch.setattr(t._shoe, "_cards", [])
    monkeypatch.setattr(t._shoe, "shuffle", lambda **kwargs: None)
    with pytest.raises(BlackjackError, match="Every card of the 6-deck shoe is on the table"):
        t.act("hit")


# ----------------------------------------------------------------------------------------
# Splits, doubles, surrender and auto-stand
# ----------------------------------------------------------------------------------------


def test_split_eights_with_double_after_split() -> None:
    t = stacked(["8h", "6c", "8d", "Ts", "3h", "Tc", "Kd", "5s"])
    t.act("bet:10")
    assert "split" in t.legal_actions()
    t.act("split")
    view = t.view()
    hands = view["seats"][0]["hands"]
    assert [h["cards"] for h in hands] == [["8h", "3h"], ["8d"]]
    assert all(h["from_split"] for h in hands)
    assert view["active_hand"] == 0
    assert "double" in t.legal_actions() and "surrender" not in t.legal_actions()
    t.act("double")  # 8 + 3 + T = 21
    assert t.active_hand == 1 and t.view()["seats"][0]["hands"][1]["cards"] == ["8d", "Kd"]
    t.act("stand")
    seat = t.last_round["seats"][0]
    assert [h["outcome"] for h in seat["hands"]] == ["push", "lose"]  # dealer 6 + T + 5 = 21
    assert seat["net"] == -10 and seat["wagered"] == 30
    assert t.bankrolls == [990]


def test_no_double_after_split_without_das() -> None:
    t = stacked(["8h", "6c", "8d", "Ts", "3h"], rules=Rules(double_after_split=False))
    t.act("bet:10")
    t.act("split")
    assert t.view()["seats"][0]["hands"][0]["cards"] == ["8h", "3h"]
    assert t.legal_actions() == ["hit", "stand"]


def test_split_aces_get_one_card_and_21_is_not_blackjack() -> None:
    t = stacked(["Ah", "7c", "Ad", "Ts", "Kh", "5c"])
    t.act("bet:10")
    t.act("split")
    assert t.phase == "betting"  # both split aces were finished automatically
    seat = t.last_round["seats"][0]
    assert [h["cards"] for h in seat["hands"]] == [["Ah", "Kh"], ["Ad", "5c"]]
    assert [h["status"] for h in seat["hands"]] == ["done", "done"]
    assert [h["outcome"] for h in seat["hands"]] == ["win", "lose"]
    assert [h["payout"] for h in seat["hands"]] == [10, -10]  # A+K after a split pays 1:1
    assert t.bankrolls == [1000]


def test_split_aces_no_resplit_by_default_but_allowed_with_rsa() -> None:
    cards = ["Ah", "7c", "Ad", "Ts", "As", "5c", "9h"]
    t = stacked(cards)
    t.act("bet:10")
    t.act("split")  # first ace draws another ace: no resplit, so it is done at soft 12
    assert t.phase == "betting"
    t2 = stacked(cards, rules=Rules(resplit_aces=True))
    t2.act("bet:10")
    t2.act("split")
    assert t2.phase == "playing" and t2.legal_actions() == ["stand", "split"]
    t2.act("split")
    assert len(t2.view()["seats"][0]["hands"]) == 3


def test_hit_split_aces_rule() -> None:
    t = stacked(["Ah", "7c", "Ad", "Ts", "5h"], rules=Rules(hit_split_aces=True))
    t.act("bet:10")
    t.act("split")
    assert t.phase == "playing" and "hit" in t.legal_actions()


def test_resplit_to_four_hands_maximum() -> None:
    t = stacked(["8h", "6c", "8d", "Ts", "8s", "8c", "8h"])
    t.act("bet:10")
    t.act("split")
    t.act("split")
    t.act("split")
    view = t.view()
    assert len(view["seats"][0]["hands"]) == 4
    assert view["seats"][0]["hands"][0]["cards"] == ["8h", "8h"]
    assert "split" not in t.legal_actions()
    assert t.bankrolls == [960]


def test_split_by_value() -> None:
    t = stacked(["Kh", "6c", "Td", "Ts"])
    t.act("bet:10")
    assert "split" in t.legal_actions()


def test_split_and_double_need_the_chips() -> None:
    t = stacked(["8h", "6c", "8d", "Ts"], bankroll=15)
    t.act("bet:10")
    assert t.legal_actions() == ["hit", "stand", "surrender"]


def test_no_surrender_or_double_after_a_hit() -> None:
    t = stacked(["2h", "Tc", "3d", "8s", "4c"])
    t.act("bet:10")
    t.act("hit")
    assert t.legal_actions() == ["hit", "stand"]


def test_no_surrender_rule() -> None:
    t = stacked(["Th", "Tc", "6d", "7s"], rules=Rules(surrender=None))
    t.act("bet:10")
    assert "surrender" not in t.legal_actions()


def test_hitting_to_21_auto_stands() -> None:
    t = stacked(["5h", "9c", "6d", "8s", "Tc"])
    t.act("bet:10")
    t.act("hit")
    assert t.phase == "betting"
    assert t.last_round["seats"][0]["hands"][0]["status"] == "stood"
    assert t.bankrolls == [1010]


def test_all_hands_bust_dealer_only_reveals() -> None:
    t = stacked(["Th", "9c", "6d", "5s", "Kc"])
    t.act("bet:10")
    t.act("hit")
    assert t.last_round["seats"][0]["hands"][0]["status"] == "bust"
    assert "dealer_draw" not in types(t) and "dealer_reveal" in types(t)
    assert t.last_round["dealer"]["cards"] == ["9c", "5s"]


@pytest.mark.parametrize(("h17", "net"), [(False, 10), (True, 0)])
def test_soft_17(h17: bool, net: int) -> None:
    t = stacked(["Th", "6c", "9d", "As", "2h"], rules=Rules(dealer_hits_soft_17=h17))
    t.act("bet:10")
    t.act("stand")
    assert t.last_round["seats"][0]["net"] == net
    assert t.last_round["dealer"]["total"] == (19 if h17 else 17)


# ----------------------------------------------------------------------------------------
# Betting, turn order and errors
# ----------------------------------------------------------------------------------------


def test_bet_validation() -> None:
    t = BlackjackTable(seats=1, bankroll=100, seed=0)
    assert t.legal_actions() == {"kind": "bet", "min": 10, "max": 100, "step": 5}
    for bad in ("bet:12", "bet:5", "bet:105", "bet:abc", "bet:10.5", "hit", "deal"):
        with pytest.raises(IllegalAction):
            t.act(bad)
    big = BlackjackTable(seats=1, bankroll=10_000, seed=0)
    with pytest.raises(IllegalAction, match="table maximum"):
        big.act("bet:505")
    t.act("bet:100")
    assert t.bankrolls == [0]


def test_turn_order_and_wrong_seat() -> None:
    t = BlackjackTable(seats=2, seed=0)
    assert t.to_act == 0
    with pytest.raises(NotYourTurn):
        t.act("bet:10", seat=1)
    t.act("bet:10", seat=0)
    assert t.to_act == 1
    t.act("bet:10", seat=1)
    assert t.phase == "playing"
    with pytest.raises(IllegalAction):
        t.act("bet:10")
    with pytest.raises(IllegalAction):
        t.act("fold")


def test_broke_seat_is_out_and_skipped() -> None:
    t = BlackjackTable(seats=2, bankroll=[5, 1000], seed=0)
    assert t.to_act == 1 and t.is_out(0)
    assert t.view()["seats"][0]["out"]
    t.act("bet:10")
    assert t.view()["seats"][0]["hands"] == []
    assert t.last_round is None or not t.last_round["seats"][0]["played"]


def test_game_over_and_rebuy() -> None:
    t = BlackjackTable(seats=1, bankroll=10, seed=1)
    while not t.game_over:
        legal = t.legal_actions()
        t.act("bet:10" if isinstance(legal, dict) else "stand")
    assert t.legal_actions() == [] and t.to_act is None
    with pytest.raises(IllegalAction):
        t.act("bet:10")
    t.set_bankroll(0, 50)
    assert not t.game_over and t.to_act == 0
    assert_conserved(t)


def test_set_bankroll_only_between_rounds() -> None:
    t = stacked(["Th", "9c", "6d", "5s"], seats=2)
    t.act("bet:10")
    with pytest.raises(IllegalAction):
        t.set_bankroll(0, 10)  # seat 0 has bet
    t.set_bankroll(1, 2000)
    t.act("bet:10")
    assert t.phase == "playing"
    with pytest.raises(IllegalAction):
        t.set_bankroll(1, 10)
    assert t.chips_in == 3000
    assert_conserved(t)


# ----------------------------------------------------------------------------------------
# Views, events, history
# ----------------------------------------------------------------------------------------


def test_view_schema_is_json_friendly() -> None:
    t = BlackjackTable(seats=2, seed=4)
    for _ in range(3):
        while t.phase == "betting":
            t.act("bet:15")
        while t.phase == "playing":
            t.act("stand")
    t.act("bet:10")
    view = t.view(1)
    assert json.loads(json.dumps(view)) == view
    assert set(view) >= {
        "phase",
        "round_no",
        "to_act",
        "active_hand",
        "dealer",
        "seats",
        "shoe",
        "running_count",
        "true_count",
        "last_round",
        "rules",
        "legal_actions",
        "remaining_counts",
        "viewer",
    }
    assert view["viewer"] == 1 and view["round_no"] == 4
    assert set(view["dealer"]) == {"cards", "total", "soft", "blackjack", "bust", "revealed"}
    seat = view["seats"][0]
    assert set(seat) == {"seat", "bankroll", "bet", "out", "hands"}
    assert seat["bet"] == 10
    assert set(view["shoe"]) == {
        "decks",
        "remaining",
        "total",
        "penetration",
        "dealt_fraction",
        "reshuffle_next",
    }
    hand_keys = {
        "cards",
        "total",
        "soft",
        "blackjack",
        "bust",
        "doubled",
        "from_split",
        "bet",
        "status",
        "outcome",
        "payout",
    }
    for s in view["seats"]:
        for h in s["hands"]:
            assert set(h) == hand_keys
    assert view["last_round"]["round"] == 3
    assert json.loads(json.dumps(t.events)) == t.events


def test_finished_round_stays_visible_until_next_deal() -> None:
    t = stacked(["Th", "9c", "9d", "8s"], seats=2)
    t.act("bet:10")
    t.act("bet:10")
    t.act("stand")
    t.act("stand")
    view = t.view()
    assert view["phase"] == "betting"
    assert view["dealer"]["revealed"] and view["dealer"]["cards"][1] is not None
    assert view["seats"][0]["hands"][0]["outcome"] is not None
    assert view["seats"][0]["bet"] is None
    t.act("bet:10")
    assert t.view()["seats"][0]["bet"] == 10 and t.view()["seats"][0]["hands"]
    t.act("bet:10")
    assert t.view()["dealer"]["revealed"] is False


def test_event_log() -> None:
    t = stacked(["Th", "9c", "6d", "5s", "4h", "Kc"])  # 16, hit to 20; dealer 14, draws K
    t.act("bet:10")
    t.act("hit")
    t.act("stand")
    ev = t.events
    assert [e["seq"] for e in ev] == list(range(len(ev)))
    assert [e["type"] for e in ev] == [
        "shuffle",
        "bet",
        "deal",
        "deal",
        "deal",
        "deal",
        "action",
        "deal",
        "action",
        "dealer_reveal",
        "dealer_draw",
        "settle",
        "round_end",
    ]
    deals = [e for e in ev if e["type"] == "deal"]
    assert deals[1] == {**deals[1], "to": "dealer", "card": "9c"}
    assert deals[3]["to"] == "dealer" and deals[3]["card"] is None
    reveal, draw, settle, end = ev[-4:]
    assert reveal["card"] == "5s" and reveal["cards"] == ["9c", "5s"] and reveal["total"] == 14
    assert draw["card"] == "Kc" and draw["total"] == 24
    assert settle == {**settle, "seat": 0, "hand": 0, "outcome": "win", "payout": 10}
    assert end["net"] == [10] and end["dealer_total"] == 24
    assert {e["round"] for e in ev} == {1}
    assert t.events_since(ev[-3]["seq"]) == ev[-2:]


def test_round_numbers_and_history() -> None:
    t = BlackjackTable(seats=1, seed=2, max_history=3)
    assert t.round_no == 1 and t.last_round is None
    for _ in range(5):
        t.act("bet:10")
        while t.phase == "playing":
            t.act("stand")
    assert t.round_no == 6 and t.rounds_played == 5
    assert [r["round"] for r in t.history] == [3, 4, 5]


def test_bounded_event_log() -> None:
    t = BlackjackTable(seats=1, seed=2, max_events=10)
    for _ in range(5):
        t.act("bet:10")
        while t.phase == "playing":
            t.act("stand")
    assert len(t.events) == 10


# ----------------------------------------------------------------------------------------
# Hidden information
# ----------------------------------------------------------------------------------------


def test_hole_card_never_leaks() -> None:
    # Identical tables except for the dealer's hole card (7 v 8, neither a blackjack).
    a = stacked(["Th", "9c", "6d", "7s"], seed=9)
    b = stacked(["Th", "9c", "6d", "8s"], seed=9)
    for t in (a, b):
        t.act("bet:10")
    assert a.view(0) == b.view(0)
    assert a.events == b.events
    assert a.public_remaining_counts() == b.public_remaining_counts()
    assert (a.running_count, a.true_count) == (b.running_count, b.true_count)
    text = json.dumps(a.view(0)) + json.dumps(a.events)
    assert "7s" not in text
    assert a.view()["dealer"]["cards"] == ["9c", None]
    assert a.view()["dealer"]["total"] == 9 and not a.view()["dealer"]["blackjack"]
    # Once revealed by play, the hole card is public.
    a.act("stand")
    assert a.view()["dealer"]["cards"][:2] == ["9c", "7s"]


def test_hole_card_with_ten_up_does_not_leak_through_peek() -> None:
    a = stacked(["Th", "Kc", "6d", "7s"], seed=1)
    b = stacked(["Th", "Kc", "6d", "9s"], seed=1)
    for t in (a, b):
        t.act("bet:10")
    assert a.view(0) == b.view(0) and a.events == b.events


def test_view_does_not_contain_shoe_order() -> None:
    t = BlackjackTable(seats=2, seed=3)
    t.act("bet:10")
    t.act("bet:10")
    view = t.view()
    upcoming = list(reversed(t._shoe._cards))[:15]  # private, inspected by the test only
    for key in ("shoe", "rules", "remaining_counts"):
        text = json.dumps(view[key])
        assert not any(c in text for c in upcoming)


# ----------------------------------------------------------------------------------------
# Shuffles
# ----------------------------------------------------------------------------------------


def test_reshuffle_at_penetration() -> None:
    rules = Rules(decks=1, penetration=0.5)
    t = BlackjackTable(rules, seats=1, seed=0)
    shuffled_rounds = []
    for _ in range(20):
        t.act("bet:10")
        while t.phase == "playing":
            t.act("stand")
        if t.last_round["shuffled"]:
            shuffled_rounds.append(t.last_round["round"])
            # After a shuffle only this round's cards have been seen.
            seen = sum(len(h["cards"]) for h in t.last_round["seats"][0]["hands"])
            seen += len(t.last_round["dealer"]["cards"])
            assert sum(t.public_remaining_counts()) == 52 - seen
    assert shuffled_rounds
    shuffles = [e for e in t.events if e["type"] == "shuffle"]
    assert {e["reason"] for e in shuffles} >= {"new_shoe", "penetration"}


def test_mid_round_reshuffle_when_the_shoe_runs_dry() -> None:
    rules = Rules(decks=1, penetration=0.95, max_hands=8)
    rng = np.random.default_rng(0)
    t = BlackjackTable(rules, seats=4, bankroll=10_000, seed=0)
    empties = 0
    for _ in range(400):
        while t.phase == "betting":
            t.act("bet:10")
        while t.phase == "playing":
            legal = t.legal_actions()
            pick = "split" if "split" in legal else ("hit" if rng.random() < 0.6 else "stand")
            t.act(pick if pick in legal else "stand")
        empties += sum(
            1
            for e in t.events
            if e["type"] == "shuffle"
            and e["reason"] == "empty"
            and e["round"] == t.last_round["round"]
        )
        # One deck: no card may appear twice within a round.
        cards = list(t.last_round["dealer"]["cards"])
        for s in t.last_round["seats"]:
            for h in s["hands"]:
                cards.extend(h["cards"])
        assert max(Counter(cards).values()) == 1
        assert_conserved(t)
    assert empties > 0


# ----------------------------------------------------------------------------------------
# Accounting invariant over random sessions
# ----------------------------------------------------------------------------------------

EXPECTED_PAYOUT = {
    "win": lambda bet: bet,
    "lose": lambda bet: -bet,
    "push": lambda bet: 0,
    "blackjack": lambda bet: bet * 1.5,
    "surrender": lambda bet: -bet / 2,
}


def run_random_session(seed: int, rounds: int, rules: Rules = DEFAULT_RULES) -> BlackjackTable:
    rng = np.random.default_rng(seed)
    t = BlackjackTable(rules, seats=2, bankroll=300, seed=seed)
    start = list(t.bankrolls)
    while t.rounds_played < rounds:
        if t.game_over:
            t.set_bankroll(int(rng.integers(2)), 200)
            start = list(t.bankrolls)
            continue
        legal = t.legal_actions()
        if isinstance(legal, dict):
            steps = (legal["max"] - legal["min"]) // legal["step"]
            amount = legal["min"] + legal["step"] * int(rng.integers(0, min(steps, 12) + 1))
            t.act(f"bet:{amount}")
        else:
            t.act(legal[int(rng.integers(len(legal)))])
        assert_conserved(t)
        settled = t.phase == "betting" and t.last_round and t.last_round["round"] == t.round_no - 1
        if settled and all(s["bet"] is None for s in t.view()["seats"]):
            summary = t.last_round
            for s in summary["seats"]:
                i = s["seat"]
                if not s["played"]:
                    assert t.bankrolls[i] == start[i]
                    continue
                payouts = [h["payout"] for h in s["hands"]]
                assert s["net"] == pytest.approx(sum(payouts))
                assert t.bankrolls[i] - start[i] == pytest.approx(s["net"])
                for h in s["hands"]:
                    expected = EXPECTED_PAYOUT[h["outcome"]](h["bet"])
                    assert h["payout"] == pytest.approx(expected)
                    if h["outcome"] == "blackjack":
                        assert not h["from_split"] and len(h["cards"]) == 2
            assert t.chips_on_table == 0
            start = list(t.bankrolls)
    return t


@pytest.mark.parametrize("seed", range(12))
def test_accounting_invariant_random_sessions(seed: int) -> None:
    rules = [
        Rules(),
        Rules(dealer_hits_soft_17=True, double_after_split=False),
        Rules(dealer_peeks=False, surrender=None, resplit_aces=True),
        Rules(decks=2, penetration=0.9, blackjack_payout=1.5, max_hands=3),
    ][seed % 4]
    t = run_random_session(seed, 150, rules)
    settles = [e for e in t.events if e["type"] == "settle"]
    assert settles and {e["outcome"] for e in settles} <= set(EXPECTED_PAYOUT)
    assert_conserved(t)


def test_rng_factory_gives_every_shuffle_a_fresh_stream() -> None:
    calls = []

    def factory() -> np.random.Generator:
        calls.append(1)
        return np.random.default_rng(len(calls))

    shoe = Shoe(6, rng_factory=factory)
    first = [shoe.draw() for _ in range(20)]
    shoe.shuffle()
    second = [shoe.draw() for _ in range(20)]
    assert len(calls) == 2 and first != second
    # the same factory sequence reproduces the same shoes
    calls.clear()
    again = Shoe(6, rng_factory=factory)
    assert [again.draw() for _ in range(20)] == first


def test_table_passes_the_rng_factory_to_its_shoe() -> None:
    calls = []

    def factory() -> np.random.Generator:
        calls.append(1)
        return np.random.default_rng(99)

    table = BlackjackTable(seats=1, rng_factory=factory)
    assert calls == [1]
    table.act("bet:10")
    assert table.phase in ("playing", "betting")
