"""Heads-up hold'em: PokerStars-style hand histories."""

from __future__ import annotations

import datetime as dt

from sekai.envs.games.poker import HoldemHand, format_hand_history

TS = dt.datetime(2026, 10, 18, 12, 0, 0, tzinfo=dt.timezone.utc)


def _fold_hand() -> HoldemHand:
    h = HoldemHand(deck="AsKdQhJcAh7d2cJs3s", button=0)
    h.act("raise:300")
    h.act("call")
    h.act("check")
    h.act("bet:400")
    h.act("fold")
    return h


def test_fold_hand_history_exact() -> None:
    h = _fold_hand()
    text = format_hand_history(h, hand_no=7, names=("You", "Agent"), hero=0, timestamp=TS)
    assert (
        text
        == """\
sekai Hand #7: Hold'em No Limit (50/100) - 2026/10/18 12:00:00 UTC
Table 'sekai' 2-max Seat #1 is the button
Seat 1: You (10000 in chips)
Seat 2: Agent (10000 in chips)
You: posts small blind 50
Agent: posts big blind 100
*** HOLE CARDS ***
Dealt to You [As Kd]
You: raises 200 to 300
Agent: calls 200
*** FLOP *** [Ah 7d 2c]
Agent: checks
You: bets 400
Agent: folds
Uncalled bet (400) returned to You
You collected 600 from pot
You: doesn't show hand
*** SUMMARY ***
Total pot 600 | Rake 0
Board [Ah 7d 2c]
Seat 1: You (button) (small blind) collected (600)
Seat 2: Agent (big blind) folded on the Flop
"""
    )
    assert "Qh" not in text and "Jc" not in text  # the opponent mucked
    revealed = format_hand_history(h, hand_no=7, names=("You", "Agent"), reveal=True, timestamp=TS)
    assert "Dealt to Agent [Qh Jc]" in revealed
    assert "folded on the Flop [Qh Jc]" in revealed


def test_site_and_table_names() -> None:
    h = _fold_hand()
    text = format_hand_history(h, hand_no=3, site="Home Game", table="Main", timestamp=TS)
    first, second = text.splitlines()[:2]
    assert first == "Home Game Hand #3: Hold'em No Limit (50/100) - 2026/10/18 12:00:00 UTC"
    assert second == "Table 'Main' 2-max Seat #1 is the button"


def test_timestamps_are_shown_in_utc() -> None:
    h = _fold_hand()
    tokyo = dt.datetime(2026, 10, 18, 21, 0, 0, tzinfo=dt.timezone(dt.timedelta(hours=9)))
    assert format_hand_history(h, hand_no=1, timestamp=tokyo) == format_hand_history(
        h, hand_no=1, timestamp=TS
    )
    naive = dt.datetime(2026, 10, 18, 12, 0, 0)  # taken to be UTC already
    assert format_hand_history(h, hand_no=1, timestamp=naive) == format_hand_history(
        h, hand_no=1, timestamp=TS
    )
    assert format_hand_history(h, hand_no=1).splitlines()[0].endswith(" UTC")


def test_showdown_history() -> None:
    h = HoldemHand(deck="AsKdQhQc2c7d9hJs3s", button=1)
    h.act("allin")
    h.act("call")
    text = format_hand_history(h, hand_no=1, hero=0, timestamp=TS)
    assert "Seat 2: raises 9900 to 10000 and is all-in" in text
    assert "Seat 1: calls 9900 and is all-in" in text
    assert "*** TURN *** [2c 7d 9h] [Js]" in text
    assert "*** RIVER *** [2c 7d 9h Js] [3s]" in text
    assert "*** SHOW DOWN ***" in text
    assert "Seat 1: shows [As Kd] (High Card, Ace high)" in text
    assert "Seat 2: shows [Qh Qc] (One Pair, Queens)" in text
    assert "Seat 2 collected 20000 from pot" in text
    assert "Seat 1: Seat 1 (big blind) showed [As Kd] and lost with High Card, Ace high" in text
    assert (
        "Seat 2: Seat 2 (button) (small blind) showed [Qh Qc] and won (20000) with One Pair, Queens"
    ) in text


def test_in_progress_history_does_not_leak() -> None:
    h = HoldemHand(deck="AsKdQhQc2c7d9hJs3s", button=0)
    h.act("call")
    h.act("check")
    for hero in (0, 1, None):
        text = format_hand_history(h, hand_no=2, hero=hero, reveal=True, timestamp=TS)
        assert "*** SUMMARY ***" not in text
        assert "Js" not in text and "3s" not in text  # undealt board cards
        if hero != 0:
            assert "As Kd" not in text
        if hero != 1:
            assert "Qh Qc" not in text
    view_text = format_hand_history(h.view(1), hand_no=2, timestamp=TS)
    assert "Dealt to Seat 2 [Qh Qc]" in view_text and "As Kd" not in view_text


def test_split_pot_history() -> None:
    h = HoldemHand(deck="2c3d2h3sAsKsQsJsTs", button=0)
    h.act("call")
    for _ in range(7):
        h.act("check")
    text = format_hand_history(h, hand_no=3, timestamp=TS)
    assert text.count("collected 100 from pot") == 2
    assert text.count("and won (100) with Royal Flush") == 2
