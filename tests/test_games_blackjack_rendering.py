"""Blackjack: the table renderer and the text renderer."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

import sekai
from sekai.envs.games.blackjack import (
    BlackjackEnv,
    BlackjackRenderer,
    BlackjackTable,
    BlackjackTextRenderer,
    Rules,
    Shoe,
)
from sekai.envs.games.blackjack.rendering import (
    dealer_label,
    format_chips,
    format_signed,
    outcome_label,
    print_lines,
    total_label,
)

MINUS = "\u2212"


def stacked_env(cards: Sequence[str], rules: Rules | None = None, seed: int = 0) -> BlackjackEnv:
    """An env whose next round deals `cards` first (seat, dealer up, seat, hole, ...)."""
    env = BlackjackEnv(rules=rules or Rules())
    env.reset(seed=seed)
    assert env.table is not None
    env.table._shoe.stack_top(cards)
    return env


def test_frame_shape_and_themes() -> None:
    env = stacked_env(["8h", "6c", "8d", "Ts", "8s", "3c"])
    renderer = BlackjackRenderer()
    with pytest.raises(sekai.ResetRequired):
        renderer.render(BlackjackEnv())
    frames = [renderer.render(env)]
    env.step(6)
    env.step(3)  # split: two hands, the first active
    for theme in ("dark", "light"):
        frame = BlackjackRenderer(theme=theme).render(env)
        width, height = BlackjackRenderer(theme=theme).frame_size()
        assert frame.shape == (height, width, 3) == (436, 672, 3) and frame.dtype == np.uint8
        frames.append(frame)
    assert not np.array_equal(frames[0], frames[1])  # the deal is drawn
    assert not np.array_equal(frames[1], frames[2])  # the themes differ
    assert np.array_equal(renderer.render(env), frames[1])  # same state, same frame
    small = BlackjackRenderer(320, header=False).render(env)
    assert small.shape == (195, 336, 3)  # the table 320 x 179.2, margins 8
    with pytest.raises(ValueError):
        BlackjackRenderer(100)


def test_renders_a_wrapped_env_and_rejects_others() -> None:
    env = sekai.TimeLimit(sekai.make("Blackjack-v0"), 10)
    env.reset(seed=1)
    frame = BlackjackRenderer().render(env)
    assert frame.shape == (436, 672, 3)
    assert "Place your bet" in BlackjackRenderer().status(env.unwrapped).text
    assert BlackjackTextRenderer().render(env).startswith("Blackjack (6D S17 DAS LS 3:2)")
    other = sekai.make("CartPole-v1")
    other.reset(seed=0)
    with pytest.raises(TypeError):
        BlackjackRenderer().render(other)
    with pytest.raises(TypeError):
        BlackjackTextRenderer().render(other)


def test_hole_card_is_never_drawn_before_it_is_revealed() -> None:
    # Identical rounds except for the hole card (7 or 8 under a 9: neither a blackjack).
    a = stacked_env(["Th", "9c", "6d", "7s", "2c"], seed=9)
    b = stacked_env(["Th", "9c", "6d", "8s", "2c"], seed=9)
    renderer, text = BlackjackRenderer(), BlackjackTextRenderer()
    for env in (a, b):
        env.step(5)
    assert np.array_equal(renderer.render(a), renderer.render(b))
    assert text.render(a) == text.render(b) and "9c ??" in text.render(a)
    for env in (a, b):
        env.step(1)  # hit: 18, still playing
    assert not a.done and np.array_equal(renderer.render(a), renderer.render(b))
    for env in (a, b):
        env.step(0)  # stand: the dealer reveals
    assert not np.array_equal(renderer.render(a), renderer.render(b))
    assert "9c 7s" in text.render(a) and "9c 8s" in text.render(b)


def test_status() -> None:
    renderer = BlackjackRenderer()
    env = stacked_env(["Ah", "6c", "6d", "Ts", "Kc"])
    status = renderer.status(env)
    assert (status.text, status.sub, status.tone) == ("Place your bet", "Round 1", "yours")
    env.step(6)
    status = renderer.status(env)
    assert status.text == "Soft 17 against the dealer's 6" and status.tone == "yours"
    env.step(0)  # stand on soft 17; the dealer has 16 and draws the king
    status = renderer.status(env)
    assert status.text == "Round over: +20" and status.tone == "win"
    assert status.sub == "Dealer busts with 26 \u00b7 bankroll 1,020"
    env.reset()
    assert renderer.status(env).sub == "Last round +20"


def test_split_hands_double_and_outcomes() -> None:
    env = stacked_env(["8h", "6c", "8d", "Ts", "3h", "Tc", "Kd", "5s"])
    renderer, text = BlackjackRenderer(), BlackjackTextRenderer()
    env.step(6)
    env.step(3)  # split the eights
    assert "Hand 1 of 2" in (renderer.status(env).sub or "")
    assert "hand 1*: 8h 3h (11)" in text.render(env)
    env.step(2)  # double: 21
    env.step(0)  # stand on 18; the dealer makes 21
    frame = renderer.render(env)
    assert frame.shape == (436, 672, 3)
    summary = text.render(env)
    assert "round 1 over, net -20" in summary
    assert "hand 1: 8h 3h Tc (21), bet 40 doubled, push" in summary
    assert "hand 2: 8d Kd (18), bet 20, lose -20" in summary


def test_table_with_several_seats() -> None:
    rules = Rules(decks=1, dealer_hits_soft_17=True, blackjack_payout=1.2)
    table = BlackjackTable(rules, seats=3, bankroll=[1000, 5, 300], seed=3)
    names = ["Ann", "Bo", "Cy"]
    renderer = BlackjackRenderer(names=names)
    assert renderer.status(table).text == "Ann to bet"
    table.act("bet:25")
    frame = renderer.render(table)
    assert frame.shape == (436, 672, 3)
    lines = BlackjackTextRenderer(names=names).render(table).splitlines()
    assert lines[1:4] == [
        "Ann: bankroll 975, bet 25",
        "Bo: bankroll 5, out",
        "Cy: bankroll 300, to bet",
    ]
    assert BlackjackRenderer(names={0: "Ann"}).seat_name(2, 3) == "Player 3"
    assert BlackjackRenderer().seat_name(0, 1) == "Player"
    table.act("bet:10")
    while table.phase == "playing":
        table.act("stand")
    assert renderer.render(table).shape == (436, 672, 3)
    crowded = BlackjackTable(rules, seats=7, seed=0)
    assert BlackjackRenderer().render(crowded).shape == (436, 672, 3)


def test_printed_rules() -> None:
    assert print_lines(Rules()) == (
        "BLACKJACK PAYS 3 TO 2",
        "DEALER STANDS ON ALL 17s \u00b7 NO INSURANCE",
    )
    h17 = Rules(dealer_hits_soft_17=True, blackjack_payout=1.2)
    assert print_lines(h17) == ("BLACKJACK PAYS 6 TO 5", "DEALER HITS SOFT 17 \u00b7 NO INSURANCE")
    a, b = BlackjackEnv(), BlackjackEnv(rules=h17)
    a.reset(seed=0)
    b.reset(seed=0)
    assert not np.array_equal(BlackjackRenderer().render(a), BlackjackRenderer().render(b))


def test_labels() -> None:
    assert format_signed(10) == "+10" and format_signed(-2.5) == f"{MINUS}3"
    assert format_signed(0) == "0" and format_signed(-0.04, 1) == "0.0"
    assert format_signed(1.25, 1) == "+1.3" and format_signed(1.005, 2) == "+1.00"
    assert format_chips(1022.5) == "1,023" and format_chips(999.49) == "999"
    hand = {"cards": ["Ah", "6d"], "total": 17, "soft": True, "blackjack": False, "bust": False}
    assert total_label(hand) == "Soft 17"
    assert total_label({**hand, "cards": ["Ah", "Td"], "total": 21}) == "21"
    assert total_label({**hand, "blackjack": True}) == "Blackjack"
    assert total_label({**hand, "total": 24, "bust": True}) == "Bust 24"
    dealer = {"cards": ["Kc", None], "total": 10, "blackjack": False, "bust": False}
    assert dealer_label({**dealer, "revealed": False}) == "Shows 10"
    assert dealer_label({**dealer, "revealed": True, "total": 17}) == "17"
    assert outcome_label({"outcome": "lose", "payout": -20}) == f"Lose {MINUS}20"
    assert outcome_label({"outcome": "push", "payout": 0}) == "Push"
    assert outcome_label({"outcome": "blackjack", "payout": 22.5}) == "Blackjack +23"


def test_out_of_chips_and_game_over() -> None:
    table = BlackjackTable(seats=1, bankroll=10, seed=1)
    while not table.game_over:
        legal = table.legal_actions()
        table.act("bet:10" if isinstance(legal, dict) else "stand")
    renderer = BlackjackRenderer()
    assert renderer.status(table).text == "Every seat is out of chips"
    assert renderer.render(table).shape == (436, 672, 3)
    assert "every seat is out of chips" in BlackjackTextRenderer().render(table)


def test_rendering_reads_the_public_view_only() -> None:
    env = stacked_env(["Th", "9c", "6d", "7s"])
    env.step(5)
    assert env.table is not None
    order = list(env.table._shoe._cards)
    view = env.view()
    BlackjackRenderer().render(env)
    BlackjackTextRenderer().render(env)
    assert env.view() == view and env.table._shoe._cards == order
    assert "7s" not in BlackjackTextRenderer().render(env)
    assert str(Shoe(1, seed=0)) == "Shoe(decks=1, remaining=52/52, running_count=0)"
