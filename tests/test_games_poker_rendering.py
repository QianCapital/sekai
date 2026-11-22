"""Heads-up hold'em: the table renderer and the text renderer."""

from __future__ import annotations

import numpy as np
import pytest

import sekai
from sekai.envs.games import OpponentWrapper, RandomPolicy
from sekai.envs.games.poker import HoldemEnv, HoldemRenderer, HoldemTextRenderer

DECK = "AsKdQhQc2c7d9hJs3s"


def play(actions: list[int], deck: str = DECK, button: int = 0) -> HoldemEnv:
    env = HoldemEnv()
    env.reset(options={"button": button, "deck": deck})
    for action in actions:
        env.step(action)
    return env


def test_frame_shape_and_type() -> None:
    renderer = HoldemRenderer()
    frame = renderer.render(play([]))
    width, height = renderer.frame_size()
    assert frame.shape == (height, width, 3) and frame.dtype == np.uint8
    assert width == 640 + 2 * 16
    wide = HoldemRenderer(width=800)
    assert wide.render(play([])).shape[1] == 800 + 2 * 20
    bare = HoldemRenderer(header=False)
    assert bare.render(play([])).shape[0] < height


def test_rendering_is_deterministic_and_themed() -> None:
    env = play([4, 1, 3])
    dark, light = HoldemRenderer(), HoldemRenderer(theme="light")
    assert np.array_equal(dark.render(env), dark.render(env))
    assert np.array_equal(dark.render(env), HoldemRenderer().render(env))
    assert not np.array_equal(dark.render(env), light.render(env))
    # The page differs between the themes, the felt does not.
    d, li = dark.render(env), light.render(env)
    assert not np.array_equal(d[2, 2], li[2, 2])
    cx, cy = d.shape[1] // 2 - 200, d.shape[0] // 2
    assert np.array_equal(d[cy, cx], li[cy, cx])


def test_every_state_renders() -> None:
    renderers = [
        HoldemRenderer(),
        HoldemRenderer(
            width=400, perspective=1, reveal="perspective", names={"p1": "A very long name"}
        ),
    ]
    env = HoldemEnv()
    rng = np.random.default_rng(0)
    for seed in range(5):
        r = env.reset(seed=seed, options={"stacks": (int(rng.integers(150, 3000)), 10000)})
        while True:
            for renderer in renderers:
                frame = renderer.render(env)
                assert frame.shape[::-1][1:] == renderer.frame_size()
            if r.done:
                break
            r = env.step(int(env.action_space.sample(mask=r.info["legal_action_mask"])))


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_perspectives(theme: str) -> None:
    env = play([4, 1])
    bottom0 = HoldemRenderer(theme=theme, perspective=0).render(env)
    bottom1 = HoldemRenderer(theme=theme, perspective="p2").render(env)
    assert not np.array_equal(bottom0, bottom1)
    assert np.array_equal(bottom1, HoldemRenderer(theme=theme, perspective=1).render(env))


def test_perspective_reveal_hides_the_other_seat_until_the_end() -> None:
    # Two deals that differ only in the top seat's (seat 1's) hole cards.
    other = "AsKd8h8c2c7d9hJs3s"
    for actions in ([], [1], [1, 1], [4, 1, 1, 1], [4, 1, 1, 1, 1, 1]):
        a = HoldemRenderer(reveal="perspective").render(play(actions))
        b = HoldemRenderer(reveal="perspective").render(play(actions, deck=other))
        assert np.array_equal(a, b), actions
        shown_a = HoldemRenderer().render(play(actions))
        shown_b = HoldemRenderer().render(play(actions, deck=other))
        assert not np.array_equal(shown_a, shown_b)  # "all" shows both hands
    # Once the hand is over both hands are shown.
    end = [4, 1, 1, 1, 1, 1, 1, 1]
    a = HoldemRenderer(reveal="perspective").render(play(end))
    b = HoldemRenderer(reveal="perspective").render(play(end, deck=other))
    assert not np.array_equal(a, b)


def test_undealt_cards_are_never_drawn() -> None:
    # Two deals that differ only in the turn and river.
    other = "AsKdQhQc2c7d9h4h8d"
    for actions in ([], [4, 1], [4, 1, 1]):
        a = HoldemRenderer().render(play(actions))
        b = HoldemRenderer().render(play(actions, deck=other))
        assert np.array_equal(a, b), actions
    # The turn, once dealt, is drawn.
    a = HoldemRenderer().render(play([4, 1, 1, 1]))
    b = HoldemRenderer().render(play([4, 1, 1, 1], deck=other))
    assert not np.array_equal(a, b)


def test_folded_and_showdown_hands_are_marked() -> None:
    renderer = HoldemRenderer(header=False)
    fold = renderer.render(play([4, 1, 3, 0]))
    assert fold.shape == renderer.render(play([4, 1, 3])).shape
    status = HoldemRenderer().status(play([4, 1, 3, 0]))
    assert status.text == "Seat 2 wins 600" and status.sub == "Seat 1 folded"
    status = HoldemRenderer(names={"p2": "Agent"}).status(play([6, 1]))
    assert status.text == "Agent wins 20,000" and status.sub == "Showdown · One Pair, Queens"
    split = play([1, 1, 1, 1, 1, 1, 1, 1], deck="2c3d2h3sAsKsQsJsTs")
    status = HoldemRenderer().status(split)
    assert status.text == "Split pot" and status.sub == "Royal Flush"


def test_status_while_playing() -> None:
    renderer = HoldemRenderer(names={"p1": "Agent"})
    env = play([])
    status = renderer.status(env)
    assert status.text == "Agent to act" and status.sub == "Preflop · Hand 1"
    renderer.render(env)
    env.reset(seed=0)  # a new hand: the renderer numbers it
    sub = renderer.status(env).sub
    assert sub is not None and sub.endswith(" · Hand 2")
    mine = HoldemRenderer(reveal="perspective", perspective=0).status(play([]))
    assert mine.tone == "yours"
    theirs = HoldemRenderer(reveal="perspective", perspective=1).status(play([]))
    assert theirs.tone == "idle"


def test_renders_wrapped_envs_and_rejects_others() -> None:
    env = OpponentWrapper(HoldemEnv(), RandomPolicy(seed=0), agent_color="p2")
    env.reset(seed=2)
    renderer = HoldemRenderer(perspective="p2", reveal="perspective")
    assert renderer.render(env).dtype == np.uint8
    assert "Seat 2" in HoldemTextRenderer().render(env)
    with pytest.raises(TypeError):
        renderer.render(sekai.make("CartPole-v1"))
    with pytest.raises(TypeError):
        HoldemTextRenderer().render(sekai.make("CartPole-v1"))
    with pytest.raises(sekai.ResetRequired):
        renderer.render(HoldemEnv())
    with pytest.raises(sekai.ResetRequired):
        HoldemTextRenderer().render(HoldemEnv())


def test_invalid_options() -> None:
    with pytest.raises(ValueError):
        HoldemRenderer(width=100)
    with pytest.raises(ValueError):
        HoldemRenderer(reveal="none")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        HoldemRenderer(perspective=2)
    with pytest.raises(ValueError):
        HoldemTextRenderer(perspective="p3")
    with pytest.raises(ValueError):
        HoldemRenderer(theme="sepia")


def test_text_renderer() -> None:
    env = play([4, 1, 3])
    text = HoldemTextRenderer(names={"p2": "Agent"}).render(env)
    lines = text.splitlines()
    assert lines[0] == "Hold'em 50/100 · Flop"
    assert lines[1] == "Board  2c 7d 9h -- --    Pot 600"
    assert lines[2].startswith("  Agent ") and "[Qh Qc]" in lines[2] and "Bet 300" in lines[2]
    assert lines[3].startswith("D Seat 1") and "[As Kd]" in lines[3]
    assert lines[3].endswith("<- to act")
    hidden = HoldemTextRenderer(reveal="perspective").render(env)
    assert "Qh" not in hidden and "[?? ??]" in hidden and "[As Kd]" in hidden
    flipped = HoldemTextRenderer(reveal="perspective", perspective=1).render(env)
    assert "As" not in flipped.split("Board")[1] and "[Qh Qc]" in flipped
    assert "Js" not in text and "3s" not in text  # undealt
    end = HoldemTextRenderer(reveal="perspective").render(play([4, 1, 3, 0]))
    assert end.splitlines()[-1] == "Seat 2 wins 600 (Seat 1 folded)" and "[Qh Qc]" in end
