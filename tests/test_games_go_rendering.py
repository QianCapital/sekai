"""The Go renderers: frames, themes, the finished board (territory, dead stones), text."""

from __future__ import annotations

import numpy as np
import pytest

import sekai
from sekai.envs.classic_control import CartPoleEnv
from sekai.envs.games import OpponentWrapper, RandomPolicy
from sekai.envs.games.go import GoEnv, GoRenderer, GoTextRenderer

# Black owns the bottom, White the top; black D7 and white C3 are dead.
DEAD_ROWS = [
    ".O.O.OO.O",
    "OOOOOO.OO",
    "O..X.OOOO",
    "OOOOOOOOO",
    "XXXXXXXXX",
    "X..XX.XXX",
    "X.OX.X.XX",
    "XXXXXXX.X",
    ".X.X.XXX.",
]


def finished(passes: int = 2) -> GoEnv:
    env = GoEnv(9)
    env.reset(options={"board": "".join(DEAD_ROWS), "to_play": "black"})
    for _ in range(passes):
        env.step(81)
    return env


def pixel(frame: np.ndarray, renderer: GoRenderer, env: GoEnv, vertex: str) -> np.ndarray:
    r, c = divmod(env.encode_vertex(vertex), env.size)
    x, y = renderer.point(env.size, r, c)
    return frame[int(y), int(x)].astype(int)


@pytest.mark.parametrize("size", [9, 13, 19])
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_frames(size: int, theme: str) -> None:
    env = GoEnv(size)
    result = env.reset(seed=0)
    policy = RandomPolicy(seed=0)
    for _ in range(size * 2):
        result = env.step(policy.act(result.observation, result.info))
    renderer = GoRenderer(theme=theme)
    frame = renderer.render(env)
    width, height = renderer.frame_size()
    assert frame.shape == (height, width, 3) and frame.dtype == np.uint8
    assert width == 480 + 32
    # Deterministic: the same frame again, from this renderer (cached) and a new one.
    np.testing.assert_array_equal(renderer.render(env), frame)
    np.testing.assert_array_equal(GoRenderer(theme=theme).render(env), frame)
    background = frame[2, 2]
    expected = (18, 24, 16) if theme == "dark" else (246, 243, 236)
    assert tuple(background) == expected


def test_themes_share_the_board() -> None:
    env = GoEnv(9)
    env.reset(options={"moves": ["E5", "C3"]})
    dark, light = GoRenderer(theme="dark"), GoRenderer(theme="light")
    a, b = dark.render(env), light.render(env)
    left, top, right, bottom = dark.board_box()
    inner = (slice(top + 40, bottom - 40), slice(left + 40, right - 40))
    assert np.abs(a[inner].astype(int) - b[inner].astype(int)).max() <= 1
    assert not np.array_equal(a[:20], b[:20])


def test_stones_last_move_and_wood() -> None:
    env = GoEnv(9)
    env.reset(options={"moves": ["C3", "E5"]})
    renderer = GoRenderer()
    frame = renderer.render(env)
    black, white = pixel(frame, renderer, env, "C3"), pixel(frame, renderer, env, "E5")
    assert black.sum() < 200 and white.sum() > 600
    unit = renderer.unit(9)

    def ring(frame: np.ndarray, vertex: str) -> int:
        r, c = divmod(env.encode_vertex(vertex), 9)
        x, y = renderer.point(9, r, c)
        return int(frame[int(y), int(x + 0.2 * unit)].astype(int).sum())

    assert ring(frame, "E5") < 200  # a dark ring on the last move, a white stone
    env.step(env.encode_vertex("D4"))
    frame = renderer.render(env)
    assert ring(frame, "E5") > 600  # gone from E5...
    assert ring(frame, "D4") > 600  # ...and light on the black stone just played
    left, top, _, _ = renderer.board_box()
    wood = frame[top + 5, left + int(unit * 3)].astype(int)
    assert 180 < wood[0] < 230 and wood[0] > wood[1] > wood[2]


def test_territory_and_dead_stones_only_at_the_end() -> None:
    renderer = GoRenderer()
    playing = finished(passes=1)
    over = finished(passes=2)
    before, after = renderer.render(playing), renderer.render(over)
    # B3 is Black's territory, A9 White's: squares only once the game is over.
    lines = pixel(before, renderer, playing, "B3") - (74, 56, 36)  # the grid's colour
    assert np.abs(lines).max() <= 2
    assert pixel(after, renderer, over, "B3").sum() < 100  # a dark square
    assert pixel(after, renderer, over, "A9").sum() > 700  # a light square
    # Dead stones fade: black D7 over White's square, white C3 over Black's.
    live = pixel(after, renderer, over, "D5")
    dead = pixel(after, renderer, over, "D7")
    assert dead.sum() > live.sum() + 200
    assert pixel(after, renderer, over, "C3").sum() < pixel(before, renderer, playing, "C3").sum()
    score = renderer.final(over)
    assert score.dead_stones == ["D7", "C3"] and score.result == "B+1.5"
    assert renderer.final(over) is score  # cached per position


def test_status() -> None:
    renderer = GoRenderer(names={"white": "Engine"})
    env = GoEnv(9)
    env.reset()
    status = renderer.status(env)
    assert (status.text, status.sub, status.tone) == ("Black to move", None, "idle")
    env.step(81)
    status = renderer.status(env)
    assert (status.text, status.sub) == ("Engine to move", "Black passed.")
    status = renderer.status(finished())
    assert status.text == "Black won"
    assert status.sub == "Black wins by 1.5 points. Black 45 · White 43.5 (komi 7.5)."
    assert status.tone == "draw"
    env = GoEnv(5, komi=0)
    env.reset()
    env.step(25)
    env.step(25)
    draw = renderer.status(env)
    assert (draw.text, draw.sub) == ("Draw", "A draw. Black 0 · White 0 (komi 0).")


def test_layout_options() -> None:
    env = GoEnv(13)
    env.reset(seed=1)
    small = GoRenderer(320, header=False, strips=False)
    frame = small.render(env)
    assert frame.shape == (small.frame_size()[1], small.frame_size()[0], 3)
    assert small.frame_size()[0] == 320 + 2 * round(small.margin())
    assert GoRenderer(perspective="white").bottom_side(env) == "white"
    assert GoRenderer().bottom_side(env) == "black"
    with pytest.raises(ValueError):
        GoRenderer(perspective="red")


def test_renders_wrapped_envs_and_rejects_others() -> None:
    env = OpponentWrapper(GoEnv(9), RandomPolicy(seed=0), agent_color="white")
    env.reset(seed=0)
    frame = GoRenderer().render(env)
    assert frame.ndim == 3
    made = sekai.make("Go19x19-v0")
    made.reset(seed=0)
    assert GoRenderer().render(made).shape == GoRenderer().render(made.unwrapped).shape
    with pytest.raises(TypeError):
        GoRenderer().render(CartPoleEnv())
    with pytest.raises(sekai.ResetRequired):
        GoRenderer().render(GoEnv(9))
    with pytest.raises(TypeError):
        GoTextRenderer().render(CartPoleEnv())


def test_text_renderer() -> None:
    env = GoEnv(9)
    env.reset(options={"moves": ["E5"]})
    text = GoTextRenderer().render(env)
    assert isinstance(text, str)
    lines = text.splitlines()
    assert lines[:11] == env.board.pretty().splitlines()
    assert lines[0] == "   A B C D E F G H J"
    assert lines[5] == " 5 . . . . X . . . .  5"
    assert lines[-1] == "White to move. Last move: Black E5. Captured: Black 0, White 0. Komi 7.5."
    over = finished()
    assert GoTextRenderer().render(over).splitlines()[-1] == (
        "Game over (two passes): B+1.5. Captured: Black 0, White 0. Komi 7.5."
    )
    with pytest.raises(sekai.ResetRequired):
        GoTextRenderer().render(GoEnv(9))
