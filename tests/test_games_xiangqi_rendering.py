"""XiangqiRenderer and XiangqiTextRenderer: frames, themes, orientation, status, strips."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import sekai
from sekai.envs.games import OpponentWrapper, RandomPolicy
from sekai.envs.games.xiangqi import XiangqiEnv, XiangqiRenderer, XiangqiTextRenderer
from sekai.envs.games.xiangqi.engine import (
    ADVISOR,
    CANNON,
    CHARIOT,
    ELEPHANT,
    HORSE,
    SOLDIER,
    Position,
)
from sekai.envs.games.xiangqi.rendering import _material

MATE_IN_ONE = "4k4/R8/9/9/9/9/9/9/9/1R1K5 w - - 0 1"
CHECK = "3k5/9/9/9/9/9/9/9/4p4/4K4 w - - 0 1"  # Red is in check


def midgame(plies: int = 30, seed: int = 4) -> XiangqiEnv:
    env = XiangqiEnv(history_length=1)
    result = env.reset()
    rng = np.random.default_rng(seed)
    for _ in range(plies):
        legal = result.info["legal_actions"]
        result = env.step(int(legal[rng.integers(len(legal))]))
    assert not env.done
    return env


def board(renderer: XiangqiRenderer, frame: np.ndarray) -> np.ndarray:
    left, top, right, bottom = (int(v) for v in renderer.board_box())
    return frame[top:bottom, left:right]


def test_frame_shape_and_dtype() -> None:
    env = midgame()
    for size in (480, 320):
        renderer = XiangqiRenderer(size)
        frame = renderer.render(env)
        assert frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 3
        width, height = renderer.frame_size()
        assert frame.shape[:2] == (height, width)
        assert renderer.board_height() == round(size * (9 + 2 * 1.18) / (8 + 2 * 1.18))
    assert XiangqiRenderer().frame_size()[0] == 480 + 2 * 16
    no_frills = XiangqiRenderer(header=False, strips=False).render(env)
    assert no_frills.shape[0] < XiangqiRenderer().render(env).shape[0]


def test_themes_share_the_board() -> None:
    env = midgame()
    dark, light = XiangqiRenderer(theme="dark"), XiangqiRenderer(theme="light")
    dark_frame, light_frame = dark.render(env), light.render(env)
    assert not np.array_equal(dark_frame, light_frame)  # the page around the board differs
    inset = 16  # inside the board's rounded corners the boards are the same
    a, b = board(dark, dark_frame), board(light, light_frame)
    np.testing.assert_array_equal(a[inset:-inset, inset:-inset], b[inset:-inset, inset:-inset])
    with pytest.raises(ValueError):
        XiangqiRenderer(theme="sepia")


def test_orientation() -> None:
    env = midgame()
    red, black = XiangqiRenderer(orientation="red"), XiangqiRenderer(orientation="black")
    mover = XiangqiRenderer(orientation="mover")
    red_board, black_board = board(red, red.render(env)), board(black, black.render(env))
    assert not np.array_equal(red_board, black_board)
    # The board flips: Red's back rank is at the bottom of one and at the top of the other.
    assert red.bottom_side(env) == "red" and black.bottom_side(env) == "black"
    expected = red if env.turn == "red" else black
    np.testing.assert_array_equal(mover.render(env), expected.render(env))
    env.step(int(env.legal_actions()[0]))
    expected = red if env.turn == "red" else black
    np.testing.assert_array_equal(mover.render(env), expected.render(env))
    with pytest.raises(ValueError, match="orientation"):
        XiangqiRenderer(orientation="white")  # type: ignore[arg-type]


def test_deterministic() -> None:
    env = midgame()
    renderer = XiangqiRenderer()
    first = renderer.render(env)
    np.testing.assert_array_equal(first, renderer.render(env))  # cached layers
    np.testing.assert_array_equal(first, XiangqiRenderer().render(env))  # a fresh renderer


def test_renders_wrapped_envs_and_rejects_others() -> None:
    made = sekai.make("Xiangqi-v0")
    made.reset(seed=0)
    renderer = XiangqiRenderer()
    np.testing.assert_array_equal(renderer.render(made), renderer.render(made.unwrapped))
    wrapped = OpponentWrapper(XiangqiEnv(), RandomPolicy(seed=1), agent_color="black")
    wrapped.reset(seed=1)
    assert renderer.render(wrapped).shape == renderer.render(made).shape
    assert XiangqiTextRenderer().render(wrapped) == wrapped.unwrapped.position.ascii()
    with pytest.raises(TypeError):
        renderer.render(sekai.make("CartPole-v1"))
    with pytest.raises(TypeError):
        XiangqiTextRenderer().render(sekai.make("CartPole-v1"))


def test_text_renderer() -> None:
    env = XiangqiEnv()
    env.reset()
    text = XiangqiTextRenderer().render(env)
    assert isinstance(text, str) and text == Position().ascii()
    assert text.splitlines()[0] == "9 r n b a k a b n r"


def test_status() -> None:
    renderer = XiangqiRenderer(names={"black": "Random"})
    env = XiangqiEnv()
    env.reset()
    status = renderer.status(env)
    assert (status.text, status.sub) == ("Red to move", None)
    env.step(env.encode_move("h2e2"))
    assert renderer.status(env).text == "Random to move"
    env.reset(options={"fen": CHECK})
    assert renderer.status(env).sub == "In check."
    env.reset(options={"fen": MATE_IN_ONE})
    env.step(env.encode_move("b0b9"))
    status = renderer.status(env)
    assert (status.text, status.sub) == ("Red won", "Red wins by checkmate.")
    env.reset(options={"fen": "3k1a3/4a4/9/9/9/9/9/4B4/4A4/4KA3 w - - 0 1"})
    status = renderer.status(env)
    assert (status.text, status.sub) == ("Draw", "Drawn by insufficient material.")
    forfeit = XiangqiEnv(illegal_action="lose")
    forfeit.reset()
    forfeit.step(0)  # a0a0: Red forfeits
    status = renderer.status(forfeit)
    assert (status.text, status.sub) == ("Random won", "Black wins by an illegal move.")


def test_header_shows_the_move_limit() -> None:
    env = sekai.make("Xiangqi-v0", max_episode_steps=4)
    result = env.reset(seed=0)
    while not result.done:
        result = env.step(int(result.info["legal_actions"][0]))
    assert result.truncated and not env.unwrapped.done
    renderer = XiangqiRenderer()
    wrapped, bare = renderer.render(env), renderer.render(env.unwrapped)
    header = int(renderer.header_height())
    assert not np.array_equal(wrapped[:header], bare[:header])  # "Draw", not "Red to move"
    np.testing.assert_array_equal(board(renderer, wrapped), board(renderer, bare))


def test_highlights_change_the_frame() -> None:
    renderer = XiangqiRenderer(header=False, strips=False)
    env = XiangqiEnv()
    env.reset(options={"fen": CHECK})  # the red general on e0 glows
    glowing = renderer.render(env)
    env.reset(options={"fen": "3k5/9/9/9/9/9/9/9/3p5/4K4 w - - 0 1"})  # not in check
    plain = renderer.render(env)
    x, y = renderer._point(4, 9 + 0.51)  # just below the general: glow, not piece
    redness = [int(f[int(y), int(x), 0]) - int(f[int(y), int(x), 1]) for f in (glowing, plain)]
    assert redness[0] > redness[1] + 20
    env.reset()
    before = renderer.render(env)
    play = ["b0c2", "b9c7", "c2b0", "c7b9"]
    for uci in play:
        env.step(env.encode_move(uci))
    # The start position again, now with the last move ringed.
    assert env.fen.split()[0] == Position().fen().split()[0]
    assert not np.array_equal(before, renderer.render(env))


def test_captured_pieces_and_lead() -> None:
    board = Position("1C1k2b2/7n1/4na3/9/p1N5p/P1P1p4/5p3/B2AKC1rr/R8/5A3 w - - 12 76").board
    captured, score = _material(board)
    assert captured["black"] == [CHARIOT, HORSE, ELEPHANT, SOLDIER, SOLDIER, SOLDIER]
    assert captured["red"] == [CANNON, CANNON, ELEPHANT, ADVISOR, SOLDIER]
    assert score == -4.0
    assert _material(Position().board) == ({"red": [], "black": []}, 0.0)


def test_strips_show_captures() -> None:
    env = XiangqiEnv()
    env.reset()
    renderer = XiangqiRenderer()
    _, _, right, bottom = (int(v) for v in renderer.board_box())
    height = int(renderer.strip_height())

    def red_captures(frame: np.ndarray) -> np.ndarray:
        return frame[bottom : bottom + height, right - 160 : right]

    before = red_captures(renderer.render(env))
    env.step(env.encode_move("b2b9"))  # Red's cannon takes a horse
    after = red_captures(renderer.render(env))
    assert not np.array_equal(before, after)


def test_cached_layers_are_bounded() -> None:
    env = midgame(plies=10)
    renderer = XiangqiRenderer(orientation="mover")
    for _ in range(12):
        renderer.render(env)
        if env.done:
            break
        env.step(int(env.legal_actions()[0]))
    assert len(renderer._static) <= 9 and len(renderer._sprites) < 4096


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_renders_every_kind_of_position(theme: str) -> None:
    renderer = XiangqiRenderer(240, theme=theme)
    policy = RandomPolicy(seed=3)
    env = XiangqiEnv(history_length=1, illegal_action="lose")
    result: Any = env.reset(seed=3)
    for _ in range(60):
        frame = renderer.render(env)
        assert frame.shape[:2] == renderer.frame_size()[::-1]
        if result.done:
            result = env.reset()
        result = env.step(policy.act(result.observation, result.info))
