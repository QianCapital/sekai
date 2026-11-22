"""What the strategy games share: drawing, cards, artwork, the frame, policies and wrappers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import sekai
from sekai.envs import games
from sekai.envs.games import (
    DARK,
    LIGHT,
    GameError,
    GameOver,
    IllegalMove,
    OpponentWrapper,
    Policy,
    RandomPolicy,
)
from sekai.envs.games._assets import chess_piece, glyph, has_glyph
from sekai.envs.games._cards import draw_card, draw_chip, draw_pill, draw_slot, draw_suit
from sekai.envs.games._draw import (
    Painter,
    fractal_noise,
    gaussian_blur,
    rotate_mask,
    rounded_box_coverage,
)
from sekai.envs.games.rendering import game_theme, time_limit_reached
from sekai.rendering import Canvas


def painter(width: int = 120, height: int = 80) -> Painter:
    return Painter(Canvas(width, height, (0, 0, 0), antialias=True))


def test_import_is_lazy_and_exports_resolve() -> None:
    assert games.ChessEnv is sekai.envs.ChessEnv
    with pytest.raises(AttributeError):
        _ = games.NoSuchGame
    with pytest.raises(AttributeError):
        _ = sekai.envs.NoSuchGame
    assert issubclass(IllegalMove, sekai.InvalidAction) and issubclass(GameOver, GameError)


def test_painter_uses_screen_coordinates() -> None:
    p = painter()
    p.rect(0, 0, 10, 10, (255, 0, 0))  # top-left corner
    pixels = p.canvas.to_array()
    assert (pixels[0, 0] == [255, 0, 0]).all() and (pixels[-1, 0] == 0).all()
    p.circle(60, 70, 5, "#00ff00")
    assert p.canvas.to_array()[70, 60, 1] == 255


def test_gradients_rings_and_shadows() -> None:
    p = painter()
    p.linear_rect(0, 0, 120, 80, [(0, "#000000"), (1, "#ffffff")], radius=10)
    row = p.canvas.to_array()[40]
    assert row[5, 0] < row[60, 0] < row[115, 0]  # dark to light along the diagonal
    p.radial_disc(60, 40, 20, [(0, "#ffffff"), (1, "#000000")], focus=(0.5, 0.5), spread=0.5)
    pixels = p.canvas.to_array()
    assert pixels[40, 60, 0] > 240 and pixels[40, 41, 0] < 40
    q = painter()
    q.ring(60, 40, 20, 4, "#ffffff")
    ring = q.canvas.to_array()[:, :, 0]
    assert ring[40, 80] > 200 and ring[40, 60] == 0  # the outline, not the inside
    q.shadow(40, 20, 80, 60, blur=10, color="#ffffff", alpha=1.0)
    assert q.canvas.to_array()[40, 60, 0] > 200 and q.canvas.to_array()[2, 2, 0] < 10


def test_rounded_box_coverage_per_corner() -> None:
    xs = np.arange(40, dtype=np.float64)[None, :] + 0.5
    ys = np.arange(40, dtype=np.float64)[:, None] + 0.5
    full = rounded_box_coverage(xs, ys, 0, 0, 40, 40, 0.0)
    assert full.sum() == pytest.approx(1600)
    one = rounded_box_coverage(xs, ys, 0, 0, 40, 40, [(20, 20), (0, 0), (0, 0), (0, 0)])
    assert one[0, 0] == 0 and one[0, 39] == 1 and one[39, 0] == 1
    assert 1600 - one.sum() == pytest.approx(400 - np.pi * 100, rel=0.05)
    with pytest.raises(ValueError):
        rounded_box_coverage(xs, ys, 0, 0, 40, 40, [(1, 1), (1, 1)])  # type: ignore[list-item]


def test_noise_blur_and_rotation() -> None:
    noise = fractal_noise(64, 32, (0.05, 0.2))
    assert noise.shape == (32, 64) and noise.min() >= 0 and noise.max() <= 1
    assert fractal_noise(64, 32, (0.05, 0.2)) is noise  # cached, read-only
    with pytest.raises(ValueError):
        noise[0, 0] = 1.0
    spot = np.zeros((21, 21))
    spot[10, 10] = 1.0
    blurred = gaussian_blur(spot, 2.0)
    assert blurred.sum() == pytest.approx(1.0, abs=1e-6) and blurred[10, 10] < 0.1
    mask = np.zeros((4, 10), dtype=np.float32)
    mask[1:3, :] = 1.0
    turned, left, _ = rotate_mask(mask, np.pi / 2, -5.0, -2.0)
    assert turned.shape[0] > turned.shape[1]  # a horizontal bar becomes a vertical one
    assert left < 0 < left + turned.shape[1]


def test_text_on_a_curve() -> None:
    p = painter(300, 120)
    points = np.array([[20 + 260 * t, 60 + 40 * t * (1 - t)] for t in np.linspace(0, 1, 40)])
    p.text_on_curve(points, "CURVED TEXT", "#ffffff", size=16)
    lit = np.argwhere(p.canvas.to_array()[:, :, 0] > 128)
    assert lit.size and 100 < lit[:, 1].mean() < 200  # centred along the curve


def test_cards_chips_and_pills() -> None:
    p = painter(320, 160)
    draw_card(p, 10, 10, 60, "As")
    draw_card(p, 80, 10, 60, "Td", highlight=True)
    draw_card(p, 150, 10, 60, None)
    draw_card(p, 220, 10, 60, "7c", dim=True, corners_only=True)
    draw_slot(p, 10, 100, 30)
    draw_chip(p, 290, 130, 10, "#b8352b")
    left, right = draw_pill(p, 200, 140, "POT 300", size=12, color="#ffffff", background="#000000")
    assert right > left
    pixels = p.canvas.to_array().astype(int)
    face, back = pixels[20, 60], pixels[50, 160]  # away from the ranks and pips
    assert face.sum() > 600 and back.sum() < 300  # an ivory face and a green back
    red = pixels[20:90, 80:140]
    assert ((red[..., 0] > 150) & (red[..., 1] < 90)).any()  # diamonds are red
    q = painter(60, 60)
    for suit in "shdc":
        draw_suit(q, 30, 30, 40, suit, "#ffffff")
    with pytest.raises(KeyError):
        draw_suit(q, 30, 30, 40, "x", "#ffffff")


def test_artwork() -> None:
    king = chess_piece("wK", 48)
    assert king.shape == (48, 48, 4) and king.dtype == np.uint8
    assert king[..., 3].max() == 255 and king[0, 0, 3] == 0  # opaque inside, clear outside
    assert chess_piece("wK", 48) is king
    with pytest.raises(KeyError):
        chess_piece("xK", 48)
    assert has_glyph("帥") and not has_glyph("A")
    mask = glyph("車", 40)
    assert mask.shape == (40, 40) and 0.05 < mask.mean() < 0.6
    with pytest.raises(KeyError):
        glyph("A", 40)


def test_themes() -> None:
    assert game_theme("dark") is DARK and game_theme("light") is LIGHT
    assert game_theme(DARK) is DARK
    assert DARK.tone("yours") == DARK.good and LIGHT.tone("loss") == LIGHT.bad
    with pytest.raises(ValueError):
        game_theme("pink")


def test_time_limit_detection() -> None:
    env = sekai.make("Chess-v0", max_episode_steps=2)
    env.reset(seed=0)
    assert not time_limit_reached(env)
    for _ in range(2):
        result = env.step(int(env.unwrapped.legal_actions()[0]))
    assert result.truncated and time_limit_reached(env)


def test_random_policy() -> None:
    policy = RandomPolicy(seed=0)
    assert isinstance(policy, Policy)
    mask = np.zeros(10, dtype=np.int8)
    mask[[2, 7]] = 1
    picks = {policy.act(None, {"legal_action_mask": mask}) for _ in range(50)}
    assert picks == {2, 7}
    assert policy.seed(3) == 3
    with pytest.raises(ValueError):
        policy.act(None, {"legal_action_mask": np.zeros(10, dtype=np.int8)})


def test_opponent_wrapper_needs_a_two_player_game() -> None:
    with pytest.raises(TypeError):
        OpponentWrapper(sekai.make("CartPole-v1"))  # type: ignore[arg-type]
    env = OpponentWrapper(sekai.make("Chess-v0"), agent_color="second")
    result: Any = env.reset(seed=0)
    assert result.info["agent_color"] is False and result.info["to_play"] is False
    assert "OpponentWrapper" in str(env)
