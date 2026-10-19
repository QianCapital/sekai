from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import sekai
from sekai.envs.classic_control import (
    AcrobotRenderer,
    CartPoleRenderer,
    MountainCarRenderer,
    PendulumRenderer,
)
from sekai.rendering import (
    BaseRenderer,
    Canvas,
    RecordFrames,
    Renderer,
    rotate,
    save_gif,
    save_video,
)


class TestCanvas:
    def test_rect_fills_expected_pixels(self) -> None:
        c = Canvas(10, 10)
        c.rect(0, 0, 5, 3, (255, 0, 0))
        img = c.to_array()
        assert img.shape == (10, 10, 3) and img.dtype == np.uint8
        red = (img == [255, 0, 0]).all(-1)
        assert red.sum() == 15
        assert red[-1, 0] and not red[0, 0]  # y-up: bottom rows are filled

    def test_circle_area(self) -> None:
        c = Canvas(101, 101)
        c.circle(50.5, 50.5, 20, (0, 0, 0))
        area = (c.to_array() == 0).all(-1).sum()
        assert area == pytest.approx(np.pi * 400, rel=0.03)

    def test_offscreen_shapes_are_clipped(self) -> None:
        c = Canvas(5, 5)
        c.polygon([(-10, -10), (-5, -10), (-5, -5)], (0, 0, 0))
        c.circle(100, 100, 3, (0, 0, 0))
        assert (c.to_array() == 255).all()

    def test_line_and_rotate(self) -> None:
        c = Canvas(20, 20)
        c.line((0, 10), (20, 10), (0, 0, 0), width=2)
        assert (c.to_array()[:, 5] == 0).all(-1).sum() == 2
        np.testing.assert_allclose(rotate([(1, 0)], np.pi / 2), [[0, 1]], atol=1e-12)


@pytest.mark.parametrize(
    ("env_id", "renderer", "shape"),
    [
        ("CartPole-v1", CartPoleRenderer(), (400, 600, 3)),
        ("MountainCar-v0", MountainCarRenderer(), (400, 600, 3)),
        ("MountainCarContinuous-v0", MountainCarRenderer(), (400, 600, 3)),
        ("Pendulum-v1", PendulumRenderer(), (500, 500, 3)),
        ("Acrobot-v1", AcrobotRenderer(), (500, 500, 3)),
    ],
)
def test_classic_renderers(env_id: str, renderer: Renderer, shape: tuple[int, ...]) -> None:
    env = sekai.make(env_id)
    with pytest.raises(sekai.ResetRequired):
        renderer.render(env)
    env.reset(seed=0)
    frame = renderer.render(env)
    assert frame.shape == shape and frame.dtype == np.uint8
    assert (frame != 255).any()


def test_renderer_rejects_other_envs() -> None:
    with pytest.raises(TypeError):
        CartPoleRenderer().render(sekai.make("Pendulum-v1"))


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1", "Acrobot-v1"])
def test_renders_match_gymnasium(env_id: str) -> None:
    gym = pytest.importorskip("gymnasium")
    pytest.importorskip("pygame")
    renderers = {
        "CartPole-v1": CartPoleRenderer,
        "Pendulum-v1": PendulumRenderer,
        "Acrobot-v1": AcrobotRenderer,
    }
    g = gym.make(env_id, render_mode="rgb_array")
    s = sekai.make(env_id)
    g.reset(seed=2)
    s.reset(seed=2)
    g.action_space.seed(2)
    for _ in range(5):
        a = g.action_space.sample()
        g.step(a)
        s.step(a)
    ours, theirs = renderers[env_id]().render(s), g.render()
    agreement = (np.abs(ours.astype(int) - theirs.astype(int)).sum(-1) < 30).mean()
    assert agreement > 0.98


def test_record_frames(tmp_path: Path) -> None:
    env = RecordFrames(sekai.make("CartPole-v1"), CartPoleRenderer(width=60, height=40))
    env.reset(seed=0)
    for _ in range(4):
        env.step(0)
    assert len(env.frames) == 5
    env.enabled = False
    env.step(0)
    assert len(env.frames) == 5
    pytest.importorskip("PIL")
    path = env.save(tmp_path / "cartpole.gif")
    assert path.stat().st_size > 0
    env.clear()
    assert env.frames == []


@pytest.mark.parametrize("fps", [0, -5, float("nan")])
@pytest.mark.parametrize("save", [save_gif, save_video])
def test_save_needs_positive_fps(save: Any, fps: float, tmp_path: Path) -> None:
    frames = [np.zeros((4, 4, 3), np.uint8)]
    with pytest.raises(ValueError, match="fps > 0"):
        save(frames, tmp_path / "out.gif", fps=fps)


def test_record_frames_save_checks_fps(tmp_path: Path) -> None:
    env = RecordFrames(sekai.make("CartPole-v1"), CartPoleRenderer(width=20, height=20))
    env.reset(seed=0)
    with pytest.raises(ValueError, match="fps > 0"):
        env.save(tmp_path / "cartpole.gif", fps=0)  # 0 is not "use the default"


def test_record_frames_rejects_text_frames(tmp_path: Path) -> None:
    class TextRenderer(BaseRenderer):
        def render(self, env: object) -> str:
            return "x=0.01"

    env = RecordFrames(sekai.make("CartPole-v1"), TextRenderer())
    env.reset(seed=0)
    for suffix in (".gif", ".mp4"):
        with pytest.raises(TypeError, match="Frame 0 is text"):
            env.save(tmp_path / f"cartpole{suffix}")


@pytest.mark.parametrize("save", [save_gif, save_video])
def test_save_rejects_non_image_frames(save: Any, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"Frame 1 has shape \(4, 4, 2\)"):
        save([np.zeros((4, 4, 3)), np.zeros((4, 4, 2))], tmp_path / "out.gif")
    with pytest.raises(ValueError, match="No frames"):
        save([], tmp_path / "out.gif")


@pytest.mark.parametrize("save", [save_gif, save_video])
@pytest.mark.parametrize("shape", [(40, 60, 3), (40, 60, 4), (40, 60)])
def test_save_rejects_a_single_frame(save: Any, shape: tuple[int, ...], tmp_path: Path) -> None:
    # Iterating one image would otherwise save each of its rows as a frame.
    with pytest.raises(ValueError, match="single frame"):
        save(np.zeros(shape, np.uint8), tmp_path / "out.gif")


def test_save_gif_accepts_a_stacked_array(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    from PIL import Image

    frames = np.zeros((3, 4, 5, 3), np.uint8)  # (T, H, W, 3), e.g. np.stack(env.frames)
    assert save_gif(frames, tmp_path / "stack.gif").stat().st_size > 0
    # (T, H, W) greyscale, distinct frames (Pillow merges identical consecutive ones)
    grey = np.broadcast_to(np.array([0, 100, 200], np.uint8)[:, None, None], (3, 4, 5))
    with Image.open(save_gif(grey, tmp_path / "grey.gif")) as gif:
        assert gif.n_frames == 3 and gif.size == (5, 4)


def test_record_frames_max_frames() -> None:
    env = RecordFrames(sekai.make("Pendulum-v1"), PendulumRenderer(size=20), max_frames=2)
    env.reset(seed=0)
    for _ in range(5):
        env.step(np.zeros(1, np.float32))
    assert len(env.frames) == 2


def test_canvas_ignores_non_finite_shapes() -> None:
    c = Canvas(10, 10)
    c.polygon([(0, 0), (np.nan, 5), (5, 5)], (0, 0, 0))
    c.circle(np.inf, 3, 2, (0, 0, 0))
    c.line((0, 0), (np.nan, np.nan), (0, 0, 0))
    assert (c.to_array() == 255).all()


def test_duck_typed_renderer_satisfies_protocol() -> None:
    class TextRenderer:
        def render(self, env: object) -> str:
            return "frame"

        def close(self) -> None:
            pass

    assert isinstance(TextRenderer(), Renderer)
    assert isinstance(CartPoleRenderer(), Renderer)
    with CartPoleRenderer() as r:  # BaseRenderer adds context-manager support
        assert isinstance(r, CartPoleRenderer)
