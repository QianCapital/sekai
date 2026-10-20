"""Canvas anti-aliasing, text, themes, chart helpers and dashboards."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

import sekai
from sekai.rendering import (
    DARK,
    LIGHT,
    BarChart,
    Canvas,
    Column,
    Dashboard,
    LineChart,
    Panel,
    RecordFrames,
    Row,
    Series,
    StackedArea,
    Stat,
    Theme,
    get_theme,
    text_width,
)
from sekai.rendering.charts import (
    MINUS,
    Box,
    Plot,
    decimals_for,
    format_number,
    nice_ticks,
    tick_labels,
    value_range,
)
from sekai.rendering.dashboard import read, resolve_color
from sekai.rendering.text import face, font_sizes
from sekai.rendering.theme import mix, rgb

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


def ink(canvas: Canvas) -> np.ndarray:
    """How much of each pixel is covered by black drawn on white (0 to 1)."""
    return 1.0 - canvas.to_array()[:, :, 0].astype(np.float64) / 255.0


class TestAntialiasing:
    def test_line_covers_its_area(self) -> None:
        c = Canvas(40, 20, antialias=True)
        c.line((5, 10), (35, 10), BLACK, width=2)
        # A capsule: a 30 x 2 rectangle and two half-discs of radius 1.
        assert ink(c).sum() == pytest.approx(30 * 2 + math.pi, rel=0.03)

    def test_rect_on_fractional_pixels(self) -> None:
        c = Canvas(4, 1, antialias=True)
        c.rect(0.5, 0, 2.5, 1, BLACK)
        np.testing.assert_allclose(ink(c)[0], [0.5, 1.0, 0.5, 0.0], atol=0.01)

    def test_rounded_rect_is_smaller_than_square(self) -> None:
        square, rounded = Canvas(20, 20, antialias=True), Canvas(20, 20, antialias=True)
        square.rect(2, 2, 18, 18, BLACK)
        rounded.rect(2, 2, 18, 18, BLACK, radius=4)
        lost = ink(square).sum() - ink(rounded).sum()
        assert lost == pytest.approx((4 - math.pi) * 16, rel=0.05)  # four corners

    def test_circle_area(self) -> None:
        c = Canvas(40, 40, antialias=True)
        c.circle(20.3, 19.7, 9.5, BLACK)
        assert ink(c).sum() == pytest.approx(math.pi * 9.5**2, rel=0.01)

    def test_polygon_area(self) -> None:
        c = Canvas(30, 30, antialias=True)
        c.polygon([(3, 3), (27, 3), (15, 25)], BLACK)
        assert ink(c).sum() == pytest.approx(0.5 * 24 * 22, rel=0.02)

    def test_translucent_polyline_overlaps_are_drawn_once(self) -> None:
        c = Canvas(40, 10, antialias=True)
        c.polyline([(5, 5), (35, 5), (5, 5)], BLACK, 3, alpha=0.5)  # retraces itself
        assert ink(c).max() == pytest.approx(0.5, abs=0.01)

    def test_nan_breaks_a_polyline(self) -> None:
        c = Canvas(40, 10, antialias=True)
        c.polyline([(2, 5), (10, 5), (np.nan, np.nan), (30, 5), (38, 5)], BLACK, 2)
        coverage = ink(c).sum(axis=0)
        assert coverage[15:25].max() == 0 and coverage[4:9].min() > 0

    def test_dashed_line_has_gaps(self) -> None:
        c = Canvas(60, 6, antialias=True)
        c.line((0, 3), (60, 3), BLACK, width=1, dash=(4, 4))
        column = ink(c).sum(axis=0)
        assert (column < 0.05).sum() >= 14 and (column > 0.9).sum() >= 28  # 4 on, 4 off
        with pytest.raises(ValueError, match="dash"):
            c.line((0, 3), (60, 3), BLACK, dash=(0, 4))

    def test_alpha_blends(self) -> None:
        c = Canvas(2, 2)
        c.rect(0, 0, 2, 2, BLACK, alpha=0.25)
        assert (c.to_array() == round(255 * 0.75)).all()

    def test_fill_between_area(self) -> None:
        c = Canvas(50, 30, antialias=True)
        x = np.array([5.0, 25.0, 45.0])
        c.fill_between(x, 5.0, np.array([5.0, 25.0, 5.0]), BLACK)  # a triangle
        assert ink(c).sum() == pytest.approx(0.5 * 40 * 20, rel=0.02)

    def test_fill_between_skips_gaps(self) -> None:
        c = Canvas(30, 10, antialias=True)
        c.fill_between([0, 10, 20, 30], 0, [8, np.nan, 8, 8], BLACK)
        assert ink(c)[:, 5:15].sum() == 0 and ink(c)[:, 22:28].sum() > 0

    def test_clip(self) -> None:
        c = Canvas(20, 20, antialias=True)
        with c.clip(5, 5, 15, 15):
            c.rect(0, 0, 20, 20, BLACK)
            with c.clip(0, 0, 10, 10):  # nested clips intersect
                c.circle(10, 10, 3, (255, 0, 0))
        covered = (c.to_array() != 255).any(axis=-1)
        assert covered.sum() == 100 and covered[5:15, 5:15].all()
        with pytest.raises(ValueError), c.clip(0, 0, np.inf, 1):
            pass

    def test_image_scales_with_crisp_blocks(self) -> None:
        c = Canvas(8, 8)
        image = np.zeros((2, 2, 4), np.uint8)
        image[0, 0] = (255, 0, 0, 255)  # top-left red
        image[1, 1] = (0, 0, 255, 128)  # bottom-right half-transparent blue
        c.image(image, 0, 0, 8, 8)
        frame = c.to_array()
        assert (frame[:4, :4] == (255, 0, 0)).all()
        assert (frame[:4, 4:] == WHITE).all()  # transparent cell
        assert (frame[4:, 4:, 2] == 255).all() and (frame[4:, 4:, 0] < 140).all()
        with pytest.raises(ValueError):
            c.image(np.zeros((2, 2)), 0, 0, 1, 1)

    def test_default_canvas_is_unchanged_by_antialiasing_features(self) -> None:
        # The classic-control renderers depend on the original pixel-centre fills.
        c = Canvas(10, 10)
        c.line((0, 5), (10, 5), BLACK, width=2)
        assert set(np.unique(c.to_array())) == {0, 255}


class TestText:
    def test_draws_and_measures(self) -> None:
        c = Canvas(200, 40)
        width = c.text(10, 15, "Equity 1.0432", BLACK, size=14)
        assert width == pytest.approx(text_width("Equity 1.0432", 14))
        cols = np.flatnonzero(ink(c).sum(axis=0) > 0)
        assert cols.min() >= 9 and cols.max() <= 10 + width + 1

    @pytest.mark.parametrize("align", ["left", "center", "right"])
    def test_alignment(self, align: str) -> None:
        c = Canvas(200, 40)
        width = c.text(100, 15, "Price", BLACK, size=12, align=align)  # type: ignore[arg-type]
        cols = np.flatnonzero(ink(c).sum(axis=0) > 0.1)
        centre = {"left": 100 + width / 2, "center": 100, "right": 100 - width / 2}[align]
        assert (cols.min() + cols.max()) / 2 == pytest.approx(centre, abs=2)

    def test_vertical_anchors(self) -> None:
        c = Canvas(60, 60)
        c.text(5, 30, "HH", BLACK, size=20, valign="middle")
        rows = np.flatnonzero(ink(c).sum(axis=1) > 0.5)
        assert (rows.min() + rows.max()) / 2 == pytest.approx(60 - 30, abs=1.5)

    def test_sizes_snap_and_scale(self) -> None:
        assert 9 in font_sizes() and 32 in font_sizes()
        assert face(12.4).size == 12 and face(12.6).size == 13
        assert face(60).size == 60 and face(60).cap_height > face(32).cap_height * 1.7
        assert text_width("Bold", 14, bold=True) > text_width("Bold", 14)

    def test_digits_are_tabular(self) -> None:
        assert text_width("1111", 12) == text_width("8888", 12) == text_width("0000", 12)

    def test_unknown_characters(self) -> None:
        assert text_width("caf\u00e9", 12) == text_width("cafe", 12)  # accent dropped
        assert text_width("\u03c3", 12) == text_width("?", 12)  # not in the font

    def test_text_respects_clip(self) -> None:
        c = Canvas(100, 30)
        with c.clip(0, 0, 20, 30):
            c.text(2, 10, "Clipped text", BLACK, size=14)
        assert ink(c)[:, 21:].sum() == 0 and ink(c)[:, :20].sum() > 0


class TestTheme:
    def test_rgb(self) -> None:
        assert rgb("#1f271b") == (31, 39, 27) and rgb("#fff") == WHITE
        assert rgb((1, 2, 3)) == (1, 2, 3)
        with pytest.raises(ValueError):
            rgb("#12345")
        assert mix(BLACK, WHITE, 0.5) == (128, 128, 128)

    def test_builtin_themes(self) -> None:
        for theme in (LIGHT, DARK):
            assert len(theme.series) == 8 and len(set(theme.series)) == 8
            assert theme.color(8) == theme.neutral  # never cycled
        assert get_theme("dark") is DARK and get_theme(LIGHT) is LIGHT
        with pytest.raises(ValueError, match="theme"):
            get_theme("neon")

    def test_with_colors(self) -> None:
        custom = LIGHT.with_colors(up="#2a78d6", series=["#000000"])
        assert isinstance(custom, Theme) and custom.up == (42, 120, 214)
        assert custom.series == (BLACK,) and custom.down == LIGHT.down

    def test_ink_on_fills(self) -> None:
        assert LIGHT.ink_on(LIGHT.text) == LIGHT.background
        assert LIGHT.ink_on("#f0f0f0") == LIGHT.text
        assert DARK.ink_on(DARK.text) == DARK.background

    def test_resolve_color(self) -> None:
        assert resolve_color(LIGHT, None, 2) == LIGHT.series[2]
        assert resolve_color(LIGHT, "accent", 0) == LIGHT.accent
        assert resolve_color(LIGHT, 1, 0) == LIGHT.series[1]
        assert resolve_color(LIGHT, "#000000", 0) == BLACK
        with pytest.raises(ValueError, match="role"):
            resolve_color(LIGHT, "teal", 0)


class TestChartHelpers:
    @pytest.mark.parametrize(
        ("low", "high"), [(0.0, 1.0), (0.917, 1.003), (9990.2, 9997.4), (-3.5, 3.5), (1e6, 1e6 + 1)]
    )
    def test_nice_ticks(self, low: float, high: float) -> None:
        ticks = nice_ticks(low, high, 3)
        assert ticks.size >= 2 and ticks.min() >= low - 1e-9 and ticks.max() <= high + 1e-9
        steps = np.diff(ticks)
        np.testing.assert_allclose(steps, steps[0])

    def test_integer_ticks(self) -> None:
        ticks = nice_ticks(-3.5, 3.5, 3, integer=True)
        assert (ticks == np.round(ticks)).all() and ticks.size >= 2
        assert nice_ticks(np.nan, 1).size == 0 and nice_ticks(2, 2).tolist() == [2]

    def test_format_number(self) -> None:
        assert format_number(-1234.5, 1) == MINUS + "1,234.5"
        assert format_number(0.0432, 2, sign=True, percent=True) == "+4.32%"
        assert format_number(-0.0001, 2, sign=True) == "0.00"  # no sign on zero
        assert format_number(2_500_000, compact=True) == "2.50M"
        assert format_number(math.nan) == "—"
        assert decimals_for(2.5) == 1 and decimals_for(0.05) == 2 and decimals_for(10) == 0

    def test_tick_labels(self) -> None:
        assert tick_labels([0.9, 1.0, 1.1]) == ["0.9", "1.0", "1.1"]
        assert tick_labels([0, 0.5, 1.0], percent=True) == ["0%", "50%", "100%"]
        assert tick_labels([0, 500_000, 1_000_000]) == ["0", "500K", "1.00M"]

    def test_value_range(self) -> None:
        lo, hi = value_range([1.0, 2.0], [np.nan], include=[0.0])
        assert lo < 0 < 2 < hi
        assert value_range([]) == (0.0, 1.0)
        lo, hi = value_range([5.0, 5.0])
        assert lo < 5 < hi

    def test_boxes(self) -> None:
        rows = Box(0, 0, 100, 100).rows([3, 1], gap=10)
        assert rows[0].top == 100 and rows[1].bottom == 0
        assert rows[0].height == pytest.approx(67.5) and rows[0].bottom - rows[1].top == 10
        cols = Box(0, 0, 100, 50).columns([1, 1])
        assert cols[0].right == cols[1].left == 50
        with pytest.raises(ValueError):
            Box(0, 0, 1, 1).rows([])

    def test_plot_maps_data_to_pixels(self) -> None:
        c = Canvas(100, 100, antialias=True)
        plot = Plot(c, Box(10, 10, 90, 90), (0, 10), (0, 1), LIGHT)
        assert plot.x(5) == pytest.approx(50) and plot.y(1) == pytest.approx(90)
        plot.line([0, 10], [0, 1], LIGHT.accent)
        plot.marker(5, 0.5, LIGHT.down, shape="down")
        plot.tag(0.5, "0.50", LIGHT.price)
        assert (c.to_array() != LIGHT.background).any()


# ----------------------------------------------------------------------
# Dashboards
# ----------------------------------------------------------------------


class Walk(sekai.Env[np.ndarray, int]):
    """A toy market: a random-walk price, a position chosen by the action, equity."""

    metadata = {"render_fps": 10}

    def __init__(self, episode_length: int = 12) -> None:
        self.episode_length = episode_length
        self.observation_space = sekai.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32)
        self.action_space = sekai.Discrete(3)
        self.elapsed_steps = 0
        self.price = 100.0
        self.position = 0.0
        self.equity = 1.0
        self.weights = np.array([0.0, 1.0])

    def reset(self, *, seed: Any = None, options: dict[str, Any] | None = None) -> Any:
        self._set_rng(seed)
        self.elapsed_steps, self.price, self.position, self.equity = 0, 100.0, 0.0, 1.0
        self.weights = np.array([0.0, 1.0])
        return sekai.ResetResult(np.array([self.price], np.float32), {})

    def step(self, action: int) -> Any:
        self.position = float(action) - 1.0
        change = float(self.rng.normal(0, 0.01))
        self.equity *= 1 + self.position * change
        self.price *= 1 + change
        self.weights = np.array([abs(self.position), 1 - abs(self.position)])
        self.elapsed_steps += 1
        done = self.elapsed_steps >= self.episode_length
        return sekai.StepResult(np.array([self.price], np.float32), 0.0, False, done, {})


def walk_dashboard(**kwargs: Any) -> Dashboard:
    return Dashboard(
        LineChart(Series("price", label="Price"), Series("equity", label="Equity", rebase=100)),
        Row(
            StackedArea("weights", labels=["Asset", "Cash"]),
            BarChart("weights", labels=["Asset", "Cash"], title="Weights"),
        ),
        stats=[Stat("Equity", "equity", "{:.3f}", delta="relative")],
        width=360,
        height=225,
        **kwargs,
    )


def run(env: sekai.Env[Any, Any], renderer: Any, steps: int, seed: int = 0) -> list[np.ndarray]:
    env.reset(seed=seed)
    frames = [renderer.render(env)]
    for i in range(steps):
        env.step(i % 3)
        frames.append(renderer.render(env))
    return frames


class TestDashboard:
    def test_renders_an_episode(self) -> None:
        frames = run(Walk(), walk_dashboard(), 12)
        assert all(f.shape == (225, 360, 3) and f.dtype == np.uint8 for f in frames)
        assert not np.array_equal(frames[1], frames[-1])

    def test_records_one_sample_per_step(self) -> None:
        env, renderer = Walk(), walk_dashboard()
        run(env, renderer, 5)
        renderer.render(env)  # the same step again: redrawn, not recorded twice
        history = renderer._histories[env]
        assert history.steps == [0, 1, 2, 3, 4, 5]
        env.reset(seed=1)  # a new episode starts a new history
        renderer.render(env)
        assert renderer._histories[env].steps == [0]

    def test_deterministic(self) -> None:
        a = run(Walk(), walk_dashboard(), 6, seed=3)
        b = run(Walk(), walk_dashboard(), 6, seed=3)
        assert all(np.array_equal(x, y) for x, y in zip(a, b))

    def test_one_renderer_follows_several_envs(self) -> None:
        renderer = walk_dashboard()
        one, two = Walk(), Walk()
        one.reset(seed=0)
        two.reset(seed=1)
        for _ in range(3):
            one.step(2)
            renderer.render(one)
        renderer.render(two)
        assert renderer._histories[one].steps == [1, 2, 3]
        assert renderer._histories[two].steps == [0]
        renderer.reset(one)
        assert one not in renderer._histories and two in renderer._histories

    def test_horizon_fixes_the_time_axis(self) -> None:
        env = Walk(episode_length=40)
        renderer = walk_dashboard()
        assert renderer._horizon(env) == 40
        assert walk_dashboard(horizon=None)._horizon(env) is None
        assert walk_dashboard(horizon=7)._horizon(env) == 7

    def test_without_a_clock_every_render_is_a_step(self) -> None:
        env, renderer = Walk(), walk_dashboard(step=None)
        run(env, renderer, 3)
        assert renderer._histories[env].steps == [0, 1, 2, 3]

    def test_window_scrolls(self) -> None:
        env, renderer = Walk(episode_length=50), walk_dashboard(window=10)
        run(env, renderer, 30)
        assert len(renderer._histories[env].steps) == 11

    def test_through_wrappers_and_record_frames(self) -> None:
        env = RecordFrames(sekai.wrappers.TimeLimit(Walk(), 8), walk_dashboard())
        result = env.reset(seed=0)
        while not result.done:
            result = env.step(2)
        assert len(env.frames) == 9

    def test_title_and_theme(self) -> None:
        light = run(Walk(), walk_dashboard(), 2)[-1]
        dark = run(Walk(), walk_dashboard(theme="dark"), 2)[-1]
        assert tuple(light[0, 0]) == LIGHT.background and tuple(dark[0, 0]) == DARK.background

    def test_sources(self) -> None:
        env = sekai.wrappers.TimeLimit(Walk(), 5)
        env.reset(seed=0)
        assert read(env, "price") == 100.0
        assert read(env, lambda e: e.unwrapped.equity) == 1.0
        assert read(env, "weights.size") == 2
        assert read(env, "max_episode_steps") == 5  # found on the wrapper
        with pytest.raises(AttributeError):
            read(env, "missing")

    def test_missing_source_raises(self) -> None:
        env = Walk()
        env.reset(seed=0)
        with pytest.raises(AttributeError):
            Dashboard(LineChart("nothing_here")).render(env)

    def test_custom_panel_and_layout(self) -> None:
        drawn: list[tuple[float, float]] = []

        class Dot(Panel):
            def sources(self) -> dict[str, Any]:
                return {"price": "price"}

            def draw(self, canvas: Canvas, box: Box, view: Any) -> None:
                plot_box = self.frame(canvas, box, view)
                drawn.append(view.x_domain)
                Plot(canvas, plot_box, view.x_domain, (90, 110), view.theme).marker(
                    view.step, float(view.values("price")[-1]), view.theme.accent
                )

        renderer = Dashboard(Row(Column(Dot("A"), Dot("B")), Dot("C"), widths=(2, 1)))
        run(Walk(episode_length=10), renderer, 2)
        assert len(drawn) == 9 and drawn[-1] == (0.0, 10.0)
        renderer.set_layout(Dot("only"))
        assert len(renderer.panels) == 1

    def test_invalid_arguments(self) -> None:
        with pytest.raises(ValueError):
            Dashboard(width=100, height=100)
        with pytest.raises(ValueError):
            Dashboard(window=0)
        with pytest.raises(TypeError):
            Dashboard("not a panel")
        with pytest.raises(ValueError):
            LineChart()
        with pytest.raises(ValueError):
            Panel(width=0)


class TestSaveGif:
    def test_shared_palette_is_exact_and_stores_only_changes(self, tmp_path: Any) -> None:
        pytest.importorskip("PIL")
        from PIL import Image

        from sekai.rendering import save_gif

        rng = np.random.default_rng(0)
        base = rng.integers(0, 256, size=(40, 60, 3)).astype(np.uint8)
        colors = np.unique(base.reshape(-1, 3), axis=0)[:200]  # 200 colours in all
        base = colors[rng.integers(0, len(colors), size=(40, 60))]
        frames = [base.copy() for _ in range(6)]
        for i, frame in enumerate(frames):
            frame[5:9, 10 * i : 10 * i + 4] = colors[i]  # a small change per frame
        with Image.open(save_gif(frames, tmp_path / "anim.gif", fps=10)) as gif:
            for i, frame in enumerate(frames):
                gif.seek(i)
                np.testing.assert_array_equal(np.asarray(gif.convert("RGB")), frame)
        one = save_gif(frames[:1], tmp_path / "one.gif").stat().st_size
        later = (tmp_path / "anim.gif").stat().st_size - one
        assert later < 5 * one / 2  # each later frame stores little more than its change
