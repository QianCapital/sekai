"""Unbranded dashboards draw exactly what sekai 0.2.0 drew.

tests/data/render_reference.npz holds digests of every frame of short episodes drawn by
sekai 0.2.0 (made by tests/data/render_reference.py; read it before regenerating the
file). The finance renderers with `branding=False`, and dashboards built from
`sekai.rendering.Dashboard` with its default arguments, must draw the same frames, byte
for byte, as must dashboards built from charts and themes that 0.2.0 pickled.
"""

from __future__ import annotations

import pickle
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sekai.rendering import DARK, LIGHT, Dashboard
from tests.data.render_reference import (
    DASHBOARDS,
    RENDERERS,
    SIZES,
    THEMES,
    Desk,
    dashboard_frames,
    digest,
    finance_frames,
    finance_renderer,
)

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def reference() -> dict[str, np.ndarray]:
    with np.load(DATA / "render_reference.npz") as data:
        return {key: data[key] for key in data.files}


def differs(frames: Iterable[np.ndarray], expected: np.ndarray) -> str | None:
    """Where `frames` first differ from the reference digests, or None."""
    digests = [digest(frame) for frame in frames]
    if len(digests) != len(expected):
        return f"{len(digests)} frames, not {len(expected)}"
    first = next((i for i, (a, b) in enumerate(zip(digests, expected.tolist())) if a != b), None)
    return None if first is None else f"frame {first} differs from sekai 0.2.0's"


@pytest.mark.parametrize("size", SIZES, ids=lambda size: f"{size[0]}x{size[1]}")
@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("name", RENDERERS)
def test_finance_renderers_without_branding(
    reference: dict[str, np.ndarray], name: str, theme: str, size: tuple[int, int]
) -> None:
    renderer = finance_renderer(name, theme, size, branding=False)
    expected = reference[f"finance_{name}_{theme}_{size[0]}x{size[1]}"]
    assert differs(finance_frames(name, renderer), expected) is None


@pytest.mark.parametrize("name", sorted(DASHBOARDS))
def test_dashboards_you_build(reference: dict[str, np.ndarray], name: str) -> None:
    build, make_env = DASHBOARDS[name]
    expected = reference[f"dashboard_{name}"]
    assert differs(dashboard_frames(build(), make_env()), expected) is None  # the defaults
    assert differs(dashboard_frames(build(branding=False), make_env()), expected) is None


def test_charts_and_themes_pickled_by_0_2_0(reference: dict[str, np.ndarray]) -> None:
    objects: dict[str, Any] = pickle.loads(reference["pickle"].tobytes())
    assert objects["light"] == LIGHT and objects["dark"] == DARK
    renderer = Dashboard(*objects["items"], stats=objects["stats"], theme=objects["light"])
    assert differs(dashboard_frames(renderer, Desk()), reference["dashboard_desk"]) is None


def test_a_renderer_without_a_branding_setting_draws_as_0_2_0(
    reference: dict[str, np.ndarray],
) -> None:
    # A renderer restored from a state saved by 0.2.0 has no `branding` of its own.
    renderer = finance_renderer("trading", "light", (720, 450))
    assert renderer.branding  # the finance renderers are branded by default
    del renderer.branding
    assert renderer.branding is False
    expected = reference["finance_trading_light_720x450"]
    assert differs(finance_frames("trading", renderer), expected) is None
