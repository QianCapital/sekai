"""Rendering: renderers that draw environments into frames, and the tools to build them.

- `Renderer` / `BaseRenderer`: the protocol (`render(env)` returns a frame) and a base.
- `RecordFrames`, `save_gif`, `save_video`: record frames and save them.
- `Canvas`: a NumPy raster with anti-aliased shapes, opacity, clipping and text.
- `Dashboard` and its charts (`LineChart`, `BarChart`, `StackedArea`, ...): renderers
  that follow an environment through an episode; `sekai.finance.rendering` adds
  market charts.
- `Theme`, `LIGHT`, `DARK`: colours by role, with validated categorical palettes.
- `charts`: the layout, tick, number-format and plot helpers charts are drawn with.
"""

from sekai.rendering.renderer import ANSIString, BaseRenderer, Renderer, RenderFrame, RGBArray
from sekai.rendering.canvas import Canvas, rotate
from sekai.rendering.text import text_width
from sekai.rendering.theme import DARK, LIGHT, Theme, get_theme
from sekai.rendering.recorder import RecordFrames, save_gif, save_video
from sekai.rendering.dashboard import (
    BarChart,
    Column,
    Dashboard,
    LineChart,
    Panel,
    Row,
    Series,
    StackedArea,
    Stat,
    View,
)
from sekai.rendering import charts

__all__ = [
    "Renderer",
    "BaseRenderer",
    "RenderFrame",
    "RGBArray",
    "ANSIString",
    "Canvas",
    "rotate",
    "text_width",
    "RecordFrames",
    "save_gif",
    "save_video",
    # Dashboards
    "Dashboard",
    "Panel",
    "View",
    "Row",
    "Column",
    "Series",
    "Stat",
    "LineChart",
    "BarChart",
    "StackedArea",
    # Themes
    "Theme",
    "LIGHT",
    "DARK",
    "get_theme",
    "charts",
]
