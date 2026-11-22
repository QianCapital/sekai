"""The Qian Capital marks of branded dashboards: the gold mark, the wordmark and the gold rule.

sekai is made by Qian Capital. Its finance renderers draw their header in the house style
by default (`branding=True`; a `Dashboard` you build yourself is plain by default): the
gold mark beside a header that carries the wordmark above a Hedvig Letters Serif title,
and the episode's progress drawn as a rule of gold metal. This module draws those marks;
`Dashboard` lays them out.

The mark and wordmark are Qian Capital's logo, cut from the documentation's lockup
(`docs/assets/brand/qian-capital.png`) by `tools/build_brand_assets.py` and stored as
RGBA arrays in `assets/qian-capital.npz`. They are resampled to the size a frame needs
(area-averaged, with premultiplied alpha), so they stay smooth at any frame size.

The Qian Capital name and logo are trademarks of Qian Capital, and the artwork is not
licensed under the Apache License 2.0 that covers sekai's code (nor under the SIL Open
Font License of its fonts). You may show them as sekai draws them in its own dashboards
(the finance renderers), including in figures and recordings of those dashboards, and
redistribute them unmodified as part of sekai. Anything else, such as branding a
dashboard or layout of your own (with `branding=True`, `draw_mark` or `draw_wordmark`),
needs written permission from Qian Capital: see the TRADEMARKS file that comes with sekai.
(`draw_rule` and `gold_metal` draw only a colour gradient, which is not a mark.)
"""

from __future__ import annotations

import functools
from importlib import resources
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from sekai.rendering.canvas import Canvas
from sekai.rendering.theme import Color, rgb

_ASSETS = "assets/qian-capital.npz"

GOLD_METAL: tuple[tuple[float, Color], ...] = tuple(
    (stop, rgb(color))
    for stop, color in (
        (0.0, "#946127"),
        (0.2, "#b18948"),
        (0.45, "#e2cf86"),
        (0.55, "#d3ba70"),
        (0.8, "#aa8040"),
        (1.0, "#936026"),
    )
)
"""The gold of the logo, dark bronze to pale gold and back, as (position, colour) stops."""


def _frozen(array: NDArray[Any]) -> NDArray[Any]:
    """`array`, made read-only: cached images are shared by every frame that draws them."""
    array.flags.writeable = False
    return array


@functools.lru_cache(maxsize=1)
def _assets() -> dict[str, NDArray[np.generic]]:
    source = resources.files("sekai.rendering").joinpath(_ASSETS)
    with source.open("rb") as f, np.load(f) as archive:
        return {name: _frozen(archive[name]) for name in archive.files}


def _weights(n_in: int, n_out: int) -> NDArray[np.float64]:
    """`(n_out, n_in)` resampling weights: area-averaging down, linear up."""
    scale = n_in / n_out
    if scale >= 1.0:  # each output pixel averages the input it covers
        edges = np.arange(n_out + 1) * scale
        lo, hi = edges[:-1, None], edges[1:, None]
        cells = np.arange(n_in)[None, :]
        overlap = np.clip(np.minimum(hi, cells + 1) - np.maximum(lo, cells), 0, None)
        weights: NDArray[np.float64] = overlap / scale
        return weights
    centres = np.clip((np.arange(n_out) + 0.5) * scale - 0.5, 0, n_in - 1)
    weights = np.clip(1 - np.abs(centres[:, None] - np.arange(n_in)[None, :]), 0, None)
    out: NDArray[np.float64] = weights / weights.sum(axis=1, keepdims=True)
    return out


def _taps(weights: NDArray[np.float64]) -> tuple[NDArray[np.intp], NDArray[np.float64]]:
    """The inputs each output reads, as `(n_out, taps)` indices and weights, in input order.

    Each row of `weights` is non-zero over a short run of inputs; rows with shorter runs
    are padded with zero weights.
    """
    n_in = weights.shape[1]
    nonzero = weights != 0
    first = nonzero.argmax(axis=1)
    last = n_in - 1 - nonzero[:, ::-1].argmax(axis=1)
    span = first[:, None] + np.arange(int((last - first).max()) + 1)[None, :]
    index = np.minimum(span, n_in - 1)
    taps = np.where(span <= last[:, None], np.take_along_axis(weights, index, axis=1), 0.0)
    return index, taps


def _resample(image: NDArray[np.generic], height: int, width: int) -> NDArray[np.uint8]:
    """An RGBA image resampled to `(height, width)`, in premultiplied alpha.

    Each output pixel is the weighted sum `sum_jk (rows[i, j] * image[j, k]) * cols[l, k]`
    over the few input pixels it covers, accumulated in input order. Weights outside them
    are zero and would add nothing, so the sum is exactly that over the whole image (the
    product of the full weight matrices with the image), without its cost.
    """
    pixels = image.astype(np.float64) / 255.0
    alpha = pixels[..., 3:]
    premultiplied = np.concatenate([pixels[..., :3] * alpha, alpha], axis=-1)
    rows, row_weights = _taps(_weights(image.shape[0], height))
    cols, col_weights = _taps(_weights(image.shape[1], width))
    out = np.zeros((height, width, 4))
    for j in range(rows.shape[1]):
        band = row_weights[:, j, None, None] * premultiplied[rows[:, j]]  # (height, w_in, 4)
        for k in range(cols.shape[1]):
            out += band[:, cols[:, k]] * col_weights[None, :, k, None]
    a = out[..., 3:]
    color = np.divide(out[..., :3], a, out=np.zeros_like(out[..., :3]), where=a > 1e-6)
    rgba = np.concatenate([color, a], axis=-1)
    result: NDArray[np.uint8] = np.clip(np.round(rgba * 255.0), 0, 255).astype(np.uint8)
    return result


@functools.lru_cache(maxsize=16)
def _mark(height: int) -> NDArray[np.uint8]:
    source = _assets()["mark"]
    width = max(round(source.shape[1] * height / source.shape[0]), 1)
    image = _resample(source, height, width)
    # The mark is drawn in hairlines, which averaging thins to a faint trace when it is
    # shrunk: strengthen their coverage by up to twice, as much as the image shrank.
    shrink = min(max(source.shape[0] / height, 1.0), 2.0)
    alpha = image[..., 3].astype(np.float64) / 255.0
    image[..., 3] = np.round(255.0 * (1.0 - (1.0 - alpha) ** shrink)).astype(np.uint8)
    return _frozen(image)


@functools.lru_cache(maxsize=16)
def _wordmark(height: int) -> NDArray[np.uint8]:
    source = _assets()["wordmark"]
    width = max(round(source.shape[1] * height / source.shape[0]), 1)
    return _frozen(_resample(source, height, width))


def mark(height: float) -> NDArray[np.uint8]:
    """The gold Q mark as an `(h, w, 4)` RGBA image `height` pixels tall.

    The image is cached and shared, so it is read-only (copy it to change it).
    """
    return _mark(max(round(height), 1))


def wordmark(cap_height: float) -> tuple[NDArray[np.uint8], int]:
    """The "QIAN CAPITAL" wordmark as RGBA, its capitals `cap_height` pixels tall.

    Returns the image (cached and shared, so read-only) and the row (from its top) where
    the capitals end: the Q's tail hangs below them.
    """
    top, bottom = (int(v) for v in _assets()["wordmark_caps"])
    height = _assets()["wordmark"].shape[0]
    image = _wordmark(max(round(height * cap_height / (bottom - top)), 1))
    return image, round(bottom * image.shape[0] / height)


def draw_mark(canvas: Canvas, left: float, top: float, height: float) -> float:
    """Draw the mark with its top-left corner at `(left, top)`; return its width."""
    image = mark(height)
    x, y = round(left), round(top)
    canvas.image(image, x, y - image.shape[0], x + image.shape[1], y)
    return float(image.shape[1])


def draw_wordmark(
    canvas: Canvas,
    x: float,
    y: float,
    cap_height: float,
    *,
    valign: Literal["top", "middle", "baseline"] = "middle",
) -> float:
    """Draw the wordmark from `x`, its capitals' top, middle or baseline at `y`; return its width.

    Placed like `Canvas.text`, so it lines up with text drawn beside it.
    """
    image, baseline = wordmark(cap_height)
    caps_top, _ = (int(v) for v in _assets()["wordmark_caps"])
    caps_top = round(caps_top * image.shape[0] / _assets()["wordmark"].shape[0])
    caps = baseline - caps_top
    if valign == "top":
        y += caps_top
    elif valign == "middle":
        y += caps_top + caps / 2
    else:
        y += baseline
    left, top = round(x), round(y)
    canvas.image(image, left, top - image.shape[0], left + image.shape[1], top)
    return float(image.shape[1])


def gold_metal(width: int) -> NDArray[np.uint8]:
    """`(width, 3)` colours of the gold-metal gradient across `width` pixels."""
    stops = np.asarray([s for s, _ in GOLD_METAL])
    colors = np.asarray([c for _, c in GOLD_METAL], dtype=np.float64)
    t = (np.arange(max(width, 1)) + 0.5) / max(width, 1)
    channels = [np.interp(t, stops, colors[:, k]) for k in range(3)]
    out: NDArray[np.uint8] = np.round(np.stack(channels, axis=-1)).astype(np.uint8)
    return out


def draw_rule(
    canvas: Canvas,
    left: float,
    bottom: float,
    right: float,
    top: float,
    *,
    span: float | None = None,
) -> None:
    """Fill a box with gold metal, its gradient spread across `span` pixels from `left`.

    With `span` the full width of a track and `right` short of it, the rule uncovers one
    gradient as it grows, as the dashboard's progress rule does. Edges that fall inside a
    pixel are drawn with partial coverage.
    """
    if right <= left or top <= bottom:
        return
    span = right - left if span is None else span
    x0, x1 = int(np.floor(left)), int(np.ceil(right))
    y0, y1 = int(np.floor(bottom)), int(np.ceil(top))
    cx = np.arange(x0, x1, dtype=np.float64)
    cy = np.arange(y1 - 1, y0 - 1, -1, dtype=np.float64)  # image rows, top first
    cover_x = np.clip(np.minimum(cx + 1, right) - np.maximum(cx, left), 0, 1)
    cover_y = np.clip(np.minimum(cy + 1, top) - np.maximum(cy, bottom), 0, 1)
    metal = gold_metal(max(round(span), 1))
    columns = np.clip(np.floor(cx + 0.5 - left).astype(np.intp), 0, metal.shape[0] - 1)
    rgba = np.empty((y1 - y0, x1 - x0, 4), dtype=np.uint8)
    rgba[..., :3] = metal[columns][None, :, :]
    rgba[..., 3] = np.round(255 * cover_y[:, None] * cover_x[None, :]).astype(np.uint8)
    canvas.image(rgba, x0, y0, x1, y1)


__all__ = [
    "GOLD_METAL",
    "draw_mark",
    "draw_rule",
    "draw_wordmark",
    "gold_metal",
    "mark",
    "wordmark",
]
