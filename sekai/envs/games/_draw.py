"""Drawing primitives for the game renderers, in screen coordinates.

`Painter` draws on a `sekai.rendering.Canvas` with the origin at the **top-left** and y
pointing **down**, the coordinates of SVG and CSS. It adds what boards and tables need on
top of the canvas: gradient fills (linear and radial, with colour stops), rings, soft
drop shadows, noise textures, RGBA sprites, filters (brightness and saturation of a
region), playing cards, chips, pills and text along a curve.

Shapes that the canvas lacks are composited as RGBA patches through `Canvas.image`, one
patch pixel per canvas pixel, with exact anti-aliased coverage at their edges.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from sekai.rendering.canvas import Canvas
from sekai.rendering.text import face
from sekai.rendering.theme import Color, ColorLike, rgb

RGBA = NDArray[np.uint8]
Stops = Sequence[tuple[float, ColorLike]]
HAlign = Literal["left", "center", "right"]
VAlign = Literal["baseline", "middle", "top", "bottom"]


def _stops(stops: Stops) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    positions = np.asarray([float(p) for p, _ in stops], dtype=np.float64)
    colors = np.asarray([rgb(c) for _, c in stops], dtype=np.float64)
    return positions, colors


def _interp_colors(t: NDArray[np.float64], stops: Stops) -> NDArray[np.float64]:
    """Colours at positions `t` (clamped to the end stops, as SVG pads), shape `t.shape + (3,)`."""
    positions, colors = _stops(stops)
    return np.stack([np.interp(t, positions, colors[:, k]) for k in range(3)], axis=-1)


class Painter:
    """Screen-coordinate drawing on a `Canvas` (origin top-left, y down).

    Args:
        canvas: The canvas to draw on; create it with `antialias=True` for smooth shapes.
    """

    def __init__(self, canvas: Canvas) -> None:
        self.canvas = canvas
        self.width = canvas.width
        self.height = canvas.height

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------

    def _y(self, y: float) -> float:
        return self.height - y

    def _box(
        self, left: float, top: float, right: float, bottom: float, pad: float = 1.0
    ) -> tuple[int, int, int, int] | None:
        """Integer pixel bounds `(x0, y0, x1, y1)` covering a box, clipped to the canvas."""
        if not np.all(np.isfinite([left, top, right, bottom])):
            return None
        x0 = max(math.floor(min(left, right) - pad), 0)
        y0 = max(math.floor(min(top, bottom) - pad), 0)
        x1 = min(math.ceil(max(left, right) + pad), self.width)
        y1 = min(math.ceil(max(top, bottom) + pad), self.height)
        if x0 >= x1 or y0 >= y1:
            return None
        return x0, y0, x1, y1

    @staticmethod
    def _centres(box: tuple[int, int, int, int]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        x0, y0, x1, y1 = box
        xs = np.arange(x0, x1, dtype=np.float64) + 0.5
        ys = np.arange(y0, y1, dtype=np.float64) + 0.5
        return xs[None, :], ys[:, None]

    def patch(self, rgba: NDArray[np.uint8], x0: int, y0: int) -> None:
        """Composite an RGBA patch whose top-left pixel lands on canvas pixel `(x0, y0)`."""
        h, w = rgba.shape[:2]
        if h == 0 or w == 0:
            return
        self.canvas.image(rgba, x0, self.height - (y0 + h), x0 + w, self.height - y0)

    def _fill(
        self,
        box: tuple[int, int, int, int],
        coverage: NDArray[np.float64],
        color: NDArray[np.float64] | Color,
        alpha: float,
    ) -> None:
        """Composite `color` (one colour, or one per pixel) weighted by `coverage`."""
        a = np.clip(coverage, 0.0, 1.0) * min(max(alpha, 0.0), 1.0)
        if not np.any(a > 0):
            return
        x0, y0, x1, y1 = box
        rgba = np.empty((y1 - y0, x1 - x0, 4), dtype=np.uint8)
        ink = np.asarray(color, dtype=np.float64)
        rgba[..., :3] = np.clip(np.round(np.broadcast_to(ink, (*a.shape, 3))), 0, 255)
        rgba[..., 3] = np.round(a * 255.0).astype(np.uint8)
        self.patch(rgba, x0, y0)

    # ------------------------------------------------------------------
    # Canvas shapes in screen coordinates
    # ------------------------------------------------------------------

    def rect(
        self,
        left: float,
        top: float,
        right: float,
        bottom: float,
        color: ColorLike,
        *,
        alpha: float = 1.0,
        radius: Radius = 0.0,
    ) -> None:
        """Fill a rectangle, optionally with rounded corners (one radius, or per corner as
        `Radius` describes)."""
        if isinstance(radius, (int, float)):
            self.canvas.rect(
                left, self._y(bottom), right, self._y(top), rgb(color), alpha=alpha, radius=radius
            )
            return
        box = self._box(left, top, right, bottom)
        if box is None:
            return
        xs, ys = self._centres(box)
        coverage = rounded_box_coverage(xs, ys, left, top, right, bottom, radius)
        self._fill(box, coverage, rgb(color), alpha)

    def circle(
        self, cx: float, cy: float, r: float, color: ColorLike, *, alpha: float = 1.0
    ) -> None:
        """Fill a disc."""
        self.canvas.circle(cx, self._y(cy), r, rgb(color), alpha=alpha)

    def polygon(
        self, points: Sequence[tuple[float, float]], color: ColorLike, *, alpha: float = 1.0
    ) -> None:
        """Fill a simple polygon."""
        self.canvas.polygon([(x, self._y(y)) for x, y in points], rgb(color), alpha=alpha)

    def line(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        color: ColorLike,
        width: float = 1.0,
        *,
        alpha: float = 1.0,
        cap: Literal["butt", "round"] = "butt",
        dash: tuple[float, float] | None = None,
    ) -> None:
        """A straight segment. `cap="butt"` ends it square at its end points, as SVG does."""
        if cap == "round" or dash is not None:
            self.canvas.line(
                (x0, self._y(y0)), (x1, self._y(y1)), rgb(color), width, alpha=alpha, dash=dash
            )
            return
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy)
        if length == 0:
            return
        nx, ny = -dy / length * width / 2, dx / length * width / 2
        self.polygon(
            [(x0 + nx, y0 + ny), (x1 + nx, y1 + ny), (x1 - nx, y1 - ny), (x0 - nx, y0 - ny)],
            color,
            alpha=alpha,
        )

    def polyline(
        self,
        points: Sequence[tuple[float, float]],
        color: ColorLike,
        width: float = 1.0,
        *,
        alpha: float = 1.0,
        dash: tuple[float, float] | None = None,
    ) -> None:
        """Connected segments with round joins and caps."""
        self.canvas.polyline(
            [(x, self._y(y)) for x, y in points], rgb(color), width, alpha=alpha, dash=dash
        )

    def text(
        self,
        x: float,
        y: float,
        text: str,
        color: ColorLike,
        *,
        size: float = 12,
        bold: bool = False,
        align: HAlign = "left",
        valign: VAlign = "baseline",
        alpha: float = 1.0,
        serif: bool = False,
    ) -> float:
        """A line of text (see `Canvas.text`); returns its width."""
        return self.canvas.text(
            x,
            self._y(y),
            text,
            rgb(color),
            size=size,
            bold=bold,
            align=align,
            valign=valign,
            alpha=alpha,
            serif=serif,
        )

    # ------------------------------------------------------------------
    # Fills the canvas lacks
    # ------------------------------------------------------------------

    def ring(
        self,
        cx: float,
        cy: float,
        r: float,
        width: float,
        color: ColorLike,
        *,
        alpha: float = 1.0,
        dash: tuple[float, float] | None = None,
    ) -> None:
        """A circle's outline of `width` centred on radius `r` (dashed: `(on, off)` in px)."""
        outer, inner = r + width / 2, max(r - width / 2, 0.0)
        box = self._box(cx - outer, cy - outer, cx + outer, cy + outer)
        if box is None:
            return
        xs, ys = self._centres(box)
        d = np.hypot(xs - cx, ys - cy)
        coverage = np.clip(outer + 0.5 - d, 0, 1) - np.clip(inner + 0.5 - d, 0, 1)
        if dash is not None:
            on, off = dash
            along = (np.arctan2(ys - cy, xs - cx) % (2 * math.pi)) * max(r, 1e-9)
            coverage = coverage * ((along % (on + off)) < on)
        self._fill(box, coverage, rgb(color), alpha)

    def radial_disc(
        self,
        cx: float,
        cy: float,
        r: float,
        stops: Stops,
        *,
        focus: tuple[float, float] = (0.5, 0.5),
        spread: float = 0.5,
        alpha: float = 1.0,
    ) -> None:
        """A disc filled with an SVG `radialGradient` in object-bounding-box units.

        `focus` is the gradient centre as fractions of the disc's bounding box (SVG `cx`,
        `cy`), `spread` its radius as a fraction of the box (SVG `r`), and `stops` the
        `(offset, colour)` stops; beyond the last stop the colour stays the last one.
        """
        box = self._box(cx - r, cy - r, cx + r, cy + r)
        if box is None:
            return
        xs, ys = self._centres(box)
        side = 2 * r
        gx, gy = cx - r + focus[0] * side, cy - r + focus[1] * side
        t = np.hypot(xs - gx, ys - gy) / max(spread * side, 1e-9)
        colors = _interp_colors(np.broadcast_to(t, (ys.shape[0], xs.shape[1])), stops)
        coverage = np.clip(r + 0.5 - np.hypot(xs - cx, ys - cy), 0, 1)
        self._fill(box, coverage, colors, alpha)

    def radial_rect(
        self,
        left: float,
        top: float,
        right: float,
        bottom: float,
        stops: Stops,
        *,
        centre: tuple[float, float] = (0.5, 0.5),
        radii: tuple[float, float] | None = None,
        radius: Radius = 0.0,
        alpha: float = 1.0,
    ) -> None:
        """A (rounded) box filled with a CSS `radial-gradient(ellipse at x y, ...)`.

        `centre` is the ellipse's centre as fractions of the box; `radii` its radii in
        pixels, by default CSS's `farthest-corner` ellipse (the box's aspect ratio, through
        the farthest corner). `radius` rounds the corners: one radius, or `(rx, ry)`.
        """
        box = self._box(left, top, right, bottom)
        if box is None:
            return
        xs, ys = self._centres(box)
        w, h = right - left, bottom - top
        ex, ey = left + centre[0] * w, top + centre[1] * h
        if radii is None:
            dx = max(abs(left - ex), abs(right - ex))
            dy = max(abs(top - ey), abs(bottom - ey))
            # farthest-corner: the ellipse with the box's aspect ratio through that corner
            rx, ry = dx * math.sqrt(2), dy * math.sqrt(2)
        else:
            rx, ry = radii
        t = np.hypot((xs - ex) / max(rx, 1e-9), (ys - ey) / max(ry, 1e-9))
        colors = _interp_colors(np.broadcast_to(t, (ys.shape[0], xs.shape[1])), stops)
        inside = rounded_box_coverage(xs, ys, left, top, right, bottom, radius)
        self._fill(box, inside, colors, alpha)

    def linear_rect(
        self,
        left: float,
        top: float,
        right: float,
        bottom: float,
        stops: Stops,
        *,
        start: tuple[float, float] = (0.0, 0.0),
        end: tuple[float, float] = (1.0, 1.0),
        radius: Radius = 0.0,
        alpha: float = 1.0,
    ) -> None:
        """A (rounded) box filled with an SVG `linearGradient` in object-bounding-box units."""
        box = self._box(left, top, right, bottom)
        if box is None:
            return
        xs, ys = self._centres(box)
        u = (xs - left) / max(right - left, 1e-9)
        v = (ys - top) / max(bottom - top, 1e-9)
        dx, dy = end[0] - start[0], end[1] - start[1]
        t = ((u - start[0]) * dx + (v - start[1]) * dy) / max(dx * dx + dy * dy, 1e-12)
        colors = _interp_colors(np.broadcast_to(t, (ys.shape[0], xs.shape[1])), stops)
        inside = rounded_box_coverage(xs, ys, left, top, right, bottom, radius)
        self._fill(box, inside, colors, alpha)

    def rounded_outline(
        self,
        left: float,
        top: float,
        right: float,
        bottom: float,
        width: float,
        color: ColorLike,
        *,
        radius: float = 0.0,
        alpha: float = 1.0,
    ) -> None:
        """The outline of a (rounded) box, `width` wide and centred on its edge."""
        half = width / 2
        box = self._box(left - half, top - half, right + half, bottom + half)
        if box is None:
            return
        xs, ys = self._centres(box)
        outer = rounded_box_coverage(
            xs, ys, left - half, top - half, right + half, bottom + half, radius + half
        )
        inner = rounded_box_coverage(
            xs, ys, left + half, top + half, right - half, bottom - half, max(radius - half, 0)
        )
        self._fill(box, outer - inner, rgb(color), alpha)

    def shadow(
        self,
        left: float,
        top: float,
        right: float,
        bottom: float,
        *,
        blur: float,
        offset: tuple[float, float] = (0.0, 0.0),
        spread: float = 0.0,
        radius: float = 0.0,
        color: ColorLike = (0, 0, 0),
        alpha: float = 0.5,
    ) -> None:
        """A CSS `box-shadow` under a (rounded) box: offset, blurred and spread like CSS's.

        `blur` is CSS's blur radius (a Gaussian of standard deviation `blur / 2`).
        """
        sigma = max(blur / 2, 1e-6)
        x0, y0 = left + offset[0] - spread, top + offset[1] - spread
        x1, y1 = right + offset[0] + spread, bottom + offset[1] + spread
        if x1 <= x0 or y1 <= y0:
            return
        pad = 3 * sigma + 1
        box = self._box(x0 - pad, y0 - pad, x1 + pad, y1 + pad, pad=0.0)
        if box is None:
            return
        xs, ys = self._centres(box)
        sharp = rounded_box_coverage(xs, ys, x0, y0, x1, y1, max(radius + spread, 0.0))
        soft = gaussian_blur(sharp, sigma)
        self._fill(box, soft, rgb(color), alpha)

    def texture(
        self,
        left: int,
        top: int,
        values: NDArray[np.float64],
        color: ColorLike,
        *,
        alpha: float = 1.0,
        mask: NDArray[np.float64] | None = None,
    ) -> None:
        """Overlay `color` with per-pixel opacity `values * alpha` (an `(h, w)` array)."""
        h, w = values.shape
        box = (left, top, left + w, top + h)
        coverage = values if mask is None else values * mask
        self._fill(box, coverage, rgb(color), alpha)

    def sprite(self, image: RGBA, x: float, y: float, *, anchor: str = "center") -> None:
        """Draw an RGBA image at `(x, y)`: its centre (`anchor="center"`) or top-left."""
        h, w = image.shape[:2]
        if anchor == "center":
            x0, y0 = round(x - w / 2), round(y - h / 2)
        else:
            x0, y0 = round(x), round(y)
        self.patch(image, x0, y0)

    def adjust(
        self,
        left: float,
        top: float,
        right: float,
        bottom: float,
        *,
        brightness: float = 1.0,
        saturate: float = 1.0,
    ) -> None:
        """Apply CSS `brightness()` and `saturate()` filters to a region already drawn."""
        box = self._box(left, top, right, bottom, pad=0.0)
        if box is None:
            return
        x0, y0, x1, y1 = box
        view = self.canvas.pixels[y0:y1, x0:x1].astype(np.float64)
        if saturate != 1.0:  # the CSS saturate() matrix
            s = saturate
            matrix = np.array(
                [
                    [0.213 + 0.787 * s, 0.715 - 0.715 * s, 0.072 - 0.072 * s],
                    [0.213 - 0.213 * s, 0.715 + 0.285 * s, 0.072 - 0.072 * s],
                    [0.213 - 0.213 * s, 0.715 - 0.715 * s, 0.072 + 0.928 * s],
                ]
            )
            view = view @ matrix.T
        view = view * brightness
        self.canvas.pixels[y0:y1, x0:x1] = np.clip(np.round(view), 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Text along a path, and masks
    # ------------------------------------------------------------------

    def mask(
        self,
        coverage: NDArray[np.float32] | NDArray[np.float64],
        x: float,
        y: float,
        color: ColorLike,
        *,
        alpha: float = 1.0,
    ) -> None:
        """Composite a coverage mask (rows top to bottom) with its top-left at `(x, y)`."""
        h, w = coverage.shape
        x0, y0 = round(x), round(y)
        self._fill((x0, y0, x0 + w, y0 + h), coverage.astype(np.float64), rgb(color), alpha)

    def text_on_curve(
        self,
        points: NDArray[np.float64],
        text: str,
        color: ColorLike,
        *,
        size: float,
        bold: bool = False,
        serif: bool = False,
        letter_spacing: float = 0.0,
        alpha: float = 1.0,
    ) -> None:
        """Text centred along a polyline (an SVG `textPath` at 50%, `text-anchor: middle`).

        Each glyph sits upright on the curve's tangent at its centre, its baseline on the
        curve, as a browser lays out text on a path.
        """
        typeface = face(size, bold, serif)
        placed, _ = typeface.layout(text)
        if not placed:
            return
        advances = [glyph.advance + letter_spacing for glyph, _ in placed]
        total = sum(advances) - letter_spacing
        seg = np.hypot(*np.diff(points, axis=0).T)
        along = np.concatenate([[0.0], np.cumsum(seg)])
        start = along[-1] / 2 - total / 2
        pen = start
        for (glyph, _), advance in zip(placed, advances):
            centre = pen + glyph.advance / 2
            pen += advance
            mask = glyph.mask
            if mask.size == 0:
                continue
            px = float(np.interp(centre, along, points[:, 0]))
            py = float(np.interp(centre, along, points[:, 1]))
            k = int(np.clip(np.searchsorted(along, centre) - 1, 0, len(seg) - 1))
            angle = math.atan2(points[k + 1, 1] - points[k, 1], points[k + 1, 0] - points[k, 0])
            # The glyph's pen-relative box: left edge glyph.left, top row glyph.top (baseline 0).
            gx = glyph.left - glyph.advance / 2
            rotated, ox, oy = rotate_mask(mask, angle, gx, float(glyph.top))
            self.mask(rotated, px + ox, py + oy, color, alpha=alpha)


Radius = float | tuple[float, float] | Sequence[tuple[float, float]]
"""Corner rounding: one radius, `(rx, ry)` for every corner, or four `(rx, ry)` pairs
(top-left, top-right, bottom-right, bottom-left, as CSS `border-radius` lists them)."""


def _corners(radius: Radius) -> list[tuple[float, float]]:
    if isinstance(radius, (int, float)):
        return [(float(radius), float(radius))] * 4
    pairs = list(radius)
    if len(pairs) == 2 and all(isinstance(v, (int, float)) for v in pairs):
        rx, ry = (float(v) for v in pairs)  # type: ignore[arg-type]
        return [(rx, ry)] * 4
    if len(pairs) != 4:
        raise ValueError(f"radius needs one value, (rx, ry) or four (rx, ry) pairs: {radius!r}.")
    return [(float(p[0]), float(p[1])) for p in pairs]  # type: ignore[index]


def rounded_box_coverage(
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    left: float,
    top: float,
    right: float,
    bottom: float,
    radius: Radius,
) -> NDArray[np.float64]:
    """Anti-aliased coverage, at pixel centres `xs` (a row) and `ys` (a column), of a box
    whose corners are rounded by ellipses (see `Radius`)."""
    shape = (ys.shape[0], xs.shape[1])
    if right <= left or bottom <= top:
        return np.zeros(shape)
    half_x, half_y = (right - left) / 2, (bottom - top) / 2
    # Signed distance to the box's edges (positive outside), then each rounded corner.
    signed = np.maximum(np.maximum(left - xs, xs - right), np.maximum(top - ys, ys - bottom))
    signed = np.broadcast_to(signed, shape).copy()
    corners = _corners(radius)
    for (rx, ry), (sx, sy) in zip(corners, ((-1, -1), (1, -1), (1, 1), (-1, 1))):
        rx, ry = min(max(rx, 0.0), half_x), min(max(ry, 0.0), half_y)
        if rx <= 0 or ry <= 0:
            continue
        cx = left + rx if sx < 0 else right - rx
        cy = top + ry if sy < 0 else bottom - ry
        dx = np.broadcast_to((xs - cx) * sx, shape)  # > 0: beyond the corner's centre
        dy = np.broadcast_to((ys - cy) * sy, shape)
        zone = (dx > 0) & (dy > 0)
        if not zone.any():
            continue
        ex, ey = dx[zone] / rx, dy[zone] / ry
        g = np.hypot(ex, ey)
        grad = np.hypot(ex / rx, ey / ry) / np.maximum(g, 1e-12)
        signed[zone] = (g - 1.0) / np.maximum(grad, 1e-12)  # first-order distance to the ellipse
    return np.asarray(np.clip(0.5 - signed, 0, 1), dtype=np.float64)


def gaussian_blur(values: NDArray[np.float64], sigma: float) -> NDArray[np.float64]:
    """A separable Gaussian blur of a 2-D array (outside it counts as zero), by FFT."""
    radius = max(math.ceil(3 * sigma), 1)
    taps = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    taps /= taps.sum()
    out = values
    for axis in (0, 1):
        n = out.shape[axis]
        size = 1 << math.ceil(math.log2(n + 2 * radius + 1))
        spectrum = np.fft.rfft(out, n=size, axis=axis)
        kernel = np.fft.rfft(taps, n=size)
        shape = [1, 1]
        shape[axis] = kernel.shape[0]
        full = np.fft.irfft(spectrum * kernel.reshape(shape), n=size, axis=axis)
        out = np.take(full, np.arange(radius, radius + n), axis=axis)
    return np.asarray(np.clip(out, 0.0, None), dtype=np.float64)


def rotate_mask(
    mask: NDArray[np.float32], angle: float, left: float, top: float
) -> tuple[NDArray[np.float64], float, float]:
    """A coverage mask rotated by `angle` (radians, clockwise on screen) about the origin.

    The mask's top-left pixel sits at `(left, top)` relative to the origin. Returns the
    rotated mask and its top-left corner relative to the origin (bilinear sampling).
    """
    h, w = mask.shape
    c, s = math.cos(angle), math.sin(angle)
    corners = np.array([[left, top], [left + w, top], [left, top + h], [left + w, top + h]])
    rx = corners[:, 0] * c - corners[:, 1] * s
    ry = corners[:, 0] * s + corners[:, 1] * c
    x0, x1 = math.floor(np.min(rx)), math.ceil(np.max(rx))
    y0, y1 = math.floor(np.min(ry)), math.ceil(np.max(ry))
    xs = np.arange(x0, x1, dtype=np.float64)[None, :] + 0.5
    ys = np.arange(y0, y1, dtype=np.float64)[:, None] + 0.5
    # Inverse rotation back into the mask's frame (pixel centres at +0.5).
    sx = xs * c + ys * s - left - 0.5
    sy = -xs * s + ys * c - top - 0.5
    ix, iy = np.floor(sx), np.floor(sy)
    fx, fy = sx - ix, sy - iy
    padded = np.pad(mask.astype(np.float64), 1)
    ix = np.clip(ix.astype(np.intp) + 1, 0, w)
    iy = np.clip(iy.astype(np.intp) + 1, 0, h)
    out = (
        padded[iy, ix] * (1 - fx) * (1 - fy)
        + padded[iy, ix + 1] * fx * (1 - fy)
        + padded[iy + 1, ix] * (1 - fx) * fy
        + padded[iy + 1, ix + 1] * fx * fy
    )
    outside = (sx < -1) | (sx > w) | (sy < -1) | (sy > h)
    out[outside] = 0.0
    return out, float(x0), float(y0)


# ----------------------------------------------------------------------
# Noise (wood grain)
# ----------------------------------------------------------------------


@functools.lru_cache(maxsize=16)
def fractal_noise(
    width: int,
    height: int,
    frequency: tuple[float, float],
    *,
    octaves: int = 2,
    seed: int = 7,
) -> NDArray[np.float64]:
    """Smooth value noise in [0, 1], shape `(height, width)`, like SVG's `fractalNoise`.

    `frequency` is cycles per pixel along x and y (SVG `baseFrequency`); each further
    octave doubles it at half the amplitude. Deterministic for a given `seed`; cached and
    read-only.
    """
    rng = np.random.default_rng(seed)
    total = np.zeros((height, width))
    weight = 0.0
    amplitude = 1.0
    fx, fy = frequency
    for _ in range(max(octaves, 1)):
        cells_x = max(math.ceil(width * fx) + 2, 2)
        cells_y = max(math.ceil(height * fy) + 2, 2)
        lattice = rng.random((cells_y, cells_x))
        gx = np.arange(width) * fx
        gy = np.arange(height) * fy
        ix, iy = np.floor(gx).astype(np.intp), np.floor(gy).astype(np.intp)
        tx, ty = gx - ix, gy - iy
        sx = tx * tx * (3 - 2 * tx)  # smoothstep between lattice points
        sy = ty * ty * (3 - 2 * ty)
        top = lattice[iy][:, ix] * (1 - sx) + lattice[iy][:, ix + 1] * sx
        bottom = lattice[iy + 1][:, ix] * (1 - sx) + lattice[iy + 1][:, ix + 1] * sx
        total += amplitude * (top * (1 - sy[:, None]) + bottom * sy[:, None])
        weight += amplitude
        amplitude /= 2
        fx, fy = fx * 2, fy * 2
    noise = total / weight
    noise.flags.writeable = False
    return noise


# ----------------------------------------------------------------------
# Sprites
# ----------------------------------------------------------------------


def resample(image: RGBA, height: int, width: int) -> RGBA:
    """An RGBA image resampled to `(height, width)`: area-averaged down, linear up."""
    from sekai.rendering.brand import _resample

    return _resample(image, max(height, 1), max(width, 1))


__all__ = [
    "Painter",
    "RGBA",
    "Stops",
    "fractal_noise",
    "gaussian_blur",
    "resample",
    "rotate_mask",
    "rounded_box_coverage",
]
