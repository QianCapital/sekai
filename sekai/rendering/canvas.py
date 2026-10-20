from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sekai.rendering.text import face

Color = tuple[int, int, int]
HAlign = Literal["left", "center", "right"]
VAlign = Literal["baseline", "middle", "top", "bottom"]

# Anti-aliased strokes are drawn piece by piece, each piece at most this long (pixels), so
# the pixels tested around a piece stay few however long or diagonal its segment is.
_PIECE = 8.0


class Canvas:
    """A tiny dependency-free RGB raster for drawing environment frames.

    Coordinates are in pixels with the origin at the bottom-left and y
    pointing up, which suits physics drawings. to_array() returns an
    (H, W, 3) uint8 array, the RGB frame format used throughout sekai.rendering.

    By default shapes are filled by testing pixel centres (crisp, no
    anti-aliasing). With `antialias=True` every shape is drawn with exact
    coverage at its edges: smooth lines and curves, rectangles that land on
    fractional pixels, round line caps and joins. Text is always anti-aliased.
    Every shape takes `alpha` (opacity in [0, 1]), and `clip()` restricts
    drawing to a rectangle.

    Args:
        width: Width in pixels.
        height: Height in pixels.
        background: Initial fill colour.
        antialias: Draw shapes with anti-aliased edges.
    """

    def __init__(
        self,
        width: int,
        height: int,
        background: Color = (255, 255, 255),
        *,
        antialias: bool = False,
    ) -> None:
        if width < 1 or height < 1:
            raise ValueError("Canvas width and height must be >= 1.")
        self.width = width
        self.height = height
        self.antialias = antialias
        self.pixels: NDArray[np.uint8] = np.empty((height, width, 3), dtype=np.uint8)
        self._clip = (0, 0, width, height)
        self.clear(background)

    def clear(self, color: Color = (255, 255, 255)) -> None:
        """Fill the whole canvas (ignoring any clip)."""
        self.pixels[:] = color

    @contextmanager
    def clip(self, left: float, bottom: float, right: float, top: float) -> Iterator[None]:
        """Restrict drawing to a rectangle while the `with` block runs.

        Pixels partly inside the rectangle are drawable; nested clips intersect.

        Example:
            ```python
            with canvas.clip(40, 20, 300, 180):
                canvas.polyline(points, (0, 0, 0))  # nothing lands outside the box
            ```
        """
        if not np.all(np.isfinite([left, bottom, right, top])):
            raise ValueError("clip() needs finite bounds.")
        previous = self._clip
        x0, y0, x1, y1 = previous
        self._clip = (
            max(x0, int(np.floor(left))),
            max(y0, int(np.floor(bottom))),
            min(x1, int(np.ceil(right))),
            min(y1, int(np.ceil(top))),
        )
        try:
            yield
        finally:
            self._clip = previous

    # ------------------------------------------------------------------
    # Pixel plumbing
    # ------------------------------------------------------------------

    def _region(
        self, x_min: float, y_min: float, x_max: float, y_max: float, pad: float = 0.0
    ) -> tuple[int, int, int, int] | None:
        if not np.all(np.isfinite([x_min, y_min, x_max, y_max])):
            return None  # e.g. a diverged (NaN) state: draw nothing rather than crash
        cx0, cy0, cx1, cy1 = self._clip
        x0 = max(int(np.floor(x_min - pad)), cx0)
        y0 = max(int(np.floor(y_min - pad)), cy0)
        x1 = min(int(np.ceil(x_max + pad)) + 1, cx1)
        y1 = min(int(np.ceil(y_max + pad)) + 1, cy1)
        if x0 >= x1 or y0 >= y1:
            return None
        return x0, y0, x1, y1

    def _paint(
        self, region: tuple[int, int, int, int], mask: NDArray[np.bool_], color: Color
    ) -> None:
        x0, y0, x1, y1 = region
        # mask rows run bottom-to-top; image rows run top-to-bottom.
        view = self.pixels[self.height - y1 : self.height - y0, x0:x1]
        view[mask[::-1]] = color

    def _blend(
        self,
        region: tuple[int, int, int, int],
        coverage: NDArray[np.floating] | NDArray[np.bool_],
        color: Color,
        alpha: float,
    ) -> None:
        """Composite `color` over the region, weighted by coverage (rows bottom-to-top)."""
        if coverage.dtype == np.bool_ and alpha >= 1.0:
            self._paint(region, coverage.astype(np.bool_, copy=False), color)
            return
        x0, y0, x1, y1 = region
        view = self.pixels[self.height - y1 : self.height - y0, x0:x1]
        self._composite(view, coverage[::-1], color, alpha)

    @staticmethod
    def _composite(
        view: NDArray[np.uint8],
        coverage: NDArray[np.floating] | NDArray[np.bool_],
        color: Color,
        alpha: float,
    ) -> None:
        a = coverage.astype(np.float32) * np.float32(min(max(alpha, 0.0), 1.0))
        a = a[..., None]
        ink = np.asarray(color, dtype=np.float32)
        view[...] = (view * (1.0 - a) + ink * a + 0.5).astype(np.uint8)

    def _grid(
        self, region: tuple[int, int, int, int]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        x0, y0, x1, y1 = region
        return np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)

    # ------------------------------------------------------------------
    # Shapes
    # ------------------------------------------------------------------

    def polygon(self, points: ArrayLike, color: Color, *, alpha: float = 1.0) -> None:
        """Fill a simple polygon given as a sequence of (x, y) vertices."""
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 3:
            return
        region = self._region(
            np.min(pts[:, 0]),
            np.min(pts[:, 1]),
            np.max(pts[:, 0]),
            np.max(pts[:, 1]),
            pad=1.0 if self.antialias else 0.0,
        )
        if region is None:
            return
        X, Y = self._grid(region)
        inside = np.zeros(X.shape, dtype=np.bool_)
        xj, yj = pts[-1]
        with np.errstate(divide="ignore", invalid="ignore"):
            for xi, yi in pts:
                crosses = (yi > Y) != (yj > Y)
                x_at_y = (xj - xi) * (Y - yi) / (yj - yi) + xi
                inside ^= crosses & (x_at_y > X)
                xj, yj = xi, yi
        if not self.antialias:
            self._blend(region, inside, color, alpha)
            return
        # Coverage from the signed distance to the nearest edge.
        distance = np.full(X.shape, np.inf)
        for (ax, ay), (bx, by) in zip(pts, np.roll(pts, -1, axis=0)):
            distance = np.minimum(distance, _segment_distance(X, Y, ax, ay, bx, by))
        signed = np.where(inside, distance, -distance)
        self._blend(region, np.clip(0.5 + signed, 0.0, 1.0), color, alpha)

    def rect(
        self,
        left: float,
        bottom: float,
        right: float,
        top: float,
        color: Color,
        *,
        alpha: float = 1.0,
        radius: float = 0.0,
    ) -> None:
        """Fill an axis-aligned rectangle, optionally with corners rounded by `radius`."""
        if radius <= 0 and not self.antialias:
            self.polygon(
                [(left, bottom), (left, top), (right, top), (right, bottom)], color, alpha=alpha
            )
            return
        left, right = min(left, right), max(left, right)
        bottom, top = min(bottom, top), max(bottom, top)
        region = self._region(left, bottom, right, top)
        if region is None:
            return
        x0, y0, x1, y1 = region
        xs = np.arange(x0, x1) + 0.5
        ys = np.arange(y0, y1) + 0.5
        if radius <= 0:  # exact area of each pixel inside the rectangle
            cover_x = np.clip(np.minimum(xs + 0.5, right) - np.maximum(xs - 0.5, left), 0, 1)
            cover_y = np.clip(np.minimum(ys + 0.5, top) - np.maximum(ys - 0.5, bottom), 0, 1)
            self._blend(region, cover_y[:, None] * cover_x[None, :], color, alpha)
            return
        half_x, half_y = (right - left) / 2, (top - bottom) / 2
        r = min(radius, half_x, half_y)
        qx = np.abs(xs - (left + half_x))[None, :] - (half_x - r)
        qy = np.abs(ys - (bottom + half_y))[:, None] - (half_y - r)
        outside = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0))
        signed = outside + np.minimum(np.maximum(qx, qy), 0) - r
        coverage = np.clip(0.5 - signed, 0, 1) if self.antialias else signed <= 0
        self._blend(region, coverage, color, alpha)

    def circle(
        self, cx: float, cy: float, radius: float, color: Color, *, alpha: float = 1.0
    ) -> None:
        """Fill a disc."""
        region = self._region(
            cx - radius, cy - radius, cx + radius, cy + radius, pad=1.0 if self.antialias else 0
        )
        if region is None:
            return
        X, Y = self._grid(region)
        squared = (X - cx) ** 2 + (Y - cy) ** 2
        if self.antialias:
            self._blend(region, np.clip(radius + 0.5 - np.sqrt(squared), 0, 1), color, alpha)
        else:
            self._blend(region, squared <= radius**2, color, alpha)

    def line(
        self,
        start: Sequence[float],
        end: Sequence[float],
        color: Color,
        width: float = 1.0,
        *,
        alpha: float = 1.0,
        dash: tuple[float, float] | None = None,
    ) -> None:
        """Draw a straight segment of the given width (in pixels).

        `dash=(on, off)` draws it dashed, `on` pixels drawn then `off` skipped.
        """
        if self.antialias or dash is not None:
            self.polyline([start, end], color, width, alpha=alpha, dash=dash)
            return
        self._segment(
            np.asarray(start, np.float64), np.asarray(end, np.float64), color, width, alpha
        )

    def _segment(
        self,
        p0: NDArray[np.float64],
        p1: NDArray[np.float64],
        color: Color,
        width: float,
        alpha: float,
    ) -> None:
        d = p1 - p0
        length = float(np.hypot(*d))
        if length == 0:
            self.circle(p0[0], p0[1], width / 2, color, alpha=alpha)
            return
        n = np.array([-d[1], d[0]]) / length * max(width, 1.0) / 2
        self.polygon([p0 + n, p1 + n, p1 - n, p0 - n], color, alpha=alpha)

    def polyline(
        self,
        points: ArrayLike,
        color: Color,
        width: float = 1.0,
        *,
        alpha: float = 1.0,
        dash: tuple[float, float] | None = None,
    ) -> None:
        """Draw connected segments through `points` ((x, y) rows).

        A non-finite point breaks the line, so NaN marks a gap in a series.
        Anti-aliased polylines have round joins and caps, and overlapping
        segments are drawn once (a translucent line stays even). `dash=(on, off)`
        draws it dashed.
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if len(pts) < 2:
            return
        if dash is not None:
            starts, ends = _dashes(pts, dash)
        else:
            starts, ends = pts[:-1], pts[1:]
            finite = np.isfinite(starts).all(axis=1) & np.isfinite(ends).all(axis=1)
            starts, ends = starts[finite], ends[finite]
        if not self.antialias:
            for a, b in zip(starts, ends):
                self._segment(a, b, color, width, alpha)
            return
        self._stroke(starts, ends, width, color, alpha)

    def _stroke(
        self,
        starts: NDArray[np.float64],
        ends: NDArray[np.float64],
        width: float,
        color: Color,
        alpha: float,
    ) -> None:
        """Anti-aliased union of capsules around segments, composited once."""
        if len(starts) == 0:
            return
        if width < 1.0:  # thinner than a pixel: a pixel-wide stroke, fainter
            alpha, width = alpha * max(width, 0.0), 1.0
        half = width / 2
        # Split each segment into short pieces (their capsules' union is the segment's).
        d = ends - starts
        counts = np.maximum(np.ceil(np.hypot(d[:, 0], d[:, 1]) / _PIECE), 1).astype(np.intp)
        seg = np.repeat(np.arange(len(starts)), counts)
        k = np.arange(len(seg)) - np.repeat(np.cumsum(counts) - counts, counts)
        t0 = (k / counts[seg])[:, None]
        t1 = ((k + 1) / counts[seg])[:, None]
        a = starts[seg] + d[seg] * t0
        b = starts[seg] + d[seg] * t1

        cx0, cy0, cx1, cy1 = self._clip
        pad = half + 1.0
        bx0 = np.maximum(np.floor(np.minimum(a[:, 0], b[:, 0]) - pad), cx0).astype(np.intp)
        by0 = np.maximum(np.floor(np.minimum(a[:, 1], b[:, 1]) - pad), cy0).astype(np.intp)
        bx1 = np.minimum(np.ceil(np.maximum(a[:, 0], b[:, 0]) + pad), cx1).astype(np.intp)
        by1 = np.minimum(np.ceil(np.maximum(a[:, 1], b[:, 1]) + pad), cy1).astype(np.intp)
        w = np.maximum(bx1 - bx0, 0)
        h = np.maximum(by1 - by0, 0)
        live = (w > 0) & (h > 0)
        if not live.any():
            return
        a, b, bx0, by0, bx1, by1, w, h = (v[live] for v in (a, b, bx0, by0, bx1, by1, w, h))
        region = (int(bx0.min()), int(by0.min()), int(bx1.max()), int(by1.max()))
        rw, rh = region[2] - region[0], region[3] - region[1]

        n = w * h
        piece = np.repeat(np.arange(len(n)), n)
        local = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
        px = bx0[piece] + local % w[piece]
        py = by0[piece] + local // w[piece]
        dist = _segment_distance(
            px + 0.5, py + 0.5, a[piece, 0], a[piece, 1], b[piece, 0], b[piece, 1]
        )
        cover = np.clip(half + 0.5 - dist, 0.0, 1.0)
        hit = cover > 0
        coverage = np.zeros(rw * rh)
        index = (py[hit] - region[1]) * rw + (px[hit] - region[0])
        np.maximum.at(coverage, index, cover[hit])
        self._blend(region, coverage.reshape(rh, rw), color, alpha)

    def fill_between(
        self,
        x: ArrayLike,
        y1: ArrayLike,
        y2: ArrayLike,
        color: Color,
        *,
        alpha: float = 1.0,
    ) -> None:
        """Fill the area between two curves sampled at increasing `x`.

        The curves are linear between samples; a non-finite y leaves a gap. For a
        step (piecewise-constant) curve, repeat each x with the old and new value.
        """
        xs = np.asarray(x, dtype=np.float64).ravel()
        lo = np.broadcast_to(np.asarray(y1, dtype=np.float64), xs.shape)
        hi = np.broadcast_to(np.asarray(y2, dtype=np.float64), xs.shape)
        keep = np.isfinite(xs)
        xs, lo, hi = xs[keep], lo[keep], hi[keep]
        if len(xs) < 2:
            return
        order = np.argsort(xs, kind="stable")
        xs, lo, hi = xs[order], lo[order], hi[order]
        both = np.concatenate([lo, hi])
        if not np.isfinite(both).any():
            return
        region = self._region(xs[0], np.nanmin(both), xs[-1], np.nanmax(both))
        if region is None:
            return
        x0, y0, x1, y1 = region
        centres = np.arange(x0, x1) + 0.5
        with np.errstate(invalid="ignore"):
            a = np.interp(centres, xs, lo)
            b = np.interp(centres, xs, hi)
            bottom, top = np.fmin(a, b), np.fmax(a, b)
            bottom[np.isnan(a) | np.isnan(b)] = np.nan
            rows = np.arange(y0, y1, dtype=np.float64)[:, None]
            coverage = np.minimum(top, rows + 1) - np.maximum(bottom, rows)
        coverage = np.nan_to_num(np.clip(coverage, 0.0, 1.0), nan=0.0)
        coverage[:, (centres < xs[0]) | (centres > xs[-1])] = 0.0
        self._blend(region, coverage if self.antialias else coverage >= 0.5, color, alpha)

    def image(
        self,
        image: ArrayLike,
        left: float,
        bottom: float,
        right: float,
        top: float,
        *,
        alpha: float = 1.0,
    ) -> None:
        """Draw an RGB `(h, w, 3)` or RGBA `(h, w, 4)` uint8 image scaled into a box.

        Each source pixel becomes a crisp block (nearest-neighbour scaling): this
        draws heatmaps and matrices of cells, with an image's first row at the top.
        """
        pixels = np.asarray(image)
        if pixels.ndim != 3 or pixels.shape[2] not in (3, 4) or 0 in pixels.shape[:2]:
            raise ValueError(f"image() needs an (h, w, 3) or (h, w, 4) array, got {pixels.shape}.")
        left, right = min(left, right), max(left, right)
        bottom, top = min(bottom, top), max(bottom, top)
        region = self._region(left, bottom, right, top)
        if region is None or right <= left or top <= bottom:
            return
        x0, y0, x1, y1 = region
        h, w = pixels.shape[:2]
        cx = np.arange(x0, x1) + 0.5
        cy = np.arange(y1 - 1, y0 - 1, -1) + 0.5  # image rows, top first
        cols = np.clip(((cx - left) / (right - left) * w).astype(np.intp), 0, w - 1)
        rows = np.clip(((top - cy) / (top - bottom) * h).astype(np.intp), 0, h - 1)
        inside = ((cx >= left) & (cx < right))[None, :] & ((cy >= bottom) & (cy < top))[:, None]
        block = pixels[rows][:, cols].astype(np.float32)
        a = inside.astype(np.float32) * np.float32(min(max(alpha, 0.0), 1.0))
        if pixels.shape[2] == 4:
            a = a * block[..., 3] / 255.0
        view = self.pixels[self.height - y1 : self.height - y0, x0:x1]
        a3 = a[..., None]
        view[...] = (view * (1.0 - a3) + block[..., :3] * a3 + 0.5).astype(np.uint8)

    # ------------------------------------------------------------------
    # Text
    # ------------------------------------------------------------------

    def text(
        self,
        x: float,
        y: float,
        text: str,
        color: Color,
        *,
        size: float = 12,
        bold: bool = False,
        align: HAlign = "left",
        valign: VAlign = "baseline",
        alpha: float = 1.0,
    ) -> float:
        """Draw a line of text and return its width in pixels.

        `align` places `x` at the text's left edge, centre or right edge.
        `valign` places `y` at the baseline, the middle of the capital letters,
        their top, or the bottom of the descenders. The typeface is BDO Grotesk
        at sizes 9-32 px (other sizes snap to the nearest; larger ones are
        scaled), regular or bold, with tabular digits; see `sekai.rendering.text`.
        """
        typeface = face(size, bold)
        placed, width = typeface.layout(text)
        if align == "center":
            x -= width / 2
        elif align == "right":
            x -= width
        if valign == "middle":
            y -= typeface.cap_height / 2
        elif valign == "top":
            y -= typeface.cap_height
        elif valign == "bottom":
            y += typeface.descent
        if not np.isfinite([x, y]).all():
            return width
        baseline = round(self.height - y)  # image row of the baseline
        cx0, cy0, cx1, cy1 = self._clip
        row_lo, row_hi = self.height - cy1, self.height - cy0
        for glyph, pen in placed:
            mask = glyph.mask
            if mask.size == 0:
                continue
            gx = round(x + pen) + glyph.left
            gy = baseline + glyph.top
            ix0, ix1 = max(gx, cx0), min(gx + mask.shape[1], cx1)
            iy0, iy1 = max(gy, row_lo), min(gy + mask.shape[0], row_hi)
            if ix0 >= ix1 or iy0 >= iy1:
                continue
            self._composite(
                self.pixels[iy0:iy1, ix0:ix1],
                mask[iy0 - gy : iy1 - gy, ix0 - gx : ix1 - gx],
                color,
                alpha,
            )
        return width

    def to_array(self) -> NDArray[np.uint8]:
        frame: NDArray[np.uint8] = self.pixels.copy()
        return frame


def _segment_distance(
    px: NDArray[np.float64],
    py: NDArray[np.float64],
    ax: NDArray[np.float64] | float,
    ay: NDArray[np.float64] | float,
    bx: NDArray[np.float64] | float,
    by: NDArray[np.float64] | float,
) -> NDArray[np.float64]:
    """Distance from points (px, py) to segments a-b (broadcasting)."""
    dx, dy = np.subtract(bx, ax), np.subtract(by, ay)
    length2 = dx * dx + dy * dy
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(length2 > 0, ((px - ax) * dx + (py - ay) * dy) / length2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    distance: NDArray[np.float64] = np.hypot(px - (ax + t * dx), py - (ay + t * dy))
    return distance


def _dashes(
    points: NDArray[np.float64], dash: tuple[float, float]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The drawn pieces of a dashed polyline, as (starts, ends); the pattern restarts at gaps."""
    on, off = (float(v) for v in dash)
    if not (on > 0 and off >= 0):
        raise ValueError(f"dash needs on > 0 and off >= 0, got {dash!r}.")
    finite = np.isfinite(points).all(axis=1)
    starts, ends = [], []
    breaks = np.flatnonzero(np.diff(np.concatenate([[0], finite.astype(np.int8), [0]])))
    for first, stop in zip(breaks[::2], breaks[1::2]):
        run = points[first:stop]
        step = np.hypot(*np.diff(run, axis=0).T)
        run = run[np.concatenate([[True], step > 0])]
        if len(run) < 2:
            continue
        along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(run, axis=0).T))])
        u = np.unique(np.concatenate([np.arange(0.0, along[-1], 0.5), along]))
        pos = np.column_stack([np.interp(u, along, run[:, 0]), np.interp(u, along, run[:, 1])])
        drawn = ((u[:-1] + u[1:]) / 2) % (on + off) < on
        starts.append(pos[:-1][drawn])
        ends.append(pos[1:][drawn])
    if not starts:
        empty = np.zeros((0, 2))
        return empty, empty
    return np.concatenate(starts), np.concatenate(ends)


def rotate(
    points: ArrayLike, angle: float, origin: Sequence[float] = (0.0, 0.0)
) -> NDArray[np.float64]:
    """Rotate (x, y) points counter-clockwise by angle (radians) about origin."""
    pts = np.asarray(points, dtype=np.float64)
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    out: NDArray[np.float64] = pts @ rot.T + np.asarray(origin, dtype=np.float64)
    return out
