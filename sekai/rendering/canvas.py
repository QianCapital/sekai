from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

import numpy as np
from numpy.typing import ArrayLike, NDArray

Color = tuple[int, int, int]


class Canvas:
    """A tiny dependency-free RGB raster for drawing environment frames.

    Coordinates are in pixels with the origin at the bottom-left and y
    pointing up, which suits physics drawings. Shapes are filled by testing
    pixel centres (no anti-aliasing). to_array() returns an (H, W, 3) uint8
    array, the RGB frame format used throughout sekai.rendering.
    """

    def __init__(self, width: int, height: int, background: Color = (255, 255, 255)) -> None:
        if width < 1 or height < 1:
            raise ValueError("Canvas width and height must be >= 1.")
        self.width = width
        self.height = height
        self.pixels: NDArray[np.uint8] = np.empty((height, width, 3), dtype=np.uint8)
        self.clear(background)

    def clear(self, color: Color = (255, 255, 255)) -> None:
        self.pixels[:] = color

    def _region(
        self, x_min: float, y_min: float, x_max: float, y_max: float
    ) -> tuple[int, int, int, int] | None:
        if not np.all(np.isfinite([x_min, y_min, x_max, y_max])):
            return None  # e.g. a diverged (NaN) state: draw nothing rather than crash
        x0 = max(int(np.floor(x_min)), 0)
        y0 = max(int(np.floor(y_min)), 0)
        x1 = min(int(np.ceil(x_max)) + 1, self.width)
        y1 = min(int(np.ceil(y_max)) + 1, self.height)
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

    def _grid(
        self, region: tuple[int, int, int, int]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        x0, y0, x1, y1 = region
        return np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)

    def polygon(self, points: ArrayLike, color: Color) -> None:
        """Fill a simple polygon given as a sequence of (x, y) vertices."""
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 3:
            return
        region = self._region(
            np.min(pts[:, 0]), np.min(pts[:, 1]), np.max(pts[:, 0]), np.max(pts[:, 1])
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
        self._paint(region, inside, color)

    def rect(self, left: float, bottom: float, right: float, top: float, color: Color) -> None:
        self.polygon([(left, bottom), (left, top), (right, top), (right, bottom)], color)

    def circle(self, cx: float, cy: float, radius: float, color: Color) -> None:
        region = self._region(cx - radius, cy - radius, cx + radius, cy + radius)
        if region is None:
            return
        X, Y = self._grid(region)
        self._paint(region, (X - cx) ** 2 + (Y - cy) ** 2 <= radius**2, color)

    def line(
        self,
        start: Sequence[float],
        end: Sequence[float],
        color: Color,
        width: float = 1.0,
    ) -> None:
        """Draw a straight segment of the given width (in pixels)."""
        p0 = np.asarray(start, dtype=np.float64)
        p1 = np.asarray(end, dtype=np.float64)
        d = p1 - p0
        length = float(np.hypot(*d))
        if length == 0:
            self.circle(p0[0], p0[1], width / 2, color)
            return
        n = np.array([-d[1], d[0]]) / length * max(width, 1.0) / 2
        self.polygon([p0 + n, p1 + n, p1 - n, p0 - n], color)

    def polyline(self, points: ArrayLike, color: Color, width: float = 1.0) -> None:
        for a, b in pairwise(np.asarray(points, dtype=np.float64)):
            self.line(a, b, color, width)

    def to_array(self) -> NDArray[np.uint8]:
        frame: NDArray[np.uint8] = self.pixels.copy()
        return frame


def rotate(
    points: ArrayLike, angle: float, origin: Sequence[float] = (0.0, 0.0)
) -> NDArray[np.float64]:
    """Rotate (x, y) points counter-clockwise by angle (radians) about origin."""
    pts = np.asarray(points, dtype=np.float64)
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    out: NDArray[np.float64] = pts @ rot.T + np.asarray(origin, dtype=np.float64)
    return out
