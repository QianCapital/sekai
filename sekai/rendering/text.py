"""Text for `Canvas`: a typeface rasterised ahead of time, so drawing text needs only NumPy.

The glyphs are BDO Grotesk (the typeface of sekai's documentation, SIL Open Font License
1.1; see `fonts/OFL.txt`), stored as anti-aliased coverage masks in `fonts/bdo-grotesk.npz`
at pixel sizes 9 to 32 in a regular and a bold (semibold) weight, with advances and
kerning. Other sizes snap to the nearest stored one; sizes above 32 are resampled from
32. Digits are tabular (all the same width), so a number that changes from frame to frame
keeps its width. `tools/build_font_atlas.py` rebuilds the atlas.

The character set is printable ASCII plus the minus sign, middle dot, em and en dashes,
ellipsis, degree, plus-minus and multiplication signs, euro, pound and yen, bullet, the
four arrows, and `≈ ≤ ≥ ∞`.
Other characters are drawn without their accents (é as e) where possible, else as `?`.
"""

from __future__ import annotations

import functools
import unicodedata
from dataclasses import dataclass
from importlib import resources

import numpy as np
from numpy.typing import NDArray

_ATLAS = "fonts/bdo-grotesk.npz"
# The largest stored size; bigger text is resampled from it.
_MAX_STORED = 32


@dataclass(frozen=True)
class Glyph:
    """One rasterised character.

    Attributes:
        mask: `(h, w)` coverage in [0, 1], rows from top to bottom (empty for a space).
        left: Offset of the mask's left column from the pen position, in pixels.
        top: Offset of the mask's top row from the baseline, in pixels (negative is above).
        advance: How far the pen moves after the glyph, in pixels.
    """

    mask: NDArray[np.float32]
    left: int
    top: int
    advance: float


@functools.lru_cache(maxsize=1)
def _atlas() -> dict[str, NDArray[np.generic]]:
    source = resources.files("sekai.rendering").joinpath(_ATLAS)
    with source.open("rb") as f, np.load(f) as archive:
        return {name: archive[name] for name in archive.files}


def font_sizes() -> tuple[int, ...]:
    """The pixel sizes stored in the atlas (text at other sizes snaps to the nearest)."""
    return tuple(int(s) for s in _atlas()["sizes"])


class Face:
    """The typeface at one pixel size and weight: glyphs, metrics, kerning and layout.

    Get one with `face(size, bold)`, which caches them.

    Attributes:
        size: The pixel size (the em height).
        bold: Whether this is the bold weight.
        ascent: Pixels from the baseline to the top of the tallest glyphs.
        descent: Pixels from the baseline to the bottom of the descenders.
        cap_height: Height of capital letters above the baseline.
        x_height: Height of lowercase letters such as x.
    """

    def __init__(self, size: int, bold: bool) -> None:
        atlas = _atlas()
        weight = "bold" if bold else "regular"
        stored = min(font_sizes(), key=lambda s: (abs(s - size), s))
        if size > _MAX_STORED:
            stored = _MAX_STORED
        scale = size / stored if size > _MAX_STORED else 1.0
        key = f"{weight}_{stored}"
        self.size = size if size > _MAX_STORED else stored
        self.bold = bold
        ascent, descent, cap, x = (float(v) * scale for v in atlas[f"{key}_metrics"])
        self.ascent, self.descent, self.cap_height, self.x_height = ascent, descent, cap, x

        charset = [chr(int(c)) for c in atlas["charset"]]
        self._index = {ch: i for i, ch in enumerate(charset)}
        advances = atlas[f"{key}_advance"].astype(np.float64) * scale
        boxes = atlas[f"{key}_box"].astype(np.int64)
        bits = atlas[f"{key}_bits"]
        glyphs: list[Glyph] = []
        offset = 0
        for i in range(len(charset)):
            left, top, width, height = (int(v) for v in boxes[i])
            mask = bits[offset : offset + width * height].reshape(height, width)
            offset += width * height
            coverage = mask.astype(np.float32) / 255.0
            if scale != 1.0 and coverage.size:
                coverage = _resample(coverage, scale)
                left, top = round(left * scale), round(top * scale)
            glyphs.append(Glyph(coverage, left, top, float(advances[i])))
        self._glyphs = glyphs
        # Kerning in whole pixels: glyphs sit on whole pixels (they are hinted), and
        # fractional adjustments would only add uneven rounding between letters.
        units = float(atlas["units_per_em"])
        kern = np.asarray(atlas[f"{weight}_kern"], dtype=np.float64)
        self._kern: NDArray[np.float64] = np.round(kern * (self.size / units))
        self._fallback = self._index["?"]

    def _lookup(self, ch: str) -> int:
        i = self._index.get(ch)
        if i is not None:
            return i
        base = unicodedata.normalize("NFKD", ch)[:1]
        return self._index.get(base, self._fallback)

    def glyph(self, ch: str) -> Glyph:
        """The glyph drawn for a character."""
        return self._glyphs[self._lookup(ch)]

    def layout(self, text: str) -> tuple[list[tuple[Glyph, float]], float]:
        """Glyphs with their pen positions (from the start of the text), and the width."""
        placed: list[tuple[Glyph, float]] = []
        pen, previous = 0.0, -1
        for ch in text:
            i = self._lookup(ch)
            if previous >= 0:
                pen += self._kern[previous, i]
            glyph = self._glyphs[i]
            placed.append((glyph, pen))
            pen += glyph.advance
            previous = i
        return placed, pen

    def width(self, text: str) -> float:
        """The advance width of `text` in pixels."""
        return self.layout(text)[1]


@functools.lru_cache(maxsize=64)
def _face(size: int, bold: bool) -> Face:
    return Face(size, bold)


def face(size: float = 12, bold: bool = False) -> Face:
    """The typeface at a pixel size (rounded, at least 1) and weight, cached."""
    return _face(max(round(size), 1), bool(bold))


def text_width(text: str, size: float = 12, bold: bool = False) -> float:
    """The width in pixels of `text` drawn by `Canvas.text` at this size and weight."""
    return face(size, bold).width(text)


def _resample(mask: NDArray[np.float32], scale: float) -> NDArray[np.float32]:
    """Bilinear resampling of a coverage mask by `scale` (> 1)."""
    h, w = mask.shape
    out_h, out_w = max(round(h * scale), 1), max(round(w * scale), 1)
    ys = np.clip((np.arange(out_h) + 0.5) / scale - 0.5, 0, h - 1)
    xs = np.clip((np.arange(out_w) + 0.5) / scale - 0.5, 0, w - 1)
    y0, x0 = np.floor(ys).astype(np.intp), np.floor(xs).astype(np.intp)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    top = mask[y0][:, x0] * (1 - fx) + mask[y0][:, x1] * fx
    bottom = mask[y1][:, x0] * (1 - fx) + mask[y1][:, x1] * fx
    out: NDArray[np.float32] = (top * (1 - fy) + bottom * fy).astype(np.float32)
    return out


__all__ = ["Face", "Glyph", "face", "font_sizes", "text_width"]
