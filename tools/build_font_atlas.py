"""Build the glyph atlases that `sekai.rendering.Canvas.text()` draws with.

sekai renders text without a font engine at runtime: this script rasterises a fixed
character set of each typeface of sekai's documentation (both SIL Open Font License 1.1)
at a range of pixel sizes and stores the anti-aliased glyph coverage masks, advances and
kerning pairs in a compressed NumPy archive per typeface that ships with the package:

- BDO Grotesk, the text face, in two weights. Digits are taken from the font's tabular
  figures (`tnum`), so numbers that change from frame to frame keep their width.
- Hedvig Letters Serif, the heading face (one weight), for the titles of Qian
  Capital-branded dashboards.

Run it from the repository root after changing the character set, the sizes or the
weights (it needs fontTools, brotli and Pillow, which sekai itself does not):

    pip install fonttools brotli pillow
    python tools/build_font_atlas.py

It reads docs/assets/fonts/BDOGrotesk-VF.woff2 and HedvigLettersSerif.woff2 and writes
sekai/rendering/fonts/bdo-grotesk.npz and hedvig-letters-serif.npz.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from fontTools.varLib import instancer
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "docs" / "assets" / "fonts"
TARGETS = ROOT / "sekai" / "rendering" / "fonts"

# Printable ASCII; minus, middle dot, em and en dash, ellipsis, degree, plus-minus, times,
# euro, pound, yen, bullet, the four arrows, and four maths symbols.
CHARSET = "".join(chr(c) for c in range(32, 127)) + (
    "\u2212\u00b7\u2014\u2013\u2026\u00b0\u00b1\u00d7\u20ac\u00a3\u00a5\u2022"
    "\u2192\u2190\u2191\u2193\u2248\u2264\u2265\u221e"
)
# Kerning pairs smaller than this (font units; 1000 per em) are below a pixel at every size.
MIN_KERN = 15
# Per typeface: the source, the atlas, each weight's variable-font axis values, and the
# pixel sizes (larger text is resampled from the largest).
FONTS = {
    "BDOGrotesk-VF.woff2": (
        "bdo-grotesk.npz",
        {"regular": {"wght": 400}, "bold": {"wght": 600}},
        (9, 10, 11, 12, 13, 14, 16, 18, 20, 24, 28, 32),
    ),
    "HedvigLettersSerif.woff2": (
        "hedvig-letters-serif.npz",
        {"regular": {"opsz": 24}},
        (9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 28, 32),
    ),
}


def static_instance(source: Path, axes: dict[str, float]) -> TTFont:
    """The variable font pinned at `axes`, with digits mapped to tabular figures if it has them."""
    font = TTFont(source)
    font = instancer.instantiateVariableFont(font, axes)
    tabular = _tnum_mapping(font)
    for table in font["cmap"].tables:
        if table.isUnicode():
            for digit in "0123456789":
                glyph = table.cmap.get(ord(digit))
                if glyph in tabular:
                    table.cmap[ord(digit)] = tabular[glyph]
    return font


def _tnum_mapping(font: TTFont) -> dict[str, str]:
    gsub = font["GSUB"].table
    mapping: dict[str, str] = {}
    for record in gsub.FeatureList.FeatureRecord:
        if record.FeatureTag != "tnum":
            continue
        for index in record.Feature.LookupListIndex:
            for sub in gsub.LookupList.Lookup[index].SubTable:
                mapping.update(getattr(sub, "mapping", {}))
    return mapping


def kerning(font: TTFont, glyphs: list[str]) -> dict[tuple[int, int], int]:
    """Horizontal pair kerning (font units) between the glyphs of the character set.

    Applies the font's `kern` lookups in order; within a lookup the first subtable that
    covers a pair decides it, as an OpenType shaper does.
    """
    gpos = font["GPOS"].table
    lookups: list[int] = sorted(
        {
            index
            for record in gpos.FeatureList.FeatureRecord
            if record.FeatureTag == "kern"
            for index in record.Feature.LookupListIndex
        }
    )
    position = {name: i for i, name in enumerate(glyphs)}
    total: dict[tuple[int, int], int] = {}
    for index in lookups:
        decided: set[tuple[int, int]] = set()
        for sub in gpos.LookupList.Lookup[index].SubTable:
            if getattr(sub, "LookupType", None) == 9:  # extension
                sub = sub.ExtSubTable
            if getattr(sub, "LookupType", 2) != 2:
                continue
            for pair, value in _pair_values(sub, glyphs, position):
                if pair in decided:
                    continue
                decided.add(pair)
                if value:
                    total[pair] = total.get(pair, 0) + value
    return {pair: value for pair, value in total.items() if abs(value) >= MIN_KERN}


def _pair_values(sub, glyphs, position):  # type: ignore[no-untyped-def]
    covered = [g for g in sub.Coverage.glyphs if g in position]
    if sub.Format == 1:
        for first in covered:
            pair_set = sub.PairSet[sub.Coverage.glyphs.index(first)]
            for record in pair_set.PairValueRecord:
                if record.SecondGlyph in position:
                    value = getattr(record.Value1, "XAdvance", 0) if record.Value1 else 0
                    yield (position[first], position[record.SecondGlyph]), value or 0
    elif sub.Format == 2:
        class1 = sub.ClassDef1.classDefs if sub.ClassDef1 else {}
        class2 = sub.ClassDef2.classDefs if sub.ClassDef2 else {}
        for first in covered:
            row = sub.Class1Record[class1.get(first, 0)]
            for second in glyphs:
                record = row.Class2Record[class2.get(second, 0)]
                value = getattr(record.Value1, "XAdvance", 0) if record.Value1 else 0
                yield (position[first], position[second]), value or 0


def rasterise(font_bytes: bytes, size: int) -> dict[str, np.ndarray]:
    """Glyph masks and metrics of the character set at one pixel size."""
    font = ImageFont.truetype(io.BytesIO(font_bytes), size)
    advances, boxes, bitmaps = [], [], []
    for ch in CHARSET:
        advances.append(font.getlength(ch))
        left, top, right, bottom = font.getbbox(ch, anchor="ls")
        width, height = max(right - left, 0), max(bottom - top, 0)
        if ch == " " or width == 0 or height == 0:
            boxes.append((0, 0, 0, 0))
            continue
        image = Image.new("L", (width, height), 0)
        ImageDraw.Draw(image).text((-left, -top), ch, font=font, fill=255, anchor="ls")
        # 16 coverage levels: indistinguishable from 256 on screen, and they compress well.
        mask = (np.round(np.asarray(image, dtype=np.float64) / 17) * 17).astype(np.uint8)
        boxes.append((left, top, width, height))
        bitmaps.append(mask.ravel())
    ascent, descent = font.getmetrics()
    cap_height = -font.getbbox("H", anchor="ls")[1]
    x_height = -font.getbbox("x", anchor="ls")[1]
    return {
        "advance": np.asarray(advances, dtype=np.float32),
        "box": np.asarray(boxes, dtype=np.int16),
        "bits": np.concatenate(bitmaps).astype(np.uint8),
        "metrics": np.asarray([ascent, descent, cap_height, x_height], dtype=np.float32),
    }


def build(
    source: Path, target: Path, weights: dict[str, dict[str, float]], sizes: tuple[int, ...]
) -> None:
    """Rasterise one typeface into its atlas."""
    arrays: dict[str, np.ndarray] = {
        "charset": np.asarray([ord(c) for c in CHARSET], dtype=np.int32),
        "sizes": np.asarray(sizes, dtype=np.int16),
    }
    for name, axes in weights.items():
        font = static_instance(source, axes)
        arrays["units_per_em"] = np.asarray(font["head"].unitsPerEm, dtype=np.int16)
        cmap = font.getBestCmap()
        glyphs = [cmap[ord(c)] for c in CHARSET]
        # kern[i, j]: adjustment (font units) after charset[i] when charset[j] follows.
        kern = np.zeros((len(CHARSET), len(CHARSET)), dtype=np.int16)
        for (first, second), value in kerning(font, glyphs).items():
            kern[first, second] = value
        arrays[f"{name}_kern"] = kern
        buffer = io.BytesIO()
        font.flavor = None
        font.save(buffer)
        for size in sizes:
            for key, value in rasterise(buffer.getvalue(), size).items():
                arrays[f"{name}_{size}_{key}"] = value
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)
    print(f"wrote {target.relative_to(ROOT)} ({target.stat().st_size / 1024:.0f} KiB)")


def main() -> None:
    for source, (target, weights, sizes) in FONTS.items():
        build(SOURCES / source, TARGETS / target, weights, sizes)


if __name__ == "__main__":
    main()
