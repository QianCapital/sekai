"""Build the images the strategy-game renderers draw with.

The game renderers need two kinds of artwork that sekai's text atlases do not have, and
draw them at any size by resampling high-resolution images stored with the package:

- sekai/envs/games/assets/chess-pieces.npz: the twelve chess pieces of Colin M.L.
  Burnett's set (the "Cburnett" pieces of Wikimedia Commons, used under their
  BSD-3-Clause licence option), rasterised from the SVG sources in
  tools/assets/cburnett/ as 256 x 256 RGBA images (supersampled 4x by a small SVG
  renderer below: paths, circles, transforms, fills with their fill rule, and strokes
  with round joins).
- sekai/envs/games/assets/xiangqi-glyphs.npz: the Chinese characters of the xiangqi
  board (the fourteen pieces, the river's 楚河漢界, Red's file numerals 一 to 九, and the
  digits 1 to 9) from Noto Serif CJK TC Bold (SIL Open Font License 1.1), as 96 x 96
  coverage masks of each character's em box.

Run it from the repository root after changing the sources (it needs fontTools and
Pillow, which sekai itself does not):

    pip install fonttools pillow
    python tools/build_game_assets.py --cjk-font /path/to/NotoSerifCJK-Bold.ttc

The font is the Traditional Chinese face of Noto Serif CJK (in Debian and Ubuntu's
fonts-noto-cjk package as /usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc, the
default here; also at https://github.com/notofonts/noto-cjk).
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTCollection
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SVG_DIR = ROOT / "tools" / "assets" / "cburnett"
TARGET = ROOT / "sekai" / "envs" / "games" / "assets"
DEFAULT_CJK_FONT = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"

PIECE_SIZE = 256
SUPERSAMPLE = 4
PIECES = {
    f"{color}{kind}": f"Chess_{kind.lower()}{'l' if color == 'w' else 'd'}t45.svg"
    for color in "wb"
    for kind in "KQRBNP"
}

GLYPH_EM = 96
XIANGQI_CHARS = "帥仕相傌俥炮兵將士象馬車砲卒楚河漢界一二三四五六七八九123456789"

SVG_NS = "{http://www.w3.org/2000/svg}"

# ----------------------------------------------------------------------
# A small SVG renderer: enough for the Cburnett pieces
# ----------------------------------------------------------------------

Matrix = np.ndarray


def _matrix(a: float, b: float, c: float, d: float, e: float, f: float) -> Matrix:
    return np.array([[a, c, e], [b, d, f], [0.0, 0.0, 1.0]])


def parse_transform(text: str | None) -> Matrix:
    m = np.eye(3)
    if not text:
        return m
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", text):
        v = [float(x) for x in re.split(r"[\s,]+", args.strip()) if x]
        if name == "translate":
            t = _matrix(1, 0, 0, 1, v[0], v[1] if len(v) > 1 else 0.0)
        elif name == "scale":
            t = _matrix(v[0], 0, 0, v[1] if len(v) > 1 else v[0], 0, 0)
        elif name == "matrix":
            t = _matrix(*v)
        elif name == "rotate":
            a = math.radians(v[0])
            t = _matrix(math.cos(a), math.sin(a), -math.sin(a), math.cos(a), 0, 0)
            if len(v) == 3:
                t = _matrix(1, 0, 0, 1, v[1], v[2]) @ t @ _matrix(1, 0, 0, 1, -v[1], -v[2])
        else:
            raise ValueError(f"Unsupported transform {name!r}.")
        m = m @ t
    return m


_TOKEN = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _arc_points(
    x0: float,
    y0: float,
    rx: float,
    ry: float,
    phi: float,
    large: bool,
    sweep: bool,
    x1: float,
    y1: float,
) -> list[tuple[float, float]]:
    """Points along an SVG elliptical arc (the endpoint-to-centre conversion of SVG F.6.5)."""
    if rx == 0 or ry == 0:
        return [(x1, y1)]
    rx, ry = abs(rx), abs(ry)
    cp, sp = math.cos(phi), math.sin(phi)
    dx, dy = (x0 - x1) / 2, (y0 - y1) / 2
    x1p, y1p = cp * dx + sp * dy, -sp * dx + cp * dy
    lam = (x1p / rx) ** 2 + (y1p / ry) ** 2
    if lam > 1:
        rx, ry = rx * math.sqrt(lam), ry * math.sqrt(lam)
    num = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    den = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    coef = math.sqrt(max(num / den, 0.0)) if den else 0.0
    if large == sweep:
        coef = -coef
    cxp, cyp = coef * rx * y1p / ry, -coef * ry * x1p / rx
    cx = cp * cxp - sp * cyp + (x0 + x1) / 2
    cy = sp * cxp + cp * cyp + (y0 + y1) / 2

    def angle(ux: float, uy: float, vx: float, vy: float) -> float:
        a = math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
        return a

    t1 = angle(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dt = angle((x1p - cxp) / rx, (y1p - cyp) / ry, (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not sweep and dt > 0:
        dt -= 2 * math.pi
    elif sweep and dt < 0:
        dt += 2 * math.pi
    n = max(int(abs(dt) / (math.pi / 64)), 4)
    out = []
    for i in range(1, n + 1):
        t = t1 + dt * i / n
        x = cx + rx * math.cos(t) * cp - ry * math.sin(t) * sp
        y = cy + rx * math.cos(t) * sp + ry * math.sin(t) * cp
        out.append((x, y))
    return out


def parse_path(d: str) -> list[tuple[list[tuple[float, float]], bool]]:
    """Subpaths of an SVG path as (points, closed), curves flattened to polylines."""
    tokens = _TOKEN.findall(d)
    subpaths: list[tuple[list[tuple[float, float]], bool]] = []
    points: list[tuple[float, float]] = []
    x = y = sx = sy = 0.0
    last_ctrl: tuple[float, float] | None = None
    last_cmd = ""
    i = 0
    cmd = ""

    def number() -> float:
        nonlocal i
        value = float(tokens[i])
        i += 1
        return value

    def flush(closed: bool) -> None:
        nonlocal points
        if len(points) > 1:
            subpaths.append((points, closed))
        points = []

    def cubic(p0, p1, p2, p3) -> None:  # type: ignore[no-untyped-def]
        for k in range(1, 33):
            t = k / 32
            mt = 1 - t
            points.append(
                (
                    mt**3 * p0[0] + 3 * mt * mt * t * p1[0] + 3 * mt * t * t * p2[0] + t**3 * p3[0],
                    mt**3 * p0[1] + 3 * mt * mt * t * p1[1] + 3 * mt * t * t * p2[1] + t**3 * p3[1],
                )
            )

    while i < len(tokens):
        if re.fullmatch(r"[A-Za-z]", tokens[i]):
            cmd = tokens[i]
            i += 1
        elif cmd in ("M", "m"):
            cmd = "L" if cmd == "M" else "l"  # extra coordinate pairs are line-tos
        rel = cmd.islower()
        c = cmd.upper()
        if c == "Z":
            if points:
                points.append((sx, sy))
            flush(True)
            x, y = sx, sy
            last_ctrl = None
            last_cmd = "Z"
            continue
        if c == "M":
            flush(False)
            nx, ny = number(), number()
            x, y = (x + nx, y + ny) if rel else (nx, ny)
            sx, sy = x, y
            points = [(x, y)]
            last_ctrl = None
        elif c == "L":
            nx, ny = number(), number()
            x, y = (x + nx, y + ny) if rel else (nx, ny)
            points.append((x, y))
            last_ctrl = None
        elif c == "H":
            nx = number()
            x = x + nx if rel else nx
            points.append((x, y))
            last_ctrl = None
        elif c == "V":
            ny = number()
            y = y + ny if rel else ny
            points.append((x, y))
            last_ctrl = None
        elif c in ("C", "S"):
            if c == "C":
                c1 = (number(), number())
                if rel:
                    c1 = (x + c1[0], y + c1[1])
            else:
                c1 = (
                    (2 * x - last_ctrl[0], 2 * y - last_ctrl[1])
                    if (last_ctrl is not None and last_cmd in ("C", "S"))
                    else (x, y)
                )
            c2 = (number(), number())
            end = (number(), number())
            if rel:
                c2 = (x + c2[0], y + c2[1])
                end = (x + end[0], y + end[1])
            cubic((x, y), c1, c2, end)
            last_ctrl = c2
            x, y = end
        elif c in ("Q", "T"):
            if c == "Q":
                q = (number(), number())
                if rel:
                    q = (x + q[0], y + q[1])
            else:
                q = (
                    (2 * x - last_ctrl[0], 2 * y - last_ctrl[1])
                    if (last_ctrl is not None and last_cmd in ("Q", "T"))
                    else (x, y)
                )
            end = (number(), number())
            if rel:
                end = (x + end[0], y + end[1])
            c1 = (x + 2 / 3 * (q[0] - x), y + 2 / 3 * (q[1] - y))
            c2 = (end[0] + 2 / 3 * (q[0] - end[0]), end[1] + 2 / 3 * (q[1] - end[1]))
            cubic((x, y), c1, c2, end)
            last_ctrl = q
            x, y = end
        elif c == "A":
            rx, ry, rot = number(), number(), number()
            large, sweep = bool(number()), bool(number())
            ex, ey = number(), number()
            if rel:
                ex, ey = x + ex, y + ey
            points.extend(_arc_points(x, y, rx, ry, math.radians(rot), large, sweep, ex, ey))
            x, y = ex, ey
            last_ctrl = None
        last_cmd = c
    flush(False)
    return subpaths


def _style(element: ET.Element, inherited: dict[str, str]) -> dict[str, str]:
    style = dict(inherited)
    for key in (
        "fill",
        "fill-opacity",
        "fill-rule",
        "stroke",
        "stroke-width",
        "stroke-linecap",
        "stroke-linejoin",
        "stroke-opacity",
        "opacity",
    ):
        if key in element.attrib:
            style[key] = element.attrib[key]
    for item in element.attrib.get("style", "").split(";"):
        if ":" in item:
            key, value = item.split(":", 1)
            style[key.strip()] = value.strip()
    return style


def _color(value: str) -> tuple[float, float, float] | None:
    value = value.strip().lower()
    if value in ("none", ""):
        return None
    named = {"black": "#000000", "white": "#ffffff"}
    value = named.get(value, value)
    if not value.startswith("#"):
        raise ValueError(f"Unsupported colour {value!r}.")
    h = value[1:]
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    return tuple(int(h[k : k + 2], 16) / 255 for k in (0, 2, 4))  # type: ignore[return-value]


class Raster:
    """A premultiplied RGBA canvas that SVG shapes are painted on, in painter's order."""

    def __init__(self, size: int, units: float) -> None:
        self.size = size
        self.scale = size / units
        self.rgba = np.zeros((size, size, 4))
        centres = np.arange(size) + 0.5
        self.xs, self.ys = np.meshgrid(centres, centres)

    def to_px(self, pts: list[tuple[float, float]], m: Matrix) -> np.ndarray:
        arr = np.asarray(pts, dtype=np.float64)
        homo = np.column_stack([arr, np.ones(len(arr))]) @ m.T
        return homo[:, :2] * self.scale

    def fill(self, subpaths: list[np.ndarray], evenodd: bool) -> np.ndarray:
        """Coverage (0/1 per pixel centre) of closed subpaths under the fill rule."""
        winding = np.zeros((self.size, self.size), dtype=np.int32)
        for pts in subpaths:
            if len(pts) < 3:
                continue
            closed = np.vstack([pts, pts[:1]])
            for (x0, y0), (x1, y1) in itertools.pairwise(closed):
                if y0 == y1:
                    continue
                lo, hi = (y0, y1) if y0 < y1 else (y1, y0)
                r0 = max(math.ceil(lo - 0.5), 0)
                r1 = min(math.ceil(hi - 0.5), self.size)
                if r0 >= r1:
                    continue
                ys = np.arange(r0, r1) + 0.5
                xs = x0 + (ys - y0) * (x1 - x0) / (y1 - y0)
                direction = 1 if y1 > y0 else -1
                for row, xc in zip(range(r0, r1), xs):
                    col = math.ceil(xc - 0.5)
                    if col < self.size:
                        winding[row, max(col, 0) :] += direction
        return (winding % 2 != 0) if evenodd else (winding != 0)

    def stroke(self, subpaths: list[tuple[np.ndarray, bool]], width: float, cap: str) -> np.ndarray:
        """Coverage of strokes `width` pixels wide: segments, round joins, the given caps."""
        out = np.zeros((self.size, self.size), dtype=bool)
        half = width / 2
        for pts, closed in subpaths:
            if len(pts) < 2:
                continue
            for k, ((ax, ay), (bx, by)) in enumerate(itertools.pairwise(pts)):
                x0 = max(int(min(ax, bx) - half - 1), 0)
                x1 = min(int(max(ax, bx) + half + 2), self.size)
                y0 = max(int(min(ay, by) - half - 1), 0)
                y1 = min(int(max(ay, by) + half + 2), self.size)
                if x0 >= x1 or y0 >= y1:
                    continue
                px, py = self.xs[y0:y1, x0:x1], self.ys[y0:y1, x0:x1]
                dx, dy = bx - ax, by - ay
                length2 = dx * dx + dy * dy
                raw = ((px - ax) * dx + (py - ay) * dy) / length2 if length2 else 0.0 * px
                first, last = k == 0 and not closed, k == len(pts) - 2 and not closed
                t = np.clip(raw, 0.0, 1.0)
                dist = np.hypot(px - (ax + t * dx), py - (ay + t * dy))
                hit = dist <= half
                if cap == "butt":
                    if first:
                        hit &= raw >= 0
                    if last:
                        hit &= raw <= 1
                out[y0:y1, x0:x1] |= hit
        return out

    def paint(self, coverage: np.ndarray, color: tuple[float, float, float], alpha: float) -> None:
        a = coverage.astype(np.float64) * alpha
        ink = np.asarray(color) * a[..., None]
        self.rgba[..., :3] = self.rgba[..., :3] * (1 - a[..., None]) + ink
        self.rgba[..., 3] = self.rgba[..., 3] * (1 - a) + a


def render_svg(path: Path, size: int) -> np.ndarray:
    """The SVG at `path` as an `(size, size, 4)` uint8 RGBA image (straight alpha)."""
    root = ET.parse(path).getroot()
    units = float(root.attrib.get("width", "45"))
    big = size * SUPERSAMPLE
    raster = Raster(big, units)

    def walk(element: ET.Element, style: dict[str, str], m: Matrix) -> None:
        tag = element.tag.replace(SVG_NS, "")
        style = _style(element, style)
        m = m @ parse_transform(element.attrib.get("transform"))
        if tag in ("svg", "g"):
            for child in element:
                walk(child, style, m)
            return
        if tag == "path":
            subpaths = parse_path(element.attrib["d"])
        elif tag == "circle":
            cx, cy, r = (float(element.attrib[k]) for k in ("cx", "cy", "r"))
            ring = [
                (cx + r * math.cos(t), cy + r * math.sin(t))
                for t in np.linspace(0, 2 * math.pi, 97)
            ]
            subpaths = [(ring, True)]
        else:
            return
        opacity = float(style.get("opacity", "1"))
        fill = _color(style.get("fill", "#000000"))
        px = [(raster.to_px(pts, m), closed) for pts, closed in subpaths]
        if fill is not None:
            coverage = raster.fill([p for p, _ in px], style.get("fill-rule") == "evenodd")
            raster.paint(coverage, fill, opacity * float(style.get("fill-opacity", "1")))
        stroke = _color(style.get("stroke", "none"))
        if stroke is not None:
            scale = math.sqrt(abs(np.linalg.det(m[:2, :2])))
            width = float(style.get("stroke-width", "1")) * scale * raster.scale
            coverage = raster.stroke(px, width, style.get("stroke-linecap", "butt"))
            raster.paint(coverage, stroke, opacity * float(style.get("stroke-opacity", "1")))

    walk(root, {}, np.eye(3))
    # Box-filter the supersampled premultiplied image down to `size`.
    rgba = raster.rgba.reshape(size, SUPERSAMPLE, size, SUPERSAMPLE, 4).mean(axis=(1, 3))
    alpha = rgba[..., 3:]
    color = np.divide(rgba[..., :3], alpha, out=np.zeros_like(rgba[..., :3]), where=alpha > 1e-6)
    out = np.concatenate([color, alpha], axis=-1)
    return np.clip(np.round(out * 255), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# Xiangqi glyphs
# ----------------------------------------------------------------------


def _traditional_face(path: str) -> int:
    """The index of the Traditional Chinese face in a Noto CJK collection."""
    if not path.lower().endswith(".ttc"):
        return 0
    for index, font in enumerate(TTCollection(path).fonts):
        if font["name"].getDebugName(1).endswith(" TC"):
            return index
    raise ValueError(f"{path} has no Traditional Chinese (TC) face.")


def render_glyphs(path: str) -> dict[str, np.ndarray]:
    """Each character's em box as a `(GLYPH_EM, GLYPH_EM)` uint8 coverage mask.

    The em box is the ideographic one: from the font's typographic ascender (top) to its
    descender (bottom), so a character centred in its mask is centred as a browser
    centres it with `dominant-baseline: central`.
    """
    index = _traditional_face(path)
    font_tt = TTCollection(path).fonts[index] if path.lower().endswith(".ttc") else None
    os2 = font_tt["OS/2"] if font_tt is not None else None
    upm = font_tt["head"].unitsPerEm if font_tt is not None else 1000
    ascender = (os2.sTypoAscender if os2 is not None else 880) / upm
    scale = 4
    em = GLYPH_EM * scale
    font = ImageFont.truetype(path, em, index=index)
    masks: dict[str, np.ndarray] = {}
    for ch in XIANGQI_CHARS:
        image = Image.new("L", (em, em), 0)
        draw = ImageDraw.Draw(image)
        advance = font.getlength(ch)
        draw.text(((em - advance) / 2, ascender * em), ch, fill=255, font=font, anchor="ls")
        big = np.asarray(image, dtype=np.float64) / 255.0
        small = big.reshape(GLYPH_EM, scale, GLYPH_EM, scale).mean(axis=(1, 3))
        masks[ch] = np.clip(np.round(small * 255), 0, 255).astype(np.uint8)
    return masks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cjk-font", default=DEFAULT_CJK_FONT)
    args = parser.parse_args()
    TARGET.mkdir(parents=True, exist_ok=True)

    pieces = {code: render_svg(SVG_DIR / name, PIECE_SIZE) for code, name in PIECES.items()}
    np.savez_compressed(TARGET / "chess-pieces.npz", **pieces)
    print(f"wrote {TARGET / 'chess-pieces.npz'} ({len(pieces)} pieces at {PIECE_SIZE} px)")

    masks = render_glyphs(args.cjk_font)
    np.savez_compressed(
        TARGET / "xiangqi-glyphs.npz",
        chars=np.array([ord(ch) for ch in masks], dtype=np.int32),
        masks=np.stack(list(masks.values())),
    )
    print(f"wrote {TARGET / 'xiangqi-glyphs.npz'} ({len(masks)} glyphs at {GLYPH_EM} px)")


if __name__ == "__main__":
    main()
