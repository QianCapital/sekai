"""Build the Qian Capital brand images that branded dashboards draw.

sekai draws without an image library at runtime, so the gold mark and wordmark are cut
from the logo lockup in the documentation and stored as RGBA arrays in a compressed
NumPy archive that ships with the package; `sekai.rendering.brand` resamples them to the
size a frame needs.

Run it from the repository root after the logo changes (it needs Pillow, which sekai
itself does not):

    pip install pillow
    python tools/build_brand_assets.py

It reads docs/assets/brand/qian-capital.png and writes
sekai/rendering/assets/qian-capital.npz. The logo is Qian Capital's trademark and
artwork, not licensed with sekai's code: see TRADEMARKS for what it may be used for.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "assets" / "brand" / "qian-capital.png"
TARGET = ROOT / "sekai" / "rendering" / "assets" / "qian-capital.npz"
# Pixels this transparent or more are background when cropping.
THRESHOLD = 8


def crop(pixels: np.ndarray) -> np.ndarray:
    """The smallest box holding every visible pixel, with a transparent pixel around it."""
    rows = np.flatnonzero((pixels[..., 3] > THRESHOLD).any(axis=1))
    cols = np.flatnonzero((pixels[..., 3] > THRESHOLD).any(axis=0))
    box = pixels[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    return np.pad(box, ((1, 1), (1, 1), (0, 0)))


def main() -> None:
    pixels = np.asarray(Image.open(SOURCE).convert("RGBA"))
    # The lockup is the mark, a gap of empty columns, then the wordmark.
    empty = ~(pixels[..., 3] > THRESHOLD).any(axis=0)
    first = int(np.argmax(~empty))
    gap = first + int(np.argmax(empty[first:]))
    mark, wordmark = crop(pixels[:, :gap]), crop(pixels[:, gap:])
    # The wordmark's capitals: the rows most of its letters cover (the Q's tail hangs below).
    covered = (wordmark[..., 3] > 128).sum(axis=1)
    caps = np.flatnonzero(covered > 0.15 * covered.max())
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        TARGET,
        mark=mark,
        wordmark=wordmark,
        wordmark_caps=np.asarray([caps[0], caps[-1] + 1], dtype=np.int16),
    )
    print(
        f"wrote {TARGET.relative_to(ROOT)} ({TARGET.stat().st_size / 1024:.0f} KiB): "
        f"mark {mark.shape[:2]}, wordmark {wordmark.shape[:2]}"
    )


if __name__ == "__main__":
    main()
