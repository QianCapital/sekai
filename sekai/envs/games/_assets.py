"""The artwork the game renderers draw: chess pieces and xiangqi characters.

Both are stored at high resolution in `assets/` and resampled to the size a frame needs
(area-averaged, so they stay smooth at any size), then cached:

- `chess-pieces.npz`: Colin M.L. Burnett's chess pieces (BSD-3-Clause, see
  `assets/LICENSE-chess-pieces.txt`) as 256 x 256 RGBA images, keyed `wK`, `bQ`, ...
- `xiangqi-glyphs.npz`: the characters of the xiangqi board from Noto Serif CJK TC Bold
  (SIL Open Font License 1.1, see `assets/OFL.txt`), each a 96 x 96 coverage mask of its
  em box.

`tools/build_game_assets.py` rebuilds them.
"""

from __future__ import annotations

import functools
from importlib import resources

import numpy as np
from numpy.typing import NDArray

from sekai.envs.games._draw import RGBA, resample

_PIECES = "assets/chess-pieces.npz"
_GLYPHS = "assets/xiangqi-glyphs.npz"


def _frozen(array: NDArray[np.generic]) -> NDArray[np.generic]:
    array.flags.writeable = False
    return array


@functools.lru_cache(maxsize=1)
def _pieces() -> dict[str, NDArray[np.uint8]]:
    source = resources.files("sekai.envs.games").joinpath(_PIECES)
    with source.open("rb") as f, np.load(f) as archive:
        return {name: archive[name] for name in archive.files}


@functools.lru_cache(maxsize=1)
def _glyphs() -> tuple[dict[str, int], NDArray[np.uint8]]:
    source = resources.files("sekai.envs.games").joinpath(_GLYPHS)
    with source.open("rb") as f, np.load(f) as archive:
        chars = [chr(int(c)) for c in archive["chars"]]
        return {ch: i for i, ch in enumerate(chars)}, archive["masks"]


@functools.lru_cache(maxsize=64)
def chess_piece(code: str, size: int) -> RGBA:
    """A chess piece as an `(size, size, 4)` RGBA image (read-only).

    `code` is the colour then the piece: `"wK"`, `"bQ"`, `"wN"`, `"bP"`, ...
    """
    pieces = _pieces()
    if code not in pieces:
        raise KeyError(f"Unknown chess piece {code!r}; use e.g. 'wK' or 'bP'.")
    image = resample(pieces[code], size, size)
    return _frozen(image)  # type: ignore[return-value]


def has_glyph(ch: str) -> bool:
    """Whether the xiangqi atlas has the character `ch`."""
    return ch in _glyphs()[0]


@functools.lru_cache(maxsize=256)
def glyph(ch: str, em: int) -> NDArray[np.float64]:
    """The em box of character `ch`, `em` pixels square, as coverage in [0, 1] (read-only)."""
    index, masks = _glyphs()
    if ch not in index:
        raise KeyError(f"The xiangqi glyph atlas has no {ch!r}.")
    mask = masks[index[ch]]
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[..., 3] = mask
    small = resample(rgba, em, em)
    out = small[..., 3].astype(np.float64) / 255.0
    return _frozen(out)  # type: ignore[return-value]


__all__ = ["chess_piece", "glyph", "has_glyph"]
