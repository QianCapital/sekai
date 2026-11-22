"""AlphaGo Zero style board planes for Go.

An observation is a `float32` array of shape `(2 * T + 1, size, size)` with values in
`{0, 1}`, channel-first (PyTorch convention) and indexed `[plane, row, column]` with
row 0 the **top** row and column 0 the **left** column, the same orientation as the action
indices (`sekai.envs.games.go.actions`). The board is **not** flipped or rotated for the
side to move; instead the planes are split by "player to move" versus "opponent":

| plane | meaning |
|---|---|
| `0 .. T-1` | stones of the player to move at times `t, t-1, ..., t-T+1` |
| `T .. 2T-1` | stones of the opponent at the same times |
| `2T` | colour to move: all ones if **Black** is to move, else zeros |

Time `t` is the current position, `t-1` the position before the last move (a pass
repeats the position), and so on. History that does not exist yet (early in the game) is
zero-filled. `T = 8` gives the 17-plane input of AlphaGo Zero.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from sekai.envs.games.go.board import BLACK
from sekai.spaces.box import Box

DEFAULT_HISTORY_LENGTH = 8

Planes = NDArray[np.float32]
Position = NDArray[np.int8]
"""One position: colour per point (0 empty, 1 black, 2 white), shape `(size * size,)`."""


class ObservationEncoder:
    """Stateless encoder turning a position history into a network input.

    Args:
        size: The board size.
        history_length: Positions stacked into an observation (`T`), at least 1.
    """

    def __init__(self, size: int, history_length: int = DEFAULT_HISTORY_LENGTH) -> None:
        if history_length < 1:
            raise ValueError("history_length must be at least 1.")
        self.size = size
        self.history_length = history_length
        self.num_planes = 2 * history_length + 1
        self.shape: tuple[int, int, int] = (self.num_planes, size, size)

    def space(self) -> Box:
        """The observation space: `Box(0, 1, (2T + 1, size, size), float32)`."""
        return Box(0.0, 1.0, shape=self.shape, dtype=np.float32)

    def encode(self, history: Sequence[Position], to_play: int) -> Planes:
        """Build the observation for `to_play` (`BLACK` = 1 or `WHITE` = 2).

        `history` holds positions oldest first with the **current** position last (the
        natural order of an append-only deque); only the last `history_length` are used.
        """
        if len(history) == 0:
            raise ValueError("history must contain at least the current position.")
        t = self.history_length
        size = self.size
        recent = list(history)[-t:][::-1]  # current position first
        stack = np.stack(recent).reshape(len(recent), size, size)
        obs = np.zeros(self.shape, dtype=np.float32)
        k = len(recent)
        obs[:k] = stack == to_play
        obs[t : t + k] = stack == 3 - to_play
        if to_play == BLACK:
            obs[2 * t] = 1.0
        return obs

    def __repr__(self) -> str:
        return (
            f"ObservationEncoder(size={self.size}, history_length={self.history_length}, "
            f"shape={self.shape})"
        )


def decode_observation(observation: NDArray[np.floating]) -> tuple[list[int], int]:
    """Recover `(colours, to_play)` of the **current** position from an observation.

    `colours` is one entry per point (0 empty, 1 black, 2 white), row-major from the
    top-left. This is how observation-only policies (`act(observation, info)`) see the
    board.
    """
    obs = np.asarray(observation)
    if obs.ndim != 3 or obs.shape[0] % 2 != 1 or obs.shape[1] != obs.shape[2]:
        raise ValueError(f"Not a Go observation: shape {obs.shape}.")
    t = (obs.shape[0] - 1) // 2
    to_play = BLACK if obs[2 * t, 0, 0] >= 0.5 else 3 - BLACK
    mine = obs[0].reshape(-1) >= 0.5
    theirs = obs[t].reshape(-1) >= 0.5
    colours = np.zeros(mine.shape, dtype=np.int64)
    colours[mine] = to_play
    colours[theirs] = 3 - to_play
    out: list[int] = colours.tolist()
    return out, to_play


__all__ = [
    "DEFAULT_HISTORY_LENGTH",
    "ObservationEncoder",
    "decode_observation",
]
