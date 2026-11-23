"""Scoring for humans: Tromp-Taylor area counts, Monte-Carlo ownership and dead stones.

The environment itself scores finished games with pure **Tromp-Taylor area scoring**
(every stone on the board counts as alive), which is the right contract for zero-knowledge
reinforcement learning. People, however, pass with dead stones still on the board, and a
game shown to them needs a result they will recognise:

- `area_score` is the raw Tromp-Taylor count (no komi).
- `estimate_ownership` runs uniformly random, eye-avoiding playouts from the current
  position and returns the mean final ownership of every point (`+1` Black, `-1` White).
  The playouts also refuse self-atari of two or more stones, so neither side is forced to
  fill the shared liberties of a seki once the rest of the board is settled.
- `final_score` judges each group as a whole: a group whose mean ownership has the
  opposite sign to its colour with magnitude at least `DEAD_THRESHOLD` (0.5) is dead.
  Dead stones are removed and the board is then counted by Tromp-Taylor area scoring, so a
  dead group's points go to the side that surrounds them, while seki groups keep their own
  colour's ownership, stay on the board and leave their shared liberties neutral.
  With `estimate_dead=False` it is exactly the environment's Tromp-Taylor result.

All functions accept a `GoEnv` (or anything wrapping one) or a bare `Board` (then pass
`komi=` to `final_score`). Playouts use simple ko only.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sekai.envs.games.go.actions import action_to_vertex
from sekai.envs.games.go.board import BLACK, WHITE, Board, bits_to_points
from sekai.envs.games.go.env import format_result, winner_of
from sekai.utils.seeding import np_random

SeedType = int | np.random.Generator | random.Random | None

DEFAULT_PLAYOUTS = 200
DEAD_THRESHOLD = 0.5
"""A group is dead when its mean ownership is at least this strongly the opponent's."""


@dataclass(frozen=True, slots=True)
class GoScore:
    """A human-facing final score (area scoring, optionally with dead stones removed).

    Attributes:
        black: Black's area points.
        white: White's area points including komi.
        margin: `black - white`.
        winner: `"black"`, `"white"` or `None` for a draw.
        result: `"B+3.5"`, `"W+12.5"` or `"Draw"`.
        dead_stones: GTP vertices of the stones judged dead (and removed before counting).
        territory: `{"black": [...], "white": [...]}`: points each side scores that are not
            its own live stones (empty territory and captured-dead points), for territory
            markers.
        method: Short human description of how the score was computed.
        komi: The komi counted.
    """

    black: float
    white: float
    margin: float
    winner: str | None
    result: str
    dead_stones: list[str] = field(default_factory=list)
    territory: dict[str, list[str]] = field(default_factory=dict)
    method: str = ""
    komi: float = 0.0


def _unwrap(obj: Any) -> Any:
    return getattr(obj, "unwrapped", obj)


def _board_of(obj: Any) -> Board:
    obj = _unwrap(obj)
    if isinstance(obj, Board):
        return obj
    board = getattr(obj, "board", None)
    if isinstance(board, Board):
        return board
    raise TypeError(f"Expected a GoEnv or a Board, got {type(obj).__name__}.")


def _python_rng(rng: SeedType) -> random.Random:
    """A fast stdlib RNG for playouts, derived deterministically from `rng`."""
    if isinstance(rng, random.Random):
        return rng
    gen, _ = np_random(rng)
    return random.Random(int(gen.integers(0, 2**63)))


def area_score(board: Any) -> tuple[int, int]:
    """Tromp-Taylor `(black_points, white_points)` of a board or env, **without** komi."""
    return _board_of(board).area_score()


def _unpack(bits: int, n: int) -> NDArray[np.uint8]:
    raw = np.frombuffer(bits.to_bytes((n + 7) // 8, "little"), dtype=np.uint8)
    return np.unpackbits(raw, bitorder="little")[:n]


def estimate_ownership(
    env: Any,
    playouts: int = DEFAULT_PLAYOUTS,
    rng: SeedType = None,
    *,
    time_limit: float | None = None,
    avoid_self_atari: bool = True,
) -> NDArray[np.float32]:
    """Mean final ownership per point after random eye-avoiding playouts.

    Returns a `float32` array of shape `(size * size,)` in `[-1, 1]`: `+1` means the point
    always ended up Black's (stone or Tromp-Taylor territory), `-1` White's, `0` neutral or
    contested. Playouts start from the current position with the player to move; the env
    is not modified. `time_limit` (seconds) optionally stops early, after at least one
    playout.

    By default the playouts also skip self-atari moves of two or more stones
    (`Board.is_self_atari()`). Without that, in a finished position the side to move is
    forced to fill a shared liberty of a seki (it may only pass when it has no other move)
    and the seki is always "won" by whoever is not to move.

    Args:
        env: A `GoEnv` (or a wrapper of one) or a `Board`.
        playouts: Playouts to average (at least one is played).
        rng: Seed, `numpy.random.Generator` or `random.Random` for the playouts.
        time_limit: Optional cap in seconds.
        avoid_self_atari: Skip self-atari moves of two or more stones.
    """
    board = _board_of(env)
    n = board.n
    rnd = _python_rng(rng)
    total = np.zeros(n, dtype=np.int64)
    deadline = None if time_limit is None else time.perf_counter() + time_limit
    done = 0
    for _ in range(max(int(playouts), 1)):
        b = board.copy()
        b.random_playout(rnd, avoid_self_atari=avoid_self_atari)
        black, white = b.area_bits()
        total += _unpack(black, n).astype(np.int64)
        total -= _unpack(white, n).astype(np.int64)
        done += 1
        if deadline is not None and time.perf_counter() >= deadline:
            break
    ownership: NDArray[np.float32] = (total / done).astype(np.float32)
    return ownership


def dead_groups(
    board: Board, ownership: NDArray[np.floating[Any]], threshold: float = DEAD_THRESHOLD
) -> list[int]:
    """Points of every group whose mean ownership says it belongs to the opponent."""
    dead: list[int] = []
    for _root, color, stones, _libs in board.groups():
        points = bits_to_points(stones)
        mean = float(np.mean(ownership[points]))
        if (color == BLACK and mean <= -threshold) or (color == WHITE and mean >= threshold):
            dead.extend(points)
    return sorted(dead)


def final_score(
    env: Any,
    *,
    estimate_dead: bool = True,
    playouts: int = DEFAULT_PLAYOUTS,
    rng: SeedType = None,
    komi: float | None = None,
    threshold: float = DEAD_THRESHOLD,
    time_limit: float | None = None,
) -> GoScore:
    """Area score of the current position, with dead stones estimated by playouts.

    Args:
        env: A `GoEnv` (komi is taken from it) or a `Board` (pass `komi`).
        estimate_dead: Remove groups judged dead by `estimate_ownership`; `False` gives
            exactly the Tromp-Taylor result the environment uses.
        playouts: Playouts for the ownership estimate.
        rng: Seed, `numpy.random.Generator` or `random.Random` for the playouts.
        komi: Overrides the env's komi (required for a bare board, default 0 then).
        threshold: Dead-group ownership threshold, default `DEAD_THRESHOLD`.
        time_limit: Optional cap in seconds on the ownership playouts.
    """
    board = _board_of(env)
    if komi is None:
        komi = float(getattr(_unwrap(env), "komi", 0.0))
    size, n = board.size, board.n
    geo = board.geo

    dead: list[int] = []
    if estimate_dead:
        ownership = estimate_ownership(board, playouts, rng, time_limit=time_limit)
        dead = dead_groups(board, ownership, threshold)
    dead_bits = 0
    for p in dead:
        dead_bits |= 1 << p
    black_stones = board.stones[BLACK] & ~dead_bits
    white_stones = board.stones[WHITE] & ~dead_bits
    black_area, white_area = geo.area_bits(black_stones, white_stones)
    black = black_area.bit_count()
    white = white_area.bit_count() + float(komi)
    margin = black - white

    def vertices(bits: int) -> list[str]:
        return [action_to_vertex(p, size) for p in bits_to_points(bits) if p < n]

    if estimate_dead:
        method = f"Area scoring, dead stones estimated from {max(int(playouts), 1)} playouts"
    else:
        method = "Tromp-Taylor area scoring, all stones counted as alive"
    return GoScore(
        black=float(black),
        white=float(white),
        margin=float(margin),
        winner=winner_of(margin),
        result=format_result(margin),
        dead_stones=vertices(dead_bits),
        territory={
            "black": vertices(black_area & ~black_stones),
            "white": vertices(white_area & ~white_stones),
        },
        method=method,
        komi=float(komi),
    )


__all__ = [
    "DEFAULT_PLAYOUTS",
    "DEAD_THRESHOLD",
    "GoScore",
    "area_score",
    "estimate_ownership",
    "dead_groups",
    "final_score",
]
