"""Episode samplers: where in a price history each episode starts.

Historical environments cut episodes out of a longer history. Training wants
random windows; evaluation wants to walk through a period deterministically,
episode after episode, or to replay one specific date. A sampler encapsulates
that choice and every historical sekai env accepts one (`sampler=`), plus a
per-reset override: `env.reset(options={"start": "2024-01-02"})` puts the
episode's first decision at that timestamp (an integer is a row position, on
every time axis; `Time(t)` looks up a numeric timestamp `t`).

Data cut by `MarketData.split` or `walk_forward` begins with warm-up rows
(`MarketData.warmup`) that may be observed but never traded on. Environments then
sample only among the windows whose first decision comes after them (`sample_start`),
and refuse an explicit start on a warm-up row (`resolve_start`).
"""

from __future__ import annotations

import abc
import copy
from typing import Any

import numpy as np

from sekai.finance.data import row_of


class EpisodeSampler(abc.ABC):
    """Chooses the first row of each episode's data window."""

    @abc.abstractmethod
    def sample(self, rng: np.random.Generator, n_starts: int, episode_length: int) -> int:
        """A window start in `[0, n_starts)`.

        Args:
            rng: The env's generator (use it for any randomness).
            n_starts: Number of valid window starts.
            episode_length: Steps per episode (for samplers that tile the history).
        """

    def reset(self) -> None:  # noqa: B027 (optional hook)
        """Forget any state (e.g. a sequential sampler's position)."""

    def spawn(self) -> EpisodeSampler:
        """A sampler for one more independent environment (e.g. another sub-env of a VecEnv).

        `make_vec()` and native vector envs call it once per sub-env. The default returns
        this sampler itself, shared, so later changes to it (a curriculum raising a limit,
        say) reach every environment. Override it when the sampler keeps per-environment
        state, as `SequentialStart` does with its position. (`ProcVecEnv` workers always
        hold pickled copies, which later changes in the parent cannot reach.)
        """
        return self


class RandomStart(EpisodeSampler):
    """Uniformly random windows: the default for training."""

    def sample(self, rng: np.random.Generator, n_starts: int, episode_length: int) -> int:
        return int(rng.integers(0, n_starts))

    def __repr__(self) -> str:
        return "RandomStart()"


class SequentialStart(EpisodeSampler):
    """Consecutive windows through the history, for deterministic evaluation.

    The first episode starts at `start`; each later one `stride` rows on
    (default: the episode length, so episodes tile the history without
    overlap). Handing out the last window that fits increases `epoch` and
    wraps around to `start`, so `while sampler.epoch == 0: run_episode()`
    visits every window exactly once. Rows after the last full window are not
    used.
    """

    def __init__(self, start: int = 0, stride: int | None = None) -> None:
        if start < 0:
            raise ValueError(f"start must be >= 0, got {start}.")
        if stride is not None and stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}.")
        self.start = start
        self.stride = stride
        self.epoch = 0
        self._next = start

    def sample(self, rng: np.random.Generator, n_starts: int, episode_length: int) -> int:
        if self.start >= n_starts:
            raise ValueError(f"start {self.start} is beyond the last window start {n_starts - 1}.")
        if self._next >= n_starts:  # n_starts shrank since the last call
            self._next = self.start
        out = self._next
        self._next += self.stride or episode_length
        if self._next >= n_starts:
            self._next = self.start
            self.epoch += 1
        return out

    def reset(self) -> None:
        self._next = self.start
        self.epoch = 0

    def spawn(self) -> SequentialStart:
        """A fresh copy (same start and stride) with its own position and epoch."""
        twin = copy.copy(self)
        twin.reset()
        return twin

    def __repr__(self) -> str:
        return f"SequentialStart(start={self.start}, stride={self.stride})"


class FixedStart(EpisodeSampler):
    """Every episode starts at the same window (replaying one period)."""

    def __init__(self, start: int) -> None:
        if start < 0:
            raise ValueError(f"start must be >= 0, got {start}.")
        self.start = start

    def sample(self, rng: np.random.Generator, n_starts: int, episode_length: int) -> int:
        if self.start >= n_starts:
            raise ValueError(f"start {self.start} is beyond the last window start {n_starts - 1}.")
        return self.start

    def __repr__(self) -> str:
        return f"FixedStart({self.start})"


def as_sampler(sampler: str | int | EpisodeSampler) -> EpisodeSampler:
    """Normalise a sampler argument: "random", "sequential", a fixed int, or a sampler."""
    if isinstance(sampler, EpisodeSampler):
        return sampler
    if isinstance(sampler, (int, np.integer)) and not isinstance(sampler, bool):
        return FixedStart(int(sampler))
    if sampler == "random":
        return RandomStart()
    if sampler == "sequential":
        return SequentialStart()
    raise ValueError(
        f"sampler must be 'random', 'sequential', an int or an EpisodeSampler, got {sampler!r}."
    )


def _no_room(n_starts: int, lookback: int, warmup: int) -> ValueError:
    return ValueError(
        f"No episode fits after the {warmup} warm-up rows: with lookback={lookback} the "
        f"first decision may not come before row {warmup}, but the last window starts at "
        f"row {n_starts - 1} (first decision at row {lookback + n_starts - 1}). Use a split "
        "with more test rows or a shorter episode."
    )


def first_start(lookback: int, warmup: int) -> int:
    """The first window start whose first decision (row `start + lookback`) is not a warm-up row."""
    return max(0, warmup - lookback)


def sample_start(
    sampler: EpisodeSampler,
    rng: np.random.Generator,
    *,
    n_starts: int,
    episode_length: int,
    lookback: int,
    warmup: int = 0,
) -> int:
    """A window start from `sampler` whose decisions all fall after the data's warm-up rows.

    The sampler chooses among the valid starts only: its position `k` is the window
    starting at row `first_start(lookback, warmup) + k`. Without warm-up rows (or
    with `lookback >= warmup`) that is exactly the sampler's own choice.

    Raises:
        ValueError: If no window fits after the warm-up rows, or the sampler cannot
            place an episode (with warm-up rows, the message gives the positions in
            terms of the windows after them).
    """
    first = first_start(lookback, warmup)
    if first >= n_starts:
        raise _no_room(n_starts, lookback, warmup)
    n_valid = n_starts - first
    try:
        position = sampler.sample(rng, n_valid, episode_length)
    except ValueError as error:
        if first == 0:  # positions are window starts: the sampler's message is exact
            raise
        windows = (
            f"the {n_valid} windows after the {warmup} warm-up rows (positions "
            f"0..{n_valid - 1}: window starts {first}..{n_starts - 1}, first decisions at "
            f"rows {first + lookback}..{n_starts - 1 + lookback})"
        )
        start = getattr(sampler, "start", None)
        if isinstance(start, int) and start >= n_valid:
            message = f"{sampler!r}: position {start} is beyond the last of {windows}."
        else:
            message = f"{sampler!r} could not choose among {windows}: {error}"
        raise ValueError(message) from error
    return first + position


def resolve_start(
    option: Any,
    *,
    timestamps: Any,
    lookback: int,
    n_starts: int,
    warmup: int = 0,
) -> int:
    """The window start for `reset(options={"start": option})`.

    `option` places the episode's first decision, read as `MarketData.index_of`
    reads it: an `int` is a row position on every time axis, and a timestamp puts
    the decision on the first row at or after it (on datetime timestamps anything
    `numpy.datetime64` accepts, on numeric ones a float, on either `Time(t)`). The
    window begins `lookback` rows earlier. `info["timestamp"]` reports the decision
    time, so `start=Time(info["timestamp"])` replays an episode on any data,
    including a split whose numeric timestamps no longer start at 0. A first
    decision on one of the data's `warmup` rows (lookback-only history of a split)
    is refused.

    Raises:
        TypeError: If `option` is a bool.
        ValueError: If `option` is a timestamp that does not suit the axis (such as
            a date string on numeric timestamps), or the first decision would fall
            outside the valid range: without `lookback` rows before it, on a warm-up
            row, or without a full episode after it.
    """
    times = np.asarray(timestamps)
    decision = row_of(times, option, what="start")
    first, last = max(lookback, warmup), lookback + n_starts - 1
    if first > last:
        raise _no_room(n_starts, lookback, warmup)
    if not first <= decision <= last:
        reason = (
            f" (rows before {warmup} are warm-up history of this split, never traded on)"
            if lookback <= decision < warmup
            else ""
        )
        if (
            isinstance(option, int | np.integer)
            and times.dtype.kind != "M"
            and times[0] <= option <= times[-1]
        ):
            reason += f". An int is a row position; for the timestamp {option}, pass Time({option})"
        raise ValueError(
            f"start {option!r} (row {decision}) is outside the valid range of first decisions: "
            f"rows {first}..{last} ({times[first]} .. {times[last]}){reason}."
        )
    return decision - lookback


__all__ = [
    "EpisodeSampler",
    "RandomStart",
    "SequentialStart",
    "FixedStart",
    "as_sampler",
    "first_start",
    "sample_start",
    "resolve_start",
]
