"""Plumbing shared by the finance environments: history windows and cost context."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sekai.envs.finance.market import validate_prices
from sekai.finance.data import MarketData
from sekai.finance.sampling import (
    EpisodeSampler,
    as_sampler,
    resolve_start,
    sample_start,
)


class History:
    """A price history (array or MarketData) that episodes are cut from.

    Handles validation, the sampler, and `reset(options={"start": ...})`.

    Args:
        source: An array of prices (`(T,)` or `(T, n)`) or a MarketData.
        ndim: 1 for a single-asset env (the history must hold one asset), 2 for multi-asset.
        window: Rows per episode (lookback + episode_length + 1).
        lookback: Rows before the first decision.
        sampler: Episode sampler (see `sekai.finance.sampling`).
        allow_missing: Whether NaN prices (unlisted assets) are allowed.
    """

    def __init__(
        self,
        source: ArrayLike | MarketData,
        *,
        ndim: int,
        window: int,
        lookback: int,
        sampler: str | int | EpisodeSampler,
        allow_missing: bool = False,
    ) -> None:
        if isinstance(source, MarketData):
            data = source
            if data.n_steps < window:
                raise ValueError(
                    f"The history has {data.n_steps} rows but at least {window} are needed "
                    "(lookback + episode_length + 1)."
                )
            if not allow_missing and not data.mask.all():
                raise ValueError(
                    "The history has missing prices (unlisted assets); this env needs a complete "
                    "panel. Use Market-v0 (sekai.envs.finance.MarketEnv) for changing universes."
                )
            if ndim == 1 and data.n_assets != 1:
                raise ValueError(
                    f"This env trades one asset but the MarketData has {data.n_assets}; pick one "
                    "with data.select([asset])."
                )
        else:
            prices = validate_prices(source, ndim=ndim, min_length=window)
            data = MarketData(prices)
        self.data = data
        self.prices: NDArray[np.float64] = (
            np.asarray(data.prices[:, 0]) if ndim == 1 else np.asarray(data.prices)
        )
        self.window = window
        self.lookback = lookback
        self.sampler = as_sampler(sampler)
        self.n_starts = data.n_steps - window + 1

    def start(
        self, rng: np.random.Generator, options: dict[str, Any] | None, episode_length: int
    ) -> int:
        """The first row of the next episode's window (no decision on a warm-up row)."""
        if options and options.get("start") is not None:
            return resolve_start(
                options["start"],
                timestamps=self.data.timestamps,
                lookback=self.lookback,
                n_starts=self.n_starts,
                warmup=self.data.warmup,
            )
        return sample_start(
            self.sampler,
            rng,
            n_starts=self.n_starts,
            episode_length=episode_length,
            lookback=self.lookback,
            warmup=self.data.warmup,
        )

    def timestamp(self, row: int) -> Any:
        """The timestamp of an absolute row (a numpy datetime64 or number)."""
        return self.data.timestamps[row]

    def spawn(self) -> History:
        """This history for one more independent environment, sharing the data.

        Its sampler is `self.sampler.spawn()`: the same sampler unless it keeps
        per-environment state (a sequential sampler's position), so a native vector
        env's sub-envs sample their windows like separate single envs.
        """
        sampler = self.sampler.spawn()
        if sampler is self.sampler:
            return self
        lane = copy.copy(self)
        lane.sampler = sampler
        return lane


def step_length(dt: float | None, history: History | None) -> float:
    """Years per step: as given, else implied by dated history, else one trading day."""
    if dt is not None:
        return float(dt)
    ppy = history.data.periods_per_year if history is not None else None
    return 1.0 / ppy if ppy else 1 / 252


def check_cost_inputs(
    cost_model: Any, data: MarketData | None, *, simulated: bool, lookback: int | None = None
) -> None:
    """Fail at construction (not at the first step) when a cost model needs missing data.

    With `lookback`, a model that reads per-asset volatility is also refused when the
    env's lookback holds a single return: no volatility estimate is ever possible, and
    some envs report that window's volatility as 0, a valid (flat) value that a fallback
    would never replace, so every trade would pay zero impact.
    """
    for name, unit in (("volume", "shares traded per step"), ("spread", "fraction of the price")):
        if name in cost_model.requires and (data is None or name not in data.fields):
            source = "a simulated market" if simulated else "this MarketData"
            raise ValueError(
                f"{cost_model!r} needs a {name!r} data field ({unit}), which {source} does "
                "not provide."
            )
    if lookback is not None and lookback < 2 and "volatility" in cost_model.requires:
        raise ValueError(
            f"{cost_model!r} needs a per-asset volatility estimate, which takes at least 2 "
            f"returns, but lookback={lookback} observes only {lookback}; every trade would be "
            "unpriced. Use lookback >= 2."
        )


def field_or_none(data: MarketData | None, name: str) -> NDArray[np.float64] | None:
    """A field of the data if present."""
    if data is None or name not in data.fields:
        return None
    return data.field(name)


__all__ = ["History", "check_cost_inputs", "field_or_none", "step_length"]
