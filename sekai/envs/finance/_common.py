"""Plumbing shared by the finance environments: price sources, history windows, cost
context and parameter checks."""

from __future__ import annotations

import copy
import sys
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sekai.envs.finance.market import validate_prices
from sekai.error import ResetRequired
from sekai.finance.data import MarketData
from sekai.finance.sampling import (
    EpisodeSampler,
    as_sampler,
    resolve_start,
    sample_start,
)


def check_sources(
    env: str, *, data: Any, model: Any, prices: Any = None, takes_prices: bool = True
) -> None:
    """Check the price source given to a finance env: at most one, each of the right kind.

    `prices=` takes raw historical prices (arrays, pandas; Trading-v0 and Portfolio-v0
    only), `data=` a MarketData, and `model=` a market model to simulate.

    Raises:
        TypeError: For a MarketData passed as `prices=`, or anything else as `data=`.
        ValueError: When more than one source is given.
    """
    if isinstance(prices, MarketData):
        raise TypeError(
            f"{env}: pass a MarketData as data=, not prices= (prices= takes a price array, "
            "a pandas DataFrame or a Series)."
        )
    if data is not None and not isinstance(data, MarketData):
        hint = (
            "pass a price array or pandas object as prices=, or build a MarketData"
            if takes_prices
            else "build one with MarketData(prices, ...) or MarketData.from_pandas(frame)"
        )
        raise TypeError(f"{env}: data= takes a MarketData, got {type(data).__name__}; {hint}.")
    given = [
        f"{name}="
        for name, value in (("prices", prices), ("data", data), ("model", model))
        if value is not None
    ]
    if len(given) > 1:
        options = (
            "historical prices= or data=, or a model= to simulate"
            if takes_prices
            else "historical data= or a model= to simulate"
        )
        raise ValueError(f"{env} got {' and '.join(given)}; pass one source: {options}.")


def dated_pandas(prices: Any) -> MarketData | None:
    """A pandas Series or DataFrame indexed by dates as a MarketData that keeps the dates.

    None for anything else (arrays, or pandas with another index, whose rows are then
    numbered from 0 like an array's).
    """
    pd = sys.modules.get("pandas")  # pandas objects exist only once pandas is imported
    if pd is None or not isinstance(prices, (pd.Series, pd.DataFrame)):
        return None
    if not isinstance(prices.index, pd.DatetimeIndex):
        return None
    frame = prices.to_frame() if isinstance(prices, pd.Series) else prices
    return MarketData.from_pandas(frame)


_BOUNDS = {"": "", ">0": " > 0", ">=0": " >= 0"}


def _in_bounds(values: NDArray[np.float64], bound: str) -> bool:
    ok = np.isfinite(values)
    if bound == ">0":
        ok &= values > 0
    elif bound == ">=0":
        ok &= values >= 0
    return bool(np.all(ok))


def check_number(name: str, value: Any, bound: str = "") -> float:
    """`value` as a finite float, also `> 0` (bound ">0") or `>= 0` (bound ">=0").

    Raises:
        ValueError: Naming the parameter, for a value that is not a finite number in range.
    """
    try:
        out = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        out = np.full(1, np.nan)
    if out.ndim == 0 and _in_bounds(out, bound):
        return float(out)
    raise ValueError(f"{name} must be a finite number{_BOUNDS[bound]}, got {value!r}.")


def check_numbers(name: str, values: Any, bound: str = "") -> NDArray[np.float64]:
    """`values` (a scalar or an array) as float64, each finite and within `bound`.

    Raises:
        ValueError: Naming the parameter, for values that are not finite numbers in range.
    """
    try:
        out = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        out = np.full(1, np.nan)
    if _in_bounds(out, bound):
        return out
    raise ValueError(f"{name} must hold finite numbers{_BOUNDS[bound]}, got {values!r}.")


def episode_over() -> ResetRequired:
    """The error for a step() after the episode ended (and before the next reset())."""
    return ResetRequired(
        "The episode is over (it terminated or was truncated); call reset() before stepping again."
    )


class History:
    """A price history (array, pandas object or MarketData) that episodes are cut from.

    Handles validation, the sampler, and `reset(options={"start": ...})`.

    Args:
        source: Prices (`(T,)` or `(T, n)`: an array, or a pandas Series or DataFrame) or a
            MarketData. Pandas input indexed by dates (a `DatetimeIndex`) becomes a
            MarketData with those timestamps (and column names as assets); with any other
            index its rows are numbered from 0, as an array's are.
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
        dated = source if isinstance(source, MarketData) else dated_pandas(source)
        if dated is not None:
            data = dated
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
                pick = (
                    "pick one with data.select([asset])"
                    if isinstance(source, MarketData)
                    else "pass a single column, frame[name]"
                )
                raise ValueError(
                    f"This env trades one asset but the history has {data.n_assets}; {pick}."
                )
        else:
            assert not isinstance(source, MarketData)  # a MarketData is its own dated form
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


__all__ = [
    "History",
    "check_cost_inputs",
    "check_number",
    "check_numbers",
    "check_sources",
    "dated_pandas",
    "episode_over",
    "field_or_none",
    "step_length",
]
