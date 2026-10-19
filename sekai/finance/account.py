"""Portfolio accounting in weight space: rebalancing, financing, leverage and drift.

Environments hold a portfolio as weights (fractions of equity) over risky
assets, with the remainder in cash. One step of the account:

1. the agent's action becomes target weights, projected onto the account's
   constraints (long-only or long/short, a gross-leverage cap, no positions in
   untradable assets) by `project_weights`;
2. trading from the drifted weights to the targets costs a fraction of equity
   (a `CostModel`);
3. over the period the positions earn the assets' returns, cash earns
   `cash_rate` (or pays `margin_rate` when borrowed), and shorts pay
   `borrow_rate`; `settle` gives the equity growth and the weights drifted by
   the returns, ready for the next rebalance.

All functions broadcast over leading batch dimensions, so the same code runs a
single env or every sub-env of a native vector env at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class AccountConfig:
    """Constraints and financing terms of a trading account.

    Args:
        allow_short: Allow negative weights.
        max_leverage: Cap on gross exposure `sum(|w|)` (1.0 = no leverage).
        cash_rate: Annual interest earned on positive cash.
        margin_rate: Annual interest paid on borrowed cash (default: cash_rate).
        borrow_rate: Annual fee on the value of short positions.
    """

    allow_short: bool = False
    max_leverage: float = 1.0
    cash_rate: float = 0.0
    margin_rate: float | None = None
    borrow_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.max_leverage <= 0:
            raise ValueError(f"max_leverage must be positive, got {self.max_leverage}.")
        if self.borrow_rate < 0:
            raise ValueError(f"borrow_rate must be non-negative, got {self.borrow_rate}.")

    @property
    def effective_margin_rate(self) -> float:
        return self.cash_rate if self.margin_rate is None else self.margin_rate


def project_weights(
    scores: ArrayLike,
    tradable: ArrayLike,
    config: AccountConfig,
    *,
    cash_score: ArrayLike | None = None,
) -> Array:
    """Map raw action values to target risky-asset weights that satisfy the account's rules.

    Long-only: scores (and an optional `cash_score`) are clipped at zero and
    normalised into weights summing to at most `max_leverage` (to exactly
    `max_leverage` when there is no cash score and some score is positive);
    all-zero scores mean all cash. Long/short: values are clipped to
    `[-max_leverage, max_leverage]` and scaled down if gross exposure exceeds
    `max_leverage`. Untradable assets get weight 0 in both cases.

    Args:
        scores: `(..., n)` action values per asset.
        tradable: `(..., n)` bool, False for assets that cannot be held.
        config: The account rules.
        cash_score: `(...)` long-only score of cash (the "stay out" option).

    Returns:
        `(..., n)` target weights.
    """
    a = np.where(np.asarray(tradable, dtype=bool), np.asarray(scores, dtype=np.float64), 0.0)
    lev = config.max_leverage
    if config.allow_short:
        a = np.clip(a, -lev, lev)
        gross = np.abs(a).sum(axis=-1, keepdims=True)
        scale = np.where(gross > lev, lev / np.maximum(gross, 1e-300), 1.0)
        return np.asarray(a * scale, dtype=np.float64)
    a = np.clip(a, 0.0, None)
    total = a.sum(axis=-1, keepdims=True)
    if cash_score is not None:
        total = total + np.clip(np.asarray(cash_score, dtype=np.float64), 0.0, None)[..., None]
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(total > 1e-12, lev * a / np.maximum(total, 1e-300), 0.0)
    return np.asarray(w, dtype=np.float64)


@dataclass(frozen=True)
class Settlement:
    """The outcome of holding `weights` over one period (arrays broadcast over batch dims)."""

    growth: Array  # (...): equity multiplier over the period, net of costs and financing
    drifted: Array  # (..., n): weights at the end of the period, before the next rebalance
    financing: Array  # (...): interest and borrow fees as a fraction of equity (signed)
    wiped_out: NDArray[np.bool_]  # (...): equity fell to zero or below


def settle(
    weights: ArrayLike,
    returns: ArrayLike,
    *,
    cost: ArrayLike,
    dt: float,
    config: AccountConfig,
) -> Settlement:
    """Hold `weights` (set after paying `cost`) over a period with asset `returns`.

    Args:
        weights: `(..., n)` risky weights held over the period.
        returns: `(..., n)` simple returns of the assets over the period (use 0
            for assets that are not held or have no price).
        cost: `(...)` trading cost already paid, as a fraction of equity.
        dt: Period length in years (for financing).
        config: Account financing terms.
    """
    w = np.asarray(weights, dtype=np.float64)
    r = np.nan_to_num(np.asarray(returns, dtype=np.float64))
    cash = 1.0 - w.sum(axis=-1)
    rate = np.where(cash >= 0, config.cash_rate, config.effective_margin_rate)
    financing = cash * rate * dt - config.borrow_rate * dt * np.clip(-w, 0.0, None).sum(axis=-1)
    gross = 1.0 + (w * r).sum(axis=-1) + financing
    c = np.asarray(cost, dtype=np.float64)
    # A cost of the whole account (or more) wipes it out, whatever the market did.
    growth = np.where(c >= 1.0, 0.0, (1.0 - c) * gross)
    wiped = growth <= 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        drifted = np.where(
            wiped[..., None], 0.0, w * (1.0 + r) / np.where(wiped, 1.0, gross)[..., None]
        )
    return Settlement(
        growth=np.asarray(growth, dtype=np.float64),
        drifted=np.asarray(drifted, dtype=np.float64),
        financing=np.asarray(financing, dtype=np.float64),
        wiped_out=np.asarray(wiped),
    )


def turnover(target: ArrayLike, current: ArrayLike) -> Any:
    """`sum(|target - current|)` over the last axis (cash excluded)."""
    return np.abs(np.asarray(target, np.float64) - np.asarray(current, np.float64)).sum(axis=-1)


__all__ = ["AccountConfig", "project_weights", "Settlement", "settle", "turnover"]
