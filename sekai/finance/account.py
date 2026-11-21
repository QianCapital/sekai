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
single env or every sub-env of a vector env at once, and they run on any
`sekai.backend`: NumPy arrays (float64) or torch tensors (their own dtype and
device, differentiable), so a batched environment on a GPU uses them as they are.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from numpy.typing import ArrayLike

from sekai.backend import Array, backend_and_like


@dataclass(frozen=True)
class AccountConfig:
    """Constraints and financing terms of a trading account.

    Args:
        allow_short: Allow negative weights.
        max_leverage: Cap on gross exposure `sum(|w|)` (1.0 = no leverage); finite
            and positive.
        cash_rate: Annual interest earned on positive cash; finite (negative rates
            are allowed).
        margin_rate: Annual interest paid on borrowed cash (default: cash_rate);
            finite.
        borrow_rate: Annual fee on the value of short positions; finite and
            non-negative.

    Raises:
        ValueError: If a numeric field is NaN, infinite or out of its range.
    """

    allow_short: bool = False
    max_leverage: float = 1.0
    cash_rate: float = 0.0
    margin_rate: float | None = None
    borrow_rate: float = 0.0

    def __post_init__(self) -> None:
        rates = {"cash_rate": self.cash_rate, "borrow_rate": self.borrow_rate}
        if self.margin_rate is not None:
            rates["margin_rate"] = self.margin_rate
        for name, value in {"max_leverage": self.max_leverage, **rates}.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}.")
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
    B, like = backend_and_like(scores, tradable, cash_score)
    a = B.where(B.asbool(tradable, like), B.asfloat(scores, like), 0.0)
    lev = config.max_leverage
    if config.allow_short:
        a = B.clip(a, -lev, lev)
        gross = B.sum(B.abs(a), axis=-1, keepdims=True)
        # The floor only matters where the scale is not used; it keeps gradients finite.
        scale = B.where(gross > lev, lev / B.maximum(gross, lev), 1.0)
        return B.asfloat(a * scale, like)
    a = B.clip(a, 0.0, None)
    total = B.sum(a, axis=-1, keepdims=True)
    if cash_score is not None:
        total = total + B.clip(B.asfloat(cash_score, like), 0.0, None)[..., None]
    with B.errstate():
        w = B.where(total > 1e-12, lev * a / B.maximum(total, 1e-12), 0.0)
    return B.asfloat(w, like)


@dataclass(frozen=True)
class Settlement:
    """The outcome of holding `weights` over one period (arrays broadcast over batch dims)."""

    growth: Array  # (...): equity multiplier over the period, net of costs and financing
    drifted: Array  # (..., n): weights at the end of the period, before the next rebalance
    financing: Array  # (...): interest and borrow fees as a fraction of equity (signed)
    wiped_out: Array  # (...): equity fell to zero or below


def settle(
    weights: ArrayLike,
    returns: ArrayLike,
    *,
    cost: ArrayLike,
    dt: float,
    config: AccountConfig,
    holding: ArrayLike | None = None,
) -> Settlement:
    """Hold `weights` (set after paying `cost`) over a period with asset `returns`.

    Args:
        weights: `(..., n)` risky weights held over the period.
        returns: `(..., n)` simple returns of the assets over the period (use 0
            for assets that are not held or have no price).
        cost: `(...)` trading cost already paid, as a fraction of equity.
        dt: Period length in years (for financing).
        config: Account financing terms.
        holding: `(...)` other costs of holding the positions over the period, as a
            fraction of equity (from a `sekai.finance.HoldingCost`, e.g. per-name borrow
            fees); counted in `financing`.
    """
    B, like = backend_and_like(weights, returns, cost, holding)
    w = B.asfloat(weights, like)
    r = B.nan_to_num(B.asfloat(returns, like))
    cash = 1.0 - B.sum(w, axis=-1)
    # A 0-d array branch keeps the rates in the arrays' precision (a float64 torch account
    # too); with NumPy it gives the float64 that two Python floats would.
    rate = B.where(cash >= 0, B.asfloat(config.cash_rate, w), config.effective_margin_rate)
    financing = cash * rate * dt - config.borrow_rate * dt * B.sum(B.clip(-w, 0.0, None), axis=-1)
    if holding is not None:
        financing = financing - B.asfloat(holding, like)
    gross = 1.0 + B.sum(w * r, axis=-1) + financing
    c = B.asfloat(cost, like)
    # A cost of the whole account (or more) wipes it out, whatever the market did.
    growth = B.where(c >= 1.0, 0.0, (1.0 - c) * gross)
    wiped = growth <= 0.0
    with B.errstate():
        drifted = B.where(
            wiped[..., None], 0.0, w * (1.0 + r) / B.where(wiped, 1.0, gross)[..., None]
        )
    return Settlement(
        growth=B.asfloat(growth, like),
        drifted=B.asfloat(drifted, like),
        financing=B.asfloat(financing, like),
        wiped_out=B.asbool(wiped, like),
    )


def turnover(target: ArrayLike, current: ArrayLike) -> Any:
    """`sum(|target - current|)` over the last axis (cash excluded)."""
    B, like = backend_and_like(target, current)
    return B.sum(B.abs(B.asfloat(target, like) - B.asfloat(current, like)), axis=-1)


__all__ = ["AccountConfig", "project_weights", "Settlement", "settle", "turnover"]
