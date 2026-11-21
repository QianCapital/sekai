"""Hard portfolio constraints, enforced by projection.

A policy's raw output becomes a feasible book by projection rather than by a
penalty, so the policy cannot propose a book that breaks the rules: no position in
an untradable or unshortable asset, a gross-exposure cap, per-name caps, and
neutrality to the market (dollar neutrality) or to any exposures, such as betas
or a risk model's factor loadings. `project` maps scores to such a book:

1. untradable assets get 0, as do negative scores of assets that cannot be shorted
   (all of them when `long_only`);
2. the scores are made orthogonal to the constraint columns (a column of ones for
   dollar neutrality and every exposure to neutralise) by one joint least-squares
   projection; neutralising against one column at a time would not converge to
   joint neutrality;
3. they are scaled to the requested gross exposure;
4. for `n_iter` rounds, they are clipped to the per-name bounds, re-neutralised and
   rescaled to the requested gross exposure;
5. a final clip, so the per-name caps and the gross cap hold exactly; neutrality holds
   to the residual the last clip leaves, which shrinks with `n_iter` (basis points
   unless the requested gross is close to `n * max_position`).

The requested gross is reached whenever it is feasible (at least `gross /
max_position` tradable names). Everything broadcasts over leading batch dimensions,
runs on any `sekai.backend`, and is differentiable in the scores and the gross (with
torch), so the projection can sit inside the graph a pathwise trainer backpropagates
through.

Example:
    ```python
    rules = Constraints(max_gross=2.0, max_position=0.05, dollar_neutral=True)
    w = project(scores, rules, gross=1.5, tradable=listed, exposures=betas)
    ```
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sekai.backend import Array, backend_and_like


@dataclass(frozen=True)
class Constraints:
    """The rules a portfolio must satisfy (enforced by `project`).

    Args:
        long_only: No short positions.
        max_gross: Cap on gross exposure `sum(|w|)` (finite, > 0).
        max_position: Cap on each position's size `|w_i|` (finite, > 0), or None.
        dollar_neutral: Net exposure `sum(w)` of zero (long/short only).
        neutral_to: Names of per-asset data fields to be neutral to (`sum(w * x) = 0`
            for each column of each field), such as `"beta"` or a risk model's
            `"loadings"` (long/short only). Environments look them up and pass them
            to `project` as `exposures`.
        n_iter: Rounds of clip / neutralise / rescale (>= 0).
        ridge: Ridge term of the least-squares neutralisation (> 0; keeps it defined
            when exposures are collinear or an asset universe is tiny).

    Raises:
        ValueError: If a value is out of range, or long-only is combined with a
            neutrality constraint (a long-only book cannot be neutral).
    """

    long_only: bool = False
    max_gross: float = 1.0
    max_position: float | None = None
    dollar_neutral: bool = False
    neutral_to: tuple[str, ...] = ()
    n_iter: int = 20
    ridge: float = 1e-4

    def __post_init__(self) -> None:
        object.__setattr__(self, "neutral_to", tuple(self.neutral_to))
        if not (math.isfinite(self.max_gross) and self.max_gross > 0):
            raise ValueError(f"max_gross must be finite and positive, got {self.max_gross!r}.")
        if self.max_position is not None and not (
            math.isfinite(self.max_position) and self.max_position > 0
        ):
            raise ValueError(
                f"max_position must be finite and positive, got {self.max_position!r}."
            )
        if self.n_iter < 0:
            raise ValueError(f"n_iter must be >= 0, got {self.n_iter}.")
        if not (math.isfinite(self.ridge) and self.ridge > 0):
            raise ValueError(f"ridge must be finite and positive, got {self.ridge!r}.")
        if self.long_only and (self.dollar_neutral or self.neutral_to):
            raise ValueError(
                "A long-only book cannot be dollar- or factor-neutral; set long_only=False."
            )

    @property
    def neutral(self) -> bool:
        """Whether any neutrality constraint applies."""
        return self.dollar_neutral or bool(self.neutral_to)


def neutralize(x: Array, columns: Array, *, ridge: float = 1e-4) -> Array:
    """`x` minus its least-squares fit on `columns`: `x - C (C'C + ridge I)^-1 C' x`.

    Args:
        x: `(..., n)` values.
        columns: `(..., n, k)` columns to be orthogonal to.
        ridge: Ridge term (> 0).
    """
    B, like = backend_and_like(x, columns)
    x = B.asfloat(x, like)
    c = B.asfloat(columns, like)
    k = c.shape[-1]
    gram = B.einsum("...nk,...nj->...kj", c, c) + ridge * B.eye(k, like=c)
    coef = B.solve(gram, B.einsum("...nk,...n->...k", c, x)[..., None])[..., 0]
    return x - B.einsum("...nk,...k->...n", c, coef)


def constraint_columns(
    constraints: Constraints,
    tradable: Array,
    exposures: Array | Sequence[Array] | None,
    *,
    like: Array | None = None,
) -> Array | None:
    """The `(..., n, k)` columns a book must be orthogonal to (None if there are none).

    A column of ones for dollar neutrality, then every exposure (`(..., n)` or
    `(..., n, k)` each), zeroed on untradable assets so they do not count. The columns
    take the float dtype of `like` (the scores), else torch's default.
    """
    if not constraints.neutral:
        return None
    B, first = backend_and_like(like, tradable, exposures)
    like = like if like is not None else first
    mask = B.cast(B.asbool(tradable, like), like) if like is not None else B.asfloat(tradable)
    cols = []
    if constraints.dollar_neutral:
        cols.append(mask[..., None])
    if constraints.neutral_to:
        if exposures is None:
            raise ValueError(
                f"The constraints are neutral to {list(constraints.neutral_to)}, so project() "
                "needs their values as `exposures`."
            )
        parts = list(exposures) if isinstance(exposures, (list, tuple)) else [exposures]
        for part in parts:
            e = B.nan_to_num(B.asfloat(part, like) if like is None else B.cast(part, like))
            if e.ndim == mask.ndim:
                e = e[..., None]
            cols.append(e * mask[..., None])
    return B.concat(cols, axis=-1)


def project(
    scores: Array,
    constraints: Constraints,
    *,
    gross: Any = None,
    tradable: Array | None = None,
    shortable: Array | None = None,
    exposures: Array | Sequence[Array] | None = None,
) -> Array:
    """Map scores to the closest-in-direction book that satisfies `constraints`.

    Args:
        scores: `(..., n)` raw scores (the direction of the book).
        constraints: The rules.
        gross: `(...)` requested gross exposure (default: `constraints.max_gross`),
            clipped to `[0, max_gross]`.
        tradable: `(..., n)` bool, False for assets that cannot be held (default: all).
        shortable: `(..., n)` bool, False for assets that cannot be shorted (default:
            all, unless `long_only`).
        exposures: Values of the `neutral_to` fields, `(..., n)` or `(..., n, k)`
            each (a list in the order of `neutral_to`, or one array).

    Returns:
        `(..., n)` weights. Scores that are all zero (after removing untradable and
        unshortable names) have no direction: they give an empty book, with a zero
        gradient.
    """
    B, like = backend_and_like(scores, gross, tradable, shortable, exposures)
    x = B.asfloat(scores, like)
    ok = B.bools(tuple(x.shape), True, like=x) if tradable is None else B.asbool(tradable, x)
    if constraints.long_only:
        can_short = B.bools(tuple(x.shape), False, like=x)
    elif shortable is None:
        can_short = ok
    else:
        can_short = ok & B.asbool(shortable, x)
    cap = constraints.max_position
    bound = B.full(tuple(x.shape), math.inf if cap is None else cap, like=x)
    hi = B.where(ok, bound, 0.0)
    lo = B.where(can_short, -bound, 0.0)
    target = B.clip(
        B.full(tuple(x.shape[:-1]), constraints.max_gross, like=x)
        if gross is None
        else B.asfloat(gross, x),
        0.0,
        constraints.max_gross,
    )[..., None]
    cols = constraint_columns(constraints, ok, exposures, like=x)

    def neutral(v: Array) -> Array:
        return v if cols is None else neutralize(v, cols, ridge=constraints.ridge)

    def to_gross(v: Array) -> Array:
        # All-zero scores have no direction: the book is empty and its gradient zero (a
        # floored size would instead multiply gradients by 1 / floor in every round).
        size = B.sum(B.abs(v), axis=-1, keepdims=True)
        some = size > 1e-12
        return v * B.where(some, target / B.where(some, size, 1.0), 0.0)

    w = B.clip(B.where(ok, x, 0.0), lo, hi) if constraints.long_only else B.where(ok, x, 0.0)
    w = to_gross(neutral(B.where(can_short | (w > 0), w, 0.0)))
    if cap is None and cols is None:  # nothing a clip could change: scaling kept the signs
        return w
    for _ in range(constraints.n_iter):
        w = to_gross(neutral(B.clip(w, lo, hi)))
    return B.clip(w, lo, hi)


__all__ = ["Constraints", "project", "neutralize", "constraint_columns"]
