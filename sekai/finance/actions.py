"""Action maps: how a policy's raw action becomes target portfolio weights.

What an agent controls is a modelling decision separate from the market and the
accounting: scores normalised into a long-only allocation, target weights, scores
plus a leverage knob projected onto hard constraints, changes to the current book.
A batched market environment (`sekai.envs.finance.BatchMarketEnv(action=...)`) takes
an `ActionMap`, which defines the action space and maps each action, given the
state of the account (`ActionContext`), to feasible weights per slot.

- `LongOnlyScores` and `TargetWeights` are the long-only and long/short actions of
  `Market-v0`.
- `ScoresLeverage`, `TanhWeights` and `TradeDelta` map unbounded raw actions through
  `sekai.finance.constraints.project` (gross and per-name caps, dollar and factor
  neutrality, unshortable names), the parameterisations used for long/short equity
  books; each has an inverse, `from_target`, so baselines can express "this book, at
  this fraction of the maximum gross" in any of them.

Every map runs on any `sekai.backend` and is differentiable in the action (torch),
so a pathwise trainer backpropagates through it.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

import numpy as np

from sekai.backend import Array, Backend
from sekai.finance.account import AccountConfig, project_weights
from sekai.finance.constraints import Constraints, project
from sekai.spaces.box import Box


@dataclass(frozen=True)
class ActionContext:
    """The state of every lane's account that an action map may look at.

    Attributes:
        backend: The backend the arrays are on.
        tradable: `(B, S)` bool, the slots whose assets can be traded now.
        weights: `(B, S)` current weights (before trading).
        shortable: `(B, S)` bool, the slots that may be held short, or None (all).
        exposures: Values of per-asset fields at the decision, `(B, S, ...)` each, for
            the fields the map asked for (`ActionMap.fields`, e.g. those a
            `Constraints.neutral_to` names).
    """

    backend: Backend
    tradable: Array
    weights: Array
    shortable: Array | None = None
    exposures: dict[str, Array] | None = None

    def exposure_list(self, names: tuple[str, ...]) -> list[Array] | None:
        """The exposures named, in order (None if there are none)."""
        if not names:
            return None
        found = self.exposures or {}
        missing = [n for n in names if n not in found]
        if missing:
            raise KeyError(f"The action context has no exposures {missing}.")
        return [found[n] for n in names]


class ActionMap(abc.ABC):
    """Maps raw actions `(B, *action_shape)` to target weights `(B, S)`."""

    #: Per-asset data fields the map reads (looked up into `ActionContext.exposures`).
    fields: tuple[str, ...] = ()

    @abc.abstractmethod
    def space(self, n_slots: int) -> Box:
        """The action space of one lane with `n_slots` asset slots."""

    @abc.abstractmethod
    def __call__(self, action: Array, ctx: ActionContext) -> Array:
        """Target weights for a batch of actions."""

    def from_target(self, scores: Array, gross_fraction: Any, ctx: ActionContext) -> Array:
        """A raw action that yields (approximately) the book of `scores` at
        `gross_fraction` of the maximum gross exposure.

        Raises:
            NotImplementedError: If the map has no inverse.
        """
        raise NotImplementedError(f"{type(self).__name__} has no inverse map.")


def _check_leverage(max_leverage: float) -> float:
    value = float(max_leverage)
    if not (np.isfinite(value) and value > 0):
        raise ValueError(f"max_leverage must be finite and positive, got {max_leverage!r}.")
    return value


class LongOnlyScores(ActionMap):
    """Non-negative scores normalised into weights (Market-v0's long-only action).

    The action is `(S + 1,)` in `[0, 1]`: one score per slot and, last, a score for
    cash (the "stay out" option); scores are clipped at 0 and normalised to sum to
    `max_leverage` together with cash's share. All-zero scores mean all cash. Without
    `cash`, the action is `(S,)` and the weights always sum to `max_leverage` (unless
    every score is zero).

    Args:
        max_leverage: Gross exposure of a fully invested book (finite, > 0).
        cash: Include the cash score.
    """

    def __init__(self, max_leverage: float = 1.0, *, cash: bool = True) -> None:
        self.max_leverage = _check_leverage(max_leverage)
        self.cash = bool(cash)
        self._config = AccountConfig(max_leverage=self.max_leverage)

    def space(self, n_slots: int) -> Box:
        return Box(0.0, 1.0, shape=(n_slots + int(self.cash),), dtype=np.float32)

    def __call__(self, action: Array, ctx: ActionContext) -> Array:
        if self.cash:
            return project_weights(
                action[..., :-1], ctx.tradable, self._config, cash_score=action[..., -1]
            )
        return project_weights(action, ctx.tradable, self._config)

    def __repr__(self) -> str:
        return f"LongOnlyScores(max_leverage={self.max_leverage:g}, cash={self.cash})"


class TargetWeights(ActionMap):
    """Target weights in `[-1, 1]` per slot, scaled down to a gross cap (Market-v0's
    long/short action).

    Args:
        max_leverage: Cap on gross exposure (finite, > 0).
    """

    def __init__(self, max_leverage: float = 1.0) -> None:
        self.max_leverage = _check_leverage(max_leverage)
        self._config = AccountConfig(allow_short=True, max_leverage=self.max_leverage)

    def space(self, n_slots: int) -> Box:
        return Box(-1.0, 1.0, shape=(n_slots,), dtype=np.float32)

    def __call__(self, action: Array, ctx: ActionContext) -> Array:
        tradable = ctx.tradable
        if ctx.shortable is not None:
            B = ctx.backend
            action = B.where(ctx.shortable | (action > 0), action, 0.0)
        return project_weights(action, tradable, self._config)

    def __repr__(self) -> str:
        return f"TargetWeights(max_leverage={self.max_leverage:g})"


class _Projected(ActionMap):
    """Shared parts of the maps that project onto `Constraints`."""

    def __init__(self, constraints: Constraints | None = None, *, bound: float = 10.0) -> None:
        self.constraints = constraints if constraints is not None else Constraints()
        if not (np.isfinite(bound) and bound > 0):
            raise ValueError(f"bound must be finite and positive, got {bound!r}.")
        self.bound = float(bound)
        self.fields = self.constraints.neutral_to

    def _project(self, scores: Array, gross: Array, ctx: ActionContext) -> Array:
        return project(
            scores,
            self.constraints,
            gross=gross,
            tradable=ctx.tradable,
            shortable=ctx.shortable,
            exposures=ctx.exposure_list(self.constraints.neutral_to),
        )

    def _space(self, size: int) -> Box:
        # The bound only limits what sample() draws (and what libraries that need finite
        # bounds see); actions are not clipped, the map decodes any real values.
        return Box(-self.bound, self.bound, shape=(size,), dtype=np.float32)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.constraints!r})"


def _logit(B: Backend, p: Array) -> Array:
    q = B.clip(p, 1e-6, 1.0 - 1e-6)
    return B.log(q) - B.log(1.0 - q)


class ScoresLeverage(_Projected):
    """`S` raw scores and a leverage logit: gross = `max_gross * sigmoid(logit)`.

    The scores give the book's direction and the logit how much risk to run, so the
    policy can de-risk in stressed markets; both go through `project`.

    Args:
        constraints: The rules of the book (default: `Constraints()`).
        bound: Bound of the action space (for sampling only; actions are not clipped).
    """

    def space(self, n_slots: int) -> Box:
        return self._space(n_slots + 1)

    def __call__(self, action: Array, ctx: ActionContext) -> Array:
        B = ctx.backend
        gross = self.constraints.max_gross * B.sigmoid(action[..., -1])
        return self._project(action[..., :-1], gross, ctx)

    def from_target(self, scores: Array, gross_fraction: Any, ctx: ActionContext) -> Array:
        B = ctx.backend
        g = B.asfloat(gross_fraction, scores) + B.zeros(tuple(scores.shape[:-1]), like=scores)
        return B.concat([scores, _logit(B, g)[..., None]], axis=-1)


class TanhWeights(_Projected):
    """`S` raw values bounded by `cap * tanh(a)`; gross = `min(max_gross, sum |w|)`.

    `cap` is the constraints' `max_position`, or `max_gross` without one.

    Args:
        constraints: The rules of the book (default: `Constraints()`).
        bound: Bound of the action space (for sampling only).
    """

    def _cap(self) -> float:
        c = self.constraints
        return c.max_gross if c.max_position is None else c.max_position

    def space(self, n_slots: int) -> Box:
        return self._space(n_slots)

    def __call__(self, action: Array, ctx: ActionContext) -> Array:
        B = ctx.backend
        raw = self._cap() * B.tanh(action)
        gross = B.minimum(
            B.sum(B.abs(B.where(ctx.tradable, raw, 0.0)), axis=-1), self.constraints.max_gross
        )
        return self._project(raw, gross, ctx)

    def from_target(self, scores: Array, gross_fraction: Any, ctx: ActionContext) -> Array:
        B = ctx.backend
        g = B.asfloat(gross_fraction, scores) * self.constraints.max_gross
        w = self._project(scores, g, ctx)
        x = B.clip(w / self._cap(), -0.999, 0.999)
        return 0.5 * (B.log1p(x) - B.log1p(-x))  # atanh


class TradeDelta(_Projected):
    """`S` score changes added to the current book, and a leverage logit; zero holds.

    The scores are the current weights normalised to a mean absolute value of 1,
    plus the action; the map is then that of `ScoresLeverage`. Holding still is the
    zero action, which makes low turnover easy to learn.

    Args:
        constraints: The rules of the book (default: `Constraints()`).
        bound: Bound of the action space (for sampling only).
    """

    def space(self, n_slots: int) -> Box:
        return self._space(n_slots + 1)

    @staticmethod
    def _hold(B: Backend, weights: Array) -> Array:
        return weights / (B.mean(B.abs(weights), axis=-1, keepdims=True) + 1e-8)

    def __call__(self, action: Array, ctx: ActionContext) -> Array:
        B = ctx.backend
        scores = self._hold(B, ctx.weights) + action[..., :-1]
        gross = self.constraints.max_gross * B.sigmoid(action[..., -1])
        return self._project(scores, gross, ctx)

    def from_target(self, scores: Array, gross_fraction: Any, ctx: ActionContext) -> Array:
        B = ctx.backend
        g = B.asfloat(gross_fraction, scores) + B.zeros(tuple(scores.shape[:-1]), like=scores)
        delta = scores - self._hold(B, ctx.weights)
        return B.concat([delta, _logit(B, g)[..., None]], axis=-1)


__all__ = [
    "ActionContext",
    "ActionMap",
    "LongOnlyScores",
    "TargetWeights",
    "ScoresLeverage",
    "TanhWeights",
    "TradeDelta",
]
