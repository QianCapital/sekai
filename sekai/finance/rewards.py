"""Reward functions for trading environments.

What an agent optimises is a modelling decision as important as the market
itself: raw P&L, log growth (the Kelly criterion), a risk-adjusted return, or
an online Sharpe ratio. Trading envs take a `reward=` argument, a
`RewardFunction` that turns each step's `StepOutcome` into a reward.

Reward functions compose arithmetically:

```python
reward = LogReturn() - 0.5 * DrawdownPenalty() - 1e-3 * TurnoverPenalty()
```

They are vectorised: an outcome's fields may be arrays with one entry per
sub-env (vector envs), and stateful rewards (DifferentialSharpe,
DrawdownPenalty) keep per-env state, reset selectively with `reset(mask)`.
They run on any `sekai.backend`: with torch tensors the reward is differentiable,
and `detach_state()` cuts a stateful reward's history from the autograd graph (for
truncated backpropagation through time).
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from sekai.backend import Array, backend_and_like, is_tensor


@dataclass(frozen=True)
class StepOutcome:
    """What happened to the account over one step.

    Fields are floats for a single env, or arrays of shape `(num_envs,)`.

    Attributes:
        log_return: Log growth of equity over the step, net of all costs.
        simple_return: `exp(log_return) - 1`.
        equity: Equity after the step, relative to the start of the episode.
        turnover: Sum of absolute weight changes traded at the start of the step.
        cost: Trading cost paid, as a fraction of equity.
        pnl: Change in equity over the step, in units of the initial equity (exact, also
            on a wipe-out), or None if the environment did not supply the equity before
            the step.
        weights: The weights held over the step (after trading), `(..., n)`, or None.
        variance: The ex-ante variance of the step's return under the environment's
            risk model (`w' Σ w` for a factor model), or None without one.
    """

    log_return: Any
    simple_return: Any
    equity: Any
    turnover: Any
    cost: Any
    pnl: Any = None
    weights: Any = None
    variance: Any = None

    @classmethod
    def from_growth(
        cls,
        growth: ArrayLike,
        *,
        equity: ArrayLike,
        turnover: ArrayLike,
        cost: ArrayLike,
        previous_equity: ArrayLike | None = None,
        weights: Any = None,
        variance: Any = None,
    ) -> StepOutcome:
        """Build from the equity multiplier of the step (floored, so a wipe-out is finite).

        Args:
            growth: Equity multiplier of the step (0 or below on a wipe-out).
            equity: Equity after the step, relative to the start of the episode.
            turnover: Sum of absolute weight changes traded.
            cost: Trading cost paid, as a fraction of equity.
            previous_equity: Equity before the step. When given, `pnl` is
                `equity - previous_equity`; environments should always pass it, since
                the equity lost on a wipe-out cannot be recovered from `equity` alone.
            weights: The weights held over the step, passed through.
            variance: The step's ex-ante variance, passed through.
        """
        B, like = backend_and_like(growth, equity, turnover, cost, previous_equity)
        g = B.maximum(B.asfloat(growth, like), 1e-12)
        after = B.asfloat(equity, like)
        return cls(
            log_return=B.log(g),
            simple_return=g - 1.0,
            equity=after,
            turnover=B.asfloat(turnover, like),
            cost=B.asfloat(cost, like),
            pnl=None if previous_equity is None else after - B.asfloat(previous_equity, like),
            weights=weights,
            variance=variance,
        )


class RewardFunction(abc.ABC):
    """Maps a StepOutcome to a reward (float, or an array for batched outcomes)."""

    @abc.abstractmethod
    def __call__(self, outcome: StepOutcome) -> Any: ...

    def reset(self, mask: Any = None) -> None:  # noqa: B027 (optional hook)
        """Start new episodes: all of them (mask=None) or the sub-envs where mask is True."""

    def detach_state(self) -> None:  # noqa: B027 (optional hook)
        """Cut any state carried between steps from the autograd graph (torch).

        Called by batched environments when a trainer truncates backpropagation
        through time; stateless rewards have nothing to do.
        """

    # Arithmetic ------------------------------------------------------------

    def __add__(self, other: RewardFunction | float) -> RewardFunction:
        return _Combination([(1.0, self), (1.0, _as_reward(other))])

    def __radd__(self, other: float) -> RewardFunction:
        return _Combination([(1.0, _as_reward(other)), (1.0, self)])

    def __sub__(self, other: RewardFunction | float) -> RewardFunction:
        return _Combination([(1.0, self), (-1.0, _as_reward(other))])

    def __rsub__(self, other: float) -> RewardFunction:
        return _Combination([(1.0, _as_reward(other)), (-1.0, self)])

    def __mul__(self, scale: float) -> RewardFunction:
        if not isinstance(scale, (int, float, np.integer, np.floating)):
            return NotImplemented
        return _Combination([(float(scale), self)])

    __rmul__ = __mul__

    def __neg__(self) -> RewardFunction:
        return _Combination([(-1.0, self)])


class _Constant(RewardFunction):
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def __call__(self, outcome: StepOutcome) -> Any:
        B, like = backend_and_like(outcome.log_return)
        return B.zeros_like(B.asfloat(outcome.log_return, like)) + self.value

    def __repr__(self) -> str:
        return repr(self.value)


def _as_reward(x: RewardFunction | float) -> RewardFunction:
    if isinstance(x, RewardFunction):
        return x
    if isinstance(x, (int, float, np.integer, np.floating)):
        return _Constant(float(x))
    raise TypeError(f"Cannot combine a reward function with {x!r}.")


class _Combination(RewardFunction):
    terms: list[tuple[float, RewardFunction]]

    def __init__(self, terms: Sequence[tuple[float, RewardFunction]]) -> None:
        flat: list[tuple[float, RewardFunction]] = []
        for weight, fn in terms:
            if isinstance(fn, _Combination):
                flat.extend((weight * w, f) for w, f in fn.terms)
            else:
                flat.append((weight, fn))
        self.terms = flat

    def __call__(self, outcome: StepOutcome) -> Any:
        B, like = backend_and_like(outcome.log_return)
        total: Any = 0.0
        for weight, fn in self.terms:
            total = total + weight * B.asfloat(fn(outcome), like)
        return total

    def reset(self, mask: Any = None) -> None:
        for _, fn in self.terms:
            fn.reset(mask)

    def detach_state(self) -> None:
        for _, fn in self.terms:
            fn.detach_state()

    def __repr__(self) -> str:
        out = ""
        for i, (weight, fn) in enumerate(self.terms):
            sign = "-" if weight < 0 else "+"
            size = abs(weight)
            term = repr(fn) if size == 1.0 else f"{size:g} * {fn!r}"
            out += (f"-{term}" if sign == "-" else term) if i == 0 else f" {sign} {term}"
        return out


class _Stateful(RewardFunction):
    """Helper base: per-env state arrays created lazily with the outcome's shape."""

    _fields: tuple[tuple[str, float], ...] = ()

    def __init__(self) -> None:
        self._state: dict[str, Array] | None = None

    def _get(self, like: Array) -> dict[str, Array]:
        """The state, (re)created to match the shape and backend of `like`."""
        B = backend_and_like(like)[0]
        shape = tuple(like.shape)
        state = self._state
        current = None if state is None else next(iter(state.values()))
        if (
            state is None
            or current is None
            or tuple(current.shape) != shape
            or is_tensor(current) != is_tensor(like)
        ):
            state = {name: B.full(shape, init, like=like) for name, init in self._fields}
            self._state = state
        return state

    def reset(self, mask: Any = None) -> None:
        if self._state is None:
            return
        if mask is None:
            self._state = None
            return
        state = self._state
        B, like = backend_and_like(*state.values())
        m = B.asbool(mask, like)
        shape = tuple(next(iter(state.values())).shape)
        try:
            fits = np.broadcast_shapes(tuple(m.shape), shape) == shape
        except ValueError:
            fits = False
        if not fits:  # (as a boolean index into the state, the mask raised IndexError in 0.2)
            raise IndexError(
                f"reset() got a mask of shape {tuple(m.shape)} for a state of shape {shape}."
            )
        # Out of place, so a state that is part of an autograd graph stays valid.
        self._state = {name: B.where(m, init, state[name]) for name, init in self._fields}

    def detach_state(self) -> None:
        if self._state is not None:
            B = backend_and_like(*self._state.values())[0]
            self._state = {name: B.detach(value) for name, value in self._state.items()}


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


class LogReturn(RewardFunction):
    """Log growth of equity. Summed over an episode: log(final / initial equity)."""

    def __call__(self, outcome: StepOutcome) -> Any:
        return outcome.log_return

    def __repr__(self) -> str:
        return "LogReturn()"


class SimpleReturn(RewardFunction):
    """Percentage return of the step (P&L as a fraction of current equity)."""

    def __call__(self, outcome: StepOutcome) -> Any:
        return outcome.simple_return

    def __repr__(self) -> str:
        return "SimpleReturn()"


class ProfitAndLoss(RewardFunction):
    """Change in equity, in units of the initial equity (summed: final equity - 1).

    Uses `StepOutcome.pnl`, which every built-in environment supplies, so the step that
    wipes the account out is rewarded with the whole equity it lost. For an outcome
    without `pnl` the change is recovered from the equity after the step, which is only
    possible while that equity is positive; a wipe-out then raises `ValueError`.
    """

    def __call__(self, outcome: StepOutcome) -> Any:
        B, like = backend_and_like(outcome.equity, outcome.pnl)
        if outcome.pnl is not None:
            return B.asfloat(outcome.pnl, like)
        equity = B.asfloat(outcome.equity, like)
        if B.item(B.any(equity <= 0.0)):
            raise ValueError(
                "ProfitAndLoss cannot recover the equity lost in a wipe-out from the equity "
                "after it; build the outcome with StepOutcome.from_growth(..., "
                "previous_equity=...) so that it carries `pnl`."
            )
        r = B.asfloat(outcome.simple_return, like)
        return equity * r / (1.0 + r)

    def __repr__(self) -> str:
        return "ProfitAndLoss()"


class MeanVariance(RewardFunction):
    """`r - (risk_aversion / 2) * r^2` on the log return: a per-step mean-variance utility."""

    def __init__(self, risk_aversion: float = 1.0) -> None:
        if risk_aversion < 0:
            raise ValueError("risk_aversion must be non-negative.")
        self.risk_aversion = float(risk_aversion)

    def __call__(self, outcome: StepOutcome) -> Any:
        B, like = backend_and_like(outcome.log_return)
        r = B.asfloat(outcome.log_return, like)
        return r - 0.5 * self.risk_aversion * r * r

    def __repr__(self) -> str:
        return f"MeanVariance({self.risk_aversion:g})"


class TurnoverPenalty(RewardFunction):
    """The step's turnover; subtract it to discourage over-trading beyond explicit costs."""

    def __call__(self, outcome: StepOutcome) -> Any:
        return outcome.turnover

    def __repr__(self) -> str:
        return "TurnoverPenalty()"


class DrawdownPenalty(_Stateful):
    """How much the drawdown deepened this step (0 when equity is at or recovering to a peak).

    The drawdown is `1 - equity / running_peak`; summed over an episode the
    penalty is the total amount by which drawdowns deepened.
    """

    _fields = (("peak", 1.0), ("drawdown", 0.0))

    def __call__(self, outcome: StepOutcome) -> Any:
        B, like = backend_and_like(outcome.equity)
        equity = B.asfloat(outcome.equity, like)
        s = self._get(equity)
        peak = B.maximum(s["peak"], equity)
        dd = B.where(peak > 0, 1.0 - equity / B.where(peak > 0, peak, 1.0), 1.0)
        penalty = B.maximum(dd - s["drawdown"], 0.0)
        s["peak"], s["drawdown"] = peak, dd
        return penalty

    def __repr__(self) -> str:
        return "DrawdownPenalty()"


class DifferentialSharpe(_Stateful):
    """The differential Sharpe ratio of Moody & Saffell (2001).

    An online, per-step reward whose sum approximately maximises the Sharpe
    ratio of the step returns: with exponential moving estimates `A` (mean)
    and `B` (second moment) of the returns,

    `D_t = (B dA - A dB / 2) / (B - A^2)^(3/2)`,  `dA = R_t - A`,  `dB = R_t^2 - B`.

    It is 0 until the variance estimate becomes positive.

    Args:
        eta: Adaptation rate of the moving estimates (about 1 / window length).
        initial_variance: Starting value of the second-moment estimate `B` (a prior on
            the variance of step returns, e.g. `0.01**2` for 1% daily moves), which makes
            the reward informative from the first step; 0 by default.
    """

    _fields: tuple[tuple[str, float], ...] = (("A", 0.0), ("B", 0.0))
    initial_variance: float = 0.0  # the default of rewards pickled by sekai 0.2

    def __init__(self, eta: float = 0.01, *, initial_variance: float = 0.0) -> None:
        if not 0.0 < eta < 1.0:
            raise ValueError(f"eta must be in (0, 1), got {eta}.")
        if not (np.isfinite(initial_variance) and initial_variance >= 0):
            raise ValueError(f"initial_variance must be finite and >= 0, got {initial_variance!r}.")
        super().__init__()
        self.eta = float(eta)
        self.initial_variance = float(initial_variance)
        self._fields = (("A", 0.0), ("B", self.initial_variance))

    def __call__(self, outcome: StepOutcome) -> Any:
        B, like = backend_and_like(outcome.simple_return)
        r = B.asfloat(outcome.simple_return, like)
        s = self._get(r)
        a, b = s["A"], s["B"]
        d_a, d_b = r - a, r * r - b
        var = b - a * a
        v = B.maximum(var, 1e-12)
        ratio = (b * d_a - 0.5 * a * d_b) / (v * B.sqrt(v))  # v**1.5, identical for any shape
        d = B.where(var > 1e-12, ratio, 0.0)
        s["A"], s["B"] = a + self.eta * d_a, b + self.eta * d_b
        return d

    def __repr__(self) -> str:
        extra = f", initial_variance={self.initial_variance:g}" if self.initial_variance else ""
        return f"DifferentialSharpe(eta={self.eta:g}{extra})"


class PortfolioVariance(RewardFunction):
    """The step's ex-ante portfolio variance under the environment's risk model.

    Subtract it for a mean-variance objective that penalises the risk the portfolio
    takes rather than the noise of realised returns, e.g.
    `SimpleReturn() - 2.0 * PortfolioVariance()`. Needs an environment with a risk
    model (`StepOutcome.variance`), such as a `BatchMarketEnv` with `risk=`.
    """

    def __call__(self, outcome: StepOutcome) -> Any:
        if outcome.variance is None:
            raise ValueError(
                "PortfolioVariance needs the step's ex-ante variance (StepOutcome.variance), "
                "which an environment computes only with a risk model, e.g. "
                "BatchMarketEnv(..., risk=FactorRiskModel(...))."
            )
        return outcome.variance

    def __repr__(self) -> str:
        return "PortfolioVariance()"


__all__ = [
    "StepOutcome",
    "RewardFunction",
    "LogReturn",
    "SimpleReturn",
    "ProfitAndLoss",
    "MeanVariance",
    "TurnoverPenalty",
    "DrawdownPenalty",
    "DifferentialSharpe",
    "PortfolioVariance",
]
