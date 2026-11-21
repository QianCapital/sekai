"""Reproducible evaluation of trading policies: tasks, baselines and comparisons.

A learned policy means little until it beats simple rules on the same episodes, and a
backtest means little until the number of things tried is accounted for. This module
fixes both:

- a `Task` is an environment and a fixed set of evaluation episodes (seeds, or the
  windows of a test period), so every policy is scored on the same scenarios;
- `evaluate(policy, task)` runs a policy on them and returns an `Evaluation`: every
  episode's equity curve, pooled per-period returns, a `PerformanceSummary`, the
  probabilistic Sharpe ratio and the average turnover;
- `compare(policies, task)` evaluates several (your agent, its ablations, the
  baselines) and reports them side by side, with each one's deflated Sharpe ratio,
  counting every policy compared as a trial;
- the baselines (`Cash`, `EqualWeight`, `InverseVolatility`, `Momentum`) express a
  target book that `target_action` turns into an action of `Market-v0`, `Market-v1`
  and `Portfolio-v0` / `-v1`, whatever the action map.

A policy is any callable `policy(observation, env) -> action`, where `env` is the
unwrapped environment (for `env.action_mask()` and the like); wrap a trained model in a
lambda, e.g. `lambda obs, env: model.predict(obs, deterministic=True)[0]`.

Example:
    ```python
    task = Task("market", "Market-v1", episodes=50)
    print(compare({"equal weight": EqualWeight(), "momentum": Momentum(20)}, task))
    ```
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sekai.finance.metrics import (
    PerformanceSummary,
    deflated_sharpe_ratio,
    performance_summary,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)

Policy = Callable[[Any, Any], Any]


# ---------------------------------------------------------------------------
# Tasks and evaluation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    """An environment and the episodes every policy is evaluated on.

    Episode `k` resets with seed `seed + k`, so evaluations are reproducible and never
    share seeds with training (keep training seeds below `seed`). For a historical test
    period, pass `sampler="sequential"` in `kwargs` and as many episodes as there are
    windows: each episode then replays the next window of the period.

    Args:
        name: A label for reports.
        env_id: A registered environment ID.
        kwargs: Keyword arguments for `sekai.make`.
        episodes: Number of evaluation episodes (>= 1).
        seed: Seed of the first episode.
        periods_per_year: Steps per year, for annualised metrics.
    """

    name: str
    env_id: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    episodes: int = 100
    seed: int = 10_000
    periods_per_year: float = 252.0

    def __post_init__(self) -> None:
        if self.episodes < 1:
            raise ValueError(f"episodes must be >= 1, got {self.episodes}.")

    def make(self) -> Any:
        """A fresh environment for this task."""
        import sekai

        return sekai.make(self.env_id, **dict(self.kwargs))


@dataclass(frozen=True)
class Evaluation:
    """How one policy did on a task's episodes.

    Attributes:
        name: The policy's label.
        task: The task.
        equity: One equity curve per episode (starting at 1, one value per step).
        rewards: Each episode's total reward.
        turnover: Average turnover per step (`info["turnover"]`), NaN if not reported.
        periods_per_year: Steps per year.
    """

    name: str
    task: Task
    equity: tuple[NDArray[np.float64], ...]
    rewards: NDArray[np.float64]
    turnover: float
    periods_per_year: float

    @property
    def returns(self) -> NDArray[np.float64]:
        """Every step's simple return of equity, all episodes pooled."""
        parts = []
        for curve in self.equity:
            before = np.concatenate([[1.0], curve[:-1]])  # equity starts each episode at 1
            parts.append(curve / before - 1.0)
        return np.concatenate(parts) if parts else np.zeros(0)

    @property
    def episode_returns(self) -> NDArray[np.float64]:
        """Each episode's total return (final equity - 1)."""
        return np.array([curve[-1] - 1.0 if curve.size else 0.0 for curve in self.equity])

    @property
    def max_drawdowns(self) -> NDArray[np.float64]:
        """Each episode's maximum drawdown (a fraction of the peak)."""
        out = []
        for curve in self.equity:
            path = np.concatenate([[1.0], curve])
            peak = np.maximum.accumulate(path)
            out.append(float(np.max(1.0 - path / peak)))
        return np.asarray(out)

    @property
    def summary(self) -> PerformanceSummary:
        """Performance of the pooled returns (episodes chained: its drawdowns span
        episodes; see `max_drawdowns` for each episode's)."""
        return performance_summary(self.returns, periods_per_year=self.periods_per_year)

    @property
    def sharpe(self) -> float:
        """Annualised Sharpe ratio of the pooled returns."""
        return sharpe_ratio(self.returns, periods_per_year=self.periods_per_year)

    @property
    def probabilistic_sharpe(self) -> float:
        """Probability that the true Sharpe ratio is above 0 (see `probabilistic_sharpe_ratio`)."""
        return probabilistic_sharpe_ratio(self.returns, periods_per_year=self.periods_per_year)

    def __repr__(self) -> str:
        return (
            f"Evaluation({self.name!r} on {self.task.name!r}: Sharpe {self.sharpe:.2f}, "
            f"mean episode return {np.mean(self.episode_returns):+.2%})"
        )


def evaluate(policy: Policy, task: Task, *, name: str = "policy") -> Evaluation:
    """Run `policy(observation, env)` on every episode of `task`.

    Args:
        policy: A callable from the observation and the unwrapped env to an action.
        task: The task.
        name: A label for the result.

    Raises:
        ValueError: If the environment reports no account value (`info["equity"]` or
            `info["portfolio_value"]`), which the statistics are computed from.
    """
    env = task.make()
    base = env.unwrapped
    curves, totals, turnover = [], [], []
    for k in range(task.episodes):
        result = env.reset(seed=task.seed + k)
        equity, total = [], 0.0
        while not result.done:
            result = env.step(policy(result.observation, base))
            total += float(result.reward)
            info = result.info
            value = info.get("equity", info.get("portfolio_value"))
            if value is None:
                env.close()
                raise ValueError(
                    f"{task.env_id} reports no account value (info['equity'] or "
                    "info['portfolio_value']), which evaluate() computes its statistics from."
                )
            equity.append(float(value))
            if "turnover" in info:
                turnover.append(float(info["turnover"]))
        curves.append(np.asarray(equity, dtype=np.float64))
        totals.append(total)
    env.close()
    return Evaluation(
        name=name,
        task=task,
        equity=tuple(curves),
        rewards=np.asarray(totals),
        turnover=float(np.mean(turnover)) if turnover else math.nan,
        periods_per_year=task.periods_per_year,
    )


@dataclass(frozen=True)
class Comparison:
    """Several policies evaluated on one task (see `compare`)."""

    evaluations: tuple[Evaluation, ...]

    @property
    def task(self) -> Task:
        return self.evaluations[0].task

    def deflated_sharpe(self, name: str) -> float:
        """The deflated Sharpe ratio of one policy, every policy compared counting as a trial."""
        sharpes = [e.sharpe for e in self.evaluations]
        chosen = next(e for e in self.evaluations if e.name == name)
        if len(sharpes) < 2:
            return chosen.probabilistic_sharpe
        return deflated_sharpe_ratio(chosen.returns, sharpes, chosen.periods_per_year)

    def table(self) -> list[dict[str, float | str]]:
        """One row per policy: Sharpe, PSR, DSR, mean and std of episode returns, the
        average of the episodes' maximum drawdowns, turnover."""
        rows: list[dict[str, float | str]] = []
        for e in self.evaluations:
            episodes = e.episode_returns
            rows.append(
                {
                    "policy": e.name,
                    "sharpe": e.sharpe,
                    "psr": e.probabilistic_sharpe,
                    "dsr": self.deflated_sharpe(e.name),
                    "mean_return": float(np.mean(episodes)),
                    "std_return": float(np.std(episodes)),
                    "max_drawdown": float(np.mean(e.max_drawdowns)),
                    "turnover": e.turnover,
                }
            )
        return rows

    def __str__(self) -> str:
        rows = self.table()
        width = max(16, *(len(str(r["policy"])) for r in rows))
        header = f"{self.task.name}: {self.task.episodes} episodes of {self.task.env_id}"
        cols = ("Sharpe", "PSR", "DSR", "mean ret", "std ret", "avg max DD", "turnover")
        lines = [header, f"{'policy':<{width}}" + "".join(f"  {c:>10}" for c in cols)]
        for r in rows:
            lines.append(
                f"{r['policy']!s:<{width}}  {r['sharpe']:>10.2f}  {r['psr']:>10.3f}  "
                f"{r['dsr']:>10.3f}  {r['mean_return']:>+10.2%}  {r['std_return']:>10.2%}  "
                f"{r['max_drawdown']:>10.2%}  {r['turnover']:>10.3f}"
            )
        return "\n".join(lines)


def compare(policies: Mapping[str, Policy], task: Task) -> Comparison:
    """Evaluate every policy on the same episodes of `task`.

    The deflated Sharpe ratio of each counts every policy in `policies` as a trial, so
    include the variants you tried, not only the winner.
    """
    if not policies:
        raise ValueError("compare() needs at least one policy.")
    return Comparison(tuple(evaluate(p, task, name=n) for n, p in policies.items()))


# ---------------------------------------------------------------------------
# From a target book to an action
# ---------------------------------------------------------------------------


def target_action(env: Any, weights: NDArray[np.float64]) -> NDArray[np.float32]:
    """The action that asks `env` for the target `weights` (one per asset slot).

    Works for `Market-v0`, `Market-v1` (any action map: `LongOnlyScores`,
    `TargetWeights`, or a projected map through its `from_target`) and `Portfolio-v0` /
    `-v1`, long-only or long/short. Long-only, negative weights are dropped and the
    rest of the equity is cash.
    """
    from sekai.backend import get_backend
    from sekai.finance.actions import ActionContext, LongOnlyScores, TargetWeights

    w = np.asarray(weights, dtype=np.float64)
    batch = getattr(env, "batch_env", None)
    if batch is not None:  # Market-v1: the env's own action map decides
        amap = batch.action_map
        if isinstance(amap, LongOnlyScores):
            return _long_only(w, amap.cash)
        if isinstance(amap, TargetWeights):
            return np.clip(w, -1.0, 1.0).astype(np.float32)
        ctx = ActionContext(
            backend=get_backend("numpy"),
            tradable=np.asarray(env.action_mask()[: batch.max_assets] > 0)[None],
            weights=np.asarray(env.weights)[None],
        )
        gross = float(np.abs(w).sum()) / amap.constraints.max_gross
        action = amap.from_target(w[None], min(max(gross, 1e-6), 1.0 - 1e-6), ctx)
        return np.asarray(action[0], dtype=np.float32)
    shape: tuple[int, ...] = tuple(env.action_space.shape)
    if float(env.action_space.low.min()) < 0:  # long/short target weights
        out = np.clip(w, -1.0, 1.0).astype(np.float32)
    else:
        out = _long_only(w, cash=shape[0] == w.size + 1)
    return out.reshape(shape)


def _long_only(w: NDArray[np.float64], cash: bool) -> NDArray[np.float32]:
    w = np.clip(w, 0.0, None)
    if not cash:
        return w.astype(np.float32)
    return np.append(w, max(0.0, 1.0 - w.sum())).astype(np.float32)


def _returns_window(observation: Any, env: Any) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """`(assets, lookback)` log returns and the tradable assets, from a Market or
    Portfolio observation."""
    if isinstance(observation, Mapping) and "assets" in observation:
        values = np.asarray(observation["assets"]["values"], dtype=np.float64)
        return values[..., 0], np.asarray(observation["assets"]["mask"]) > 0
    if isinstance(observation, Mapping) and "returns" in observation:
        r = np.asarray(observation["returns"], dtype=np.float64).T
        return r, np.ones(r.shape[0], dtype=bool)
    raise TypeError(
        "The baselines read Market-v0/v1 and Portfolio-v0/v1 observations; got "
        f"{type(observation).__name__}."
    )


def _normalise(scores: NDArray[np.float64], tradable: NDArray[np.bool_]) -> NDArray[np.float64]:
    s = np.where(tradable, np.clip(scores, 0.0, None), 0.0)
    total = s.sum()
    return s / total if total > 0 else s


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


class Cash:
    """Hold no assets: the zero-risk benchmark (and a check of cost accounting)."""

    def __call__(self, observation: Any, env: Any) -> NDArray[np.float32]:
        r, _ = _returns_window(observation, env)
        return target_action(env, np.zeros(r.shape[0]))

    def __repr__(self) -> str:
        return "Cash()"


class EqualWeight:
    """Equal weights across the tradable assets, scaled to `gross` (rebalanced each step)."""

    def __init__(self, gross: float = 1.0) -> None:
        self.gross = float(gross)

    def __call__(self, observation: Any, env: Any) -> NDArray[np.float32]:
        r, tradable = _returns_window(observation, env)
        return target_action(env, self.gross * _normalise(np.ones(r.shape[0]), tradable))

    def __repr__(self) -> str:
        return f"EqualWeight(gross={self.gross:g})"


class InverseVolatility:
    """Weights proportional to 1 / volatility of the observed window (risk parity
    without correlations)."""

    def __init__(self, gross: float = 1.0) -> None:
        self.gross = float(gross)

    def __call__(self, observation: Any, env: Any) -> NDArray[np.float32]:
        r, tradable = _returns_window(observation, env)
        vol = r.std(axis=1)
        scores = np.where(vol > 0, 1.0 / np.where(vol > 0, vol, 1.0), 0.0)
        return target_action(env, self.gross * _normalise(scores, tradable))

    def __repr__(self) -> str:
        return f"InverseVolatility(gross={self.gross:g})"


class Momentum:
    """Long the assets with the highest return over the last `window` steps.

    Holds the top `fraction` of tradable assets, equally weighted (long-only), or, with
    `long_short=True`, also shorts the bottom `fraction`, dollar neutral.

    Args:
        window: Look-back in steps (at most the env's lookback).
        fraction: Share of assets held long (and short).
        long_short: Short the losers too.
        gross: Gross exposure of the book.
    """

    def __init__(
        self,
        window: int = 20,
        fraction: float = 0.3,
        *,
        long_short: bool = False,
        gross: float = 1.0,
    ) -> None:
        largest = 0.5 if long_short else 1.0  # long/short holds a fraction each side
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}.")
        if not 0.0 < fraction <= largest:
            raise ValueError(f"fraction must be in (0, {largest:g}], got {fraction}.")
        self.window, self.fraction = int(window), float(fraction)
        self.long_short, self.gross = long_short, float(gross)

    def __call__(self, observation: Any, env: Any) -> NDArray[np.float32]:
        r, tradable = _returns_window(observation, env)
        past = r[:, -self.window :].sum(axis=1)
        idx = np.flatnonzero(tradable)
        w = np.zeros(r.shape[0])
        if idx.size:
            k = max(1, round(self.fraction * idx.size))
            order = idx[np.argsort(past[idx], kind="stable")]
            if self.long_short and idx.size >= 2:
                k = min(k, idx.size // 2)
                w[order[-k:]] = 0.5 * self.gross / k
                w[order[:k]] = -0.5 * self.gross / k
            else:
                w[order[-k:]] = self.gross / k
        return target_action(env, w)

    def __repr__(self) -> str:
        kind = ", long_short=True" if self.long_short else ""
        return f"Momentum({self.window}, {self.fraction:g}{kind})"


def standard_baselines() -> dict[str, Policy]:
    """`Cash`, `EqualWeight`, `InverseVolatility` and 20-step `Momentum`, by name."""
    return {
        "cash": Cash(),
        "equal weight": EqualWeight(),
        "inverse vol": InverseVolatility(),
        "momentum": Momentum(20),
    }


def standard_tasks(episodes: int = 100) -> list[Task]:
    """The reference tasks of the docs' results table (simulated markets)."""
    return [
        Task("market", "Market-v1", {"episode_length": 252}, episodes=episodes),
        Task(
            "market-close",
            "Market-v1",
            {"episode_length": 252, "execution": "close"},
            episodes=episodes,
        ),
        Task("portfolio", "Portfolio-v1", {"episode_length": 252}, episodes=episodes),
    ]


__all__ = [
    "Task",
    "Evaluation",
    "Comparison",
    "Policy",
    "evaluate",
    "compare",
    "target_action",
    "Cash",
    "EqualWeight",
    "InverseVolatility",
    "Momentum",
    "standard_baselines",
    "standard_tasks",
]
