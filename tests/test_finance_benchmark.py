"""sekai.finance.benchmark: tasks, evaluation, comparisons and baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import sekai
from sekai.finance import Constraints, ScoresLeverage
from sekai.finance.benchmark import (
    Cash,
    EqualWeight,
    InverseVolatility,
    Momentum,
    Task,
    compare,
    evaluate,
    standard_baselines,
    standard_tasks,
    target_action,
)

SHORT = {"episode_length": 20}


def test_evaluation_is_reproducible_and_consistent() -> None:
    task = Task("market", "Market-v1", SHORT, episodes=3)
    a, b = evaluate(EqualWeight(), task), evaluate(EqualWeight(), task)
    assert len(a.equity) == 3 and all(len(c) == 20 for c in a.equity)
    for x, y in zip(a.equity, b.equity):
        np.testing.assert_array_equal(x, y)
    # Pooled returns rebuild each episode's equity curve.
    returns = a.returns.reshape(3, 20)
    np.testing.assert_allclose(np.cumprod(1 + returns, axis=1), np.stack(a.equity), rtol=1e-12)
    np.testing.assert_allclose(a.episode_returns, [c[-1] - 1 for c in a.equity])
    np.testing.assert_allclose(a.rewards, np.log([c[-1] for c in a.equity]), rtol=1e-10)
    assert np.all((a.max_drawdowns >= 0) & (a.max_drawdowns <= 1))
    assert "Sharpe" in repr(a) and 0.0 <= a.probabilistic_sharpe <= 1.0
    with pytest.raises(ValueError, match="episodes"):
        Task("x", "Market-v1", episodes=0)


def test_cash_and_equal_weight_books() -> None:
    task = Task("market", "Market-v1", SHORT, episodes=2)
    cash = evaluate(Cash(), task)
    assert all(np.allclose(c, 1.0) for c in cash.equity) and cash.turnover == 0.0
    env = task.make()
    obs = env.reset(seed=0).observation
    action = EqualWeight()(obs, env.unwrapped)
    tradable = obs["assets"]["mask"] > 0
    np.testing.assert_allclose(action[:-1][tradable], 1.0 / tradable.sum(), rtol=1e-6)
    assert action[-1] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    "env_id,kwargs",
    [
        ("Market-v0", {}),
        ("Market-v0", {"allow_short": True}),
        ("Market-v1", {}),
        ("Market-v1", {"allow_short": True}),
        ("Market-v1", {"action": ScoresLeverage(Constraints(max_gross=2.0, max_position=0.5))}),
        ("Portfolio-v0", {}),
        ("Portfolio-v1", {"allow_short": True}),
    ],
)
def test_target_action_gives_the_target_book(env_id: str, kwargs: dict[str, Any]) -> None:
    env = sekai.make(env_id, episode_length=5, **kwargs)
    obs = env.reset(seed=0).observation
    base = env.unwrapped
    n = obs["assets"]["mask"].shape[0] if "assets" in obs else obs["returns"].shape[1]
    tradable = obs["assets"]["mask"] > 0 if "assets" in obs else np.ones(n, dtype=bool)
    w = np.where(tradable, np.linspace(0.05, 0.3, n), 0.0)
    w = 0.9 * w / w.sum()
    short = kwargs.get("allow_short") or "action" in kwargs
    if short:
        w = w * np.where(np.arange(n) % 2, -1.0, 1.0)
    action = target_action(base, w)
    assert env.action_space.contains(action) or "action" in kwargs
    if hasattr(base, "batch_env"):
        from sekai.backend import get_backend
        from sekai.finance.actions import ActionContext

        ctx = ActionContext(
            backend=get_backend("numpy"), tradable=tradable[None], weights=np.zeros((1, n))
        )
        got = base.batch_env.action_map(action[None].astype(np.float64), ctx)[0]
    elif env_id.startswith("Market"):
        got = base.target_weights(action)
    else:
        got = base.action_to_weights(action)[:-1]
    np.testing.assert_allclose(got, w, atol=1e-3 if "action" in kwargs else 1e-6)


def test_momentum_and_inverse_volatility() -> None:
    class Env:
        action_space = sekai.Box(0.0, 1.0, shape=(5,), dtype=np.float32)

    r = np.array([[0.01] * 4, [0.03] * 4, [-0.02] * 4, [0.02] * 4])
    obs = {"returns": r.T}
    w = Momentum(4, 0.5)(obs, Env())
    np.testing.assert_allclose(w, [0, 0.5, 0, 0.5, 0], atol=1e-7)
    vol = np.array([[0.01, -0.01] * 2, [0.02, -0.02] * 2, [0.04, -0.04] * 2, [0.0] * 4])
    w = InverseVolatility()({"returns": vol.T}, Env())
    np.testing.assert_allclose(w[:3], np.array([4, 2, 1]) / 7, rtol=1e-6)
    assert w[3] == 0.0  # no volatility estimate: no weight
    with pytest.raises(ValueError, match="fraction"):
        Momentum(5, 0.6, long_short=True)
    with pytest.raises(TypeError, match="Market-v0/v1 and Portfolio"):
        EqualWeight()(np.zeros(3), Env())


def test_compare_reports_every_policy() -> None:
    task = Task("portfolio", "Portfolio-v1", SHORT, episodes=3)
    result = compare(standard_baselines(), task)
    rows = result.table()
    assert [r["policy"] for r in rows] == list(standard_baselines())
    assert all(0.0 <= r["dsr"] <= 1.0 and 0.0 <= r["psr"] <= 1.0 for r in rows)
    # With many trials, the deflated Sharpe ratio is below the probabilistic one.
    best = max(rows, key=lambda r: r["sharpe"])
    assert best["dsr"] <= best["psr"]
    text = str(result)
    assert "portfolio: 3 episodes of Portfolio-v1" in text and "inverse vol" in text
    with pytest.raises(ValueError, match="at least one"):
        compare({}, task)
    assert {t.env_id for t in standard_tasks(5)} == {"Market-v1", "Portfolio-v1"}


def test_evaluate_needs_an_account_value() -> None:
    from sekai.finance.benchmark import Task, evaluate

    task = Task("making", "MarketMaking-v0", episodes=1)
    with pytest.raises(ValueError, match="reports no account value"):
        evaluate(lambda obs, env: env.action_space.sample(), task)
