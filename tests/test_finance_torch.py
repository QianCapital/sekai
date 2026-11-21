"""The finance components on torch: same values as NumPy, finite gradients, detachable state."""

from __future__ import annotations

import numpy as np
import pytest

from sekai.finance import (
    BorrowFee,
    DifferentialSharpe,
    DrawdownPenalty,
    FixedCost,
    GrossFinancing,
    LogReturn,
    MeanVariance,
    NoCost,
    PortfolioVariance,
    ProfitAndLoss,
    ProportionalCost,
    SimpleReturn,
    SpreadCost,
    SquareRootImpact,
    StepOutcome,
    TurnoverPenalty,
)
from sekai.finance.account import AccountConfig, project_weights, settle, turnover

torch = pytest.importorskip("torch")


def t64(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float64)


def context(rng: np.random.Generator, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    ctx = {
        "equity": rng.uniform(1e5, 1e7, shape[0]),
        "volatility": np.abs(rng.normal(0.02, 0.01, shape)),
        "volume": rng.lognormal(14, 1, shape),
        "spread": rng.uniform(1e-4, 2e-3, shape),
        "borrow_rate": rng.uniform(0.0, 0.2, shape),
    }
    ctx["volatility"][1, 2] = np.nan  # replaced by the cross-sectional median
    ctx["spread"][2, :3] = np.nan
    ctx["borrow_rate"][3, 0] = np.nan
    return ctx


COSTS = [
    NoCost(),
    ProportionalCost(1e-3),
    ProportionalCost(np.linspace(0, 1e-3, 7)),
    SpreadCost(),
    SpreadCost(half_spread=2e-4),
    FixedCost(5.0),
    SquareRootImpact(0.3),
    SpreadCost() + SquareRootImpact(0.1) + FixedCost(1.0),
]


@pytest.mark.parametrize("model", COSTS, ids=repr)
def test_cost_models_agree(model: object) -> None:
    rng = np.random.default_rng(0)
    trades = rng.normal(0, 0.1, (6, 7))
    trades[0] = 0.0
    ctx = {k: v for k, v in context(rng, trades.shape).items() if k in model.requires}  # type: ignore[attr-defined]
    want = model(trades, **ctx)  # type: ignore[operator]
    got = model(t64(trades), **{k: t64(v) for k, v in ctx.items()})  # type: ignore[operator]
    assert isinstance(got, torch.Tensor)
    np.testing.assert_allclose(got.numpy(), want, rtol=1e-12, atol=1e-18)


def test_per_asset_parameters_follow_assets() -> None:
    """With `assets=`, entry i of the trades is priced with asset assets[i]'s parameter."""
    rates = np.array([1e-4, 2e-4, 3e-4, 4e-4])
    trades = np.array([[0.1, -0.2, 0.0]])
    assets = np.array([[3, 0, -1]])  # slot 2 is empty
    want = 0.1 * 4e-4 + 0.2 * 1e-4
    assert ProportionalCost(rates)(trades, assets=assets) == pytest.approx(want)
    assert float(
        ProportionalCost(rates)(t64(trades), assets=torch.as_tensor(assets))
    ) == pytest.approx(want)
    assert SpreadCost(half_spread=rates)(trades, assets=assets) == pytest.approx(want)


def test_cost_gradients_are_finite_at_zero_trades() -> None:
    """The square root in the impact model has an infinite slope at 0; it must not leak NaNs."""
    rng = np.random.default_rng(1)
    ctx = {k: t64(v) for k, v in context(rng, (4, 5)).items()}
    trades = t64(rng.normal(0, 0.05, (4, 5)))
    trades[:, 0] = 0.0
    trades[1] = 0.0
    trades.requires_grad_(True)
    model = SpreadCost() + SquareRootImpact(0.5)
    model(
        trades,
        spread=ctx["spread"],
        equity=ctx["equity"],
        volatility=ctx["volatility"],
        volume=ctx["volume"],
    ).sum().backward()
    assert trades.grad is not None and bool(torch.isfinite(trades.grad).all())


def test_cost_inputs_still_validated_on_torch() -> None:
    trades = torch.ones(1, 3)
    with pytest.raises(ValueError, match="no valid 'spread'"):
        SpreadCost()(trades, spread=torch.full((1, 3), float("nan")))
    assert float(
        SpreadCost(fallback_spread=0.01)(trades, spread=torch.full((1, 3), float("nan")))
    ) == pytest.approx(0.015)


def test_holding_costs() -> None:
    rng = np.random.default_rng(2)
    w = rng.normal(0, 0.2, (4, 6))
    borrow = context(rng, (4, 6))["borrow_rate"]
    dt = 1 / 252
    fee = BorrowFee()(w, dt=dt, borrow_rate=borrow)
    filled = np.where(np.isfinite(borrow), borrow, np.nanmedian(borrow, axis=-1, keepdims=True))
    np.testing.assert_allclose(fee, dt * (np.clip(-w, 0, None) * filled).sum(-1))
    np.testing.assert_allclose(BorrowFee(0.05)(w, dt=dt), dt * 0.05 * np.clip(-w, 0, None).sum(-1))
    np.testing.assert_allclose(GrossFinancing(0.02)(w, dt=dt), dt * 0.02 * np.abs(w).sum(-1))
    both = BorrowFee() + GrossFinancing(0.02)
    assert both.requires == {"borrow_rate"}
    np.testing.assert_allclose(
        both(t64(w), dt=dt, borrow_rate=t64(borrow)).numpy(),
        fee + GrossFinancing(0.02)(w, dt=dt),
        rtol=1e-12,
    )
    with pytest.raises(ValueError, match="fallback_rate only applies"):
        BorrowFee(0.1, fallback_rate=0.2)


@pytest.mark.parametrize(
    "config",
    [
        AccountConfig(),
        AccountConfig(allow_short=True, max_leverage=2.0, cash_rate=0.02, borrow_rate=0.01),
        AccountConfig(max_leverage=1.5, cash_rate=0.01, margin_rate=0.05),
    ],
)
def test_accounting_agrees(config: AccountConfig) -> None:
    rng = np.random.default_rng(3)
    scores = rng.normal(0, 0.3, (8, 5))
    tradable = rng.random((8, 5)) > 0.2
    cash = rng.random(8)
    for args in ((scores, tradable), (np.abs(scores), tradable)):
        want = project_weights(*args, config, cash_score=cash)
        got = project_weights(t64(args[0]), torch.as_tensor(args[1]), config, cash_score=t64(cash))
        np.testing.assert_allclose(got.numpy(), want, rtol=1e-13)
    w = project_weights(scores, tradable, config)
    r = rng.normal(0, 0.02, (8, 5))
    cost = rng.uniform(0, 0.01, 8)
    hold = rng.uniform(0, 1e-4, 8)
    a = settle(w, r, cost=cost, dt=1 / 252, config=config, holding=hold)
    b = settle(t64(w), t64(r), cost=t64(cost), dt=1 / 252, config=config, holding=t64(hold))
    for name in ("growth", "drifted", "financing"):
        np.testing.assert_allclose(getattr(b, name).numpy(), getattr(a, name), rtol=1e-13)
    np.testing.assert_array_equal(b.wiped_out.numpy(), a.wiped_out)
    np.testing.assert_allclose(turnover(t64(w), t64(w * 0.5)).numpy(), turnover(w, w * 0.5))


def test_holding_cost_enters_financing() -> None:
    config = AccountConfig(cash_rate=0.0)
    w = np.array([0.5, 0.5])
    base = settle(w, np.zeros(2), cost=0.0, dt=1.0, config=config)
    held = settle(w, np.zeros(2), cost=0.0, dt=1.0, config=config, holding=0.01)
    assert float(held.financing) == pytest.approx(float(base.financing) - 0.01)
    assert float(held.growth) == pytest.approx(0.99)


def test_projection_gradients_are_finite_for_all_zero_scores() -> None:
    for config in (AccountConfig(), AccountConfig(allow_short=True, max_leverage=2.0)):
        scores = torch.zeros(3, 4, requires_grad=True)
        project_weights(scores, torch.ones(3, 4, dtype=torch.bool), config).sum().backward()
        assert scores.grad is not None and bool(torch.isfinite(scores.grad).all())


REWARDS = [
    LogReturn(),
    SimpleReturn(),
    ProfitAndLoss(),
    MeanVariance(2.0),
    TurnoverPenalty(),
    DrawdownPenalty(),
    DifferentialSharpe(0.05),
    LogReturn() - 0.5 * DrawdownPenalty() - 1e-3 * TurnoverPenalty(),
]


@pytest.mark.parametrize("make", [lambda r=r: r for r in REWARDS], ids=[repr(r) for r in REWARDS])
def test_rewards_agree(make: object) -> None:
    import copy

    rng = np.random.default_rng(4)
    growth = rng.uniform(0.95, 1.05, (30, 6))
    np_reward, th_reward = copy.deepcopy(make()), copy.deepcopy(make())  # type: ignore[operator]
    eq_np, eq_th = np.ones(6), torch.ones(6, dtype=torch.float64)
    for t in range(30):
        prev_np, prev_th = eq_np, eq_th
        eq_np, eq_th = eq_np * growth[t], eq_th * t64(growth[t])
        a = np_reward(
            StepOutcome.from_growth(
                growth[t],
                equity=eq_np,
                turnover=np.full(6, 0.1),
                cost=np.full(6, 1e-4),
                previous_equity=prev_np,
            )
        )
        b = th_reward(
            StepOutcome.from_growth(
                t64(growth[t]),
                equity=eq_th,
                turnover=torch.full((6,), 0.1, dtype=torch.float64),
                cost=torch.full((6,), 1e-4, dtype=torch.float64),
                previous_equity=prev_th,
            )
        )
        np.testing.assert_allclose(b.numpy(), a, rtol=1e-12, atol=1e-15)
        if t == 15:
            mask = np.arange(6) % 2 == 0
            np_reward.reset(mask)
            th_reward.reset(torch.as_tensor(mask))


def test_stateful_reward_detach_state() -> None:
    reward = DrawdownPenalty() + DifferentialSharpe(0.1)
    g = torch.tensor([1.01, 0.97], requires_grad=True)
    out = StepOutcome.from_growth(g, equity=g * 1.0, turnover=torch.zeros(2), cost=torch.zeros(2))
    reward(out).sum()
    states = [term._state for _, term in reward.terms]  # type: ignore[attr-defined]
    assert any(v.requires_grad for s in states for v in s.values())
    reward.detach_state()
    states = [term._state for _, term in reward.terms]  # type: ignore[attr-defined]
    assert not any(v.requires_grad for s in states for v in s.values())


def test_reward_state_follows_backend() -> None:
    """A stateful reward used on NumPy and then on torch rebuilds its state."""
    reward = DrawdownPenalty()
    reward(StepOutcome.from_growth(np.array([0.9]), equity=np.array([0.9]), turnover=0.0, cost=0.0))
    out = reward(
        StepOutcome.from_growth(
            torch.tensor([0.9]), equity=torch.tensor([0.9]), turnover=0.0, cost=0.0
        )
    )
    assert isinstance(out, torch.Tensor)


def test_portfolio_variance_reward() -> None:
    outcome = StepOutcome.from_growth(
        np.array([1.0]), equity=np.array([1.0]), turnover=0.0, cost=0.0, variance=np.array([2e-4])
    )
    assert float(PortfolioVariance()(outcome)[0]) == pytest.approx(2e-4)
    assert float((SimpleReturn() - 3.0 * PortfolioVariance())(outcome)[0]) == pytest.approx(-6e-4)
    bare = StepOutcome.from_growth(np.array([1.0]), equity=np.array([1.0]), turnover=0.0, cost=0.0)
    with pytest.raises(ValueError, match="risk model"):
        PortfolioVariance()(bare)
