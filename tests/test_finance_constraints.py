"""Constraint projection, factor risk models, action maps and market panels."""

from __future__ import annotations

import numpy as np
import pytest

from sekai.backend import get_backend
from sekai.finance import (
    GBM,
    ActionContext,
    Constraints,
    FactorRiskModel,
    ListingProcess,
    LongOnlyScores,
    MarketData,
    MarketPanel,
    ModelScenarios,
    ScoresLeverage,
    TanhWeights,
    TargetWeights,
    TradeDelta,
    factor_exposures,
    portfolio_variance,
    project,
)
from sekai.finance.account import AccountConfig, project_weights
from sekai.finance.constraints import neutralize

B = get_backend("numpy")


def scores(seed: int = 0, shape: tuple[int, int] = (6, 40)) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=shape)


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


class TestProject:
    def test_gross_and_caps(self) -> None:
        rules = Constraints(max_gross=2.0, max_position=0.1)
        w = project(scores(), rules, gross=1.5)
        np.testing.assert_allclose(np.abs(w).sum(-1), 1.5, rtol=1e-6)
        assert np.abs(w).max() <= 0.1 + 1e-12
        w = project(scores(), rules)  # default: the maximum gross
        np.testing.assert_allclose(np.abs(w).sum(-1), 2.0, rtol=1e-6)
        w = project(scores(), rules, gross=5.0)  # clipped to max_gross
        np.testing.assert_allclose(np.abs(w).sum(-1), 2.0, rtol=1e-6)

    def test_dollar_and_factor_neutral(self) -> None:
        rng = np.random.default_rng(1)
        loadings = rng.normal(size=(6, 40, 3))
        beta = 1.0 + 0.3 * rng.normal(size=(6, 40))
        rules = Constraints(
            max_gross=2.0, max_position=0.1, dollar_neutral=True, neutral_to=("loadings", "beta")
        )
        w = project(scores(), rules, gross=1.0, exposures=[loadings, beta])
        assert np.abs(w.sum(-1)).max() < 1e-3
        assert np.abs(factor_exposures(w, loadings)).max() < 1e-3
        assert np.abs((w * beta).sum(-1)).max() < 1e-3
        assert np.abs(w).max() <= 0.1 + 1e-12
        np.testing.assert_allclose(np.abs(w).sum(-1), 1.0, rtol=1e-6)

    def test_untradable_and_unshortable(self) -> None:
        rng = np.random.default_rng(2)
        tradable = rng.random((6, 40)) > 0.3
        shortable = rng.random((6, 40)) > 0.5
        w = project(
            scores(),
            Constraints(max_gross=1.0, max_position=0.2),
            tradable=tradable,
            shortable=shortable,
        )
        assert np.all(w[~tradable] == 0.0)
        assert np.all(w[~shortable] >= 0.0)
        w = project(scores(), Constraints(long_only=True, max_gross=1.0, max_position=0.05))
        assert np.all(w >= 0.0) and np.abs(w).max() <= 0.05 + 1e-12
        # Only positive scores are held, so the gross is capped by how many there are.
        reachable = np.minimum(1.0, (scores() > 0).sum(-1) * 0.05)
        np.testing.assert_allclose(w.sum(-1), reachable, rtol=1e-6)

    def test_zero_scores_and_infeasible_gross(self) -> None:
        w = project(np.zeros((2, 5)), Constraints(max_gross=1.0))
        assert np.all(w == 0.0)
        # 5 names at a 0.1 cap cannot reach a gross of 1: every name sits at the cap.
        w = project(scores(shape=(2, 5)), Constraints(max_gross=1.0, max_position=0.1))
        np.testing.assert_allclose(np.abs(w), 0.1, rtol=1e-9)

    def test_numpy_and_torch_agree_and_gradients_flow(self) -> None:
        torch = pytest.importorskip("torch")
        rng = np.random.default_rng(3)
        loadings = rng.normal(size=(6, 40, 2))
        rules = Constraints(max_gross=2.0, max_position=0.1, dollar_neutral=True, neutral_to=("x",))
        want = project(scores(), rules, gross=1.2, exposures=loadings)
        s = torch.tensor(scores(), dtype=torch.float64, requires_grad=True)
        g = torch.full((6,), 1.2, dtype=torch.float64, requires_grad=True)
        got = project(s, rules, gross=g, exposures=torch.as_tensor(loadings))
        np.testing.assert_allclose(got.detach().numpy(), want, rtol=1e-9, atol=1e-12)
        (got * torch.as_tensor(rng.normal(size=(6, 40)))).sum().backward()
        assert s.grad is not None and g.grad is not None
        assert bool(torch.isfinite(s.grad).all()) and float(s.grad.abs().sum()) > 0
        assert bool(torch.isfinite(g.grad).all())

    def test_zero_scores_have_finite_gradients(self) -> None:
        torch = pytest.importorskip("torch")
        rules = Constraints(max_gross=2.0, max_position=0.05, dollar_neutral=True)
        s = torch.zeros(2, 30, requires_grad=True)
        w = project(s, rules, gross=torch.tensor([1.0, 1.0]))
        (w * torch.randn(2, 30)).sum().backward()
        assert float(w.detach().abs().sum()) == 0.0
        assert s.grad is not None and bool(torch.isfinite(s.grad).all())

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="long-only book cannot be"):
            Constraints(long_only=True, dollar_neutral=True)
        with pytest.raises(ValueError, match="max_gross"):
            Constraints(max_gross=0.0)
        with pytest.raises(ValueError, match="max_position"):
            Constraints(max_position=float("nan"))
        with pytest.raises(ValueError, match="n_iter"):
            Constraints(n_iter=-1)
        with pytest.raises(ValueError, match="needs their values as `exposures`"):
            project(scores(), Constraints(neutral_to=("beta",)))

    def test_neutralize_is_orthogonal(self) -> None:
        rng = np.random.default_rng(4)
        c = rng.normal(size=(3, 30, 4))
        x = neutralize(rng.normal(size=(3, 30)), c, ridge=1e-10)
        np.testing.assert_allclose(np.einsum("bnk,bn->bk", c, x), 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------


def test_factor_variance_matches_the_full_covariance() -> None:
    rng = np.random.default_rng(5)
    n, k = 12, 3
    loadings = rng.normal(size=(4, n, k))
    a = rng.normal(size=(4, k, k))
    factor_cov = a @ np.transpose(a, (0, 2, 1)) / k
    idio = rng.uniform(1e-4, 4e-4, (4, n))
    w = rng.normal(0, 0.1, (4, n))
    sigma = loadings @ factor_cov @ np.transpose(loadings, (0, 2, 1)) + idio[:, :, None] * np.eye(n)
    want = np.einsum("bi,bij,bj->b", w, sigma, w)
    got = portfolio_variance(w, loadings=loadings, factor_cov=factor_cov, idio_var=idio)
    np.testing.assert_allclose(got, want, rtol=1e-12)
    np.testing.assert_allclose(portfolio_variance(w, idio_var=idio), (w * w * idio).sum(-1))
    torch = pytest.importorskip("torch")
    th = portfolio_variance(
        torch.as_tensor(w),
        loadings=torch.as_tensor(loadings),
        factor_cov=torch.as_tensor(factor_cov),
        idio_var=torch.as_tensor(idio),
    )
    np.testing.assert_allclose(th.numpy(), want, rtol=1e-12)
    with pytest.raises(ValueError, match="go together"):
        portfolio_variance(w, loadings=loadings)
    with pytest.raises(ValueError, match="needs"):
        portfolio_variance(w)


def test_factor_risk_model_fields() -> None:
    model = FactorRiskModel()
    assert model.asset_fields == ("loadings", "idio_var")
    assert model.market_fields == ("factor_cov",)
    assert FactorRiskModel(loadings=None, factor_cov=None).asset_fields == ("idio_var",)
    with pytest.raises(ValueError, match="go together"):
        FactorRiskModel(factor_cov=None)
    with pytest.raises(ValueError, match="needs"):
        FactorRiskModel(loadings=None, factor_cov=None, idio_var=None)


# ---------------------------------------------------------------------------
# Action maps
# ---------------------------------------------------------------------------


def ctx(n: int = 4, s: int = 10, seed: int = 0, backend: object = B) -> ActionContext:
    rng = np.random.default_rng(seed)
    tradable = rng.random((n, s)) > 0.2
    weights = np.where(tradable, rng.normal(0, 0.05, (n, s)), 0.0)
    if backend is B:
        return ActionContext(backend=B, tradable=tradable, weights=weights)
    import torch

    from sekai.backend import get_backend

    return ActionContext(
        backend=get_backend("torch"),
        tradable=torch.as_tensor(tradable),
        weights=torch.as_tensor(weights),
    )


def test_market_v0_action_maps() -> None:
    c = ctx()
    a = np.random.default_rng(1).random((4, 11))
    np.testing.assert_array_equal(
        LongOnlyScores(1.5)(a, c),
        project_weights(
            a[:, :-1], c.tradable, AccountConfig(max_leverage=1.5), cash_score=a[:, -1]
        ),
    )
    b = np.random.default_rng(2).uniform(-1, 1, (4, 10))
    np.testing.assert_array_equal(
        TargetWeights(2.0)(b, c),
        project_weights(b, c.tradable, AccountConfig(allow_short=True, max_leverage=2.0)),
    )
    assert LongOnlyScores().space(10).shape == (11,)
    assert LongOnlyScores(cash=False).space(10).shape == (10,)
    assert TargetWeights().space(10).shape == (10,)
    with pytest.raises(ValueError, match="max_leverage"):
        LongOnlyScores(0.0)


@pytest.mark.parametrize("cls", [ScoresLeverage, TanhWeights, TradeDelta])
def test_projected_maps_round_trip(cls: type) -> None:
    rules = Constraints(max_gross=2.0, max_position=0.3, dollar_neutral=True)
    m = cls(rules)
    c = ctx()
    target_scores = np.random.default_rng(3).normal(size=(4, 10))
    action = m.from_target(target_scores, 0.6, c)
    assert action.shape == (4, *m.space(10).shape)
    got = m(action, c)
    want = project(target_scores, rules, gross=1.2, tradable=c.tradable)
    np.testing.assert_allclose(got, want, atol=2e-3)
    assert np.all(got[~c.tradable] == 0.0)


def test_scores_leverage_controls_gross() -> None:
    m = ScoresLeverage(Constraints(max_gross=3.0, max_position=0.5))
    c = ctx()
    a = np.concatenate(
        [np.random.default_rng(4).normal(size=(4, 10)), np.full((4, 1), 0.0)], axis=1
    )
    np.testing.assert_allclose(np.abs(m(a, c)).sum(-1), 1.5, rtol=1e-6)


def test_trade_delta_zero_action_keeps_the_book_direction() -> None:
    m = TradeDelta(Constraints(max_gross=1.0))
    c = ctx()
    gross_fraction = np.abs(c.weights).sum(-1)
    action = np.concatenate(
        [np.zeros((4, 10)), np.log(gross_fraction / (1 - gross_fraction))[:, None]], axis=1
    )
    np.testing.assert_allclose(m(action, c), c.weights, atol=1e-9)


def test_action_maps_on_torch() -> None:
    torch = pytest.importorskip("torch")
    c = ctx(backend="torch")
    m = ScoresLeverage(Constraints(max_gross=2.0, max_position=0.3, dollar_neutral=True))
    a = torch.zeros(4, 11, dtype=torch.float64, requires_grad=True)
    w = m(a + torch.as_tensor(np.random.default_rng(5).normal(size=(4, 11))), c)
    w.pow(2).sum().backward()
    assert a.grad is not None and bool(torch.isfinite(a.grad).all())


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def market_data() -> MarketData:
    rng = np.random.default_rng(6)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (50, 4)), axis=0))
    prices[:10, 3] = np.nan
    return MarketData(
        prices,
        fields={"volume": rng.lognormal(10, 1, (50, 4)), "loadings": rng.normal(size=(50, 4, 2))},
        exogenous={
            "factor_cov": np.broadcast_to(np.eye(2) * 1e-4, (50, 2, 2)),
            "rate": np.full(50, 0.01),
        },
    )


def test_panel_from_market_data() -> None:
    data = market_data()
    p = MarketPanel.from_market_data(data, fields=["loadings"], exogenous=["factor_cov"])
    assert (p.n_paths, p.n_steps, p.n_assets) == (1, 50, 4)
    np.testing.assert_allclose(p.returns[0, 1:, 0], data.prices[1:, 0] / data.prices[:-1, 0] - 1.0)
    assert np.isnan(p.returns[0, 0]).all() and np.isnan(p.returns[0, 10, 3])
    np.testing.assert_array_equal(p.valid[0], data.mask)
    assert p.fields["loadings"].shape == (1, 50, 4, 2)
    assert p.exogenous["factor_cov"].shape == (1, 50, 2, 2)
    torch = pytest.importorskip("torch")
    t = p.to("torch")
    assert t.returns.dtype == torch.get_default_dtype() and t.valid.dtype == torch.bool
    assert t.to("torch", dtype=torch.float64).returns.dtype == torch.float64
    np.testing.assert_allclose(t.to("numpy").returns, p.returns, rtol=1e-6)


def test_panel_validation() -> None:
    r = np.zeros((2, 5, 3))
    with pytest.raises(ValueError, match="valid has shape"):
        MarketPanel(returns=r, valid=np.ones((2, 5, 2), dtype=bool))
    with pytest.raises(ValueError, match="first three dimensions"):
        MarketPanel.from_returns(r, fields={"x": np.zeros((2, 5, 4))})
    with pytest.raises(ValueError, match="first two dimensions"):
        MarketPanel.from_returns(r, exogenous={"x": np.zeros((2, 4))})
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="another backend"):
        MarketPanel(returns=r, valid=torch.ones(2, 5, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="asset ids"):
        MarketPanel.from_returns(r, assets=["a"])


def test_model_scenarios_are_deterministic() -> None:
    gen = ModelScenarios(ListingProcess(GBM(mu=0.05, sigma=0.2, n_assets=5)), n_steps=30)
    a, b = gen.sample(3, seed=7), gen.sample(3, seed=7)
    assert a.returns.shape == (3, 31, 5)
    np.testing.assert_array_equal(a.returns, b.returns)
    assert not np.array_equal(gen.sample(3, seed=8).returns, a.returns)
    with pytest.raises(TypeError):
        ModelScenarios("not a model", 3)  # type: ignore[arg-type]


def test_market_data_higher_rank_fields() -> None:
    data = market_data()
    assert data.field("loadings").shape == (50, 4, 2)
    assert data.exogenous_width("factor_cov") == 4
    sub = data.select(["asset_0", "asset_2"])[5:20]
    assert sub.field("loadings").shape == (15, 2, 2)
    with pytest.raises(ValueError, match=r"first two dimensions are \(T, N\)"):
        data.with_fields(bad=np.zeros((50, 3, 2)))
    with pytest.raises(ValueError, match="must be 2-D"):
        MarketData(fields={"close": np.ones((5, 2, 2))})


def test_long_only_caps_keep_the_scores_direction() -> None:
    """The cap applies after scaling, so the book depends on the direction only."""
    c = Constraints(long_only=True, max_position=0.4)
    scores = np.array([10.0, 1.0, 0.5])
    np.testing.assert_allclose(project(scores, c), [0.4, 0.4, 0.2], atol=1e-9)
    np.testing.assert_allclose(project(scores / 10, c), project(scores, c), atol=1e-12)
    held = np.array([0.05, 0.03, 0.02])  # a feasible book maps to itself
    tight = Constraints(long_only=True, max_position=0.05)
    np.testing.assert_allclose(project(held, tight, gross=0.1), held, atol=1e-12)
    torch = pytest.importorskip("torch")
    x = torch.tensor([10.0, 1.0, 0.5], dtype=torch.float64, requires_grad=True)
    project(x, c)[2].backward()
    assert x.grad is not None and x.grad.abs().sum() > 0  # was all zeros


def test_dollar_neutral_with_limited_shorting_is_exactly_neutral() -> None:
    rng = np.random.default_rng(0)
    c = Constraints(dollar_neutral=True, max_position=0.2)
    for _ in range(300):
        x = rng.normal(size=(3, 10))
        shortable = rng.uniform(size=(3, 10)) > 0.8
        gross = rng.uniform(0, 1, 3)
        w = project(x, c, gross=gross, shortable=shortable)
        np.testing.assert_allclose(w.sum(-1), 0.0, atol=1e-12)
        assert np.all(np.abs(w) <= 0.2 + 1e-12) and np.all(w[~shortable] >= 0)
        assert np.all(np.abs(w).sum(-1) <= gross + 1e-12)
    nothing = project(
        rng.normal(size=4), Constraints(dollar_neutral=True), shortable=np.zeros(4, bool)
    )
    np.testing.assert_array_equal(nothing, np.zeros(4))  # no short side: an empty book


def test_projection_is_exact_however_far_apart_the_scores_are() -> None:
    w = project(np.array([[3.0, 2.0, 1e-16, 1e-16, 1e-16]]), Constraints(max_position=0.3))
    np.testing.assert_allclose(w, [[0.3, 0.3, 0.4 / 3, 0.4 / 3, 0.4 / 3]], rtol=1e-12)
    rng = np.random.default_rng(5)
    logits = rng.normal(0, 12, (200, 20))  # long-only softmax scores
    scores = np.exp(logits - logits.max(-1, keepdims=True))
    w = project(scores, Constraints(long_only=True, max_position=0.1))
    np.testing.assert_allclose(np.abs(w).sum(-1), 1.0, rtol=1e-12)
    assert np.abs(w).max() <= 0.1


def test_dollar_neutral_books_are_exactly_neutral() -> None:
    rng = np.random.default_rng(6)
    c = Constraints(max_position=0.1, max_gross=2.0, dollar_neutral=True)
    w = project(rng.normal(size=(500, 10)), c, gross=np.full(500, 1.0))
    np.testing.assert_allclose(w.sum(-1), 0.0, atol=1e-14)
    assert np.abs(w).max() <= 0.1 and np.all(np.abs(w).sum(-1) <= 1.0 + 1e-12)
    w = project(
        np.array([[1.0, 1.0, 1.0, -1.0]]),
        Constraints(max_gross=2, max_position=0.3, dollar_neutral=True),
        gross=1.2,
    )
    np.testing.assert_allclose(w, [[0.1, 0.1, 0.1, -0.3]])
