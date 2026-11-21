"""Market-v1 / BatchMarketEnv: parity with Market-v0, execution timing, backends, scenarios."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import sekai
from sekai.envs.finance import BatchMarketEnv, MarketVecEnv
from sekai.finance import (
    GBM,
    BorrowFee,
    Constraints,
    FactorRiskModel,
    ListingProcess,
    LogReturn,
    MarketData,
    MarketPanel,
    NoCost,
    ObservationBuilder,
    PortfolioVariance,
    ScoresLeverage,
    SimpleReturn,
    SpreadCost,
    SquareRootImpact,
    TargetWeights,
    portfolio_variance,
)


def history(n_steps: int = 500, n_assets: int = 12, seed: int = 0) -> MarketData:
    rng = np.random.default_rng(seed)
    sim = ListingProcess(GBM(mu=0.05, sigma=rng.uniform(0.15, 0.45, n_assets))).simulate(
        rng, n_steps - 1
    )
    prices = sim.prices
    gap = np.exp(rng.normal(0, 0.005, prices.shape))
    return MarketData(
        prices,
        timestamps=np.busday_offset("2015-01-01", np.arange(n_steps), roll="forward"),
        fields={
            "volume": rng.lognormal(12, 1, prices.shape),
            "spread": rng.uniform(1e-4, 1e-3, prices.shape),
            "open": np.vstack([prices[:1], prices[:-1] * gap[1:]]),  # NaN where unlisted
            "borrow_rate": rng.uniform(0.0, 0.3, prices.shape),
            "shortable": (rng.random(prices.shape) > 0.3).astype(float),
            "loadings": rng.normal(1.0, 0.3, (*prices.shape, 2)),
            "idio_var": rng.uniform(1e-4, 4e-4, prices.shape),
        },
        exogenous={
            "rate": rng.normal(0.02, 0.001, n_steps),
            "factor_cov": np.broadcast_to(np.diag([1e-4, 4e-5]), (n_steps, 2, 2)),
        },
    )


DATA = history()

COMMON: dict[str, Any] = dict(
    data=DATA,
    lookback=10,
    episode_length=60,
    features=("volume",),
    exogenous=("rate",),
    transaction_cost=SpreadCost() + SquareRootImpact(0.2),
    cash_rate=0.01,
    borrow_rate=0.02,
    delisting_return=-0.3,
)


def rollout(env: sekai.Env, actions: list[np.ndarray], start: int) -> list[dict[str, Any]]:
    out = []
    env.reset(seed=1, options={"start": start})
    for a in actions:
        res = env.step(a)
        out.append(
            {k: np.asarray(v, dtype=float) for k, v in res.info.items() if k in INFO}
            | {"reward": res.reward}
        )
        if res.done:
            break
    return out


INFO = ("equity", "turnover", "cost", "financing", "gross_exposure", "n_assets", "weights")


def assert_same(a: list[dict[str, Any]], b: list[dict[str, Any]], atol: float = 1e-15) -> None:
    assert len(a) == len(b)
    for x, y in zip(a, b):
        for k in x:
            np.testing.assert_allclose(x[k], y[k], rtol=0, atol=atol, err_msg=k)


@pytest.mark.parametrize("allow_short", [False, True])
@pytest.mark.parametrize("max_assets", [None, 6])
def test_close_execution_reproduces_market_v0(allow_short: bool, max_assets: int | None) -> None:
    kw = dict(COMMON, allow_short=allow_short, max_leverage=1.5, max_assets=max_assets)
    v0, v1 = sekai.make("Market-v0", **kw), sekai.make("Market-v1", execution="close", **kw)
    for start in (10, 113, 400):
        v0.reset(seed=1, options={"start": start})
        v1.reset(seed=1, options={"start": start})
        assert v0.unwrapped.slot_assets == v1.unwrapped.slot_assets
        while True:
            mask = v0.unwrapped.action_mask()
            np.testing.assert_array_equal(mask, v1.unwrapped.action_mask())
            a = v0.action_space.sample(mask=mask)
            s0, s1 = v0.step(a), v1.step(a)
            for key in INFO:
                np.testing.assert_allclose(
                    s1.info[key], s0.info[key], rtol=0, atol=1e-15, err_msg=key
                )
            assert s1.reward == pytest.approx(s0.reward, abs=1e-15)
            assert s1.info["timestamp"] == s0.info["timestamp"]
            np.testing.assert_array_equal(
                s1.observation["assets"]["mask"], s0.observation["assets"]["mask"]
            )
            np.testing.assert_allclose(
                s1.observation["assets"]["values"], s0.observation["assets"]["values"], atol=1e-6
            )
            np.testing.assert_allclose(
                s1.observation["portfolio"], s0.observation["portfolio"], atol=1e-7
            )
            np.testing.assert_allclose(s1.observation["market"], s0.observation["market"])
            assert v0.unwrapped.slot_assets == v1.unwrapped.slot_assets
            assert (s0.terminated, s0.truncated) == (s1.terminated, s1.truncated)
            if s0.done:
                break


def test_next_close_is_close_delayed_by_one_decision() -> None:
    """With next-close execution, decision k fills at the close of k + 1: the same account
    as same-close execution fed every action one decision later (costs included), while
    the universe does not change (a pending order drops assets that stop trading)."""
    rng0 = np.random.default_rng(5)
    prices = 100 * np.exp(np.cumsum(rng0.normal(0, 0.01, (300, 6)), axis=0))
    fixed = MarketData(
        prices, fields={"volume": np.full(prices.shape, 1e6), "spread": np.full(prices.shape, 4e-4)}
    )
    kw = dict(COMMON, data=fixed, features=("volume",), exogenous=())
    close = sekai.make("Market-v1", execution="close", **kw)
    delayed = sekai.make("Market-v1", execution="next_close", **kw)
    delayed.action_space.seed(0)
    actions = [delayed.action_space.sample() for _ in range(60)]
    cash = np.zeros_like(actions[0])  # all-zero scores: all cash, the book before the first fill
    a = rollout(delayed, actions, start=50)
    b = rollout(close, [cash, *actions[:-1]], start=50)
    assert_same(a, b, atol=1e-14)


def test_next_open_splits_the_period_at_the_open() -> None:
    """Old book earns close -> open, the new one open -> close; open == close and zero
    costs make it same-close execution fed every action one decision later."""
    flat = MarketData(DATA.prices, timestamps=DATA.timestamps, fields={"open": DATA.prices})
    kw = dict(data=flat, lookback=5, episode_length=40, transaction_cost=NoCost())
    nxt = sekai.make("Market-v1", execution="next_open", **kw)
    close = sekai.make("Market-v1", execution="close", **kw)
    actions = [nxt.action_space.sample() for _ in range(40)]
    a = rollout(nxt, actions, start=30)
    b = rollout(close, [np.zeros_like(actions[0]), *actions[:-1]], start=30)
    for x, y in zip(a, b):
        assert x["equity"] == pytest.approx(y["equity"], rel=1e-13)

    # One asset, one step, by hand.
    p = np.array([[100.0], [101.0], [99.0], [102.0], [103.0]])
    o = np.array([[100.0], [100.5], [100.0], [101.0], [104.0]])
    one = MarketData(p, fields={"open": o})
    env = BatchMarketEnv(
        1,
        data=one,
        lookback=1,
        episode_length=2,
        execution="next_open",
        transaction_cost=NoCost(),
        action=TargetWeights(),
    )
    env.reset(seed=0, options={"start": 1})
    r1 = env.step(np.array([[1.0]]))  # flat overnight, long from the open of row 2 to its close
    assert float(r1.info["equity"][0]) == pytest.approx(99.0 / 100.0)
    r2 = env.step(np.array([[0.0]]))  # long overnight (99 -> 101 open), flat from the open
    assert float(r2.info["equity"][0]) == pytest.approx(0.99 * 101.0 / 99.0)


def test_next_open_needs_open_prices() -> None:
    bare = MarketData(DATA.prices)
    with pytest.raises(KeyError, match="open"):
        BatchMarketEnv(1, data=bare, execution="next_open")


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        dict(COMMON),
        dict(COMMON, max_assets=5, execution="next_open", allow_short=True),
        dict(
            COMMON,
            execution="close",
            privileged_state=False,
            holding_cost=BorrowFee(),
            shortable="shortable",
            allow_short=True,
        ),
        {"model": ListingProcess(GBM(mu=0.05, sigma=0.2, n_assets=6)), "episode_length": 30},
    ],
)
def test_check_env_passes(kwargs: dict[str, Any]) -> None:
    assert sekai.check_env(sekai.make("Market-v1", **kwargs), check_determinism=True) == []


def test_native_vectorisation_matches_separate_envs() -> None:
    kw = dict(COMMON, max_assets=6, sampler="sequential", execution="next_open", allow_short=True)
    native = sekai.make_vec("Market-v1", num_envs=4, **kw)
    sync = sekai.make_vec("Market-v1", num_envs=4, vectorization_mode="sync", **kw)
    assert isinstance(native, MarketVecEnv)
    a, b = native.reset(seed=3), sync.reset(seed=3)
    np.testing.assert_allclose(
        a.observations["assets"]["values"], b.observations["assets"]["values"]
    )
    rng = np.random.default_rng(0)
    for _ in range(150):  # crosses episode boundaries (autoreset)
        act = rng.uniform(-1, 1, (4, 6)).astype(np.float32)
        x, y = native.step(act), sync.step(act)
        np.testing.assert_allclose(x.rewards, y.rewards, atol=1e-15)
        np.testing.assert_array_equal(x.dones, y.dones)
        np.testing.assert_allclose(
            x.observations["assets"]["values"], y.observations["assets"]["values"]
        )
        for i in range(4):
            assert x.infos[i]["equity"] == pytest.approx(y.infos[i]["equity"], abs=1e-15)
            if x.dones[i]:
                np.testing.assert_allclose(
                    x.infos[i]["final_observation"]["portfolio"],
                    y.infos[i]["final_observation"]["portfolio"],
                )
    assert native.get_attr("max_assets") == [6] * 4
    native.close()
    sync.close()


def test_torch_float64_matches_numpy() -> None:
    torch = pytest.importorskip("torch")
    kw = dict(COMMON, max_assets=6, allow_short=True, sampler="sequential", risk=FactorRiskModel())
    a = BatchMarketEnv(5, backend="numpy", **kw)
    b = BatchMarketEnv(5, backend="torch", dtype=torch.float64, **kw)
    x, y = a.reset(seed=2), b.reset(seed=2)
    rng = np.random.default_rng(1)
    for _ in range(130):
        act = rng.uniform(-1, 1, (5, 6))
        x, y = a.step(act), b.step(torch.as_tensor(act))
        np.testing.assert_allclose(y.rewards.numpy(), x.rewards, atol=1e-13)
        for key in ("equity", "cost", "turnover", "financing", "variance", "weights", "exposures"):
            np.testing.assert_allclose(
                b.backend.to_numpy(y.info[key]), x.info[key], atol=1e-13, err_msg=key
            )
        np.testing.assert_array_equal(y.dones.numpy(), x.dones)
        np.testing.assert_array_equal(
            y.observations["assets"]["values"].numpy().astype(np.float32),
            x.observations["assets"]["values"],
        )
        np.testing.assert_array_equal(b.backend.to_numpy(b.slot_assets), a.slot_assets)


def test_float32_torch_default_and_differentiable_rollout() -> None:
    torch = pytest.importorskip("torch")
    rules = Constraints(max_gross=2.0, max_position=0.25, dollar_neutral=True)
    env = BatchMarketEnv(
        8,
        data=DATA,
        max_assets=8,
        lookback=10,
        episode_length=30,
        execution="close",
        action=ScoresLeverage(rules),
        transaction_cost=SpreadCost() + SquareRootImpact(0.3),
        backend="torch",
        autoreset=False,
    )
    assert env.panel.returns.dtype == torch.float32
    env.differentiable = True
    theta = torch.zeros(9, requires_grad=True)
    res = env.reset(seed=0)
    values = res.observations["assets"]["values"]
    total = torch.zeros(())
    k = 0
    while not bool(res.dones.all()):
        momentum = values[..., -5:, 0].sum(-1)
        action = torch.cat([theta[:8] + 10 * momentum, theta[8:].expand(8, 1)], dim=-1)
        res = env.step(action)
        values = res.observations["assets"]["values"]
        total = total + res.rewards.mean()
        k += 1
        if k == 15:  # truncate backpropagation through time
            env.detach_state()
            values = values.detach()
            assert not env.weights.requires_grad and not env.equity.requires_grad
    (-total).backward()
    assert (
        theta.grad is not None
        and bool(torch.isfinite(theta.grad).all())
        and float(theta.grad.abs().sum()) > 0
    )


class GaussianScenarios:
    """A toy generative model: i.i.d. normal returns, a volume field and a beta field."""

    def __init__(self, n_assets: int = 5, n_steps: int = 40) -> None:
        self.n_assets, self.n_steps = n_assets, n_steps
        self.calls: list[int] = []

    def sample(self, n_paths: int, *, seed: int) -> MarketPanel:
        self.calls.append(n_paths)
        rng = np.random.default_rng(seed)
        shape = (n_paths, self.n_steps, self.n_assets)
        return MarketPanel.from_returns(
            rng.normal(0.0005, 0.02, shape),
            fields={"beta": rng.normal(1, 0.2, shape), "dollar_volume": np.full(shape, 1e8)},
        )


def test_scenario_generators() -> None:
    gen = GaussianScenarios()
    env = BatchMarketEnv(
        4,
        scenarios=gen,
        lookback=5,
        episode_length=20,
        features=("beta",),
        max_drawdown=0.02,
        transaction_cost=SquareRootImpact(0.5) + SpreadCost(half_spread=1e-3),
    )
    env.reset(seed=0)
    first = np.array(env.panel.returns, copy=True)
    rng = np.random.default_rng(0)
    stopped = False
    for _ in range(25):
        res = env.step(rng.random((4, 6)) * 5)
        if res.reset_indices is not None and len(res.reset_indices) < 4:
            stopped = True
            kept = np.setdiff1d(np.arange(4), res.reset_indices)
            np.testing.assert_array_equal(env.panel.returns[kept], first[kept])
            first = np.array(env.panel.returns, copy=True)
    assert stopped, "some lanes should hit the drawdown stop before the horizon"
    assert gen.calls[0] == 1 and gen.calls[1] == 4  # a probe at construction, then all lanes
    a, b = (
        BatchMarketEnv(2, scenarios=GaussianScenarios(), lookback=5, episode_length=20),
        BatchMarketEnv(2, scenarios=GaussianScenarios(), lookback=5, episode_length=20),
    )
    np.testing.assert_array_equal(
        a.reset(seed=9).observations["assets"]["values"],
        b.reset(seed=9).observations["assets"]["values"],
    )


def test_scenario_validation() -> None:
    with pytest.raises(KeyError, match="lack the fields"):
        BatchMarketEnv(
            2, scenarios=GaussianScenarios(), features=("alpha",), lookback=5, episode_length=10
        )
    with pytest.raises(ValueError, match="an episode needs"):
        BatchMarketEnv(2, scenarios=GaussianScenarios(n_steps=10), lookback=5, episode_length=10)

    class Wrong:
        def sample(self, n_paths: int, *, seed: int) -> Any:
            return np.zeros((n_paths, 50, 3))

    with pytest.raises(TypeError, match="must return a MarketPanel"):
        BatchMarketEnv(2, scenarios=Wrong(), lookback=5, episode_length=10)
    env = BatchMarketEnv(1, scenarios=GaussianScenarios(), lookback=5, episode_length=10)
    with pytest.raises(ValueError, match="needs historical data"):
        env.reset(seed=0, options={"start": 3})


def test_risk_model_variance_and_reward() -> None:
    env = BatchMarketEnv(
        3,
        data=DATA,
        lookback=5,
        episode_length=10,
        max_assets=6,
        allow_short=True,
        risk=FactorRiskModel(),
        reward=SimpleReturn() - 2.0 * PortfolioVariance(),
        execution="close",
    )
    env.reset(seed=0, options={"start": 100})
    rows = env._rows()
    slots = env.backend.to_numpy(env._slots)
    occ, cols = slots >= 0, np.where(slots >= 0, slots, 0)
    action = np.random.default_rng(0).uniform(-1, 1, (3, 6))
    held = TargetWeights()(action, env_context(env))
    res = env.step(action)
    loadings = np.where(occ[..., None], DATA.field("loadings")[rows[:, None], cols], 0.0)
    idio = np.where(occ, DATA.field("idio_var")[rows[:, None], cols], 0.0)
    cov = DATA.exogenous_series("factor_cov")[rows]
    want = portfolio_variance(held, loadings=loadings, factor_cov=cov, idio_var=idio)
    np.testing.assert_allclose(res.info["variance"], want, rtol=1e-12)
    np.testing.assert_allclose(
        res.info["exposures"], np.einsum("bn,bnk->bk", held, loadings), rtol=1e-12
    )
    growth = res.info["equity"]  # equity started at 1
    np.testing.assert_allclose(res.rewards, growth - 1.0 - 2.0 * want, rtol=1e-12)


def env_context(env: BatchMarketEnv) -> Any:
    from sekai.finance import ActionContext

    return ActionContext(
        backend=env.backend, tradable=env._tradable(env._rows()), weights=env.weights
    )


def test_holding_costs_shortability_and_financing() -> None:
    kw = dict(
        data=DATA,
        lookback=5,
        episode_length=30,
        max_assets=8,
        allow_short=True,
        execution="close",
        transaction_cost=NoCost(),
        sampler="sequential",
    )
    plain = BatchMarketEnv(2, **kw)
    fees = BatchMarketEnv(2, holding_cost=BorrowFee(), shortable="shortable", **kw)
    plain.reset(seed=0)
    fees.reset(seed=0)
    short = -np.ones((2, 8))
    for _ in range(10):
        cols = fees.backend.to_numpy(fees._slots)
        rows = fees._rows()  # the decision: the flags in force when the order is placed
        allowed = DATA.field("shortable")[rows[:, None], np.where(cols >= 0, cols, 0)] > 0
        a, b = plain.step(short), fees.step(short)
        assert np.all(b.info["holding_cost"] >= 0) and np.all(a.info["holding_cost"] == 0)
        assert np.all(b.info["weights"][~allowed] >= 0.0)
        assert np.any(b.info["weights"] < 0) and np.any(b.info["holding_cost"] > 0)


def test_privileged_state_and_vector_infos() -> None:
    vec = sekai.make_vec("Market-v1", num_envs=3, privileged_state=True, exogenous=("regime",))
    res = vec.reset(seed=0)
    assert set(res.infos[0]["state"]) >= {"regime"}
    out = vec.step(vec.batch_action_space.sample())
    assert "regime" in out.infos[2]["state"] and "market" in out.observations
    env = sekai.make("Market-v1", privileged_state=True)
    assert "regime" in env.reset(seed=0).info["state"]


def test_custom_observation_builder() -> None:
    class Weights(ObservationBuilder):
        def space(self, env: Any) -> sekai.Space[Any]:
            return sekai.Box(-np.inf, np.inf, shape=(env.n_slots + 1,), dtype=np.float32)

        def __call__(self, view: Any) -> Any:
            B = view.backend
            out = B.concat([view.weights, B.asfloat(view.tradable, view.weights)[:, :1]], axis=-1)
            return out.astype(np.float32)

    env = sekai.make("Market-v1", observation=Weights(), episode_length=20)
    assert sekai.check_env(env) == []
    with pytest.raises(ValueError, match="features= and exogenous="):
        BatchMarketEnv(1, observation=Weights(), features=("volume",))


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="Give one of"):
        BatchMarketEnv(1, data=DATA, model=GBM(mu=0, sigma=0.1, n_assets=2))
    with pytest.raises(ValueError, match="execution must be"):
        BatchMarketEnv(1, execution="tomorrow")
    with pytest.raises(ValueError, match="an episode needs"):
        BatchMarketEnv(1, data=DATA[:50], lookback=20, episode_length=40)
    with pytest.raises(ValueError, match="delisting_return"):
        BatchMarketEnv(1, delisting_return=-2.0)
    with pytest.raises(ValueError, match="max_drawdown"):
        BatchMarketEnv(1, max_drawdown=1.5)
    with pytest.raises(ValueError, match="lookback >= 2"):
        BatchMarketEnv(1, data=DATA, lookback=1, transaction_cost=SquareRootImpact(0.1))
    with pytest.raises(KeyError):
        BatchMarketEnv(1, data=DATA, features=("nope",))
    with pytest.raises(TypeError, match="make_vec"):
        sekai.envs.finance.make_market(num_envs=3)
    env = BatchMarketEnv(2, episode_length=5)
    env.reset(seed=0)
    with pytest.raises(sekai.InvalidAction):
        env.step(np.full((2, 11), np.nan))
    with pytest.raises(sekai.InvalidAction):
        env.step(np.zeros((3, 11)))


def test_info_and_introspection() -> None:
    env = sekai.make("Market-v1", **COMMON)
    res = env.reset(seed=0, options={"start": 42})
    assert res.info["timestamp"] == DATA.timestamps[42]
    assert set(res.info) >= {
        "equity",
        "turnover",
        "cost",
        "holding_cost",
        "financing",
        "gross_exposure",
        "net_exposure",
        "drawdown",
        "n_assets",
        "weights",
        "timestamp",
        "dt",
    }
    assert res.info["dt"] == pytest.approx(1 / DATA.periods_per_year)
    assert env.unwrapped.action_mask().shape == env.action_space.shape
    assert len(env.unwrapped.slot_assets) == DATA.n_assets
    assert env.unwrapped.lookback == 10 and env.unwrapped.episode_length == 60
    assert "pending" in res.observation
    with pytest.raises(AttributeError):
        env.unwrapped.batch_env.lookback = 3  # type: ignore[misc]
    assert isinstance(env.unwrapped.batch_env.reward_fn, LogReturn)


def test_volatility_field_prices_impact_and_step_return() -> None:
    """A `volatility` data field (e.g. a risk model's forecast) replaces the window estimate."""
    vol = np.full(DATA.prices.shape, 0.03)
    data = DATA.with_fields(volatility=vol, dollar_volume=np.full(DATA.prices.shape, 1e7))
    kw: dict[str, Any] = dict(
        lookback=1, episode_length=5, execution="close", action=TargetWeights(1.0)
    )
    env = BatchMarketEnv(1, data=data, transaction_cost=SquareRootImpact(0.5), capital=1e6, **kw)
    env.reset(seed=0, options={"start": 30})
    trade = np.zeros((1, DATA.n_assets))
    trade[0, 0] = 0.5
    out = env.step(trade)
    want = (
        0.5 * 0.03 * 0.5 * np.sqrt(0.5 * 1e6 / 1e7)
    )  # coefficient * vol * |w| * sqrt(participation)
    assert float(out.info["cost"][0]) == pytest.approx(want)
    assert float(out.info["return"][0]) == pytest.approx(float(out.info["equity"][0]) - 1.0)


def test_custom_cost_models_written_for_0_2_work_combined() -> None:
    """A cost model without `assets=` (sekai 0.2's interface) works on Market-v1, also
    inside a CompositeCost, which passes `assets` only to the models that take it."""
    from sekai.finance import CostModel, ProportionalCost

    calls = []

    class Flat(CostModel):
        def __call__(self, trades: Any, *, equity: Any = None) -> Any:  # type: ignore[override]
            calls.append(1)
            return 1e-4 * np.abs(np.asarray(trades)).sum(axis=-1)

    for cost in (Flat(), Flat() + ProportionalCost(1e-4), ProportionalCost(1e-4) + Flat()):
        env = sekai.make("Market-v1", transaction_cost=cost)
        env.reset(seed=0)
        env.step(env.action_space.sample())
    assert len(calls) == 3


def test_attributes_reach_the_batch_env_as_on_market_v0() -> None:
    """get_attr returns the objects themselves and set_attr changes the running env."""
    from sekai.finance import LogReturn, SimpleReturn

    def rewards(env: Any) -> list[float]:
        env.reset(seed=0)
        env.action_space.seed(0)
        return [float(env.step(env.action_space.sample()).reward) for _ in range(5)]

    env = sekai.make("Market-v1", features=("volume",), data=DATA)
    assert isinstance(env.get_attr("reward_fn"), LogReturn)
    assert env.unwrapped.features == ("volume",) and env.unwrapped.exogenous == ()
    env.set_attr("reward_fn", SimpleReturn())
    expected = sekai.make("Market-v1", features=("volume",), data=DATA, reward=SimpleReturn())
    assert rewards(env) == rewards(expected)
    env.unwrapped.delisting_return = -0.5
    assert env.unwrapped.batch_env.delisting_return == -0.5
    vec = sekai.make_vec("Market-v1", num_envs=2, data=DATA)
    assert all(isinstance(r, LogReturn) for r in vec.get_attr("reward_fn"))
