from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

import sekai
from sekai.envs.finance import (
    MarketMakingEnv,
    OptimalExecutionEnv,
    PortfolioEnv,
    TradingEnv,
    random_correlation,
    simulate_gbm,
)
from sekai.finance import GBM

FINANCE = [
    "Trading-v0",
    "Portfolio-v0",
    "MarketMaking-v0",
    "OptimalExecution-v0",
    "Market-v0",
    "LimitOrderBook-v0",
]


@pytest.mark.parametrize("env_id", FINANCE)
def test_check_env(env_id: str) -> None:
    assert sekai.check_env(sekai.make(env_id), n_steps=400, check_determinism=True) == []


@pytest.mark.parametrize("env_id", FINANCE)
def test_episode_ends(env_id: str) -> None:
    env = sekai.make(env_id)
    env.reset(seed=0)
    for _ in range(10_000):
        if env.step(env.action_space.sample()).done:
            return
    pytest.fail("episode never ended")


class TestSimulation:
    def test_gbm_shapes_and_positivity(self) -> None:
        rng = np.random.default_rng(0)
        single = simulate_gbm(rng, 100)
        assert single.shape == (101,) and single[0] == 100.0 and np.all(single > 0)
        multi = simulate_gbm(
            rng, 50, s0=[1, 2, 3], sigma=[0.1, 0.2, 0.3], corr=random_correlation(rng, 3)
        )
        assert multi.shape == (51, 3)
        np.testing.assert_array_equal(multi[0], [1, 2, 3])

    def test_gbm_drift(self) -> None:
        rng = np.random.default_rng(1)
        paths = np.stack([simulate_gbm(rng, 252, mu=0.1, sigma=0.2) for _ in range(2000)])
        assert paths[:, -1].mean() / 100 == pytest.approx(np.exp(0.1), rel=0.02)

    def test_correlation(self) -> None:
        rng = np.random.default_rng(2)
        corr = np.array([[1.0, 0.8], [0.8, 1.0]])
        paths = simulate_gbm(rng, 20_000, s0=[1.0, 1.0], sigma=[0.2, 0.2], corr=corr)
        returns = np.diff(np.log(paths), axis=0)
        assert np.corrcoef(returns.T)[0, 1] == pytest.approx(0.8, abs=0.03)

    def test_random_correlation_is_valid(self) -> None:
        c = random_correlation(np.random.default_rng(0), 6)
        np.testing.assert_allclose(np.diag(c), 1.0)
        assert np.all(np.linalg.eigvalsh(c) > 0)


class TestTrading:
    def test_buy_and_hold_reward_is_log_growth(self) -> None:
        env = TradingEnv(transaction_cost=1e-3)
        env.reset(seed=1)
        p0, total = env.price, 0.0
        while True:
            r = env.step(2)  # long
            total += r.reward
            if r.done:
                break
        assert r.truncated and not r.terminated
        assert total == pytest.approx(np.log(env.price / p0) + np.log(1 - 1e-3))
        assert r.info["equity"] == pytest.approx(np.exp(total))

    def test_flat_earns_nothing(self) -> None:
        env = TradingEnv()
        env.reset(seed=0)
        assert all(env.step(1).reward == 0.0 for _ in range(10))

    def test_historical_prices(self) -> None:
        prices = np.linspace(100, 200, 400)
        env = TradingEnv(prices, episode_length=50, lookback=10, positions=(0, 1))
        obs = env.reset(seed=0).observation
        assert obs.shape == (11,) and np.all(obs[:10] > 0)  # rising prices
        assert env.step(1).reward > 0

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            TradingEnv(np.ones(10), episode_length=50)
        with pytest.raises(ValueError):
            TradingEnv(positions=(1,))
        with pytest.raises(ValueError):
            TradingEnv(np.full(500, -1.0))

    def test_position_drifts_and_rebalancing_back_is_charged(self) -> None:
        prices = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.02, 300)))
        env = TradingEnv(prices, episode_length=50, lookback=5, transaction_cost=1e-3, sampler=0)
        env.reset(seed=0)
        first = env.step(0)  # 100% short
        r = first.info["asset_return"]
        drifted = -(1 + r) / (1 - r)  # the short's value over the new equity
        assert first.info["position"] == pytest.approx(drifted, rel=1e-12)
        assert first.observation[-1] == pytest.approx(drifted, rel=1e-6)
        again = env.step(0)  # back to -1: the drift is traded and paid for
        assert again.info["turnover"] == pytest.approx(abs(-1 - drifted), rel=1e-12)
        assert again.info["cost"] == pytest.approx(1e-3 * abs(-1 - drifted), rel=1e-12)
        env.reset(seed=0)  # long and flat positions do not drift
        assert [env.step(2).info["position"] for _ in range(3)] == [1.0, 1.0, 1.0]
        assert env.step(2).info["turnover"] == 0.0 and env.step(1).info["position"] == 0.0

    def test_matches_a_one_asset_portfolio(self) -> None:
        prices = 100 * np.exp(np.cumsum(np.random.default_rng(1).normal(0, 0.02, 300)))
        common = {"episode_length": 120, "lookback": 5, "transaction_cost": 1e-3, "sampler": 0}
        trading = TradingEnv(prices, **common)
        portfolio = PortfolioEnv(prices[:, None], allow_short=True, **common)
        trading.reset(seed=0)
        portfolio.reset(seed=0)
        for a in np.random.default_rng(2).integers(0, 3, 120):
            x = trading.step(int(a))
            y = portfolio.step(np.array([trading.positions[a]], np.float32))
            assert x.reward == pytest.approx(y.reward, rel=1e-12, abs=1e-15)
            assert x.info["equity"] == pytest.approx(y.info["equity"], rel=1e-12)
            assert x.info["turnover"] == pytest.approx(y.info["turnover"], rel=1e-12, abs=1e-15)
            assert x.info["position"] == pytest.approx(y.info["weights"][0], rel=1e-12, abs=1e-15)

    def test_observation_bounds_cover_drifted_positions(self) -> None:
        def bounds(positions: tuple[float, ...]) -> tuple[float, float]:
            space = TradingEnv(positions=positions).observation_space
            return float(space.low[-1]), float(space.high[-1])

        assert bounds((0.0, 1.0)) == bounds((0.0, 0.5)) == (0.0, 1.0)
        assert bounds((-1.0, 0.0, 1.0)) == (-np.inf, 1.0)
        assert bounds((0.0, 2.0)) == (0.0, np.inf)
        levered = TradingEnv(positions=(-2.0, 0.0, 2.0), sigma=1.5)
        assert sekai.check_env(levered, n_steps=500) == []

    def test_dt_follows_the_model(self) -> None:
        intraday = 1 / (252 * 78)
        assert TradingEnv(model=GBM(n_assets=1, dt=intraday)).reset(seed=0).info["dt"] == intraday
        portfolio = PortfolioEnv(model=GBM(n_assets=3, dt=intraday))
        assert portfolio.reset(seed=0).info["dt"] == intraday
        assert TradingEnv(model=GBM(n_assets=1, dt=intraday), dt=0.5).dt == 0.5  # explicit wins
        vec = sekai.make_vec("Trading-v0", 2, model=GBM(n_assets=1, dt=intraday))
        assert vec.reset(seed=0).infos[0]["dt"] == intraday


class TestPortfolio:
    def test_weights_long_only(self) -> None:
        env = PortfolioEnv(n_assets=3)
        np.testing.assert_allclose(
            env.action_to_weights(np.array([1, 1, 0, 2])), [0.25, 0.25, 0, 0.5]
        )
        np.testing.assert_allclose(env.action_to_weights(np.zeros(4)), [0, 0, 0, 1])

    def test_weights_long_short(self) -> None:
        env = PortfolioEnv(n_assets=2, allow_short=True)
        np.testing.assert_allclose(env.action_to_weights(np.array([1.0, -1.0])), [0.5, -0.5, 1.0])
        np.testing.assert_allclose(env.action_to_weights(np.array([0.2, 0.3])), [0.2, 0.3, 0.5])

    def test_all_cash_is_riskless(self) -> None:
        env = PortfolioEnv()
        env.reset(seed=0)
        cash = np.zeros(6, np.float32)
        cash[-1] = 1
        for _ in range(20):
            r = env.step(cash)
            assert r.reward == 0.0 and r.info["turnover"] == 0.0

    def test_turnover_cost_and_value(self) -> None:
        env = PortfolioEnv(n_assets=2, transaction_cost=0.01)
        env.reset(seed=0)
        r = env.step(np.array([1, 0, 0], np.float32))
        assert r.info["turnover"] == pytest.approx(1.0) and r.info["cost"] == pytest.approx(0.01)
        value = 1.0
        env.reset(seed=0)
        total = 0.0
        while True:
            r = env.step(np.array([1, 1, 1], np.float32))
            total += r.reward
            if r.done:
                break
        value = r.info["portfolio_value"]
        assert value == pytest.approx(np.exp(total))

    def test_historical_prices_set_n_assets(self) -> None:
        prices = np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, (300, 4)), axis=0))
        env = PortfolioEnv(prices, episode_length=100, lookback=20)
        assert env.n_assets == 4
        assert env.reset(seed=0).observation["returns"].shape == (20, 4)

    def test_invalid_action_shape(self) -> None:
        env = PortfolioEnv(n_assets=2)
        env.reset(seed=0)
        with pytest.raises(sekai.InvalidAction):
            env.step(np.ones(5, np.float32))


class TestMarketMaking:
    def test_avellaneda_stoikov_beats_naive_quotes(self) -> None:
        env = MarketMakingEnv()

        def mean_reward(policy: str) -> float:
            totals = []
            for seed in range(100):
                env.reset(seed=seed)
                total = 0.0
                while True:
                    a = (
                        env.avellaneda_stoikov_action()
                        if policy == "as"
                        else np.full(2, 0.2, np.float32)
                    )
                    r = env.step(a)
                    total += r.reward
                    if r.done:
                        break
                totals.append(total)
            return float(np.mean(totals))

        assert mean_reward("as") > mean_reward("naive") + 10

    def test_inventory_limit(self) -> None:
        env = MarketMakingEnv(max_inventory=3)
        env.reset(seed=0)
        for _ in range(100):  # bid at the mid, never quote the ask: inventory climbs to the cap
            r = env.step(np.array([0.0, 5.0], np.float32))
            assert r.info["inventory"] <= 3
        assert r.info["inventory"] == 3

    def test_terminates_at_horizon(self) -> None:
        env = MarketMakingEnv(n_steps=10)
        env.reset(seed=0)
        flags = [env.step(np.ones(2, np.float32)).terminated for _ in range(10)]
        assert flags == [False] * 9 + [True]

    def test_skew_direction(self) -> None:
        env = MarketMakingEnv()
        env.reset(seed=0)
        env.inventory = 10  # long: quote lower to sell, i.e. wider bid, tighter ask
        bid, ask = env.avellaneda_stoikov_action()
        assert bid > ask


class TestOptimalExecution:
    def test_twap_when_risk_neutral(self) -> None:
        env = OptimalExecutionEnv(risk_aversion=0.0, n_steps=4)
        np.testing.assert_allclose(env.almgren_chriss_trajectory(), [1e6, 7.5e5, 5e5, 2.5e5, 0])

    def test_risk_aversion_front_loads(self) -> None:
        x = OptimalExecutionEnv(risk_aversion=1e-6).almgren_chriss_trajectory()
        trades = -np.diff(x)
        assert x[0] == 1e6 and x[-1] == pytest.approx(0, abs=1e-6)
        assert np.all(np.diff(trades) < 0)

    def test_trajectory_matches_the_closed_form_and_stays_finite(self) -> None:
        env = OptimalExecutionEnv(risk_aversion=1e-5)  # kappa * T is about 9.5
        t = np.arange(env.n_steps + 1) * env.tau
        eta_tilde = env.eta - 0.5 * env.gamma * env.tau
        kappa_tilde_sq = env.risk_aversion * env.sigma**2 / eta_tilde
        kappa = np.arccosh(1 + 0.5 * kappa_tilde_sq * env.tau**2) / env.tau
        closed_form = (
            env.X0 * np.sinh(kappa * (env.total_time - t)) / np.sinh(kappa * env.total_time)
        )
        np.testing.assert_allclose(
            env.almgren_chriss_trajectory(), closed_form, rtol=1e-10, atol=1e-6
        )
        steep = OptimalExecutionEnv(risk_aversion=0.1, n_steps=390)  # kappa * T > 1000
        x = steep.almgren_chriss_trajectory()
        assert np.all(np.isfinite(x)) and x[0] == steep.X0 and np.all(np.diff(x) <= 0)
        assert x[-1] == pytest.approx(0, abs=1e-6)
        assert np.all(np.isfinite(steep.almgren_chriss_actions()))

    @pytest.mark.parametrize(
        "params",
        [
            {},
            {  # one day in minute bars
                "total_time": 1.0,
                "n_steps": 390,
                "sigma": 0.05,
                "permanent_impact": 2.5e-8,
                "temporary_impact": 2.5e-6,
            },
        ],
    )
    def test_trajectory_is_continuous_in_risk_aversion(self, params: dict[str, float]) -> None:
        holdings = []
        for risk_aversion in [0.0, 1e-20, 1e-14, 1e-8, 1e-4, 0.1]:
            env = OptimalExecutionEnv(risk_aversion=risk_aversion, **params)
            x = env.almgren_chriss_trajectory()
            assert np.all(np.isfinite(x)) and x[0] == env.X0 and np.all(np.diff(x) <= 0)
            t = np.arange(env.n_steps + 1) * env.tau
            twap = env.X0 * (1 - t / env.total_time)
            eta_tilde = env.eta - 0.5 * env.gamma * env.tau
            kappa_tilde_sq = risk_aversion * env.sigma**2 / eta_tilde
            if 0.5 * kappa_tilde_sq * env.tau**2 > 1e-6:  # well conditioned: the closed form
                kappa = np.arccosh(1 + 0.5 * kappa_tilde_sq * env.tau**2) / env.tau
                if kappa * env.total_time < 700:
                    closed = np.sinh(kappa * (env.total_time - t)) / np.sinh(kappa * env.total_time)
                    np.testing.assert_allclose(x, env.X0 * closed, rtol=1e-9, atol=1e-6)
            elif risk_aversion <= 1e-14:  # nearly risk neutral: TWAP
                np.testing.assert_allclose(x, twap, rtol=1e-6, atol=1e-6)
                actions = env.almgren_chriss_actions()[:3, 0]
                n = env.n_steps
                np.testing.assert_allclose(actions, [1 / n, 1 / (n - 1), 1 / (n - 2)], rtol=1e-5)
            holdings.append(x)
        # More risk aversion sells sooner: holdings fall at every time, never jump back up.
        assert all(np.all(b <= a + 1e-9 * a[0]) for a, b in pairwise(holdings))

    def test_monte_carlo_matches_closed_form(self) -> None:
        env = OptimalExecutionEnv()
        actions = env.almgren_chriss_actions()
        shortfalls = []
        for seed in range(1500):
            env.reset(seed=seed)
            for a in actions:
                r = env.step(a)
            assert r.terminated and r.info["inventory"] == 0
            shortfalls.append(r.info["implementation_shortfall"])
        x = env.almgren_chriss_trajectory()
        mean, var = env.expected_shortfall(x[:-1] - x[1:])
        stderr = np.sqrt(var / len(shortfalls))
        assert abs(np.mean(shortfalls) - mean) < 4 * stderr
        assert np.std(shortfalls) == pytest.approx(np.sqrt(var), rel=0.1)

    def test_optimal_schedule_beats_twap_on_objective(self) -> None:
        env = OptimalExecutionEnv()
        twap = np.array([[1 / (env.n_steps - k)] for k in range(env.n_steps)], np.float32)

        def objective(actions: np.ndarray) -> float:
            totals = []
            for seed in range(300):
                env.reset(seed=seed)
                totals.append(sum(env.step(a).reward for a in actions))
            return float(np.mean(totals))

        assert objective(env.almgren_chriss_actions()) > objective(twap)

    def test_forced_liquidation_at_horizon(self) -> None:
        env = OptimalExecutionEnv(n_steps=3)
        env.reset(seed=0)
        env.step(np.zeros(1, np.float32))
        env.step(np.zeros(1, np.float32))
        r = env.step(np.zeros(1, np.float32))
        assert r.terminated and r.info["shares_sold"] == 1e6 and r.info["inventory"] == 0

    def test_selling_everything_terminates_early(self) -> None:
        env = OptimalExecutionEnv()
        env.reset(seed=0)
        assert env.step(np.ones(1, np.float32)).terminated


@pytest.mark.parametrize("env_id", FINANCE)
def test_elapsed_steps(env_id: str) -> None:
    env = sekai.make(env_id)
    env.reset(seed=0)
    assert env.elapsed_steps == 0
    env.step(env.action_space.sample())
    env.step(env.action_space.sample())
    assert env.elapsed_steps == 2
