from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

import cloudpickle
import numpy as np
import pytest

from sekai.envs.finance.market import random_correlation, simulate_gbm
from sekai.finance.simulators import (
    GARCH,
    GBM,
    BlockBootstrap,
    CointegratedPair,
    Heston,
    ListingProcess,
    MarketModel,
    MeanReverting,
    MertonJumpDiffusion,
    Randomized,
    RegimeSwitching,
    Simulation,
)

CORR3 = random_correlation(np.random.default_rng(42), 3)
HISTORY = np.random.default_rng(7).normal(0.0, 0.01, size=(120, 3))


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def log_returns(prices: np.ndarray) -> np.ndarray:
    return np.diff(np.log(prices), axis=-2)


def excess_kurtosis(x: np.ndarray) -> float:
    x = x.ravel() - x.mean()
    return float(np.mean(x**4) / np.mean(x**2) ** 2 - 3.0)


def lag1_autocorr(x: np.ndarray) -> float:
    x = x - x.mean()
    return float(np.sum(x[1:] * x[:-1]) / np.sum(x * x))


def two_regimes(**kwargs: Any) -> RegimeSwitching:
    calm = GBM(0.1, [0.1, 0.15], corr=[[1.0, 0.3], [0.3, 1.0]])
    stressed = GBM(-0.3, [0.4, 0.5], corr=[[1.0, 0.8], [0.8, 1.0]])
    return RegimeSwitching([calm, stressed], [[0.98, 0.02], [0.05, 0.95]], **kwargs)


class ConstantModel(MarketModel):
    """A minimal user-defined model: flat prices of 1."""

    def __init__(self, n_assets: int = 2) -> None:
        self._n = n_assets

    @property
    def n_assets(self) -> int:
        return self._n

    @property
    def dt(self) -> float:
        return 1 / 252

    def simulate(
        self, rng: np.random.Generator, n_steps: int, *, n_paths: int | None = None
    ) -> Simulation:
        lead = () if n_paths is None else (n_paths,)
        return Simulation(prices=np.ones((*lead, n_steps + 1, self._n)))


MODELS: dict[str, Callable[[], MarketModel]] = {
    "gbm": lambda: GBM(0.05, 0.2),
    "gbm-corr": lambda: GBM([0.05, 0.1, 0.0], 0.2, s0=[10.0, 20.0, 30.0], corr=CORR3),
    "heston": lambda: Heston(n_assets=2, corr=[[1.0, 0.5], [0.5, 1.0]]),
    "merton": lambda: MertonJumpDiffusion(n_assets=3, jump_intensity=20.0),
    "regime": two_regimes,
    "garch": lambda: GARCH(n_assets=2, corr=[[1.0, -0.4], [-0.4, 1.0]]),
    "mean-reverting": lambda: MeanReverting(n_assets=2, s0=[50.0, 150.0]),
    "cointegrated": lambda: CointegratedPair(s0=(100.0, 80.0)),
    "bootstrap": lambda: BlockBootstrap(HISTORY, block_size=7, s0=[1.0, 2.0, 3.0]),
    "randomized": lambda: Randomized(
        lambda r: Heston(mu=r.uniform(-0.1, 0.1), theta=r.uniform(0.01, 0.09), n_assets=2)
    ),
    "listing": lambda: ListingProcess(Heston(n_assets=4), min_listed=1),
    "listing-nested": lambda: ListingProcess(
        ListingProcess(MeanReverting(n_assets=5), delisting_rate=3.0, min_listed=0),
        delisting_rate=3.0,
        min_listed=2,
    ),
}


@pytest.fixture(params=sorted(MODELS))
def model(request: pytest.FixtureRequest) -> MarketModel:
    return MODELS[request.param]()


# --------------------------------------------------------------------------------------------
# Contract shared by every model
# --------------------------------------------------------------------------------------------


class TestContract:
    @pytest.mark.parametrize("n_paths", [None, 1, 4])
    def test_shapes(self, model: MarketModel, n_paths: int | None) -> None:
        sim = model.simulate(rng(), 30, n_paths=n_paths)
        lead = (31,) if n_paths is None else (n_paths, 31)
        assert sim.prices.shape == (*lead, model.n_assets)
        assert sim.prices.dtype == np.float64
        assert sim.mask is None or (sim.mask.shape == sim.prices.shape and sim.mask.dtype == bool)
        spec = model.state_spec()
        assert set(sim.state) == set(spec)
        for key, trailing in spec.items():
            assert sim.state[key].shape == (*lead, *trailing)
            assert sim.state[key].dtype == np.float64

    def test_positive_where_listed(self, model: MarketModel) -> None:
        sim = model.simulate(rng(1), 200, n_paths=8)
        valid = np.ones(sim.prices.shape, bool) if sim.mask is None else sim.mask
        assert np.all(np.isfinite(sim.prices[valid])) and np.all(sim.prices[valid] > 0)
        assert np.all(np.isnan(sim.prices[~valid]))

    def test_initial_row_is_s0(self, model: MarketModel) -> None:
        s0 = getattr(model, "s0", None)
        if s0 is None:
            pytest.skip("model has no fixed s0")
        sim = model.simulate(rng(), 5, n_paths=3)
        np.testing.assert_array_equal(sim.prices[:, 0, :], np.broadcast_to(s0, (3, len(s0))))

    def test_reproducible(self, model: MarketModel) -> None:
        a = model.simulate(rng(3), 40, n_paths=2)
        b = model.simulate(rng(3), 40, n_paths=2)
        c = model.simulate(rng(4), 40, n_paths=2)
        np.testing.assert_array_equal(a.prices, b.prices)
        assert (a.mask is None) == (b.mask is None)
        if a.mask is not None and b.mask is not None:
            np.testing.assert_array_equal(a.mask, b.mask)
        for key in a.state:
            np.testing.assert_array_equal(a.state[key], b.state[key])
        assert not np.array_equal(a.prices, c.prices, equal_nan=True)

    def test_single_path_matches_seeded_rerun(self, model: MarketModel) -> None:
        a = model.simulate(rng(5), 10).prices
        b = model.simulate(rng(5), 10).prices
        np.testing.assert_array_equal(a, b)

    @pytest.mark.parametrize(
        ("n_steps", "n_paths", "error"),
        [
            (0, None, ValueError),
            (-3, 2, ValueError),
            (5, 0, ValueError),
            (2.5, None, TypeError),
            (True, None, TypeError),
            (5, 1.0, TypeError),
        ],
    )
    def test_rejects_bad_counts(
        self, model: MarketModel, n_steps: Any, n_paths: Any, error: type[Exception]
    ) -> None:
        with pytest.raises(error):
            model.simulate(rng(), n_steps, n_paths=n_paths)

    def test_numpy_integer_counts(self, model: MarketModel) -> None:
        sim = model.simulate(rng(), np.int64(3), n_paths=np.int32(2))
        assert sim.prices.shape[:2] == (2, 4)

    def test_repr_and_pickle(self, model: MarketModel) -> None:
        assert repr(model).startswith(type(model).__name__ + "(")
        clone = cloudpickle.loads(cloudpickle.dumps(model))
        np.testing.assert_array_equal(
            clone.simulate(rng(2), 10, n_paths=2).prices,
            model.simulate(rng(2), 10, n_paths=2).prices,
        )

    def test_immutable(self, model: MarketModel) -> None:
        with pytest.raises(AttributeError):
            model.foo = 1
        with pytest.raises(AttributeError):
            del model.dt  # type: ignore[misc]


# --------------------------------------------------------------------------------------------
# Individual models
# --------------------------------------------------------------------------------------------


class TestGBM:
    @pytest.mark.parametrize(
        ("s0", "mu", "sigma", "corr"),
        [
            (50.0, 0.05, 0.3, None),
            (100.0, 0.0, 0.2, [[1.0]]),
            (
                np.array([1.0, 2.0, 3.0]),
                np.array([0.1, 0.0, -0.1]),
                np.array([0.1, 0.2, 0.3]),
                CORR3,
            ),
            (100.0, np.array([0.1, 0.0, -0.1]), np.array([0.1, 0.2, 0.3]), None),
        ],
    )
    def test_bit_identical_to_simulate_gbm(self, s0: Any, mu: Any, sigma: Any, corr: Any) -> None:
        for seed in (0, 11):
            got = GBM(mu, sigma, s0=s0, corr=corr, dt=1 / 52).simulate(rng(seed), 300).prices
            want = simulate_gbm(rng(seed), 300, s0=s0, mu=mu, sigma=sigma, dt=1 / 52, corr=corr)
            np.testing.assert_array_equal(got, want.reshape(301, -1))

    def test_mean_terminal_price(self) -> None:
        sim = GBM(0.1, 0.3, dt=1 / 12).simulate(rng(1), 12, n_paths=20_000)
        assert sim.prices[:, -1, 0].mean() / 100 == pytest.approx(np.exp(0.1), rel=0.02)

    def test_correlation_and_volatility(self) -> None:
        model = GBM(0.0, [0.1, 0.2, 0.3], corr=CORR3)
        returns = log_returns(model.simulate(rng(2), 500, n_paths=40).prices).reshape(-1, 3)
        np.testing.assert_allclose(np.corrcoef(returns.T), CORR3, atol=0.02)
        np.testing.assert_allclose(returns.std(axis=0) / np.sqrt(model.dt), model.sigma, rtol=0.02)

    def test_asset_count_inference(self) -> None:
        assert GBM().n_assets == 1
        assert GBM(n_assets=4).mu.shape == (4,)
        assert GBM(corr=CORR3).n_assets == 3
        assert GBM([0.1], [0.1, 0.2]).n_assets == 2
        with pytest.raises(ValueError, match="assets"):
            GBM([0.1, 0.2], n_assets=3)
        with pytest.raises(ValueError, match="assets"):
            GBM([0.1, 0.2], [0.1, 0.2, 0.3])
        with pytest.raises(ValueError):
            GBM(np.zeros((2, 2)))

    def test_parameters_are_read_only(self) -> None:
        model = GBM([0.1, 0.2])
        with pytest.raises(ValueError):
            model.mu[0] = 1.0
        with pytest.raises(AttributeError):
            model.mu = np.zeros(2)

    @pytest.mark.parametrize(
        "corr",
        [
            [[1.0, 0.5], [0.4, 1.0]],  # not symmetric
            [[1.0, 1.0], [1.0, 1.0]],  # singular
            [[1.0, 2.0], [2.0, 1.0]],  # not positive definite
            [[2.0, 0.0], [0.0, 2.0]],  # not a correlation matrix
            [[1.0, np.nan], [np.nan, 1.0]],
            np.eye(3),  # wrong size
        ],
    )
    def test_rejects_bad_corr(self, corr: Any) -> None:
        with pytest.raises(ValueError, match="corr"):
            GBM(sigma=[0.2, 0.2], corr=corr)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"sigma": -0.1},
            {"s0": 0.0},
            {"dt": 0.0},
            {"dt": np.inf},
            {"mu": np.nan},
            {"n_assets": 0},
        ],
    )
    def test_validation(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            GBM(**kwargs)


class TestHeston:
    def test_variance_non_negative_when_feller_fails(self) -> None:
        model = Heston(kappa=1.0, theta=0.04, xi=1.0, n_assets=3)
        sim = model.simulate(rng(0), 1000, n_paths=20)
        assert np.all(sim.state["variance"] >= 0)
        assert np.any(sim.state["variance"] == 0)
        assert np.all(sim.prices > 0)

    def test_variance_averages_to_theta(self) -> None:
        model = Heston(mu=0.0, kappa=3.0, theta=0.04, xi=0.3, rho=-0.7)
        sim = model.simulate(rng(1), 2520, n_paths=200)
        variance = sim.state["variance"]
        np.testing.assert_array_equal(variance[:, 0, 0], 0.04)
        assert variance.mean() == pytest.approx(0.04, rel=0.05)
        realised = log_returns(sim.prices).var() / model.dt
        assert realised == pytest.approx(0.04, rel=0.05)

    def test_leverage_effect(self) -> None:
        sim = Heston(rho=-0.7).simulate(rng(2), 1000, n_paths=20)
        dlog = log_returns(sim.prices).ravel()
        dvar = np.diff(sim.state["variance"], axis=-2).ravel()
        assert np.corrcoef(dlog, dvar)[0, 1] < -0.5

    def test_v0(self) -> None:
        sim = Heston(v0=[0.0, 0.09]).simulate(rng(), 3)
        np.testing.assert_array_equal(sim.state["variance"][0], [0.0, 0.09])

    @pytest.mark.parametrize(
        "kwargs",
        [{"kappa": 0.0}, {"theta": -0.01}, {"xi": 0.0}, {"rho": 1.5}, {"v0": -0.1}],
    )
    def test_validation(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            Heston(**kwargs)


class TestMerton:
    def test_mean_terminal_price(self) -> None:
        model = MertonJumpDiffusion(0.05, 0.2, 5.0, -0.05, 0.1, dt=1 / 12)
        sim = model.simulate(rng(1), 12, n_paths=40_000)
        assert sim.prices[:, -1, 0].mean() / 100 == pytest.approx(np.exp(0.05), rel=0.02)

    def test_fat_tails(self) -> None:
        merton = MertonJumpDiffusion(0.0, 0.2, jump_intensity=10.0, jump_mean=-0.05)
        gbm = GBM(0.0, 0.2)
        k_merton = excess_kurtosis(log_returns(merton.simulate(rng(2), 1000, n_paths=50).prices))
        k_gbm = excess_kurtosis(log_returns(gbm.simulate(rng(2), 1000, n_paths=50).prices))
        assert abs(k_gbm) < 0.2
        assert k_merton > k_gbm + 5

    def test_jump_state(self) -> None:
        model = MertonJumpDiffusion(jump_intensity=20.0, n_assets=2)
        jumps = model.simulate(rng(3), 2520, n_paths=10).state["jumps"]
        np.testing.assert_array_equal(jumps[:, 0], 0.0)
        np.testing.assert_array_equal(jumps, np.round(jumps))
        assert jumps[:, 1:].mean() == pytest.approx(20.0 * model.dt, rel=0.05)

    def test_without_jumps_matches_gbm(self) -> None:
        merton = MertonJumpDiffusion(0.07, 0.25, jump_intensity=0.0, n_assets=2)
        gbm = GBM(0.07, 0.25, n_assets=2)
        np.testing.assert_allclose(
            merton.simulate(rng(4), 100).prices, gbm.simulate(rng(4), 100).prices, rtol=1e-12
        )

    @pytest.mark.parametrize(
        "kwargs", [{"sigma": -0.1}, {"jump_intensity": -1.0}, {"jump_std": -0.1}]
    )
    def test_validation(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            MertonJumpDiffusion(**kwargs)


class TestRegimeSwitching:
    def test_stationary_initial_distribution(self) -> None:
        np.testing.assert_allclose(two_regimes().initial, [5 / 7, 2 / 7])
        np.testing.assert_allclose(two_regimes(initial=[0.0, 1.0]).initial, [0.0, 1.0])

    def test_regime_frequencies_and_volatility(self) -> None:
        model = two_regimes()
        sim = model.simulate(rng(0), 2000, n_paths=100)
        regime = sim.state["regime"]
        assert set(np.unique(regime)) == {0.0, 1.0}
        assert np.mean(regime == 1) == pytest.approx(2 / 7, abs=0.02)
        returns = log_returns(sim.prices)
        for k, gbm in enumerate(model.regimes):
            in_regime = returns[regime[:, :-1] == k]
            np.testing.assert_allclose(
                in_regime.std(axis=0) / np.sqrt(model.dt), gbm.sigma, rtol=0.03
            )
            assert np.corrcoef(in_regime.T)[0, 1] == pytest.approx(gbm.corr[0, 1], abs=0.03)

    def test_initial_and_absorbing(self) -> None:
        model = RegimeSwitching(
            [GBM(0.0, 0.1), GBM(0.0, 0.3)], [[0.9, 0.1], [0.0, 1.0]], initial=[1.0, 0.0]
        )
        regime = model.simulate(rng(1), 500, n_paths=50).state["regime"]
        np.testing.assert_array_equal(regime[:, 0], 0.0)
        assert np.all(np.diff(regime, axis=1) >= 0)  # regime 1 is absorbing
        assert np.all(regime[:, -1] == 1.0)
        np.testing.assert_allclose(model.initial, [1.0, 0.0])

    def test_single_regime(self) -> None:
        sim = RegimeSwitching([GBM(0.0, 0.2, n_assets=2)], [[1.0]]).simulate(rng(), 10)
        assert sim.prices.shape == (11, 2)
        np.testing.assert_array_equal(sim.state["regime"], 0.0)

    @pytest.mark.parametrize(
        ("regimes", "transition", "initial"),
        [
            ([], np.zeros((0, 0)), None),
            ([GBM(), GBM(n_assets=2)], np.eye(2), None),
            ([GBM(), GBM(dt=1 / 52)], np.eye(2), None),
            ([GBM(), GBM(s0=50.0)], np.eye(2), None),
            ([GBM(), GBM()], [[0.5, 0.6], [0.5, 0.5]], None),
            ([GBM(), GBM()], [[1.5, -0.5], [0.5, 0.5]], None),
            ([GBM(), GBM()], np.eye(3), None),
            ([GBM(), GBM()], np.eye(2), [0.5, 0.6]),
            ([GBM(), GBM()], np.eye(2), [1.0]),
        ],
    )
    def test_validation(self, regimes: Any, transition: Any, initial: Any) -> None:
        with pytest.raises(ValueError):
            RegimeSwitching(regimes, transition, initial=initial)

    def test_rejects_non_gbm_regime(self) -> None:
        with pytest.raises(TypeError):
            RegimeSwitching([GBM(), Heston()], np.eye(2))  # type: ignore[list-item]


class TestGARCH:
    def test_unconditional_variance_and_clustering(self) -> None:
        model = GARCH(omega=1e-6, alpha=0.08, beta=0.9, mu=0.05)
        sim = model.simulate(rng(0), 1500, n_paths=200)
        uncond = 1e-6 / (1 - 0.08 - 0.9)
        np.testing.assert_allclose(model.unconditional_variance, uncond)
        eps = log_returns(sim.prices)[..., 0] - 0.05 * model.dt
        assert eps.var() == pytest.approx(uncond, rel=0.1)
        acf = np.mean([lag1_autocorr(path**2) for path in eps])
        assert acf > 0.1

    def test_variance_recursion(self) -> None:
        model = GARCH(omega=2e-6, alpha=[0.05, 0.1], beta=[0.9, 0.8], mu=0.1)
        sim = model.simulate(rng(1), 200, n_paths=3)
        h = sim.state["variance"]
        eps = log_returns(sim.prices) - 0.1 * model.dt
        np.testing.assert_allclose(h[:, 0], np.broadcast_to(model.unconditional_variance, (3, 2)))
        np.testing.assert_allclose(h[:, 1:], 2e-6 + model.alpha * eps**2 + model.beta * h[:, :-1])

    @pytest.mark.parametrize(
        "kwargs",
        [{"omega": 0.0}, {"alpha": -0.1}, {"beta": -0.1}, {"alpha": 0.2, "beta": 0.8}],
    )
    def test_validation(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            GARCH(**kwargs)


class TestMeanReverting:
    def test_reverts_to_mean(self) -> None:
        theta, sigma, horizon = 5.0, 0.2, 252
        model = MeanReverting(theta=theta, mean=100.0, sigma=sigma, s0=50.0)
        x_t = np.log(model.simulate(rng(0), horizon, n_paths=2000).prices[:, -1, 0])
        decay = np.exp(-theta * horizon * model.dt)
        expected = np.log(100.0) + (np.log(50.0) - np.log(100.0)) * decay
        assert x_t.mean() == pytest.approx(expected, abs=0.005)
        assert x_t.mean() == pytest.approx(np.log(100.0), abs=0.01)
        assert x_t.var() == pytest.approx(sigma**2 * (1 - decay**2) / (2 * theta), rel=0.08)

    @pytest.mark.parametrize("kwargs", [{"theta": 0.0}, {"mean": 0.0}, {"sigma": -0.2}])
    def test_validation(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            MeanReverting(**kwargs)


class TestCointegratedPair:
    def test_spread_state_matches_prices(self) -> None:
        model = CointegratedPair(s0=(120.0, 80.0))
        sim = model.simulate(rng(0), 500, n_paths=4)
        log_ratio = np.log(sim.prices[..., 0] / sim.prices[..., 1]) - np.log(120.0 / 80.0)
        np.testing.assert_allclose(log_ratio, sim.state["spread"], atol=1e-10)
        np.testing.assert_array_equal(sim.state["spread"][:, 0], 0.0)

    def test_spread_mean_reverts_while_prices_wander(self) -> None:
        model = CointegratedPair(sigma=0.2, theta=50.0, spread_sigma=0.1)
        sim = model.simulate(rng(1), 2000, n_paths=20)
        spread_acf = np.mean([lag1_autocorr(s) for s in sim.state["spread"]])
        price_acf = np.mean([lag1_autocorr(np.log(p)) for p in sim.prices[..., 0]])
        assert spread_acf < 0.9
        assert price_acf > 0.95
        terminal = model.simulate(rng(2), 300, n_paths=1000).state["spread"][:, -1]
        assert terminal.var() == pytest.approx(0.1**2 / (2 * 50.0), rel=0.15)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"theta": 0.0},
            {"sigma": -0.1},
            {"spread_sigma": -0.1},
            {"mu": [0.1, 0.2]},
            {"s0": (1.0, 2.0, 3.0)},
            {"s0": (1.0, -2.0)},
        ],
    )
    def test_validation(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            CointegratedPair(**kwargs)


class TestBlockBootstrap:
    def test_rows_and_blocks_come_from_history(self) -> None:
        history = np.random.default_rng(3).normal(0.0, 0.01, size=(30, 3))
        size = 7
        sim = BlockBootstrap(history, block_size=size).simulate(rng(0), 100, n_paths=5)
        wrapped = False
        for path in log_returns(sim.prices):
            distance = np.abs(path[:, None, :] - history[None, :, :]).max(axis=-1)
            rows = distance.argmin(axis=1)
            assert np.all(distance.min(axis=1) < 1e-10)
            for j in range(len(rows) - 1):
                if (j + 1) % size:
                    assert rows[j + 1] == (rows[j] + 1) % len(history)
                    wrapped |= bool(rows[j] == len(history) - 1)
        assert wrapped

    def test_one_dimensional_returns(self) -> None:
        model = BlockBootstrap(np.linspace(-0.01, 0.01, 50), block_size=50, s0=10.0)
        assert model.n_assets == 1
        sim = model.simulate(rng(), 120)
        assert sim.prices.shape == (121, 1) and sim.prices[0, 0] == 10.0

    @pytest.mark.parametrize(
        ("returns", "block_size"),
        [
            (np.zeros((10, 2)), 0),
            (np.zeros((10, 2)), 11),
            (np.array([0.01, np.nan, 0.0]), 1),
            (np.zeros((4, 2, 2)), 1),
            (np.zeros((0, 2)), 1),
        ],
    )
    def test_validation(self, returns: np.ndarray, block_size: int) -> None:
        with pytest.raises(ValueError):
            BlockBootstrap(returns, block_size=block_size)

    def test_block_size_must_be_integer(self) -> None:
        with pytest.raises(TypeError):
            BlockBootstrap(np.zeros(10), block_size=2.5)  # type: ignore[arg-type]


class TestRandomized:
    @staticmethod
    def factory(r: np.random.Generator) -> MarketModel:
        return GBM(sigma=r.uniform(0.05, 0.5), s0=r.uniform(50.0, 150.0), n_assets=2)

    def test_parameters_differ_across_paths(self) -> None:
        model = Randomized(self.factory)
        sim = model.simulate(rng(0), 20, n_paths=8)
        assert len(np.unique(sim.prices[:, 0, 0])) == 8
        assert model.n_assets == 2 and model.dt == 1 / 252 and model.state_spec() == {}

    def test_reference_and_draw_order(self) -> None:
        model = Randomized(self.factory)
        reference = self.factory(np.random.default_rng(0))
        assert isinstance(model.reference, GBM) and isinstance(reference, GBM)
        np.testing.assert_array_equal(model.reference.s0, reference.s0)
        single = model.simulate(rng(9), 15).prices
        batch = model.simulate(rng(9), 15, n_paths=1).prices
        np.testing.assert_array_equal(batch[0], single)

    def test_stacks_state_and_masks(self) -> None:
        model = Randomized(
            lambda r: ListingProcess(Heston(theta=r.uniform(0.01, 0.1), n_assets=3), min_listed=0)
        )
        sim = model.simulate(rng(1), 50, n_paths=4)
        assert sim.mask is not None and sim.mask.shape == (4, 51, 3)
        assert sim.state["variance"].shape == (4, 51, 3)
        np.testing.assert_array_equal(np.isnan(sim.prices), ~sim.mask)

    @pytest.mark.parametrize(
        "models",
        [
            [GBM(), GBM(n_assets=2)],
            [GBM(), GBM(dt=1 / 52)],
            [Heston(), GBM()],
        ],
    )
    def test_mismatch_raises(self, models: list[MarketModel]) -> None:
        counter = itertools.count()
        model = Randomized(lambda r: models[min(next(counter), 1)])
        with pytest.raises(ValueError, match="reference"):
            model.simulate(rng(), 5, n_paths=3)

    def test_bad_factory(self) -> None:
        with pytest.raises(TypeError):
            Randomized(lambda r: "not a model")  # type: ignore[arg-type, return-value]
        with pytest.raises(TypeError):
            Randomized(GBM())  # type: ignore[arg-type]


class TestListingProcess:
    def test_mask_and_nan_agree(self) -> None:
        sim = ListingProcess(GBM(n_assets=6), delisting_rate=2.0).simulate(rng(0), 500, n_paths=5)
        assert sim.mask is not None
        np.testing.assert_array_equal(np.isnan(sim.prices), ~sim.mask)
        assert sim.mask.any() and not sim.mask.all()

    def test_min_listed_respected(self) -> None:
        model = ListingProcess(
            GBM(n_assets=6), initial_listed=0.1, listing_rate=0.5, delisting_rate=5.0, min_listed=3
        )
        mask = model.simulate(rng(1), 1000, n_paths=10).mask
        assert mask is not None
        assert np.all(mask.sum(axis=-1) >= 3)
        assert np.any(mask.sum(axis=-1) == 3)

    def test_min_listed_fills_lowest_indices(self) -> None:
        model = ListingProcess(
            GBM(n_assets=5), initial_listed=0.0, listing_rate=0.0, delisting_rate=5.0, min_listed=2
        )
        mask = model.simulate(rng(2), 300, n_paths=3).mask
        assert mask is not None
        assert np.all(mask[..., :2]) and not np.any(mask[..., 2:])

    def test_long_run_listed_fraction(self) -> None:
        model = ListingProcess(
            GBM(n_assets=10),
            initial_listed=2 / 3,
            listing_rate=2.0,
            delisting_rate=1.0,
            min_listed=0,
        )
        mask = model.simulate(rng(3), 2520, n_paths=20).mask
        assert mask is not None
        assert mask.mean() == pytest.approx(2.0 / (2.0 + 1.0), abs=0.02)

    def test_always_listed(self) -> None:
        model = ListingProcess(GBM(n_assets=3), initial_listed=1.0, delisting_rate=0.0)
        sim = model.simulate(rng(), 100)
        assert sim.mask is not None and sim.mask.all()
        np.testing.assert_array_equal(sim.prices, GBM(n_assets=3).simulate(rng(), 100).prices)

    def test_passes_through_inner_path_and_state(self) -> None:
        inner = Heston(n_assets=3)
        outer = ListingProcess(inner, min_listed=1).simulate(rng(4), 200, n_paths=2)
        plain = inner.simulate(rng(4), 200, n_paths=2)
        assert outer.mask is not None
        np.testing.assert_array_equal(outer.state["variance"], plain.state["variance"])
        np.testing.assert_array_equal(outer.prices[outer.mask], plain.prices[outer.mask])

    def test_combines_inner_mask(self) -> None:
        inner = ListingProcess(GBM(n_assets=5), delisting_rate=3.0, min_listed=0)
        inner_mask = inner.simulate(rng(5), 400, n_paths=3).mask
        outer_mask = (
            ListingProcess(inner, delisting_rate=3.0, min_listed=0)
            .simulate(rng(5), 400, n_paths=3)
            .mask
        )
        assert inner_mask is not None and outer_mask is not None
        assert np.all(inner_mask | ~outer_mask)  # outer is a subset of inner
        assert np.any(inner_mask & ~outer_mask)
        # With every own flag forced on, the combined mask is exactly the inner one.
        forced = ListingProcess(inner, min_listed=5).simulate(rng(5), 400, n_paths=3).mask
        assert forced is not None
        np.testing.assert_array_equal(forced, inner_mask)

    def test_wraps_user_model(self) -> None:
        sim = ListingProcess(ConstantModel(4), min_listed=4).simulate(rng(), 10, n_paths=2)
        assert sim.mask is not None and sim.mask.all()
        np.testing.assert_array_equal(sim.prices, 1.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"initial_listed": 1.5},
            {"initial_listed": -0.1},
            {"listing_rate": -1.0},
            {"delisting_rate": -1.0},
            {"min_listed": 7},
            {"min_listed": -1},
        ],
    )
    def test_validation(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            ListingProcess(GBM(n_assets=6), **kwargs)

    def test_rejects_non_model(self) -> None:
        with pytest.raises(TypeError):
            ListingProcess("GBM")  # type: ignore[arg-type]


class TestSimulation:
    def test_validates_shapes(self) -> None:
        prices = np.ones((5, 2))
        Simulation(prices, np.ones((5, 2), bool), {"x": np.zeros(5), "y": np.zeros((5, 2, 3))})
        with pytest.raises(ValueError):
            Simulation(np.ones(5))
        with pytest.raises(ValueError):
            Simulation(prices, np.ones((5, 3), bool))
        with pytest.raises(ValueError):
            Simulation(prices, np.ones((5, 2)))  # not boolean
        with pytest.raises(ValueError):
            Simulation(prices, state={"x": np.zeros(4)})

    def test_user_model_defaults(self) -> None:
        model = ConstantModel(3)
        assert model.state_spec() == {}
        sim = model.simulate(rng(), 4, n_paths=2)
        assert sim.mask is None and dict(sim.state) == {}
