"""`MarketModel.simulate_torch`: the market models simulated with torch, on a device.

The same model as `simulate` (distributions, parameters, shapes, mask and state), not the
same numbers: checked by shapes, and by statistics of large samples against the NumPy
simulation and the models' defining properties, with tolerances of several standard errors
(so they hold for any seed). Then the paths' way into environments: `MarketPanel`,
`ModelScenarios(device=...)` and `BatchMarketEnv(simulate_on_device=True)`, whose resets copy
nothing to the device (counted as in test_device_syncs.py).
"""

from __future__ import annotations

import copy
import gc
import math
import pickle
import subprocess
import sys
import weakref
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import sekai
from sekai.envs.finance import BatchMarketEnv
from sekai.envs.finance.market_env import default_market
from sekai.finance import MarketData, MarketPanel, ModelScenarios
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
from tests.test_finance_simulators import CORR3, MODELS, ConstantModel, lag1_autocorr, two_regimes

torch = pytest.importorskip("torch")
from tests.test_device_syncs import Waits  # noqa: E402  (needs torch)

ALL_MODELS: dict[str, Callable[[], MarketModel]] = {**MODELS, "default-market": default_market}

GPUS = [
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
    ),
    pytest.param(
        "mps",
        marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device"),
    ),
]


def gen(seed: int = 0, device: str = "cpu") -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def host(x: Any) -> np.ndarray:
    """A tensor's (or array's) values in NumPy, as float64 (bools stay bools)."""
    out = np.asarray(x) if isinstance(x, np.ndarray) else x.detach().cpu().numpy()
    return out if out.dtype == np.bool_ else out.astype(np.float64)


def log_returns(prices: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.diff(np.log(prices), axis=-2)


@pytest.fixture(params=sorted(ALL_MODELS))
def model(request: pytest.FixtureRequest) -> MarketModel:
    return ALL_MODELS[request.param]()


# --------------------------------------------------------------------------------------------
# Contract shared by every model
# --------------------------------------------------------------------------------------------


class TestContract:
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize("n_paths", [None, 1, 4])
    def test_shapes_dtypes_and_device(
        self, model: MarketModel, n_paths: int | None, dtype: str
    ) -> None:
        want = model.simulate(np.random.default_rng(0), 30, n_paths=n_paths)
        sim = model.simulate_torch(gen(), 30, n_paths=n_paths, dtype=getattr(torch, dtype))
        assert isinstance(sim.prices, torch.Tensor) and sim.prices.dtype == getattr(torch, dtype)
        assert tuple(sim.prices.shape) == want.prices.shape and sim.prices.device.type == "cpu"
        assert (sim.mask is None) == (want.mask is None)
        if sim.mask is not None:
            assert sim.mask.dtype == torch.bool and tuple(sim.mask.shape) == want.mask.shape
        assert sim.state.keys() == want.state.keys() == model.state_spec().keys()
        for key, value in sim.state.items():
            assert tuple(value.shape) == want.state[key].shape
            assert value.dtype == getattr(torch, dtype)

    def test_default_dtype(self, model: MarketModel) -> None:
        assert model.simulate_torch(gen(), 3).prices.dtype == torch.get_default_dtype()

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_positive_where_listed_and_nan_elsewhere(self, model: MarketModel, dtype: str) -> None:
        sim = model.simulate_torch(gen(1), 200, n_paths=8, dtype=getattr(torch, dtype))
        prices = host(sim.prices)
        valid = np.ones(prices.shape, bool) if sim.mask is None else host(sim.mask)
        assert np.all(np.isfinite(prices[valid])) and np.all(prices[valid] > 0)
        assert np.all(np.isnan(prices[~valid]))
        for value in sim.state.values():
            assert np.all(np.isfinite(host(value)))

    def test_initial_row_is_s0(self, model: MarketModel) -> None:
        s0 = getattr(model, "s0", None)
        if s0 is None:
            pytest.skip("model has no fixed s0")
        for dtype in (torch.float32, torch.float64):
            prices = model.simulate_torch(gen(), 5, n_paths=3, dtype=dtype).prices
            want = torch.tensor(np.broadcast_to(s0, (3, len(s0))).copy(), dtype=dtype)
            assert torch.equal(prices[:, 0, :], want)

    def test_deterministic_for_a_generator_state(self, model: MarketModel) -> None:
        a = model.simulate_torch(gen(3), 40, n_paths=2)
        b = model.simulate_torch(gen(3), 40, n_paths=2)
        torch.testing.assert_close(a.prices, b.prices, rtol=0, atol=0, equal_nan=True)
        assert (a.mask is None) == (b.mask is None)
        if a.mask is not None:
            assert torch.equal(a.mask, b.mask)
        for key in a.state:
            assert torch.equal(a.state[key], b.state[key])
        g = gen(3)
        first = model.simulate_torch(g, 40, n_paths=2).prices
        second = model.simulate_torch(g, 40, n_paths=2).prices  # the generator moved on
        other = model.simulate_torch(gen(4), 40, n_paths=2).prices
        for c in (second, other):
            assert not torch.equal(torch.nan_to_num(c), torch.nan_to_num(first))

    def test_a_single_path_is_a_batch_of_one_without_its_axis(self, model: MarketModel) -> None:
        one = model.simulate_torch(gen(5), 10)
        batch = model.simulate_torch(gen(5), 10, n_paths=1)
        torch.testing.assert_close(one.prices, batch.prices[0], rtol=0, atol=0, equal_nan=True)
        for key in one.state:
            assert torch.equal(one.state[key], batch.state[key][0])

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
            model.simulate_torch(gen(), n_steps, n_paths=n_paths)

    def test_rejects_a_numpy_generator_and_dtypes_but_float32_and_float64(
        self, model: MarketModel
    ) -> None:
        with pytest.raises(TypeError, match=r"torch\.Generator"):
            model.simulate_torch(np.random.default_rng(0), 5)  # type: ignore[arg-type]
        for dtype in (torch.int64, torch.float16, torch.bfloat16):  # (half: not the same model)
            with pytest.raises(TypeError, match=r"torch\.float32 or torch\.float64"):
                model.simulate_torch(gen(), 5, dtype=dtype)

    def test_numpy_integer_counts(self, model: MarketModel) -> None:
        sim = model.simulate_torch(gen(), np.int64(3), n_paths=np.int32(2))  # type: ignore[arg-type]
        assert tuple(sim.prices.shape[:2]) == (2, 4)

    def test_reads_nothing_back_and_copies_nothing_after_the_first_call(
        self, model: MarketModel, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On a GPU, each would make the host wait for the device. The parameters are copied
        to the device once (`Randomized` copies those it draws, once per call)."""
        model.simulate_torch(gen(), 20, n_paths=8)
        waits = Waits(monkeypatch)
        with waits:
            model.simulate_torch(gen(1), 20, n_paths=8)
        expected = {"upload": 1} if isinstance(model, Randomized) else {}
        assert dict(waits.count) == expected


def test_custom_models_simulate_on_the_host_only() -> None:
    model = ConstantModel(3)
    with pytest.raises(NotImplementedError, match="ConstantModel has no torch simulation"):
        model.simulate_torch(gen(), 5)


def test_parameters_are_copied_once_per_device_and_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    model = default_market()
    pickled = pickle.dumps(model)
    waits = Waits(monkeypatch)
    for dtype, uploads in ((torch.float32, 1), (torch.float32, 0), (torch.float64, 1)):
        with waits:
            model.simulate_torch(gen(), 10, n_paths=2, dtype=dtype)
        assert dict(waits.count) == ({"upload": uploads} if uploads else {})
    assert pickle.dumps(model) == pickled  # the copies are not part of the model
    alive = weakref.ref(model)
    del model
    gc.collect()
    assert alive() is None  # nor do they keep it alive
    model = default_market()
    clone = pickle.loads(pickled)
    torch.testing.assert_close(
        clone.simulate_torch(gen(4), 10, n_paths=2).prices,
        model.simulate_torch(gen(4), 10, n_paths=2).prices,
        rtol=0,
        atol=0,
        equal_nan=True,
    )


def test_simulation_of_tensors_checks_shapes_and_devices() -> None:
    prices = torch.ones(5, 2)
    Simulation(prices, torch.ones(5, 2, dtype=torch.bool), {"x": torch.zeros(5, 3)})
    with pytest.raises(ValueError, match="2-D"):
        Simulation(torch.ones(5))
    with pytest.raises(ValueError, match="boolean tensor"):
        Simulation(prices, torch.ones(5, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="boolean tensor"):
        Simulation(prices, torch.ones(5, 2))
    with pytest.raises(ValueError, match="boolean tensor"):
        Simulation(prices, np.ones((5, 2), dtype=bool))
    with pytest.raises(ValueError, match="leading dimensions"):
        Simulation(prices, state={"x": torch.zeros(4)})
    with pytest.raises(ValueError, match="must be a tensor"):
        Simulation(prices, state={"x": np.zeros(5)})


def test_import_does_not_import_torch() -> None:
    code = (
        "import sys, numpy as np, sekai, sekai.finance as f\n"
        "f.ModelScenarios(f.GBM(), 5).sample(2, seed=0)\n"
        "f.Simulation(np.ones((3, 2)))\n"
        "print('torch' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def path_statistics(sim: Simulation) -> dict[str, np.ndarray]:
    """Statistics of each path (a leading axis of paths): the mean and mean square of each
    asset's log returns, the mean cross product of the first two assets' (where listed), the
    log return over the whole path and its square (drift, diffusion or mean reversion), the
    fraction of the time each asset is listed, and the mean of each state series."""
    prices = host(sim.prices)
    r = log_returns(prices)
    ok = np.isfinite(r)
    r0 = np.where(ok, r, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        whole = np.log(prices[:, -1] / prices[:, 0])  # NaN where unlisted at either end
        out = {
            "mean": r0.sum(1) / ok.sum(1),
            "square": (r0**2).sum(1) / ok.sum(1),
            "whole": whole,
            "whole square": whole**2,
        }
        if prices.shape[-1] > 1:
            both = ok[..., 0] & ok[..., 1]
            out["cross"] = (r0[..., 0] * r0[..., 1]).sum(1) / both.sum(1)
    if sim.mask is not None:
        out["listed"] = host(sim.mask).mean(axis=1)
    for key, value in sim.state.items():
        out[f"state[{key}]"] = host(value).mean(axis=1)
    return out


def assert_same_distribution(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> None:
    """The means over paths of each statistic agree within 6 standard errors."""
    assert a.keys() == b.keys()
    for key in a:
        x, y = a[key], b[key]
        mx, my = np.nanmean(x, axis=0), np.nanmean(y, axis=0)
        se = np.sqrt(
            np.nanvar(x, axis=0) / np.sum(np.isfinite(x), axis=0)
            + np.nanvar(y, axis=0) / np.sum(np.isfinite(y), axis=0)
        )
        assert np.all(np.abs(mx - my) <= 6 * se + 1e-12), (key, mx, my, se)


@pytest.mark.parametrize("name", sorted(ALL_MODELS))
def test_the_same_distribution_as_simulate(name: str) -> None:
    model = ALL_MODELS[name]()
    n_paths = 150 if name == "randomized" else 600
    want = path_statistics(model.simulate(np.random.default_rng(1), 120, n_paths=n_paths))
    got = path_statistics(model.simulate_torch(gen(1), 120, n_paths=n_paths, dtype=torch.float64))
    assert_same_distribution(got, want)


# --------------------------------------------------------------------------------------------
# Individual models: their defining properties
# --------------------------------------------------------------------------------------------


def test_gbm_drift_volatility_and_correlation() -> None:
    model = GBM([0.3, 0.0, -0.2], [0.1, 0.2, 0.3], s0=[10.0, 20.0, 30.0], corr=CORR3, dt=1 / 12)
    sim = model.simulate_torch(gen(2), 12, n_paths=20_000, dtype=torch.float64)
    terminal = host(sim.prices)[:, -1] / model.s0  # E = exp(mu T), sd = exp(mu) sqrt(e^s2 - 1)
    sd = np.exp(model.mu) * np.sqrt(np.expm1(model.sigma**2)) / np.sqrt(20_000)
    np.testing.assert_array_less(np.abs(terminal.mean(axis=0) - np.exp(model.mu)), 5 * sd)
    r = log_returns(host(sim.prices)).reshape(-1, 3)
    np.testing.assert_allclose(r.std(axis=0) / np.sqrt(model.dt), model.sigma, rtol=0.02)
    np.testing.assert_allclose(np.corrcoef(r.T), CORR3, atol=0.02)


class TestHeston:
    def test_variance_averages_to_theta_and_prices_have_it(self) -> None:
        model = Heston(mu=0.0, kappa=3.0, theta=0.04, xi=0.3, rho=-0.7)
        sim = model.simulate_torch(gen(1), 1260, n_paths=400, dtype=torch.float64)
        variance = host(sim.state["variance"])
        np.testing.assert_array_equal(variance[:, 0, 0], 0.04)
        assert variance.mean() == pytest.approx(0.04, rel=0.05)
        realised = log_returns(host(sim.prices)).var() / model.dt
        assert realised == pytest.approx(0.04, rel=0.05)

    def test_leverage_effect(self) -> None:
        sim = Heston(rho=-0.7).simulate_torch(gen(2), 500, n_paths=40, dtype=torch.float64)
        dlog = log_returns(host(sim.prices)).ravel()
        dvar = np.diff(host(sim.state["variance"]), axis=-2).ravel()
        assert np.corrcoef(dlog, dvar)[0, 1] < -0.5

    def test_variance_truncated_at_zero_when_feller_fails(self) -> None:
        model = Heston(kappa=1.0, theta=0.04, xi=1.0, n_assets=3)
        sim = model.simulate_torch(gen(0), 1000, n_paths=20)
        variance = host(sim.state["variance"])
        assert np.all(variance >= 0) and np.any(variance == 0)

    def test_v0_and_correlated_price_shocks(self) -> None:
        sim = Heston(v0=[0.0, 0.09], corr=[[1.0, 0.6], [0.6, 1.0]]).simulate_torch(gen(), 3)
        np.testing.assert_allclose(host(sim.state["variance"][0]), [0.0, 0.09], rtol=1e-7)
        model = Heston(theta=0.04, xi=0.05, corr=[[1.0, 0.6], [0.6, 1.0]])
        r = log_returns(host(model.simulate_torch(gen(3), 200, n_paths=200).prices))
        assert np.corrcoef(r[..., 0].ravel(), r[..., 1].ravel())[0, 1] == pytest.approx(
            0.6, abs=0.03
        )


class TestMerton:
    def test_mean_terminal_price_and_jump_frequency(self) -> None:
        model = MertonJumpDiffusion(0.05, 0.2, 5.0, -0.05, 0.1, dt=1 / 12)
        sim = model.simulate_torch(gen(1), 12, n_paths=40_000, dtype=torch.float64)
        assert host(sim.prices)[:, -1, 0].mean() / 100 == pytest.approx(np.exp(0.05), rel=0.02)
        jumps = host(sim.state["jumps"])
        np.testing.assert_array_equal(jumps[:, 0], 0.0)
        np.testing.assert_array_equal(jumps, np.round(jumps))
        assert jumps[:, 1:].mean() == pytest.approx(5.0 / 12, rel=0.02)

    @pytest.mark.parametrize("rate", [0.004, 0.7, 3.0, 40.0, 900.0])
    def test_jump_counts_are_poisson(self, rate: float) -> None:
        """Counts drawn by inverting the Poisson CDF: the probabilities of small counts, and
        the mean and variance, at rates per step from rare jumps to hundreds."""
        model = MertonJumpDiffusion(jump_intensity=rate, dt=1.0)
        counts = host(model.simulate_torch(gen(7), 200, n_paths=500).state["jumps"])[:, 1:]
        n = counts.size  # (the sample variance has a variance of (rate + 2 rate^2) / n)
        assert counts.mean() == pytest.approx(rate, abs=6 * np.sqrt(rate / n))
        assert counts.var() == pytest.approx(rate, rel=6 * np.sqrt(1 / (rate * n) + 2 / n))
        for k in range(3):
            p = np.exp(-rate) * rate**k / np.prod(np.arange(1, k + 1))
            assert np.mean(counts == k) == pytest.approx(p, abs=6 * np.sqrt(p * (1 - p) / n))

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize("per_path", [False, True])
    def test_a_uniform_of_zero_inverts_to_the_half_step_above_it(
        self, dtype: str, per_path: bool, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """torch.rand gives exactly 0 once in 2**24 float32 draws. It stands for the half step
        above 0, whose quantile at 40 jumps a step is 15 or so, not the table's lowest count
        (0 jumps, where Poisson gives 4e-18)."""
        from sekai.finance.simulators import _poisson_counts, _poisson_table, _TorchContext

        def zeros(self: Any, *shape: int) -> torch.Tensor:
            return torch.zeros(shape, dtype=self.dtype)

        dt = getattr(torch, dtype)
        monkeypatch.setattr(_TorchContext, "uniform", zeros)
        low, cdf = (torch.as_tensor(a, dtype=dt)[None] for a in _poisson_table(np.array([40.0])))
        if per_path:
            low, cdf = low.repeat(3, 1), cdf.repeat(3, 1, 1)  # (contiguous, as real ones)
        counts = _poisson_counts(_TorchContext(gen(), dt), low, cdf, 3, 2)
        assert tuple(counts.shape) == (3, 2, 1) and torch.all(counts == counts[0, 0, 0])

        def poisson_cdf(k: float) -> float:
            return sum(
                math.exp(j * math.log(40.0) - 40.0 - math.lgamma(j + 1)) for j in range(int(k) + 1)
            )

        half_step, count = torch.finfo(dt).eps / 4, float(counts[0, 0, 0])
        assert poisson_cdf(count - 1) <= half_step < poisson_cdf(count)

    def test_fat_tails_without_jumps_gbm(self) -> None:
        def kurtosis(model: MarketModel) -> float:
            r = log_returns(host(model.simulate_torch(gen(2), 1000, n_paths=50).prices)).ravel()
            r = r - r.mean()
            return float(np.mean(r**4) / np.mean(r**2) ** 2 - 3.0)

        merton = MertonJumpDiffusion(0.0, 0.2, jump_intensity=10.0, jump_mean=-0.05)
        gbm_like = MertonJumpDiffusion(0.0, 0.2, jump_intensity=0.0)
        assert abs(kurtosis(gbm_like)) < 0.2 and kurtosis(merton) > 5


class TestRegimeSwitching:
    def test_regime_frequencies_volatility_and_correlation(self) -> None:
        model = two_regimes()
        sim = model.simulate_torch(gen(0), 2000, n_paths=100, dtype=torch.float64)
        regime = host(sim.state["regime"])
        assert set(np.unique(regime)) == {0.0, 1.0}
        assert np.mean(regime == 1) == pytest.approx(2 / 7, abs=0.02)
        returns = log_returns(host(sim.prices))
        for k, gbm in enumerate(model.regimes):
            in_regime = returns[regime[:, :-1] == k]
            np.testing.assert_allclose(
                in_regime.std(axis=0) / np.sqrt(model.dt), gbm.sigma, rtol=0.03
            )
            assert np.corrcoef(in_regime.T)[0, 1] == pytest.approx(gbm.corr[0, 1], abs=0.03)

    def test_initial_distribution_and_absorbing_regime(self) -> None:
        model = RegimeSwitching(
            [GBM(0.0, 0.1), GBM(0.0, 0.3)], [[0.9, 0.1], [0.0, 1.0]], initial=[1.0, 0.0]
        )
        regime = host(model.simulate_torch(gen(1), 500, n_paths=50).state["regime"])
        np.testing.assert_array_equal(regime[:, 0], 0.0)
        assert np.all(np.diff(regime, axis=1) >= 0) and np.all(regime[:, -1] == 1.0)
        start = RegimeSwitching(
            [GBM(), GBM(), GBM()], np.full((3, 3), 1 / 3), initial=[0.2, 0, 0.8]
        )
        first = host(start.simulate_torch(gen(2), 1, n_paths=20_000).state["regime"])[:, 0]
        assert np.mean(first == 1) == 0 and np.mean(first == 2) == pytest.approx(0.8, abs=0.02)

    def test_single_regime(self) -> None:
        sim = RegimeSwitching([GBM(0.0, 0.2, n_assets=2)], [[1.0]]).simulate_torch(gen(), 10)
        assert tuple(sim.prices.shape) == (11, 2)
        assert torch.all(sim.state["regime"] == 0)

    def test_the_chain_is_the_same_whatever_its_chunks_of_steps(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The moves are worked out a chunk of steps at a time, so that the memory does not
        grow with the number of steps; the chunks change no regime and no price."""
        from sekai.finance import simulators

        def factory(r: np.random.Generator) -> MarketModel:  # a transition matrix per path
            stay = r.uniform(0.5, 0.99)
            return RegimeSwitching([GBM(0.0, 0.1), GBM(0.0, 0.3)], [[stay, 1 - stay]] * 2)

        for model in (two_regimes(), Randomized(factory)):
            whole = model.simulate_torch(gen(3), 50, n_paths=7)  # one chunk
            for steps in (1, 4, 11, 49):
                monkeypatch.setattr(simulators, "_MOVES_ELEMENTS", steps * 7 * 2 * 2)
                same_simulation(model.simulate_torch(gen(3), 50, n_paths=7), whole)


class TestGARCH:
    def test_unconditional_variance_and_clustering(self) -> None:
        model = GARCH(omega=1e-6, alpha=0.08, beta=0.9, mu=0.05)
        sim = model.simulate_torch(gen(0), 1500, n_paths=200, dtype=torch.float64)
        eps = log_returns(host(sim.prices))[..., 0] - 0.05 * model.dt
        assert eps.var() == pytest.approx(float(model.unconditional_variance[0]), rel=0.1)
        assert np.mean([lag1_autocorr(path**2) for path in eps]) > 0.1

    def test_variance_recursion(self) -> None:
        model = GARCH(omega=2e-6, alpha=[0.05, 0.1], beta=[0.9, 0.8], mu=0.1)
        sim = model.simulate_torch(gen(1), 200, n_paths=3, dtype=torch.float64)
        h = host(sim.state["variance"])
        eps = log_returns(host(sim.prices)) - 0.1 * model.dt
        np.testing.assert_allclose(h[:, 0], np.broadcast_to(model.unconditional_variance, (3, 2)))
        np.testing.assert_allclose(
            h[:, 1:], 2e-6 + model.alpha * eps**2 + model.beta * h[:, :-1], rtol=1e-6
        )


def test_mean_reverting_terminal_distribution() -> None:
    theta, sigma, horizon = 5.0, 0.2, 252
    model = MeanReverting(theta=theta, mean=100.0, sigma=sigma, s0=50.0)
    x_t = np.log(host(model.simulate_torch(gen(0), horizon, n_paths=4000).prices)[:, -1, 0])
    decay = np.exp(-theta * horizon * model.dt)
    variance = sigma**2 * (1 - decay**2) / (2 * theta)
    expected = np.log(100.0) + (np.log(50.0) - np.log(100.0)) * decay
    assert x_t.mean() == pytest.approx(expected, abs=6 * np.sqrt(variance / 4000))
    assert x_t.var() == pytest.approx(variance, rel=6 * np.sqrt(2 / 4000))


def test_cointegrated_pair_spread_and_trend() -> None:
    model = CointegratedPair(mu=0.1, sigma=0.2, theta=50.0, spread_sigma=0.1, s0=(120.0, 80.0))
    sim = model.simulate_torch(gen(0), 252, n_paths=2000, dtype=torch.float64)
    prices, spread = host(sim.prices), host(sim.state["spread"])
    log_ratio = np.log(prices[..., 0] / prices[..., 1]) - np.log(120.0 / 80.0)
    np.testing.assert_allclose(log_ratio, spread, atol=1e-10)
    np.testing.assert_array_equal(spread[:, 0], 0.0)
    assert spread[:, -1].var() == pytest.approx(0.1**2 / (2 * 50.0), rel=6 * np.sqrt(2 / 2000))
    trend = 0.5 * np.log(prices[:, -1, 0] * prices[:, -1, 1] / (120.0 * 80.0))  # x_T
    sd = 0.2 / np.sqrt(2000)
    assert trend.mean() == pytest.approx(0.1 - 0.5 * 0.2**2, abs=6 * sd)
    assert trend.var() == pytest.approx(0.2**2, rel=6 * np.sqrt(2 / 2000))


def test_block_bootstrap_draws_circular_blocks_of_history() -> None:
    history = np.random.default_rng(3).normal(0.0, 0.01, size=(30, 3))
    size = 7
    model = BlockBootstrap(history, block_size=size)
    sim = model.simulate_torch(gen(0), 100, n_paths=20, dtype=torch.float64)
    wrapped, starts = False, set()
    for path in log_returns(host(sim.prices)):
        distance = np.abs(path[:, None, :] - history[None, :, :]).max(axis=-1)
        rows = distance.argmin(axis=1)
        assert np.all(distance.min(axis=1) < 1e-10)
        for j in range(len(rows) - 1):
            if (j + 1) % size:
                assert rows[j + 1] == (rows[j] + 1) % len(history)
                wrapped |= bool(rows[j] == len(history) - 1)
        starts.update(rows[::size].tolist())
    assert wrapped and len(starts) > 20  # blocks start anywhere


class TestListingProcess:
    def test_min_listed_respected_and_attained(self) -> None:
        model = ListingProcess(
            GBM(n_assets=6), initial_listed=0.1, listing_rate=0.5, delisting_rate=5.0, min_listed=3
        )
        sim = model.simulate_torch(gen(1), 1000, n_paths=10)
        mask = host(sim.mask)
        assert np.all(mask.sum(axis=-1) >= 3) and np.any(mask.sum(axis=-1) == 3)
        np.testing.assert_array_equal(np.isnan(host(sim.prices)), ~mask)

    def test_min_listed_fills_lowest_indices(self) -> None:
        model = ListingProcess(
            GBM(n_assets=5), initial_listed=0.0, listing_rate=0.0, delisting_rate=5.0, min_listed=2
        )
        mask = host(model.simulate_torch(gen(2), 300, n_paths=3).mask)
        assert np.all(mask[..., :2]) and not np.any(mask[..., 2:])

    @pytest.mark.parametrize("dt", [1 / 252, 1.0])
    def test_long_run_listed_fraction(self, dt: float) -> None:
        """p_list / (p_list + p_delist): 0.8 for daily steps, 0.741 for yearly ones."""
        p_list, p_delist = -np.expm1(-1.0 * dt), -np.expm1(-0.25 * dt)
        fraction = p_list / (p_list + p_delist)
        model = ListingProcess(
            GBM(n_assets=50, dt=dt),
            initial_listed=fraction,  # (started there, the chain stays there)
            listing_rate=1.0,
            delisting_rate=0.25,
            min_listed=0,
        )
        mask = host(model.simulate_torch(gen(6), 400, n_paths=200).mask)
        assert mask.mean() == pytest.approx(fraction, abs=0.025)  # 6 standard errors

    def test_combines_the_inner_mask_and_passes_its_state(self) -> None:
        inner = ListingProcess(Heston(n_assets=5), delisting_rate=3.0, min_listed=0)
        outer = ListingProcess(inner, delisting_rate=3.0, min_listed=0)
        sim = outer.simulate_torch(gen(5), 400, n_paths=3)
        assert set(sim.state) == {"variance"} and tuple(sim.state["variance"].shape) == (3, 401, 5)
        forced = ListingProcess(inner, min_listed=5).simulate_torch(gen(5), 400, n_paths=3)
        alone = inner.simulate_torch(gen(5), 400, n_paths=3)
        assert torch.equal(forced.mask, alone.mask)  # every own flag forced on: the inner mask
        assert not torch.equal(sim.mask, alone.mask)

    def test_always_listed(self) -> None:
        model = ListingProcess(GBM(n_assets=3), initial_listed=1.0, delisting_rate=0.0)
        sim = model.simulate_torch(gen(), 100)
        assert sim.mask is not None and bool(sim.mask.all())

    def test_wraps_a_custom_model_with_a_torch_simulation(self) -> None:
        class Flat(ConstantModel):
            def simulate_torch(self, generator: Any, n_steps: int, **kw: Any) -> Simulation:
                n = kw.get("n_paths")
                lead = () if n is None else (n,)
                return Simulation(prices=torch.ones((*lead, n_steps + 1, self._n)))

        sim = ListingProcess(Flat(4), min_listed=4).simulate_torch(gen(), 10, n_paths=2)
        assert sim.mask is not None and bool(sim.mask.all()) and bool((sim.prices == 1).all())
        with pytest.raises(NotImplementedError, match="ConstantModel"):
            ListingProcess(ConstantModel(4)).simulate_torch(gen(), 10)


class TestRandomized:
    @staticmethod
    def factory(r: np.random.Generator) -> MarketModel:
        s0 = r.uniform(50.0, 150.0)
        return GBM(sigma=s0 / 500, s0=s0, n_assets=2)  # the path's s0 tells its sigma

    def test_the_default_dtype_changes_nothing(self) -> None:
        """Seeding its host generator moves the torch generator on by a float32 draw, which
        every device has (MPS has no float64), whatever torch's default dtype."""
        model = Randomized(self.factory)
        want = model.simulate_torch(gen(4), 20, n_paths=6, dtype=torch.float32)
        default = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64 if default == torch.float32 else torch.float32)
        try:
            got = model.simulate_torch(gen(4), 20, n_paths=6, dtype=torch.float32)
        finally:
            torch.set_default_dtype(default)
        same_simulation(got, want)

    def test_each_path_has_its_own_parameters(self) -> None:
        sim = Randomized(self.factory).simulate_torch(gen(0), 500, n_paths=64)
        prices = host(sim.prices)
        s0 = prices[:, 0, 0]
        assert len(np.unique(s0)) == 64
        vol = log_returns(prices).std(axis=(1, 2)) / np.sqrt(1 / 252)
        np.testing.assert_allclose(vol, s0 / 500, rtol=0.15)

    def test_paths_of_several_kinds_stay_in_order(self) -> None:
        def factory(r: np.random.Generator) -> MarketModel:
            if r.random() < 0.5:  # s0 below 100: a GBM, else a fast reversion to 200
                gbm = GBM(sigma=0.05, s0=r.uniform(50, 99), n_assets=2)
                return ListingProcess(gbm, initial_listed=1.0, delisting_rate=0.0)
            if r.random() < 0.5:
                return MeanReverting(theta=50.0, mean=200.0, s0=r.uniform(101, 150), n_assets=2)
            corr = [[1.0, 0.5], [0.5, 1.0]]
            return MeanReverting(50.0, 200.0, s0=r.uniform(101, 150), corr=corr)

        model = Randomized(factory)
        g = gen(1)
        sim = model.simulate_torch(g, 100, n_paths=40)
        prices = host(sim.prices)
        gbm = prices[:, 0, 0] < 100
        assert 5 < gbm.sum() < 35
        np.testing.assert_allclose(prices[~gbm, -1], 200.0, rtol=0.2)
        assert np.all(prices[gbm, -1] < 150)
        assert sim.mask is not None and bool(sim.mask[torch.as_tensor(~gbm)].all())
        again = model.simulate_torch(gen(1), 100, n_paths=40)
        torch.testing.assert_close(again.prices, sim.prices, rtol=0, atol=0, equal_nan=True)
        assert not torch.equal(model.simulate_torch(g, 100, n_paths=40).prices, sim.prices)

    def test_stacks_state_and_masks(self) -> None:
        model = Randomized(
            lambda r: ListingProcess(Heston(theta=r.uniform(0.01, 0.1), n_assets=3), min_listed=0)
        )
        sim = model.simulate_torch(gen(1), 50, n_paths=4)
        assert sim.mask is not None and tuple(sim.mask.shape) == (4, 51, 3)
        theta = host(sim.state["variance"])[:, 0, 0]  # v0 = theta, drawn per path
        assert len(np.unique(theta)) == 4 and np.all((theta > 0.01) & (theta < 0.1))
        np.testing.assert_array_equal(np.isnan(host(sim.prices)), ~host(sim.mask))

    def test_models_without_their_own_torch_simulation(self) -> None:
        with pytest.raises(NotImplementedError, match="ConstantModel"):
            Randomized(lambda r: ConstantModel(2)).simulate_torch(gen(), 5, n_paths=2)

        class Doubled(GBM):  # a custom simulate: no torch simulation of its own
            def simulate(self, rng: Any, n_steps: int, *, n_paths: int | None = None) -> Any:
                return super().simulate(rng, n_steps, n_paths=n_paths)

        with pytest.raises(NotImplementedError, match="overrides simulate"):
            Doubled().simulate_torch(gen(), 5)
        with pytest.raises(NotImplementedError, match="overrides simulate"):
            Randomized(lambda r: Doubled()).simulate_torch(gen(), 5, n_paths=2)

    def test_mismatched_models_raise(self) -> None:
        models = iter([GBM(), GBM(n_assets=2)])
        model = Randomized(lambda r: next(models, GBM(n_assets=2)))
        with pytest.raises(ValueError, match="reference"):
            model.simulate_torch(gen(), 5, n_paths=3)


# --------------------------------------------------------------------------------------------
# MarketPanel and ModelScenarios
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("single", [False, True])
def test_a_panel_of_a_tensor_simulation_is_computed_on_its_device(single: bool) -> None:
    sim = default_market().simulate_torch(
        gen(3), 30, n_paths=None if single else 4, dtype=torch.float64
    )
    as_numpy = Simulation(
        prices=host(sim.prices),
        mask=host(sim.mask),
        state={k: host(v) for k, v in sim.state.items()},
    )
    want = MarketPanel.from_simulation(as_numpy)  # the NumPy panel of the same paths
    got = MarketPanel.from_simulation(sim, backend="torch")
    for a, b in (
        (got.returns, want.returns),
        (got.valid, want.valid),
        (got.fields["close"], want.fields["close"]),
        (got.exogenous["regime"], want.exogenous["regime"]),
        (got.state["regime"], want.state["regime"]),
    ):
        assert isinstance(a, torch.Tensor) and a.device.type == "cpu"
        np.testing.assert_array_equal(host(a), b)  # float64: the same operations, bit for bit
    assert got.returns.dtype == torch.float64 and got.valid.dtype == torch.bool
    narrow = MarketPanel.from_simulation(sim, backend="torch", dtype=torch.float32)
    assert torch.equal(torch.isnan(narrow.returns), torch.isnan(got.returns))
    assert torch.equal(torch.nan_to_num(narrow.returns), torch.nan_to_num(got.returns.float()))
    on_host = MarketPanel.from_simulation(sim)  # the NumPy backend copies it to the host
    np.testing.assert_array_equal(on_host.returns, want.returns)
    np.testing.assert_array_equal(on_host.valid, want.valid)


class TestModelScenarios:
    def test_on_a_device_the_panels_are_tensors_made_there(self) -> None:
        model = ListingProcess(GBM(mu=0.05, sigma=0.2, n_assets=5))
        scenarios = ModelScenarios(model, 30, device="cpu", dtype=torch.float64)
        a, b = scenarios.sample(3, seed=7), scenarios.sample(3, seed=7)
        assert isinstance(a.returns, torch.Tensor) and a.returns.dtype == torch.float64
        assert a.device == torch.device("cpu") and tuple(a.returns.shape) == (3, 31, 5)
        torch.testing.assert_close(a.returns, b.returns, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(  # NumPy integers seed it too, as they seed NumPy
            scenarios.sample(3, seed=np.int64(7)).returns, a.returns, rtol=0, atol=0, equal_nan=True
        )
        c = scenarios.sample(3, seed=8)
        assert not torch.equal(torch.nan_to_num(c.returns), torch.nan_to_num(a.returns))
        seed = int(np.random.SeedSequence(7).generate_state(1, np.uint64)[0])  # (the seed's)
        sim = model.simulate_torch(gen(seed), 30, n_paths=3, dtype=torch.float64)
        torch.testing.assert_close(a.fields["close"], sim.prices, rtol=0, atol=0, equal_nan=True)
        assert ModelScenarios(model, 30, device="cpu").sample(2, seed=0).returns.dtype == (
            torch.get_default_dtype()
        )
        assert repr(scenarios).endswith("n_steps=30, device='cpu', dtype=torch.float64)")
        assert repr(ModelScenarios(model, 30, device="cpu")).endswith(", device='cpu')")

    def test_without_a_device_nothing_changes(self) -> None:
        model = GBM(sigma=[0.2, 0.3])
        scenarios = ModelScenarios(model, 20)
        assert vars(scenarios).keys() == {"model", "n_steps"}  # pickles as it did
        assert repr(scenarios) == f"ModelScenarios({model!r}, n_steps=20)"
        panel = scenarios.sample(4, seed=3)
        want = MarketPanel.from_simulation(model.simulate(np.random.default_rng(3), 20, n_paths=4))
        assert isinstance(panel.returns, np.ndarray)
        np.testing.assert_array_equal(panel.returns, want.returns)

    def test_on_a_device_it_takes_the_seeds_numpy_takes(self) -> None:
        model = GBM(n_assets=2)
        on_device, on_host = ModelScenarios(model, 10, device="cpu"), ModelScenarios(model, 10)
        entropy = np.random.SeedSequence().entropy  # 128 bits, as sekai draws when unseeded
        assert isinstance(entropy, int)
        for seed in (0, 2**63, 2**64, 2**100, entropy):
            on_host.sample(2, seed=seed)
            same(on_device.sample(2, seed=seed).returns, on_device.sample(2, seed=seed).returns)
        returns = [on_device.sample(2, seed=s).returns.nan_to_num() for s in (0, 2**64, 2**64 - 1)]
        assert not torch.equal(returns[0], returns[1]) and not torch.equal(returns[1], returns[2])
        for scenarios in (on_device, on_host):
            with pytest.raises(ValueError, match="non-negative"):
                scenarios.sample(2, seed=-1)
            with pytest.raises(TypeError):
                scenarios.sample(2, seed=1.5)  # type: ignore[arg-type]

    def test_validation(self) -> None:
        with pytest.raises(ValueError, match="give device= too"):
            ModelScenarios(GBM(), 5, dtype=torch.float32)
        for dtype in (torch.int32, torch.float16, torch.bfloat16):
            with pytest.raises(TypeError, match=r"torch\.float32 or torch\.float64"):
                ModelScenarios(GBM(), 5, device="cpu", dtype=dtype)
        with pytest.raises(NotImplementedError):
            ModelScenarios(ConstantModel(), 5, device="cpu").sample(2, seed=0)


# --------------------------------------------------------------------------------------------
# BatchMarketEnv(simulate_on_device=True)
# --------------------------------------------------------------------------------------------


def make_env(n: int = 5, **kwargs: Any) -> BatchMarketEnv:
    kw: dict[str, Any] = dict(
        model=default_market(),
        simulate_on_device=True,
        backend="torch",
        lookback=5,
        episode_length=12,
        max_drawdown=0.03,  # (lanes stop at different times: partial resets)
    )
    kw.update(kwargs)
    return BatchMarketEnv(n, **kw)


def actions(env: BatchMarketEnv, k: int, seed: int = 0) -> list[Any]:
    g = torch.Generator().manual_seed(seed)
    shape = (env.num_envs, *env.action_space.shape)  # type: ignore[misc]
    return [torch.rand(shape, generator=g, dtype=env._like.dtype) for _ in range(k)]


def same(a: Any, b: Any) -> None:
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            same(a[key], b[key])
    elif a is None or b is None:
        assert a is b
    elif isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
    else:
        np.testing.assert_array_equal(a, b)


def same_simulation(a: Simulation, b: Simulation) -> None:
    same(
        {"prices": a.prices, "mask": a.mask, "state": a.state},
        {"prices": b.prices, "mask": b.mask, "state": b.state},
    )


def rollout(
    env: BatchMarketEnv, steps: list[Any], reset: int | None = None
) -> list[dict[str, Any]]:
    out = [] if reset is None else [vars(env.reset(seed=reset))]
    for i, a in enumerate(steps):
        if not env.autoreset and i in (7, 15, 23):
            done = np.flatnonzero(env.backend.to_numpy(env.frozen))
            if len(done):
                out.append(vars(env.reset(indices=done[::2])))  # some of them
        out.append(vars(env.step(a)))
    return out


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("autoreset", [True, False])
def test_an_env_simulating_on_the_device_steps_and_resets(autoreset: bool, dtype: str) -> None:
    float_dtype = getattr(torch, dtype)
    env = make_env(autoreset=autoreset, dtype=float_dtype, privileged_state=True)
    assert env.scenarios is not None and env.scenarios.device == torch.device("cpu")
    steps = actions(env, 30)
    out = rollout(env, steps, reset=1)
    resets = 0
    for result in out:
        obs = result["observations"]
        for value in (obs["assets"]["values"], obs["portfolio"], obs["account"]):
            assert value.device.type == "cpu" and value.dtype == float_dtype
        assert bool(torch.isfinite(obs["assets"]["values"]).all())
        lanes = result.get("reset_indices", result.get("indices"))
        resets += lanes is not None and 0 < len(lanes) < env.num_envs
    assert resets >= 2  # some lanes at a time: episodes of 12 steps, and drawdown stops
    assert env.panel.returns.dtype == float_dtype and env.panel.returns.device.type == "cpu"
    assert set(env.latent_state) == {"regime"}
    again = rollout(
        make_env(autoreset=autoreset, dtype=float_dtype, privileged_state=True), steps, 1
    )
    for x, y in zip(out, again, strict=True):  # a seed reproduces the rollout
        same(x, y)
    other = make_env(autoreset=autoreset, dtype=float_dtype).reset(seed=2).observations
    assert not torch.equal(other["assets"]["values"], out[0]["observations"]["assets"]["values"])


def test_a_reset_on_the_device_copies_nothing_to_it(monkeypatch: pytest.MonkeyPatch) -> None:
    waits = Waits(monkeypatch)
    for autoreset in (True, False):
        env, on_host = (
            make_env(autoreset=autoreset),
            make_env(autoreset=autoreset, simulate_on_device=False),
        )
        env._defer = on_host._defer = True  # run the device path on the CPU
        with waits:
            env.reset(seed=0)
        assert dict(waits.count) == {}
        with waits:  # the paths simulated on the host are copied over
            on_host.reset(seed=0)
        assert waits.count["upload"] > 0
        if not autoreset:
            for a in actions(env, 13):
                env.step(a)
            with waits:  # the lanes' indices, given on the host
                env.reset(indices=np.array([1, 3]))
            assert dict(waits.count) == {"upload": 1}
            continue
        resets = 0
        for a in actions(env, 30):
            with waits:
                result = env.step(a)
            resets += result.reset_indices is not None
            # the step's read, and the indices of the lanes autoreset (as on the host)
            assert dict(waits.count) == (
                {"read": 1, "upload": 1} if result.reset_indices is not None else {"read": 1}
            )
        assert resets > 0


def test_flag_validation() -> None:
    data = MarketData(np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, (60, 3)), 0)))
    with pytest.raises(ValueError, match="does not apply to data="):
        BatchMarketEnv(
            2, data, simulate_on_device=True, backend="torch", lookback=5, episode_length=10
        )
    with pytest.raises(ValueError, match="does not apply to scenarios="):
        BatchMarketEnv(
            2,
            scenarios=ModelScenarios(GBM(), 40),
            simulate_on_device=True,
            backend="torch",
            lookback=5,
            episode_length=10,
        )
    with pytest.raises(ValueError, match="needs backend='torch'"):
        BatchMarketEnv(2, model=GBM(), simulate_on_device=True)
    with pytest.raises(NotImplementedError, match="ConstantModel"):
        BatchMarketEnv(2, model=ConstantModel(), simulate_on_device=True, backend="torch")
    for dtype in (torch.float16, torch.bfloat16):  # (simulated on the host, they are allowed)
        with pytest.raises(TypeError, match=r"torch\.float32 or torch\.float64"):
            BatchMarketEnv(2, simulate_on_device=True, backend="torch", dtype=dtype)
    default = BatchMarketEnv(
        2, simulate_on_device=True, backend="torch", lookback=5, episode_length=10
    )
    assert default.scenarios is not None and default.scenarios.device == torch.device("cpu")
    assert BatchMarketEnv(2, backend="torch").scenarios.device is None  # type: ignore[union-attr]


def test_pickles_copies_and_snapshots_continue_identically() -> None:
    env = make_env(autoreset=True)
    env.reset(seed=3)
    steps = actions(env, 40, seed=1)
    for a in steps[:5]:
        env.step(a)
    clone, twin, snapshot = pickle.loads(pickle.dumps(env)), copy.deepcopy(env), env.snapshot()
    expected = rollout(env, steps[5:])  # autoresets included, which simulate on the device
    for other in (clone, twin):
        for x, y in zip(rollout(other, steps[5:]), expected, strict=True):
            same(x, y)
    env.restore(snapshot)
    for x, y in zip(rollout(env, steps[5:]), expected, strict=True):
        same(x, y)


def test_make_and_make_vec_pass_the_flag_on() -> None:
    kw: dict[str, Any] = dict(
        backend="torch", simulate_on_device=True, lookback=5, episode_length=10
    )
    env = sekai.make("Market-v1", **kw)
    vec = sekai.make_vec("Market-v1", num_envs=3, **kw)
    for batch in (env.unwrapped.batch_env, vec.batch_env):  # type: ignore[attr-defined]
        assert batch.scenarios.device == torch.device("cpu")
    env.reset(seed=0)
    env.step(env.action_space.sample())
    assert vec.reset(seed=0).observations["portfolio"].shape == (3, 11)
    vec.step(vec.batch_action_space.sample())
    vec.close()
    with pytest.raises(ValueError, match="needs backend='torch'"):
        sekai.make_vec("Market-v1", num_envs=2, simulate_on_device=True)


# --------------------------------------------------------------------------------------------
# On a GPU
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("device", GPUS)
@pytest.mark.parametrize("name", sorted(ALL_MODELS))
def test_simulates_on_a_gpu(device: str, name: str) -> None:
    model = ALL_MODELS[name]()
    sim = model.simulate_torch(gen(0, device), 60, n_paths=16, dtype=torch.float32)
    tensors = [sim.prices, *sim.state.values()] + ([] if sim.mask is None else [sim.mask])
    assert all(t.device.type == device for t in tensors)
    prices = host(sim.prices)
    valid = np.ones(prices.shape, bool) if sim.mask is None else host(sim.mask)
    assert np.all(prices[valid] > 0) and np.all(np.isnan(prices[~valid]))
    if device == "cuda" and not isinstance(model, Randomized):  # (it copies its parameters)
        generator = gen(1, device)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")  # waiting for the device raises
        try:
            model.simulate_torch(generator, 60, n_paths=16, dtype=torch.float32)
        finally:
            torch.cuda.set_sync_debug_mode("default")


@pytest.mark.parametrize("device", GPUS)
def test_an_env_simulating_on_a_gpu(device: str) -> None:
    env = make_env(device=device, dtype=torch.float32)
    result = env.reset(seed=0)
    assert result.observations["portfolio"].device.type == device
    assert env.panel.returns.device.type == device
    for a in actions(env, 15):
        result = env.step(a.to(device))
    assert result.rewards.device.type == device and bool(torch.isfinite(result.rewards).all())
    scenarios = ModelScenarios(default_market(), 20, device=device)
    assert scenarios.sample(4, seed=0).returns.device.type == device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_a_cuda_generator_without_an_index_simulates_on_the_current_device() -> None:
    """torch.Generator(device="cuda") may have no index: its simulations, and the parameters
    kept for them, are on the device current when it simulates."""
    from sekai.finance.simulators import _TorchContext

    ctx = _TorchContext(torch.Generator(device="cuda"), torch.float32)
    assert ctx.device == torch.device("cuda", torch.cuda.current_device())
    model = GBM(n_assets=2)
    for index in range(min(torch.cuda.device_count(), 2)):
        with torch.cuda.device(index):
            sim = model.simulate_torch(torch.Generator(device="cuda"), 5, n_paths=2)
            assert sim.prices.device == torch.device("cuda", index)
