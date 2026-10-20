"""Regression tests for issues found in review of the dynamic-space, vector and finance code."""

from __future__ import annotations

import os
import signal
import sys
from typing import Any

import numpy as np
import pytest

import sekai
from sekai.envs.finance import MarketEnv, PortfolioEnv, TradingEnv
from sekai.finance import (
    DifferentialSharpe,
    DrawdownPenalty,
    FixedCost,
    ListingProcess,
    MarketData,
    ProportionalCost,
    SequentialStart,
    SquareRootImpact,
    StepOutcome,
    Time,
)
from sekai.finance.simulators import GBM
from sekai.spaces import MaskedBox, Sequence, batch_space
from sekai.utils.running_stats import ExponentialMovingStats
from tests.conftest import CountingEnv
from tests.test_dynamic import UniverseEnv

# ---------------------------------------------------------------------------
# Core framework
# ---------------------------------------------------------------------------


def test_make_vec_records_time_limits_on_the_spec() -> None:
    vec = sekai.make_vec("CartPole-v1", 2, max_episode_steps=100)
    assert vec.spec.max_episode_steps == 100 and "max_episode_steps" not in vec.spec.kwargs
    rebuilt = sekai.make_vec(vec.spec, 2)  # the spec rebuilds the same vector env
    assert rebuilt.spec.max_episode_steps == 100
    unlimited = sekai.make_vec("CartPole-v1", 2, max_episode_steps=-1)
    assert unlimited.spec.max_episode_steps is None


def test_ewm_stats_are_not_dominated_by_the_first_sample() -> None:
    stats = ExponentialMovingStats((), halflife=1000)
    for x in np.random.default_rng(0).normal(5.0, 1.0, 300):
        stats.update(np.array([x]))
    assert stats.mean == pytest.approx(5.0, abs=0.2) and stats.var == pytest.approx(1.0, abs=0.25)
    restored = ExponentialMovingStats((), halflife=1000)
    restored.load_state_dict(stats.state_dict())
    assert restored.weight == stats.weight


def test_ewm_halflife_counts_steps_not_rows() -> None:
    class Wide(sekai.Env[dict[str, Any], int]):
        """100 occupied slots; their level jumps from 0 to 10 at step 50."""

        def __init__(self) -> None:
            self.observation_space = MaskedBox(-np.inf, np.inf, shape=(1,), max_size=100)
            self.action_space = sekai.Discrete(1)
            self.t = 0

        def _obs(self) -> dict[str, Any]:
            space = self.observation_space
            assert isinstance(space, MaskedBox)
            return space.pack(np.full((100, 1), 0.0 if self.t < 50 else 10.0))

        def reset(self, *, seed: Any = None, options: Any = None) -> sekai.ResetResult:
            self.t = 0
            return sekai.ResetResult(self._obs(), {})

        def step(self, action: Any) -> sekai.StepResult:
            self.t += 1
            return sekai.StepResult(self._obs(), 0.0, False, False, {})

    env = sekai.NormalizeObservation(Wide(), stats="ewm", halflife=100)
    env.reset()
    for _ in range(60):
        env.step(0)
    # Ten steps after the jump, with a 100-step halflife, the mean has moved only part
    # way (counting each of the 100 rows as a sample would put it near 10 already).
    assert 0.5 < float(env.obs_rms.mean[0]) < 3.0


def test_masked_box_observations_keep_their_type_in_wrappers() -> None:
    class SlotsOnly(UniverseEnv):
        """UniverseEnv whose observation is just the MaskedBox of assets."""

        def __init__(self) -> None:
            super().__init__()
            self._full = self.observation_space
            self.observation_space = self._full["assets"]  # type: ignore[assignment]

        def _obs(self) -> Any:
            space = self.observation_space
            self.observation_space = self._full  # UniverseEnv._obs reads the full space
            try:
                return super()._obs()["assets"]
            finally:
                self.observation_space = space

    env = sekai.ActionMaskObservation(SlotsOnly())
    assert isinstance(env.observation_space["observation"], MaskedBox)
    assert env.observation_space.contains(env.reset(seed=0).observation)
    with pytest.raises(TypeError):
        sekai.TimeAwareObservation(SlotsOnly(), max_episode_steps=10)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals and start methods")
def test_proc_workers_survive_sigint_and_fork_does_not_unlink_early() -> None:
    from multiprocessing import shared_memory

    vec = sekai.ProcVecEnv([CountingEnv] * 2, context="fork")
    try:
        vec.reset(seed=0)
        os.kill(vec._processes[0].pid, signal.SIGINT)  # Ctrl-C reaches the worker too
        assert vec.step([0, 0]).rewards.tolist() == [0.0, 0.0]
        assert vec._shared is not None
        name = vec._shared.block.name
        shared_memory.SharedMemory(name=name).close()  # still exists while in use
    finally:
        vec.close()
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name)  # and is gone after close()


def test_batched_sequences_are_independent() -> None:
    space = batch_space(Sequence(sekai.Box(-1, 1, (2,)), max_length=5, seed=0), 3)
    samples = space.sample()
    assert len({str(s) for s in samples}) == 3


def test_check_env_compares_nan_observations_correctly() -> None:
    class WithNaN(CountingEnv):
        def __init__(self) -> None:
            super().__init__()
            self.observation_space = sekai.Box(-np.inf, np.inf, shape=(2,))

        def _obs(self) -> np.ndarray:
            return np.array([np.nan, self.t], dtype=np.float32)

    issues = sekai.check_env(WithNaN(), check_determinism=True)
    assert not any("mutated" in i or "different" in i or "differed" in i for i in issues)


def test_stacked_sequence_rejects_the_tuple_form() -> None:
    space = Sequence(sekai.Box(-1, 1, (2,)), stack=True)
    rows = (np.zeros(2, np.float32), np.zeros(2, np.float32))
    assert not space.contains(rows) and space.contains(np.stack(rows))


# ---------------------------------------------------------------------------
# Finance
# ---------------------------------------------------------------------------


def history(n: int = 300, k: int = 3) -> MarketData:
    rng = np.random.default_rng(0)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, k)), axis=0))
    return MarketData(prices, fields={"volume": np.full((n, k), 1e5)})


def test_stateful_rewards_are_not_shared_between_sub_envs() -> None:
    vec = sekai.make_vec("Trading-v0", 2, vectorization_mode="sync", reward=DrawdownPenalty())
    a, b = vec.envs  # type: ignore[attr-defined]
    assert a.unwrapped.reward_fn is not b.unwrapped.reward_fn


def test_a_cost_of_the_whole_account_wipes_it_out() -> None:
    env = TradingEnv(
        positions=(0.0, 3.0), transaction_cost=FixedCost(2.0), capital=1.0, episode_length=5
    )
    env.reset(seed=0)
    r = env.step(1)
    assert r.terminated and r.info["equity"] == 0.0


def test_integer_start_is_a_row_and_time_a_timestamp_on_numeric_axes() -> None:
    _, test = history(1000, 1).split(0.5)  # timestamps 500..999
    env = TradingEnv(data=test, episode_length=20, lookback=10)
    r = env.reset(seed=0, options={"start": Time(800)})
    assert r.info["timestamp"] == 800
    replay = env.reset(seed=0, options={"start": Time(r.info["timestamp"])})
    assert replay.info["timestamp"] == 800
    assert env.reset(seed=0, options={"start": 300}).info["timestamp"] == 800  # row 300


def test_sequential_epoch_flips_after_the_last_window() -> None:
    env = PortfolioEnv(data=history(), episode_length=50, lookback=10, sampler=SequentialStart())
    sampler = env._history.sampler  # type: ignore[union-attr]
    starts = []
    while sampler.epoch == 0:  # type: ignore[attr-defined]
        starts.append(env.reset(seed=0).info["timestamp"])
    assert len(starts) == len(set(starts)) == 5  # 300 rows: windows at 10, 60, ..., 210


def test_impact_is_never_free_for_illiquid_assets() -> None:
    cost = SquareRootImpact(0.5)(
        np.ones(3), equity=1e6, volatility=np.full(3, 0.02), volume=np.array([1e6, np.nan, 0.0])
    )
    assert cost >= 2.0  # the illiquid and missing-volume trades each cost their full value


def test_per_asset_cost_parameters_follow_assets_not_slots() -> None:
    data = history(200, 4)
    rates = ProportionalCost([0.0, 0.0, 0.0, 0.01])
    for slots in (3, 6):
        env = MarketEnv(
            data, max_assets=slots, lookback=5, episode_length=10, transaction_cost=rates
        )
        env.reset(seed=0)
        w = np.zeros(slots + 1, np.float32)
        w[env.slot_assets.index("asset_0")] = 1.0
        assert env.step(w).info["cost"] == 0.0  # asset_0 is free wherever its slot is


def test_echoing_observed_weights_charges_no_ticket_fees() -> None:
    env = PortfolioEnv(
        data=history(), episode_length=20, lookback=5, transaction_cost=FixedCost(5.0)
    )
    env.reset(seed=0)
    obs = env.step(np.ones(4, np.float32)).observation
    for _ in range(5):
        r = env.step(obs["weights"])  # hold: float32 rounding must not count as trades
        obs = r.observation
        assert r.info["cost"] == 0.0


def test_differential_sharpe_is_identical_for_scalars_and_batches() -> None:
    rng = np.random.default_rng(1)
    growth = 1 + rng.normal(0.001, 0.01, (300, 4))
    single = [DifferentialSharpe(0.05) for _ in range(4)]
    batched = DifferentialSharpe(0.05)
    for g in growth:
        batch = batched(StepOutcome.from_growth(g, equity=1.0, turnover=0.0, cost=0.0))
        one = [
            float(s(StepOutcome.from_growth(x, equity=1.0, turnover=0.0, cost=0.0)))
            for s, x in zip(single, g)
        ]
        assert batch.tolist() == one


def test_native_portfolio_is_bit_identical_and_infos_match() -> None:
    data = history(400, 5)
    common = {
        "data": data,
        "episode_length": 30,
        "lookback": 8,
        "sampler": 40,
        "allow_short": True,
    }
    native = sekai.make_vec("Portfolio-v0", 33, **common)
    sync = sekai.make_vec("Portfolio-v0", 33, vectorization_mode="sync", **common)
    x, y = native.reset(seed=0), sync.reset(seed=0)
    assert x.infos[0].keys() == y.infos[0].keys()
    actions = native.batch_action_space
    actions.seed(0)
    for _ in range(40):
        a = actions.sample()
        xs, ys = native.step(a), sync.step(a)
        assert xs.rewards.tolist() == ys.rewards.tolist()
    assert xs.infos[0].keys() == ys.infos[0].keys()


def test_native_timestamps_stay_datetimes() -> None:
    data = MarketData(history().prices, timestamps=np.datetime64("2024-01-01") + np.arange(300))
    vec = sekai.make_vec(
        "Trading-v0", 2, data=data.select(["asset_0"]), episode_length=10, lookback=5
    )
    info = vec.reset(seed=0).infos[0]
    assert isinstance(info["timestamp"], np.datetime64)
    assert isinstance(vec.step(np.array([1, 1])).infos[0]["timestamp"], np.datetime64)


def test_native_envs_validate_like_single_envs() -> None:
    listed = ListingProcess(GBM(n_assets=3), initial_listed=0.3, min_listed=1)
    with pytest.raises(ValueError, match="listed"):
        sekai.make_vec("Portfolio-v0", 2, model=listed).reset(seed=0)
    vec = sekai.make_vec("Trading-v0", 2)
    vec.reset(seed=0)
    with pytest.raises(sekai.InvalidAction):
        vec.step(np.array([1.7, 1.0]))
