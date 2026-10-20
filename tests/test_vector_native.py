"""Native (batched) vector envs, vector entry points, and ProcVecEnv's shared memory / workers."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import pytest

import sekai
from sekai.finance.costs import FixedCost
from sekai.finance.data import MarketData
from sekai.finance.sampling import EpisodeSampler, FixedStart, RandomStart, SequentialStart
from sekai.vector.native import NativeVecEnv, shared, split_infos, take_index
from tests.conftest import CountingEnv, DictObsEnv
from tests.test_dynamic import TradesEnv, UniverseEnv


class BatchCounting(NativeVecEnv[np.ndarray, Any]):
    """CountingEnv, natively vectorised: obs [t, noise], reward = action, ends at horizon."""

    def __init__(self, num_envs: int, horizon: int = 3, **kwargs: Any) -> None:
        super().__init__(
            num_envs,
            sekai.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
            sekai.Discrete(3),
            **kwargs,
        )
        self.horizon = np.full(num_envs, horizon)
        self.t = np.zeros(num_envs, dtype=np.int64)
        self.noise = np.zeros(num_envs)

    def _reset_envs(self, indices: np.ndarray, options: Sequence[dict[str, Any] | None]) -> None:
        self.t[indices] = 0
        self.noise[indices] = self.rng.standard_normal(len(indices))
        for i, opts in zip(indices, options):
            if opts and "horizon" in opts:
                self.horizon[i] = opts["horizon"]

    def _observe(self) -> np.ndarray:
        return np.stack([self.t, self.noise], axis=1).astype(np.float32)

    def _step(self, actions: Any) -> tuple[Any, Any, Any, dict[str, np.ndarray]]:
        a = np.asarray(actions)
        self.t += 1
        self.noise = self.rng.standard_normal(self.num_envs)
        return a.astype(float), self.t >= self.horizon, np.zeros(self.num_envs, bool), {"t": self.t}

    def _reset_infos(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        return {"reset": np.ones(len(indices), dtype=bool)}

    def double(self) -> np.ndarray:
        return self.t * 2


@pytest.fixture
def registered() -> Iterator[str]:
    env_id = "test/NativeCounting-v0"
    sekai.register(
        env_id,
        entry_point=CountingEnv,
        vector_entry_point=BatchCounting,
        kwargs={"horizon": 3},
    )
    yield env_id
    sekai.deregister(env_id)


class TestNativeVecEnv:
    def test_reset_step_and_infos(self) -> None:
        vec = BatchCounting(4)
        r = vec.reset(seed=0)
        assert r.observations.shape == (4, 2) and r.infos[0] == {"reset": True}
        s = vec.step(np.array([0, 1, 2, 1]))
        assert s.rewards.tolist() == [0.0, 1.0, 2.0, 1.0] and s.rewards.dtype == np.float64
        assert s.infos[2] == {"t": 1} and s.observations[:, 0].tolist() == [1, 1, 1, 1]

    def test_same_step_autoreset(self) -> None:
        vec = BatchCounting(2)
        vec.reset(seed=0, options=[{"horizon": 2}, {"horizon": 5}])
        vec.step(np.zeros(2, int))
        s = vec.step(np.zeros(2, int))
        assert s.terminated.tolist() == [True, False]
        assert s.observations[0, 0] == 0  # first observation of the new episode
        info = s.infos[0]
        assert info["final_observation"][0] == 2 and info["final_info"] == {"t": 2}
        assert info["reset_info"] == {"reset": True} and "final_observation" not in s.infos[1]

    def test_autoreset_off(self) -> None:
        vec = BatchCounting(2, horizon=1, autoreset=False)
        vec.reset(seed=0)
        s = vec.step(np.zeros(2, int))
        assert s.terminated.all() and s.observations[:, 0].tolist() == [1, 1]

    def test_seeding(self) -> None:
        a, b = BatchCounting(3), BatchCounting(3)
        np.testing.assert_array_equal(a.reset(seed=7).observations, b.reset(seed=7).observations)
        c = BatchCounting(3)
        np.testing.assert_array_equal(
            c.reset(seed=[1, 2, 3]).observations,
            BatchCounting(3).reset(seed=[1, 2, 3]).observations,
        )
        with pytest.raises(ValueError):
            c.reset(seed=[1, 2])

    def test_seeds_with_none_entries_continue_the_stream(self) -> None:
        a, b = BatchCounting(3), BatchCounting(3)
        for vec in (a, b):
            vec.reset(seed=0)
        # A None entry draws from the current stream: reproducible after a seeded reset...
        mixed = a.reset(seed=[1, None, 3]).observations
        np.testing.assert_array_equal(mixed, b.reset(seed=[1, None, 3]).observations)
        assert not np.array_equal(mixed, BatchCounting(3).reset(seed=[1, 2, 3]).observations)
        # ...and a list of only None continues it, exactly like seed=None.
        np.testing.assert_array_equal(a.reset(seed=[None] * 3).observations, b.reset().observations)
        # A Generator entry draws its contribution from that generator.
        rng = np.random.default_rng
        np.testing.assert_array_equal(
            a.reset(seed=[rng(5), 2, rng(6)]).observations,
            b.reset(seed=[rng(5), 2, rng(6)]).observations,
        )

    @pytest.mark.parametrize(
        "seed",
        [
            None,
            7,
            np.int64(7),
            [1, 2, 3],
            [1, None, 3],
            [None, None, None],
            [np.random.default_rng(0), 2, None],
            np.random.default_rng(0),
            [1, 2],
            "seven",
        ],
        ids=[
            "none",
            "int",
            "np-int",
            "ints",
            "ints-and-none",
            "nones",
            "generator-entry",
            "generator",
            "wrong-length",
            "str",
        ],
    )
    def test_accepts_exactly_what_sync_accepts(self, seed: Any) -> None:
        def outcome(vec: sekai.VecEnv) -> str:
            try:
                vec.reset(seed=seed)
            except Exception as e:
                return type(e).__name__
            return "ok"

        sync = sekai.SyncVecEnv([CountingEnv] * 3)
        assert outcome(BatchCounting(3)) == outcome(sync)

    def test_step_before_reset_raises(self) -> None:
        vec = BatchCounting(2)
        with pytest.raises(sekai.ResetRequired):
            vec.step(np.zeros(2, int))
        vec.reset(seed=0)
        assert vec.step(np.zeros(2, int)).observations[:, 0].tolist() == [1, 1]

    @pytest.mark.parametrize(
        "env_id", ["Trading-v0", "Portfolio-v0", "MarketMaking-v0", "OptimalExecution-v0"]
    )
    def test_finance_native_step_before_reset_raises_like_sync(self, env_id: str) -> None:
        for mode in ["native", "sync"]:
            vec = sekai.make_vec(env_id, 2, vectorization_mode=mode)
            with pytest.raises(sekai.ResetRequired):
                vec.step(vec.batch_action_space.sample())
            vec.close()

    def test_shared_members_are_not_split(self) -> None:
        class Scheduled(BatchCounting):
            @shared
            def schedule(self) -> np.ndarray:
                return np.linspace(0.0, 1.0, self.num_envs)[:, None]  # (n_steps, 1), n_steps == N

            @property
            @shared
            def levels(self) -> np.ndarray:
                return np.arange(float(self.num_envs))

            def per_env(self) -> np.ndarray:
                return np.arange(float(self.num_envs))[:, None]

        vec = Scheduled(4)
        vec.reset(seed=0)
        assert all(s.shape == (4, 1) for s in vec.call("schedule"))  # whole, for every env
        assert all(level.shape == (4,) for level in vec.get_attr("levels"))
        assert all(level.shape == (4,) for level in vec.call("levels"))
        assert [p.tolist() for p in vec.call("per_env")] == [[0.0], [1.0], [2.0], [3.0]]

    def test_attribute_access(self) -> None:
        vec = BatchCounting(3)
        vec.reset(seed=0)
        vec.step(np.zeros(3, int))
        assert vec.get_attr("t") == [1, 1, 1]
        assert vec.get_attr("num_envs") == [3, 3, 3]
        assert vec.call("double") == [2, 2, 2]
        vec.set_attr("horizon", [5, 6, 7])
        assert vec.horizon.tolist() == [5, 6, 7]

    def test_wrappers_and_close(self) -> None:
        vec = sekai.VecRecordEpisodeStatistics(sekai.VecNormalizeObservation(BatchCounting(2)))
        vec.reset(seed=0)
        lengths = []
        for _ in range(9):
            s = vec.step(np.ones(2, int))
            lengths += [i["episode"].episode_length for i in s.infos if "episode" in i]
        assert lengths == [3] * 6
        vec.close()
        with pytest.raises(sekai.VecEnvClosed):
            vec.step(np.ones(2, int))

    def test_episode_statistics_without_autoreset(self) -> None:
        vec = sekai.VecRecordEpisodeStatistics(BatchCounting(3, autoreset=False))
        vec.reset(seed=0, options=[{"horizon": 1}, {"horizon": 2}, {"horizon": 4}])
        finished = np.zeros(3, bool)
        episodes = []
        while not finished.all():
            result = vec.step(np.ones(3, int))
            finished |= result.dones
            episodes += [info["episode"] for info in result.infos if "episode" in info]
        assert [e.episode_length for e in episodes] == [1, 2, 4]  # each recorded once
        assert vec.tracker.summary()["num_episodes"] == 3

    def test_helpers(self) -> None:
        space = sekai.Dict(a=sekai.Box(0, 1, (2,)), b=sekai.Discrete(3))
        batch = {"a": np.arange(6.0).reshape(3, 2), "b": np.array([0, 1, 2])}
        row = take_index(space, batch, 1)
        assert row["a"].tolist() == [2.0, 3.0] and row["b"] == 1
        row["a"][0] = 99  # a copy
        assert batch["a"][1, 0] == 2.0
        infos = split_infos({"x": np.array([1.5, 2.5]), "w": np.ones((2, 3))}, 2)
        assert infos[0]["x"] == 1.5 and infos[1]["w"].shape == (3,)
        with pytest.raises(ValueError):
            split_infos({"x": np.ones(3)}, 2)


class TestMakeVec:
    def test_native_is_the_default(self, registered: str) -> None:
        vec = sekai.make_vec(registered, 4)
        assert isinstance(vec, BatchCounting) and vec.horizon.tolist() == [3] * 4
        assert vec.spec.id == registered
        assert isinstance(
            sekai.make_vec(registered, 2, vectorization_mode="sync"), sekai.SyncVecEnv
        )
        alias = sekai.make_vec(registered, 2, vectorization_mode="vector_entry_point")
        assert isinstance(alias, BatchCounting)

    def test_kwargs_and_vector_kwargs(self, registered: str) -> None:
        vec = sekai.make_vec(registered, 2, horizon=7, vector_kwargs={"autoreset": False})
        assert vec.horizon.tolist() == [7, 7] and vec.autoreset is False
        assert vec.spec.kwargs["horizon"] == 7

    def test_wrappers_fall_back_or_raise(self, registered: str) -> None:
        wrapped = sekai.make_vec(registered, 2, wrappers=[sekai.FlattenObservation])
        assert isinstance(wrapped, sekai.SyncVecEnv)  # wrappers need per-env objects
        with pytest.raises(ValueError, match="wrappers"):
            sekai.make_vec(
                registered, 2, vectorization_mode="native", wrappers=[sekai.FlattenObservation]
            )

    def test_registered_wrappers_fall_back_or_raise(self) -> None:
        env_id = "test/NativeClipped-v0"
        sekai.register(
            env_id,
            entry_point=CountingEnv,
            vector_entry_point=BatchCounting,
            additional_wrappers=(
                sekai.WrapperSpec("ClipReward", "sekai.wrappers:ClipReward", {"max_reward": 0.5}),
            ),
        )
        try:
            vec = sekai.make_vec(env_id, 2)
            assert isinstance(vec, sekai.SyncVecEnv)  # the native env would drop the wrapper
            vec.reset(seed=0)
            assert vec.step([2, 2]).rewards.tolist() == [0.5, 0.5]
            with pytest.raises(ValueError, match=r"additional_wrappers \(ClipReward\)"):
                sekai.make_vec(env_id, 2, vectorization_mode="native")
        finally:
            sekai.deregister(env_id)

    def test_no_native_implementation(self) -> None:
        with pytest.raises(ValueError, match="native"):
            sekai.make_vec("CartPole-v1", 2, vectorization_mode="native")

    def test_spec_serialisation(self) -> None:
        spec = sekai.EnvSpec("x/Y-v0", entry_point="m:E", vector_entry_point="m:VecE")
        assert sekai.EnvSpec.from_json(spec.to_json()).vector_entry_point == "m:VecE"
        assert "vector_entry_point=m:VecE" in spec.pprint()


class TestProcSharedMemory:
    def test_matches_sync(self) -> None:
        fns = [lambda: CountingEnv(horizon=3)] * 4
        sync = sekai.SyncVecEnv(fns)
        proc = sekai.ProcVecEnv(fns, num_workers=2)
        try:
            assert proc.shared_memory and proc.num_workers == 2
            np.testing.assert_array_equal(
                sync.reset(seed=3).observations, proc.reset(seed=3).observations
            )
            for _ in range(7):
                a, b = sync.step([1, 2, 0, 1]), proc.step([1, 2, 0, 1])
                np.testing.assert_array_equal(a.observations, b.observations)
                assert a.rewards.tolist() == b.rewards.tolist()
                for ia, ib in zip(a.infos, b.infos):
                    assert ia.keys() == ib.keys()
                    if "final_observation" in ia:
                        np.testing.assert_array_equal(
                            ia["final_observation"], ib["final_observation"]
                        )
        finally:
            sync.close()
            proc.close()

    def test_returned_batches_are_copies(self) -> None:
        proc = sekai.ProcVecEnv([lambda: CountingEnv(horizon=50)] * 2)
        try:
            first = proc.reset(seed=0).observations
            proc.step([0, 0])
            assert first[:, 0].tolist() == [0, 0]
        finally:
            proc.close()

    def test_nested_and_masked_observations(self) -> None:
        fns = [UniverseEnv] * 3
        proc = sekai.ProcVecEnv(fns, num_workers=2)
        sync = sekai.SyncVecEnv(fns)
        try:
            assert proc.shared_memory
            a, b = sync.reset(seed=1).observations, proc.reset(seed=1).observations
            np.testing.assert_array_equal(a["assets"]["values"], b["assets"]["values"])
            np.testing.assert_array_equal(a["assets"]["mask"], b["assets"]["mask"])
            masks = proc.call("action_mask")
            r = proc.step(np.stack(masks).astype(np.float32) * 0)
            assert proc.batch_observation_space.contains(r.observations)
        finally:
            proc.close()
            sync.close()
        d = sekai.ProcVecEnv([DictObsEnv] * 2)
        try:
            assert d.shared_memory and d.reset(seed=0).observations["b"].dtype == np.intp
        finally:
            d.close()

    def test_variable_size_observations_fall_back(self) -> None:
        proc = sekai.ProcVecEnv([TradesEnv] * 2)
        try:
            assert not proc.shared_memory
            assert len(proc.reset(seed=0).observations["trades"]) == 2
        finally:
            proc.close()
        with pytest.raises(TypeError, match="shared_memory"):
            sekai.ProcVecEnv([TradesEnv], shared_memory=True)

    def test_workers_keep_env_order_and_errors_name_the_env(self) -> None:
        from tests.test_vector import FailingEnv

        proc = sekai.ProcVecEnv(
            [lambda h=h: CountingEnv(horizon=h) for h in range(1, 6)], num_workers=2
        )
        try:
            assert proc.get_attr("horizon") == [1, 2, 3, 4, 5]
            proc.set_attr("horizon", [9, 8, 7, 6, 5])
            assert proc.get_attr("horizon") == [9, 8, 7, 6, 5]
        finally:
            proc.close()
        failing = sekai.ProcVecEnv([FailingEnv] * 3, num_workers=2, shared_memory=False)
        try:
            failing.reset(seed=0)
            with pytest.raises(sekai.VecEnvWorkerError) as exc:
                failing.step([0, 0, 2])
            assert exc.value.rank == 2
            assert failing.step([0, 0, 0]).rewards.tolist() == [0.0, 0.0, 0.0]
        finally:
            failing.close()

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            sekai.ProcVecEnv([CountingEnv], num_workers=0)


# ---------------------------------------------------------------------------
# Time limits, samplers and parameters of native envs
# ---------------------------------------------------------------------------


def assert_same(a: Any, b: Any, path: str = "") -> None:
    """Recursive equality for observations and infos (arrays and floats to rounding)."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys(), (path, a.keys(), b)
        for key in a:
            assert_same(a[key], b[key], f"{path}.{key}")
        return
    x, y = np.asarray(a), np.asarray(b)
    assert x.shape == y.shape, (path, x.shape, y.shape)
    if x.dtype.kind in "fc" or y.dtype.kind in "fc":
        np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-12, err_msg=path)
    else:
        np.testing.assert_array_equal(x, y, err_msg=path)


def same_value(a: Any, b: Any) -> bool:
    """Whether two attribute values (of a native env and of a sub-env) agree."""
    if isinstance(a, MarketData):
        return np.array_equal(a.prices, b.prices) and np.array_equal(a.timestamps, b.timestamps)
    if isinstance(a, (np.ndarray, np.number, int, float)) and not isinstance(a, bool):
        x, y = np.asarray(a), np.asarray(b)
        return x.shape == y.shape and bool(np.allclose(x, y, rtol=1e-12, equal_nan=True))
    if isinstance(a, (bool, np.bool_, str, dict, sekai.Space)) or a is None:
        return bool(a == b)
    return repr(a) == repr(b)  # cost models, reward functions, market models


class CurriculumStart(EpisodeSampler):
    """Random windows among the first `limit` starts; the caller raises `limit` over time."""

    def __init__(self) -> None:
        self.limit = 1

    def sample(self, rng: np.random.Generator, n_starts: int, episode_length: int) -> int:
        return int(rng.integers(0, min(self.limit, n_starts)))


def prices(n_assets: int | None = None, n_steps: int = 300) -> np.ndarray:
    rng = np.random.default_rng(1)
    shape = (n_steps,) if n_assets is None else (n_steps, n_assets)
    return 100 * np.exp(np.cumsum(rng.normal(0.0, 0.01, shape), axis=0))


# Deterministic configurations, so a native env and a SyncVecEnv agree exactly.
DETERMINISTIC = {
    "Trading-v0": {"prices": prices(), "episode_length": 30, "lookback": 5, "sampler": 7},
    "Portfolio-v0": {"prices": prices(3), "episode_length": 30, "lookback": 5, "sampler": 7},
    "MarketMaking-v0": {"sigma": 0.0, "fill_intensity": 1e9, "n_steps": 20, "max_inventory": 4},
    "OptimalExecution-v0": {"sigma": 0.0},
}


def run_both(native: Any, sync: Any, steps: int, seed: int = 0) -> list[Any]:
    """Step a native env and a SyncVecEnv with the same actions; check they agree."""
    assert_same(native.reset(seed=0).observations, sync.reset(seed=0).observations)
    actions = sync.batch_action_space
    actions.seed(seed)
    results = []
    for t in range(steps):
        a = actions.sample()
        x, y = native.step(a), sync.step(a)
        assert_same(x.observations, y.observations, f"t{t}.obs")
        assert_same(x.rewards, y.rewards, f"t{t}.rewards")
        assert x.terminated.tolist() == y.terminated.tolist(), t
        assert x.truncated.tolist() == y.truncated.tolist(), t
        for i, (ix, iy) in enumerate(zip(x.infos, y.infos)):
            assert_same(ix, iy, f"t{t}.infos[{i}]")
        results.append(x)
    return results


class TestNativeTimeLimit:
    def test_base_class_truncates_like_time_limit(self) -> None:
        native = BatchCounting(2, horizon=5, max_episode_steps=3)
        sync = sekai.SyncVecEnv([lambda: sekai.TimeLimit(CountingEnv(horizon=5), 3)] * 2)
        native.reset(seed=0, options=[{"horizon": 2}, None])
        sync.reset(seed=0)
        sync.set_attr("horizon", [2, 5])
        for _ in range(7):
            x, y = native.step(np.zeros(2, int)), sync.step([0, 0])
            assert x.terminated.tolist() == y.terminated.tolist()
            assert x.truncated.tolist() == y.truncated.tolist()
            for ix, iy in zip(x.infos, y.infos):
                assert ix["elapsed_steps"] == iy["elapsed_steps"]
                assert ("final_info" in ix) == ("final_info" in iy)
                if "final_info" in ix:
                    assert ix["final_info"]["elapsed_steps"] == iy["final_info"]["elapsed_steps"]
                    assert "elapsed_steps" not in ix["reset_info"]
        native.set_attr("max_episode_steps", 1)
        assert native.step(np.zeros(2, int)).truncated.all()
        with pytest.raises(ValueError, match="max_episode_steps"):
            BatchCounting(2, max_episode_steps=0)

    @pytest.mark.parametrize("env_id", sorted(DETERMINISTIC))
    @pytest.mark.parametrize("limit", [7, 45])  # before / after the env's own episode end
    def test_finance_native_matches_time_limited_sync(self, env_id: str, limit: int) -> None:
        kwargs = DETERMINISTIC[env_id]
        native = sekai.make_vec(env_id, 3, max_episode_steps=limit, **kwargs)
        sync = sekai.make_vec(
            env_id, 3, vectorization_mode="sync", max_episode_steps=limit, **kwargs
        )
        assert isinstance(native, sekai.NativeVecEnv) and native.max_episode_steps == limit
        results = run_both(native, sync, steps=100)
        assert all(info["elapsed_steps"] <= limit for r in results for info in r.infos)
        if limit == 7:  # shorter than every env's own episodes
            assert all(r.truncated.all() == (t % 7 == 6) for t, r in enumerate(results))

    @pytest.mark.parametrize("limit", [50, None])
    def test_elapsed_steps_matches_sync(self, limit: int | None) -> None:
        kwargs = {**DETERMINISTIC["Trading-v0"], "max_episode_steps": limit or -1}
        native = sekai.make_vec("Trading-v0", 2, **kwargs)
        sync = sekai.make_vec("Trading-v0", 2, vectorization_mode="sync", **kwargs)
        native.reset(seed=0)
        sync.reset(seed=0)
        assert native.get_attr("elapsed_steps") == sync.get_attr("elapsed_steps") == [0, 0]
        for t in range(1, 34):  # the episodes end after 30 steps
            native.step(np.ones(2, int))
            sync.step(np.ones(2, int))
            assert native.get_attr("elapsed_steps") == sync.get_attr("elapsed_steps"), t
            assert native.call("elapsed_steps") == [t % 30] * 2
        native.reset(seed=0)
        assert native.get_attr("elapsed_steps") == [0, 0]


# Frozen sub-envs keep being stepped (with their last action) while the others run; the
# native envs must leave their state alone (the Trading / Portfolio ones used to read past
# their price windows).
FROZEN_CASES = {
    "OptimalExecution-v0": np.array([[1.0], [0.3], [0.1], [0.04]], np.float32),  # sells at once
    "MarketMaking-v0": np.full((4, 2), 1.0, np.float32),
    "Trading-v0": np.array([0, 1, 2, 2]),
    "Portfolio-v0": np.array([[1, 0, 0, 0], [0, 0, 0, 1], [1, 1, 1, 0], [0, 2, 1, 1]], np.float32),
}


@pytest.mark.parametrize("limit", [None, 7])
@pytest.mark.parametrize("env_id", sorted(FROZEN_CASES))
def test_finance_native_freezes_finished_sub_envs_like_sync(env_id: str, limit: int | None) -> None:
    kwargs = {**DETERMINISTIC[env_id], "max_episode_steps": limit or -1}
    off = {"autoreset": False}
    native = sekai.make_vec(env_id, 4, vector_kwargs=off, **kwargs)
    sync = sekai.make_vec(env_id, 4, vectorization_mode="sync", vector_kwargs=off, **kwargs)
    assert_same(native.reset(seed=0).observations, sync.reset(seed=0).observations)
    done = np.zeros(4, bool)
    for t in range(75):  # well past every sub-env's end; sync never steps a finished one
        x, y = native.step(FROZEN_CASES[env_id]), sync.step(FROZEN_CASES[env_id])
        assert_same(x.observations, y.observations, f"t{t}.obs")
        assert_same(x.rewards, y.rewards, f"t{t}.rewards")
        assert x.terminated.tolist() == y.terminated.tolist(), t
        assert x.truncated.tolist() == y.truncated.tolist(), t
        for i, (ix, iy) in enumerate(zip(x.infos, y.infos)):
            assert_same(ix, iy, f"t{t}.infos[{i}]")
        assert not any(x.infos[i] for i in np.flatnonzero(done))  # frozen: empty infos
        done |= y.dones
    assert done.all() and native.frozen.all()


class TestNativeSamplers:
    def first_decisions(self, vec: Any, episodes: int) -> list[list[Any]]:
        """Each sub-env's first-decision timestamps over its first `episodes` episodes."""
        result = vec.reset(seed=0)
        out = [[info["timestamp"]] for info in result.infos]
        while min(len(seq) for seq in out) < episodes:
            result = vec.step(np.ones(vec.num_envs, int))
            for seq, info in zip(out, result.infos):
                if "reset_info" in info:
                    seq.append(info["reset_info"]["timestamp"])
        return [[int(t) for t in seq[:episodes]] for seq in out]

    @pytest.mark.parametrize("sampler", ["sequential", "instance"])
    def test_every_backend_gives_each_sub_env_its_own_sampler(self, sampler: str) -> None:
        kwargs = {"prices": prices(n_steps=120), "episode_length": 20, "lookback": 5}
        shared = SequentialStart()
        seen = {}
        for mode, vector_kwargs in [
            ("native", {}),
            ("sync", {}),
            ("async", {}),
            ("proc", {"num_workers": 2}),
        ]:
            vec = sekai.make_vec(
                "Trading-v0",
                3,
                vectorization_mode=mode,
                vector_kwargs=vector_kwargs,
                sampler=shared if sampler == "instance" else "sequential",
                **kwargs,
            )
            try:
                seen[mode] = self.first_decisions(vec, episodes=6)
            finally:
                vec.close()
        # Five windows of 20 steps, replayed in order by every sub-env (epoch wraps after 5).
        expected = [[5, 25, 45, 65, 85, 5]] * 3
        assert seen == dict.fromkeys(seen, expected)
        assert shared.epoch == 0 and shared.sample(np.random.default_rng(), 5, 20) == 0

    @pytest.mark.parametrize("mode", ["native", "sync", "async"])
    def test_a_shared_sampler_sees_the_callers_changes(self, mode: str) -> None:
        sampler = CurriculumStart()
        vec = sekai.make_vec(
            "Trading-v0",
            4,
            vectorization_mode=mode,
            prices=prices(n_steps=300),
            episode_length=20,
            lookback=5,
            sampler=sampler,
        )
        assert {info["timestamp"] for info in vec.reset(seed=0).infos} == {5}
        sampler.limit = 200  # the curriculum widens after make_vec()
        seen = set()
        for seed in range(5):
            seen |= {int(info["timestamp"]) for info in vec.reset(seed=seed).infos}
        assert len(seen) > 5 and max(seen) > 5
        vec.close()

    def test_a_stateless_sampler_is_not_copied_per_sub_env(self) -> None:
        sampler = CurriculumStart()
        kwargs = {"prices": prices(n_steps=300), "episode_length": 20, "lookback": 5}
        native = sekai.make_vec("Trading-v0", 64, sampler=sampler, **kwargs)
        assert all(lane is native.template._history for lane in native._windows.lanes)
        assert native.template._history.sampler is sampler
        sync = sekai.make_vec("Trading-v0", 3, vectorization_mode="sync", sampler=sampler, **kwargs)
        assert all(env.unwrapped._history.sampler is sampler for env in sync.envs)

    def test_spawn(self) -> None:
        rng = np.random.default_rng(0)
        walker = SequentialStart(start=3, stride=7)
        walker.sample(rng, 100, 20)
        twin = walker.spawn()
        assert twin is not walker and isinstance(twin, SequentialStart)
        assert (twin.start, twin.stride, twin.epoch, twin.sample(rng, 100, 20)) == (3, 7, 0, 3)
        assert walker.sample(rng, 100, 20) == 10  # the original keeps its own position
        random = RandomStart()
        assert random.spawn() is random and FixedStart(2).spawn().start == 2

    def test_a_single_env_keeps_the_callers_sampler(self) -> None:
        sampler = SequentialStart()
        env = sekai.make(
            "Trading-v0", prices=prices(n_steps=120), episode_length=20, lookback=5, sampler=sampler
        )
        for _ in range(5):
            env.reset(seed=0)
        assert sampler.epoch == 1

    def test_split_history_with_offset_samplers(self) -> None:
        kwargs = {"prices": prices(n_steps=179), "episode_length": 21, "lookback": 10}
        n = 4
        vec = sekai.SyncVecEnv(
            [
                lambda i=i: sekai.make(
                    "Trading-v0", sampler=SequentialStart(start=21 * i, stride=21 * n), **kwargs
                )
                for i in range(n)
            ]
        )
        firsts = sorted(t for seq in self.first_decisions(vec, episodes=2) for t in seq)
        assert firsts == list(range(10, 179 - 21, 21))  # every window exactly once

    @pytest.mark.parametrize("mode", ["native", "sync"])
    def test_split_history_with_start_options(self, mode: str) -> None:
        kwargs = {"prices": prices(n_steps=179), "episode_length": 21, "lookback": 10}
        starts = list(range(10, 179 - 21, 21))
        n = 4
        vec = sekai.make_vec(
            "Trading-v0", n, vectorization_mode=mode, vector_kwargs={"autoreset": False}, **kwargs
        )
        covered = []
        for k in range(0, len(starts), n):
            result = vec.reset(options=[{"start": t} for t in starts[k : k + n]])
            covered += [int(info["timestamp"]) for info in result.infos]
            done = np.zeros(n, bool)
            while not done.all():
                result = vec.step(np.ones(n, int))
                done |= result.dones
        assert covered == starts


class TestNativeParameters:
    @pytest.mark.parametrize(
        "env_id, kwargs, name, value",
        [
            ("OptimalExecution-v0", {}, "sigma", 0.0),
            ("MarketMaking-v0", {"fill_intensity": 1e9}, "sigma", 0.0),
            (
                "Trading-v0",
                {**DETERMINISTIC["Trading-v0"], "transaction_cost": FixedCost(50.0)},
                "capital",
                1e4,
            ),
            (
                "Portfolio-v0",
                {**DETERMINISTIC["Portfolio-v0"], "transaction_cost": FixedCost(50.0)},
                "capital",
                1e4,
            ),
        ],
    )
    def test_set_attr_changes_the_simulation_like_sync(
        self, env_id: str, kwargs: dict[str, Any], name: str, value: Any
    ) -> None:
        native = sekai.make_vec(env_id, 3, **kwargs)
        sync = sekai.make_vec(env_id, 3, vectorization_mode="sync", **kwargs)
        assert native.get_attr(name) == sync.get_attr(name)
        native.set_attr(name, value)
        sync.set_attr(name, [value] * 3)
        assert native.get_attr(name) == sync.get_attr(name) == [value] * 3
        run_both(native, sync, steps=40)

    def test_shared_parameters(self) -> None:
        vec = sekai.make_vec("Portfolio-v0", 5)  # five assets (the default), five sub-envs
        mu = vec.get_attr("mu")
        assert len(mu) == 5 and all(m.shape == (5,) for m in mu)  # repeated, not split
        with pytest.raises(ValueError, match="shared by every sub-env"):
            vec.set_attr("sigma", [0.1, 0.2, 0.3, 0.4, 0.5])
        vec.set_attr("dt", [1 / 52] * 5)  # one repeated value is fine
        assert vec.template.dt == 1 / 52
        with pytest.raises(AttributeError, match="lookback"):
            vec.set_attr("lookback", 5)
        with pytest.raises(AttributeError, match="episode_length is read-only"):
            vec.set_attr("episode_length", 40)
        with pytest.raises(AttributeError, match="set `cost_model` instead"):
            vec.set_attr("transaction_cost", 0.0)
        with pytest.raises(AttributeError, match="has no attribute 'no_such_parameter'"):
            vec.set_attr("no_such_parameter", 1.0)
        vec.set_attr("weights", vec.weights)  # state arrays stay settable

    @pytest.mark.parametrize("env_id", sorted(DETERMINISTIC))
    def test_get_attr_matches_sync(self, env_id: str) -> None:
        native = sekai.make_vec(env_id, 2, **DETERMINISTIC[env_id])
        sync = sekai.make_vec(env_id, 2, vectorization_mode="sync", **DETERMINISTIC[env_id])
        run_both(native, sync, steps=3)
        single = sync.envs[0].unwrapped
        # Everything the single env exposes but methods, the per-env objects / generators
        # a native env does not have (it has one generator for all), and the spec.
        names = [
            name
            for name in dir(single)
            if not name.startswith("_")
            and not callable(getattr(type(single), name, None))
            and name not in {"unwrapped", "rng", "spec"}
        ]
        assert {"metadata", "elapsed_steps", "observation_space"} <= set(names)
        for name in names:
            for x, y in zip(native.get_attr(name), sync.get_attr(name)):
                assert same_value(x, y), (name, x, y)

    def test_base_class_rejects_unknown_attributes(self) -> None:
        vec = BatchCounting(2)
        with pytest.raises(AttributeError, match="horizn"):
            vec.set_attr("horizn", 5)
        assert not hasattr(vec, "horizn")
