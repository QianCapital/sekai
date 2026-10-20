from __future__ import annotations

import asyncio
import functools
import gc
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import sekai
import sekai.vector.proc_vec_env as vec_module
from tests.conftest import AsyncOnlyEnv, CountingEnv, DictObsEnv, run
from tests.test_dynamic import TradesEnv


class FailingEnv(CountingEnv):
    def step(self, action: int) -> sekai.StepResult:
        if action == 2:
            raise ValueError("boom")
        return super().step(action)


def make_counting(horizon: int = 3) -> CountingEnv:
    return CountingEnv(horizon=horizon)


@pytest.fixture(params=["sync", "async", "proc"])
def vec_cls(request: pytest.FixtureRequest) -> Any:
    return {"sync": sekai.SyncVecEnv, "async": sekai.AsyncVecEnv, "proc": sekai.ProcVecEnv}[
        request.param
    ]


class TestAllBackends:
    def test_reset_and_step_shapes(self, vec_cls: Any) -> None:
        with vec_cls([make_counting] * 3) as vec:
            r = vec.reset(seed=0)
            assert r.observations.shape == (3, 2) and len(r.infos) == 3
            s = vec.step(np.array([0, 1, 2]))
            assert s.rewards.tolist() == [0.0, 1.0, 2.0]
            assert s.rewards.dtype == np.float64 and s.dones.dtype == np.bool_
            assert vec.batch_observation_space.contains(s.observations)

    def test_seeding_is_deterministic(self, vec_cls: Any) -> None:
        with vec_cls([make_counting] * 2) as a, vec_cls([make_counting] * 2) as b:
            np.testing.assert_array_equal(
                a.reset(seed=5).observations, b.reset(seed=5).observations
            )
            obs = a.reset(seed=[1, 2]).observations
            assert obs[0, 1] != obs[1, 1]

    def test_autoreset(self, vec_cls: Any) -> None:
        with vec_cls([make_counting] * 2) as vec:
            vec.reset(seed=0)
            vec.step([0, 0])
            vec.step([0, 0])
            r = vec.step([0, 0])  # horizon 3: episode ends, sub-envs reset
            assert r.terminated.all()
            assert (r.observations[:, 0] == 0).all()
            assert all(info["final_observation"][0] == 3 for info in r.infos)
            assert all(info["final_info"]["t"] == 3 for info in r.infos)

    def test_autoreset_off(self, vec_cls: Any) -> None:
        with vec_cls([make_counting] * 2, autoreset=False) as vec:
            vec.reset(seed=0)
            for _ in range(3):
                r = vec.step([0, 0])
            assert (r.observations[:, 0] == 3).all() and "final_observation" not in r.infos[0]

    def test_attr_access(self, vec_cls: Any) -> None:
        with vec_cls([make_counting] * 2) as vec:
            assert vec.get_attr("horizon") == [3, 3]
            vec.set_attr("horizon", [7, 8])
            assert vec.get_attr("horizon") == [7, 8]
            assert vec.call("horizon") == [7, 8]
            assert len(vec.call("_obs")) == 2

    def test_dict_observations(self, vec_cls: Any) -> None:
        with vec_cls([DictObsEnv] * 2) as vec:
            r = vec.reset(seed=0)
            assert r.observations["a"].shape == (2, 3) and r.observations["b"].shape == (2,)
            s = vec.step(np.zeros((2, 2), np.float32))
            assert vec.batch_observation_space.contains(s.observations)

    def test_wrong_action_count(self, vec_cls: Any) -> None:
        with vec_cls([make_counting] * 2) as vec:
            vec.reset(seed=0)
            with pytest.raises(ValueError):
                vec.step([0, 0, 0])


def test_proc_worker_error_is_reported_and_recoverable() -> None:
    vec = sekai.ProcVecEnv([FailingEnv] * 2)
    try:
        vec.reset(seed=0)
        with pytest.raises(sekai.VecEnvWorkerError, match="boom") as exc:
            vec.step([0, 2])
        assert exc.value.rank == 1 and "ValueError" in str(exc.value)
        assert vec.step([0, 0]).rewards.tolist() == [0.0, 0.0]  # workers are still alive
    finally:
        vec.close()
    with pytest.raises(sekai.VecEnvClosed):
        vec.step([0, 0])


def test_proc_env_fn_error() -> None:
    def broken() -> CountingEnv:
        raise RuntimeError("cannot build")

    with pytest.raises(sekai.VecEnvWorkerError, match="cannot build"):
        sekai.ProcVecEnv([broken])


def test_async_vec_uses_async_methods() -> None:
    envs: list[AsyncOnlyEnv] = []

    def factory() -> AsyncOnlyEnv:
        envs.append(AsyncOnlyEnv())
        return envs[-1]

    async def main() -> None:
        vec = sekai.AsyncVecEnv([factory] * 3)
        await vec.async_reset()
        r = await vec.async_step([0, 1, 0])
        assert r.observations[:, 0].tolist() == [1.0, 1.0, 1.0]
        with pytest.raises(RuntimeError, match="running event loop"):
            vec.step([0, 0, 0])
        vec.close()

    run(main())
    assert all(env.async_calls == 2 for env in envs)


def test_async_vec_sync_api_without_running_loop() -> None:
    vec = sekai.AsyncVecEnv([make_counting] * 2)
    vec.reset(seed=0)
    assert vec.step([1, 1]).rewards.tolist() == [1.0, 1.0]
    vec.close()
    vec.close()  # idempotent
    with pytest.raises(RuntimeError):
        asyncio.get_running_loop()


def test_skip_double_reset_with_autoreset_wrapper() -> None:
    vec = sekai.SyncVecEnv([lambda: sekai.AutoReset(CountingEnv(horizon=1))] * 2)
    vec.reset(seed=0)
    r = vec.step([0, 0])
    assert (r.observations[:, 0] == 0).all()
    assert r.infos[0]["final_observation"][0] == 1


class TestVecWrappers:
    def test_record_episode_statistics(self) -> None:
        vec = sekai.VecRecordEpisodeStatistics(
            sekai.SyncVecEnv([lambda: CountingEnv(2), lambda: CountingEnv(3)])
        )
        vec.reset(seed=0)
        r1 = vec.step([1, 1])
        r2 = vec.step([1, 1])
        assert "episode" not in r1.infos[0]
        assert r2.infos[0]["episode"].episode_return == 2.0 and "episode" not in r2.infos[1]
        r3 = vec.step([1, 1])
        assert r3.infos[1]["episode"].episode_length == 3
        assert r3.infos[1]["episode"].extra["env_index"] == 1

    def test_normalize_observation(self) -> None:
        vec = sekai.VecNormalizeObservation(sekai.SyncVecEnv([lambda: CountingEnv(4)] * 4))
        vec.reset(seed=0)
        for _ in range(50):
            r = vec.step([0] * 4)
        assert np.abs(r.observations).max() < 5
        finals = [i for i in r.infos if "final_observation" in i]
        assert all(i["final_observation"].dtype == np.float64 for i in finals)
        assert vec.observation_space.dtype == np.float64

    def test_normalize_reward(self) -> None:
        vec = sekai.VecNormalizeReward(sekai.SyncVecEnv([lambda: CountingEnv(5)] * 2))
        vec.reset(seed=0)
        for _ in range(30):
            r = vec.step([2, 2])
        assert np.all(np.abs(r.rewards) < 5)

    def test_record_episode_statistics_without_autoreset(self) -> None:
        # Evaluation: step until every sub-env has finished; the finished ones keep being
        # stepped (and keep reporting done) but their episode is recorded once.
        vec = sekai.VecRecordEpisodeStatistics(
            sekai.SyncVecEnv([lambda: CountingEnv(1), lambda: CountingEnv(3)], autoreset=False)
        )
        for round_ in range(1, 3):
            vec.reset(seed=0)
            finished = np.zeros(2, bool)
            episodes = []
            while not finished.all():
                result = vec.step([1, 2])
                finished |= result.dones
                episodes += [info["episode"] for info in result.infos if "episode" in info]
            assert [(e.episode_length, e.episode_return) for e in episodes] == [(1, 1), (3, 6)]
            assert vec.tracker.summary()["num_episodes"] == 2 * round_  # reset() re-arms

    def test_record_episode_statistics_with_self_resetting_sub_envs(self) -> None:
        # Without vec autoreset, a sub-env wrapped in AutoReset still plays episode after episode.
        vec = sekai.VecRecordEpisodeStatistics(
            sekai.SyncVecEnv([lambda: sekai.AutoReset(CountingEnv(2))], autoreset=False)
        )
        vec.reset(seed=0)
        for _ in range(6):
            vec.step([1])
        assert vec.tracker.summary()["num_episodes"] == 3

    def test_normalize_observation_ewm_halflife_counts_vector_steps(self) -> None:
        # One vector step is one update, however many sub-envs: no need to scale halflife.
        moments = []
        for n in (1, 8):
            vec = sekai.VecNormalizeObservation(
                sekai.SyncVecEnv([lambda: CountingEnv(100)] * n), stats="ewm", halflife=10.0
            )
            vec.reset(seed=0)
            for _ in range(20):
                vec.step([0] * n)
            moments.append((vec.obs_rms.count, vec.obs_rms.weight))
        assert moments[0] == moments[1] and moments[0][0] == 21  # the reset, then 20 steps

    def test_transform_wrappers(self) -> None:
        base = sekai.SyncVecEnv([make_counting] * 2)
        vec = sekai.VecTransformReward(
            sekai.VecTransformObservation(base, lambda o: o + 100), np.negative
        )
        assert vec.reset(seed=0).observations[0, 0] == 100
        assert vec.step([1, 2]).rewards.tolist() == [-1.0, -2.0]
        assert vec.unwrapped is base

    def test_transform_wrappers_take_func_like_gymnasium(self) -> None:
        base = sekai.SyncVecEnv([make_counting] * 2)
        obs = sekai.VecTransformObservation(base, func=lambda o: o + 100)
        vec = sekai.VecTransformReward(obs, func=np.negative)
        assert vec.func is np.negative and obs.func(np.zeros(1))[0] == 100
        vec.reset(seed=0)
        assert vec.step([1, 2]).rewards.tolist() == [-1.0, -2.0]


def test_make_vec() -> None:
    vec = sekai.make_vec("CartPole-v1", num_envs=3, wrappers=[sekai.RecordEpisodeStatistics])
    assert isinstance(vec, sekai.SyncVecEnv) and vec.num_envs == 3
    assert isinstance(vec.envs[0], sekai.RecordEpisodeStatistics)
    vec.reset(seed=0)
    vec.step(vec.batch_action_space.sample())
    vec.close()
    with pytest.raises(ValueError):
        sekai.make_vec("CartPole-v1", num_envs=2, vectorization_mode="gpu")  # type: ignore[arg-type]


def test_vec_action_wrapper() -> None:
    class Plus1(sekai.VecActionWrapper):
        def action(self, actions: Any) -> Any:
            return np.asarray(actions) + 1

    vec = Plus1(sekai.SyncVecEnv([make_counting] * 2))
    vec.reset(seed=0)
    assert vec.step([0, 1]).rewards.tolist() == [1.0, 2.0]
    assert vec.batch_action_space == sekai.MultiDiscrete([3, 3])


def test_stack_infos() -> None:
    stacked = sekai.stack_infos([{"a": 1, "b": 2}, {"a": 3}])
    assert stacked.keys() == {"a"} and stacked["a"].tolist() == [1, 3]
    assert sekai.stack_infos([]) == {}


def test_seed_and_option_validation() -> None:
    with sekai.SyncVecEnv([make_counting] * 2) as vec:
        with pytest.raises(ValueError):
            vec.reset(seed=[1])
        with pytest.raises(ValueError):
            vec.reset(options=[{}])
        with pytest.raises(TypeError):
            vec.reset(seed=np.random.default_rng(0))  # type: ignore[arg-type]
        vec.reset(seed=0, options=[{"x": 1}, None])


@pytest.mark.parametrize("vec_cls", [sekai.SyncVecEnv, sekai.AsyncVecEnv, sekai.ProcVecEnv])
def test_closed_vec_env_raises(vec_cls: Any) -> None:
    vec = vec_cls([make_counting] * 2)
    vec.close()
    assert vec.closed
    with pytest.raises(sekai.VecEnvClosed):
        vec.reset(seed=0)
    with pytest.raises(sekai.VecEnvClosed):
        vec.step([0, 0])


class CloseRecordingEnv(CountingEnv):
    """Touches `<directory>/closed-<index>` when closed; sub-env 0 fails to close."""

    def __init__(self, index: int, directory: str) -> None:
        super().__init__()
        self.index, self.directory = index, directory

    def close(self) -> None:
        if self.index == 0:
            raise RuntimeError("close failed in sub-env 0")
        (Path(self.directory) / f"closed-{self.index}").touch()


class StubbornEnv(CountingEnv):
    """Ignores SIGTERM and never finishes closing."""

    def __init__(self) -> None:
        super().__init__()
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

    def close(self) -> None:
        time.sleep(60)


@pytest.mark.parametrize(
    "build",
    [
        sekai.SyncVecEnv,
        sekai.AsyncVecEnv,
        sekai.ProcVecEnv,
        lambda fns: sekai.ProcVecEnv(fns, num_workers=2),  # sub-envs 0 and 1 share a worker
    ],
    ids=["sync", "async", "proc", "proc-grouped"],
)
def test_close_closes_every_sub_env_then_raises(build: Any, tmp_path: Path) -> None:
    vec = build([lambda i=i: CloseRecordingEnv(i, str(tmp_path)) for i in range(3)])
    vec.reset(seed=0)
    start = time.monotonic()
    with pytest.raises((RuntimeError, sekai.VecEnvWorkerError), match="close failed in sub-env 0"):
        vec.close()
    assert time.monotonic() - start < 2.0  # no wait for close_timeout
    assert sorted(p.name for p in tmp_path.iterdir()) == ["closed-1", "closed-2"]
    assert vec.closed
    vec.close()  # a no-op now
    if isinstance(vec, sekai.ProcVecEnv):
        assert not any(proc.is_alive() for proc in vec._processes)
    if isinstance(vec, sekai.AsyncVecEnv):
        assert vec._loop is None


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM cannot be ignored on Windows")
def test_proc_close_kills_workers_that_ignore_terminate() -> None:
    vec = sekai.ProcVecEnv([StubbornEnv] * 6, close_timeout=0.5)
    vec.reset(seed=0)
    start = time.monotonic()
    vec.close()
    # close_timeout, then one shared grace period before the kill (not one per worker).
    assert time.monotonic() - start < 3.5
    assert not any(proc.is_alive() for proc in vec._processes)


def test_batch_action_space_is_cached_so_seeding_sticks() -> None:
    vec = sekai.SyncVecEnv([make_counting] * 8)
    assert vec.batch_action_space is vec.batch_action_space
    vec.batch_action_space.seed(0)
    first = vec.batch_action_space.sample()
    vec.batch_action_space.seed(0)
    np.testing.assert_array_equal(vec.batch_action_space.sample(), first)


def test_proc_unpicklable_payload_does_not_desync() -> None:
    import threading

    vec = sekai.ProcVecEnv([make_counting] * 3)
    try:
        vec.reset(seed=0)
        with pytest.raises(TypeError):
            vec.set_attr("tag", ["a", threading.Lock(), "c"])  # nothing is sent
        assert vec.get_attr("horizon") == [3, 3, 3]
        assert vec.step([1, 1, 1]).rewards.tolist() == [1.0, 1.0, 1.0]
    finally:
        vec.close()


def test_proc_dead_worker_raises_worker_error() -> None:
    vec = sekai.ProcVecEnv([make_counting] * 2)
    try:
        vec.reset(seed=0)
        vec._processes[1].kill()
        vec._processes[1].join()
        for _ in range(2):  # the failure is reported consistently, never as stale data
            with pytest.raises(sekai.VecEnvWorkerError):
                vec.step([0, 0])
    finally:
        vec.close()


def test_proc_drains_unread_replies() -> None:
    vec = sekai.ProcVecEnv([make_counting] * 2)
    try:
        vec.reset(seed=0)
        vec._send_all(vec_module._Cmd.STEP, [0, 0])  # simulate a call interrupted before _gather
        obs = vec.reset(seed=1).observations
        expected = sekai.SyncVecEnv([make_counting] * 2).reset(seed=1).observations
        np.testing.assert_array_equal(obs, expected)
    finally:
        vec.close()


def test_async_vec_error_leaves_no_pending_sub_env_calls() -> None:
    class Flaky(CountingEnv):
        fail_next = False

        async def async_step(self, action: int) -> sekai.StepResult:
            if self.fail_next:
                self.fail_next = False
                raise ConnectionError("feed dropped")
            await asyncio.sleep(0.01)
            return self.step(action)

    vec = sekai.AsyncVecEnv([lambda: Flaky(horizon=100)] * 2)
    vec.reset(seed=0)
    vec.envs[0].fail_next = True  # type: ignore[attr-defined]
    with pytest.raises(ConnectionError):
        vec.step([0, 0])
    assert vec.envs[1].t == 1  # finished before the error was raised...
    vec.step([0, 0])
    assert vec.envs[1].t == 2  # ...and nothing from the failed call ran again
    vec.close()


def test_vec_wrappers_keep_async_api() -> None:
    envs: list[AsyncOnlyEnv] = []

    def factory() -> AsyncOnlyEnv:
        envs.append(AsyncOnlyEnv())
        return envs[-1]

    async def main() -> None:
        vec = sekai.VecRecordEpisodeStatistics(
            sekai.VecNormalizeReward(
                sekai.VecTransformObservation(sekai.AsyncVecEnv([factory] * 2), lambda o: o * 2)
            )
        )
        await vec.async_reset(seed=0)
        r = await vec.async_step([0, 0])
        assert r.observations[:, 0].tolist() == [2.0, 2.0]
        vec.close()

    run(main())
    assert all(env.async_calls == 2 for env in envs)


def test_vec_observation_wrappers_transform_final_observation() -> None:
    vec = sekai.VecTransformObservation(
        sekai.SyncVecEnv([lambda: CountingEnv(1)] * 2), lambda o: o + 100
    )
    vec.reset(seed=0)
    r = vec.step([0, 0])
    assert r.observations[0, 0] == 100  # new episode's first obs, transformed
    assert r.infos[0]["final_observation"][0] == 101  # terminal obs (t=1), transformed too


def test_sync_only_vec_wrapper_gets_async_fallback() -> None:
    class Negate(sekai.VecWrapper):
        def step(self, actions: Any) -> sekai.VecStepResult:
            r = self.venv.step(actions)
            return sekai.VecStepResult(
                r.observations, -r.rewards, r.terminated, r.truncated, r.infos
            )

    vec = Negate(sekai.SyncVecEnv([make_counting] * 2))
    run(vec.async_reset(seed=0))
    assert run(vec.async_step([1, 2])).rewards.tolist() == [-1.0, -2.0]


@pytest.mark.parametrize("mode", ["sync", "async", "proc"])
def test_attribute_access_reaches_through_wrappers(mode: str) -> None:
    vec = sekai.make_vec("CartPole-v1", num_envs=2, vectorization_mode=mode)
    try:
        assert vec.get_attr("x_threshold") == [2.4, 2.4]  # on CartPoleEnv, under TimeLimit
        assert vec.get_attr("max_episode_steps") == [500, 500]  # on the TimeLimit wrapper
        vec.set_attr("x_threshold", [2.0, 3.0])
        assert vec.get_attr("x_threshold") == [2.0, 3.0]
        assert vec.call("elapsed_steps") == [0, 0]
    finally:
        vec.close()


class ExitingEnv(CountingEnv):
    """Sub-env `index` kills its worker process on its first step if it is `fail`."""

    def __init__(self, index: int, fail: int = 1) -> None:
        super().__init__()
        self.index, self.fail = index, fail

    def step(self, action: int) -> sekai.StepResult:
        if self.index == self.fail:
            os._exit(3)
        return super().step(action)


@pytest.mark.parametrize(
    "num_workers, rank, message",
    [
        (None, 1, r"^Sub-env 1 raised "),  # the worker ran sub-env 1 only
        (2, 0, r"^Worker 0 \(sub-envs 0-1\) raised "),  # sub-env 0 or 1: unknown
    ],
)
def test_proc_dead_worker_error_names_what_is_known(
    num_workers: int | None, rank: int, message: str
) -> None:
    vec = sekai.ProcVecEnv([lambda i=i: ExitingEnv(i) for i in range(4)], num_workers=num_workers)
    try:
        vec.reset(seed=0)
        for _ in range(2):  # the failed call, then the next one on the dead worker
            with pytest.raises(sekai.VecEnvWorkerError, match=message) as exc:
                vec.step([0, 0, 0, 0])
            assert exc.value.rank == rank
    finally:
        vec.close()


class MarkClosedEnv(CountingEnv):
    """Touches `<directory>/closed-<index>` when closed."""

    def __init__(self, index: int, directory: str) -> None:
        super().__init__()
        self.index, self.directory = index, directory

    def close(self) -> None:
        (Path(self.directory) / f"closed-{self.index}").touch()


class MarkClosedTradesEnv(TradesEnv):
    """Variable-size observations (no shared memory); touches a file when closed."""

    def __init__(self, index: int, directory: str) -> None:
        super().__init__()
        self.index, self.directory = index, directory

    def close(self) -> None:
        (Path(self.directory) / f"closed-{self.index}").touch()


def build_or_fail(index: int, directory: str, fail: int, *unpicklable: Any) -> MarkClosedEnv:
    if index == fail:
        raise RuntimeError(f"cannot build sub-env {index}")
    return MarkClosedEnv(index, directory)


@pytest.mark.parametrize(
    "build, closed",
    [
        (sekai.SyncVecEnv, [0, 1]),  # sub-env 3 is never built
        (sekai.AsyncVecEnv, [0, 1]),
        (sekai.ProcVecEnv, [0, 1, 3]),  # one worker each: 3 was built, then closed
        (lambda fns: sekai.ProcVecEnv(fns, num_workers=1), [0, 1]),  # closed by the worker
        (lambda fns: sekai.ProcVecEnv(fns, num_workers=2), [0, 1]),  # [0, 1] | [2, 3]
    ],
    ids=["sync", "async", "proc", "proc-one-worker", "proc-two-workers"],
)
def test_failed_env_fn_closes_the_sub_envs_already_built(
    build: Any, closed: list[int], tmp_path: Path
) -> None:
    fns = [functools.partial(build_or_fail, i, str(tmp_path), 2) for i in range(4)]
    with pytest.raises((RuntimeError, sekai.VecEnvWorkerError), match="cannot build sub-env 2"):
        build(fns)
    assert sorted(p.name for p in tmp_path.iterdir()) == [f"closed-{i}" for i in closed]


def test_proc_failed_constructor_closes_the_sub_envs(tmp_path: Path) -> None:
    # Failing after the workers built their sub-envs: they are closed, not just terminated.
    fns = [functools.partial(MarkClosedTradesEnv, i, str(tmp_path)) for i in range(2)]
    with pytest.raises(TypeError, match="shared_memory"):
        sekai.ProcVecEnv(fns, shared_memory=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["closed-0", "closed-1"]
    # Failing to start a worker (its env_fn cannot be pickled) closes those already started.
    lock = threading.Lock()
    fns = [functools.partial(build_or_fail, i, str(tmp_path / "b"), -1) for i in range(3)]
    fns[1] = functools.partial(build_or_fail, 1, str(tmp_path / "b"), -1, lock)
    (tmp_path / "b").mkdir()
    with pytest.raises(TypeError, match="pickle"):
        sekai.ProcVecEnv(fns, context="spawn")
    assert sorted(p.name for p in (tmp_path / "b").iterdir()) == ["closed-0"]


def test_proc_close_error_leaves_no_pickle_buffer_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unraisable: list[Any] = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)
    vec = sekai.ProcVecEnv([lambda i=i: CloseRecordingEnv(i, str(tmp_path)) for i in range(2)])
    vec.reset(seed=0)
    error: Exception | None = None
    try:
        vec.close()
    except sekai.VecEnvWorkerError as e:
        error = e  # keeps close()'s frame alive through the traceback, as a caller may
    assert error is not None
    del error
    gc.collect()
    # The pickled CLOSE command was a memoryview pinning its BytesIO: collecting the
    # traceback cycle printed "Exception ignored ... BufferError: Existing exports of data".
    assert not [u for u in unraisable if isinstance(u.exc_value, BufferError)]
