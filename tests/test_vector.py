from __future__ import annotations

import asyncio
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import sekai
import sekai.vector.proc_vec_env as vec_module
from tests.conftest import AsyncOnlyEnv, CountingEnv, DictObsEnv, run


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

    def test_transform_wrappers(self) -> None:
        base = sekai.SyncVecEnv([make_counting] * 2)
        vec = sekai.VecTransformReward(
            sekai.VecTransformObservation(base, lambda o: o + 100), np.negative
        )
        assert vec.reset(seed=0).observations[0, 0] == 100
        assert vec.step([1, 2]).rewards.tolist() == [-1.0, -2.0]
        assert vec.unwrapped is base


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
