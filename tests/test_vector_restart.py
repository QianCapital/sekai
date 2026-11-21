"""ProcVecEnv fault tolerance: replacing lost workers (max_restarts, step_timeout)."""

from __future__ import annotations

import functools
import os
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import sekai
from tests.conftest import CountingEnv

LOST = r"\w+: worker process exited unexpectedly"  # EOFError, or e.g. ConnectionResetError


class CrashingEnv(CountingEnv):
    """A CountingEnv that kills its worker process on its `on`-th call to `method`.

    It first creates the `token` file, and does nothing if that file exists, so the
    replacement worker (built from the same env_fn) plays on. With `hang`, it sleeps
    instead, until step_timeout terminates the worker. Without a token it never fails.
    """

    def __init__(
        self,
        token: str | None = None,
        method: str = "step",
        on: int = 1,
        hang: bool = False,
        horizon: int = 5,
    ) -> None:
        super().__init__(horizon=horizon)
        self.token, self.method, self.on, self.hang = token, method, on, hang
        self.calls = 0

    def _maybe_fail(self, method: str) -> None:
        if self.token is None or method != self.method:
            return
        self.calls += 1
        if self.calls == self.on and not os.path.exists(self.token):
            Path(self.token).touch()
            if self.hang:
                time.sleep(60)
            os._exit(3)

    def reset(
        self, *, seed: Any = None, options: dict[str, Any] | None = None
    ) -> sekai.ResetResult:
        self._maybe_fail("reset")
        return super().reset(seed=seed, options=options)

    def step(self, action: int) -> sekai.StepResult:
        self._maybe_fail("step")
        return super().step(action)

    def exit_process(self) -> None:
        os._exit(3)


def build_once(token: str) -> CrashingEnv:
    """A sub-env whose env_fn fails once it has crashed: its replacement cannot start."""
    if os.path.exists(token):
        raise RuntimeError("cannot rebuild")
    return CrashingEnv(token)


def env_fns(num_envs: int, directory: Path, **failures: dict[str, Any]) -> list[Any]:
    """CrashingEnv factories; `env1={"on": 2}` makes sub-env 1 fail on its second step."""
    fns: list[Any] = []
    for i in range(num_envs):
        failure = failures.get(f"env{i}")
        if failure is None:
            fns.append(CrashingEnv)
        else:
            fns.append(functools.partial(CrashingEnv, str(directory / f"failed-{i}"), **failure))
    return fns


def assert_closed(vec: sekai.ProcVecEnv) -> None:
    start = time.monotonic()
    vec.close()
    assert time.monotonic() - start < 5.0
    assert vec.closed and not any(proc.is_alive() for proc in vec._processes)


def restart_warning(worker: int, envs: str, cause: str = LOST, count: str = "1 of 1") -> Any:
    restart, of = count.split(" of ")
    return pytest.warns(
        RuntimeWarning,
        match=(
            rf"^ProcVecEnv worker {worker} \({envs}\) was lost \({cause}\); restarting it "
            rf"\(restart {restart} of max_restarts={of}\)\.$"
        ),
    )


@pytest.mark.parametrize(
    "num_workers, shared_memory",
    [(None, None), (None, False), (2, None), (2, False)],
    ids=["one-each", "one-each-pickled", "grouped", "grouped-pickled"],
)
def test_step_restarts_a_lost_worker(
    num_workers: int | None, shared_memory: bool | None, tmp_path: Path
) -> None:
    fns = env_fns(4, tmp_path, env1={"on": 2})
    vec = sekai.ProcVecEnv(
        fns, num_workers=num_workers, shared_memory=shared_memory, max_restarts=1
    )
    try:
        assert vec.shared_memory is (shared_memory is None)
        vec.reset(seed=0)
        vec.step([1, 1, 1, 1])
        if num_workers is None:
            restarted, warning = [1], restart_warning(1, "sub-env 1")
        else:  # sub-envs 0 and 1 share worker 0: both are restarted
            restarted, warning = [0, 1], restart_warning(0, "sub-envs 0-1")
        with warning:
            result = vec.step([2, 2, 2, 2])
        for i in range(4):
            if i in restarted:  # a new episode, reported as a truncation
                assert result.observations[i, 0] == 0 and result.rewards[i] == 0.0
                assert not result.terminated[i] and result.truncated[i]
                assert result.infos[i] == {"worker_restarted": True, "reset_info": {"reset": True}}
            else:  # unaffected
                assert result.observations[i, 0] == 2 and result.rewards[i] == 2.0
                assert not result.dones[i] and result.infos[i] == {"t": 2}
        assert vec.restarts == 1
        assert vec.worker_restarts == ((0, 1, 0, 0) if num_workers is None else (1, 0))
        result = vec.step([1, 1, 1, 1])  # the replacement plays on
        assert result.observations[:, 0].tolist() == [1 if i in restarted else 3 for i in range(4)]
        assert not result.dones.any()
    finally:
        assert_closed(vec)


def test_restart_with_spawn_reattaches_shared_memory(tmp_path: Path) -> None:
    vec = sekai.ProcVecEnv(env_fns(2, tmp_path, env0={"on": 1}), context="spawn", max_restarts=1)
    try:
        assert vec.shared_memory
        vec.reset(seed=0)
        with restart_warning(0, "sub-env 0"):
            result = vec.step([1, 1])
        assert result.observations[:, 0].tolist() == [0, 1] and result.infos[0]["worker_restarted"]
        assert vec.step([1, 1]).observations[:, 0].tolist() == [1, 2]
    finally:
        assert_closed(vec)


def test_reset_is_redone_on_the_replacement(tmp_path: Path) -> None:
    fns = env_fns(3, tmp_path, env0={"method": "reset"})
    vec = sekai.ProcVecEnv(fns, num_workers=2, max_restarts=1)  # [0, 1] | [2]
    try:
        with restart_warning(0, "sub-envs 0-1"):
            result = vec.reset(seed=3, options={"x": 1})
        expected = sekai.SyncVecEnv([CountingEnv] * 3).reset(seed=3)
        np.testing.assert_array_equal(result.observations, expected.observations)
        assert result.infos == [{"reset": True, "worker_restarted": True}] * 2 + [{"reset": True}]
        assert vec.step([1, 1, 1]).observations[:, 0].tolist() == [1, 1, 1]
    finally:
        assert_closed(vec)


@pytest.mark.parametrize("method", ["step", "reset"])
def test_step_timeout_restarts_a_hung_worker(method: str, tmp_path: Path) -> None:
    fns = env_fns(2, tmp_path, env1={"method": method, "hang": True})
    vec = sekai.ProcVecEnv(fns, max_restarts=1, step_timeout=1.0)
    try:
        cause = r"TimeoutError: no reply within step_timeout=1.0 s; the worker was terminated"
        start = time.monotonic()
        with restart_warning(1, "sub-env 1", cause):
            reset = vec.reset(seed=0)
            result = vec.step([1, 1])
        assert time.monotonic() - start < 10.0
        if method == "reset":  # redone on the replacement
            assert reset.infos == [{"reset": True}, {"reset": True, "worker_restarted": True}]
            assert result.observations[:, 0].tolist() == [1, 1] and not result.dones.any()
        else:
            assert result.rewards.tolist() == [1.0, 0.0]
            assert (
                result.truncated.tolist() == [False, True] and result.infos[1]["worker_restarted"]
            )
            assert vec.step([1, 1]).observations[:, 0].tolist() == [2, 1]
    finally:
        assert_closed(vec)


def test_step_timeout_without_restarts_raises(tmp_path: Path) -> None:
    vec = sekai.ProcVecEnv(env_fns(2, tmp_path, env1={"hang": True}), step_timeout=1.0)
    try:
        vec.reset(seed=0)
        start = time.monotonic()
        with pytest.raises(
            sekai.VecEnvWorkerError,
            match=r"^Sub-env 1 raised TimeoutError: no reply within step_timeout=1.0 s; the "
            r"worker was terminated$",
        ) as exc:
            vec.step([1, 1])
        assert exc.value.rank == 1 and exc.value.exc_type == "TimeoutError"
        assert time.monotonic() - start < 5.0
        assert not vec._processes[1].is_alive()  # terminated, not left hanging
        with pytest.raises(sekai.VecEnvWorkerError, match=r"^Sub-env 1 raised "):
            vec.step([1, 1])
    finally:
        assert_closed(vec)


def test_without_max_restarts_a_lost_worker_raises_as_before(tmp_path: Path) -> None:
    vec = sekai.ProcVecEnv(env_fns(2, tmp_path, env1={"on": 1}))
    try:
        assert vec.max_restarts == 0 and vec.step_timeout is None
        vec.reset(seed=0)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(sekai.VecEnvWorkerError, match=rf"^Sub-env 1 raised {LOST}$"):
                vec.step([1, 1])
            with pytest.raises(sekai.VecEnvWorkerError, match=r"^Sub-env 1 raised \w+: worker"):
                vec.step([1, 1])
        assert not [w for w in caught if "ProcVecEnv worker" in str(w.message)]
        assert vec.restarts == 0
    finally:
        assert_closed(vec)


def test_used_up_restarts_raise(tmp_path: Path) -> None:
    vec = sekai.ProcVecEnv(env_fns(3, tmp_path, env0={"on": 1}, env1={"on": 2}), max_restarts=1)
    try:
        vec.reset(seed=0)
        with restart_warning(0, "sub-env 0"):
            vec.step([1, 1, 1])
        message = (
            rf"^Worker 1 \(sub-env 1\) raised {LOST}; not restarted \(1 worker lost, 1 of "
            r"max_restarts=1 restarts used\)$"
        )
        with pytest.raises(sekai.VecEnvWorkerError, match=message) as exc:
            vec.step([1, 1, 1])
        assert exc.value.rank == 1
        # The worker stays lost; every call that needs it says so.
        with pytest.raises(sekai.VecEnvWorkerError, match=r"not restarted \(1 worker lost"):
            vec.reset(seed=0)
        with pytest.raises(sekai.VecEnvWorkerError, match=r"not restarted \(1 worker lost"):
            vec.get_attr("t")
        assert vec.restarts == 1 and vec.worker_restarts == (1, 0, 0)
    finally:
        assert_closed(vec)


def test_more_lost_workers_than_restarts_left_restarts_none(tmp_path: Path) -> None:
    vec = sekai.ProcVecEnv(env_fns(2, tmp_path, env0={}, env1={}), max_restarts=1)
    try:
        vec.reset(seed=0)
        message = rf"^Worker 0 \(sub-env 0\) raised {LOST}; not restarted \(2 workers lost, 0 of"
        with pytest.raises(sekai.VecEnvWorkerError, match=message):
            vec.step([1, 1])
        assert vec.restarts == 0
    finally:
        assert_closed(vec)


def test_failed_replacement_raises(tmp_path: Path) -> None:
    fns = [functools.partial(build_once, str(tmp_path / "failed-0")), CrashingEnv]
    vec = sekai.ProcVecEnv(fns, max_restarts=2)
    try:
        vec.reset(seed=0)
        for attempt in (1, 2):  # each attempt uses a restart
            message = (
                rf"^Worker 0 \(sub-env 0\) was lost \(\w+: worker process "
                rf"(exited unexpectedly|is not running)\) and could not be restarted \(restart "
                rf"{attempt} of max_restarts=2\): Sub-env 0 raised RuntimeError: cannot rebuild"
            )
            with (
                pytest.warns(RuntimeWarning, match=f"restart {attempt} of max_restarts=2"),
                pytest.raises(sekai.VecEnvWorkerError, match=message) as exc,
            ):
                vec.step([1, 1])
            assert exc.value.rank == 0 and exc.value.exc_type == "RuntimeError"
            assert (
                "Worker traceback" in str(exc.value)
                and "cannot rebuild" in exc.value.remote_traceback
            )
        with pytest.raises(sekai.VecEnvWorkerError, match=r"not restarted \(1 worker lost, 2 of"):
            vec.step([1, 1])
        assert vec.restarts == 2
    finally:
        assert_closed(vec)


def test_lost_in_call_raises_and_the_next_step_restarts(tmp_path: Path) -> None:
    vec = sekai.ProcVecEnv([CrashingEnv] * 2, max_restarts=2)
    try:
        vec.reset(seed=0)
        message = (
            rf"^Worker 0 \(sub-env 0\) raised {LOST}; the next step\(\) or reset\(\) restarts it$"
        )
        with pytest.raises(sekai.VecEnvWorkerError, match=message):
            vec.call("exit_process")
        with pytest.raises(sekai.VecEnvWorkerError, match=r"the next step\(\) or reset\(\)"):
            vec.get_attr("t")
        assert vec.restarts == 0
        with pytest.warns(RuntimeWarning, match=r"^ProcVecEnv worker [01] \(sub-env [01]\) was"):
            result = vec.step([1, 1])
        assert result.truncated.all() and all(info["worker_restarted"] for info in result.infos)
        assert vec.restarts == 2 and vec.get_attr("t") == [0, 0]
    finally:
        assert_closed(vec)


@pytest.mark.parametrize("shared_memory", [None, False], ids=["shared", "pickled"])
def test_autoreset_off_restarted_sub_envs_are_frozen(
    shared_memory: bool | None, tmp_path: Path
) -> None:
    # One worker: sub-env 0 finishes (and freezes) on step 1, sub-env 1 kills the worker
    # on step 2. Sub-env 0 stays frozen on its own final step; 1 and 2 restart, frozen.
    fns = [
        functools.partial(CrashingEnv, horizon=1),
        functools.partial(CrashingEnv, str(tmp_path / "failed-1"), on=2),
        CrashingEnv,
    ]
    vec = sekai.ProcVecEnv(
        fns, num_workers=1, autoreset=False, shared_memory=shared_memory, max_restarts=1
    )
    try:
        vec.reset(seed=0)
        first = vec.step([1, 1, 1])
        assert first.terminated.tolist() == [True, False, False]
        with restart_warning(0, "sub-envs 0-2"):
            restart = vec.step([1, 1, 1])
        np.testing.assert_array_equal(restart.observations[0], first.observations[0])
        assert restart.terminated.tolist() == [True, False, False]
        assert restart.truncated.tolist() == [False, True, True]
        assert restart.rewards.tolist() == [0.0, 0.0, 0.0]
        assert restart.infos[0] == {}
        assert restart.infos[1] == {"worker_restarted": True, "reset_info": {"reset": True}}
        assert restart.observations[1:, 0].tolist() == [0, 0]
        later = vec.step([1, 1, 1])  # all three frozen until reset()
        np.testing.assert_array_equal(later.observations, restart.observations)
        assert later.dones.all() and later.rewards.tolist() == [0.0, 0.0, 0.0]
        assert later.terminated.tolist() == [True, False, False]
        assert later.infos == [{}, {}, {}]
        vec.reset(seed=1)
        result = vec.step([1, 1, 1])
        assert result.observations[:, 0].tolist() == [1, 1, 1]
        assert result.rewards.tolist() == [1.0, 1.0, 1.0]
    finally:
        assert_closed(vec)


def restart_seed(root: np.random.SeedSequence, k: int) -> int:
    """The documented seed of a sub-env's k-th restart since root seeded it."""
    return int(root.spawn(k + 1)[k].generate_state(1)[0])


@pytest.mark.parametrize("seed", [11, [5, 6, 7]], ids=["int", "list"])
def test_seeded_runs_with_the_same_failures_replay_exactly(seed: Any, tmp_path: Path) -> None:
    def run(directory: Path) -> list[np.ndarray]:
        directory.mkdir()
        fns = env_fns(3, directory, env1={"on": 2}, env2={"method": "reset", "on": 2})
        vec = sekai.ProcVecEnv(fns, max_restarts=2)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                observations = [vec.reset(seed=seed).observations]
                for _ in range(3):  # sub-env 1 is lost on step 2
                    observations.append(vec.step([1, 1, 1]).observations)
                observations.append(vec.reset().observations)  # and sub-env 2 here
                observations.append(vec.step([1, 1, 1]).observations)
            assert vec.restarts == 2
        finally:
            assert_closed(vec)
        return observations

    first, second = run(tmp_path / "a"), run(tmp_path / "b")
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    if isinstance(seed, int):
        roots = np.random.SeedSequence(seed).spawn(3)
    else:
        roots = [np.random.SeedSequence(s) for s in seed]
    # Sub-env 1's first restart, in step(); sub-env 2's, redoing a reset() given no seed.
    for i, observations in ((1, first[2]), (2, first[4])):
        expected = CountingEnv().reset(seed=restart_seed(roots[i], 0)).observation
        np.testing.assert_array_equal(observations[i], expected)


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="max_restarts"):
        sekai.ProcVecEnv([CountingEnv], max_restarts=-1)
    with pytest.raises(ValueError, match="step_timeout"):
        sekai.ProcVecEnv([CountingEnv], step_timeout=0)
