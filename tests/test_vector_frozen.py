"""With autoreset off, a finished sub-env is frozen until reset(), on every backend."""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

import sekai
from sekai.stats.collectors import InMemoryTracker
from sekai.vector.native import NativeVecEnv

HORIZONS = (1, 3, 6)  # sub-env 0 ends after one step, 1 after three; 2 runs into the limit
LIMIT = 4


def _space() -> sekai.Box:
    return sekai.Box(0.0, 100.0, shape=(3,), dtype=np.float32)


class StrictCountdown(sekai.Env[np.ndarray, int]):
    """obs [t, horizon, last action]; reward = action; terminates at `horizon`.

    Like the finance envs, it raises ResetRequired when stepped after its episode ended,
    and InvalidAction for an action outside its space.
    """

    def __init__(self, horizon: int) -> None:
        self.horizon = horizon
        self.observation_space = _space()
        self.action_space = sekai.Discrete(3)
        self.t = self.last = 0
        self.over = True

    def _obs(self) -> np.ndarray:
        return np.array([self.t, self.horizon, self.last], dtype=np.float32)

    def reset(
        self, *, seed: Any = None, options: dict[str, Any] | None = None
    ) -> sekai.ResetResult:
        self._set_rng(seed)
        self.t = self.last = 0
        self.over = False
        return sekai.ResetResult(self._obs(), {"t": 0})

    def step(self, action: int) -> sekai.StepResult:
        if self.over:
            raise sekai.ResetRequired()
        if not self.action_space.contains(action):
            raise sekai.InvalidAction(action, self.action_space)
        self.t += 1
        self.last = int(action)
        self.over = self.t >= self.horizon
        return sekai.StepResult(self._obs(), float(action), self.over, False, {"t": self.t})


class BatchCountdown(NativeVecEnv[np.ndarray, Any]):
    """StrictCountdown natively vectorised; frozen sub-envs keep their state."""

    def __init__(self, num_envs: int, horizons: Sequence[int], **kwargs: Any) -> None:
        super().__init__(num_envs, _space(), sekai.Discrete(3), **kwargs)
        self.horizon = np.asarray(horizons, dtype=np.int64)
        self.t = np.zeros(num_envs, dtype=np.int64)
        self.last = np.zeros(num_envs, dtype=np.int64)

    def _reset_envs(self, indices: np.ndarray, options: Sequence[dict[str, Any] | None]) -> None:
        self.t[indices] = 0
        self.last[indices] = 0

    def _observe(self) -> np.ndarray:
        return np.stack([self.t, self.horizon, self.last], axis=1).astype(np.float32)

    def _step(self, actions: Any) -> tuple[Any, Any, Any, dict[str, np.ndarray]]:
        a = np.asarray(actions)
        if not np.all((a >= 0) & (a < 3)):  # a frozen sub-env gets its last action again
            raise sekai.InvalidAction(actions, self.action_space)
        active = ~self.frozen
        self.t = self.t + active
        self.last = np.where(active, a, self.last)
        return (
            a.astype(np.float64),
            self.t >= self.horizon,
            np.zeros(self.num_envs, bool),
            {"t": self.t.copy()},
        )

    def _reset_infos(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        return {"t": self.t[indices]}


def strict(horizon: int) -> sekai.Env[np.ndarray, int]:
    return sekai.TimeLimit(StrictCountdown(horizon), LIMIT)


BACKENDS = {
    "native": lambda: BatchCountdown(3, HORIZONS, autoreset=False, max_episode_steps=LIMIT),
    "sync": lambda: sekai.SyncVecEnv(
        [functools.partial(strict, h) for h in HORIZONS], autoreset=False
    ),
    "async": lambda: sekai.AsyncVecEnv(
        [functools.partial(strict, h) for h in HORIZONS], autoreset=False
    ),
    "proc": lambda: sekai.ProcVecEnv(
        [functools.partial(strict, h) for h in HORIZONS], autoreset=False, num_workers=2
    ),
    "proc-pickled": lambda: sekai.ProcVecEnv(
        [functools.partial(strict, h) for h in HORIZONS], autoreset=False, shared_memory=False
    ),
}


def play(vec: sekai.VecEnv, steps: int = 8) -> list[Any]:
    """Two rounds of reset + `steps` steps; finished sub-envs get an invalid action (99)."""
    results: list[Any] = []
    for _ in range(2):
        results.append(vec.reset(seed=0))
        done = np.zeros(vec.num_envs, bool)
        for t in range(steps):
            actions = np.where(done, 99, [(t + i) % 3 for i in range(vec.num_envs)])
            result = vec.step(actions)
            done |= result.dones
            results.append(result)
    return results


def as_plain(result: Any) -> dict[str, Any]:
    out = {"obs": result.observations.tolist(), "infos": result.infos}
    if isinstance(result, sekai.VecStepResult):
        out.update(
            rewards=result.rewards.tolist(),
            terminated=result.terminated.tolist(),
            truncated=result.truncated.tolist(),
        )
    return out


@pytest.fixture(scope="module")
def played() -> dict[str, list[dict[str, Any]]]:
    out = {}
    for name, build in BACKENDS.items():
        vec = build()
        try:
            out[name] = [as_plain(r) for r in play(vec)]
        finally:
            vec.close()
    return out


def test_finished_sub_envs_are_frozen(played: dict[str, list[dict[str, Any]]]) -> None:
    sync = played["sync"]
    # Step 1: sub-env 0 finishes (terminated) with a real transition.
    assert sync[1]["terminated"] == [True, False, False]
    assert sync[1]["infos"][0] == {"t": 1, "elapsed_steps": 1}
    # Step 2 on: it repeats its final observation and flags, with reward 0 and no info.
    assert sync[2]["obs"][0] == sync[1]["obs"][0] == [1.0, 1.0, 0.0]
    assert sync[2]["rewards"][0] == 0.0 and sync[2]["infos"][0] == {}
    # Step 8: every sub-env is frozen where it finished (sub-env 2 truncated by the limit).
    last = sync[8]
    assert [row[0] for row in last["obs"]] == [1.0, 3.0, 4.0]
    assert last["rewards"] == [0.0, 0.0, 0.0] and last["infos"] == [{}, {}, {}]
    assert last["terminated"] == [True, True, False] and last["truncated"] == [False, False, True]
    # reset() unfreezes them all: the second round repeats the first.
    assert sync[9:] == sync[:9]


@pytest.mark.parametrize("backend", sorted(set(BACKENDS) - {"sync"}))
def test_every_backend_agrees_step_by_step(
    played: dict[str, list[dict[str, Any]]], backend: str
) -> None:
    for t, (got, expected) in enumerate(zip(played[backend], played["sync"])):
        assert got == expected, t


def test_native_frozen_mask_and_held_actions() -> None:
    vec = BatchCountdown(3, HORIZONS, autoreset=False)
    vec.reset(seed=0)
    vec.step(np.array([2, 1, 1]))  # sub-env 0 finishes
    assert vec.frozen.tolist() == [True, False, False]
    vec.step(np.array([99, 1, 1]))  # its action is replaced by its last one, 2
    assert vec.last.tolist() == [2, 1, 1] and vec.t.tolist() == [1, 2, 2]
    assert vec.elapsed_steps.tolist() == [1, 2, 2]  # a frozen sub-env's clock stops too
    vec.reset()
    assert not vec.frozen.any()


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_episode_statistics_agree(backend: str) -> None:
    tracker = InMemoryTracker()
    vec = sekai.VecRecordEpisodeStatistics(BACKENDS[backend](), tracker=tracker)
    try:
        play(vec)
    finally:
        vec.close()
    summary = [(e.episode_length, e.episode_return, e.terminated) for e in tracker.episodes]
    # Each round: sub-env 0 (1 step, action 0), 1 (3 steps, 1+2+0), 2 (truncated at 4: 2+0+1+2).
    assert summary == [(1, 0.0, True), (3, 3.0, True), (4, 5.0, False)] * 2


def test_self_resetting_sub_envs_are_not_frozen() -> None:
    vec = sekai.SyncVecEnv([lambda: sekai.AutoReset(StrictCountdown(2))], autoreset=False)
    vec.reset(seed=0)
    steps = [vec.step([1]) for _ in range(5)]
    assert [int(s.observations[0, 0]) for s in steps] == [1, 0, 1, 0, 1]  # never frozen
    assert [float(s.rewards[0]) for s in steps] == [1.0] * 5
