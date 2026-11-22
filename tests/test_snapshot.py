"""env.snapshot() / env.restore(): the same actions after a restore give the same results."""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pytest

import sekai
from sekai.core.snapshot import Snapshot, register_immutable
from sekai.envs.finance import BatchMarketEnv
from sekai.finance import MarketData

DATA = MarketData(
    100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, (300, 5)), axis=0))
)


def flat(x: Any) -> Any:
    if isinstance(x, dict):
        return [flat(x[k]) for k in sorted(x)]
    if isinstance(x, (list, tuple)):
        return [flat(v) for v in x]
    return np.asarray(x).tolist()


def play(env: Any, actions: list[Any]) -> list[Any]:
    out = []
    for a in actions:
        r = env.step(a)
        out.append(
            flat(
                [
                    r.observation,
                    r.reward,
                    r.terminated,
                    r.truncated,
                    {k: v for k, v in r.info.items() if k != "episode"},
                ]
            )
        )
        if r.done:
            break
    return out


@pytest.mark.parametrize(
    "make",
    [
        lambda: sekai.RecordEpisodeStatistics(sekai.make("CartPole-v1")),
        lambda: sekai.NormalizeObservation(sekai.make("Pendulum-v1")),
        lambda: sekai.make("Trading-v1"),
        lambda: sekai.make("Portfolio-v1", prices=DATA.prices, episode_length=40),
        lambda: sekai.make("Market-v0"),
        lambda: sekai.make("Market-v1"),
        lambda: sekai.make("Market-v1", data=DATA, episode_length=40),
        lambda: sekai.make("MarketMaking-v0"),
        lambda: sekai.make("LimitOrderBook-v0"),
    ],
    ids=[
        "cartpole+stats",
        "pendulum+normalize",
        "trading",
        "portfolio",
        "market-v0",
        "market-v1",
        "market-v1-data",
        "market-making",
        "lob",
    ],
)
def test_restore_replays_exactly(make: Any) -> None:
    env = make()
    env.reset(seed=0)
    env.action_space.seed(0)
    for _ in range(5):
        env.step(env.action_space.sample())
    snap = env.snapshot()
    actions = [env.action_space.sample() for _ in range(15)]
    first = play(env, actions)
    env.restore(snap)
    assert play(env, actions) == first
    env.restore(snap)  # a snapshot can be restored again
    assert play(env, actions) == first
    restored = pickle.loads(pickle.dumps(snap))  # and survives pickling
    env.restore(restored)
    assert play(env, actions) == first


def test_restore_is_in_place_and_shares_data() -> None:
    env = sekai.TimeLimit(sekai.make("Market-v1", data=DATA, episode_length=40), 30)
    inner = env.env
    batch = env.unwrapped.batch_env
    env.reset(seed=0)
    snap = env.snapshot()
    # The wrapper chain, then the batch env behind the single-env adapter.
    assert [k.__name__ for k in snap.kinds] == ["TimeLimit", "UnbatchedEnv", "BatchMarketEnv"]
    assert snap.kind is sekai.TimeLimit
    env.step(env.action_space.sample())
    env.restore(snap)
    assert env.env is inner and env.unwrapped.batch_env is batch  # references stay valid
    assert batch.data is DATA  # shared, not copied
    assert snap.layers[2]["data"] is DATA


def test_vector_envs() -> None:
    for vec in (
        sekai.make_vec("Market-v1", num_envs=3, data=DATA, episode_length=20),
        sekai.make_vec("Portfolio-v1", num_envs=3),
        sekai.make_vec("CartPole-v1", num_envs=3, vectorization_mode="sync"),
    ):
        vec.reset(seed=1)
        vec.batch_action_space.seed(1)
        actions = [vec.batch_action_space.sample() for _ in range(30)]
        snap = vec.snapshot()
        a = [flat([s.rewards, s.dones]) for s in map(vec.step, actions)]
        vec.restore(snap)
        b = [flat([s.rewards, s.dones]) for s in map(vec.step, actions)]
        assert a == b
        vec.close()


def test_batch_env_and_torch() -> None:
    torch = pytest.importorskip("torch")
    env = BatchMarketEnv(4, data=DATA, episode_length=30, backend="torch", allow_short=True)
    env.reset(seed=0)
    snap = env.snapshot()
    acts = [torch.rand(4, 5) * 2 - 1 for _ in range(10)]
    a = [s.rewards.tolist() for s in map(env.step, acts)]
    env.restore(snap)
    assert [s.rewards.tolist() for s in map(env.step, acts)] == a
    env.differentiable = True
    env.reset(seed=0)
    env.step(torch.zeros(4, 5, requires_grad=True) + 0.1)
    with pytest.raises(RuntimeError, match="autograd graph"):
        env.snapshot()
    env.detach_state()
    assert isinstance(env.snapshot(), Snapshot)


def test_errors() -> None:
    env = sekai.make("CartPole-v1")
    env.reset(seed=0)
    with pytest.raises(TypeError, match=r"The snapshot is of"):
        sekai.make("Pendulum-v1").restore(env.snapshot())
    with pytest.raises(TypeError, match="Expected a Snapshot"):
        env.restore("nope")  # type: ignore[arg-type]
    vec = sekai.make_vec("CartPole-v1", num_envs=2, vectorization_mode="proc")
    with pytest.raises(NotImplementedError, match="worker processes"):
        vec.snapshot()
    vec.close()


def test_register_immutable() -> None:
    @register_immutable
    class Table:
        def __init__(self) -> None:
            self.values = np.ones(3)

    env = sekai.make("CartPole-v1")
    env.reset(seed=0)
    table = Table()
    env.unwrapped.table = table
    snap = env.snapshot()
    assert snap.layers[-1]["table"] is table


class _OwnSnapshotEnv(sekai.Env):  # an env written for sekai 0.2, with its own methods
    def __init__(self) -> None:
        self.observation_space = sekai.Box(-1.0, 1.0, (1,))
        self.action_space = sekai.Discrete(2)
        self.x = 0

    def reset(self, *, seed: Any = None, options: Any = None) -> sekai.ResetResult:
        self.x = 0
        return sekai.ResetResult(np.zeros(1, np.float32), {})

    def step(self, action: Any) -> sekai.StepResult:
        self.x += 1
        return sekai.StepResult(np.zeros(1, np.float32), 0.0, False, False, {})

    def snapshot(self) -> dict[str, int]:  # type: ignore[override]
        return {"x": self.x}

    def restore(self, state: dict[str, int]) -> None:  # type: ignore[override]
        self.x = state["x"]


def test_an_envs_own_snapshot_methods_are_found_through_wrappers() -> None:
    """As in 0.2, get_attr and a vector env's call reach the base env's own methods."""
    env = sekai.TimeLimit(sekai.OrderEnforcing(_OwnSnapshotEnv()), max_episode_steps=10)
    env.reset(seed=0)
    env.step(0)
    env.step(1)
    assert env.get_attr("snapshot")() == {"x": 2}
    env.get_attr("restore")({"x": 5})
    assert env.unwrapped.x == 5
    vec = sekai.SyncVecEnv([lambda: sekai.TimeLimit(_OwnSnapshotEnv(), 10)] * 2)
    vec.reset(seed=0)
    assert list(vec.call("snapshot")) == [{"x": 0}, {"x": 0}]
    # sekai's own snapshots of the stack still work as before.
    plain = sekai.make("CartPole-v1")
    plain.reset(seed=0)
    assert isinstance(plain.get_attr("snapshot")(), Snapshot)


class _ViewEnv(sekai.Env):  # hands out a read-only view of its mutable state as the obs
    observation_space = sekai.Box(-np.inf, np.inf, (3,))
    action_space = sekai.Box(-1.0, 1.0, (3,))

    def __init__(self) -> None:
        self._state = np.zeros(3)
        self._view = self._state.view()
        self._view.flags.writeable = False

    def reset(self, *, seed: Any = None, options: Any = None) -> sekai.ResetResult:
        self._state[:] = 0
        return sekai.ResetResult(self._view, {})

    def step(self, action: Any) -> sekai.StepResult:
        self._state += action
        return sekai.StepResult(self._view, float(self._state.sum()), False, False, {})


class _Cache(sekai.Wrapper):  # keeps a handle on the base env, as wrappers may
    def __init__(self, env: sekai.Env) -> None:
        super().__init__(env)
        self.base = env.unwrapped


def test_restore_keeps_every_reference_and_view() -> None:
    env = _Cache(sekai.TimeLimit(_ViewEnv(), 50))
    env.reset(seed=0)
    env.step(np.ones(3))
    snap = env.snapshot()
    env.step(np.ones(3))
    env.restore(snap)
    assert env.base is env.unwrapped  # the wrapper's handle is still the live base env
    base = env.unwrapped
    assert np.shares_memory(base._state, base._view)  # the view still views the state
    np.testing.assert_array_equal(env.step(np.ones(3)).observation, [2.0, 2.0, 2.0])
    vec = sekai.SyncVecEnv([lambda: sekai.make("CartPole-v1")] * 2)
    vec.reset(seed=0)
    first = vec.envs[0]
    snap = vec.snapshot()
    vec.step(vec.batch_action_space.sample())
    vec.restore(snap)
    assert vec.envs[0] is first


def test_snapshots_share_the_scenario_generator() -> None:
    from sekai.finance import GBM, ModelScenarios

    gen = ModelScenarios(GBM(sigma=[0.2, 0.3]), n_steps=60)
    env = BatchMarketEnv(2, scenarios=gen, lookback=5, episode_length=20)
    env.reset(seed=0)
    snap = env.snapshot()
    env.step(np.stack([env.action_space.sample() for _ in range(2)]))
    env.restore(snap)
    assert env.scenarios is gen


def test_unsupported_layers_are_refused_before_anything_is_copied() -> None:
    proc = sekai.ProcVecEnv([lambda: sekai.make("CartPole-v1")] * 2, num_workers=1)
    vec = sekai.VecRecordEpisodeStatistics(proc)
    try:
        vec.reset(seed=0)
        with pytest.raises(NotImplementedError, match="worker processes"):
            vec.snapshot()
        vec.step(vec.batch_action_space.sample())  # the workers are untouched
    finally:
        vec.close()


def test_async_vec_env_snapshots_after_sync_calls() -> None:
    vec = sekai.AsyncVecEnv([lambda: sekai.make("CartPole-v1")] * 2)
    try:
        vec.reset(seed=0)
        vec.step(vec.batch_action_space.sample())  # creates its event loop
        snap = vec.snapshot()
        actions = [vec.batch_action_space.sample() for _ in range(5)]
        first = [vec.step(a).rewards.tolist() for a in actions]
        vec.restore(snap)
        assert [vec.step(a).rewards.tolist() for a in actions] == first
    finally:
        vec.close()


def test_restore_leaves_arrays_callers_hold_alone() -> None:
    """A tree search keeps what it read from each branch: restore rebinds, it doesn't
    write into arrays from before (only into buffers the env itself hands out views of)."""
    env = sekai.make("Portfolio-v0")
    env.reset(seed=0)
    env.action_space.seed(0)
    root = env.snapshot()
    seen = []
    for _ in range(3):
        env.step(env.action_space.sample())
        weights = env.unwrapped.weights
        seen.append((weights, weights.copy()))
        env.restore(root)
    for held, value in seen:
        np.testing.assert_array_equal(held, value)
    masked = sekai.make("CartPole-v1")
    masked.reset(seed=0)
    masked.unwrapped.extra = np.ma.MaskedArray([1.0, 2.0], mask=[False, True])
    snap = masked.snapshot()
    masked.unwrapped.extra = np.ma.MaskedArray([5.0, 6.0], mask=[True, False])
    masked.restore(snap)
    assert masked.unwrapped.extra.mask.tolist() == [False, True]
