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
    env.reset(seed=0)
    snap = env.snapshot()
    assert len(snap.layers) == 2 and snap.kind is sekai.TimeLimit
    env.step(env.action_space.sample())
    env.restore(snap)
    assert env.env is inner  # references to the layers stay valid
    batch = inner.batch_env
    assert batch.data is DATA  # shared, not copied
    assert snap.layers[1]["batch_env"].data is DATA


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
