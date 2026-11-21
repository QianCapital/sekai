"""The BatchEnv contract (autoreset, frozen lanes, partial resets) and its adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

import sekai
from sekai.backend import Array
from sekai.vector.batch import BatchEnv, BatchVecEnv, UnbatchedEnv, map_tree


class Counter(BatchEnv[dict[str, Any], Any]):
    """Lane i counts up by its action; it ends after `horizon + i` steps (terminated if
    the count went negative). Its observation is the count and the step; `noise` is a
    draw from the shared rng at reset."""

    def __init__(self, num_envs: int = 3, horizon: int = 2, **kwargs: Any) -> None:
        obs = sekai.Dict(
            {
                "count": sekai.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
                "t": sekai.Box(0, np.inf, shape=(), dtype=np.float32),
            }
        )
        super().__init__(num_envs, obs, sekai.Box(-1.0, 1.0, shape=(), dtype=np.float32), **kwargs)
        B = self.backend
        self.horizon = horizon
        self.count = B.zeros((num_envs,), like=self._like)
        self.t = np.zeros(num_envs, dtype=np.int64)
        self.noise = np.zeros(num_envs)
        self.resets = np.zeros(num_envs, dtype=np.int64)
        self.limit = np.arange(num_envs) + horizon

    def _reset_lanes(
        self, indices: Array, options: Sequence[dict[str, Any] | None]
    ) -> dict[str, Any]:
        B = self.backend
        idx = B.to_numpy(indices)
        lanes = B.asbool(np.isin(np.arange(self.num_envs), idx), self._like)
        start = np.array([(o or {}).get("start", 0.0) for o in options])
        full = np.zeros(self.num_envs)
        full[idx] = start
        self.count = B.where(lanes, B.asfloat(full, self._like), self.count)
        self.t[idx] = 0
        self.noise[idx] = self.rng.random(len(idx))
        self.resets[idx] += 1
        return {"noise": self.noise[idx].copy()}

    def _observe(self) -> dict[str, Any]:
        B = self.backend
        return {"count": self.count, "t": B.asfloat(self.t.astype(np.float64), self._like)}

    def _step(self, actions: Any) -> tuple[Array, Array, Array, dict[str, Any]]:
        B = self.backend
        a = B.cast(actions, self._like)
        frozen = self.frozen
        self.count = B.where(frozen, self.count, self.count + a)
        self.t = self.t + (~B.to_numpy(frozen)).astype(np.int64)
        terminated = self.count < 0
        truncated = ~terminated & B.asbool(self.t >= self.limit, self._like)
        return (
            self.count * 1.0,
            terminated,
            truncated,
            {"t": B.asfloat(self.t.astype(np.float64), self._like)},
        )

    def _detach(self) -> None:
        self.count = self.backend.detach(self.count)


def test_reset_required_and_partial_reset_order() -> None:
    env = Counter()
    with pytest.raises(sekai.ResetRequired):
        env.step(np.zeros(3))
    with pytest.raises(sekai.ResetRequired):
        env.reset(indices=[0])
    res = env.reset(seed=0, options=[{"start": 1.0}, None, {"start": 2.0}])
    np.testing.assert_array_equal(res.observations["count"], [1.0, 0.0, 2.0])
    assert res.info["noise"].shape == (3,) and not res.dones.any()
    res = env.reset(indices=[2], options={"start": 5.0})
    np.testing.assert_array_equal(res.observations["count"], [1.0, 0.0, 5.0])
    assert list(res.indices) == [2]
    with pytest.raises(ValueError, match="Expected 2 options"):
        env.reset(indices=[0, 1], options=[None])
    with pytest.raises(TypeError, match="int seed"):
        env.reset(seed=[1, 2, 3])  # type: ignore[arg-type]


def test_seeding_is_reproducible() -> None:
    a, b = Counter(), Counter()
    assert np.array_equal(a.reset(seed=3).info["noise"], b.reset(seed=3).info["noise"])
    assert np.array_equal(a.reset().info["noise"], b.reset().info["noise"])  # continues the stream


def test_autoreset_resets_finished_lanes_in_the_same_step() -> None:
    env = Counter(horizon=1)
    env.reset(seed=0)
    res = env.step(np.full(3, 0.5))  # lane 0 truncates (limit 1)
    assert list(res.truncated) == [True, False, False]
    assert list(res.reset_indices) == [0]
    assert res.final_observations is not None and res.final_observations["count"][0] == 0.5
    assert res.observations["count"][0] == 0.0  # new episode
    assert res.info["t"][0] == 1.0  # the info of the finished episode's last step
    assert env.resets.tolist() == [2, 1, 1]
    res = env.step(np.full(3, -2.0))  # everyone goes negative
    assert res.terminated.all() and sorted(res.reset_indices.tolist()) == [0, 1, 2]


def test_without_autoreset_finished_lanes_freeze() -> None:
    env = Counter(horizon=1, autoreset=False)
    env.reset(seed=0)
    first = env.step(np.full(3, 0.5))
    assert list(first.truncated) == [True, False, False] and first.final_observations is None
    assert list(env.frozen) == [True, False, False]
    second = env.step(np.full(3, 0.5))
    assert (
        second.rewards[0] == 0.0 and second.truncated[0] and second.observations["count"][0] == 0.5
    )
    assert second.truncated[1]  # lane 1's limit is 2
    third = env.step(np.full(3, 0.5))
    assert list(third.dones) == [True, True, True]
    env.reset(indices=[0])
    assert list(env.frozen) == [False, True, True]


def test_batch_vec_env_adapter() -> None:
    vec = BatchVecEnv(Counter(horizon=1))
    res = vec.reset(seed=0)
    assert isinstance(res.observations["count"], np.ndarray) and len(res.infos) == 3
    assert "noise" in res.infos[0]
    out = vec.step(np.full(3, 0.5))
    assert out.truncated[0] and out.infos[0]["final_observation"]["count"] == 0.5
    assert "noise" in out.infos[0]["reset_info"] and out.infos[1]["t"] == 1.0
    assert vec.get_attr("horizon") == [1, 1, 1]
    np.testing.assert_array_equal(vec.get_attr("limit"), [1, 2, 3])
    vec.set_attr("horizon", 7)
    assert vec.batch_env.horizon == 7
    # Seeding goes through the adapter's generator, as for every native VecEnv.
    a, b = BatchVecEnv(Counter()), BatchVecEnv(Counter())
    assert a.reset(seed=5).infos[2]["noise"] == b.reset(seed=5).infos[2]["noise"]
    vec.close()
    assert vec.batch_env.closed


def test_batch_vec_env_without_autoreset_and_time_limit() -> None:
    vec = BatchVecEnv(Counter(horizon=5), autoreset=False, max_episode_steps=2)
    vec.reset(seed=0)
    vec.step(np.zeros(3))
    out = vec.step(np.zeros(3))
    assert out.truncated.all() and out.infos[0]["elapsed_steps"] == 2
    again = vec.step(np.zeros(3))
    assert again.truncated.all() and (again.rewards == 0).all() and again.infos[0] == {}


def test_unbatched_env() -> None:
    env = UnbatchedEnv(Counter(num_envs=1, horizon=2))
    with pytest.raises(sekai.ResetRequired):
        env.step(0.0)
    res = env.reset(seed=1)
    assert res.observation["count"] == 0.0 and isinstance(res.info["noise"], float)
    env.step(np.float32(0.5))
    last = env.step(np.float32(0.5))
    assert last.truncated and last.reward == 1.0
    with pytest.raises(sekai.ResetRequired, match="ended"):
        env.step(0.0)
    assert env.horizon == 2 and env.limit == 2  # attributes of the one lane
    with pytest.raises(ValueError, match="one-lane"):
        UnbatchedEnv(Counter(num_envs=2))
    assert sekai.check_env(UnbatchedEnv(Counter(num_envs=1)), check_determinism=True) == []


def test_map_tree() -> None:
    assert map_tree(lambda x: x * 2, {"a": (1, [2]), "b": 3}) == {"a": (2, [4]), "b": 6}


def test_torch_backend_and_gradients() -> None:
    torch = pytest.importorskip("torch")
    env = Counter(horizon=3, backend="torch", autoreset=False)
    with pytest.raises(ValueError, match="device= is for the torch backend"):
        Counter(device="cpu")
    assert env.device == torch.device("cpu")
    env.reset(seed=0)
    a = torch.full((3,), 0.25, requires_grad=True)
    res = env.step(a)
    assert not res.rewards.requires_grad  # differentiable is off: no graph
    env.differentiable = True
    env.reset(seed=0)
    total = env.step(a).rewards.sum() + env.step(a).rewards.sum()
    total.backward()
    assert a.grad is not None and a.grad.tolist() == [3.0, 3.0, 3.0]
    env.detach_state()
    assert not env.count.requires_grad
