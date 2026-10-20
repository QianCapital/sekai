from __future__ import annotations

import logging
import pickle
from typing import Any

import numpy as np
import pytest

import sekai
from sekai.utils.pickle import CloudpickleWrapper
from tests.conftest import run


class Duel(sekai.MultiAgentEnv[np.ndarray, int, str]):
    def __init__(self) -> None:
        self.possible_agents = ["a", "b"]
        self.agents = list(self.possible_agents)
        self.observation_spaces = {a: sekai.Box(0, 1, shape=(1,)) for a in self.possible_agents}
        self.action_spaces = {a: sekai.Discrete(2) for a in self.possible_agents}

    def reset(self, *, seed: Any = None, options: Any = None) -> sekai.MAResetResult:
        return sekai.MAResetResult(
            {a: np.zeros(1, np.float32) for a in self.agents}, {a: {} for a in self.agents}
        )

    def step(self, actions: dict[str, int]) -> sekai.MAStepResult:
        rewards = {a: float(actions[a] == 1) for a in self.agents}
        flags = {a: False for a in self.agents}
        return sekai.MAStepResult(
            {a: np.zeros(1, np.float32) for a in self.agents},
            rewards,
            flags,
            dict(flags),
            {a: {} for a in self.agents},
        )


def test_multi_agent() -> None:
    env = Duel()
    assert env.num_agents == 2 and env.max_num_agents == 2
    assert env.action_space("a") == sekai.Discrete(2)
    env.reset(seed=0)
    r = env.step({"a": 1, "b": 0})
    assert r.rewards == {"a": 1.0, "b": 0.0} and not r.any_done
    assert run(env.async_step({"a": 0, "b": 1})).rewards["b"] == 1.0
    assert "agents" in str(env)


class TestRunningMeanStd:
    def test_matches_numpy(self) -> None:
        rng = np.random.default_rng(0)
        data = rng.normal(3.0, 2.0, size=(1000, 3))
        rms = sekai.RunningMeanStd(shape=(3,))
        for chunk in np.array_split(data, 17):
            rms.update(chunk)
        np.testing.assert_allclose(rms.mean, data.mean(0), rtol=1e-3)
        np.testing.assert_allclose(rms.var, data.var(0), rtol=1e-3)

    def test_state_dict_round_trip(self) -> None:
        rms = sekai.RunningMeanStd()
        rms.update(np.array([1.0, 2.0, 3.0]))
        other = sekai.RunningMeanStd()
        other.load_state_dict(rms.state_dict())
        assert other.count == rms.count and other.mean == rms.mean

    def test_prior_does_not_bias_data_far_from_zero(self) -> None:
        # The prior mean of 0 used to add about mean**2 * 1e-4 / n to the variance:
        # 10.95 instead of 0.954 here.
        data = 1e4 + np.random.default_rng(0).standard_normal(1000)
        rms = sekai.RunningMeanStd()
        for x in data:
            rms.update(x[None])
        np.testing.assert_allclose(rms.var, data.var(), rtol=1e-6)
        np.testing.assert_allclose(rms.mean, data.mean(), rtol=1e-10)
        assert rms.count == pytest.approx(1000 + 1e-4)  # the pseudo-count is still counted

    def test_first_batch_sets_the_mean_and_keeps_a_variance_floor(self) -> None:
        rms = sekai.RunningMeanStd()
        rms.update(np.array([5.0]))
        assert rms.mean == 5.0
        # The prior's unit variance keeps its pseudo-count weight: one sample does not
        # give a zero variance (NormalizeReward divides by its square root).
        assert rms.var == pytest.approx(1e-4 / (1 + 1e-4))

    def test_untouched_state_dict_still_starts_from_the_first_batch(self) -> None:
        fresh = sekai.RunningMeanStd(shape=(2,)).state_dict()
        rms = sekai.RunningMeanStd(shape=(2,))
        rms.load_state_dict(fresh)
        rms.update(np.array([[100.0, -100.0], [102.0, -98.0]]))
        np.testing.assert_allclose(rms.mean, [101.0, -99.0])
        np.testing.assert_allclose(rms.var, [1.0, 1.0], rtol=1e-3)

    @pytest.mark.parametrize("cls", [sekai.RunningMeanStd, sekai.ExponentialMovingStats])
    def test_empty_batches_are_ignored(self, cls: Any) -> None:
        stats = cls(shape=(2,))
        stats.update(np.ones((3, 2)))
        before = stats.state_dict()
        stats.update(np.empty((0, 2)))  # used to make RunningMeanStd NaN for good
        stats.update_from_moments(np.zeros(2), np.zeros(2), 0)
        after = stats.state_dict()
        for key in before:
            np.testing.assert_array_equal(after[key], before[key])
        assert np.all(np.isfinite(stats.normalize(np.ones(2))))


def test_backend_mismatch_is_deprecated() -> None:
    import sekai.error

    assert "BackendMismatch" not in sekai.__all__
    with pytest.warns(DeprecationWarning, match="removed in sekai 0.3"):
        cls = sekai.BackendMismatch
    with pytest.warns(DeprecationWarning, match="sekai.error.BackendMismatch is deprecated"):
        assert sekai.error.BackendMismatch is cls
    err = cls("numpy", "torch")
    assert isinstance(err, sekai.SekaiError) and "expected 'numpy'" in str(err)
    with pytest.warns(DeprecationWarning):
        assert type(pickle.loads(pickle.dumps(err))) is cls
    with pytest.raises(AttributeError):
        sekai.NoSuchName


class TestStats:
    def test_in_memory_tracker_summary(self) -> None:
        t = sekai.InMemoryTracker(max_episodes=2)
        assert t.summary() == {} and t.mean_return() == 0.0
        for ret in (1.0, 2.0, 3.0):
            t.on_episode_end(sekai.EpisodeStats(ret, 1, 0.0, 0.0, 0.0, True, False))
        assert t.summary()["num_episodes"] == 2 and t.mean_return() == 2.5

    def test_logging_tracker(self, caplog: pytest.LogCaptureFixture) -> None:
        env = sekai.RecordEpisodeStatistics(
            sekai.make("CartPole-v1"), tracker=sekai.LoggingTracker()
        )
        with caplog.at_level(logging.INFO, logger="sekai.stats"):
            env.reset(seed=0)
            while not env.step(0).done:
                pass
        assert any(r.message == "episode_end" for r in caplog.records)

    def test_trackers_satisfy_protocol(self) -> None:
        assert isinstance(sekai.InMemoryTracker(), sekai.StatsTracker)
        assert isinstance(sekai.LoggingTracker(), sekai.StatsTracker)


def test_cloudpickle_wrapper_handles_lambdas() -> None:
    fn = CloudpickleWrapper(lambda: 41 + 1)
    assert pickle.loads(pickle.dumps(fn))() == 42


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Portfolio-v0", "MarketMaking-v0"])
def test_pickled_env_continues_identically(env_id: str) -> None:
    env = sekai.make(env_id)
    env.reset(seed=0)
    clone = pickle.loads(pickle.dumps(env))
    for _ in range(5):
        action = env.action_space.sample()
        a, b = env.step(action), clone.step(action)
        assert sekai.utils.checker._equal(a.observation, b.observation)
        assert a.reward == b.reward


def test_public_api_is_importable() -> None:
    for name in sekai.__all__:
        assert hasattr(sekai, name), name


def test_version() -> None:
    from importlib.metadata import version

    assert sekai.__version__ == version("sekai")


def test_multi_agent_seeding_api() -> None:
    env = Duel()
    env._set_rng(3)
    a = env.rng.random()
    env._set_rng(3)
    assert env.rng.random() == a
    env.rng = np.random.default_rng(0)  # 0.1.x-style subclasses assign their own generator
    assert env._rng_seed == -1
