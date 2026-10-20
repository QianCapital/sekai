from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

import sekai
from tests.conftest import AsyncOnlyEnv, CountingEnv, DictObsEnv, run


class BoxEnv(sekai.Env[np.ndarray, np.ndarray]):
    """Obs = last action, padded; bounded Box spaces for action/observation wrappers."""

    def __init__(self) -> None:
        self.observation_space = sekai.Box(-2.0, 2.0, shape=(2,), dtype=np.float32)
        self.action_space = sekai.Box(-2.0, 2.0, shape=(2,), dtype=np.float32)
        self.last_action: np.ndarray | None = None

    def reset(self, *, seed: Any = None, options: Any = None) -> sekai.ResetResult:
        self._set_rng(seed)
        return sekai.ResetResult(np.zeros(2, np.float32), {})

    def step(self, action: np.ndarray) -> sekai.StepResult:
        self.last_action = np.asarray(action)
        return sekai.StepResult(np.asarray(action, np.float32), 1.0, False, False, {})


class TestTimeLimit:
    def test_truncates(self, counting_env: CountingEnv) -> None:
        env = sekai.TimeLimit(CountingEnv(horizon=100), max_episode_steps=3)
        env.reset(seed=0)
        flags = [env.step(0).truncated for _ in range(3)]
        assert flags == [False, False, True]
        assert env.elapsed_steps == 3

    def test_reset_clears(self) -> None:
        env = sekai.TimeLimit(CountingEnv(horizon=100), max_episode_steps=2)
        env.reset(seed=0)
        env.step(0)
        env.reset()
        assert not env.step(0).truncated

    def test_invalid(self, counting_env: CountingEnv) -> None:
        with pytest.raises(ValueError):
            sekai.TimeLimit(counting_env, max_episode_steps=0)


def test_autoreset(counting_env: CountingEnv) -> None:
    env = sekai.AutoReset(CountingEnv(horizon=2))
    env.reset(seed=0)
    env.step(0)
    r = env.step(0)
    assert r.terminated
    assert r.observation[0] == 0  # first obs of the new episode
    assert r.info["final_observation"][0] == 2
    assert r.info["final_info"] == {"t": 2}
    assert r.info["reset_info"] == {"reset": True}


def test_autoreset_async() -> None:
    env = sekai.AutoReset(CountingEnv(horizon=1))
    run(env.async_reset(seed=0))
    r = run(env.async_step(0))
    assert r.observation[0] == 0 and r.info["final_observation"][0] == 1


def test_order_enforcing(counting_env: CountingEnv) -> None:
    env = sekai.OrderEnforcing(counting_env)
    with pytest.raises(sekai.ResetRequired):
        env.step(0)
    with pytest.raises(sekai.ResetRequired):
        run(env.async_step(0))
    env.reset(seed=0)
    assert env.has_reset
    env.step(0)


class TestRecordEpisodeStatistics:
    def test_episode_stats(self) -> None:
        env = sekai.RecordEpisodeStatistics(CountingEnv(horizon=3))
        env.reset(seed=0)
        rewards = [env.step(2) for _ in range(3)]
        stats = rewards[-1].info["episode"]
        assert isinstance(stats, sekai.EpisodeStats)
        assert stats.episode_return == 6.0 and stats.episode_length == 3 and stats.terminated
        assert env.episode_stats is stats
        assert env.tracker.summary()["num_episodes"] == 1  # type: ignore[attr-defined]

    def test_tracker_receives_seed(self) -> None:
        seen: list[Any] = []

        class Tracker(sekai.InMemoryTracker):
            def on_reset(self, env_id: str | None, seed: int | None) -> None:
                seen.append(seed)

        env = sekai.RecordEpisodeStatistics(sekai.TimeLimit(CountingEnv(), 10), tracker=Tracker())
        env.reset(seed=123)
        assert seen == [123]


class TestActionWrappers:
    def test_clip_action(self) -> None:
        base = BoxEnv()
        env = sekai.ClipAction(base)
        env.reset()
        env.step(np.array([5.0, -5.0], np.float32))
        np.testing.assert_array_equal(base.last_action, [2.0, -2.0])

    def test_rescale_action(self) -> None:
        base = BoxEnv()
        env = sekai.RescaleAction(base, -1.0, 1.0)
        env.reset()
        assert env.action_space == sekai.Box(-1.0, 1.0, shape=(2,))
        env.step(np.array([1.0, 0.0], np.float32))
        np.testing.assert_allclose(base.last_action, [2.0, 0.0])
        np.testing.assert_allclose(env.reverse_action(np.array([2.0, 0.0])), [1.0, 0.0])

    def test_rescale_action_needs_bounds(self, counting_env: CountingEnv) -> None:
        with pytest.raises(TypeError):
            sekai.RescaleAction(counting_env, -1, 1)  # type: ignore[arg-type]

    def test_sticky_action_repeats_deterministically(self) -> None:
        def actions_seen(seed: int) -> list[float]:
            base = BoxEnv()
            env = sekai.StickyAction(base, repeat_action_probability=0.5)
            env.reset(seed=seed)
            seen = []
            for i in range(30):
                env.step(np.full(2, float(i), np.float32))
                seen.append(float(base.last_action[0]))  # type: ignore[index]
            return seen

        seen = actions_seen(0)
        assert seen == actions_seen(0)
        assert any(a != i for i, a in enumerate(seen))  # some actions were repeated
        with pytest.raises(ValueError):
            sekai.StickyAction(BoxEnv(), 1.0)

    def test_action_repeat(self) -> None:
        env = sekai.ActionRepeat(CountingEnv(horizon=5), repeat=3)
        env.reset(seed=0)
        r = env.step(1)
        assert r.reward == 3.0 and r.info["action_repeats"] == 3 and r.observation[0] == 3
        r = env.step(1)  # stops early at the terminal step
        assert r.terminated and r.reward == 2.0 and r.info["action_repeats"] == 2
        r_async = sekai.ActionRepeat(CountingEnv(horizon=5), repeat=2)
        run(r_async.async_reset(seed=0))
        assert run(r_async.async_step(1)).reward == 2.0


class TestObservationWrappers:
    def test_flatten(self) -> None:
        env = sekai.FlattenObservation(DictObsEnv())
        obs = env.reset(seed=0).observation
        assert obs.shape == (7,) and env.observation_space.contains(obs)

    def test_normalize_observation(self) -> None:
        env = sekai.NormalizeObservation(sekai.TimeLimit(CountingEnv(horizon=1000), 1000))
        env.reset(seed=0)
        obs = np.stack([env.step(0).observation for _ in range(500)])
        assert abs(obs[-100:, 1].mean()) < 0.5 and 0.5 < obs[-100:, 1].std() < 1.5
        env.update_running_mean = False
        count = env.obs_rms.count
        env.step(0)
        assert env.obs_rms.count == count

    def test_frame_stack(self) -> None:
        env = sekai.FrameStackObservation(CountingEnv(horizon=10), num_stack=3)
        obs = env.reset(seed=0).observation
        assert obs.shape == (3, 2) and (obs[:, 0] == 0).all()
        obs = env.step(0).observation
        assert obs[:, 0].tolist() == [0, 0, 1]
        assert env.observation_space.shape == (3, 2)

    def test_transform_observation(self, counting_env: CountingEnv) -> None:
        env = sekai.TransformObservation(counting_env, lambda o: o * 2)
        env.reset(seed=0)
        assert env.step(0).observation[0] == 2

    def test_rescale_observation(self) -> None:
        env = sekai.RescaleObservation(BoxEnv(), 0.0, 1.0)
        env.reset()
        obs = env.step(np.array([2.0, -2.0], np.float32)).observation
        np.testing.assert_allclose(obs, [1.0, 0.0])
        assert env.observation_space == sekai.Box(0.0, 1.0, shape=(2,))

    def test_dtype_observation(self, counting_env: CountingEnv) -> None:
        env = sekai.DtypeObservation(counting_env, np.float64)
        assert env.reset(seed=0).observation.dtype == np.float64
        assert env.observation_space.dtype == np.float64

    def test_filter_observation(self) -> None:
        env = sekai.FilterObservation(DictObsEnv(), ["b"])
        assert set(env.reset(seed=0).observation) == {"b"}
        with pytest.raises(KeyError):
            sekai.FilterObservation(DictObsEnv(), ["zzz"])

    def test_time_aware_box(self) -> None:
        env = sekai.TimeAwareObservation(sekai.TimeLimit(CountingEnv(horizon=10), 4))
        assert env.max_episode_steps == 4
        assert env.reset(seed=0).observation[-1] == 0.0
        assert env.step(0).observation[-1] == pytest.approx(0.25)
        assert env.observation_space.shape == (3,)

    def test_time_aware_dict_and_requires_limit(self) -> None:
        env = sekai.TimeAwareObservation(DictObsEnv(), max_episode_steps=10)
        assert "time" in env.reset(seed=0).observation
        with pytest.raises(ValueError):
            sekai.TimeAwareObservation(CountingEnv())

    def test_delay_observation(self) -> None:
        env = sekai.DelayObservation(CountingEnv(horizon=10), delay=2)
        assert env.reset(seed=0).observation[0] == 0  # zero padding
        assert env.step(0).observation.tolist() == [0.0, 0.0]
        assert env.step(0).observation[0] == 0  # the reset observation, delayed 2 steps
        assert env.step(0).observation[0] == 1

    def test_delay_observation_reset_padding(self) -> None:
        env = sekai.DelayObservation(CountingEnv(horizon=10), delay=1, padding="reset")
        first = env.reset(seed=0).observation
        np.testing.assert_array_equal(env.step(0).observation, first)


class TestRewardWrappers:
    def test_transform_and_clip(self, counting_env: CountingEnv) -> None:
        env = sekai.ClipReward(sekai.TransformReward(counting_env, lambda r: r * 10), -5, 5)
        env.reset(seed=0)
        assert env.step(2).reward == 5.0
        with pytest.raises(ValueError):
            sekai.ClipReward(counting_env, 1, 1)

    def test_normalize_reward_scales_and_resets(self) -> None:
        env = sekai.NormalizeReward(
            sekai.TransformReward(CountingEnv(horizon=50), lambda r: r * 100)
        )
        env.reset(seed=0)
        rewards = []
        for _ in range(200):
            r = env.step(1)
            rewards.append(r.reward)
            if r.done:
                assert env._returns == 0.0
                env.reset()
        assert max(abs(x) for x in rewards[-50:]) < 5

    def test_reward_wrapper_async(self) -> None:
        env = sekai.TransformReward(AsyncOnlyEnv(), lambda r: r + 1)
        run(env.async_reset())
        assert run(env.async_step(0)).reward == 2.0


def test_validation_raises_real_exceptions(counting_env: CountingEnv) -> None:
    # Not AssertionError (which python -O strips).
    with pytest.raises(TypeError):
        sekai.ClipAction(counting_env)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        sekai.NormalizeObservation(DiscreteObsEnv())  # nothing to normalise
    with pytest.raises(TypeError):
        sekai.ActionMaskObservation(counting_env)  # no action_mask()


class DiscreteObsEnv(sekai.Env[Any, int]):
    def __init__(self) -> None:
        self.observation_space = sekai.Discrete(3)
        self.action_space = sekai.Discrete(2)

    def reset(self, *, seed: Any = None, options: Any = None) -> sekai.ResetResult:
        return sekai.ResetResult(np.intp(0), {})

    def step(self, action: int) -> sekai.StepResult:
        return sekai.StepResult(np.intp(1), 0.0, False, False, {})


class IntObsEnv(sekai.Env[np.ndarray, int]):
    def __init__(self) -> None:
        self.observation_space = sekai.Box(0, 255, shape=(4,), dtype=np.uint8)
        self.action_space = sekai.Discrete(2)

    def reset(self, *, seed: Any = None, options: Any = None) -> sekai.ResetResult:
        return sekai.ResetResult(np.array([0, 64, 128, 255], np.uint8), {})

    def step(self, action: int) -> sekai.StepResult:
        return sekai.StepResult(np.array([0, 64, 128, 255], np.uint8), 0.0, False, False, {})


def test_rescale_integer_observations_to_float() -> None:
    env = sekai.RescaleObservation(IntObsEnv(), -1.0, 1.0)
    obs = env.reset().observation
    assert obs.dtype == np.float32 and env.observation_space.dtype == np.float32
    np.testing.assert_allclose(obs, [-1.0, -0.498, 0.004, 1.0], atol=1e-3)
    assert env.observation_space.contains(obs)


def test_time_aware_integer_observations() -> None:
    env = sekai.TimeAwareObservation(IntObsEnv(), max_episode_steps=10)
    env.reset()
    for _ in range(5):
        obs = env.step(0).observation
    assert obs[-1] == pytest.approx(0.5) and env.observation_space.contains(obs)


def test_sticky_action_copies_reused_buffers() -> None:
    base = BoxEnv()
    env = sekai.StickyAction(base, repeat_action_probability=0.999)
    env.reset(seed=0)
    buffer = np.full(2, 2.0, np.float32)
    env.step(buffer)
    for value in (-2.0, 1.0, -1.0):
        buffer[:] = value  # the caller reuses its action array
        env.step(buffer)
        np.testing.assert_array_equal(base.last_action, [2.0, 2.0])


def test_episode_statistics_around_autoreset() -> None:
    env = sekai.RecordEpisodeStatistics(sekai.AutoReset(CountingEnv(horizon=4)))
    env.reset(seed=0)
    lengths = []
    for _ in range(12):
        r = env.step(1)
        if "episode" in r.info:
            lengths.append(r.info["episode"].episode_length)
    assert lengths == [4, 4, 4]


class StrictBoxEnv(sekai.Env[np.ndarray, np.ndarray]):
    """Rejects actions outside its action space, as the environment guide recommends."""

    def __init__(self, low: float, high: float) -> None:
        self.observation_space = sekai.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = sekai.Box(low, high, shape=(1,), dtype=np.float32)

    def reset(self, *, seed: Any = None, options: Any = None) -> sekai.ResetResult:
        self._set_rng(seed)
        return sekai.ResetResult(np.zeros(1, np.float32), {})

    def step(self, action: np.ndarray) -> sekai.StepResult:
        if not self.action_space.contains(action):
            raise sekai.InvalidAction(action, self.action_space)
        return sekai.StepResult(np.zeros(1, np.float32), 0.0, False, False, {})


@pytest.mark.parametrize(("low", "high"), [(-0.3, 1.1), (-0.3, 0.9), (-2.5, 0.7), (-2.5, 3.3)])
def test_rescale_action_stays_within_bounds(low: float, high: float) -> None:
    # float32 rounding used to put the endpoints one ulp outside the env's bounds.
    env = sekai.RescaleAction(StrictBoxEnv(low, high), -1.0, 1.0)
    env.reset(seed=0)
    for value in (-1.0, 1.0):
        action = np.array([value], np.float32)
        env.step(action)  # used to raise InvalidAction
        inner = env.action(action)
        assert env.env.action_space.contains(inner)
        assert env.action_space.contains(env.reverse_action(inner))
    assert env.action_space.contains(env.reverse_action(np.array([high + 1.0], np.float32)))


def test_sticky_action_reproducible_with_generator_seed() -> None:
    def rollout() -> tuple[list[float], float]:
        base = BoxEnv()
        env = sekai.StickyAction(base, repeat_action_probability=0.5)
        env.reset(seed=np.random.default_rng(0))
        # Deriving the sticky stream must not draw from the env's generator.
        first_draw = float(env.rng.random())
        seen = []
        for i in range(30):
            env.step(np.full(2, float(i), np.float32))
            seen.append(float(base.last_action[0]))  # type: ignore[index]
        return seen, first_draw

    seen, first_draw = rollout()
    assert (seen, first_draw) == rollout()
    assert any(a != i for i, a in enumerate(seen))
    assert first_draw == float(np.random.default_rng(0).random())


class Float32RewardEnv(CountingEnv):
    def step(self, action: int) -> sekai.StepResult:
        r = super().step(action)
        return sekai.StepResult(r.observation, np.float32(0.1), False, False, r.info)


def test_reward_accumulators_use_python_floats() -> None:
    # np.float32 rewards used to keep the running sums in float32 (NumPy 2 promotion).
    env = sekai.RecordEpisodeStatistics(sekai.TimeLimit(Float32RewardEnv(), 1000))
    env.reset(seed=0)
    for _ in range(1000):
        r = env.step(0)
    stats = r.info["episode"]
    assert type(stats.episode_return) is float
    assert stats.episode_return == pytest.approx(1000 * float(np.float32(0.1)), rel=1e-12)

    repeat = sekai.ActionRepeat(Float32RewardEnv(), repeat=3)
    repeat.reset(seed=0)
    assert type(repeat.step(0).reward) is float
    assert type(run(repeat.async_step(0)).reward) is float

    norm = sekai.NormalizeReward(Float32RewardEnv())
    norm.reset(seed=0)
    norm.step(0)
    assert type(norm._returns) is float


# ---------------------------------------------------------------------------
# Regressions found in the 0.2.0 pre-release review
# ---------------------------------------------------------------------------


class MaskedBoxActionEnv(BoxEnv):
    """BoxEnv with a [1, 3] action range on 3 components, the middle one masked."""

    def __init__(self) -> None:
        super().__init__()
        self.action_space = sekai.Box(1.0, 3.0, shape=(3,), dtype=np.float32)
        self.observation_space = sekai.Box(-5.0, 5.0, shape=(3,), dtype=np.float32)

    def reset(self, *, seed: Any = None, options: Any = None) -> sekai.ResetResult:
        self._set_rng(seed)
        return sekai.ResetResult(np.zeros(3, np.float32), {})

    def action_mask(self) -> np.ndarray:
        return np.array([1, 0, 1], np.int8)


class TestRescaleActionMasks:
    def test_masked_components_reach_the_env_as_its_zero(self) -> None:
        env = sekai.RescaleAction(MaskedBoxActionEnv(), -1.0, 1.0)
        env.reset(seed=0)
        env.action_space.seed(0)
        action = env.action_space.sample(mask=sekai.get_action_mask(env))
        assert action[1] == 0.0  # the wrapper's zero, the middle of [1, 3] once rescaled
        env.step(action)
        inner = env.unwrapped
        assert isinstance(inner, MaskedBoxActionEnv) and inner.last_action is not None
        assert inner.last_action[1] == 1.0  # the env's zero(): 0 clipped into [1, 3]
        np.testing.assert_allclose(inner.last_action[[0, 2]], 2.0 + action[[0, 2]], rtol=1e-6)
        run(env.async_step(np.ones(3, np.float32)))  # same rule on the async path
        assert inner.last_action.tolist() == [3.0, 1.0, 3.0]

    def test_envs_without_a_mask_are_unchanged(self) -> None:
        env = sekai.RescaleAction(BoxEnv(), -1.0, 1.0)
        env.reset(seed=0)
        env.step(np.zeros(2, np.float32))
        assert env.unwrapped.last_action.tolist() == [0.0, 0.0]  # type: ignore[attr-defined]

        class NoMaskNow(MaskedBoxActionEnv):
            def action_mask(self) -> Any:
                return None

        env = sekai.RescaleAction(NoMaskNow(), -1.0, 1.0)
        env.reset(seed=0)
        env.step(np.ones(3, np.float32))
        assert env.unwrapped.last_action.tolist() == [3.0, 3.0, 3.0]  # type: ignore[attr-defined]

    def test_market_env_receives_zero_for_unlisted_assets(self) -> None:
        env = sekai.RescaleAction(sekai.make("Market-v0"), -1.0, 1.0)
        env.reset(seed=0)
        mask = sekai.get_action_mask(env)
        env.action_space.seed(0)
        sent = env.action(env.action_space.sample(mask=mask))
        assert np.all(sent[mask == 0] == 0.0)  # used to be 0.5, the middle of [0, 1]


@pytest.mark.parametrize(
    "build",
    [
        lambda env: sekai.TimeLimit(env, 5),
        lambda env: sekai.TimeAwareObservation(env, 5),
        lambda env: sekai.FrameStackObservation(env, 2),
        lambda env: sekai.DelayObservation(env, 1),
        lambda env: sekai.DelayAction(env, 1, noop=0),
        lambda env: sekai.StickyAction(env, 0.5),
    ],
)
def test_stateful_wrappers_warn_outside_autoreset(build: Any) -> None:
    with pytest.warns(UserWarning, match="Make AutoReset the outermost wrapper"):
        build(sekai.AutoReset(CountingEnv(horizon=3)))
    with pytest.warns(UserWarning, match="outermost"):  # found below other wrappers too
        build(sekai.OrderEnforcing(sekai.AutoReset(CountingEnv(horizon=3))))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sekai.AutoReset(build(CountingEnv(horizon=3)))  # the right order: no warning
        sekai.RecordEpisodeStatistics(sekai.AutoReset(CountingEnv(horizon=3)))


def test_time_limit_inside_autoreset_restarts_each_episode() -> None:
    env = sekai.AutoReset(sekai.TimeLimit(CountingEnv(horizon=100), 5))
    env.reset(seed=0)
    truncated = [env.step(0).truncated for _ in range(15)]
    assert truncated == ([False] * 4 + [True]) * 3


class TestCallableArgument:
    def test_func_keyword_matches_gymnasium(self, counting_env: CountingEnv) -> None:
        env = sekai.TransformReward(counting_env, func=lambda r: r * 10)
        obs_env = sekai.TransformObservation(env, func=lambda o: o * 0, new_observation_space=None)
        obs_env.reset(seed=0)
        result = obs_env.step(2)
        assert result.reward == 20.0 and not np.any(result.observation)

    def test_fn_keyword_is_deprecated_but_works(self, counting_env: CountingEnv) -> None:
        with pytest.warns(DeprecationWarning, match=r"TransformReward\(fn=...\) is deprecated"):
            env = sekai.TransformReward(counting_env, fn=lambda r: r + 1)
        with pytest.warns(DeprecationWarning, match="TransformObservation"):
            env2 = sekai.TransformObservation(env, fn=lambda o: o)
        env2.reset(seed=0)
        assert env2.step(1).reward == 2.0

    def test_func_is_required_once(self, counting_env: CountingEnv) -> None:
        with pytest.raises(TypeError, match="missing required argument: 'func'"):
            sekai.TransformReward(counting_env)
        with pytest.raises(TypeError, match="both func and fn"):
            sekai.TransformObservation(counting_env, lambda o: o, fn=lambda o: o)


def test_clip_reward_accepts_none_for_a_side(counting_env: CountingEnv) -> None:
    upper = sekai.ClipReward(counting_env, None, 0.5)  # used to raise TypeError
    upper.reset(seed=0)
    assert upper.step(2).reward == 0.5 and upper.min_reward == -np.inf
    lower = sekai.ClipReward(CountingEnv(), min_reward=1.5, max_reward=None)
    lower.reset(seed=0)
    assert lower.step(0).reward == 1.5 and lower.max_reward == np.inf
