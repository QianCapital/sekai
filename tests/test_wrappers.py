"""Tests for environment wrappers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import sekai
from sekai.core.env import Env
from sekai.spaces.box import Box
from sekai.spaces.discrete import Discrete
from sekai.utils.typing import ResetResult, StepResult
from sekai.wrappers import (
    ActionWrapper,
    ClipReward,
    ObservationWrapper,
    RecordEpisode,
    RewardWrapper,
    TimeLimit,
    Wrapper,
)


# ---------------------------------------------------------------------------
# Minimal test environment
# ---------------------------------------------------------------------------


class _CounterEnv(Env[np.ndarray, np.intp]):
    """Tiny env that counts steps and ends after `max_steps`."""

    def __init__(self, max_steps: int = 5, reward: float = 1.0) -> None:
        self.observation_space = Box(0.0, 1.0, shape=(1,))
        self.action_space = Discrete(2)
        self._max_steps = max_steps
        self._fixed_reward = reward
        self._step_count = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> ResetResult:
        self._step_count = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action: np.intp) -> StepResult:
        self._step_count += 1
        terminated = self._step_count >= self._max_steps
        obs = np.array([self._step_count / self._max_steps], dtype=np.float32)
        return obs, self._fixed_reward, terminated, False, {"step": self._step_count}


# ---------------------------------------------------------------------------
# Wrapper (transparent delegation)
# ---------------------------------------------------------------------------


class TestWrapper:
    def test_delegation(self):
        env = _CounterEnv()
        wrapper = Wrapper(env)
        obs, info = wrapper.reset()
        assert obs is not None
        obs, reward, terminated, truncated, info = wrapper.step(np.intp(0))
        assert isinstance(reward, float)

    def test_spaces_forwarded(self):
        env = _CounterEnv()
        wrapper = Wrapper(env)
        assert wrapper.observation_space is env.observation_space
        assert wrapper.action_space is env.action_space

    def test_unwrapped(self):
        env = _CounterEnv()
        w1 = Wrapper(env)
        w2 = Wrapper(w1)
        assert w2.unwrapped() is env

    def test_repr(self):
        env = _CounterEnv()
        wrapper = Wrapper(env)
        assert "Wrapper" in repr(wrapper)

    def test_context_manager(self):
        with Wrapper(_CounterEnv()) as w:
            w.reset()


# ---------------------------------------------------------------------------
# ObservationWrapper
# ---------------------------------------------------------------------------


class _DoubleObsWrapper(ObservationWrapper):
    """Doubles each value in the observation."""

    def observation(self, obs):
        return obs * 2.0


class TestObservationWrapper:
    def test_reset_transforms_obs(self):
        env = _DoubleObsWrapper(_CounterEnv())
        obs, _ = env.reset()
        # _CounterEnv resets to [0.0], doubled = [0.0]
        np.testing.assert_array_equal(obs, np.array([0.0], dtype=np.float32))

    def test_step_transforms_obs(self):
        env = _DoubleObsWrapper(_CounterEnv(max_steps=10))
        env.reset()
        obs, _, _, _, _ = env.step(np.intp(0))
        # first step obs is [1/10] = [0.1], doubled = [0.2]
        np.testing.assert_allclose(obs, np.array([0.2], dtype=np.float32), atol=1e-6)


# ---------------------------------------------------------------------------
# RewardWrapper
# ---------------------------------------------------------------------------


class _NegateRewardWrapper(RewardWrapper):
    def reward(self, reward: float) -> float:
        return -reward


class TestRewardWrapper:
    def test_reward_transformed(self):
        env = _NegateRewardWrapper(_CounterEnv(reward=5.0))
        env.reset()
        _, reward, _, _, _ = env.step(np.intp(0))
        assert reward == -5.0


# ---------------------------------------------------------------------------
# ActionWrapper
# ---------------------------------------------------------------------------


class _FlipActionWrapper(ActionWrapper):
    """Inverts a binary action (0 -> 1, 1 -> 0)."""

    def action(self, action):
        return np.intp(1 - int(action))


class TestActionWrapper:
    def test_action_transformed(self):
        inner = _CounterEnv()
        env = _FlipActionWrapper(inner)
        env.reset()
        # We step with action 0, wrapper sends 1 — env doesn't care about action value
        obs, reward, _, _, _ = env.step(np.intp(0))
        assert reward == 1.0


# ---------------------------------------------------------------------------
# TimeLimit
# ---------------------------------------------------------------------------


class TestTimeLimit:
    def test_truncates_at_limit(self):
        env = TimeLimit(_CounterEnv(max_steps=100), max_episode_steps=3)
        env.reset()
        for i in range(3):
            _, _, terminated, truncated, info = env.step(np.intp(0))
            assert info["elapsed_steps"] == i + 1
        assert not terminated
        assert truncated

    def test_natural_termination_before_limit(self):
        env = TimeLimit(_CounterEnv(max_steps=2), max_episode_steps=10)
        env.reset()
        env.step(np.intp(0))
        _, _, terminated, truncated, _ = env.step(np.intp(0))
        assert terminated
        assert not truncated

    def test_reset_resets_step_count(self):
        env = TimeLimit(_CounterEnv(max_steps=100), max_episode_steps=2)
        env.reset()
        env.step(np.intp(0))
        env.step(np.intp(0))  # truncated
        env.reset()
        _, _, _, truncated, info = env.step(np.intp(0))
        assert not truncated
        assert info["elapsed_steps"] == 1

    def test_invalid_limit(self):
        with pytest.raises(ValueError):
            TimeLimit(_CounterEnv(), max_episode_steps=0)

    def test_max_episode_steps_property(self):
        env = TimeLimit(_CounterEnv(), max_episode_steps=7)
        assert env.max_episode_steps == 7


# ---------------------------------------------------------------------------
# RecordEpisode
# ---------------------------------------------------------------------------


class TestRecordEpisode:
    def test_records_after_episode(self):
        env = RecordEpisode(_CounterEnv(max_steps=3, reward=2.0))
        env.reset()
        for _ in range(3):
            _, _, terminated, truncated, info = env.step(np.intp(0))
        assert len(env.episode_records) == 1
        rec = env.episode_records[0]
        assert rec["r"] == pytest.approx(6.0)
        assert rec["l"] == 3

    def test_info_augmented_on_done(self):
        env = RecordEpisode(_CounterEnv(max_steps=2))
        env.reset()
        env.step(np.intp(0))
        _, _, _, _, info = env.step(np.intp(0))
        assert "episode" in info

    def test_deque_size_limits_records(self):
        env = RecordEpisode(_CounterEnv(max_steps=1), deque_size=2)
        for _ in range(5):
            env.reset()
            env.step(np.intp(0))
        assert len(env.episode_records) == 2

    def test_total_steps_accumulates(self):
        env = RecordEpisode(_CounterEnv(max_steps=3))
        for _ in range(2):
            env.reset()
            for _ in range(3):
                env.step(np.intp(0))
        assert env.episode_records[-1]["t"] == 6


# ---------------------------------------------------------------------------
# ClipReward
# ---------------------------------------------------------------------------


class TestClipReward:
    def test_clips_high_reward(self):
        env = ClipReward(_CounterEnv(reward=100.0), min_reward=-1.0, max_reward=1.0)
        env.reset()
        _, reward, _, _, _ = env.step(np.intp(0))
        assert reward == 1.0

    def test_clips_low_reward(self):
        env = ClipReward(_CounterEnv(reward=-50.0), min_reward=-1.0, max_reward=1.0)
        env.reset()
        _, reward, _, _, _ = env.step(np.intp(0))
        assert reward == -1.0

    def test_no_clip_within_range(self):
        env = ClipReward(_CounterEnv(reward=0.5), min_reward=-1.0, max_reward=1.0)
        env.reset()
        _, reward, _, _, _ = env.step(np.intp(0))
        assert reward == pytest.approx(0.5)

    def test_invalid_bounds(self):
        with pytest.raises(ValueError):
            ClipReward(_CounterEnv(), min_reward=1.0, max_reward=-1.0)


# ---------------------------------------------------------------------------
# Wrapper stacking
# ---------------------------------------------------------------------------


class TestWrapperStacking:
    def test_time_limit_and_record(self):
        inner = _CounterEnv(max_steps=100)
        env = RecordEpisode(TimeLimit(inner, max_episode_steps=3))
        env.reset()
        for _ in range(3):
            _, _, _, _, _ = env.step(np.intp(0))
        assert len(env.episode_records) == 1
        assert env.episode_records[0]["l"] == 3
