"""Tests for built-in environments."""

from __future__ import annotations

import numpy as np
import pytest

import sekai
from sekai.envs.cartpole import CartPoleEnv
from sekai.envs.gridworld import GridWorldEnv


class TestCartPoleEnv:
    def test_reset_returns_obs_and_info(self):
        env = CartPoleEnv()
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4,)
        assert isinstance(info, dict)

    def test_step_returns_correct_types(self):
        env = CartPoleEnv()
        env.reset()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert obs.shape == (4,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_before_reset_raises(self):
        env = CartPoleEnv()
        with pytest.raises(RuntimeError):
            env.step(np.intp(0))

    def test_invalid_action_raises(self):
        env = CartPoleEnv()
        env.reset()
        with pytest.raises(ValueError):
            env.step(np.intp(5))

    def test_seed_reproducibility(self):
        env1 = CartPoleEnv()
        env2 = CartPoleEnv()
        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

    def test_episode_runs_to_completion(self):
        env = CartPoleEnv()
        env.reset(seed=0)
        for _ in range(300):
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                break

    def test_context_manager(self):
        with CartPoleEnv() as env:
            obs, _ = env.reset()
            assert obs is not None

    def test_observation_space_contains_reset_obs(self):
        env = CartPoleEnv()
        obs, _ = env.reset(seed=1)
        # The obs is within the space bounds after reset (small values)
        assert env.observation_space.shape == obs.shape

    def test_repr(self):
        env = CartPoleEnv()
        assert "CartPoleEnv" in repr(env)


class TestGridWorldEnv:
    def test_reset(self):
        env = GridWorldEnv(size=5)
        obs, info = env.reset()
        assert obs.shape == (2,)
        assert info["agent"] == (0, 0)
        assert info["goal"] == (4, 4)

    def test_step_towards_goal(self):
        env = GridWorldEnv(size=2)
        env.reset()
        # Move right then down to reach (1,1) from (0,0) on a 2x2 grid
        obs, reward, terminated, truncated, info = env.step(np.intp(1))  # right
        assert not terminated
        obs, reward, terminated, truncated, info = env.step(np.intp(2))  # down
        assert terminated
        assert reward == 1.0

    def test_wall_clipping(self):
        env = GridWorldEnv(size=3)
        env.reset()
        # Try to move up from (0,0) — should stay at (0,0)
        obs, _, _, _, info = env.step(np.intp(0))  # up
        assert info["agent"] == (0, 0)

    def test_invalid_action(self):
        env = GridWorldEnv()
        env.reset()
        with pytest.raises(ValueError):
            env.step(np.intp(99))

    def test_observation_in_space(self):
        env = GridWorldEnv()
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    def test_size_validation(self):
        with pytest.raises(ValueError):
            GridWorldEnv(size=1)

    def test_seed_reproducibility(self):
        e1 = GridWorldEnv()
        e2 = GridWorldEnv()
        o1, _ = e1.reset(seed=7)
        o2, _ = e2.reset(seed=7)
        np.testing.assert_array_equal(o1, o2)


class TestMakeIntegration:
    """Integration tests for sekai.make() with built-in envs."""

    def test_make_cartpole(self):
        env = sekai.make("CartPole-v0")
        obs, _ = env.reset()
        assert obs.shape == (4,)

    def test_make_gridworld(self):
        env = sekai.make("GridWorld-v0")
        obs, _ = env.reset()
        assert obs.shape == (2,)

    def test_make_applies_time_limit(self):
        from sekai.wrappers import TimeLimit

        env = sekai.make("CartPole-v0")
        assert isinstance(env, TimeLimit)
        assert env.max_episode_steps == 200

    def test_make_override_time_limit(self):
        from sekai.wrappers import TimeLimit

        env = sekai.make("CartPole-v0", max_episode_steps=50)
        assert isinstance(env, TimeLimit)
        assert env.max_episode_steps == 50

    def test_make_disable_time_limit(self):
        from sekai.wrappers import TimeLimit

        env = sekai.make("CartPole-v0", max_episode_steps=0)
        assert not isinstance(env, TimeLimit)
