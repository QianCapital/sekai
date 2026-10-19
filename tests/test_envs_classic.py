from __future__ import annotations

import numpy as np
import pytest

import sekai
from sekai.envs.classic_control import (
    AcrobotEnv,
    CartPoleEnv,
    ContinuousMountainCarEnv,
    MountainCarEnv,
    PendulumEnv,
)

CLASSIC = {
    "CartPole-v1": CartPoleEnv,
    "MountainCar-v0": MountainCarEnv,
    "MountainCarContinuous-v0": ContinuousMountainCarEnv,
    "Pendulum-v1": PendulumEnv,
    "Acrobot-v1": AcrobotEnv,
}


@pytest.mark.parametrize("env_id", CLASSIC)
def test_check_env(env_id: str) -> None:
    assert sekai.check_env(sekai.make(env_id), n_steps=300, check_determinism=True) == []


@pytest.mark.parametrize("env_id", CLASSIC)
def test_registered_time_limits(env_id: str) -> None:
    env = sekai.make(env_id)
    gym_limits = {
        "CartPole-v1": 500,
        "MountainCar-v0": 200,
        "MountainCarContinuous-v0": 999,
        "Pendulum-v1": 200,
        "Acrobot-v1": 500,
    }
    assert env.max_episode_steps == gym_limits[env_id]


@pytest.mark.parametrize("env_id", CLASSIC)
def test_step_before_reset_raises(env_id: str) -> None:
    env = CLASSIC[env_id]()
    with pytest.raises(sekai.ResetRequired):
        env.step(env.action_space.sample())


def test_reset_bounds_option() -> None:
    env = CartPoleEnv()
    obs = env.reset(seed=0, options={"low": -0.001, "high": 0.001}).observation
    assert np.all(np.abs(obs) <= 0.001)
    with pytest.raises(ValueError):
        env.reset(options={"low": 1.0, "high": 0.0})


def test_cartpole_invalid_action() -> None:
    env = CartPoleEnv()
    env.reset(seed=0)
    with pytest.raises(sekai.InvalidAction):
        env.step(2)


def test_cartpole_warns_after_termination() -> None:
    env = CartPoleEnv()
    env.reset(seed=0)
    while not env.step(1).terminated:
        pass
    with pytest.warns(UserWarning, match="after the episode terminated"):
        assert env.step(1).reward == 0.0


def test_pendulum_reward_range() -> None:
    env = sekai.make("Pendulum-v1")
    env.reset(seed=0)
    for _ in range(200):
        r = env.step(env.action_space.sample())
        assert -16.2736045 <= r.reward <= 0.0


# ---------------------------------------------------------------------------
# Bit-for-bit parity with gymnasium's implementations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_id", CLASSIC)
def test_matches_gymnasium(env_id: str) -> None:
    gym = pytest.importorskip("gymnasium")
    for seed in range(5):
        g = gym.make(env_id).unwrapped
        s = CLASSIC[env_id]()
        g_obs, _ = g.reset(seed=seed)
        np.testing.assert_array_equal(s.reset(seed=seed).observation, g_obs)
        g.action_space.seed(seed)
        for _ in range(200):
            action = g.action_space.sample()
            g_obs, g_rew, g_term, _, _ = g.step(action)
            r = s.step(action)
            np.testing.assert_array_equal(r.observation, g_obs)
            assert r.reward == pytest.approx(float(g_rew), abs=0)
            assert r.terminated == g_term
            if g_term:
                break
