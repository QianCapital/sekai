"""Tests for the environment registry."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import sekai
from sekai.core.env import Env
from sekai.registry import EnvSpec, Registry, register, spec
from sekai.spaces.box import Box
from sekai.spaces.discrete import Discrete
from sekai.utils.typing import ResetResult, StepResult


# ---------------------------------------------------------------------------
# Minimal environment for registration tests
# ---------------------------------------------------------------------------


class _DummyEnv(Env[np.ndarray, np.intp]):
    def __init__(self, size: int = 4) -> None:
        self.observation_space = Box(0.0, 1.0, shape=(size,))
        self.action_space = Discrete(2)
        self.size = size

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> ResetResult:
        return np.zeros(self.size, dtype=np.float32), {}

    def step(self, action: np.intp) -> StepResult:
        return np.zeros(self.size, dtype=np.float32), 0.0, False, False, {}


# ---------------------------------------------------------------------------
# Isolated registry (not the global singleton) to avoid side-effects
# ---------------------------------------------------------------------------


@pytest.fixture
def reg() -> Registry:
    return Registry()


class TestRegistryRegister:
    def test_register_and_make(self, reg: Registry):
        reg.register("Dummy-v0", _DummyEnv)
        env = reg.make("Dummy-v0")
        assert isinstance(env, _DummyEnv)

    def test_register_with_kwargs(self, reg: Registry):
        reg.register("Dummy-v1", _DummyEnv, kwargs={"size": 8})
        env = reg.make("Dummy-v1")
        assert env.size == 8  # type: ignore[union-attr]

    def test_register_duplicate_raises(self, reg: Registry):
        reg.register("Dummy-v0", _DummyEnv)
        with pytest.raises(ValueError, match="already registered"):
            reg.register("Dummy-v0", _DummyEnv)

    def test_invalid_id_raises(self, reg: Registry):
        with pytest.raises(ValueError, match="Invalid environment ID"):
            reg.register("nover", _DummyEnv)

    def test_register_by_import_path(self, reg: Registry):
        reg.register("Dummy-v0", "sekai.envs.cartpole:CartPoleEnv")
        env = reg.make("Dummy-v0")
        from sekai.envs.cartpole import CartPoleEnv

        assert isinstance(env, CartPoleEnv)

    def test_contains(self, reg: Registry):
        reg.register("Dummy-v0", _DummyEnv)
        assert "Dummy-v0" in reg
        assert "Dummy-v1" not in reg

    def test_all_specs(self, reg: Registry):
        reg.register("Dummy-v0", _DummyEnv)
        reg.register("Dummy-v1", _DummyEnv)
        specs = reg.all_specs()
        assert len(specs) == 2
        assert all(isinstance(s, EnvSpec) for s in specs)

    def test_repr(self, reg: Registry):
        reg.register("Dummy-v0", _DummyEnv)
        assert "Dummy-v0" in repr(reg)


class TestRegistryMake:
    def test_unknown_id_raises(self, reg: Registry):
        with pytest.raises(KeyError, match="No environment registered"):
            reg.make("Unknown-v0")

    def test_make_applies_time_limit(self, reg: Registry):
        from sekai.wrappers import TimeLimit

        reg.register("Dummy-v0", _DummyEnv, max_episode_steps=10)
        env = reg.make("Dummy-v0")
        assert isinstance(env, TimeLimit)
        assert env.max_episode_steps == 10

    def test_make_override_time_limit(self, reg: Registry):
        from sekai.wrappers import TimeLimit

        reg.register("Dummy-v0", _DummyEnv, max_episode_steps=10)
        env = reg.make("Dummy-v0", max_episode_steps=5)
        assert isinstance(env, TimeLimit)
        assert env.max_episode_steps == 5

    def test_make_disable_time_limit(self, reg: Registry):
        from sekai.wrappers import TimeLimit

        reg.register("Dummy-v0", _DummyEnv, max_episode_steps=10)
        env = reg.make("Dummy-v0", max_episode_steps=0)
        assert not isinstance(env, TimeLimit)

    def test_make_kwargs_override(self, reg: Registry):
        reg.register("Dummy-v0", _DummyEnv, kwargs={"size": 4})
        env = reg.make("Dummy-v0", size=8)
        assert env.size == 8  # type: ignore[union-attr]


class TestRegistrySpec:
    def test_spec_returns_env_spec(self, reg: Registry):
        reg.register("Dummy-v0", _DummyEnv, max_episode_steps=50)
        s = reg.spec("Dummy-v0")
        assert isinstance(s, EnvSpec)
        assert s.id == "Dummy-v0"
        assert s.max_episode_steps == 50

    def test_spec_unknown_raises(self, reg: Registry):
        with pytest.raises(KeyError):
            reg.spec("Unknown-v0")


class TestGlobalRegistry:
    """Test the module-level `sekai.make` / `sekai.spec` interface."""

    def test_make_cartpole(self):
        env = sekai.make("CartPole-v0")
        obs, _ = env.reset()
        assert obs.shape == (4,)

    def test_spec_cartpole(self):
        s = sekai.spec("CartPole-v0")
        assert s.id == "CartPole-v0"

    def test_register_and_make_custom(self):
        # Use a unique ID to avoid collision with other test runs
        env_id = "TestDummy-v99"
        register(env_id, _DummyEnv)
        env = sekai.make(env_id)
        assert isinstance(env, _DummyEnv)


class TestEnvSpecDataclass:
    def test_fields(self):
        s = EnvSpec(id="Test-v0", entry_point=_DummyEnv)
        assert s.id == "Test-v0"
        assert s.entry_point is _DummyEnv
        assert s.kwargs == {}
        assert s.max_episode_steps is None
        assert s.reward_threshold is None
