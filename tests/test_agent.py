"""Tests for core agent classes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from sekai.core.agent import Agent, RandomAgent
from sekai.core.env import Env
from sekai.spaces.box import Box
from sekai.spaces.discrete import Discrete
from sekai.utils.typing import ResetResult, StepResult


class _SimpleAgent(Agent[np.ndarray, np.intp]):
    """Always returns action 0."""

    def act(self, observation: np.ndarray, *, explore: bool = True) -> np.intp:
        return np.intp(0)


class TestAgent:
    def test_act_returns_action(self):
        agent = _SimpleAgent()
        obs = np.zeros(4, dtype=np.float32)
        assert agent.act(obs) == np.intp(0)

    def test_learn_returns_dict(self):
        agent = _SimpleAgent()
        obs = np.zeros(4, dtype=np.float32)
        result = agent.learn(obs, np.intp(0), 1.0, obs, False, False)
        assert isinstance(result, dict)

    def test_reset_is_callable(self):
        agent = _SimpleAgent()
        agent.reset()  # should not raise

    def test_repr(self):
        assert "_SimpleAgent" in repr(_SimpleAgent())


class TestRandomAgent:
    def test_samples_valid_actions(self):
        space = Discrete(5)
        agent = RandomAgent(space)
        for _ in range(50):
            action = agent.act(None)  # type: ignore[arg-type]
            assert action in space
