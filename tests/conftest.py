from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pytest

import sekai


class CountingEnv(sekai.Env[np.ndarray, int]):
    """Deterministic toy env: obs is [step, noise]; terminates after `horizon` steps."""

    def __init__(self, horizon: int = 5) -> None:
        self.horizon = horizon
        self.observation_space = sekai.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
        self.action_space = sekai.Discrete(3)
        self.t = 0
        self.closed = False

    def _obs(self) -> np.ndarray:
        return np.array([self.t, self.rng.standard_normal()], dtype=np.float32)

    def reset(
        self, *, seed: Any = None, options: dict[str, Any] | None = None
    ) -> sekai.ResetResult:
        self._set_rng(seed)
        self.t = 0
        return sekai.ResetResult(observation=self._obs(), info={"reset": True})

    def step(self, action: int) -> sekai.StepResult:
        self.t += 1
        return sekai.StepResult(
            observation=self._obs(),
            reward=float(action),
            terminated=self.t >= self.horizon,
            truncated=False,
            info={"t": self.t},
        )

    def close(self) -> None:
        self.closed = True


class DictObsEnv(sekai.Env[dict, np.ndarray]):
    def __init__(self) -> None:
        self.observation_space = sekai.Dict(
            {"a": sekai.Box(-1, 1, shape=(3,)), "b": sekai.Discrete(4)}
        )
        self.action_space = sekai.Box(-1, 1, shape=(2,))

    def reset(
        self, *, seed: Any = None, options: dict[str, Any] | None = None
    ) -> sekai.ResetResult:
        self._set_rng(seed)
        self.observation_space.seed(int(self.rng.integers(2**31)))
        return sekai.ResetResult(observation=self.observation_space.sample(), info={})

    def step(self, action: np.ndarray) -> sekai.StepResult:
        return sekai.StepResult(
            self.observation_space.sample(), float(action.sum()), False, False, {}
        )


class AsyncOnlyEnv(sekai.Env[np.ndarray, int]):
    """An env whose async methods do real awaiting and record that they were used."""

    def __init__(self) -> None:
        self.observation_space = sekai.Box(0, 100, shape=(1,), dtype=np.float32)
        self.action_space = sekai.Discrete(2)
        self.t = 0
        self.async_calls = 0

    def reset(
        self, *, seed: Any = None, options: dict[str, Any] | None = None
    ) -> sekai.ResetResult:
        self.t = 0
        return sekai.ResetResult(np.zeros(1, np.float32), {})

    def step(self, action: int) -> sekai.StepResult:
        self.t += 1
        return sekai.StepResult(np.full(1, self.t, np.float32), 1.0, False, False, {})

    async def async_reset(
        self, *, seed: Any = None, options: dict[str, Any] | None = None
    ) -> sekai.ResetResult:
        await asyncio.sleep(0)
        self.async_calls += 1
        return self.reset(seed=seed, options=options)

    async def async_step(self, action: int) -> sekai.StepResult:
        await asyncio.sleep(0)
        self.async_calls += 1
        return self.step(action)


@pytest.fixture
def counting_env() -> CountingEnv:
    return CountingEnv()


def run(coro: Any) -> Any:
    return asyncio.run(coro)
