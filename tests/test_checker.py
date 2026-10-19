from __future__ import annotations

from typing import Any

import numpy as np

import sekai
from tests.conftest import AsyncOnlyEnv, CountingEnv


def test_clean_env_passes() -> None:
    assert sekai.check_env(sekai.TimeLimit(CountingEnv(), 10), check_determinism=True) == []


def test_async_env_passes() -> None:
    assert sekai.check_env(AsyncOnlyEnv()) == []


def test_missing_spaces() -> None:
    class NoSpaces(sekai.Env[Any, Any]):
        def reset(self, *, seed: Any = None, options: Any = None) -> sekai.ResetResult:
            return sekai.ResetResult(0, {})

        def step(self, action: Any) -> sekai.StepResult:
            return sekai.StepResult(0, 0.0, False, False, {})

    issues = sekai.check_env(NoSpaces())
    assert any("observation_space is not defined" in i for i in issues)


def test_detects_unseeded_randomness() -> None:
    class Unseeded(CountingEnv):
        def reset(self, *, seed: Any = None, options: Any = None) -> sekai.ResetResult:
            self.t = 0
            return sekai.ResetResult(np.random.default_rng().random(2).astype(np.float32), {})

    issues = sekai.check_env(Unseeded())
    assert any("reset(seed=0)" in i for i in issues)


def test_detects_in_place_mutation() -> None:
    class Aliasing(CountingEnv):
        def __init__(self) -> None:
            super().__init__(horizon=100)
            self.state = np.zeros(2, dtype=np.float32)

        def reset(self, *, seed: Any = None, options: Any = None) -> sekai.ResetResult:
            self.state[:] = 0
            return sekai.ResetResult(self.state, {})

        def step(self, action: int) -> sekai.StepResult:
            self.state[0] += 1
            return sekai.StepResult(self.state, 0.0, False, False, {})

    issues = sekai.check_env(Aliasing())
    assert any("mutated a previously returned observation" in i for i in issues)


def test_detects_bad_types_and_out_of_space_obs() -> None:
    class Sloppy(CountingEnv):
        def step(self, action: int) -> sekai.StepResult:
            return sekai.StepResult(
                np.array([1e9, 0.0]),
                "1",
                0,
                False,
                [],  # type: ignore[arg-type]
            )

        def __init__(self) -> None:
            super().__init__()
            self.observation_space = sekai.Box(-1, 1, shape=(2,), dtype=np.float32)

    issues = sekai.check_env(Sloppy(), n_steps=3)
    text = "\n".join(issues)
    assert "reward" in text and "terminated" in text and "info" in text
    assert "not within observation_space" in text


def test_detects_step_exception() -> None:
    class Crashes(CountingEnv):
        def step(self, action: int) -> sekai.StepResult:
            raise RuntimeError("kaboom")

    issues = sekai.check_env(Crashes())
    assert any(i.startswith("ERROR") and "kaboom" in i for i in issues)


def test_all_masked_state_is_reported_not_raised() -> None:
    class DeadEnd(CountingEnv):
        """Masks every action from t == 3 on."""

        def __init__(self) -> None:
            super().__init__(horizon=10)

        def action_mask(self) -> np.ndarray:
            return np.full(3, 1 if self.t < 3 else 0, dtype=np.int8)

    issues = sekai.check_env(DeadEnd(), check_determinism=True)  # used to raise ValueError
    errors = [i for i in issues if i.startswith("ERROR")]
    assert len(errors) == 1
    assert "at step 3" in errors[0] and "All actions masked" in errors[0]


def test_malformed_mask_mid_rollout_gets_a_neutral_hint() -> None:
    class WrongShape(CountingEnv):
        def __init__(self) -> None:
            super().__init__(horizon=10)

        def action_mask(self) -> np.ndarray:
            return np.ones(3 if self.t < 3 else 2, dtype=np.int8)  # Discrete(3) needs (3,)

    errors = [i for i in sekai.check_env(WrongShape()) if i.startswith("ERROR")]
    assert len(errors) == 1 and "at step 3" in errors[0] and "mask.shape" in errors[0]
    assert "at least one action in every state" not in errors[0]  # not the all-masked hint
    assert "valid mask" in errors[0]


def test_warnings_can_be_silenced() -> None:
    class Unseeded(CountingEnv):
        def reset(self, *, seed: Any = None, options: Any = None) -> sekai.ResetResult:
            return sekai.ResetResult(np.random.default_rng().random(2).astype(np.float32), {})

    assert sekai.check_env(Unseeded(), warn_on_anomaly=False) == []
