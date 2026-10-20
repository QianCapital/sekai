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


class FiveStepEnv(sekai.Env[np.ndarray, int]):
    """Episodes of five steps with a Box(-1, 1, (2,)) observation."""

    def __init__(self) -> None:
        self.observation_space = sekai.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.action_space = sekai.Discrete(2)
        self.t = 0
        self.resets = 0

    def reset(self, *, seed: Any = None, options: Any = None) -> sekai.ResetResult:
        self._set_rng(seed)
        self.t = 0
        self.resets += 1
        return sekai.ResetResult(np.zeros(2, np.float32), {})

    def step(self, action: int) -> sekai.StepResult:
        self.t += 1
        return sekai.StepResult(np.zeros(2, np.float32), 0.0, self.t >= 5, False, {})


def test_resets_between_episodes_are_checked() -> None:
    class BadLaterResets(FiveStepEnv):
        """Only reset(seed=...) returns a valid observation."""

        def reset(self, *, seed: Any = None, options: Any = None) -> sekai.ResetResult:
            result = super().reset(seed=seed, options=options)
            if seed is None:
                return sekai.ResetResult(np.full(3, 99.0, np.float32), {})
            return result

    issues = sekai.check_env(BadLaterResets(), n_steps=100, check_determinism=True)
    assert len(issues) == 1  # used to pass with []
    assert issues[0].startswith("WARNING: Observation from reset() after the episode ended")
    assert "seen at 20 resets" in issues[0]


def test_reset_observation_problems_are_reported_once_for_all_resets() -> None:
    class Float64Resets(FiveStepEnv):
        def reset(self, *, seed: Any = None, options: Any = None) -> sekai.ResetResult:
            super().reset(seed=seed, options=options)
            return sekai.ResetResult(np.zeros(2), {})

    issues = sekai.check_env(Float64Resets(), n_steps=100, check_async=False)
    # The first reset, 20 between episodes and reset(seed=0): one line, not three.
    assert issues == [
        "WARNING: Observation from reset() has dtype float64, space has float32. "
        "(seen at 22 resets; first shown)"
    ]


def test_raising_reset_between_episodes_is_reported_not_raised() -> None:
    class SecondEpisodeBug(FiveStepEnv):
        def reset(self, *, seed: Any = None, options: Any = None) -> sekai.ResetResult:
            if self.t >= 5:
                raise RuntimeError("stale buffer")
            return super().reset(seed=seed, options=options)

    issues = sekai.check_env(SecondEpisodeBug())  # used to raise RuntimeError
    assert issues == [
        "ERROR: reset() after the episode ended at step 4 raised RuntimeError: stale buffer"
    ]


def test_reset_overwriting_the_terminal_observation_is_detected() -> None:
    class ResetClobbers(FiveStepEnv):
        """step() returns fresh arrays; reset() writes into the one step() returned last."""

        def __init__(self) -> None:
            super().__init__()
            self.observation_space = sekai.Box(-10.0, 10.0, shape=(2,), dtype=np.float32)
            self.state = np.zeros(2, np.float32)

        def reset(self, *, seed: Any = None, options: Any = None) -> sekai.ResetResult:
            super().reset(seed=seed, options=options)
            self.state[:] = self.rng.uniform(-0.1, 0.1, 2)
            return sekai.ResetResult(self.state, {})

        def step(self, action: int) -> sekai.StepResult:
            result = super().step(action)
            self.state = self.state + np.float32(0.5)
            return sekai.StepResult(self.state, 0.0, result.terminated, False, {})

    issues = sekai.check_env(ResetClobbers(), check_determinism=True)
    assert len(issues) == 1
    assert "reset() overwrote, in place" in issues[0] and "final_observation" in issues[0]
    # The determinism trace holds copies, so the reused array is not mistaken for
    # nondeterminism (it used to be the only warning).
    assert not any("rollouts" in issue for issue in issues)


def test_seed_check_compares_a_copy_and_is_guarded() -> None:
    class InPlaceUnseeded(FiveStepEnv):
        """Ignores the seed, writing into the same array on every reset()."""

        def __init__(self) -> None:
            super().__init__()
            self.state = np.zeros(2, np.float32)

        def reset(self, *, seed: Any = None, options: Any = None) -> sekai.ResetResult:
            self.t = 0
            self.state[:] = np.random.default_rng().uniform(-1, 1, 2)
            return sekai.ResetResult(self.state, {})

    issues = sekai.check_env(InPlaceUnseeded(), check_async=False)
    assert any("reset(seed=0) returned different observations" in i for i in issues)

    class RejectsSeedZero(FiveStepEnv):
        def reset(self, *, seed: Any = None, options: Any = None) -> sekai.ResetResult:
            if seed == 0:
                raise ValueError("seed 0 is reserved")
            return super().reset(seed=seed, options=options)

    issues = sekai.check_env(RejectsSeedZero(), check_async=False)
    assert issues == ["ERROR: reset(seed=0) raised ValueError: seed 0 is reserved"]


def test_wrong_length_multidiscrete_mask_is_an_error() -> None:
    class ShortMask(FiveStepEnv):
        def __init__(self) -> None:
            super().__init__()
            self.action_space = sekai.MultiDiscrete([6, 6, 3])  # type: ignore[assignment]

        def action_mask(self) -> tuple[np.ndarray, ...]:
            # Two flags for three actions: used to pass, and ActionMaskObservation then
            # emitted 14 flags in a MultiBinary(15) space.
            return (np.ones(6, np.int8), np.ones(6, np.int8), np.array([1, 1], np.int8))

    issues = sekai.check_env(ShortMask())
    assert len(issues) == 1 and issues[0].startswith("ERROR")
    assert "dimension (2,) must have shape (3,)" in issues[0]
