from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import sekai
from tests.conftest import AsyncOnlyEnv, CountingEnv, run


class TestResults:
    def test_step_result_done(self) -> None:
        r = sekai.StepResult(0, 1.0, terminated=False, truncated=True, info={})
        assert r.done
        assert not sekai.StepResult(0, 1.0, False, False, {}).done

    def test_reset_results_are_never_done(self) -> None:
        assert not sekai.ResetResult(observation=0, info={}).done
        assert not sekai.MAResetResult({}, {}).all_done
        assert not sekai.VecResetResult(observations=[0, 0], infos=[{}, {}]).dones.any()

    def test_unified_episode_loop(self, counting_env: CountingEnv) -> None:
        result = counting_env.reset(seed=0)
        steps = 0
        while not result.done:
            result = counting_env.step(0)
            steps += 1
        assert steps == counting_env.horizon

    def test_results_are_frozen(self) -> None:
        r = sekai.ResetResult(observation=1, info={})
        with pytest.raises(AttributeError):
            r.observation = 2  # type: ignore[misc]

    def test_multi_agent_done_flags(self) -> None:
        r = sekai.MAStepResult(
            observations={"a": 0, "b": 0},
            rewards={"a": 0.0, "b": 0.0},
            terminated={"a": True, "b": False},
            truncated={"a": False, "b": False},
            info={"a": {}, "b": {}},
        )
        assert r.any_done and not r.all_done

    @pytest.mark.parametrize(
        ("terminated", "truncated", "all_done", "any_done"),
        [
            # Matched by agent ID, not by position (the dicts used to be zipped).
            ({"a": True, "b": False}, {"b": True, "a": False}, True, True),
            # A missing key counts as not done: truncated={} no longer makes all_done True.
            ({"a": False, "b": False}, {}, False, False),
            ({"a": True, "b": True}, {}, True, True),
            ({"a": False, "b": False}, {"a": True}, False, True),
            ({"a": True}, {"b": True}, True, True),
            ({}, {}, False, False),
        ],
    )
    def test_multi_agent_done_flags_by_agent(
        self,
        terminated: dict[str, bool],
        truncated: dict[str, bool],
        all_done: bool,
        any_done: bool,
    ) -> None:
        r = sekai.MAStepResult({}, {}, terminated, truncated, {})
        assert r.all_done is all_done and r.any_done is any_done


class TestSeeding:
    def test_reset_seed_reproducible(self, counting_env: CountingEnv) -> None:
        a = counting_env.reset(seed=3).observation
        b = counting_env.reset(seed=3).observation
        np.testing.assert_array_equal(a, b)

    def test_unseeded_reset_continues_stream(self) -> None:
        # reset(seed=s) then reset() must be reproducible across runs (gymnasium semantics).
        def episode_starts() -> list[float]:
            env = CountingEnv()
            starts = [float(env.reset(seed=11).observation[1])]
            starts += [float(env.reset().observation[1]) for _ in range(3)]
            return starts

        first, second = episode_starts(), episode_starts()
        assert first == second
        assert len(set(first)) == 4  # and the episodes differ from one another

    def test_set_rng_returns_seed(self, counting_env: CountingEnv) -> None:
        assert counting_env._set_rng(42) == 42
        assert counting_env._rng_seed == 42
        assert counting_env._set_rng(None) == 42  # keeps the existing stream

    def test_generator_seed(self, counting_env: CountingEnv) -> None:
        g = np.random.default_rng(0)
        counting_env.reset(seed=g)
        assert counting_env.rng is g

    def test_np_random_entropy_is_128_bit(self) -> None:
        seeds = {sekai.np_random()[1] for _ in range(5)}
        assert len(seeds) == 5 and max(seeds) > 2**64

    def test_make_seed_sequence(self) -> None:
        assert sekai.make_seed_sequence(0, 3) == sekai.make_seed_sequence(0, 3)
        assert len(set(sekai.make_seed_sequence(0, 10))) == 10


class TestEnvBasics:
    def test_context_manager_closes(self) -> None:
        with CountingEnv() as env:
            env.reset(seed=0)
        assert env.closed

    def test_unwrapped_and_str(self, counting_env: CountingEnv) -> None:
        assert counting_env.unwrapped is counting_env
        assert str(counting_env) == "<CountingEnv>"

    def test_async_defaults(self, counting_env: CountingEnv) -> None:
        r = run(counting_env.async_reset(seed=0))
        assert r.observation[0] == 0
        assert run(counting_env.async_step(1)).reward == 1.0


class TestWrapperBase:
    def test_delegation_and_stack(self, counting_env: CountingEnv) -> None:
        env = sekai.TimeLimit(sekai.OrderEnforcing(counting_env), max_episode_steps=10)
        assert env.unwrapped is counting_env
        assert len(env) == 3
        assert [type(e).__name__ for e in env] == ["TimeLimit", "OrderEnforcing", "CountingEnv"]
        assert env.observation_space is counting_env.observation_space
        assert env.get_attr("horizon") == 5 and env.has_attr("horizon")

    def test_wrapper_shares_rng_with_base(self, counting_env: CountingEnv) -> None:
        env = sekai.TimeLimit(counting_env, max_episode_steps=10)
        env.reset(seed=5)
        assert env.rng is counting_env.rng
        assert env._rng_seed == 5

    def test_rejects_non_env(self) -> None:
        with pytest.raises(TypeError, match=r"sekai\.Env"):
            sekai.TimeLimit(object(), max_episode_steps=3)  # type: ignore[arg-type]

    def test_spec_setter_reaches_base(self, counting_env: CountingEnv) -> None:
        env = sekai.TimeLimit(counting_env, max_episode_steps=3)
        env.spec = "spec"
        assert counting_env.spec == "spec"

    def test_close_propagates(self, counting_env: CountingEnv) -> None:
        sekai.TimeLimit(counting_env, max_episode_steps=3).close()
        assert counting_env.closed


class TestAsyncFallback:
    """Wrappers that only override step() must not be bypassed by async_step()."""

    def test_custom_sync_only_wrapper_runs_on_async_path(self, counting_env: CountingEnv) -> None:
        class Doubler(sekai.Wrapper):
            def step(self, action: Any) -> sekai.StepResult:
                r = self.env.step(action)
                return sekai.StepResult(
                    r.observation, r.reward * 2, r.terminated, r.truncated, r.info
                )

        env = Doubler(counting_env)
        run(env.async_reset(seed=0))
        assert run(env.async_step(2)).reward == 4.0

    def test_subclass_of_concrete_wrapper(self, counting_env: CountingEnv) -> None:
        class Negate(sekai.TimeLimit):
            def step(self, action: Any) -> sekai.StepResult:
                r = super().step(action)
                return sekai.StepResult(r.observation, -r.reward, r.terminated, r.truncated, r.info)

        env = Negate(counting_env, max_episode_steps=2)
        run(env.async_reset(seed=0))
        assert run(env.async_step(1)).reward == -1.0
        assert run(env.async_step(1)).truncated

    def test_builtin_wrappers_keep_true_async(self) -> None:
        base = AsyncOnlyEnv()
        env = sekai.RecordEpisodeStatistics(sekai.TimeLimit(base, max_episode_steps=2))
        run(env.async_reset())
        run(env.async_step(0))
        last = run(env.async_step(0))
        assert last.truncated and "episode" in last.info
        assert base.async_calls == 3  # every call went through the env's own async methods

    def test_full_override_keeps_async_override(self) -> None:
        class Both(sekai.Wrapper):
            def step(self, action: Any) -> sekai.StepResult:
                raise AssertionError("sync path must not be used")

            async def async_step(self, action: Any) -> sekai.StepResult:
                return await self.env.async_step(action)

        base = AsyncOnlyEnv()
        env = Both(base)
        run(env.async_reset())
        run(env.async_step(1))
        assert base.async_calls == 2


class TestAsyncFallbackMRO:
    def test_mixin_step_is_honoured(self, counting_env: CountingEnv) -> None:
        calls: list[str] = []

        class LogMixin:
            def step(self, action: Any) -> sekai.StepResult:
                calls.append("mixin")
                return super().step(action)  # type: ignore[misc]

        class Logged(LogMixin, sekai.Wrapper):
            pass

        env = Logged(counting_env)
        run(env.async_reset(seed=0))
        run(env.async_step(0))
        assert calls == ["mixin"]

    def test_no_double_application(self, counting_env: CountingEnv) -> None:
        count = 0

        class StepOnly(sekai.Wrapper):
            def step(self, action: Any) -> sekai.StepResult:
                nonlocal count
                count += 1
                return self.env.step(action)

        class Both(StepOnly):
            def step(self, action: Any) -> sekai.StepResult:
                nonlocal count
                count += 1
                return super().step(action)

            async def async_step(self, action: Any) -> sekai.StepResult:
                nonlocal count
                count += 1
                return await super().async_step(action)

        env = Both(counting_env)
        run(env.async_reset(seed=0))
        run(env.async_step(0))
        assert count == 2  # Both.async_step + StepOnly.step, not Both.step again


@pytest.mark.parametrize(
    "error",
    [
        sekai.EnvNotFound("Foo-v0", ["Fo-v0"]),
        sekai.EnvAlreadyRegistered("Foo-v0"),
        sekai.InvalidAction(5, sekai.Discrete(2)),
        sekai.ResetRequired(),
        sekai.VecEnvWorkerError(1, "ValueError", "boom", "Traceback ..."),
        sekai.VecEnvClosed(),
        sekai.SekaiError("plain"),
    ],
    ids=lambda e: type(e).__name__,
)
def test_errors_pickle(error: sekai.SekaiError) -> None:
    import pickle

    clone = pickle.loads(pickle.dumps(error))
    assert type(clone) is type(error) and str(clone) == str(error)


def test_wrapper_set_attr_targets_the_layer_that_has_it(counting_env: CountingEnv) -> None:
    env = sekai.TimeLimit(sekai.OrderEnforcing(counting_env), max_episode_steps=5)
    env.set_attr("horizon", 9)
    assert counting_env.horizon == 9 and "horizon" not in vars(env)
    env.set_attr("max_episode_steps", 7)
    assert env.max_episode_steps == 7
    env.set_attr("brand_new", 1)  # nothing has it: set on the outermost wrapper
    assert vars(env)["brand_new"] == 1
