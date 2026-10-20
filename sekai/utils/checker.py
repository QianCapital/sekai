from __future__ import annotations

import asyncio
import copy
from typing import Any

import numpy as np

from sekai.core.env import Env
from sekai.spaces.box import Box
from sekai.spaces.discrete import Discrete
from sekai.spaces.multi_discrete import MultiDiscrete
from sekai.spaces.space import Space
from sekai.utils.masking import flatten_action_mask, get_action_mask, supports_action_mask


def _equal(a: Any, b: Any) -> bool:
    """Structural equality for observations (arrays, dicts, tuples, scalars)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b))
    try:
        return bool(np.array_equal(a, b, equal_nan=True))
    except TypeError:  # equal_nan needs numeric dtypes
        try:
            return bool(np.array_equal(a, b))
        except Exception:
            return bool(a == b)
    except Exception:
        return bool(a == b)


def _loop_running() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def check_env(
    env: Env[Any, Any],
    *,
    n_steps: int = 100,
    warn_on_anomaly: bool = True,
    check_obs_space: bool = True,
    check_action_space: bool = True,
    check_reset_seed: bool = True,
    check_determinism: bool = False,
    check_async: bool = True,
) -> list[str]:
    """Validate that env correctly implements the sekai Env API.

    Returns a list of issue strings, each prefixed "ERROR:" or "WARNING:".
    An empty list means the env passed every check.

    This is a testing utility — call it in your test suite, not in production.
    Unlike gymnasium's PassiveEnvChecker, it adds zero overhead to normal training.

    Exceptions from reset(), step() and action_mask() are reported as ERROR lines,
    not raised. Every reset() is checked, including those that start a new episode
    during the rollout (which must not overwrite the previous episode's terminal
    observation in place).

    Args:
        env: The environment to check. It is reset and stepped, so pass a fresh one.
        n_steps: Number of random-action steps to take.
        warn_on_anomaly: Report WARNING-level issues (ERRORs are always reported).
        check_obs_space: Check every observation is inside observation_space.
        check_action_space: Check action_space.sample() is inside action_space.
        check_reset_seed: Check reset(seed=0) twice gives the same observation.
        check_determinism: Check two rollouts with the same seed and actions match.
        check_async: Check async_reset()/async_step() return valid results. Skipped
            when called from inside a running event loop.
    """
    issues: list[str] = []

    def warn(msg: str) -> None:
        if warn_on_anomaly:
            issues.append(f"WARNING: {msg}")

    def error(msg: str) -> None:
        issues.append(f"ERROR: {msg}")

    # ------------------------------------------------------------------
    # Spaces
    # ------------------------------------------------------------------
    for name in ("observation_space", "action_space"):
        if not hasattr(env, name):
            error(f"{name} is not defined. Set it in __init__.")
        elif not isinstance(getattr(env, name), Space):
            error(f"{name} must be a sekai.Space, got {type(getattr(env, name)).__name__}.")
    if issues:
        return issues

    obs_space = env.observation_space
    act_space = env.action_space
    masked = supports_action_mask(env)

    def sample_action(space: Space[Any] = act_space) -> Any:
        """A random action, respecting the env's action_mask() when it has one.

        The mask is also checked with flatten_action_mask(), the layout that
        ActionMaskObservation and to_gymnasium() give to learners.
        """
        if not masked:
            return space.sample()
        mask = get_action_mask(env)
        action = space.sample(mask=mask)
        if isinstance(space, (Discrete, MultiDiscrete, Box)):
            flatten_action_mask(space, mask)
        return action

    if check_action_space:
        sample_action_ = act_space.sample()
        if not act_space.contains(sample_action_):
            error("action_space.sample() returned a value not in action_space.contains().")
    if check_obs_space and not obs_space.contains(obs_space.sample()):
        error("observation_space.sample() returned a value not in observation_space.contains().")

    # Per-step problems are reported once each, with how many steps (or resets) showed
    # them, instead of one line per step.
    step_problems: dict[str, list[Any]] = {}

    def step_warn(kind: str, msg: str, unit: str = "steps") -> None:
        if kind in step_problems:
            step_problems[kind][1] += 1
        else:
            step_problems[kind] = [msg, 1, unit]

    def obs_problem(obs: Any, where: str) -> tuple[str, str] | None:
        if not check_obs_space:
            return None
        if not obs_space.contains(obs):
            return (
                "obs-space",
                f"Observation from {where} is not within observation_space. obs={obs!r}",
            )
        if (
            isinstance(obs_space, Box)
            and isinstance(obs, np.ndarray)
            and obs.dtype != obs_space.dtype
        ):
            return (
                "obs-dtype",
                f"Observation from {where} has dtype {obs.dtype}, space has {obs_space.dtype}.",
            )
        return None

    def check_obs(obs: Any, where: str) -> None:
        problem = obs_problem(obs, where)
        if problem is not None:
            warn(problem[1])

    def check_reset_obs(obs: Any, where: str) -> None:
        """Like check_obs, but one line for all reset()s, with how many showed the problem."""
        problem = obs_problem(obs, where)
        if problem is not None:
            step_warn(f"reset-{problem[0]}", problem[1], unit="resets")

    def report_step_problems() -> None:
        for msg, count, unit in step_problems.values():
            warn(msg if count == 1 else f"{msg} (seen at {count} {unit}; first shown)")
        step_problems.clear()

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------
    try:
        reset_result = env.reset(seed=42)
    except Exception as e:
        error(f"reset() raised {type(e).__name__}: {e}")
        return issues

    if not hasattr(reset_result, "observation") or not hasattr(reset_result, "info"):
        error("reset() must return a ResetResult with .observation and .info attributes.")
        return issues
    if not isinstance(reset_result.info, dict):
        error(f"reset() info is {type(reset_result.info).__name__}, expected dict.")
    check_reset_obs(reset_result.observation, "reset()")

    if masked:
        try:
            masked_action = sample_action()
        except Exception as e:
            error(
                f"action_space.sample(mask=env.action_mask()) raised {type(e).__name__}: "
                f"{str(e).rstrip('.')}. "
                "action_mask() must return a mask in the format action_space.sample() accepts."
            )
            report_step_problems()
            return issues
        if not act_space.contains(masked_action):
            error("action_space.sample(mask=env.action_mask()) returned an invalid action.")

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------
    anomalous_rewards = 0
    prev_obs: Any = reset_result.observation
    prev_obs_copy: Any = copy.deepcopy(prev_obs)
    aliasing_reported = False
    reset_clobber_reported = False
    reset_failed = False
    for step_idx in range(n_steps):
        try:
            action = sample_action()
        except Exception as e:
            if masked:
                error(
                    "action_space.sample(mask=env.action_mask()) raised "
                    f"{type(e).__name__} at step {step_idx}: {str(e).rstrip('.')}. "
                    "action_mask() must return a valid mask (in the format "
                    "action_space.sample() accepts, with at least one allowed action) in "
                    "every state reached before the episode ends."
                )
            else:
                error(f"action_space.sample() raised {type(e).__name__} at step {step_idx}: {e}")
            break
        try:
            result = env.step(action)
        except Exception as e:
            error(f"step() raised {type(e).__name__} at step {step_idx}: {e}")
            break

        missing = [
            attr
            for attr in ("observation", "reward", "terminated", "truncated", "info")
            if not hasattr(result, attr)
        ]
        if missing:
            error(f"step() must return a StepResult; result has no {', '.join(missing)}.")
            break

        if not aliasing_reported and not _equal(prev_obs, prev_obs_copy):
            warn(
                "step() mutated a previously returned observation in place. Return a copy "
                "(e.g. self._state.copy()) so stored transitions are not corrupted."
            )
            aliasing_reported = True

        problem = obs_problem(result.observation, f"step() at step {step_idx}")
        if problem is not None:
            step_warn(*problem)

        if not isinstance(result.reward, (int, float, np.integer, np.floating)) or isinstance(
            result.reward, (bool, np.bool_)
        ):
            kind = type(result.reward).__name__
            step_warn("reward", f"reward at step {step_idx} is {kind}, expected float.")
        if not isinstance(result.terminated, (bool, np.bool_)):
            kind = type(result.terminated).__name__
            step_warn("terminated", f"terminated at step {step_idx} is {kind}, expected bool.")
        if not isinstance(result.truncated, (bool, np.bool_)):
            kind = type(result.truncated).__name__
            step_warn("truncated", f"truncated at step {step_idx} is {kind}, expected bool.")
        if not isinstance(result.info, dict):
            kind = type(result.info).__name__
            step_warn("info", f"info at step {step_idx} is {kind}, expected dict.")

        try:
            reward = float(result.reward)
        except (TypeError, ValueError):
            reward = 0.0
        if np.isnan(reward):
            step_warn("nan-reward", f"NaN reward at step {step_idx}.")
        elif np.isinf(reward):
            anomalous_rewards += 1

        prev_obs = result.observation
        prev_obs_copy = copy.deepcopy(prev_obs)
        if result.done:
            where = f"reset() after the episode ended at step {step_idx}"
            try:
                next_reset = env.reset()
            except Exception as e:
                error(f"{where} raised {type(e).__name__}: {e}")
                reset_failed = True
                break
            if not hasattr(next_reset, "observation") or not hasattr(next_reset, "info"):
                error(f"{where} did not return a ResetResult with .observation and .info.")
                reset_failed = True
                break
            if not reset_clobber_reported and not _equal(prev_obs, prev_obs_copy):
                warn(
                    "reset() overwrote, in place, the observation step() returned at the end "
                    "of the episode. AutoReset and vector envs pass that terminal observation "
                    'on in info["final_observation"], which would then hold the new episode\'s '
                    "first observation. Return a copy (e.g. self._state.copy())."
                )
                reset_clobber_reported = True
            check_reset_obs(next_reset.observation, where)
            prev_obs = next_reset.observation
            prev_obs_copy = copy.deepcopy(prev_obs)

    if anomalous_rewards:
        warn(f"Encountered {anomalous_rewards} infinite reward(s) during {n_steps} steps.")
    if reset_failed:
        report_step_problems()
        return issues  # every remaining check starts with reset()

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------
    if check_reset_seed:
        try:
            r1 = env.reset(seed=0)
            # A copy: a reset() that writes into the array it returned last time would
            # make the two observations the same object, and so always equal.
            first = copy.deepcopy(r1.observation)
            check_reset_obs(first, "reset(seed=0)")
            r2 = env.reset(seed=0)
            second = r2.observation
        except Exception as e:
            error(f"reset(seed=0) raised {type(e).__name__}: {e}")
        else:
            if not _equal(first, second):
                warn(
                    "reset(seed=0) returned different observations on two calls. "
                    "Call self._set_rng(seed) at the start of reset() and draw all "
                    "randomness from self.rng."
                )
    report_step_problems()

    if check_determinism:

        def rollout() -> list[Any]:
            # A freshly seeded copy of the action space gives both rollouts the same
            # actions, while still respecting state-dependent action masks.
            actions = copy.deepcopy(act_space)
            actions.seed(123)
            # Deep copies, so an env that reuses its observation arrays does not change
            # entries already recorded.
            trace: list[Any] = [copy.deepcopy(env.reset(seed=123).observation)]
            for _ in range(min(n_steps, 50)):
                try:
                    action = sample_action(actions)
                except Exception:  # no valid action: already reported by the step checks
                    break
                r = env.step(action)
                trace.append(copy.deepcopy((r.observation, r.reward, r.terminated, r.truncated)))
                if r.done:
                    break
            return trace

        try:
            differed = not _equal(rollout(), rollout())
        except Exception as e:
            error(f"The determinism check's rollout raised {type(e).__name__}: {e}")
            differed = False
        if differed:
            warn(
                "Two rollouts with the same seed and actions differed. If the env is "
                "intentionally stochastic beyond its seed, register it with "
                "nondeterministic=True."
            )

    # ------------------------------------------------------------------
    # Async API
    # ------------------------------------------------------------------
    if check_async and not _loop_running():

        async def exercise() -> None:
            r = await env.async_reset(seed=7)
            if not hasattr(r, "observation") or not hasattr(r, "info"):
                error("async_reset() must return a ResetResult.")
                return
            check_obs(r.observation, "async_reset()")
            s = await env.async_step(sample_action())
            if not all(hasattr(s, a) for a in ("observation", "reward", "terminated", "truncated")):
                error("async_step() must return a StepResult.")
                return
            check_obs(s.observation, "async_step()")

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(exercise())
        except Exception as e:
            error(f"async_reset()/async_step() raised {type(e).__name__}: {e}")
        finally:
            loop.close()

    return issues
