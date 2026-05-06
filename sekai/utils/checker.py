from __future__ import annotations

from typing import Any

import numpy as np

from sekai.core.env import Env


def check_env(
    env: Env[Any, Any],
    *,
    n_steps: int = 100,
    warn_on_anomaly: bool = True,
    check_obs_space: bool = True,
    check_action_space: bool = True,
    check_reset_seed: bool = True,
    check_determinism: bool = False,
) -> list[str]:
    """Validate that env correctly implements the sekai Env API.

    Returns a list of issue strings (warnings and errors). Empty list = clean.

    This is a testing utility — call it in your test suite, not in production.
    Unlike gymnasium's PassiveEnvChecker, it adds zero overhead to normal training.
    """
    issues: list[str] = []

    def warn(msg: str) -> None:
        if warn_on_anomaly:
            issues.append(f"WARNING: {msg}")

    def error(msg: str) -> None:
        issues.append(f"ERROR: {msg}")

    # Space checks
    if check_obs_space:
        if not hasattr(env, "observation_space"):
            error("observation_space is not defined.")
        if not hasattr(env, "action_space"):
            error("action_space is not defined.")

    if issues:
        return issues

    obs_space = env.observation_space
    act_space = env.action_space

    if check_action_space:
        sample_action = act_space.sample()
        if not act_space.contains(sample_action):
            error("action_space.sample() returned a value not in action_space.contains().")

    # Reset checks
    try:
        reset_result = env.reset(seed=42)
    except Exception as e:
        error(f"reset() raised an exception: {e}")
        return issues

    if not hasattr(reset_result, "observation") or not hasattr(reset_result, "info"):
        error("reset() must return a ResetResult with .observation and .info attributes.")
        return issues

    obs = reset_result.observation
    if check_obs_space and not obs_space.contains(obs):
        warn(
            f"Initial observation from reset() is not within observation_space. "
            f"obs={obs!r}, space={obs_space!r}"
        )

    # Step checks
    anomalous_rewards: list[float] = []
    for step_idx in range(n_steps):
        action = act_space.sample()
        try:
            result = env.step(action)
        except Exception as e:
            error(f"step() raised an exception at step {step_idx}: {e}")
            break

        if not hasattr(result, "observation"):
            error("step() must return a StepResult with .observation attribute.")
            break
        if not hasattr(result, "reward"):
            error("step() must return a StepResult with .reward attribute.")
            break
        if not hasattr(result, "terminated"):
            error("step() must return a StepResult with .terminated attribute.")
            break
        if not hasattr(result, "truncated"):
            error("step() must return a StepResult with .truncated attribute.")
            break
        if not hasattr(result, "info"):
            error("step() must return a StepResult with .info attribute.")
            break

        if check_obs_space and not result.done and not obs_space.contains(result.observation):
            warn(
                f"Observation at step {step_idx} is not within observation_space. "
                f"obs={result.observation!r}"
            )

        if not isinstance(result.reward, (int, float, np.floating)):
            warn(f"reward at step {step_idx} is {type(result.reward).__name__}, expected float.")

        if not isinstance(result.terminated, (bool, np.bool_)):
            warn(f"terminated at step {step_idx} is {type(result.terminated).__name__}, expected bool.")

        if not isinstance(result.truncated, (bool, np.bool_)):
            warn(f"truncated at step {step_idx} is {type(result.truncated).__name__}, expected bool.")

        if not isinstance(result.info, dict):
            warn(f"info at step {step_idx} is {type(result.info).__name__}, expected dict.")

        reward = float(result.reward)
        if np.isnan(reward):
            warn(f"NaN reward at step {step_idx}.")
        if np.isinf(reward):
            anomalous_rewards.append(reward)

        if result.done:
            env.reset()

    if anomalous_rewards:
        warn(f"Encountered {len(anomalous_rewards)} infinite reward(s) during {n_steps} steps.")

    # Seed reproducibility
    if check_reset_seed:
        r1 = env.reset(seed=0)
        r2 = env.reset(seed=0)
        obs1, obs2 = r1.observation, r2.observation
        try:
            equal = bool(np.array_equal(obs1, obs2))
        except Exception:
            equal = obs1 == obs2
        if not equal:
            warn(
                "reset(seed=0) returned different observations on two calls. "
                "Ensure _set_rng(seed) is called consistently."
            )

    return issues
