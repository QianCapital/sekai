# Checking environments

`sekai.check_env(env)` exercises an environment through the whole API and returns a list of
problems. It is a **testing utility**: call it from your test suite (or `sekai check` on the
command line), not in training code. Nothing is wrapped around your env, so it costs nothing
at training time; gymnasium, by contrast, wraps every `make()`d env in a `PassiveEnvChecker`.

```python
import numpy as np
import sekai

print(sekai.check_env(sekai.make("Portfolio-v0"), check_determinism=True))  # []
```

Each issue is a string prefixed `ERROR:` (the env breaks the API) or `WARNING:` (it works,
but something is likely wrong). An empty list means every check passed.

## Options

| Argument | Default | Effect |
|---|---|---|
| `n_steps` | `100` | random-action steps to take (resetting whenever an episode ends); actions respect the env's [action mask](dynamic.md#action-masks), if it has one |
| `warn_on_anomaly` | `True` | report `WARNING`s; `ERROR`s are always reported |
| `check_obs_space` | `True` | check observations against `observation_space` |
| `check_action_space` | `True` | check `action_space.sample()` is in `action_space` |
| `check_reset_seed` | `True` | check `reset(seed=0)` twice gives the same observation |
| `check_determinism` | `False` | check two seeded rollouts with the same actions match |
| `check_async` | `True` | exercise `async_reset()` / `async_step()` (skipped inside a running event loop) |

The env is reset and stepped, so pass a fresh instance.

## What is checked

| Area | Check | Level |
|---|---|---|
| Spaces | `observation_space` and `action_space` exist and are `sekai.Space` instances | ERROR (stops) |
| | `action_space.sample()` is contained in `action_space` | ERROR |
| | `observation_space.sample()` is contained in `observation_space` | ERROR |
| Action masks | if the env or a wrapper defines `action_mask()`: `action_space.sample(mask=env.action_mask())` does not raise after `reset()` | ERROR (stops) |
| | and returns an action contained in `action_space` | ERROR |
| | and does not raise in any state the random-action rollout reaches (e.g. every action masked) | ERROR (ends the rollout) |
| `reset()` | `reset(seed=42)` does not raise | ERROR (stops) |
| | it returns an object with `.observation` and `.info` | ERROR (stops) |
| | `info` is a `dict` | ERROR |
| Observations | every observation (from `reset`, `step` and the async methods) is in `observation_space` | WARNING |
| | for a `Box` space, the observation's dtype equals the space's | WARNING |
| `step()` | `step()` does not raise | ERROR (stops stepping) |
| | it returns an object with all five `StepResult` fields | ERROR (stops stepping) |
| | a previously returned observation was not mutated in place (reported once) | WARNING |
| | `reward` is a real number (not a `bool`), `terminated` / `truncated` are `bool`, `info` is a `dict` | WARNING |
| | no NaN rewards; a count of infinite rewards | WARNING |
| Seeding | `reset(seed=0)` twice returns equal observations | WARNING |
| Determinism | two rollouts from `reset(seed=123)` with the same (up to 50) actions match exactly | WARNING |
| Async | `async_reset(seed=7)` and one `async_step()` return valid results and do not raise | ERROR / WARNING |

Per-step problems are reported once each: the first occurrence, plus how many steps showed
it, e.g. `... has dtype float64, space has float32. (seen at 500 steps; first shown)`.

## Examples

A sloppy environment: it ignores the seed and mutates the array it returned on the previous
step.

```python
class SloppyEnv(sekai.Env):
    def __init__(self):
        self.observation_space = sekai.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.action_space = sekai.Discrete(2)

    def reset(self, *, seed=None, options=None):
        self.state = np.random.uniform(-0.1, 0.1, size=3).astype(np.float32)  # ignores seed
        return sekai.ResetResult(self.state, {})

    def step(self, action):
        self.state += 0.01 if action else -0.01        # mutates the array returned last step
        return sekai.StepResult(self.state, 1.0, False, False, {})


for issue in sekai.check_env(SloppyEnv(), n_steps=50):
    print(issue)
```

```text
WARNING: step() mutated a previously returned observation in place. Return a copy (e.g. self._state.copy()) so stored transitions are not corrupted.
WARNING: reset(seed=0) returned different observations on two calls. Call self._set_rng(seed) at the start of reset() and draw all randomness from self.rng.
```

A broken one: its action space promises an action it cannot handle. `check_env` samples
actions from `action_space`, so seed the space to make the report reproducible.

```python
class BrokenEnv(SloppyEnv):
    PUSH = {0: -0.01, 1: 0.01}

    def __init__(self):
        super().__init__()
        self.action_space = sekai.Discrete(3)          # but only 0 and 1 are handled

    def reset(self, *, seed=None, options=None):
        self._set_rng(seed)
        self.state = self.rng.uniform(-0.1, 0.1, size=3).astype(np.float32)
        return sekai.ResetResult(self.state.copy(), {})

    def step(self, action):
        self.state = self.state + self.PUSH[int(action)]
        return sekai.StepResult(self.state.copy(), 1.0, False, False, {})


env = BrokenEnv()
env.action_space.seed(0)
print(sekai.check_env(env))  # ['ERROR: step() raised KeyError at step 8: 2']
```

`check_env` respects action masks. If the env (or any wrapper around it) has an
`action_mask()` method, every random action, in the rollouts and in the async check, is
sampled with the current mask, so actions the env declares invalid are never tried. Declaring
that action 2 is not available makes the same env pass:

```python
class MaskedEnv(BrokenEnv):
    def action_mask(self):
        return np.array([1, 1, 0], dtype=np.int8)  # action 2 is never valid here


env = MaskedEnv()
env.action_space.seed(0)
print(sekai.check_env(env))  # []
```

A mask in the wrong format (wrong shape, every action masked), after `reset()` or in any state
the rollout reaches, is reported as an `ERROR` naming the step, and ends the rollout.

## In a test suite

```python
import pytest


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Trading-v0", "Portfolio-v0"])
def test_env_api(env_id):
    issues = sekai.check_env(sekai.make(env_id), n_steps=500, check_determinism=True)
    assert not issues, "\n".join(issues)


test_env_api("Portfolio-v0")
```

For envs that are stochastic beyond their seed (a live feed), leave `check_determinism` off
and register them with `nondeterministic=True` to document it.

## From the command line

`sekai check ID` runs the same checks on a registered environment (500 steps by default) and
exits non-zero on errors, or on any issue with `--strict`:

```console
$ sekai check CartPole-v1 --determinism
CartPole-v1: all checks passed.
```

See [Command line](cli.md#check).
