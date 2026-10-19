# Quickstart

This page walks through the whole workflow in five minutes: make an environment, run an
episode, add wrappers, vectorise, and write and validate your own environment. Every code
block runs as-is, top to bottom.

## Make an environment

```python
import sekai

env = sekai.make("CartPole-v1")
print(env)                    # <TimeLimit(<CartPole-v1>)>
print(env.observation_space)  # Box([-4.8, -inf, -0.41887903, -inf], [4.8, inf, 0.41887903, inf], (4,), float32)
print(env.action_space)       # Discrete(2)
```

`sekai.make()` looks the ID up in the [registry](../guide/registry.md), builds the env and
applies the registered time limit (`CartPole-v1` is truncated after 500 steps). Run
`sekai list` in a shell, or call `sekai.pprint_registry()`, to see every ID; the
[catalog](../environments/index.md) describes them.

## Run an episode

`reset()` returns a `ResetResult` and `step()` returns a `StepResult`: frozen dataclasses, not
tuples. Both have a `.done` property (always `False` for `ResetResult`), so one variable
drives the loop:

```python
result = env.reset(seed=42)
print(result.observation)  # [ 0.0273956  -0.00611216  0.03585979  0.0197368 ]

while not result.done:
    angle, angular_velocity = result.observation[2:]
    action = int(angle + 0.5 * angular_velocity > 0)  # push towards the lean
    result = env.step(action)

print(result.terminated, result.truncated)  # False True
print(result.info)  # {'elapsed_steps': 500}
```

| `StepResult` field | Meaning |
|---|---|
| `observation` | the next observation |
| `reward` | a `float` |
| `terminated` | the episode reached a terminal state of the MDP |
| `truncated` | the episode was cut short (e.g. by `TimeLimit`) |
| `info` | a `dict` of diagnostics |
| `done` | property: `terminated or truncated` |

!!! tip "Seeding"
    `reset(seed=s)` seeds the environment; later `reset()` calls without a seed continue the
    same random stream, exactly as in gymnasium. Action and observation *spaces* have their own
    generators: call `env.action_space.seed(s)` if you want reproducible `sample()`s.
    See [Seeding](../guide/environments.md#seeding).

## Add wrappers

Wrappers layer behaviour on top of an environment. `RecordEpisodeStatistics` puts a typed
`EpisodeStats` into `info["episode"]` at the end of every episode:

```python
env = sekai.RecordEpisodeStatistics(sekai.make("CartPole-v1"))
env.action_space.seed(0)

result = env.reset(seed=0)
for _ in range(3):
    while not result.done:
        result = env.step(env.action_space.sample())
    stats = result.info["episode"]  # EpisodeStats, not a dict
    print(stats.episode_return, stats.episode_length)  # 18.0 18 | 16.0 16 | 11.0 11
    result = env.reset()

print([type(layer).__name__ for layer in env])  # ['RecordEpisodeStatistics', 'TimeLimit', 'CartPoleEnv']
print(env.tracker.summary()["mean_return"])  # 15.0
```

The [wrappers guide](../guide/wrappers.md) lists all twenty built-in wrappers.

## Vectorise

`make_vec()` runs several copies of an environment and batches their results. Finished
sub-environments are reset automatically within the same `step()`:

```python
vec = sekai.make_vec("CartPole-v1", num_envs=8)
batch = vec.reset(seed=0)
print(batch.observations.shape)  # (8, 4)

batch = vec.step(vec.batch_action_space.sample())
print(batch.rewards.shape, batch.dones.shape)  # (8,) (8,)
print(vec.observation_space)        # Box([-4.8, -inf, -0.41887903, -inf], [4.8, inf, 0.41887903, inf], (4,), float32)
print(vec.batch_observation_space)  # Box([-4.8, -inf, -0.41887903, ..., -inf, -0.41887903, -inf], [4.8, inf, 0.41887903, ..., inf, 0.41887903, inf], (8, 4), float32)
vec.close()
```

Pass `vectorization_mode="proc"` for one process per sub-env, or `"async"` for asyncio. See
[Vector environments](../guide/vector.md).

## Write your own environment

Subclass `sekai.Env`, set the two spaces in `__init__`, and implement `reset()` and
`step()`. Call `self._set_rng(seed)` at the top of `reset()` and draw all randomness from
`self.rng`; that is all seeding needs.

```python
import numpy as np
import sekai


class MeanReversionEnv(sekai.Env[np.ndarray, np.intp]):
    """Trade a mean-reverting spread: observe it, then go short, flat or long."""

    def __init__(self, theta: float = 0.1, sigma: float = 0.2, horizon: int = 100):
        self.theta, self.sigma, self.horizon = theta, sigma, horizon
        self.observation_space = sekai.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32)
        self.action_space = sekai.Discrete(3, start=-1)  # -1 short, 0 flat, +1 long

    def reset(self, *, seed=None, options=None):
        self._set_rng(seed)  # always first
        self.spread, self.position, self.t = 0.0, 0, 0
        return sekai.ResetResult(self._obs(), info={})

    def step(self, action):
        self.position = int(action)
        change = -self.theta * self.spread + self.sigma * self.rng.standard_normal()
        self.spread += change
        self.t += 1
        return sekai.StepResult(
            observation=self._obs(),
            reward=float(self.position * change),
            terminated=False,
            truncated=self.t >= self.horizon,
            info={"spread": self.spread},
        )

    def _obs(self) -> np.ndarray:
        return np.array([self.spread, self.position], dtype=np.float32)
```

Validate it with `check_env` (an empty list means every check passed), then register it so
`make()` and `make_vec()` can build it by name:

```python
print(sekai.check_env(MeanReversionEnv(), check_determinism=True))  # []

sekai.register("quickstart/MeanReversion-v0", MeanReversionEnv, kwargs={"theta": 0.2})
env = sekai.make("quickstart/MeanReversion-v0")

returns = []
for seed in range(100):
    result, total = env.reset(seed=seed), 0.0
    while not result.done:
        result = env.step(-int(np.sign(result.observation[0])))  # bet on reversion
        total += result.reward
    returns.append(total)
print(round(float(np.mean(returns)), 2))  # 5.16
```

## Next steps

- [Migrating from gymnasium](../guide/gymnasium.md), including training with Stable-Baselines3.
- [Environments](../guide/environments.md): the full `Env` API, seeding and metadata.
- [Finance environments](../environments/finance.md): trading, portfolios, market making
  and optimal execution.
