# Wrappers

A wrapper is an `Env` that holds another `Env` and changes part of its behaviour: the
observations, the rewards, the actions, or the episode logic. Everything it does not override
is delegated to the wrapped env, including the [async methods](async.md) and the generator
(`wrapper.rng is wrapper.unwrapped.rng`).

```python
import numpy as np
import sekai

env = sekai.make("CartPole-v1")            # TimeLimit is applied by make()
env = sekai.RecordEpisodeStatistics(env)   # typed info["episode"]
env = sekai.NormalizeObservation(env)      # running mean/std normalisation
print(env)  # <NormalizeObservation(<RecordEpisodeStatistics(<TimeLimit(<CartPole-v1>)>)>)>
```

Wrappers apply from the inside out: the last one you add sees the env's output last and its
own input first. Order matters, e.g. put `RecordEpisodeStatistics` *inside* reward
normalisation so it records the raw returns.

## Inspecting a stack

Iterating over a wrapper walks from the outermost layer to the base env, and `len()` counts the
layers (base env included):

```python
print(len(env))                            # 4
print([type(layer).__name__ for layer in env])  # ['NormalizeObservation', 'RecordEpisodeStatistics', 'TimeLimit', 'CartPoleEnv']

time_limit = next(layer for layer in env if isinstance(layer, sekai.TimeLimit))
print(time_limit.max_episode_steps)        # 500
print(env.unwrapped)                       # <CartPole-v1>
print(env.get_attr("max_episode_steps"))   # 500
```

Each layer reports its own spaces: `NormalizeObservation` declares a `float64` observation
space, while `env.unwrapped.observation_space` is the base env's `float32` one.

## Built-in wrappers

All of these are exported from `sekai` and from `sekai.wrappers`.

**Episode control**

| Wrapper | What it does |
|---|---|
| `TimeLimit(env, max_episode_steps)` | sets `truncated=True` after `max_episode_steps` steps; adds `info["elapsed_steps"]`. `make()` applies it from the registry. |
| `AutoReset(env)` | resets as soon as an episode ends; the returned observation starts the new episode, the terminal one is in `info["final_observation"]` (also `final_info`, `reset_info`) |
| `OrderEnforcing(env)` | raises `ResetRequired` if `step()` is called before the first `reset()` |
| `RecordEpisodeStatistics(env, tracker=None, deque_size=100)` | puts an `EpisodeStats` in `info["episode"]` when an episode ends and reports it to a [tracker](statistics.md) |

**Observations**

| Wrapper | What it does |
|---|---|
| `FlattenObservation(env)` | flattens any [flattenable space](spaces.md#flattening) to a 1-D `float32` `Box` |
| `NormalizeObservation(env, epsilon=1e-8, *, stats="running", halflife=1000.0, clip=None)` | normalises every `Box` part of the observation (a `MaskedBox` over its occupied slots only; other parts pass through) with a running mean/variance, or with `stats="ewm"` an exponentially weighted one; optional `clip`; output is `float64`; `obs_rms`, `state_dict()` |
| `FrameStackObservation(env, num_stack, *, padding="reset")` | stacks the last `num_stack` observations on a new leading axis, leaf by leaf for any batchable space (`Dict`, `Tuple`, `MaskedBox`, `Discrete`, ...); after `reset()` it is filled with the first observation, or with `padding="zeros"` the space's [zero sample](spaces.md#zero-samples) |
| `TransformObservation(env, fn, new_observation_space=None)` | applies `fn` to every observation |
| `RescaleObservation(env, min_obs, max_obs)` | maps a bounded `Box` affinely onto `[min_obs, max_obs]`; integer observations (e.g. `uint8` pixels) become `float32` |
| `DtypeObservation(env, dtype)` | casts a `Box` observation, e.g. a `float64` simulator feeding a `float32` policy |
| `FilterObservation(env, keys)` | keeps only `keys` of a `Dict` observation |
| `TimeAwareObservation(env, max_episode_steps=None, *, normalize_time=True, dict_key="time")` | appends elapsed time (`t / max_episode_steps`) to a 1-D `Box`, or adds it under `dict_key` of a `Dict`, so time limits become observable; an integer `Box` becomes `float32` |
| `DelayObservation(env, delay, *, padding="zeros")` | returns the observation from `delay` steps ago (feed or decision latency); pads with zeros or (`padding="reset"`) the first observation |
| `PadSequenceObservation(env, max_length=None, *, truncate="tail")` | turns every `Sequence` of `Box` elements in the observation into a fixed-size `MaskedBox` of `max_length` slots; longer sequences keep their last (`"tail"`) or first (`"head"`) elements, or raise (`"error"`) |
| `ActionMaskObservation(env, *, key="action_mask")` | adds the env's current [action mask](dynamic.md#action-masks), flattened to `int8`, to the observation: a `Dict` gains a `key` entry, anything else becomes `{"observation": obs, key: mask}` |

**Actions**

| Wrapper | What it does |
|---|---|
| `ClipAction(env)` | clips `Box` actions to the bounds |
| `RescaleAction(env, min_action, max_action)` | lets the policy act in `[min_action, max_action]` (e.g. `[-1, 1]`), rescaled to the env's bounded `Box` and clipped to its bounds |
| `StickyAction(env, repeat_action_probability)` | with that probability, repeats the previous action (a stored copy) instead: execution lag (Machado et al., 2018); its own RNG is derived from the `reset()` seed (an int, or a `Generator`'s `SeedSequence`), without drawing from the env's |
| `ActionRepeat(env, repeat)` | repeats each action `repeat` times, summing rewards; stops early at episode end; `info["action_repeats"]` |
| `DelayAction(env, delay, *, noop=None)` | executes each action `delay` steps after it is chosen (order latency), and `noop` until then; `noop` defaults to zeros for a `Box` action space and is required otherwise; `pending_actions` |

**Rewards**

| Wrapper | What it does |
|---|---|
| `NormalizeReward(env, gamma=0.99, epsilon=1e-8)` | divides rewards by the running std of the discounted return (`return_rms`); the mean is not subtracted |
| `TransformReward(env, fn)` | applies `fn` to every reward |
| `ClipReward(env, min_reward=-inf, max_reward=inf)` | clips every reward |

**Rendering** (in `sekai.rendering`)

| Wrapper | What it does |
|---|---|
| `RecordFrames(env, renderer, *, enabled=True, max_frames=None)` | captures a frame after every `reset()` and `step()`; `save("run.gif")` ([Rendering](rendering.md#recording-frames)) |

The [vector wrappers](vector.md#vector-wrappers) are listed with the vector environments.
[Variable-size observations](dynamic.md) shows `PadSequenceObservation`,
`ActionMaskObservation` and normalisation of padded sets at work, and
[Time and history](time.md) the history and latency wrappers.

### Examples

Time features and stacking for a time-series policy:

```python
env = sekai.TimeAwareObservation(sekai.make("CartPole-v1"))  # reads 500 from the TimeLimit
print(env.observation_space)  # Box([-4.8, -inf, -0.41887903, -inf, 0.0], [4.8, inf, 0.41887903, inf, 1.0], (5,), float32)
print(env.reset(seed=0).observation[-1], env.step(0).observation[-1])  # 0.0 0.002

stacked = sekai.FrameStackObservation(sekai.make("Pendulum-v1"), num_stack=4)
print(stacked.observation_space)  # Box([-1.0, -1.0, -8.0, ..., -1.0, -1.0, -8.0], [1.0, 1.0, 8.0, ..., 1.0, 1.0, 8.0], (4, 3), float32)
```

Selecting and flattening a `Dict` observation:

```python
portfolio = sekai.make("Portfolio-v0")
print(sekai.FilterObservation(portfolio, ["weights"]).observation_space)  # Dict({weights: Box(0.0, 1.0, (6,), float32)})
print(sekai.FlattenObservation(portfolio).observation_space)  # Box([-inf, -inf, -inf, ..., 0.0, 0.0, 0.0], [inf, inf, inf, ..., 1.0, 1.0, 1.0], (156,), float32)
```

A `[-1, 1]` policy on Pendulum's `[-2, 2]` torque:

```python
pendulum = sekai.RescaleAction(sekai.make("Pendulum-v1"), min_action=-1.0, max_action=1.0)
print(pendulum.action_space)  # Box(-1.0, 1.0, (1,), float32)
print(pendulum.action(np.array([0.5], dtype=np.float32)))  # [1.]
```

Deciding every 4 ticks:

```python
env = sekai.ActionRepeat(sekai.make("CartPole-v1"), repeat=4)
env.reset(seed=0)
result = env.step(0)
print(result.reward, result.info)  # 4.0 {'elapsed_steps': 4, 'action_repeats': 4}
```

### Freezing normalisation statistics

`NormalizeObservation` and `NormalizeReward` keep a [`RunningMeanStd`](../api/utils.md)
(`obs_rms` / `return_rms`; an `ExponentialMovingStats` with `stats="ewm"`). Freeze it for
evaluation and save it next to your policy:

```python
train = sekai.NormalizeObservation(sekai.make("CartPole-v1"))
train.action_space.seed(0)
result = train.reset(seed=0)
for _ in range(1000):
    result = train.step(train.action_space.sample())
    if result.done:
        result = train.reset()

state = train.obs_rms.state_dict()       # {'mean': ..., 'var': ..., 'count': ...}

evaluate = sekai.NormalizeObservation(sekai.make("CartPole-v1"))
evaluate.obs_rms.load_state_dict(state)
evaluate.update_running_mean = False     # use, but do not update, the statistics
print(evaluate.obs_rms)  # RunningMeanStd(shape=(4,), count=1046)
```

For a nested observation space `obs_rms` mirrors the nesting (a dict for a `Dict`), with one
statistics object per normalised leaf. The wrapper's own `state_dict()` and
`load_state_dict()` save and restore all of them at once:

```python
portfolio = sekai.NormalizeObservation(sekai.make("Portfolio-v0"))
print(portfolio.obs_rms)  # {'returns': RunningMeanStd(shape=(30, 5), count=0.0001), 'weights': RunningMeanStd(shape=(6,), count=0.0001)}

state = portfolio.state_dict()
print(list(state), sorted(state["weights"]))  # ['returns', 'weights'] ['count', 'mean', 'var']
sekai.NormalizeObservation(sekai.make("Portfolio-v0")).load_state_dict(state)
```

## Writing a wrapper

For a pure transformation, subclass one of the three semantic base classes and implement a
single method. They handle `step`, `reset` and both async paths for you.

| Base class | Implement | Called on |
|---|---|---|
| `ObservationWrapper` | `observation(obs)` | every observation from `reset()` and `step()` |
| `RewardWrapper` | `reward(reward)` | every reward from `step()` |
| `ActionWrapper` | `action(action)` (optionally `reverse_action(action)`) | every action passed to `step()` |

If the transformation changes the space, assign the new one in `__init__`. Assigning
`self.observation_space` or `self.action_space` on a wrapper overrides it for that layer
only.

```python
class ReturnsInPercent(sekai.ObservationWrapper):
    """Trading-v0 observes log returns, then the position; scale the returns to percent."""

    def observation(self, obs):
        out = obs.copy()
        out[:-1] *= 100.0
        return out


class RewardInBps(sekai.RewardWrapper):
    def reward(self, reward):
        return 1e4 * reward


class SoftmaxWeights(sekai.ActionWrapper):
    """Let the policy output unbounded logits; the env receives softmax weights."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = sekai.Box(-10.0, 10.0, shape=env.action_space.shape)

    def action(self, logits):
        z = np.exp(logits - np.max(logits))
        return (z / z.sum()).astype(np.float32)


trading = ReturnsInPercent(sekai.make("Trading-v0"))
print(trading.reset(seed=0).observation[:3])  # [ 0.17030995 -0.15453172  0.81876147]

portfolio = RewardInBps(SoftmaxWeights(sekai.make("Portfolio-v0", n_assets=2)))
print(portfolio.action_space)  # Box(-10.0, 10.0, (3,), float32)
portfolio.reset(seed=0)
print(round(portfolio.step(np.array([10.0, -10.0, -10.0], dtype=np.float32)).reward, 2))  # -49.31
```

### Stateful wrappers and the async rule

For anything else (episode logic, state across steps) subclass `Wrapper` and override
`step()` / `reset()`. Keep in mind how the async path behaves:

!!! warning "Override both paths, or neither"
    If a class overrides `step()` but not `async_step()` (likewise `reset()` /
    `async_reset()`), sekai routes `async_step()` through your `step()`, so `await
    env.async_step(a)` never skips your logic. The price: below your wrapper the env is driven
    synchronously, even if it implements real async I/O. To keep an
    [async-native env](async.md) asynchronous end to end, implement **both** `step()` and
    `async_step()` (and the `reset` pair), sharing the logic through a helper, as every
    built-in wrapper does.

```python
class DrawdownStop(sekai.Wrapper):
    """End the episode once equity falls `limit` below its running peak."""

    def __init__(self, env, limit=0.2):
        super().__init__(env)
        self.limit = limit
        self.peak = 1.0

    def _on_reset(self, result):
        self.peak = 1.0
        return result

    def _on_step(self, result):
        equity = result.info["equity"]
        self.peak = max(self.peak, equity)
        if equity < (1 - self.limit) * self.peak:
            return sekai.StepResult(result.observation, result.reward, True, result.truncated,
                                    {**result.info, "stopped_out": True})
        return result

    def reset(self, *, seed=None, options=None):
        return self._on_reset(self.env.reset(seed=seed, options=options))

    async def async_reset(self, *, seed=None, options=None):
        return self._on_reset(await self.env.async_reset(seed=seed, options=options))

    def step(self, action):
        return self._on_step(self.env.step(action))

    async def async_step(self, action):
        return self._on_step(await self.env.async_step(action))


env = DrawdownStop(sekai.make("Trading-v0", sigma=0.3), limit=0.1)
result, steps = env.reset(seed=0), 0
while not result.done:
    result = env.step(2)  # always long
    steps += 1
print(steps, result.info.get("stopped_out", False))  # 79 True
```
