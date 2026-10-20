# Vector environments

A vector environment runs `num_envs` copies of an environment and batches their results, so a
policy evaluates one batch of observations per step instead of one observation per env.

## Backends

| Backend | Runs sub-envs | Use it for |
|---|---|---|
| native (`NativeVecEnv` subclasses) | all at once, with array operations, in this process | envs with a [native implementation](#native-vector-environments) (the built-in `Trading-v0`, `Portfolio-v0`, `MarketMaking-v0`, `OptimalExecution-v0`), or simulators you can write in array form: usually the fastest by far |
| `SyncVecEnv` | one after another, in this process | fast envs without a native version (classic control, your own), debugging |
| `AsyncVecEnv` | concurrently with `asyncio.gather`, in this process | I/O-bound envs: live feeds, REST/websocket simulators, anything with real [`async_step()`](async.md) |
| `ProcVecEnv` | in parallel, in worker processes | CPU-bound simulators where a step costs more than the IPC round trip |

`SyncVecEnv`, `AsyncVecEnv` and `ProcVecEnv` take a list of zero-argument callables that build
the sub-envs; a native vector env is a class of its own. All share the same API and result
types:

```python
import numpy as np
import sekai

vec = sekai.SyncVecEnv([lambda: sekai.make("CartPole-v1") for _ in range(4)])
print(vec, len(vec))  # SyncVecEnv(num_envs=4) 4
vec.close()
```

`sekai.make_vec()` does the same from a registry ID:

```python
vec = sekai.make_vec(
    "CartPole-v1",
    num_envs=4,
    vectorization_mode="sync",                       # None | "native" | "sync" | "async" | "proc"
    wrappers=[lambda env: sekai.TimeAwareObservation(env)],  # applied to each sub-env
    vector_kwargs={"autoreset": True},               # passed to the VecEnv constructor
    max_episode_steps=200,                           # any other kwarg goes to make()
)
print(vec.observation_space)  # Box([-4.8, -inf, -0.41887903, -inf, 0.0], [4.8, inf, 0.41887903, inf, 1.0], (5,), float32)
vec.close()
```

With the default `vectorization_mode=None`, `make_vec()` uses the env's native
implementation when it has one and no per-env wrappers apply (no `wrappers=`, and no
`additional_wrappers` in the registry entry), and `SyncVecEnv` otherwise:

```python
print(sekai.make_vec("Portfolio-v0", num_envs=64))                            # PortfolioVecEnv(num_envs=64)
print(sekai.make_vec("Portfolio-v0", num_envs=64, vectorization_mode="sync"))  # SyncVecEnv(num_envs=64)
print(sekai.make_vec("CartPole-v1", num_envs=64))                             # SyncVecEnv(num_envs=64)
```

Every `VecEnv` built by `make_vec()` carries the registry entry as `vec.spec`.

## Spaces and results

`observation_space` and `action_space` are always the **single-env** spaces;
`batch_observation_space` and `batch_action_space` are the batched ones (built with
[`batch_space`](spaces.md#batching)):

```python
vec = sekai.make_vec("CartPole-v1", num_envs=4)
print(vec.observation_space, vec.batch_observation_space)  # Box([-4.8, -inf, -0.41887903, -inf], [4.8, inf, 0.41887903, inf], (4,), float32) Box([-4.8, -inf, -0.41887903, ..., -inf, -0.41887903, -inf], [4.8, inf, 0.41887903, ..., inf, 0.41887903, inf], (4, 4), float32)
print(vec.action_space, vec.batch_action_space)            # Discrete(2) MultiDiscrete([2, 2, 2, 2])
```

`reset()` returns a `VecResetResult` and `step()` a `VecStepResult`:

| Field | `VecStepResult` | `VecResetResult` |
|---|---|---|
| `observations` | batched: an array `(N, *obs_shape)`, or a dict/tuple of arrays for `Dict`/`Tuple` spaces (a tuple of per-env samples for a [`Sequence`](spaces.md#sequence)) | same |
| `rewards` | `float64` array `(N,)` | — |
| `terminated`, `truncated` | `bool` arrays `(N,)` | — |
| `dones` | property: `terminated \| truncated` | all `False` |
| `infos` | `list[dict]`, one per sub-env | same |

Actions are a batch in the shape of `batch_action_space` (an array, or a dict of arrays for a
`Dict` action space); a plain list of per-env actions works too.

```python
actions = vec.batch_action_space  # MultiDiscrete([2, 2, 2, 2]), with its own generator
actions.seed(0)
result = vec.reset(seed=0)
result = vec.step(actions.sample())
print(result.observations.shape, result.rewards, result.dones)  # (4, 4) [1. 1. 1. 1.] [False False False False]
result = vec.step([0, 1, 0, 1])
```

`infos` is a list of dicts rather than gymnasium's dict of arrays. If you want arrays,
`stack_infos()` stacks the keys present in every sub-env's info:

```python
print(sekai.stack_infos(result.infos))  # {'elapsed_steps': array([2, 2, 2, 2])}
```

## Seeding and options

| `reset(seed=...)` | Effect |
|---|---|
| an `int` | derives `num_envs` independent seeds from it (`numpy.random.SeedSequence.spawn`) |
| a `list` of `num_envs` seeds | one seed per sub-env (a `None` entry: that sub-env continues its stream) |
| `None` | each sub-env continues its own stream |
| a `Generator` | `TypeError`: one generator cannot seed `num_envs` sub-envs independently |

`options` is either one dict passed to every sub-env or a list with one per sub-env.

!!! note "Different from gymnasium"
    gymnasium seeds sub-env `i` with `seed + i`; sekai spawns statistically independent child
    seeds. Pass an explicit list, `vec.reset(seed=[0, 1, 2, 3])`, to choose the seeds yourself.

### Episode samplers

On every backend the sub-envs are independent copies of the env. `make_vec()` gives each
sub-env `sampler.spawn()` of the [episode sampler](../finance/data.md#episode-samplers), and a
native env does the same for every sub-env it simulates. A sampler with per-env state spawns
a fresh copy: with `sampler="sequential"` (or a `SequentialStart`) every sub-env replays the
same windows, and the instance you pass is not advanced:

```python
history = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0.0, 0.01, 179)))
trading = {"prices": history, "episode_length": 21, "lookback": 10}

replay = sekai.make_vec("Trading-v0", 3, sampler="sequential", **trading)
first = replay.reset(seed=0)
print([info["timestamp"] for info in first.infos])  # [10, 10, 10]
replay.close()
```

Any other sampler is shared by default (`EpisodeSampler.spawn()` returns the sampler itself),
so a change you make to it later, such as a curriculum widening the range of start dates,
reaches every sub-env. If your own sampler keeps per-env state, override `spawn()` to return
a fresh copy. `ProcVecEnv` workers always hold pickled copies, which later changes in the
parent process do not reach.

To split a history between the sub-envs instead, so that together they visit every window
once, give each sub-env its own offset sampler when you build the sub-envs yourself
(`SyncVecEnv`, `AsyncVecEnv` or `ProcVecEnv`):

```python
from sekai.finance import SequentialStart

lanes = 4
split = sekai.SyncVecEnv(
    [
        lambda i=i: sekai.make(
            "Trading-v0", sampler=SequentialStart(start=21 * i, stride=21 * lanes), **trading
        )
        for i in range(lanes)
    ]
)
step = split.reset(seed=0)
firsts = [info["timestamp"] for info in step.infos]
for _ in range(21):  # one episode each; the autoreset starts the next window
    step = split.step(np.ones(lanes, int))
firsts += [info["reset_info"]["timestamp"] for info in step.infos]
print(sorted(int(t) for t in firsts))  # [10, 31, 52, 73, 94, 115, 136, 157]
split.close()
```

or, on any backend including a native one, turn autoreset off and start each sub-env's
episode where you want it with per-sub-env `options`:

```python
starts = list(range(10, len(history) - 21, 21))  # the first decision of every window
split = sekai.make_vec("Trading-v0", lanes, vector_kwargs={"autoreset": False}, **trading)
covered = []
for k in range(0, len(starts), lanes):
    step = split.reset(options=[{"start": t} for t in starts[k : k + lanes]])
    covered += [info["timestamp"] for info in step.infos]
    finished = np.zeros(lanes, bool)
    while not finished.all():
        step = split.step(np.ones(lanes, int))
        finished |= step.dones
print(type(split).__name__, covered)  # TradingVecEnv [10, 31, 52, 73, 94, 115, 136, 157]
split.close()
```

## Autoreset

By default (`autoreset=True`) a sub-env that finishes an episode is reset **inside the same
`step()` call**. For that sub-env:

- `observations[i]` is the **first observation of the new episode**;
- `rewards[i]`, `terminated[i]`, `truncated[i]` belong to the step that ended the episode;
- `infos[i]["final_observation"]` and `infos[i]["final_info"]` hold the terminal step's
  observation and info, and `infos[i]["reset_info"]` the info returned by the reset.

```python
actions.seed(0)
result = vec.reset(seed=0)
while not result.dones.any():
    result = vec.step(actions.sample())

i = int(np.flatnonzero(result.dones)[0])
print(sorted(result.infos[i]))  # ['elapsed_steps', 'final_info', 'final_observation', 'reset_info']
terminal_obs = result.infos[i]["final_observation"]  # last obs of the finished episode
next_obs = result.observations[i]                     # first obs of the new one
vec.close()
```

This is the `AutoReset` wrapper's convention, applied per sub-env; a sub-env that is already
wrapped in `AutoReset` is not reset twice.

!!! warning "Same-step, not next-step"
    gymnasium ≥ 1.0 vector envs default to *next-step* autoreset: the step after an episode
    ends returns the reset observation with reward 0 and ignores the action. sekai uses
    *same-step* autoreset (like Stable-Baselines3's `VecEnv`), so every `step()` is a real
    transition and there is no dummy step to mask out. When bootstrapping a value target on
    truncation, use `infos[i]["final_observation"]`, not `observations[i]`.

Pass `autoreset=False` (or `vector_kwargs={"autoreset": False}` to `make_vec`) to manage
resets yourself, e.g. for evaluation that runs exactly one episode per sub-env. A sub-env
that finishes is then **frozen** until the next `reset()`, on every backend: it is not stepped
again and its action is ignored, and every later `step()` returns, for that sub-env, its final
observation, reward `0`, the `terminated` / `truncated` flags it finished with (so `dones[i]`
stays `True`) and an empty info. So you can keep stepping until every sub-env has finished,
even with envs that refuse a step after their episode ends (the finance envs raise
`ResetRequired`), and `VecRecordEpisodeStatistics` records each episode once. `reset()`
unfreezes every sub-env. A sub-env that resets itself (one wrapped in `AutoReset`) never
freezes.

```python
vec = sekai.make_vec("OptimalExecution-v0", 3, vector_kwargs={"autoreset": False})
vec.reset(seed=0)
sell = np.array([[1.0], [0.5], [0.5]], np.float32)  # sub-env 0 sells everything at once
first = vec.step(sell)
later = vec.step(sell)
print(first.dones, later.dones)  # [ True False False] [ True False False]
print(later.rewards[0], later.infos[0])  # 0.0 {}
print(np.array_equal(later.observations[0], first.observations[0]))  # True
vec.close()
```

## Vector wrappers

Vector wrappers wrap a whole `VecEnv`. They are exported from `sekai` and `sekai.vector`.

| Wrapper | What it does |
|---|---|
| `VecRecordEpisodeStatistics(venv, tracker=None, deque_size=100)` | `infos[i]["episode"]` is an [`EpisodeStats`](statistics.md) (with `extra={"env_index": i}`) when sub-env `i` finishes an episode; without autoreset, once per episode however long a [frozen](#autoreset) sub-env keeps being stepped |
| `VecNormalizeObservation(venv, epsilon=1e-8, *, stats="running", halflife=1000.0, clip=None)` | the options of [`NormalizeObservation`](wrappers.md#built-in-wrappers) (every `Box` part normalised, a `MaskedBox` over occupied slots only, `"ewm"` statistics, clipping), with statistics shared across sub-envs (`obs_rms`, `state_dict()`); `infos[i]["final_observation"]` is normalised with the current statistics, without updating them. With `"ewm"`, `halflife` counts vector steps: one step updates the statistics once, however many sub-envs it carries, so it needs no scaling with `num_envs` |
| `VecNormalizeReward(venv, gamma=0.99, epsilon=1e-8)` | divides rewards by the running std of each sub-env's discounted return (`return_rms`) |
| `VecTransformObservation(venv, func, new_observation_space=None)` | applies `func` to every batch of observations |
| `VecTransformReward(venv, func)` | applies `func` to every `(N,)` reward array |

The base classes `VecWrapper`, `VecObservationWrapper` (implement `observation(batch)`),
`VecRewardWrapper` (`reward(rewards)`) and `VecActionWrapper` (`action(actions)`) work like
their [single-env counterparts](wrappers.md#writing-a-wrapper), and `vec.unwrapped` returns
the innermost `VecEnv`. Every vector wrapper delegates `async_step()` / `async_reset()` too,
so wrapping an `AsyncVecEnv` keeps it [concurrent](async.md#vector-wrappers-keep-the-async-path).

### Observation wrappers and terminal observations

A `VecObservationWrapper` transforms the batched observations **and** every
`infos[i]["final_observation"]` left by autoreset, so a value function bootstrapping from the
terminal state sees the same representation as the policy. Terminal observations go through
the `final_observation(obs)` hook, which by default runs `observation()` on a batch of one.
Override it when your transform has side effects (such as updating running statistics,
which is why `VecNormalizeObservation` overrides it) or only makes sense on a full batch:

```python
class CrossSectionalDemean(sekai.VecObservationWrapper):
    """Subtract the cross-sub-env mean from each observation."""

    def observation(self, batch):
        return batch - batch.mean(axis=0, keepdims=True)

    def final_observation(self, obs):
        return obs  # a single row has no cross-section; pass it through unchanged


vec = CrossSectionalDemean(sekai.make_vec("CartPole-v1", num_envs=4))
result = vec.reset(seed=0)
print(np.allclose(result.observations.mean(axis=0), 0.0))  # True
vec.close()
```

### A PPO-style stack

The usual setup for on-policy training on continuous control: record raw episode returns
first, then normalise observations and rewards.

```python
stats = sekai.VecRecordEpisodeStatistics(sekai.make_vec("Pendulum-v1", num_envs=8))
vec = sekai.VecNormalizeObservation(stats)      # stats is innermost: it sees raw rewards
vec = sekai.VecNormalizeReward(vec, gamma=0.99)
print(vec)  # VecNormalizeReward(VecNormalizeObservation(VecRecordEpisodeStatistics(SyncVecEnv(num_envs=8))))

def value_fn(obs):                           # stand-in for your critic
    return np.zeros(len(obs))

actions = vec.batch_action_space
actions.seed(0)
result = vec.reset(seed=0)
for _ in range(250):
    result = vec.step(actions.sample())
    values_next = value_fn(result.observations)
    for i in np.flatnonzero(result.truncated):           # bootstrap from the true last state
        values_next[i] = value_fn(result.infos[i]["final_observation"][None])[0]

print(stats.tracker.summary()["num_episodes"])  # 8
```

For evaluation, freeze the statistics with `update_running_mean = False` on both normalisation
wrappers, and save `obs_rms.state_dict()` with the policy.

## Reaching the sub-envs

| Method | Does |
|---|---|
| `vec.get_attr(name)` | read `name` from each sub-env |
| `vec.set_attr(name, value_or_list)` | one value for all, or a list of `num_envs` values |
| `vec.call(name, *args, **kwargs)` | call a method on each sub-env (or read the attribute if not callable) |

Names are looked up through each sub-env's wrapper stack (like `Env.get_attr()`), so
attributes of the base env and of any wrapper are both reachable, and `set_attr` changes the
layer that actually has the attribute. They work across processes with `ProcVecEnv` too, and
`vec.call("action_mask")` collects every sub-env's [action mask](dynamic.md#action-masks). A
native vector env has no sub-env objects; see [its rules](#reaching-sub-envs-of-a-native-env).

```python
vec = sekai.make_vec("CartPole-v1", num_envs=2)
print(vec.get_attr("max_episode_steps"))   # [500, 500]
print(vec.get_attr("x_threshold"))         # [2.4, 2.4]
vec.set_attr("x_threshold", [2.0, 3.0])    # reaches CartPoleEnv through TimeLimit
print(vec.get_attr("x_threshold"))         # [2.0, 3.0]
vec.close()
```

## Native vector environments

`SyncVecEnv` steps `num_envs` separate Python environments one after another, so every step
costs `num_envs` Python calls. A native vector env keeps the state of all sub-envs in arrays
with a leading `num_envs` axis and advances them together with array operations: one call per
step, however many sub-envs there are. For simulators that vectorise naturally (most market
models do) that is typically 10 to 100 times faster; see [Performance](#performance).

| `vectorization_mode` | `make_vec()` builds |
|---|---|
| `None` (default) | the native env if the registry entry has a `vector_entry_point`, no `additional_wrappers`, and no `wrappers=` are given, else a `SyncVecEnv` |
| `"native"` (alias `"vector_entry_point"`) | the native env; `ValueError` if there is none, if `wrappers=` is given, or if the registry entry has `additional_wrappers` |
| `"sync"`, `"async"`, `"proc"` | a `SyncVecEnv`, `AsyncVecEnv` or `ProcVecEnv` of `make()`d sub-envs |

A native env has no per-env objects for `wrappers=` (or a registry entry's
`additional_wrappers`) to wrap; use [vector wrappers](#vector-wrappers), which work on any
`VecEnv`.

### Writing one

Subclass `sekai.NativeVecEnv`, pass `num_envs` and the single-env spaces to its constructor,
and implement three hooks:

| Hook | Does |
|---|---|
| `_reset_envs(indices, options)` | (re)initialise the sub-envs in `indices` (an int array), with one options dict (or `None`) per index |
| `_observe()` | return the batched observation of all sub-envs |
| `_step(actions)` | advance every sub-env; return `(rewards, terminated, truncated, infos)`, the first three of shape `(num_envs,)` and `infos` a dict of arrays with a leading `num_envs` axis |
| `_reset_infos(indices)` | optional: info arrays for the sub-envs just reset |

The base class does the rest: seeding, per-env options, `ResetRequired` from a `step()`
before the first `reset()`, [same-step autoreset](#autoreset)
with `final_observation` / `final_info` / `reset_info` (or, with `autoreset=False`, frozen
finished sub-envs), the split of info arrays into one dict
per sub-env (1-D arrays become Python scalars), batched spaces, closing, and the time limit:
pass `max_episode_steps=n` to `NativeVecEnv.__init__` and every sub-env is truncated after
`n` steps with `infos[i]["elapsed_steps"]` set, exactly as the `TimeLimit` that `make()` adds
to each sub-env of a `SyncVecEnv`. A native version of a small single-agent env:

```python
class Drift(sekai.Env):
    """Push a drifting point back towards zero."""

    def __init__(self):
        self.observation_space = sekai.Box(-10.0, 10.0, shape=(1,))
        self.action_space = sekai.Discrete(2)  # push down / push up

    def reset(self, *, seed=None, options=None):
        self._set_rng(seed)
        self.x = self.rng.normal()
        return sekai.ResetResult(np.array([self.x], np.float32), {})

    def step(self, action):
        self.x += (0.5 if action == 1 else -0.5) + self.rng.normal(0.0, 0.1)
        obs = np.array([np.clip(self.x, -10, 10)], np.float32)
        return sekai.StepResult(obs, -abs(self.x), abs(self.x) > 5, False, {})


class DriftVecEnv(sekai.NativeVecEnv):
    """num_envs copies of Drift, stepped together with array operations."""

    def __init__(self, num_envs, *, max_episode_steps=None, autoreset=True):
        super().__init__(
            num_envs,
            sekai.Box(-10.0, 10.0, shape=(1,)),
            sekai.Discrete(2),
            autoreset=autoreset,
            max_episode_steps=max_episode_steps,  # the base class applies the time limit
        )
        self.x = np.zeros(num_envs)

    def _reset_envs(self, indices, options):
        self.x[indices] = self.rng.normal(size=len(indices))

    def _observe(self):
        return np.clip(self.x, -10, 10)[:, None].astype(np.float32)  # a new (N, 1) array

    def _step(self, actions):
        push = np.where(np.asarray(actions) == 1, 0.5, -0.5)
        self.x += push + self.rng.normal(0.0, 0.1, size=self.num_envs)
        terminated = np.abs(self.x) > 5
        return -np.abs(self.x), terminated, np.zeros(self.num_envs, bool), {}


vec = DriftVecEnv(4, max_episode_steps=100)
result = vec.reset(seed=0)
print(result.observations.shape, vec.batch_action_space)  # (4, 1) MultiDiscrete([2, 2, 2, 2])
result = vec.step(np.array([1, 1, 0, 0]))
print(result.rewards.round(3), result.infos[0])  # [-0.572 -0.404 -0.271 -0.3  ] {'elapsed_steps': 1}
```

!!! warning "Return new arrays"
    The observation, rewards and flags are handed to the caller as they are. Return fresh
    arrays from `_observe()` and `_step()`, not the env's own state arrays (or views of them),
    or the next step overwrites results the caller still holds.

With `autoreset=False` the base class replaces whatever `_step()` returns for a frozen
sub-env, but `_step()` still advances all of them (a frozen sub-env is passed its last action
again when actions are a numeric array). If your simulation reads per-step data, such as a
price window it would run past, or should keep a finished sub-env's state as `get_attr()`
reports it, leave the sub-envs in `self.frozen` (a boolean array) as they are: in `Drift`,
`self.x += np.where(self.frozen, 0.0, push + noise)`.

### Registering it

Register the single env as usual and name the native class as its `vector_entry_point`;
`make_vec()` then picks it by default, and the other modes still build sub-envs from the
single env:

```python
sekai.register(
    "demo/Drift-v0", entry_point=Drift, vector_entry_point=DriftVecEnv, max_episode_steps=100
)
print(sekai.make_vec("demo/Drift-v0", num_envs=8))  # DriftVecEnv(num_envs=8)
print(sekai.make_vec("demo/Drift-v0", num_envs=8, vectorization_mode="sync"))  # SyncVecEnv(num_envs=8)
```

`make_vec()` calls `vector_entry_point(num_envs=..., **kwargs)` with the registered kwargs,
the kwargs passed to `make_vec()`, `max_episode_steps` when the entry has one, and
`vector_kwargs`. So the native constructor should accept the single env's keyword arguments
plus `max_episode_steps` and `autoreset`, and pass both on to `NativeVecEnv.__init__`, which
applies the time limit. The finance envs' native versions do, so
`sekai.make_vec("Trading-v0", 8, max_episode_steps=50)` truncates every sub-env after 50
steps on every backend. In a package, use
a string, `vector_entry_point="my_package.envs:DriftVecEnv"`; `sekai list` marks such
entries `[native vec]`.

!!! note "One generator for all sub-envs"
    A native env draws every sub-env's randomness from one generator, `self.rng`.
    `reset()` takes the same seeds as any `VecEnv`: an int seeds the generator, and a list
    of `num_envs` seeds is combined into one seed, where a `None` entry contributes a draw
    from the current stream (so `[None] * num_envs` continues it, like `reset()` without a
    seed) and a `Generator` entry a draw from that generator. The results are therefore not
    draw-for-draw identical to the same env under `SyncVecEnv`, which seeds each sub-env
    separately: they are identical in distribution, and exactly identical when the dynamics
    are deterministic given the options (historical data with fixed start dates, for
    example).

### Reaching sub-envs of a native env

`get_attr(name)` and `call(name, ...)` read attributes and call methods of the native object
itself. A result that is an array with a leading `num_envs` axis is split into one value per
sub-env; anything else is repeated for every sub-env. `set_attr(name, values)` changes an
existing attribute, storing a list of `num_envs` values as one array and any other value as
it is. A name the object does not have raises `AttributeError`: a new attribute would not
change the simulation.

```python
print(vec.get_attr("max_episode_steps"))  # [100, 100, 100, 100]
print([round(float(x), 3) for x in vec.get_attr("x")])  # [0.572, 0.404, 0.271, -0.3]
vec.close()
```

A method or property whose value is the same for every sub-env, but may still be an array
whose length happens to equal `num_envs` (a schedule of `n_steps` rows, say), should be
marked with `sekai.vector.shared`, so that every sub-env gets the whole value rather than one
row of it:

```python
from sekai.vector import shared


class ScheduledDrift(DriftVecEnv):
    @shared
    def push_schedule(self):
        """The push for each of the next 4 steps, the same for every sub-env."""
        return np.array([1, 1, 0, 0])


print(ScheduledDrift(4).call("push_schedule")[0])  # [1 1 0 0]
```

The finance envs' native versions keep their parameters on `vec.template`, a single env
built from the same keyword arguments, and read them from there every step. `get_attr` and
`set_attr` reach them by name, so `vec.set_attr("sigma", 0.0)` changes the simulation just as
it does for every sub-env of a `SyncVecEnv`. All sub-envs share these parameters: `set_attr`
takes one value (or a list repeating one value) and raises `ValueError` for different values
per sub-env, which need `vectorization_mode="sync"`. Parameters that fix array shapes
(`lookback`, `episode_length`, `n_assets`, `allow_short`) are read-only, as are `prices`,
`data`, `transaction_cost` (set `cost_model` instead) and `metadata`. State such as `equity`,
`position`, `price`, `wealth`, `time_remaining` and `elapsed_steps` has one value per sub-env.

A native env that defines `action_mask()` returning an array of shape `(num_envs, ...)` thus
answers `vec.call("action_mask")` like any other `VecEnv`.

### Performance

Random-action throughput of `Portfolio-v0` (five assets, the default), measured with
[`sekai bench`](cli.md#bench) on an 8-core machine, in env steps per second:

| Backend (`--mode`) | 16 sub-envs | 1024 sub-envs |
|---|---|---|
| `native` (`PortfolioVecEnv`) | 195,000 | 1,010,000 |
| `sync` | 49,000 | 55,000 |
| `proc`, one worker per sub-env | 12,000 | (not run) |
| `proc --workers 4` | 23,000 | 89,000 |
| `proc --workers 8` | 17,000 | 122,000 |

A single `Portfolio-v0` env runs about 34,000 steps per second on the same machine.

The native env's advantage grows with the number of sub-envs (4 times `SyncVecEnv` at 16,
almost 20 times at 1024), because its per-step overhead is paid once per batch. With 16
sub-envs `ProcVecEnv` is slower than `SyncVecEnv`: a `Portfolio-v0` step is too cheap to
cover the inter-process round trip. With 1024 sub-envs on 8 workers each round trip carries
128 sub-envs and the cores work in parallel, so it overtakes `SyncVecEnv`, but stays far
behind the native env. Numbers vary with the hardware, the env and its parameters: measure
your own with `sekai bench ID --num-envs N --mode M`.

## ProcVecEnv

`ProcVecEnv` runs the sub-envs in worker processes, by default one per sub-env, and talks to
them over pipes. The env factories are sent with `cloudpickle`, so lambdas and classes
defined in your script work. Two things keep the inter-process overhead low:

- **Shared memory.** When every part of the observation space has a fixed size (`Box`,
  `Discrete`, `MultiDiscrete`, `MultiBinary`, and `Dict`, `Tuple` or `MaskedBox` of them),
  workers write observations straight into one shared-memory block, and only rewards, flags
  and infos are pickled. Large observations (order books, multi-asset windows) no longer pay
  a serialisation cost on every step: for an env with a cheap step and 8 workers, shared
  memory measured 1.6 times faster than pickling with 120 kB observations and 5 times faster
  with 1 MB ones (it makes no difference when the env's own step dominates). Other spaces,
  such as a `Sequence`, fall back to pickling.
- **Several sub-envs per worker.** `num_workers` splits the sub-envs evenly between that
  many processes. With more sub-envs than CPU cores, set it to about the number of cores: it
  avoids oversubscription and spreads each round trip over several sub-envs.

!!! danger "Guard your entry point"
    The default start method is `forkserver` (`spawn` on Windows), which re-imports your main
    module in every worker. A script that creates a `ProcVecEnv` (directly or with
    `vectorization_mode="proc"`) **must** put that code under `if __name__ == "__main__":`,
    or each worker will try to start its own workers. Pass `context="fork"` to use `fork` on
    Linux if you know your process is fork-safe.

An exception inside a worker is raised in the parent as `VecEnvWorkerError`, carrying the
sub-env's index (`rank`), the exception type (`exc_type`) and the worker's traceback
(`remote_traceback`). The other workers' replies are still collected and the workers stay
alive, so the `VecEnv` can still be used or closed cleanly. A worker that has died also
raises `VecEnvWorkerError` (see [below](#which-sub-env-failed)). Replies left unread by an interrupted call (e.g. Ctrl-C during
`step()`) are discarded before the next command, and every command is pickled before
anything is sent, so an unpicklable action fails without leaving workers out of step.

```python
import numpy as np
import sekai


class FragileEnv(sekai.Env):
    def __init__(self):
        self.observation_space = sekai.Box(-1.0, 1.0, shape=(2,))
        self.action_space = sekai.Discrete(2)

    def reset(self, *, seed=None, options=None):
        self._set_rng(seed)
        return sekai.ResetResult(np.zeros(2, np.float32), {})

    def step(self, action):
        if action == 1:
            raise ValueError("action 1 is not supported")
        return sekai.StepResult(np.zeros(2, np.float32), 0.0, False, False, {})


if __name__ == "__main__":
    with sekai.ProcVecEnv([FragileEnv] * 2) as vec:
        vec.reset(seed=0)
        try:
            vec.step(np.array([0, 1]))
        except sekai.VecEnvWorkerError as e:
            print(e.rank, e.exc_type)  # 1 ValueError
            # e.remote_traceback holds the worker's formatted traceback
        print(vec.step(np.array([0, 0])).rewards)  # [0. 0.]
```

| Constructor argument | Default | Meaning |
|---|---|---|
| `autoreset` | `True` | [same-step autoreset](#autoreset) |
| `context` | `"forkserver"` (`"spawn"` on Windows) | `multiprocessing` start method |
| `close_timeout` | `5.0` | seconds `close()` waits for workers before terminating them |
| `num_workers` | `None` (one per sub-env) | worker processes; the sub-envs are split evenly between them |
| `shared_memory` | `None` | pass observations through shared memory: `None` uses it when the observation space allows, `True` requires it (`TypeError` otherwise), `False` turns it off; `vec.shared_memory` tells whether it is in use |

With `make_vec()`, pass these through `vector_kwargs`:

```python
if __name__ == "__main__":
    vec = sekai.make_vec(
        "Portfolio-v0", num_envs=64, vectorization_mode="proc", vector_kwargs={"num_workers": 8}
    )
    print(vec.num_workers, vec.shared_memory)  # 8 True
    vec.close()
```

### Which sub-env failed

When a sub-env raises, `VecEnvWorkerError.rank` is its index, also when its worker runs
several sub-envs. When a worker process dies (killed, out of memory, or `os._exit()` in env
code), there is no traceback, and which of its sub-envs was running is known only if it runs
one: then `rank` is that sub-env, as above. A worker running several reports itself and all
of them, with `rank` set to the first:

```text
VecEnvWorkerError: Sub-env 1 raised EOFError: worker process exited unexpectedly
VecEnvWorkerError: Worker 0 (sub-envs 0-1) raised EOFError: worker process exited unexpectedly
```

Every later call that needs the dead worker raises `VecEnvWorkerError` too; `close()` still
releases the others.

## Closing

Every `VecEnv` is a context manager; leaving the `with` block calls `close()`, which closes
the sub-envs (and, for `ProcVecEnv`, shuts the workers down). Afterwards `vec.closed` is
`True` and `step()` / `reset()` raise `sekai.VecEnvClosed`; calling `close()` again is a
no-op.

If a sub-env's own `close()` raises, the other sub-envs are still closed and everything is
released first (`AsyncVecEnv`'s event loop, `ProcVecEnv`'s workers and shared memory); then
the first error is raised, as a `VecEnvWorkerError` for `ProcVecEnv`. A `ProcVecEnv` worker
that does not exit within `close_timeout` is terminated, and killed if it ignores that.

Building a `VecEnv` cleans up after itself the same way: if an env factory raises (or, for
`ProcVecEnv`, a worker cannot be started, or the constructor fails later), the sub-envs
already built are closed, and any workers stopped, before the error propagates.

```python
vec = sekai.make_vec("CartPole-v1", num_envs=2)
vec.close()
try:
    vec.reset(seed=0)
except sekai.VecEnvClosed as e:
    print(vec.closed, e)  # True This VecEnv has been closed.
```
