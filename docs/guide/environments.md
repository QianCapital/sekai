# Environments

`sekai.Env[ObsType, ActType]` is the base class of every single-agent environment. It keeps
gymnasium's shape (`reset`, `step`, two spaces) and changes what comes back: structured
results instead of tuples, async methods on every env, and no rendering on the env itself.

## The interface

| Member | Kind | Notes |
|---|---|---|
| `observation_space` | attribute | a [`sekai.Space`](spaces.md); set it in `__init__` |
| `action_space` | attribute | a [`sekai.Space`](spaces.md); set it in `__init__` |
| `reset(*, seed=None, options=None)` | **abstract** | returns a `ResetResult` |
| `step(action)` | **abstract** | returns a `StepResult` |
| `async_reset(...)`, `async_step(action)` | coroutine | default to the sync methods; override for real async I/O ([Async](async.md)) |
| `close()` | method | release resources; no-op by default |
| `rng` | property | the env's `numpy.random.Generator` |
| `_set_rng(seed)` | method | seed `rng`; call it first in `reset()`; returns the seed in use |
| `metadata` | class attribute | free-form dict, e.g. `{"render_fps": 50}` |
| `spec` | attribute | the `EnvSpec` used by `sekai.make()`, else `None` |
| `unwrapped` | property | the base env under any wrappers (itself, for a base env) |
| `has_attr(name)`, `get_attr(name)` | method | look an attribute up through the wrapper stack |
| `with env:` | context manager | calls `close()` on exit |

There is no `render()`, `render_mode` or `np_random`: rendering is done by
[`Renderer` objects](rendering.md) and the generator is `env.rng`.

## Results

`step()` returns a `StepResult` and `reset()` a `ResetResult`. Both are frozen, slotted
dataclasses, generic in the observation type.

| | `StepResult` | `ResetResult` |
|---|---|---|
| `observation` | next observation | first observation |
| `reward` | `float` | — |
| `terminated` | reached a terminal state of the MDP | — |
| `truncated` | cut short (time limit, data ran out, ...) | — |
| `info` | `dict[str, Any]` | `dict[str, Any]` |
| `done` | `terminated or truncated` | always `False` |

Because `ResetResult.done` is always `False`, a single variable can hold either result:

```python
import sekai

env = sekai.make("CartPole-v1")
env.action_space.seed(0)
result = env.reset(seed=0)
while not result.done:
    result = env.step(env.action_space.sample())
print(type(result).__name__, result.terminated, result.truncated)  # StepResult True False
```

Results are immutable, which makes them safe to store in replay buffers or send across
threads. Use `dataclasses.replace()` to derive a modified copy:

```python
import dataclasses

try:
    result.reward = 0.0
except dataclasses.FrozenInstanceError as e:
    print(e)  # cannot assign to field 'reward'

zeroed = dataclasses.replace(result, reward=0.0)
print(result.reward, zeroed.reward)  # 1.0 0.0
```

!!! info "terminated vs truncated"
    Set `terminated=True` when the episode reached a terminal state of the underlying MDP (the
    pole fell, the inventory is sold, the session closed and the agent can *see* the clock).
    Set `truncated=True` when it was cut off for a reason outside the MDP (a time limit the
    agent cannot observe, the end of a historical price file). Value-based learners bootstrap
    from truncated states but not from terminated ones, so the distinction matters.

## Seeding

Every env owns one `numpy.random.Generator`, `env.rng`. The contract:

| Call | Effect |
|---|---|
| `reset(seed=int)` | re-seed `rng` deterministically |
| `reset(seed=None)` (or `reset()`) | **keep** the current generator, so the stream continues |
| the very first `reset()` without a seed | create a generator from 128 bits of OS entropy |
| `reset(seed=np.random.Generator)` | use that generator directly (the recorded seed is `-1`) |

So `reset(seed=s)` followed by plain `reset()` calls gives one reproducible sequence of
episodes, exactly as in gymnasium:

```python
first = sekai.make("CartPole-v1")
second = sekai.make("CartPole-v1")

a = [first.reset(seed=7).observation] + [first.reset().observation for _ in range(3)]
b = [second.reset(seed=7).observation] + [second.reset().observation for _ in range(3)]
print(all((x == y).all() for x, y in zip(a, b)))  # True
```

Inside your environment, call `self._set_rng(seed)` as the **first** line of `reset()` and
draw every random number from `self.rng`. Never use `np.random.*` module functions or
Python's `random`: they bypass the seed.

!!! warning "Spaces are seeded separately"
    `env.action_space.sample()` draws from the *space's* generator, which `env.reset(seed=...)`
    does not touch. Call `env.action_space.seed(s)` for reproducible random actions (as in
    gymnasium).

Wrappers have no generator of their own: `wrapper.rng` *is* the base env's `rng`, and
`wrapper.reset(seed=s)` seeds the base env. The one exception is
[`StickyAction`](wrappers.md#built-in-wrappers), which derives a private generator from the
reset seed so that it does not perturb the env's stream.

## Writing an environment

A complete environment: an inventory manager that orders stock each day to meet random
demand over a season.

```python
import numpy as np
import sekai


class InventoryEnv(sekai.Env[dict, np.intp]):
    """Order stock each day to meet Poisson demand over a season."""

    metadata = {"render_fps": 4}

    def __init__(self, capacity: int = 20, mean_demand: float = 4.0, days: int = 30,
                 price: float = 2.0, unit_cost: float = 1.0, holding_cost: float = 0.1):
        self.capacity, self.mean_demand, self.days = capacity, mean_demand, days
        self.price, self.unit_cost, self.holding_cost = price, unit_cost, holding_cost
        self.observation_space = sekai.Dict({
            "stock": sekai.Box(0, capacity, shape=(1,), dtype=np.int64),
            "days_left": sekai.Discrete(days + 1),
        })
        self.action_space = sekai.Discrete(capacity + 1)  # units to order
        self.stock = None

    def reset(self, *, seed=None, options=None):
        self._set_rng(seed)                       # 1. seed first
        options = options or {}
        self.stock = int(options.get("initial_stock", 0))
        self.day = 0
        return sekai.ResetResult(self._obs(), info={})

    def step(self, action):
        if self.stock is None:
            raise sekai.ResetRequired()           # 2. fail loudly on misuse
        if not self.action_space.contains(action):
            raise sekai.InvalidAction(action, self.action_space)
        ordered = min(int(action), self.capacity - self.stock)
        self.stock += ordered
        demand = int(self.rng.poisson(self.mean_demand))   # 3. randomness from self.rng
        sold = min(demand, self.stock)
        self.stock -= sold
        self.day += 1
        reward = self.price * sold - self.unit_cost * ordered - self.holding_cost * self.stock
        return sekai.StepResult(
            observation=self._obs(),              # 4. a fresh object every step
            reward=float(reward),
            terminated=self.day >= self.days,     # the clock is observed: a true terminal
            truncated=False,
            info={"demand": demand, "sold": sold, "lost_sales": demand - sold},
        )

    def _obs(self):
        return {"stock": np.array([self.stock], dtype=np.int64),
                "days_left": np.intp(self.days - self.day)}
```

Check it, then run a base-stock policy ("order up to 6 units"):

```python
print(sekai.check_env(InventoryEnv(), check_determinism=True))  # []

env = InventoryEnv()
result = env.reset(seed=0, options={"initial_stock": 5})
total = 0.0
while not result.done:
    result = env.step(max(0, 6 - int(result.observation["stock"][0])))
    total += result.reward
print(round(total, 1), result.info)  # 110.5 {'demand': 3, 'sold': 3, 'lost_sales': 0}
```

The checklist, which [`check_env`](checking.md) verifies for you:

1. Set `observation_space` and `action_space` in `__init__`.
2. Call `self._set_rng(seed)` first in `reset()`; draw all randomness from `self.rng`.
3. Return observations that lie in `observation_space`, with the space's dtype.
4. Return a new observation object each step; never mutate one you already returned
   (a replay buffer may still hold it).
5. Return a `float` reward and `bool` flags; keep `info` a `dict`.
6. Accept `options` in `reset()` for anything configurable per episode.

!!! tip "Type parameters"
    `sekai.Env[ObsType, ActType]` is generic, so a type checker knows that
    `env.step(a).observation` is, say, an `np.ndarray`. The parameters are optional;
    `class MyEnv(sekai.Env)` works too.

## Metadata and spec

`metadata` is a free-form class-level dict. sekai reads one key: `render_fps`, used by
[`RecordFrames.save()`](rendering.md#recording-frames) and passed through by
[`to_gymnasium()`](gymnasium.md).

`spec` is set by `sekai.make()` to the [`EnvSpec`](registry.md#envspec) that built the env,
including the keyword arguments actually used, so `env.spec.make()` builds an identical
copy. Environments constructed directly have `spec = None`.

```python
env = sekai.make("Portfolio-v0", n_assets=3, episode_length=20)
print(env, env.spec.kwargs)  # <Portfolio-v0> {'n_assets': 3, 'episode_length': 20}

clone = env.spec.make()
print(clone.unwrapped.n_assets)  # 3
print(sekai.envs.CartPoleEnv())  # <CartPoleEnv>
```

## Lifecycle

`close()` releases whatever the env holds (files, sockets, subprocesses). Environments are
context managers, so `with` guarantees it runs:

```python
with sekai.make("CartPole-v1") as env:
    env.reset(seed=0)
    env.step(0)
# env.close() has been called here, even if the block raised
```

## Reaching through wrappers

Wrappers do **not** forward unknown attributes (gymnasium 1.x dropped that too). Use
`unwrapped` for the base env, or `get_attr()` / `has_attr()`, which search the stack from the
outside in:

```python
env = sekai.make("CartPole-v1")                 # TimeLimit(CartPoleEnv)
print(env.has_attr("x_threshold"), env.get_attr("x_threshold"))  # True 2.4
print(env.unwrapped.x_threshold)                # 2.4
print(env.get_attr("max_episode_steps"))        # 500
```

`len(wrapper)` counts the layers and iterating over a wrapper walks from the outermost layer
to the base env; see [Wrappers](wrappers.md#inspecting-a-stack).
