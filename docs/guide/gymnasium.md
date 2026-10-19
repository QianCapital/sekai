# Migrating from gymnasium

sekai keeps gymnasium's mental model (environments, spaces, wrappers, a registry, vector
envs), so most code ports line by line. This page maps the API, shows how to use sekai
environments with gymnasium-based libraries (and the reverse), and lists the differences that
change behaviour.

```bash
pip install "sekai[gymnasium]"   # the bridge needs gymnasium >= 1.0
```

## API mapping

| gymnasium | sekai |
|---|---|
| `gym.make("CartPole-v1")` | `sekai.make("CartPole-v1")` |
| `gym.make(id, render_mode="rgb_array")`, `env.render()` | `renderer = CartPoleRenderer()`, `renderer.render(env)` ([Rendering](rendering.md)) |
| `obs, info = env.reset(seed=0)` | `result = env.reset(seed=0)` → `result.observation`, `result.info` |
| `obs, reward, terminated, truncated, info = env.step(a)` | `result = env.step(a)` → `result.observation`, `.reward`, `.terminated`, `.truncated`, `.info` |
| `terminated or truncated` | `result.done` |
| `env.np_random` | `env.rng` |
| `super().reset(seed=seed)` inside `reset()` | `self._set_rng(seed)` |
| `self.np_random` inside the env | `self.rng` |
| `env.unwrapped` | `env.unwrapped` |
| `env.get_wrapper_attr(name)` | `env.get_attr(name)` |
| `gym.spaces.Box`, `Discrete`, `MultiDiscrete`, `MultiBinary`, `Dict`, `Tuple` | `sekai.Box`, `sekai.Discrete`, ... (same arguments) |
| `gym.spaces.Sequence(space, stack=False)` | `sekai.Sequence(space, max_length=None, stack=False)` |
| (none) | `sekai.MaskedBox`: a padded set with a validity mask ([Variable-size observations](dynamic.md)) |
| `gym.spaces.utils.flatten(space, x)` | `sekai.spaces.flatten(space, x)` |
| `gym.register(...)` / `gym.spec(id)` / `gym.pprint_registry()` | `sekai.register(...)` / `sekai.spec(id)` / `sekai.pprint_registry()` |
| `gym.make_vec(id, n, vectorization_mode="sync")` | `sekai.make_vec(id, n, vectorization_mode="sync")` |
| `gym.make_vec(id, n, vectorization_mode="async")` (subprocesses) | `sekai.make_vec(id, n, vectorization_mode="proc")` |
| `gym.make_vec(id, n, vectorization_mode="vector_entry_point")` | `sekai.make_vec(id, n, vectorization_mode="native")` (`"vector_entry_point"` is accepted too); both are the default when the env registers one |
| `gym.register(id, ..., vector_entry_point=...)`, a `gym.vector.VectorEnv` subclass | `sekai.register(id, ..., vector_entry_point=...)`, a [`sekai.NativeVecEnv`](vector.md#native-vector-environments) subclass |
| `gym.vector.SyncVectorEnv` / `AsyncVectorEnv` | `sekai.SyncVecEnv` / `sekai.ProcVecEnv` |
| (none) | `sekai.AsyncVecEnv`: asyncio, for I/O-bound envs |
| `envs.single_observation_space` / `envs.observation_space` | `vec.observation_space` / `vec.batch_observation_space` |
| `infos["episode"]["r"][i]` (vector) | `infos[i]["episode"].episode_return` |
| `gymnasium.utils.env_checker.check_env(env)` (raises) | `sekai.check_env(env)` (returns a list of issues) |
| `gym.wrappers.RecordVideo` | `sekai.rendering.RecordFrames` + `save()` |
| an `action_masks()` method for sb3-contrib's `MaskablePPO` | an `action_mask()` method, forwarded as `action_masks()` by `to_gymnasium()` |
| PettingZoo `ParallelEnv` | `sekai.MultiAgentEnv`, bridged with `to_pettingzoo()` / `from_pettingzoo()` |

### Wrappers

Most wrappers exist under the same name. Argument names that differ:

| gymnasium | sekai |
|---|---|
| `Autoreset(env)` | `AutoReset(env)`; also sets `info["final_observation"]` |
| `RecordEpisodeStatistics(env, buffer_length=100)` | `RecordEpisodeStatistics(env, tracker=None, deque_size=100)`; `info["episode"]` is an `EpisodeStats` |
| `FrameStackObservation(env, stack_size, padding_type="reset")` | `FrameStackObservation(env, num_stack, padding="reset")`; `padding="zeros"` for gymnasium's `"zero"` (no custom padding value); any batchable space, `Dict` included |
| `TransformObservation(env, func, observation_space)` | `TransformObservation(env, fn, new_observation_space=None)` |
| `FilterObservation(env, filter_keys)` | `FilterObservation(env, keys)` |
| `TimeAwareObservation(env, flatten=True, normalize_time=False, dict_time_key="time")` | `TimeAwareObservation(env, max_episode_steps=None, normalize_time=True, dict_key="time")` |
| `DelayObservation(env, delay)` | `DelayObservation(env, delay, padding="zeros")` (or `"reset"`) |
| `StickyAction(env, repeat_action_probability, repeat_action_duration=1)` | `StickyAction(env, repeat_action_probability)` |
| `MaxAndSkipObservation(env, skip=4)` | `ActionRepeat(env, repeat)` (sums rewards; no frame max-pooling) |

`TimeLimit`, `OrderEnforcing`, `ClipAction`, `RescaleAction`, `FlattenObservation`,
`NormalizeObservation`, `NormalizeReward`, `TransformReward`, `ClipReward`,
`RescaleObservation` and `DtypeObservation` take the same arguments. sekai's
`NormalizeObservation` also accepts keyword-only `stats=`, `halflife=` and `clip=`, and
normalises `Dict` observations part by part. `DelayAction`, `ActionMaskObservation` and
`PadSequenceObservation` have no gymnasium counterpart; see [Wrappers](wrappers.md).

## Side by side

=== "sekai"

    ```python
    import sekai

    env = sekai.make("CartPole-v1")
    env.action_space.seed(0)
    result = env.reset(seed=0)
    total = 0.0
    while not result.done:
        result = env.step(env.action_space.sample())
        total += result.reward
    print(total)  # 18.0
    ```

=== "gymnasium"

    ```python
    import gymnasium as gym

    env = gym.make("CartPole-v1")
    env.action_space.seed(0)
    obs, info = env.reset(seed=0)
    total, done = 0.0, False
    while not done:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        total += reward
        done = terminated or truncated
    print(total)  # 18.0
    ```

The dynamics, seeding and spaces are identical, so both print the same return.

A custom environment:

=== "sekai"

    ```python
    import numpy as np
    import sekai


    class Drift(sekai.Env[np.ndarray, np.intp]):
        def __init__(self):
            self.observation_space = sekai.Box(-10.0, 10.0, shape=(1,))
            self.action_space = sekai.Discrete(2)

        def reset(self, *, seed=None, options=None):
            self._set_rng(seed)
            self.x = self.rng.normal()
            return sekai.ResetResult(np.array([self.x], np.float32), {})

        def step(self, action):
            self.x += (1 if action else -1) * 0.5 + self.rng.normal(0.0, 0.1)
            done = abs(self.x) > 5
            obs = np.array([np.clip(self.x, -10, 10)], np.float32)
            return sekai.StepResult(obs, -abs(self.x), done, False, {})


    print(sekai.check_env(Drift()))  # []
    ```

=== "gymnasium"

    ```python
    import gymnasium as gym
    import numpy as np


    class Drift(gym.Env):
        def __init__(self):
            self.observation_space = gym.spaces.Box(-10.0, 10.0, shape=(1,))
            self.action_space = gym.spaces.Discrete(2)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.x = self.np_random.normal()
            return np.array([self.x], np.float32), {}

        def step(self, action):
            self.x += (1 if action else -1) * 0.5 + self.np_random.normal(0.0, 0.1)
            done = abs(self.x) > 5
            obs = np.array([np.clip(self.x, -10, 10)], np.float32)
            return obs, -abs(self.x), done, False, {}


    from gymnasium.utils.env_checker import check_env
    check_env(Drift())  # raises on failure
    ```

A vectorised loop:

=== "sekai"

    ```python
    import numpy as np
    import sekai

    vec = sekai.VecRecordEpisodeStatistics(
        sekai.make_vec("CartPole-v1", num_envs=4, vectorization_mode="sync")
    )
    actions = vec.batch_action_space
    actions.seed(0)
    result = vec.reset(seed=0)
    for _ in range(100):
        result = vec.step(actions.sample())
        for i in np.flatnonzero(result.dones):
            episode_return = result.infos[i]["episode"].episode_return
            last_obs = result.infos[i]["final_observation"]
            # result.observations[i] is already the first obs of the next episode
    vec.close()
    ```

=== "gymnasium"

    ```python
    import gymnasium as gym
    import numpy as np

    envs = gym.wrappers.vector.RecordEpisodeStatistics(
        gym.make_vec("CartPole-v1", num_envs=4, vectorization_mode="sync")
    )
    envs.action_space.seed(0)
    obs, infos = envs.reset(seed=0)
    for _ in range(100):
        obs, rewards, terminations, truncations, infos = envs.step(envs.action_space.sample())
        if "episode" in infos:
            for i in np.flatnonzero(infos["_episode"]):
                episode_return = infos["episode"]["r"][i]
        # next-step autoreset (the default): on the following step, a finished sub-env
        # returns its reset observation with reward 0 and ignores its action
    envs.close()
    ```

## Using sekai envs with gymnasium libraries

`sekai.compat.to_gymnasium(env)` wraps any sekai env (including a wrapper stack) as a
`gymnasium.Env`: spaces are converted, results become tuples, seeds pass through.

```python
import sekai
from sekai.compat import to_gymnasium

gym_env = to_gymnasium(sekai.make("Portfolio-v0"))
print(type(gym_env).__name__, gym_env.action_space)  # SekaiToGymnasium Box(0.0, 1.0, (6,), float32)
obs, info = gym_env.reset(seed=0)
print(sorted(obs))  # ['returns', 'weights']

from gymnasium.utils.env_checker import check_env as gym_check_env
gym_check_env(gym_env)  # passes
```

### Training with Stable-Baselines3

```bash
pip install "sekai[gymnasium]" stable-baselines3
```

PPO on CartPole, through the bridge. 50k steps take about 30 seconds on a laptop CPU and
reach a mean return near the maximum of 500:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import sekai
from sekai.compat import to_gymnasium

vec_env = make_vec_env(lambda: to_gymnasium(sekai.make("CartPole-v1")), n_envs=4, seed=0)
model = PPO("MlpPolicy", vec_env, verbose=0, seed=0)
model.learn(total_timesteps=50_000)

env = to_gymnasium(sekai.make("CartPole-v1"))
returns = []
for seed in range(10):
    obs, _ = env.reset(seed=seed)
    done, total = False, 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        done = terminated or truncated
    returns.append(total)
print(f"PPO mean return over 10 episodes: {sum(returns) / len(returns):.1f}")  # ~ PPO mean return over 10 episodes: 493.5
```

`Dict` observations, like `Portfolio-v0`'s, need SB3's `MultiInputPolicy`. SB3's own
`check_env` accepts every finance environment. For most of them it only adds advice, such as
using a symmetric `[-1, 1]` action space, which `RescaleAction` provides. `Market-v0` and
`LimitOrderBook-v0` are the exception: their [`MaskedBox`](dynamic.md) entry becomes a `Dict`
inside the observation `Dict`, and SB3 does not support nested observation spaces
(`check_env` warns about it, and `MultiInputPolicy` raises `NotImplementedError`). Flatten
those two first, as [shown below](#nested-observations-flatten-first).

```python
import numpy as np
from stable_baselines3.common.env_checker import check_env as sb3_check_env

portfolio = sekai.make("Portfolio-v0", n_assets=3, episode_length=63)
env = to_gymnasium(sekai.RescaleAction(portfolio, -1.0, 1.0))
sb3_check_env(env)  # passes (SB3 still suggests flattening the 2-D "returns" entry)

model = PPO("MultiInputPolicy", env, n_steps=512, batch_size=64, verbose=0, seed=0)
model.learn(total_timesteps=20_000)


def evaluate(policy, episodes=50):
    totals = []
    for seed in range(1000, 1000 + episodes):
        obs, _ = env.reset(seed=seed)
        done, total = False, 0.0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(policy(obs))
            total += reward
            done = terminated or truncated
        totals.append(total)
    return float(np.mean(totals))


ppo = evaluate(lambda obs: model.predict(obs, deterministic=True)[0])
equal_weight = evaluate(lambda obs: np.ones(4, dtype=np.float32))
print(f"PPO {ppo:+.4f}, equal weight {equal_weight:+.4f}")  # ~ PPO +0.0075, equal weight +0.0086
```

A 20k-step run is nowhere near converged; the closed-form and heuristic baselines on the
[finance page](../environments/finance.md) are the bar a learned policy has to clear.

### Nested observations: flatten first

For `Market-v0` and `LimitOrderBook-v0`, wrap the sekai env in `sekai.FlattenObservation`
before converting it, and train with `MlpPolicy`:

```python
market = to_gymnasium(sekai.FlattenObservation(sekai.make("Market-v0")))
print(type(market.observation_space).__name__)  # Box

model = PPO("MlpPolicy", market, verbose=0, seed=0)
model.learn(total_timesteps=2_048)
```

The flattened vector is the same one gymnasium's `FlattenObservation` builds from the
converted env, because `to_gymnasium()` keeps the `Dict` key order.

### Action masks

When the sekai env has an [`action_mask()` method](dynamic.md#action-masks), the gymnasium
env gets an `action_masks()` method returning the mask flattened to one boolean array, the
layout sb3-contrib's `MaskablePPO` expects. Wrappers forward `action_mask()`, so this still
works through `FlattenObservation`. `LimitOrderBook-v0` masks orders that would breach its
inventory limit (`pip install sb3-contrib`):

```python
from sb3_contrib import MaskablePPO

lob = to_gymnasium(sekai.FlattenObservation(sekai.make("LimitOrderBook-v0")))
lob.reset(seed=0)
print(lob.action_space, lob.action_masks().shape)  # MultiDiscrete([6 6 3]) (15,)

model = MaskablePPO("MlpPolicy", lob, verbose=0, seed=0)
model.learn(total_timesteps=2_048)  # every action it samples is allowed by the mask
```

### Rendering

Pass a [renderer](rendering.md) to get gymnasium's `render_mode="rgb_array"`, so tools like
`gymnasium.wrappers.RecordVideo` work:

```python
from sekai.envs.classic_control import CartPoleRenderer

gym_env = to_gymnasium(sekai.make("CartPole-v1"), renderer=CartPoleRenderer())
gym_env.reset(seed=0)
print(gym_env.render_mode, gym_env.render().shape)  # rgb_array (400, 600, 3)
print(gym_env.metadata)  # {'render_modes': ['rgb_array'], 'render_fps': 50}
```

For a text renderer pass `render_mode="ansi"` as well. Other modes raise `ValueError`,
`"human"` included: sekai renderers return frames instead of opening a window. To watch the
frames, wrap the converted env in `gymnasium.wrappers.HumanRendering` (needs pygame).

## Using gymnasium envs in sekai

`sekai.compat.from_gymnasium(env)` goes the other way, so any gymnasium environment (Atari,
MuJoCo, Box2D, your own) can use sekai's wrappers, vector envs, statistics and checker:

```python
import gymnasium as gym
import sekai
from sekai.compat import from_gymnasium

env = from_gymnasium(gym.make("Acrobot-v1"))
print(env.observation_space, env.action_space)  # Box([-1.0, -1.0, -1.0, -1.0, -12.566371, -28.274334], [1.0, 1.0, 1.0, 1.0, 12.566371, 28.274334], (6,), float32) Discrete(3)
print(sekai.check_env(env))  # []

vec = sekai.SyncVecEnv([lambda: from_gymnasium(gym.make("Acrobot-v1")) for _ in range(4)])
print(vec.reset(seed=0).observations.shape)  # (4, 6)
vec.close()
```

The gymnasium env keeps its own generator: seeds given to `reset()` go straight to it and
`env.rng` is its `np_random`. gymnasium only takes `int` seeds, so a NumPy integer seed is
passed on as an `int`, and `reset(seed=generator)` keeps sekai's meaning by making that
`Generator` the gymnasium env's `np_random`. To render such an env, create it with gymnasium's
`render_mode` and use `sekai.compat.GymnasiumRenderer`, which calls the gymnasium env's own
`render()`:

```python
from sekai.compat import GymnasiumRenderer

env = from_gymnasium(gym.make("CartPole-v1", render_mode="rgb_array"))
env.reset(seed=0)
print(GymnasiumRenderer().render(env).shape)  # (400, 600, 3)
```

`space_to_gymnasium()` and `space_from_gymnasium()` convert individual spaces. A `Dict`
keeps its key order in both directions (a gymnasium `Dict` built from a plain `dict` sorts
its keys), so flattened observations line up on both sides. A `Sequence` becomes gymnasium's
`Sequence`, which has no length bound, so `max_length` is dropped; a `MaskedBox` becomes a
gymnasium `Dict` of `values` and `mask`, and comes back as a plain `Dict`:

```python
from sekai.compat import space_to_gymnasium

print(space_to_gymnasium(sekai.Sequence(sekai.Box(0.0, 1.0, shape=(2,)), max_length=8)))  # Sequence(Box(0.0, 1.0, (2,), float32), stack=False)
print(space_to_gymnasium(sekai.MaskedBox(0.0, 1.0, shape=(2,), max_size=8)))  # Dict('values': Box(0.0, 1.0, (8, 2), float32), 'mask': MultiBinary((8,)))
```

## Multi-agent: PettingZoo

The same bridge exists for multi-agent environments: `sekai.compat.to_pettingzoo(env)` exposes
a `sekai.MultiAgentEnv` as a PettingZoo `ParallelEnv`, and `from_pettingzoo(parallel_env)`
wraps a PettingZoo parallel environment as a `MultiAgentEnv`. Install it with
`pip install "sekai[pettingzoo]"`; see [Multi-agent](multi-agent.md#pettingzoo).

## Differences that change behaviour

!!! warning "Vector autoreset is same-step"
    gymnasium ≥ 1.0 vector envs default to next-step autoreset. sekai resets a finished
    sub-env inside the same `step()`: `observations[i]` already belongs to the new episode and
    the terminal observation is in `infos[i]["final_observation"]`. Remove any code that skips
    the dummy step after an episode ends, and bootstrap truncated episodes from
    `final_observation`. See [Autoreset](vector.md#autoreset).

!!! warning "`vectorization_mode=\"async\"` means asyncio"
    In gymnasium, `"async"` starts subprocesses. In sekai, `"async"` is `AsyncVecEnv`
    (asyncio, one process) and `"proc"` is the subprocess backend.

!!! info "Results are dataclasses"
    `obs, reward, terminated, truncated, info = env.step(a)` raises `TypeError` with sekai,
    because `StepResult` is not iterable. Use the fields, or go through `to_gymnasium()`
    when a library needs tuples.

!!! info "No render mode on the env"
    `sekai.make()` takes no `render_mode`; any keyword argument goes to the environment's
    constructor. Create a renderer instead.

!!! info "Seeding details"
    Seeding an env works as in gymnasium (`reset(seed=s)` then unseeded `reset()`s continue the
    stream). Vector envs differ: an int seed is expanded into independent child seeds rather
    than `seed + i`. Pass a list for explicit per-env seeds.

!!! info "Vector infos are lists"
    `infos` is a `list` with one `dict` per sub-env, not a dict of arrays with `_key` masks.
    `sekai.stack_infos(infos)` gives a dict of arrays when you need one.
