<div align="center">

# sekai · 世界

**A modern, flexible reinforcement learning environment framework**

*Built for high-dimensional, complex environments — especially financial markets*

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-green)](sekai/_version.py)

</div>

---

sekai is a Python RL environment framework designed as a modern successor to [gym](https://github.com/openai/gym) / [gymnasium](https://github.com/Farama-Foundation/Gymnasium). It provides a clean, fully-typed API with structured return types, first-class async support, multi-agent environments, and a composable wrapper system — all built to scale to the complexity of real financial market simulation.

```python
import sekai
import numpy as np

class TradingEnv(sekai.Env):
    def __init__(self):
        self.observation_space = sekai.Box(-np.inf, np.inf, shape=(64,))
        self.action_space = sekai.Box(-1.0, 1.0, shape=(8,))

    def reset(self, *, seed=None, options=None):
        self._set_rng(seed)
        obs = self.rng.standard_normal(64).astype(np.float32)
        return sekai.ResetResult(observation=obs, info={})

    def step(self, action):
        obs = self.rng.standard_normal(64).astype(np.float32)
        reward = float(-np.sum(np.square(action)))
        return sekai.StepResult(obs, reward, terminated=False, truncated=False, info={})

with TradingEnv() as env:
    result = env.reset(seed=42)
    for _ in range(1000):
        result = env.step(env.action_space.sample())
        if result.done:
            break
```

---

## Why sekai?

gym and gymnasium have a well-known set of friction points that compound at scale:

| Pain point | gymnasium | sekai |
|---|---|---|
| Return types | `obs, rew, term, trunc, info = env.step(a)` — easy to unpack wrong | `StepResult(observation, reward, terminated, truncated, info)` frozen dataclass |
| Async environments | Not supported | `async_step` / `async_reset` on every environment by default |
| Multi-agent | Delegated to PettingZoo | `MultiAgentEnv` with per-agent typed spaces built in |
| Rendering | `render()` on Env, mode set at construction | `Renderer` protocol — injected, composable, swappable |
| Episode statistics | Untyped `info["r"]` dict | `EpisodeStats` typed dataclass with `.episode_return`, `.episode_length`, etc. |
| Vectorised env naming | `AsyncVectorEnv` = multiprocessing (confusing) | `SyncVecEnv`, `AsyncVecEnv` (asyncio), `ProcVecEnv` (multiprocessing) |
| Registry plugins | Not supported | `register_namespace("qc", loader)` for lazy plugin namespaces |
| `check_env` overhead | Runs on every step in production | Testing utility — call it in your test suite, zero prod overhead |

---

## Installation

```bash
pip install sekai
```

**Development install:**

```bash
git clone https://github.com/qiancapital/sekai
cd sekai
pip install -e ".[dev]"
```

**Requirements:** Python 3.10+, NumPy 1.24+

---

## Table of Contents

- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
  - [StepResult and ResetResult](#stepresult-and-resetresult)
  - [Writing an Environment](#writing-an-environment)
  - [Spaces](#spaces)
  - [Wrappers](#wrappers)
  - [Vectorised Environments](#vectorised-environments)
  - [Multi-Agent Environments](#multi-agent-environments)
  - [Environment Registry](#environment-registry)
  - [Episode Statistics](#episode-statistics)
  - [Rendering](#rendering)
  - [Async Environments](#async-environments)
- [Financial Market Example](#financial-market-example)
- [Validating Your Environment](#validating-your-environment)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [License](#license)

---

## Quick Start

```python
import sekai
import numpy as np

# 1. Implement an environment
class CartPoleEnv(sekai.Env):
    def __init__(self):
        self.observation_space = sekai.Box(-4.0, 4.0, shape=(4,), dtype=np.float32)
        self.action_space = sekai.Discrete(2)
        self._state = np.zeros(4, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self._set_rng(seed)
        self._state = self.rng.uniform(-0.05, 0.05, size=(4,)).astype(np.float32)
        return sekai.ResetResult(observation=self._state.copy(), info={})

    def step(self, action):
        # ... physics update ...
        obs = self._state.copy()
        reward = 1.0
        terminated = bool(np.any(np.abs(self._state) > 3.0))
        return sekai.StepResult(obs, reward, terminated, truncated=False, info={})

# 2. Run it
env = CartPoleEnv()
result = env.reset(seed=0)
print(result.observation)  # array([-0.043, 0.044, ...], dtype=float32)

for _ in range(200):
    action = env.action_space.sample()
    result = env.step(action)
    if result.done:          # result.terminated or result.truncated
        env.reset()

# 3. Add wrappers
env = sekai.TimeLimit(CartPoleEnv(), max_episode_steps=500)
env = sekai.RecordEpisodeStatistics(env)
env = sekai.NormalizeObservation(env)

# 4. Vectorise
vec = sekai.SyncVecEnv([CartPoleEnv] * 8)
batch = vec.reset(seed=0)
print(batch.observations.shape)  # (8, 4)
```

---

## Core Concepts

### StepResult and ResetResult

The most visible departure from gymnasium. sekai returns frozen dataclasses instead of tuples.

```python
# gymnasium (error-prone unpacking)
obs, reward, terminated, truncated, info = env.step(action)

# sekai
result = env.step(action)
result.observation   # typed
result.reward        # float
result.terminated    # bool
result.truncated     # bool
result.info          # dict[str, Any]
result.done          # computed: terminated or truncated
```

```python
# gymnasium
obs, info = env.reset()

# sekai
result = env.reset()
result.observation
result.info
```

Both `StepResult` and `ResetResult` are `frozen=True` value objects — safe to cache, compare, and pass across threads.

---

### Writing an Environment

Subclass `sekai.Env` and implement `reset()` and `step()`.

```python
import sekai
import numpy as np

class MyEnv(sekai.Env[np.ndarray, np.ndarray]):
    """Type params: Env[ObsType, ActType]"""

    metadata = {"description": "A custom sekai environment"}

    def __init__(self, n_assets: int = 10):
        self.n_assets = n_assets
        self.observation_space = sekai.Box(
            low=-np.inf, high=np.inf, shape=(n_assets * 5,), dtype=np.float32
        )
        self.action_space = sekai.Box(
            low=-1.0, high=1.0, shape=(n_assets,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        # Always call _set_rng to handle deterministic seeding
        self._set_rng(seed)
        obs = self.rng.standard_normal(self.n_assets * 5).astype(np.float32)
        return sekai.ResetResult(observation=obs, info={"episode_start": True})

    def step(self, action: np.ndarray) -> sekai.StepResult[np.ndarray]:
        obs = self.rng.standard_normal(self.n_assets * 5).astype(np.float32)
        pnl = float(np.dot(action, self.rng.standard_normal(self.n_assets)))
        terminated = False
        truncated = False
        return sekai.StepResult(obs, pnl, terminated, truncated, info={"pnl": pnl})

    def close(self) -> None:
        pass  # release any data feeds, file handles, etc.
```

**Use as a context manager** to ensure `close()` is always called:

```python
with MyEnv(n_assets=20) as env:
    result = env.reset(seed=42)
    for _ in range(1000):
        result = env.step(env.action_space.sample())
```

**Seeding:** `_set_rng(seed)` stores `self.rng` (a `numpy.random.Generator`) and `self._rng_seed`. Call it in `reset()` for reproducible episodes.

---

### Spaces

sekai ships the same space types as gymnasium with improved generics and JSON serialisation.

#### Box

Continuous n-dimensional space. Supports bounded, semi-bounded, and unbounded dimensions.

```python
# Bounded
price_space = sekai.Box(low=0.0, high=1000.0, shape=(8,), dtype=np.float32)

# Partially unbounded (sampled from exponential/normal distributions)
returns_space = sekai.Box(low=-np.inf, high=np.inf, shape=(50,), dtype=np.float64)

# Integer box (market depth levels)
depth_space = sekai.Box(low=0, high=10_000, shape=(10, 2), dtype=np.int32)

sample = price_space.sample()       # np.ndarray shape (8,)
print(sample in price_space)        # True
print(price_space.is_bounded)       # (True, True)
```

#### Discrete

Discrete values `{start, start+1, ..., start+n-1}`.

```python
action_space = sekai.Discrete(n=5)       # {0, 1, 2, 3, 4}
shifted = sekai.Discrete(n=5, start=-2)  # {-2, -1, 0, 1, 2}

# Masked sampling (useful for invalid action masking)
mask = np.array([1, 0, 1, 0, 1], dtype=np.int8)  # only 0, 2, 4 are valid
action = action_space.sample(mask=mask)
```

#### MultiDiscrete

Multiple independent discrete dimensions. Common for multi-asset order type selection.

```python
# 3 assets, each can hold position -1/0/1
order_space = sekai.MultiDiscrete(nvec=[3, 3, 3], start=[-1, -1, -1])
```

#### Dict

Heterogeneous named sub-spaces. Ideal for rich financial observations.

```python
obs_space = sekai.Dict({
    "prices":     sekai.Box(-np.inf, np.inf, shape=(100, 8)),   # OHLCV + 3 features
    "portfolio":  sekai.Box(-1.0, 1.0, shape=(10,)),            # current positions
    "order_book": sekai.Box(0, np.inf, shape=(20, 2)),          # bid/ask levels
    "regime":     sekai.Discrete(4),                            # market regime label
})

sample = obs_space.sample()
print(sample["prices"].shape)    # (100, 8)
print(sample["regime"])          # np.intp in {0,1,2,3}
```

#### Tuple

Ordered heterogeneous sub-spaces.

```python
obs_space = sekai.Tuple([
    sekai.Box(-np.inf, np.inf, shape=(50,)),  # technical features
    sekai.MultiBinary(10),                    # binary signals
])
```

#### Space utilities

```python
from sekai.spaces import flatdim, flatten, unflatten, flatten_space

# How many floats does this space flatten to?
dim = flatdim(obs_space)

# Flatten a sample to a 1D float32 array
flat = flatten(obs_space, sample)       # shape (dim,)

# Reconstruct the original structure
restored = unflatten(obs_space, flat)

# Get an equivalent Box space
box = flatten_space(obs_space)          # Box(-inf, inf, (dim,))
```

---

### Wrappers

Wrappers transparently modify an environment's behaviour. They compose by layering, and the full stack is introspectable.

```python
env = MyEnv()
env = sekai.TimeLimit(env, max_episode_steps=252)      # truncate at 1 trading year
env = sekai.RecordEpisodeStatistics(env)               # track episode return/length
env = sekai.NormalizeObservation(env)                  # running mean/var normalisation
env = sekai.NormalizeReward(env, gamma=0.99)           # reward scaling

# Inspect the stack
for layer in env:
    print(layer)
# <RecordEpisodeStatistics(<TimeLimit(<MyEnv>)>)>

# Reach the base env
base = env.unwrapped
```

#### Built-in wrappers

| Wrapper | Description |
|---|---|
| `TimeLimit(env, max_episode_steps)` | Truncates episodes after N steps |
| `AutoReset(env)` | Automatically resets on episode end; stores final obs in `info["final_observation"]` |
| `RecordEpisodeStatistics(env)` | Adds `info["episode"]` as a typed `EpisodeStats` on termination |
| `NormalizeObservation(env)` | Welford running mean/variance normalisation |
| `NormalizeReward(env, gamma)` | Discounted return variance scaling |
| `FlattenObservation(env)` | Flattens any space to a 1D `Box` |
| `FrameStackObservation(env, n)` | Stacks last N observations along a new leading axis |
| `ClipAction(env)` | Clips continuous actions to the action space bounds |
| `RescaleAction(env, min, max)` | Maps `[min, max]` → env's action range |
| `TransformObservation(env, fn)` | Applies an arbitrary callable to observations |
| `TransformReward(env, fn)` | Applies an arbitrary callable to rewards |
| `OrderEnforcing(env)` | Raises `ResetRequired` if `step()` is called before `reset()` |

#### Writing a custom wrapper

```python
class LogReturnObservation(sekai.ObservationWrapper):
    """Converts price observations to log returns."""

    def __init__(self, env):
        super().__init__(env)
        # Update observation space to match transformed output
        self.observation_space = sekai.Box(
            -np.inf, np.inf,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return np.log1p(obs).astype(np.float32)


class SharpeReward(sekai.RewardWrapper):
    """Replaces raw PnL reward with a rolling Sharpe estimate."""

    def __init__(self, env, window: int = 20):
        super().__init__(env)
        self._returns: list[float] = []
        self._window = window

    def reward(self, reward: float) -> float:
        self._returns.append(reward)
        window = self._returns[-self._window:]
        if len(window) < 2:
            return 0.0
        mean = sum(window) / len(window)
        std = float(np.std(window)) + 1e-8
        return mean / std


class ScaledAction(sekai.ActionWrapper):
    """Scales [-1, 1] policy outputs to actual notional sizes."""

    def action(self, action: np.ndarray) -> np.ndarray:
        return (action * 1_000_000).astype(np.float32)

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        return (action / 1_000_000).astype(np.float32)
```

---

### Vectorised Environments

Run N independent environments in parallel and receive batched results.

sekai ships three backends — choose based on your workload:

| Backend | When to use |
|---|---|
| `SyncVecEnv` | Fast envs where parallelism overhead would dominate |
| `AsyncVecEnv` | I/O-bound envs (live feeds, REST APIs, async data sources) |
| `ProcVecEnv` | CPU-bound simulators that release the GIL |

```python
# SyncVecEnv — simplest, single process
vec = sekai.SyncVecEnv([MyEnv] * 16)

# or use lambdas for parameterised envs
vec = sekai.SyncVecEnv([lambda: MyEnv(n_assets=i) for i in range(4)])

# Reset all environments
batch = vec.reset(seed=0)
print(batch.observations.shape)  # (16, obs_dim)
print(batch.infos)               # list[dict] — one per sub-env

# Step all environments
actions = np.stack([vec.action_space.sample() for _ in range(16)])
batch = vec.step(actions)
print(batch.rewards.shape)       # (16,)
print(batch.dones.shape)         # (16,)  — terminated | truncated

vec.close()
```

**VecEnv results are structured dataclasses** — `VecStepResult` and `VecResetResult`:

```python
result = vec.step(actions)
result.observations   # (N, *obs_shape)
result.rewards        # (N,)  float64
result.terminated     # (N,)  bool
result.truncated      # (N,)  bool
result.dones          # (N,)  terminated | truncated
result.infos          # list[dict]  — one per sub-env

# Stack info values into numpy arrays when needed
from sekai.vector import stack_infos
stacked = stack_infos(result.infos)  # {key: np.array([v0, v1, ...])}
```

**VecEnv wrappers:**

```python
from sekai.vector import VecObservationWrapper

class NormaliseVec(VecObservationWrapper):
    def observation(self, obs: np.ndarray) -> np.ndarray:
        return (obs - obs.mean(axis=0)) / (obs.std(axis=0) + 1e-8)
```

**Accessing sub-env attributes:**

```python
vec.get_attr("n_assets")            # [10, 10, 10, ...]
vec.set_attr("n_assets", 20)        # set all to 20
vec.call("some_method", arg=True)   # call a method on each sub-env
```

---

### Multi-Agent Environments

sekai provides `MultiAgentEnv` for heterogeneous multi-agent setups without relying on PettingZoo. Each agent gets its own typed observation and action space.

```python
import sekai
from sekai.core.multi_agent import MultiAgentEnv
from sekai.core.result import MAStepResult, MAResetResult
import numpy as np

class MarketMicrostructureEnv(MultiAgentEnv):
    """Two agents: a market maker and a directional trader."""

    def __init__(self):
        self.possible_agents = ["market_maker", "trader"]
        self.agents = list(self.possible_agents)

        self.observation_spaces = {
            "market_maker": sekai.Dict({
                "order_book": sekai.Box(0, np.inf, shape=(40,)),
                "inventory":  sekai.Box(-100, 100, shape=(1,)),
            }),
            "trader": sekai.Box(-np.inf, np.inf, shape=(20,)),
        }
        self.action_spaces = {
            "market_maker": sekai.Box(-1.0, 1.0, shape=(4,)),  # bid/ask spread + size
            "trader":       sekai.Discrete(3),                  # buy / hold / sell
        }

    def reset(self, *, seed=None, options=None):
        self.agents = list(self.possible_agents)
        obs = {agent: self.observation_spaces[agent].sample() for agent in self.agents}
        return MAResetResult(observations=obs, info={a: {} for a in self.agents})

    def step(self, actions):
        obs = {agent: self.observation_spaces[agent].sample() for agent in self.agents}
        rewards = {"market_maker": 0.01, "trader": -0.005}
        terminated = {agent: False for agent in self.agents}
        truncated = {agent: False for agent in self.agents}
        return MAStepResult(obs, rewards, terminated, truncated, info={a: {} for a in self.agents})


env = MarketMicrostructureEnv()
reset_result = env.reset(seed=0)

actions = {
    "market_maker": env.action_spaces["market_maker"].sample(),
    "trader": env.action_spaces["trader"].sample(),
}
result = env.step(actions)
print(result.rewards)     # {"market_maker": 0.01, "trader": -0.005}
print(result.all_done)    # False
print(result.any_done)    # False
```

---

### Environment Registry

Register environments by ID and instantiate them by name — the same pattern as `gym.make()`, with lazy namespace loading for plugins.

```python
# Register
sekai.register(
    id="qc/PortfolioEnv-v1",
    entry_point="mypackage.envs:PortfolioEnv",
    max_episode_steps=252,
    kwargs={"n_assets": 50},
    reward_threshold=0.25,
)

# Instantiate
env = sekai.make("qc/PortfolioEnv-v1")                     # uses registered kwargs
env = sekai.make("qc/PortfolioEnv-v1", n_assets=100)       # override kwargs

# Inspect the spec
s = sekai.spec("qc/PortfolioEnv-v1")
print(s.id, s.max_episode_steps, s.reward_threshold)
```

**Namespace plugins** — register an entire namespace lazily. The loader is called once, the first time any env in that namespace is requested:

```python
# In your plugin package's __init__.py
import sekai

sekai.register_namespace(
    "qc",
    lambda: __import__("qiancapital.envs"),  # deferred import
)

# Elsewhere — no startup cost until first make() call
env = sekai.make("qc/CryptoOrderBook-v2")
```

**List all registered environments:**

```python
for s in sekai.all_specs():
    print(s.id, s.max_episode_steps)
```

---

### Episode Statistics

`RecordEpisodeStatistics` records per-episode metrics via a pluggable `StatsTracker`. On episode end, `info["episode"]` is a typed `EpisodeStats` dataclass — not an untyped dict.

```python
env = sekai.RecordEpisodeStatistics(MyEnv())
result = env.reset(seed=0)

for _ in range(10_000):
    result = env.step(env.action_space.sample())
    if result.done:
        ep = result.info["episode"]      # EpisodeStats — fully typed
        print(ep.episode_return)         # float
        print(ep.episode_length)         # int
        print(ep.elapsed_time)           # float (seconds)
        print(ep.terminated)             # bool
```

**Custom tracker** — plug in your own monitoring backend:

```python
from sekai.stats.tracker import StatsTracker, EpisodeStats

class WandbTracker:
    def on_reset(self, env_id, seed): ...
    def on_step(self, reward, terminated, truncated, info): ...
    def on_episode_end(self, stats: EpisodeStats):
        import wandb
        wandb.log({
            "episode_return": stats.episode_return,
            "episode_length": stats.episode_length,
        })
    def summary(self): return {}

env = sekai.RecordEpisodeStatistics(MyEnv(), tracker=WandbTracker())
```

The `InMemoryTracker` (default) stores a rolling window of episodes and exposes `.mean_return()` and `.mean_length()`. `LoggingTracker` emits structured Python log records.

---

### Rendering

In sekai, rendering is **not** part of `Env`. There is no `render()` method and no `render_mode` string. Instead, `Renderer` objects are injected independently — you can swap, compose, or record without touching the environment.

```python
from sekai.rendering import Renderer

class MatplotlibRenderer:
    def render(self, env) -> np.ndarray:
        # Access env state directly and draw it
        state = env.unwrapped._state
        # ... draw to figure, return RGB array ...
        return frame  # np.ndarray uint8 (H, W, 3)

    def close(self):
        import matplotlib.pyplot as plt
        plt.close("all")

env = MyEnv()
renderer = MatplotlibRenderer()
result = env.reset(seed=0)

frames = []
with renderer:
    for _ in range(500):
        result = env.step(policy(result.observation))
        frames.append(renderer.render(env))

# frames is a list of RGB arrays — save as video, GIF, etc.
```

This separation means you can attach multiple renderers simultaneously, record video for the first 100 steps then switch to a live display, or completely skip rendering in headless training without any conditional logic in your env.

---

### Async Environments

Every sekai environment has `async_step` and `async_reset` built in. Sync envs get them for free (the defaults just call the sync methods). Environments backed by live data feeds can override them for true async operation.

```python
import asyncio

class LiveFeedEnv(sekai.Env):
    async def async_reset(self, *, seed=None, options=None):
        data = await self._feed.connect()
        obs = self._process(data)
        return sekai.ResetResult(observation=obs, info={})

    async def async_step(self, action):
        await self._feed.send_order(action)
        data = await self._feed.next_tick()
        obs = self._process(data)
        return sekai.StepResult(obs, reward=0.0, terminated=False, truncated=False, info={})

# Run multiple live feeds concurrently with AsyncVecEnv
async def main():
    vec = sekai.AsyncVecEnv([LiveFeedEnv] * 4)
    batch = await vec.async_reset(seed=0)
    actions = np.stack([vec.action_space.sample() for _ in range(4)])
    batch = await vec.async_step(actions)

asyncio.run(main())
```

---

## Financial Market Example

A realistic portfolio management environment showing sekai's capabilities for financial RL:

```python
import sekai
import numpy as np
from sekai.core.result import StepResult, ResetResult

class PortfolioEnv(sekai.Env):
    """
    Multi-asset portfolio management environment.

    Observation: (lookback, n_assets, n_features) price/volume history
    Action:      (n_assets,) target portfolio weights in [-1, 1]
    Reward:      portfolio return minus transaction costs
    """

    metadata = {"description": "Qian Capital portfolio environment"}

    def __init__(
        self,
        n_assets: int = 20,
        lookback: int = 60,
        n_features: int = 5,         # OHLCV
        transaction_cost: float = 1e-4,
        episode_length: int = 252,
    ):
        self.n_assets = n_assets
        self.lookback = lookback
        self.n_features = n_features
        self.transaction_cost = transaction_cost
        self.episode_length = episode_length

        self.observation_space = sekai.Dict({
            "prices": sekai.Box(
                low=-np.inf, high=np.inf,
                shape=(lookback, n_assets, n_features),
                dtype=np.float32,
            ),
            "holdings": sekai.Box(
                low=-1.0, high=1.0,
                shape=(n_assets,),
                dtype=np.float32,
            ),
        })
        self.action_space = sekai.Box(
            low=-1.0, high=1.0,
            shape=(n_assets,),
            dtype=np.float32,
        )

        self._holdings = np.zeros(n_assets, dtype=np.float32)
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        self._set_rng(seed)
        self._holdings = np.zeros(self.n_assets, dtype=np.float32)
        self._step_count = 0
        obs = self._get_obs()
        return ResetResult(observation=obs, info={"holdings": self._holdings.copy()})

    def step(self, action: np.ndarray) -> StepResult:
        # Normalise weights to sum to 1
        weights = np.clip(action, -1, 1)
        weights = weights / (np.abs(weights).sum() + 1e-8)

        # Simulated returns
        returns = self.rng.standard_normal(self.n_assets) * 0.01

        # Portfolio PnL
        pnl = float(np.dot(weights, returns))

        # Transaction costs
        turnover = float(np.sum(np.abs(weights - self._holdings)))
        cost = turnover * self.transaction_cost

        self._holdings = weights.copy()
        self._step_count += 1

        obs = self._get_obs()
        reward = pnl - cost
        terminated = False
        truncated = self._step_count >= self.episode_length

        return StepResult(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={"pnl": pnl, "cost": cost, "turnover": turnover},
        )

    def _get_obs(self) -> dict:
        prices = self.rng.standard_normal(
            (self.lookback, self.n_assets, self.n_features)
        ).astype(np.float32)
        return {"prices": prices, "holdings": self._holdings.copy()}


# Build a training pipeline with wrappers and vectorisation
def make_env(seed_offset: int = 0):
    env = PortfolioEnv(n_assets=20, lookback=60, episode_length=252)
    env = sekai.RecordEpisodeStatistics(env)
    return env

vec = sekai.SyncVecEnv([lambda i=i: make_env(i) for i in range(8)])
sekai.register("qc/Portfolio-v1", PortfolioEnv, max_episode_steps=252)

# Validate the environment
issues = sekai.check_env(PortfolioEnv(), n_steps=300)
assert not issues, issues
```

---

## Validating Your Environment

`check_env` is a testing utility — run it in your test suite, not in production:

```python
from sekai.utils.checker import check_env

issues = check_env(
    MyEnv(),
    n_steps=500,
    warn_on_anomaly=True,
    check_obs_space=True,
    check_action_space=True,
    check_reset_seed=True,
)

# In pytest
def test_my_env():
    issues = check_env(MyEnv(), n_steps=200)
    assert not issues, "\n".join(issues)
```

Checks performed:
- `observation_space` and `action_space` are defined
- `action_space.sample()` is contained in `action_space`
- `reset()` returns a valid `ResetResult` with a valid observation
- `step()` returns a valid `StepResult` at each of N steps
- Reward, terminated, truncated, and info are correct types
- Observations are within `observation_space` (with warnings)
- `reset(seed=X)` is deterministic (same obs on two calls)
- No NaN or infinite rewards

---

## API Reference

### `sekai.Env[ObsType, ActType]`

| Member | Description |
|---|---|
| `observation_space: Space[ObsType]` | Must be set in `__init__` |
| `action_space: Space[ActType]` | Must be set in `__init__` |
| `reset(*, seed, options) -> ResetResult` | **Abstract.** Call `_set_rng(seed)` inside. |
| `step(action) -> StepResult` | **Abstract.** |
| `async_reset(...) -> ResetResult` | Default wraps `reset()`. Override for true async. |
| `async_step(action) -> StepResult` | Default wraps `step()`. Override for true async. |
| `close()` | Release resources. |
| `rng: np.random.Generator` | Lazily-initialised RNG. |
| `_set_rng(seed) -> int` | Seed the RNG. Returns actual seed used. |
| `unwrapped: Env` | Base env underneath any wrappers. |
| `metadata: ClassVar[dict]` | Optional environment metadata. |
| `spec: EnvSpec \| None` | Set automatically by `sekai.make()`. |

### `sekai.StepResult`

```python
@dataclass(frozen=True, slots=True)
class StepResult(Generic[ObsType]):
    observation: ObsType
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    @property
    def done(self) -> bool: ...   # terminated or truncated
```

### `sekai.ResetResult`

```python
@dataclass(frozen=True, slots=True)
class ResetResult(Generic[ObsType]):
    observation: ObsType
    info: dict[str, Any]
```

### Spaces

| Space | Sample type | Key params |
|---|---|---|
| `Box(low, high, shape, dtype)` | `np.ndarray` | `is_bounded`, `low`, `high` |
| `Discrete(n, start)` | `np.intp` | `n`, `start` |
| `MultiDiscrete(nvec, start)` | `np.ndarray[intp]` | `nvec`, `start` |
| `MultiBinary(n)` | `np.ndarray[int8]` | `n` (int or shape tuple) |
| `Dict(spaces)` | `dict[str, Any]` | `spaces: OrderedDict` |
| `Tuple(spaces)` | `tuple[Any, ...]` | `spaces: tuple` |

All spaces implement:
- `sample(mask=None) -> T`
- `contains(x) -> bool` / `x in space`
- `seed(seed) -> int`
- `is_flattenable: bool`
- `to_jsonable() / from_jsonable()`

---

## Contributing

sekai is developed by [Qian Capital](https://github.com/qiancapital) and intended for open source release. Contributions are welcome.

```bash
git clone https://github.com/qiancapital/sekai
cd sekai
pip install -e ".[dev]"

# Run tests
pytest tests/

# Type check
mypy sekai/

# Lint
ruff check sekai/
```

**Adding a new environment:** implement `sekai.Env`, call `check_env` in your tests, and optionally register it with `sekai.register()`.

**Adding a new space:** subclass `sekai.spaces.Space` and register flatten/unflatten implementations using the `singledispatch` hooks in `sekai.spaces.utils`.

**Adding a new backend:** implement the `BackendOps` protocol in `sekai/backend/` and pass it to spaces/vec-envs.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

<div align="center">
<sub>sekai · 世界 — Qian Capital</sub>
</div>
