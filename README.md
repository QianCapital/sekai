<div align="center">

# sekai · 世界

**A modern, flexible reinforcement learning environment framework**

*Built for high-dimensional, complex environments, especially financial markets*

[![PyPI](https://img.shields.io/pypi/v/sekai)](https://pypi.org/project/sekai/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://pypi.org/project/sekai/)
[![CI](https://github.com/QianCapital/sekai/actions/workflows/ci.yml/badge.svg)](https://github.com/QianCapital/sekai/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-sekai.qiancapital.com-blue)](https://sekai.qiancapital.com/)
[![Typed](https://img.shields.io/badge/typing-mypy%20strict-blue)](https://mypy.readthedocs.io/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/QianCapital/sekai/blob/master/LICENSE)

[Documentation](https://sekai.qiancapital.com/) ·
[Environments](https://sekai.qiancapital.com/environments/) ·
[Migrating from gymnasium](https://sekai.qiancapital.com/guide/gymnasium/) ·
[Changelog](https://github.com/QianCapital/sekai/blob/master/CHANGELOG.md)

</div>

---

sekai is a Python framework for reinforcement learning environments, designed as a modern
successor to [gym](https://github.com/openai/gym) / [gymnasium](https://github.com/Farama-Foundation/Gymnasium).
It keeps the API you know (`make`, `reset`, `step`, spaces, wrappers, vector envs) and fixes
what gets painful at scale: structured return types instead of tuples, async-native
environments, built-in multi-agent support, typed episode statistics, and rendering that
lives outside the environment. It ships with gymnasium-identical classic control benchmarks
and a suite of financial-market environments, and it plugs straight into the gymnasium
ecosystem (Stable-Baselines3, CleanRL, RLlib).

```python
import sekai

env = sekai.make("CartPole-v1")
result = env.reset(seed=42)
while not result.done:
    result = env.step(env.action_space.sample())
    # result.observation, result.reward, result.terminated, result.truncated, result.info
```

## Highlights

- **Structured results.** `step()` returns a frozen `StepResult` dataclass, not a 5-tuple you can unpack in the wrong order.
- **Dynamic dimensionality.** `MaskedBox` padded sets and `Sequence` spaces for observations whose size changes (a universe that gains and loses assets, the trades since the last decision), plus a standard action-mask convention that `check_env`, wrappers and MaskablePPO understand.
- **Time-aware.** Frame stacking and normalisation over any (nested) space, exponentially weighted statistics for non-stationary data, observation and order latency wrappers, and `timestamp` / `dt` in every finance env's info.
- **Fast vectorisation.** Native vector environments that step thousands of sub-envs with array operations (over a million env-steps per second), plus sync, asyncio and multiprocessing backends with shared-memory observations and autoreset.
- **A finance toolkit.** Point-in-time `MarketData` with walk-forward splits and a look-ahead detector, market simulators (stochastic volatility, jumps, regimes, listings), cost and impact models, account rules, reward functions, performance metrics and a limit order book, which the built-in environments are assembled from, and yours can be too.
- **Built-in environments.** Classic control with dynamics bit-for-bit identical to gymnasium; trading, portfolio, dynamic-universe, market-making, optimal-execution and order-book environments (single- and multi-agent), with closed-form baselines.
- **Renderers for every environment, in NumPy alone.** The finance ones draw each episode as a dashboard: prices with positions and trades, equity, allocations, order-book depth with your orders in the queue. They are built from charts (`sekai.rendering.Dashboard`, `sekai.finance.rendering`) that render your own environments as well.
- **Async-native.** Every environment has `async_step()` / `async_reset()`; `AsyncVecEnv` runs I/O-bound envs (live feeds, remote simulators) concurrently.
- **Ecosystem-compatible.** `sekai.compat` converts to and from gymnasium (Stable-Baselines3, CleanRL, RLlib) and PettingZoo.
- **Production quality.** Fully typed (`mypy --strict`), tested on Python 3.10–3.14 on Linux and on macOS and Windows, with `check_env` and a `sekai` CLI for validating and benchmarking environments.

## Installation

```bash
pip install sekai
```

| Extra | Adds |
|---|---|
| `pip install "sekai[gymnasium]"` | gymnasium interop (`sekai.compat`) |
| `pip install "sekai[pettingzoo]"` | PettingZoo interop for multi-agent envs |
| `pip install "sekai[pandas]"` | building `MarketData` from pandas DataFrames |
| `pip install "sekai[render]"` | saving frames as GIF / MP4 |
| `pip install "sekai[all]"` | all of the above |

sekai itself needs only NumPy and cloudpickle.

## Quick start

### Use a built-in environment

```python
import sekai

env = sekai.RecordEpisodeStatistics(sekai.make("CartPole-v1"))
result = env.reset(seed=0)
while not result.done:
    result = env.step(env.action_space.sample())

stats = result.info["episode"]  # a typed EpisodeStats, not an untyped dict
print(stats.episode_return, stats.episode_length)
```

### Write your own

```python
import numpy as np
import sekai


class RandomWalkEnv(sekai.Env[np.ndarray, np.ndarray]):
    def __init__(self):
        self.observation_space = sekai.Box(-np.inf, np.inf, shape=(1,))
        self.action_space = sekai.Box(-1.0, 1.0, shape=(1,))

    def reset(self, *, seed=None, options=None):
        self._set_rng(seed)  # seeds self.rng; seed=None continues the current stream
        self.position = np.zeros(1, dtype=np.float32)
        return sekai.ResetResult(observation=self.position.copy(), info={})

    def step(self, action):
        self.position += action + self.rng.normal(0, 0.1, size=1).astype(np.float32)
        reward = -float(abs(self.position[0]))
        return sekai.StepResult(self.position.copy(), reward, False, False, info={})


assert sekai.check_env(RandomWalkEnv()) == []  # validates the API, spaces, seeding, async path
sekai.register("RandomWalk-v0", RandomWalkEnv, max_episode_steps=100)
env = sekai.make("RandomWalk-v0")
```

### Wrap and vectorise

```python
import sekai

vec = sekai.make_vec("Pendulum-v1", num_envs=8)  # also "async" and "proc" backends
vec = sekai.VecNormalizeObservation(sekai.VecRecordEpisodeStatistics(vec))

batch = vec.reset(seed=0)
for _ in range(500):
    batch = vec.step(vec.batch_action_space.sample())
    # finished sub-envs reset automatically; see batch.infos[i]["final_observation"]
print(batch.observations.shape)  # (8, 3)
```

### Trade a changing universe

```python
import numpy as np
import sekai
from sekai.finance import (
    GBM, ListingProcess, MarketData, RecordPerformance, SpreadCost, SquareRootImpact,
)

# closes: (T, N) prices with NaN while a stock is not listed; volumes: (T, N) shares traded.
# Simulated here; pass your own history the same way.
rng = np.random.default_rng(0)
sim = ListingProcess(GBM(mu=0.05, sigma=rng.uniform(0.15, 0.45, 80))).simulate(rng, 2519)
closes, volumes = sim.prices, rng.lognormal(12, 1, sim.prices.shape)
dates = np.busday_offset("2015-01-01", np.arange(2520), roll="forward")  # trading days
tickers = [f"STK{i:02d}" for i in range(80)]

data = MarketData(closes, timestamps=dates, assets=tickers, fields={"volume": volumes})
train, test = data.split(0.8, embargo=5, history=20)  # chronological, no leakage

env = RecordPerformance(sekai.make(
    "Market-v0", data=train, max_assets=50, lookback=20,
    transaction_cost=SpreadCost(half_spread=5e-4) + SquareRootImpact(0.3),
))
result = env.reset(seed=0)
while not result.done:
    mask = env.unwrapped.action_mask()               # only listed assets can be traded
    result = env.step(env.action_space.sample(mask=mask))
print(result.info["performance"].sharpe_ratio)
```

The observation is a padded set of at most 50 assets (`MaskedBox`) with a validity mask, so
it batches, normalises and feeds a set encoder or transformer directly.

### Vectorise natively

```python
vec = sekai.make_vec("Portfolio-v0", num_envs=1024)  # a PortfolioVecEnv: one array op per step
batch = vec.reset(seed=0)
batch = vec.step(vec.batch_action_space.sample())
```

### Train with Stable-Baselines3

```python
import sekai
from sekai.compat import to_gymnasium
from stable_baselines3 import PPO

model = PPO("MultiInputPolicy", to_gymnasium(sekai.make("Portfolio-v0")))
model.learn(total_timesteps=10_000)
```

## Environments

| ID | Observation | Action | Description |
|---|---|---|---|
| `CartPole-v1` | Box(4) | Discrete(2) | Balance a pole on a cart |
| `MountainCar-v0` | Box(2) | Discrete(3) | Build momentum to climb a hill |
| `MountainCarContinuous-v0` | Box(2) | Box(1) | Mountain car with continuous force |
| `Pendulum-v1` | Box(3) | Box(1) | Swing up and balance a pendulum |
| `Acrobot-v1` | Box(6) | Discrete(3) | Swing a two-link arm above a line |
| `Trading-v0` | Box(lookback + 1) | Discrete(3) | Go short / flat / long on one asset |
| `Portfolio-v0` | Dict | Box(n + 1) | Allocate across correlated assets and cash |
| `Market-v0` | Dict (MaskedBox) | Box(max_assets + 1) | Trade a universe that lists and delists, with features, costs and financing |
| `MarketMaking-v0` | Box(2) | Box(2) | Quote bid/ask, manage inventory (Avellaneda–Stoikov) |
| `OptimalExecution-v0` | Box(3) | Box(1) | Liquidate a large order (Almgren–Chriss) |
| `LimitOrderBook-v0` | Dict (MaskedBox) | MultiDiscrete([6, 6, 3]) | Make markets in an event-driven limit order book |

The classic control environments match gymnasium's dynamics exactly, verified step by step
in the test suite. The finance environments simulate markets by default, and `Trading-v0`
and `Portfolio-v0` also accept your own price history:

```python
import numpy as np
import sekai

prices = np.cumprod(1 + np.random.default_rng(0).normal(0, 0.01, (1000, 4)), axis=0)
env = sekai.make("Portfolio-v0", prices=prices, episode_length=126)  # random 6-month windows
```

`Market-v0`, the general-purpose finance environment, takes a `MarketData` history with
listings and delistings (or simulates a regime-switching market whose assets come and go),
per-asset features, market-wide series, any cost model and reward function, leverage and
financing. `sekai.envs.finance.LimitOrderBookMarket` is a multi-agent order-book market in
which traders act on their own clocks. The four classic finance environments also have native
vector implementations, which `make_vec()` uses automatically.

The microstructure environments come with their textbook solutions as baselines, e.g.
`env.almgren_chriss_actions()` and `env.avellaneda_stoikov_action()`. On
`OptimalExecution-v0` the Almgren–Chriss schedule scores about -320 bps against -420 bps for
TWAP, so a learned policy has a clear bar to beat.

Run `sekai list` to see every registered environment, or browse the
[environment catalogue](https://sekai.qiancapital.com/environments/).

## Why sekai?

| | gymnasium | sekai |
|---|---|---|
| Return types | `obs, rew, term, trunc, info = env.step(a)` | `StepResult` frozen dataclass with `.done` |
| Async environments | Not supported | `async_step` / `async_reset` on every env, `AsyncVecEnv` |
| Multi-agent | Delegated to PettingZoo | `MultiAgentEnv` with per-agent typed spaces, event-driven turns, PettingZoo bridge |
| Variable-size observations | `Sequence` / `Graph` spaces (not batchable) | `MaskedBox` padded sets that batch and flatten, `Sequence` with padding |
| Vectorised simulators | `vector_entry_point` | `NativeVecEnv` base class, native finance envs, shared-memory `ProcVecEnv` |
| Rendering | `render()` on Env, mode fixed at construction | Separate `Renderer` objects: swap, record, or skip freely |
| Episode statistics | Untyped `info["episode"]["r"]` | Typed `EpisodeStats` with pluggable trackers |
| Vector env naming | `AsyncVectorEnv` means multiprocessing | `SyncVecEnv`, `AsyncVecEnv` (asyncio), `ProcVecEnv` (processes) |
| Registry plugins | Entry-point plugins | `register_namespace("qc", loader)` for lazy plugin namespaces |
| Env checking | Passive checker wraps every env | `check_env()` in your tests; zero overhead in training |
| Finance | Not in scope | `sekai.finance` toolkit and six finance environments |

Coming from gymnasium? The [migration guide](https://sekai.qiancapital.com/guide/gymnasium/)
maps every API, and `sekai.compat` lets you mix both while you move over.

## Command line

```bash
sekai list                               # registered environments
sekai info Portfolio-v0                  # spec, spaces, wrapper stack
sekai check MyEnv-v0 --determinism       # run check_env (non-zero exit on errors)
sekai bench Portfolio-v0 --num-envs 1024    # native vector env when available
sekai -m my_package.envs list            # import a plugin package first
```

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](https://github.com/QianCapital/sekai/blob/master/CONTRIBUTING.md)
for the development setup, and the [examples](https://github.com/QianCapital/sekai/tree/master/examples)
for runnable scripts, from a pure-NumPy agent that solves CartPole in under a second to
finance baselines.

## Citing

If you use sekai in research, please cite it using the metadata in
[CITATION.cff](https://github.com/QianCapital/sekai/blob/master/CITATION.cff).

## License

Apache 2.0. See [LICENSE](https://github.com/QianCapital/sekai/blob/master/LICENSE).

<div align="center">
<sub>sekai · 世界 — Qian Capital</sub>
</div>
