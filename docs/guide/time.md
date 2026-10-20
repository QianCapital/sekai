# Time and history

A market environment is a time series: decisions happen at timestamps, steps can have
different lengths, the agent needs some history to act on, data and orders arrive late, and
the statistics of the data drift. This page covers the conventions and wrappers sekai provides
for each.

## Timestamps and step lengths

Environments report time through two `info` keys:

| Key | Meaning | In the built-in finance envs |
|---|---|---|
| `timestamp` | when the observation is made, i.e. the time of the decision it informs | the data's timestamp (a `numpy.datetime64` or a number) on historical data, else the step index; session time in the order-book envs |
| `dt` | the length of a step | in years in `Trading-v0`, `Portfolio-v0` and `Market-v0` (`1 / data.periods_per_year` on dated data, else the market model's `dt` when simulated, else `1/252`; an explicit `dt=` overrides all three); in the model's time units in `MarketMaking-v0` and `OptimalExecution-v0`; the time a step covered in the order-book envs |

Both appear in the infos of `reset()` and of every `step()`. On simulated data the timestamp
is a step counter:

```python
import numpy as np
import sekai

env = sekai.make("Trading-v0")
result = env.reset(seed=0)
print(result.info["timestamp"], round(result.info["dt"], 6))  # 0 0.003968
print(env.step(1).info["timestamp"])  # 1
```

On dated data it is the date of the bar the decision is made on. Here `options={"start": ...}`
picks the first decision date, and the weekend between Friday and Monday is skipped: the
timestamps follow the data, and `dt` is one bar, `1 / data.periods_per_year` years (about
1/261 for weekday bars), unless you pass `dt=` yourself.

```python
from sekai.finance import MarketData

dates = np.arange("2024-01-01", "2025-01-01", dtype="datetime64[D]")
dates = dates[np.is_busday(dates)]  # weekdays only
closes = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0.0, 0.01, len(dates))))
data = MarketData(closes, timestamps=dates)

env = sekai.make("Trading-v0", data=data, lookback=10, episode_length=20)
result = env.reset(seed=0, options={"start": "2024-03-01"})  # a Friday
print(result.info["timestamp"], env.step(1).info["timestamp"])  # 2024-03-01 2024-03-04
```

Use the same keys in your own environments, so that code built on them (time-based
discounting below, logging and evaluation by date) works with every env.

## Variable step lengths

In an event-driven environment the agent decides when something happens (a quote update, a
fill, a news item), so steps have different lengths. Discounting by a constant `gamma` per step
then treats a 2-millisecond step and a 2-second step alike. The semi-Markov fix is to discount
by elapsed time: with `gamma` the discount per unit of time, a step of length `dt` is
discounted by `gamma ** dt`.

An env that decides at every quote update, with updates arriving at random times:

```python
class QuoteUpdateEnv(sekai.Env):
    """Decide at every quote update; updates arrive at random times (a Poisson process)."""

    def __init__(self, rate=5.0, horizon=10.0):
        self.rate, self.horizon = rate, horizon
        self.observation_space = sekai.Box(-np.inf, np.inf, shape=(1,))
        self.action_space = sekai.Discrete(3, start=-1)  # short / flat / long

    def reset(self, *, seed=None, options=None):
        self._set_rng(seed)
        self.time, self.mid = 0.0, 0.0
        return sekai.ResetResult(np.array([self.mid], np.float32), {"timestamp": self.time})

    def step(self, action):
        dt = self.rng.exponential(1.0 / self.rate)  # time until the next update
        move = self.rng.normal(0.0, 0.1 * np.sqrt(dt))
        self.time += dt
        self.mid += move
        info = {"timestamp": self.time, "dt": dt}
        obs = np.array([self.mid], np.float32)
        return sekai.StepResult(obs, float(action) * move, False, self.time >= self.horizon, info)
```

Per-step discounts come straight from the infos:

```python
env = QuoteUpdateEnv()
env.action_space.seed(0)
result = env.reset(seed=0)
rewards, dts = [], []
while not result.done:
    result = env.step(env.action_space.sample())
    rewards.append(result.reward)
    dts.append(result.info["dt"])
print(len(dts), round(min(dts), 4), round(max(dts), 4))  # 45 0.0024 0.6573

gamma = 0.9                                                   # discount per unit of time
discounts = gamma ** np.asarray(dts)                          # one discount per step
weights = np.concatenate([[1.0], np.cumprod(discounts)[:-1]])  # discount applied to each reward
print(round(float(weights @ np.asarray(rewards)), 4))  # 0.0687
```

In a TD target the per-step discount replaces the constant one:
`reward + gamma ** info["dt"] * (1 - terminated) * value(next_observation)`. When only
timestamps are available, `dt` is the difference between consecutive ones. The event-driven
multi-agent market works the same way per agent; see
[Multi-agent](multi-agent.md#event-driven-environments).

## History in observations

A policy without memory needs recent history in its observation. There are three ways to put
it there.

**Lookback windows inside the env.** The finance environments already observe a window of
recent data: `Trading-v0` the last `lookback` log returns (and the current position),
`Portfolio-v0` a `(lookback, n_assets)` window of returns. The env has the data, so this is
the cheapest and most exact option; set its size with `lookback=`.

```python
print(sekai.make("Trading-v0", lookback=10).observation_space.shape)  # (11,)
```

**`FrameStackObservation`** stacks the last `num_stack` observations on a new leading axis.
It works on any space that can be batched, leaf by leaf, so a `Dict` observation keeps its
keys. Use it for what the env does not already keep, such as the agent's own past weights:

```python
portfolio = sekai.make("Portfolio-v0", n_assets=2)
history = sekai.FrameStackObservation(
    sekai.FilterObservation(portfolio, ["weights"]), num_stack=3, padding="zeros"
)
print(history.observation_space)  # Dict({weights: Box(0.0, 1.0, (3, 3), float32)})
print(history.reset(seed=0).observation["weights"].tolist())  # [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]

print(sekai.FrameStackObservation(portfolio, 4).observation_space)  # Dict({returns: Box(-inf, inf, (4, 30, 2), float32), weights: Box(0.0, 1.0, (4, 3), float32)})
```

`padding="reset"` (the default) fills the stack with the first observation after a reset;
`padding="zeros"` uses the space's [zero sample](spaces.md#zero-samples), which makes the
start of an episode visible to the agent.

**`TimeAwareObservation`** adds the elapsed fraction of the episode, which makes a time limit
observable (otherwise the end of an episode looks random to the agent). It reads the horizon
from a `TimeLimit`; the finance envs end episodes themselves, so pass it explicitly:

```python
env = sekai.TimeAwareObservation(sekai.make("Trading-v0", episode_length=50), max_episode_steps=50)
print(env.reset(seed=0).observation[-1], env.step(1).observation[-1])  # 0.0 0.02
```

## Latency

Live trading sees data late and executes orders late. Training without that latency gives a
policy that trades on prices it could never have acted on.

| Wrapper | Models |
|---|---|
| `DelayObservation(env, delay, *, padding="zeros")` | feed latency: the agent sees the observation from `delay` steps ago |
| `DelayAction(env, delay, *, noop=None)` | order latency: an action chosen now is executed `delay` steps later; `noop` is executed while the pipeline fills |
| `StickyAction(env, repeat_action_probability)` | random execution lag: with that probability, the previous action is repeated instead |
| `ActionRepeat(env, repeat)` | a slower decision clock: each action is held for `repeat` steps |

`DelayAction` needs an explicit `noop` unless the action space is a `Box` (where it defaults
to zeros), because the first index of a `Discrete` space is rarely "do nothing". In
`Trading-v0` the actions are short / flat / long, so flat is `1`:

```python
env = sekai.DelayAction(sekai.make("Trading-v0"), delay=2, noop=1)
env.reset(seed=0)
positions = [env.step(2).info["position"] for _ in range(4)]  # always choose long
print(positions, env.pending_actions)  # [0.0, 0.0, 1.0, 1.0] [2, 2]

feed = sekai.DelayObservation(sekai.make("Trading-v0"), delay=1)
print(np.all(feed.reset(seed=0).observation == 0))  # True
```

`pending_actions` lists the actions chosen but not yet executed, oldest first. The two delays
compose: `DelayAction(DelayObservation(env, 1), 1, noop=1)` decides on data one step old, and
the decision reaches the market one step later.

## Non-stationary statistics

`NormalizeObservation` and `VecNormalizeObservation` default to `stats="running"`, which
weights every observation ever seen equally. On data whose distribution changes (volatility
regimes, trending prices) those statistics drift towards a long-run average that no longer
describes the present. `stats="ewm"` uses exponentially weighted statistics instead: an
observation's weight halves every `halflife` steps.

```python
env = sekai.NormalizeObservation(sekai.make("Portfolio-v0"), stats="ewm", halflife=500, clip=5.0)
print(env.obs_rms["returns"])  # ExponentialMovingStats(shape=(30, 5), halflife=500, count=0)
```

The statistics classes are public. After a calm period and a stressed one with three times
the volatility, the running estimate is still far below the current level, while the
exponentially weighted one has caught up:

```python
rng = np.random.default_rng(0)
data = np.concatenate([rng.normal(0.0, 0.01, (2000, 1)), rng.normal(0.0, 0.03, (500, 1))])

running = sekai.RunningMeanStd(shape=(1,))
ewm = sekai.ExponentialMovingStats(shape=(1,), halflife=100)
for x in data:
    running.update(x[None])
    ewm.update(x[None])
print(running.std.round(4), ewm.std.round(4))  # [0.016] [0.0281]
```

`ExponentialMovingStats(shape, halflife)` has the same interface as `RunningMeanStd`
(`update()`, `mean`, `var`, `std`, `normalize()`, `state_dict()`). Called directly, a batch
counts as `len(batch)` samples. The normalisation wrappers count steps instead: with
`stats="ewm"`, each call of `NormalizeObservation` and each vector step of
`VecNormalizeObservation` counts once, however many sub-envs (or occupied `MaskedBox` slots)
it carries, so `halflife` is in steps whatever `num_envs` is and needs no scaling. `clip=`
bounds the normalised values, which keeps a policy stable when a regime change produces
values far outside the recent range.

## Point-in-time data

The other half of handling time correctly is never letting an observation contain information
that was not available at its timestamp. The finance toolkit covers this:
[`MarketData`](../finance/data.md) holds prices with their timestamps and marks unlisted
assets, splits data chronologically with an embargo, and builds walk-forward folds;
`sekai.finance.features.asof_join` aligns macro and fundamental data with a publication lag;
and `sekai.finance.features.check_causal` tests a feature transform for look-ahead, raising
`sekai.LookAheadError` when it finds one. See [Market data](../finance/data.md) and the
[finance overview](../finance/index.md).
