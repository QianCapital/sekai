# Episode statistics

`RecordEpisodeStatistics` measures every episode and, when it ends, puts an `EpisodeStats`
dataclass in `info["episode"]` and hands it to a pluggable **tracker**. Typed fields replace
gymnasium's `info["episode"]["r"]` / `["l"]` / `["t"]` dictionary.

## EpisodeStats

| Field | Type | Meaning |
|---|---|---|
| `episode_return` | `float` | sum of rewards (as seen by this wrapper layer) |
| `episode_length` | `int` | number of steps |
| `elapsed_time` | `float` | wall-clock seconds from `reset()` to the last step |
| `start_time`, `end_time` | `float` | `time.monotonic()` timestamps |
| `terminated`, `truncated` | `bool` | how the episode ended |
| `extra` | `dict` | free-form; trackers and vector wrappers add to it |
| `done` | property | `terminated or truncated` |

```python
import numpy as np
import sekai

env = sekai.RecordEpisodeStatistics(sekai.make("CartPole-v1"), deque_size=10)
env.action_space.seed(0)

result = env.reset(seed=0)
while not result.done:
    result = env.step(env.action_space.sample())

stats = result.info["episode"]       # also env.episode_stats, the latest episode
print(stats.episode_return, stats.episode_length, stats.terminated)  # 18.0 18 True
print(env.episode_stats is stats)  # True
```

!!! tip "Where to put it"
    Statistics are recorded at the layer where the wrapper sits. Wrap *inside* reward
    normalisation or scaling to log raw returns, and *outside* `TimeLimit` (as `make()`
    gives you) so truncated episodes are counted.

The wrapper starts counting the next episode as soon as it reports one, so the statistics stay
correct even when an inner `AutoReset` starts the next episode without a `reset()` call
reaching this layer.

## Trackers

A tracker is any object with four methods (the `sekai.StatsTracker` protocol):

| Method | Called |
|---|---|
| `on_reset(env_id, seed)` | after every `reset()`, with `env.spec.id` and the seed in use |
| `on_step(reward, terminated, truncated, info)` | after every `step()` |
| `on_episode_end(stats)` | when an episode ends, before `stats` is put in `info` |
| `summary()` | whenever you call it; returns a `dict` |

### InMemoryTracker

The default. It keeps the last `deque_size` episodes (100 unless you pass another
`deque_size` to the wrapper) in `tracker.episodes`:

```python
for _ in range(9):
    result = env.reset()
    while not result.done:
        result = env.step(env.action_space.sample())

print(env.tracker.summary())  # {'num_episodes': 10, 'mean_return': 21.5, 'min_return': 11.0, 'max_return': 58.0, 'mean_length': 21.5}
print(env.tracker.mean_return(), env.tracker.mean_length())  # 21.5 21.5
```

### LoggingTracker

Emits one `INFO` record per episode on the `sekai.stats` logger (or a logger you pass), with
the statistics as `extra` fields, ready for structured-logging handlers:

```python
import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(name)s %(message)s return=%(episode_return).1f length=%(episode_length)d",
)
env = sekai.RecordEpisodeStatistics(sekai.make("CartPole-v1"), tracker=sekai.LoggingTracker())
env.action_space.seed(0)
result = env.reset(seed=0)
while not result.done:
    result = env.step(env.action_space.sample())
```

```text
sekai.stats episode_end return=18.0 length=18
```

### A custom tracker

Trackers see every step's `info`, so they can compute statistics that need the whole path.
This one records each episode's maximum drawdown of a trading account and attaches it to
`stats.extra`:

```python
class DrawdownTracker:
    """Maximum drawdown of info["equity"] per episode."""

    def __init__(self):
        self.drawdowns = []

    def on_reset(self, env_id, seed):
        self.peak, self.worst = 1.0, 0.0

    def on_step(self, reward, terminated, truncated, info):
        self.peak = max(self.peak, info["equity"])
        self.worst = max(self.worst, 1 - info["equity"] / self.peak)

    def on_episode_end(self, stats):
        stats.extra["max_drawdown"] = self.worst
        self.drawdowns.append(self.worst)

    def summary(self):
        return {"mean_max_drawdown": float(np.mean(self.drawdowns))}


tracker = DrawdownTracker()
print(isinstance(tracker, sekai.StatsTracker))  # True

env = sekai.RecordEpisodeStatistics(sekai.make("Trading-v0"), tracker=tracker)
for seed in range(20):
    result = env.reset(seed=seed)
    while not result.done:
        result = env.step(2)  # buy and hold
print(round(result.info["episode"].extra["max_drawdown"], 3))  # 0.107
print(round(tracker.summary()["mean_max_drawdown"], 3))  # 0.202
```

The same pattern feeds TensorBoard, Weights & Biases or MLflow: log `stats` fields in
`on_episode_end`.

## Vector environments

`VecRecordEpisodeStatistics` does the same for a [vector env](vector.md#vector-wrappers):
when sub-env `i` finishes, `infos[i]["episode"]` is an `EpisodeStats` whose
`extra["env_index"]` is `i`, and the stats go to the tracker's `on_episode_end()`.

```python
vec = sekai.VecRecordEpisodeStatistics(sekai.make_vec("CartPole-v1", num_envs=4))
actions = vec.batch_action_space
actions.seed(0)
vec.reset(seed=0)
for _ in range(200):
    result = vec.step(actions.sample())
print(vec.tracker.summary()["num_episodes"])  # 33
```

!!! note
    The vector wrapper reports only finished episodes: it calls `on_episode_end()` but not
    `on_reset()` / `on_step()`. Trackers that need per-step data should wrap each sub-env in
    `RecordEpisodeStatistics` instead (e.g. through `make_vec(..., wrappers=[...])`).
