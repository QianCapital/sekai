# sekai — 世界

**sekai** is a modern, flexible deep reinforcement learning environment
framework for Python 3.10+.  It provides a clean, typed API for defining,
registering, and interacting with RL environments — similar to OpenAI Gym /
Gymnasium but written from the ground up with modern Python conventions.

---

## Features

| Feature | Details |
|---|---|
| **Modern Python** | Full type annotations, dataclasses, `abc`, generics (Python ≥ 3.10) |
| **Typed spaces** | `Discrete`, `Box`, `MultiBinary`, `MultiDiscrete`, `Tuple`, `Dict` |
| **Clean env API** | `reset()` → `(obs, info)` · `step()` → `(obs, reward, terminated, truncated, info)` |
| **Environment registry** | `sekai.register` / `sekai.make` with import-path or class entry points |
| **Composable wrappers** | `TimeLimit`, `RecordEpisode`, `ClipReward`, `ObservationWrapper`, `ActionWrapper`, `RewardWrapper` |
| **Agent interface** | Abstract `Agent` base class + `RandomAgent` baseline |
| **Reproducibility** | Per-environment & per-space seeding via `numpy.random.Generator` |
| **Built-in envs** | `CartPole-v0`, `GridWorld-v0` |

---

## Installation

```bash
pip install sekai          # runtime only (numpy is the sole dependency)
pip install sekai[dev]     # adds pytest, ruff, mypy
```

---

## Quick Start

```python
import sekai

# Create a registered environment
env = sekai.make("CartPole-v0")          # TimeLimit(200) applied automatically
obs, info = env.reset(seed=42)

for _ in range(200):
    action = env.action_space.sample()   # random policy
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

env.close()
```

---

## Defining a Custom Environment

```python
from sekai import Env
from sekai.spaces import Box, Discrete
import numpy as np
from typing import Any

class MyEnv(Env[np.ndarray, np.intp]):
    """Minimal custom environment."""

    metadata = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = Box(low=-1.0, high=1.0, shape=(4,))
        self.action_space = Discrete(2)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._seed_np_random(seed)   # seeds env + both spaces
        obs = self.np_random.uniform(-0.05, 0.05, size=(4,)).astype(np.float32)
        return obs, {}

    def step(self, action: np.intp) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs = self.observation_space.sample()
        reward = float(action)
        terminated = bool(self.np_random.random() < 0.05)
        return obs, reward, terminated, False, {}
```

### Register and `make` it

```python
import sekai

sekai.register(
    "MyEnv-v0",
    "mypackage.mymodule:MyEnv",    # dotted import path
    max_episode_steps=200,
    reward_threshold=180.0,
)

env = sekai.make("MyEnv-v0")      # TimeLimit(200) applied automatically
```

---

## Spaces

```python
from sekai.spaces import Box, Discrete, MultiBinary, MultiDiscrete, Tuple, Dict
import numpy as np

# Discrete: {0, 1, ..., n-1}
d = Discrete(4)
d.sample()          # np.intp  in {0,1,2,3}
d.contains(3)       # True

# Box: continuous n-dimensional space
b = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
b.sample()          # ndarray of shape (3,)

# MultiBinary: binary vectors
mb = MultiBinary(5)
mb.sample()         # int8 array of 0s and 1s, shape (5,)

# MultiDiscrete: product of discrete ranges
md = MultiDiscrete([3, 4, 2])
md.sample()         # int64 array e.g. [2, 0, 1]

# Tuple: ordered heterogeneous product
t = Tuple((Discrete(3), Box(-1, 1, shape=(2,))))
t.sample()          # (np.intp, ndarray)

# Dict: named heterogeneous product
d2 = Dict({"obs": Box(-1, 1, shape=(4,)), "mask": MultiBinary(4)})
d2.sample()         # {"obs": ndarray, "mask": ndarray}
```

All spaces support:
- `space.sample()` — draw a uniform random element
- `x in space` / `space.contains(x)` — membership test
- `space.seed(42)` — re-seed the space's RNG

---

## Wrappers

Wrappers transform environments without modifying their source code.

```python
from sekai.wrappers import TimeLimit, RecordEpisode, ClipReward

env = sekai.make("CartPole-v0", max_episode_steps=0)  # no built-in limit

# Stack wrappers
env = ClipReward(env, min_reward=-1.0, max_reward=1.0)
env = TimeLimit(env, max_episode_steps=500)
env = RecordEpisode(env)

obs, _ = env.reset()
while True:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        print("Episode stats:", info.get("episode"))
        break
```

### Custom wrappers

```python
from sekai.wrappers import ObservationWrapper, RewardWrapper, ActionWrapper
import numpy as np

class NormalizeObs(ObservationWrapper):
    def observation(self, obs: np.ndarray) -> np.ndarray:
        return (obs - obs.mean()) / (obs.std() + 1e-8)

class ScaleReward(RewardWrapper):
    def reward(self, reward: float) -> float:
        return reward * 0.01

class DiscreteToFloat(ActionWrapper):
    def action(self, action: float) -> np.intp:
        return np.intp(int(action > 0))
```

---

## Agent Interface

```python
from sekai.core.agent import Agent, RandomAgent
import numpy as np

# Use the built-in random agent
env = sekai.make("GridWorld-v0")
agent = RandomAgent(env.action_space)

obs, _ = env.reset(seed=0)
for _ in range(100):
    action = agent.act(obs)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        break

# Implement your own agent
class MyAgent(Agent[np.ndarray, np.intp]):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation: np.ndarray, *, explore: bool = True) -> np.intp:
        # TODO: your policy here
        return self.action_space.sample()

    def learn(self, obs, action, reward, next_obs, terminated, truncated, info=None):
        # TODO: update your policy
        return {"loss": 0.0}
```

---

## Built-in Environments

| ID | Description | Obs shape | Actions | Limit |
|---|---|---|---|---|
| `CartPole-v0` | Balance a pole on a moving cart | `(4,)` float32 | 2 (left/right) | 200 steps |
| `GridWorld-v0` | Navigate a square grid to a goal | `(2,)` float32 | 4 (URDL) | 100 steps |

---

## Development

```bash
# Install with dev extras
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check sekai tests

# Type-check
mypy sekai
```

---

## Design Principles

1. **Minimal dependencies** — only `numpy` at runtime.
2. **Explicit over implicit** — `terminated` and `truncated` are separate
   signals (following the Gymnasium convention).
3. **Type-safe** — all public APIs carry full type annotations.
4. **Composable** — wrappers chain cleanly via the `Wrapper` base class.
5. **Reproducible** — deterministic seeding at every level.

