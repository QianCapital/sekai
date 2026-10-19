# Registry

The registry maps environment IDs to `EnvSpec`s, so environments can be built by name with
`sekai.make()` and `sekai.make_vec()`, listed with `sekai list`, and checked with
`sekai check`. The built-in environments are registered when `sekai` is imported.

| Function | Does |
|---|---|
| `register(id, entry_point, *, max_episode_steps=None, reward_threshold=None, nondeterministic=False, kwargs=None, additional_wrappers=(), vector_entry_point=None)` | add an environment; returns its `EnvSpec` |
| `make(id_or_spec, *, max_episode_steps=None, **kwargs)` | build one environment |
| `make_vec(id_or_spec, num_envs=1, *, vectorization_mode=None, vector_kwargs=None, wrappers=(), **kwargs)` | build a [vector environment](vector.md) (native when the env has one) |
| `spec(id)` | look up an `EnvSpec` |
| `all_specs()` | every registered `EnvSpec` |
| `deregister(id)` | remove an environment |
| `register_namespace(namespace, loader)` | register a lazy loader for a plugin namespace |
| `pprint_registry()` | print every ID, grouped by namespace |

## IDs

IDs follow `[namespace/]Name[-vN]`:

| ID | `namespace` | `name` | `version` |
|---|---|---|---|
| `CartPole-v1` | `None` | `CartPole` | `1` |
| `qc/MarketEnv-v0` | `qc` | `MarketEnv` | `0` |
| `my_ns/CustomEnv` | `my_ns` | `CustomEnv` | `None` |

Names and namespaces may contain letters, digits, `_` and `-`. Anything else raises
`ValueError` when the spec is created.

## Registering

The entry point is either a `"module.path:ClassName"` string, imported only when the env is
first made (keeping `import yourpackage` cheap), or any callable that returns an `Env`.

```python
import sekai

sekai.register(
    id="desk/Trading-v1",
    entry_point="sekai.envs.finance.trading:TradingEnv",
    kwargs={"positions": (0.0, 1.0), "transaction_cost": 5e-4},  # long-only, 5 bp costs
    max_episode_steps=126,                                       # adds a TimeLimit
    reward_threshold=0.05,
)

env = sekai.make("desk/Trading-v1")
print(env)                          # <TimeLimit(<desk/Trading-v1>)>
print(env.unwrapped.positions)      # [0. 1.]
```

`make()` then:

1. merges the registered `kwargs` with the keyword arguments you pass (yours win);
2. calls the entry point;
3. attaches a copy of the spec, recording the kwargs and time limit actually used, as
   `env.spec`;
4. wraps the env in `TimeLimit` if a `max_episode_steps` is set (the `make()` argument
   overrides the registered one);
5. applies `additional_wrappers` in order (outermost last).

### Native vector implementations

`vector_entry_point` names a batched implementation of the same environment: a
[`NativeVecEnv`](vector.md#native-vector-environments) (or any `VecEnv`) called as
`factory(num_envs=..., **kwargs)`. `make_vec()` then uses it by default, and `sekai list`
marks the ID with `[native vec]`; `vectorization_mode="sync"` (or `"async"`, `"proc"`)
still builds the per-env version:

```python
sekai.register(
    id="desk/Portfolio-v1",
    entry_point="sekai.envs.finance.portfolio:PortfolioEnv",
    vector_entry_point="sekai.envs.finance.vector:PortfolioVecEnv",
    kwargs={"n_assets": 8, "transaction_cost": 5e-4},
)
vec = sekai.make_vec("desk/Portfolio-v1", num_envs=256)
print(type(vec).__name__, vec.spec.vector_entry_point)  # PortfolioVecEnv sekai.envs.finance.vector:PortfolioVecEnv
print(type(sekai.make_vec("desk/Portfolio-v1", 4, vectorization_mode="sync")).__name__)  # SyncVecEnv
```

`max_episode_steps` is consumed by `make()` and is **not** passed to the environment's
constructor (as in gymnasium); every other keyword argument is. Pass `max_episode_steps=-1`
to drop a registered time limit altogether.

```python
env = sekai.make("desk/Trading-v1", transaction_cost=0.0, max_episode_steps=21)
print(env.spec.kwargs)              # {'positions': (0.0, 1.0), 'transaction_cost': 0.0}
print(env.spec.max_episode_steps)   # 21
```

Registering an ID twice raises `EnvAlreadyRegistered`; `deregister()` it first to replace it.
An unknown ID raises `EnvNotFound` with close matches:

```python
try:
    sekai.make("desk/Tradin-v1")
except sekai.EnvNotFound as e:
    print(e.suggestions)  # ('desk/Trading-v1', 'Trading-v0')
```

### Default wrappers

`additional_wrappers` attaches wrappers to every `make()` of an ID. Each is a `WrapperSpec`
with a string entry point, so the spec stays serialisable:

```python
from sekai import WrapperSpec

sekai.register(
    "desk/CartPoleNormalized-v0",
    entry_point="sekai.envs.classic_control.cartpole:CartPoleEnv",
    max_episode_steps=200,
    additional_wrappers=(
        WrapperSpec("NormalizeObservation", "sekai.wrappers:NormalizeObservation"),
        WrapperSpec("ClipReward", "sekai.wrappers:ClipReward", {"max_reward": 0.5}),
    ),
)
env = sekai.make("desk/CartPoleNormalized-v0")
print([type(layer).__name__ for layer in env])  # ['ClipReward', 'NormalizeObservation', 'TimeLimit', 'CartPoleEnv']
```

`nondeterministic=True` documents that an env is not reproducible from its seed (e.g. it
reads a live feed). It is informational: sekai does not change behaviour based on it.

## EnvSpec

`EnvSpec` is a dataclass with the fields passed to `register()`, plus `namespace`, `name` and
`version` parsed from the ID.

```python
spec = env.spec
print(spec.pprint())
```

```text
id=desk/CartPoleNormalized-v0
entry_point=sekai.envs.classic_control.cartpole:CartPoleEnv
max_episode_steps=200
wrapper=NormalizeObservation (sekai.wrappers:NormalizeObservation)
wrapper=ClipReward (sekai.wrappers:ClipReward) {'max_reward': 0.5}
```

Specs with a string entry point round-trip through JSON, which is handy for experiment logs;
`spec.make()` (or `sekai.make(spec)`) rebuilds the environment without touching the registry:

```python
restored = sekai.EnvSpec.from_json(spec.to_json())
print(restored.make())  # <ClipReward(<NormalizeObservation(<TimeLimit(<desk/CartPoleNormalized-v0>)>)>)>
```

A callable entry point cannot be serialised; `to_json()` leaves it out.

## Plugins and namespaces

A package that ships environments registers them at import time, typically in its
`__init__.py` or an `envs` module:

```py
# acme_envs/__init__.py
import sekai

sekai.register("acme/OrderBook-v0", entry_point="acme_envs.order_book:OrderBookEnv",
               max_episode_steps=1_000)
```

Users then `import acme_envs` before `sekai.make("acme/OrderBook-v0")`, and the CLI takes
`-m acme_envs` to import it first ([Command line](cli.md#plugins)).

To defer even that import, register a **namespace loader**: a callable that runs once, the
first time any ID in the namespace is requested.

```python
calls = []


def load_lab_envs():
    calls.append("loaded")
    sekai.register("lab/Pendulum-v9", "sekai.envs.classic_control.pendulum:PendulumEnv",
                   kwargs={"g": 9.81}, max_episode_steps=300)


sekai.register_namespace("lab", load_lab_envs)
print([s.id for s in sekai.all_specs() if s.namespace == "lab"])  # []

env = sekai.make("lab/Pendulum-v9")  # triggers the loader
print(calls, env.unwrapped.g)  # ['loaded'] 9.81
```

In a real package the loader is usually
`lambda: importlib.import_module("acme_envs.registration")`.

!!! note "Lazy namespaces are invisible until loaded"
    `all_specs()`, `pprint_registry()` and `sekai list` only show IDs that are registered,
    so a namespace whose loader has not run yet does not appear. Any lookup of an ID in the
    namespace (`make`, `spec`, `make_vec`) runs the loader.

## Listing

```python
sekai.pprint_registry()
```

```text
Registered environments:
  [no namespace]
    Acrobot-v1
    CartPole-v1
    MarketMaking-v0
    MountainCar-v0
    MountainCarContinuous-v0
    OptimalExecution-v0
    Pendulum-v1
    Portfolio-v0
    Trading-v0
  [desk]
    desk/CartPoleNormalized-v0
    desk/Trading-v1
  [lab]
    lab/Pendulum-v9
```
