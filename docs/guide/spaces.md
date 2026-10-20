# Spaces

A space describes the set of valid observations or actions. sekai ships the same six types
as gymnasium, with the same constructor arguments, two more for observations whose size
changes from step to step, and JSON round-tripping and batching utilities used by the vector
environments.

| Space | Elements | Sample type | Constructor |
|---|---|---|---|
| `Box` | n-d arrays within `[low, high]` | `np.ndarray` (space dtype) | `Box(low, high, shape=None, dtype=np.float32, seed=None)` |
| `Discrete` | `{start, ..., start + n - 1}` | `np.intp` | `Discrete(n, seed=None, start=0)` |
| `MultiDiscrete` | per-entry discrete values | `np.ndarray` (`int64`) | `MultiDiscrete(nvec, dtype=np.int64, seed=None, start=None)` |
| `MultiBinary` | arrays of 0/1 | `np.ndarray` (`int8`) | `MultiBinary(n, seed=None)` (`n` an int or a shape) |
| `Dict` | dicts of sub-space elements | `dict[str, Any]` | `Dict(spaces=None, seed=None, **spaces)` |
| `Tuple` | tuples of sub-space elements | `tuple` | `Tuple(spaces, seed=None)` |
| `Sequence` | variable-length sequences of `feature_space` elements | `tuple`, or one stacked array (dict, ...) with `stack=True` | `Sequence(feature_space, *, max_length=None, stack=False, seed=None)` |
| `MaskedBox` | up to `max_size` `Box` elements in fixed slots, plus a validity mask | `dict`: `"values"` `(max_size, *shape)`, `"mask"` `int8` `(max_size,)` | `MaskedBox(low, high, shape=None, *, max_size, dtype=np.float32, fill_value=None, seed=None)` |

Every space implements `sample(mask=None)`, `contains(x)` (also `x in space`), `seed(seed)`,
`is_flattenable`, `to_jsonable()` / `from_jsonable()`, `==` and a readable `repr`. Spaces
compare by value (a `MultiDiscrete`'s dtype included) and, being mutable, are not hashable.
`contains()` returns `False`, rather than raising, for anything that is not a member, ragged
nested lists included.

## Box

A continuous (or integer) box. Bounds can be scalars broadcast to `shape`, or arrays whose
shape becomes the space's shape. Unbounded dimensions are allowed: `sample()` draws them from
a normal distribution (both sides unbounded) or a shifted exponential (one side).

```python
import numpy as np
import sekai

prices = sekai.Box(0.0, 1.0, shape=(3,), seed=0)
print(prices)            # Box(0.0, 1.0, (3,), float32)
print(prices.sample())   # [0.6369617  0.26978672 0.04097353]
print(prices.is_bounded) # (True, True)

returns = sekai.Box(-np.inf, np.inf, shape=(30, 5))       # unbounded
depth = sekai.Box(0, 255, shape=(2, 2), dtype=np.uint8)   # integer box
mixed = sekai.Box(low=np.array([0.0, -np.inf]), high=np.array([1.0, np.inf]))
print(mixed.is_bounded)  # (False, False)
```

Bounds must not be NaN (`ValueError`). For an integer dtype, infinite bounds become the dtype's
limits, as in gymnasium, and finite bounds outside its range are clipped to them with a
`UserWarning`:

```python
shares = sekai.Box(-np.inf, np.inf, shape=(2,), dtype=np.int64)
print(shares)  # Box(-9223372036854775808, 9223372036854775807, (2,), int64)
```

`contains()` requires the exact shape, a dtype castable to the space's (`same_kind`), and every
element within bounds. Python ints and lists of them are judged by value, so they are members of
any integer box whose bounds they fit (a NumPy `int64` array is not a member of a `uint8` box):

```python
print(np.array([0.5, 0.5, 0.5], dtype=np.float32) in prices)  # True
print([2.0, 0.0, 0.0] in prices)                               # False
print([[0, 255], [7, 8]] in depth, [[0, 256], [7, 8]] in depth)  # True False
```

## Discrete

```python
side = sekai.Discrete(3, start=-1, seed=0)   # {-1, 0, 1}: short / flat / long
print(side, [int(side.sample()) for _ in range(5)])  # Discrete(3, start=-1) [1, 0, 0, -1, -1]
print(0 in side, 1 in side, 2 in side)  # True True False
print(True in sekai.Discrete(2))        # False
```

Python ints, NumPy integers and 0-d integer arrays are members; `bool`s are not. `n` and
`start` may be NumPy integers too.

## MultiDiscrete

One discrete choice per entry; `nvec` may be multi-dimensional, and `start` offsets each entry.

```python
orders = sekai.MultiDiscrete([3, 3, 3], start=[-1, -1, -1], seed=0)  # 3 assets, -1/0/+1 each
print(orders, orders.sample())  # MultiDiscrete([3, 3, 3], start=[-1, -1, -1]) [1 0 0]
```

## MultiBinary

```python
signals = sekai.MultiBinary(4, seed=0)
print(signals, signals.sample())  # MultiBinary(4) [0 1 1 1]
print(sekai.MultiBinary((2, 3)).shape)  # (2, 3)
```

## Dict and Tuple

Composite spaces nest any other spaces. `Dict` keeps insertion order and supports
`space["key"]`, `keys()`, `values()`, `items()` and `len()`; `Tuple` supports indexing and
`len()`.

```python
obs_space = sekai.Dict(
    {"prices": sekai.Box(-np.inf, np.inf, shape=(5,)), "regime": sekai.Discrete(3)},
    seed=0,
)
print(obs_space)  # Dict({prices: Box(-inf, inf, (5,), float32), regime: Discrete(3)})
print(obs_space["regime"], list(obs_space.keys()))  # Discrete(3) ['prices', 'regime']

pair = sekai.Tuple([sekai.Discrete(2), sekai.Box(-1, 1, shape=(2,))])
print(pair[0], len(pair))  # Discrete(2) 2
```

`Dict(prices=..., regime=...)` with keyword arguments works too. `contains()` requires
exactly the space's keys (for `Dict`) or length (for `Tuple`, which also accepts lists).

## Sequence

A `Sequence` holds a variable number of elements of one feature space: the trades printed
since the last decision, the news items in a window, the orders resting in a book. By default
a sample is a tuple of elements; with `stack=True` it is one stacked batch (an array of shape
`(length, *feature_shape)` for a `Box` feature space, a dict of arrays for a `Dict`), which is
what most models consume.

```python
trades = sekai.Sequence(sekai.Box(0.0, 10.0, shape=(2,)), max_length=5, seed=0)
print(trades)  # Sequence(Box(0.0, 10.0, (2,), float32), max_length=5)
sample = trades.sample()
print(type(sample).__name__, len(sample), () in trades)  # tuple 5 True

tape = sekai.Sequence(sekai.Box(0.0, 10.0, shape=(2,)), max_length=5, stack=True, seed=0)
print(tape.sample().shape, tape.sample(mask=(3, None)).shape, tape.pack([]).shape)  # (5, 2) (3, 2) (0, 2)
```

`max_length` is optional: without it, sampled lengths are geometric (mean 3). `pack(items)`
builds a sample from a list of elements and `elements(x)` returns the list back, whichever
representation the space uses.

A `Sequence` is not flattenable, and a batch of them across vector sub-envs is a tuple of
per-env samples, not one array. When a policy needs fixed shapes, convert it into a `MaskedBox`
with [`PadSequenceObservation`](dynamic.md#event-lists-sequence).

## MaskedBox

A `MaskedBox` is a variable-size set of up to `max_size` `Box` elements, padded to a fixed
shape: the assets in a universe that gains and loses listings, the open positions of an
account. A sample is a dict with a `"values"` array of shape `(max_size, *shape)`, one row per
slot, and an `int8` `"mask"` of shape `(max_size,)` that is 1 where the slot holds an element.

```python
universe = sekai.MaskedBox(-np.inf, np.inf, shape=(2,), max_size=4, seed=0)
print(universe)  # MaskedBox(-inf, inf, (2,), max_size=4, float32)
print(universe["values"], universe["mask"])  # Box(-inf, inf, (4, 2), float32) MultiBinary(4)

obs = universe.pack([[0.01, 0.2], [-0.02, 0.3]], slots=[0, 3])  # two assets, in slots 0 and 3
print(obs["mask"], obs in universe)  # [1 0 0 1] True
values, slots = universe.unpack(obs)
print(values.shape, slots)  # (2, 2) [0 3]
```

| Member | Returns |
|---|---|
| `pack(values, slots=None)` | a sample with `k` elements placed in `slots` (default: the first `k`), the rest padded |
| `unpack(x)` | `(values[slots], slots)` for the occupied slots |
| `empty()` | a sample with every slot empty |
| `element_space`, `element_shape` | the `Box` of one element, and its shape |
| `max_size`, `fill_value` | the number of slots, and the value of empty slots |

Empty slots must hold `fill_value` (by default zero, clipped into the bounds), so a sample
with stale data in an empty slot is not a member:

```python
stale = {"values": obs["values"].copy(), "mask": obs["mask"]}
stale["values"][1] = 5.0  # slot 1 is empty
print(stale in universe)  # False
```

`MaskedBox` is a `Dict` subclass, so every `Dict` utility applies unchanged: flattening,
batching, `FilterObservation`, and gymnasium conversion (as a `Dict` of `values` and `mask`).
Shapes stay fixed, so its samples stack across vector sub-envs. See
[Variable-size observations](dynamic.md) for keeping slot assignments stable and for
normalising over occupied slots only.

## Masked sampling

`sample(mask=...)` restricts sampling to valid values, e.g. for invalid-action masking:

| Space | Mask format |
|---|---|
| `Discrete(n)` | `int8` array (or list) of shape `(n,)`; `1` marks allowed values (all zeros raises `ValueError`) |
| `MultiDiscrete` | a tuple of `int8` arrays, one per entry (nested to match `nvec`) and of length `nvec[i]`, **or** one array of shape `(*nvec.shape, max(nvec))`, each entry's flags padded to `max(nvec)`; another structure or length raises `ValueError` |
| `MultiBinary` | `int8` array (or list) of the space's shape: `1` forces 1, `0` forces 0, `-1` samples randomly |
| `Dict` | a dict of sub-masks (missing keys sample unmasked) |
| `Tuple` | a tuple with one sub-mask (or `None`) per sub-space (another length raises `ValueError`) |
| `Box` | an array broadcastable to the space's shape; entries where it is `0` are not sampled but set to `Box.zero()` (zero clipped into the bounds) |
| `MaskedBox` | an `int8` (or `bool`) array of shape `(max_size,)` fixing which slots are occupied (by default each slot is occupied with probability 1/2) |
| `Sequence` | `(length, feature_mask)`: a fixed length (or `None`) and a mask passed to every element's `sample()` |

```python
action_space = sekai.Discrete(5, seed=0)
print(action_space.sample(mask=np.array([0, 1, 0, 0, 1], dtype=np.int8)))  # 4

md = sekai.MultiDiscrete([3, 5], seed=0)
print(md.sample(mask=(np.array([1, 0, 0], np.int8), np.array([0, 0, 0, 1, 1], np.int8))))  # [0 4]

mb = sekai.MultiBinary(4, seed=0)
print(mb.sample(mask=np.array([1, 0, -1, -1], np.int8)))  # [1 0 0 1]

weights = sekai.Box(0.0, 1.0, shape=(4,), seed=0)  # asset 1 cannot be traded
print(weights.sample(mask=np.array([1, 0, 1, 1], np.int8)))  # [0.6369617  0.         0.04097353 0.01652764]
```

An environment that restricts its actions publishes the current mask through an
`action_mask()` method in exactly this format; see [Action masks](dynamic.md#action-masks).

## Seeding

Each space has its own generator, created lazily from OS entropy. Seed it with the `seed=`
constructor argument or `space.seed(s)` (which returns the seed used). Seeding a `Dict` or
`Tuple` seeds every sub-space deterministically. To seed sub-spaces individually, pass a dict
of seeds (`Dict`) or a list (`Tuple`, `None` to skip one), to the constructor or to `seed()`
(which then returns `-1`, as there is no single seed); a key that is not a sub-space, or more
seeds than sub-spaces, raises `ValueError`.

```python
a = sekai.Dict({"x": sekai.Discrete(10), "y": sekai.Box(0, 1, shape=(2,))})
b = sekai.Dict({"x": sekai.Discrete(10), "y": sekai.Box(0, 1, shape=(2,))})
a.seed(123)
b.seed(123)
print(a.sample()["x"] == b.sample()["x"])  # True

a.seed({"x": 7})
b["x"].seed(7)
print(a.sample()["x"] == b.sample()["x"])  # True
```

## Flattening

Most RL libraries want a flat vector. The utilities in `sekai.spaces` convert any
flattenable space to and from a 1-D `float32` array:

| Function | Returns |
|---|---|
| `flatdim(space)` | length of the flattened vector |
| `flatten(space, x)` | 1-D `float32` array |
| `unflatten(space, v)` | a sample of `space` |
| `flatten_space(space)` | the `float32` `Box` that `flatten()` maps into |

`Box` and `MultiBinary` are raveled; `Discrete` becomes a one-hot vector and `MultiDiscrete`
concatenated one-hots; `Dict` and `Tuple` concatenate their parts in key/index order, so a
`MaskedBox` flattens to its values followed by its mask. `Sequence` is not flattenable.

`unflatten()` always returns a member of the space: `Box` values are cast to its dtype (integer
dtypes rounded) and clipped into its bounds, so `float32` rounding cannot push a value past a
bound. The flat vector is `float32`, which represents integers exactly only up to `2**24`, so
larger integer `Box` values (and `float64` values) do not round-trip exactly.

```python
from sekai.spaces import flatdim, flatten, flatten_space, unflatten

sample = obs_space.sample()
flat = flatten(obs_space, sample)
print(flatdim(obs_space), flatten_space(obs_space))  # 8 Box([-inf, -inf, -inf, -inf, -inf, 0.0, 0.0, 0.0], [inf, inf, inf, inf, inf, 1.0, 1.0, 1.0], (8,), float32)
print(flat[5:])  # [1. 0. 0.]
print(unflatten(obs_space, flat)["regime"] == sample["regime"])  # True
```

The [`FlattenObservation`](wrappers.md#built-in-wrappers) wrapper applies this to every
observation.

## Batching

Vector environments stack per-env samples along a new leading axis. The same utilities are
public, so your own code can do it too:

| Function | Purpose |
|---|---|
| `batch_space(space, n)` | the space of `n` stacked samples |
| `concatenate(space, items)` | stack a list of samples into one batch |
| `iterate(space, batch)` | split a batch back into per-env samples (inverse of `concatenate`) |

| Single space | `batch_space(space, n)` |
|---|---|
| `Box(low, high, shape)` | `Box` of shape `(n, *shape)` |
| `Discrete(k, start=s)` | `MultiDiscrete([k] * n, start=[s] * n)` |
| `MultiDiscrete(nvec)` | `MultiDiscrete` of shape `(n, *nvec.shape)` |
| `MultiBinary(shape)` | `MultiBinary((n, *shape))` |
| `Dict` / `Tuple` | `Dict` / `Tuple` of batched sub-spaces |
| `MaskedBox(..., shape, max_size=k)` | `Dict` with `values` `Box` `(n, k, *shape)` and `mask` `MultiBinary((n, k))` |
| `Sequence` | `Tuple` of `n` copies: sequences are ragged, so a batch is a tuple of per-env samples |

```python
from sekai.spaces import batch_space, concatenate, iterate

print(batch_space(obs_space, 4))  # Dict({prices: Box(-inf, inf, (4, 5), float32), regime: MultiDiscrete([3, 3, 3, 3])})

batch = concatenate(obs_space, [obs_space.sample() for _ in range(4)])
print(batch["prices"].shape, batch["regime"].shape)  # (4, 5) (4,)
print(len(list(iterate(obs_space, batch))))  # 4
```

So a batched `Dict` observation is a dict of arrays, not a list of dicts. An empty `Dict` or
`Tuple` inside a space batches to `{}` / `()`, and `iterate()` gives every sample that same
value, taking the number of samples from the other parts. A space with nothing but empty
`Dict`s and `Tuple`s cannot be split: its batches do not record how many samples they hold, so
`iterate()` raises `ValueError` (and `Sequence(..., stack=True)` rejects it as a feature space).

## Zero samples

`zero_sample(space)` returns the neutral element of a space, used as padding by
`FrameStackObservation(padding="zeros")` and `DelayObservation`:

| Space | `zero_sample(space)` |
|---|---|
| `Box` | zeros clipped into the bounds (`Box.zero()`) |
| `Discrete`, `MultiDiscrete` | the lowest value (`start`) |
| `MultiBinary` | zeros |
| `Dict` / `Tuple` | the zero sample of every sub-space |
| `MaskedBox` | every slot empty (`empty()`) |
| `Sequence` | an empty sequence |

```python
from sekai.spaces import zero_sample

print(zero_sample(sekai.Box(1.0, 2.0, shape=(2,))), zero_sample(sekai.Discrete(3, start=-1)))  # [1. 1.] -1
print(zero_sample(universe)["mask"], zero_sample(tape).shape)  # [0 0 0 0] (0, 2)
```

Like the other utilities it is a `functools.singledispatch` function, so a
[custom space](#custom-spaces) can register its own.

## Serialisation

`to_jsonable()` returns a plain dict and `space_from_jsonable()` rebuilds any registered space
type from it, which is handy for logging environment configurations or shipping them to a
remote worker:

```python
import json
from sekai.spaces import space_from_jsonable

data = json.dumps(obs_space.to_jsonable())
print(data[:60])  # {"type": "Dict", "spaces": {"prices": {"type": "Box", "low":
print(space_from_jsonable(json.loads(data)) == obs_space)  # True
```

`Sequence` and `MaskedBox` round-trip the same way, bounds, `max_length` and `fill_value`
included:

```python
print(space_from_jsonable(json.loads(json.dumps(universe.to_jsonable()))) == universe)  # True
print(space_from_jsonable(tape.to_jsonable()))  # Sequence(Box(0.0, 10.0, (2,), float32), max_length=5, stack=True)
```

!!! note "Infinite bounds"
    Unbounded `Box` bounds serialise as `Infinity` / `-Infinity`, which Python's `json`
    module reads and writes but strict JSON parsers reject.

## Custom spaces

Subclass `sekai.Space` and implement `sample()`, `contains()` and `is_flattenable`. The
utilities are `functools.singledispatch` functions, so a custom space opts in to flattening,
batching and JSON by registering implementations. A probability simplex (long-only portfolio
weights) as an example:

```python
from sekai.spaces import utils as space_utils


class Simplex(sekai.Space[np.ndarray]):
    """Non-negative weights that sum to one."""

    def __init__(self, n: int, seed=None):
        self.n = n
        super().__init__(shape=(n,), dtype=np.float32, seed=seed)

    def sample(self, mask=None):
        return self.rng.dirichlet(np.ones(self.n)).astype(np.float32)

    def contains(self, x):
        x = np.asarray(x)
        return x.shape == (self.n,) and bool(np.all(x >= 0)) and abs(float(x.sum()) - 1) < 1e-5

    @property
    def is_flattenable(self):
        return True

    def to_jsonable(self):
        return {"type": "Simplex", "n": self.n}

    @classmethod
    def from_jsonable(cls, data):
        return cls(data["n"])

    def __repr__(self):
        return f"Simplex({self.n})"

    def __eq__(self, other):
        return isinstance(other, Simplex) and other.n == self.n


space_utils.flatdim.register(Simplex, lambda space: space.n)
space_utils.flatten.register(Simplex, lambda space, x: np.asarray(x, np.float32).ravel())
space_utils.unflatten.register(Simplex, lambda space, x: np.asarray(x, np.float32))
space_utils.flatten_space.register(Simplex, lambda space: sekai.Box(0.0, 1.0, shape=(space.n,)))
space_utils.batch_space.register(
    Simplex, lambda space, n: sekai.Box(0.0, 1.0, shape=(n, space.n))
)
space_utils.concatenate.register(Simplex, lambda space, items: np.stack(items))
sekai.spaces.register_space_type("Simplex", Simplex)

weights = Simplex(4, seed=0)
print(weights.sample() in weights)  # True
print(space_from_jsonable(weights.to_jsonable()))  # Simplex(4)
print(batch_space(weights, 8))  # Box(0.0, 1.0, (8, 4), float32)
```

`iterate()` needs no registration for array-like batches: its default splits along the first
axis.

`Simplex` defines `__eq__`. A space that does not compares by identity: two instances are equal
only if they are the same object, and so are two `Sequence`s or `Dict`s holding them.
