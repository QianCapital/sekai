# Variable-size observations

Many market observations have no natural fixed size: the assets in a universe that gains and
loses listings, the trades printed since the last decision, the orders resting in a book. This
page shows how to give a policy such data without giving up fixed shapes, how to tell it which
actions are valid, and how the wrappers treat padded data.

## Why fixed shapes

Most of the RL stack assumes that every observation has the same shape:

- vector environments stack the sub-envs' observations along a new leading axis
  ([`concatenate`](spaces.md#batching)), which needs equal shapes;
- a network has a fixed input layer, so a flattened observation must have the same length at
  every step;
- running normalisation statistics, frame stacks and replay buffers are fixed-size arrays.

The standard answer is padding plus a mask: reserve room for the largest set you expect, and
send a mask that says which entries are real. A set encoder, a transformer with an attention
mask, or a plain masked mean uses the mask; an MLP can take the flattened values and mask
together.

## Two tools

| Data | Space | Fixed-size form |
|---|---|---|
| a set of items that come and go, each with the same features: a changing asset universe, open positions | [`MaskedBox`](spaces.md#maskedbox) | already fixed: `max_size` slots and a mask |
| a list of events whose count varies from step to step: trades since the last decision, news items | [`Sequence`](spaces.md#sequence) | `PadSequenceObservation` turns it into a `MaskedBox` |

### Sets: MaskedBox

A `MaskedBox` observation is a dict of a `"values"` array with one row per slot and an `int8`
`"mask"`. Empty slots hold zeros (the `fill_value`), so a masked mean is two lines:

```python
import numpy as np
import sekai

universe = sekai.MaskedBox(-np.inf, np.inf, shape=(2,), max_size=4)
obs = universe.pack([[0.01, 0.2], [-0.03, 0.4]], slots=[0, 2])

values, mask = obs["values"], obs["mask"]
mean = values.sum(axis=0) / max(mask.sum(), 1)  # empty slots add zero
print(mean.round(3))  # [-0.01  0.3 ]
```

### Event lists: Sequence

A `Sequence` observation can have any length. Keep the raw form inside the env, where it is
natural, and convert it at the edge with `PadSequenceObservation`, which turns every
`Sequence` of `Box` elements (at the top level or inside a `Dict`/`Tuple`) into a
`MaskedBox`, elements in order in the first slots. An env that observes the trades printed
since its last decision:

```python
class TradeTapeEnv(sekai.Env):
    """Each step observes the trades printed since the last decision: (price change, size)."""

    def __init__(self):
        trade = sekai.Box(np.array([-np.inf, 0.0]), np.array([np.inf, np.inf]))
        self.observation_space = sekai.Sequence(trade, stack=True)
        self.action_space = sekai.Discrete(3, start=-1)  # short / flat / long

    def reset(self, *, seed=None, options=None):
        self._set_rng(seed)
        return sekai.ResetResult(self._tape(), {})

    def step(self, action):
        tape = self._tape()
        reward = float(action) * float(tape[:, 0].sum())  # position times the price move
        return sekai.StepResult(tape, reward, False, False, {})

    def _tape(self):
        n = self.rng.poisson(3.0)  # a different number of trades every step
        moves = self.rng.normal(0.0, 0.01, n)
        sizes = self.rng.integers(1, 10, n)
        return np.column_stack([moves, sizes]).astype(np.float32)


env = TradeTapeEnv()
env.reset(seed=0)
print([env.step(1).observation.shape for _ in range(4)])  # [(5, 2), (0, 2), (3, 2), (3, 2)]

padded = sekai.PadSequenceObservation(TradeTapeEnv(), max_length=8)
print(padded.observation_space)  # MaskedBox([-inf, 0.0], inf, (2,), max_size=8, float32)
print(padded.reset(seed=0).observation["mask"])  # [1 1 0 0 0 0 0 0]
```

`max_length` defaults to the `Sequence`'s own `max_length`. A longer sequence keeps its most
recent `max_length` elements (`truncate="tail"`, the right choice for event streams); pass
`truncate="head"` to keep the first ones, or `truncate="error"` to raise `ValueError`.

The padded env vectorises like any other:

```python
vec = sekai.SyncVecEnv([lambda: sekai.PadSequenceObservation(TradeTapeEnv(), max_length=8)] * 4)
print(vec.reset(seed=0).observations["values"].shape)  # (4, 8, 2)
vec.close()
```

Without the wrapper the batch would be a tuple of four arrays of different lengths.

## Stable slots

When assets come and go, the slot an asset occupies must not change while it stays listed.
If row 2 means one stock on Monday and another on Tuesday, per-slot features, frame stacks
and a recurrent policy's memory all silently mix the two. `sekai.finance.SlotMap` keeps each
instrument in the slot it was first given, gives newcomers the lowest free slot, and reports
which slots changed hands:

```python
from sekai.finance import SlotMap

slots = SlotMap(4)
print(slots.update(["ACME", "BOLT", "CRUX"]))  # [0 1 2]
print(slots.update(["BOLT", "CRUX", "DYNE"]))  # [1 2 0]
print(slots.assets, slots.changed)  # ('DYNE', 'BOLT', 'CRUX', None) [ True False False False]
```

`update()` returns the slot of each present instrument, in order, ready for
`MaskedBox.pack(values, slots=...)`. `changed` marks slots whose owner differs from before the
call, so you can clear per-slot state (a lagged return, an open position) when a slot is
reused; `mask` is the occupancy as `int8`. The [finance toolkit](../finance/index.md) builds
its dynamic-universe environment on it.

## Action masks

When some actions are invalid in the current state (a halted stock cannot be traded, a short
is not available, an inventory limit is hit), the environment says so with an
`action_mask()` method. It returns a mask in the format that `action_space.sample(mask=...)`
accepts ([Masked sampling](spaces.md#masked-sampling)):

| Action space | `action_mask()` returns |
|---|---|
| `Discrete(n)` | an `int8` array `(n,)`; 1 = allowed |
| `MultiDiscrete(nvec)` | a tuple of `int8` arrays, one per entry, or one array padded to `max(nvec)` |
| `Box` | an array broadcastable to the shape; 0 = that component must stay at zero |

The method can live on the env or on any wrapper. A wrapper that forbids shorting in
`Trading-v0` (whose actions are short / flat / long):

```python
class NoShortSelling(sekai.Wrapper):
    """Trading-v0 when the stock cannot be borrowed: action 0 (short) is not allowed."""

    def action_mask(self):
        return np.array([0, 1, 1], dtype=np.int8)


env = NoShortSelling(sekai.make("Trading-v0"))
print(sekai.get_action_mask(env), sekai.get_action_mask(sekai.make("Trading-v0")))  # [0 1 1] None

env.action_space.seed(0)
env.reset(seed=0)
print(sorted({int(env.action_space.sample(mask=env.action_mask())) for _ in range(20)}))  # [1, 2]
print(sekai.check_env(env))  # []
```

`sekai.get_action_mask(env)` searches the whole wrapper stack and returns `None` for an env
without a mask. Masks are information for the agent, not enforcement: the env should still
handle a masked action, or reject it with `sekai.InvalidAction`.

Everything that samples or learns picks the mask up from the same method:

| Consumer | How it gets the mask |
|---|---|
| your own loop | `sekai.get_action_mask(env)`, or `env.action_mask()` directly |
| [`check_env`](checking.md) | samples every random action with the env's mask, and reports a mask that `sample()` rejects |
| a policy that applies the mask itself (masked logits) | `ActionMaskObservation(env)` adds the mask to the observation |
| vector envs | `vec.call("action_mask")` returns one mask per sub-env |
| sb3-contrib's `MaskablePPO` | `to_gymnasium(env)` has an `action_masks()` method returning the flat boolean mask it looks for |

```python
masked = sekai.ActionMaskObservation(env)
print(masked.observation_space["action_mask"], masked.reset(seed=0).observation["action_mask"])  # MultiBinary(3) [0 1 1]

vec = sekai.SyncVecEnv([lambda: NoShortSelling(sekai.make("Trading-v0")) for _ in range(2)])
vec.reset(seed=0)
print(vec.call("action_mask"))  # [array([0, 1, 1], dtype=int8), array([0, 1, 1], dtype=int8)]
vec.close()

from sekai.compat import to_gymnasium

gym_env = to_gymnasium(env)
gym_env.reset(seed=0)
print(gym_env.action_masks())  # [False  True  True]
```

`ActionMaskObservation(env, key="action_mask")` flattens the mask to an `int8` `MultiBinary`
(per-entry masks concatenated for `MultiDiscrete`): a `Dict` observation gains a `key` entry,
and any other observation becomes `{"observation": obs, key: mask}`. The flattening helpers,
`flatten_action_mask(space, mask)` and `action_mask_space(space)`, are in
[`sekai.utils.masking`](../api/utils.md#action-masks); they support `Discrete`,
`MultiDiscrete` and `Box` action spaces. Through `to_gymnasium()`,
`MaskablePPO("MlpPolicy", to_gymnasium(env))` reads the mask on its own (MaskablePPO needs a
discrete action space). A multi-agent env takes the agent as an argument,
`action_mask(agent)`; see [Multi-agent](multi-agent.md#action-masks).

## Normalising masked sets

[`NormalizeObservation`](wrappers.md#built-in-wrappers) and
[`VecNormalizeObservation`](vector.md#vector-wrappers) understand padding. For a
`MaskedBox` they keep per-feature statistics computed over the occupied slots only, so empty
slots never bias them, and padding stays exactly zero in the output. Every `Box` part of a
`Dict` gets its own statistics, and other parts pass through:

```python
market = sekai.NormalizeObservation(sekai.make("Market-v0"))
print(market.obs_rms["assets"])  # RunningMeanStd(shape=(20, 2), count=0.0001)

market.action_space.seed(0)
result = market.reset(seed=0)
for _ in range(20):
    result = market.step(market.action_space.sample(mask=sekai.get_action_mask(market)))
assets = result.observation["assets"]
print(assets["mask"], np.abs(assets["values"][assets["mask"] == 0]).max())  # [1 1 0 1 1 1 1 0 1 1] 0.0
print(market.obs_rms["assets"].count)  # 168.0001
```

Twenty-one observations of eight listed assets: 168 rows of statistics, none from padding.

!!! note "Normalise before stacking"
    `FrameStackObservation` stacks a `MaskedBox` leaf by leaf, so the stacked observation is a
    plain `Dict` of `values` and `mask`. Put `NormalizeObservation` *inside* the frame stack,
    or a normaliser outside it treats the padding as data.

## Worked example: a changing universe

An allocation environment over up to four assets that list and delist at random. It puts
everything on this page together: a `MaskedBox` observation in stable slots, an action mask
for empty slots, and wrappers and vector envs that work unchanged.

```python
class ChangingUniverseEnv(sekai.Env):
    """Allocate across up to `max_assets` assets that list and delist at random."""

    def __init__(self, max_assets=4, episode_length=50, churn=0.05):
        self.max_assets, self.episode_length, self.churn = max_assets, episode_length, churn
        self.observation_space = sekai.Dict({
            # per listed asset: (last return, volatility), in a stable slot
            "assets": sekai.MaskedBox(-np.inf, np.inf, shape=(2,), max_size=max_assets),
            "weights": sekai.Box(0.0, 1.0, shape=(max_assets + 1,)),  # slots, then cash
        })
        self.action_space = sekai.Box(0.0, 1.0, shape=(max_assets + 1,))
        self.slots = SlotMap(max_assets)

    def reset(self, *, seed=None, options=None):
        self._set_rng(seed)
        self.t, self.next_id, self.vol = 0, 0, {}
        for _ in range(self.max_assets - 1):
            self._list()
        self.slots.reset()
        self.slots.update(self.vol)
        self.returns = np.zeros(self.max_assets)
        self.weights = np.eye(self.max_assets + 1)[-1]  # all cash
        return sekai.ResetResult(self._obs(), {"assets": self.slots.assets})

    def action_mask(self):
        """Empty slots must get zero weight; cash is always allowed."""
        return np.append(self.slots.mask, 1).astype(np.int8)

    def step(self, action):
        w = np.asarray(action, dtype=np.float64) * self.action_mask()  # enforce the mask
        self.weights = w / w.sum() if w.sum() > 0 else np.eye(self.max_assets + 1)[-1]
        self.returns = np.zeros(self.max_assets)
        for asset, vol in self.vol.items():
            self.returns[self.slots.slot_of(asset)] = self.rng.normal(0.0, vol)
        reward = float(np.log1p(self.weights[:-1] @ self.returns))

        # Churn: listed assets may delist, free slots may get a new listing.
        for asset in [a for a in self.vol if self.rng.random() < self.churn]:
            del self.vol[asset]
        for _ in range(self.max_assets - len(self.vol)):
            if self.rng.random() < self.churn:
                self._list()
        self.slots.update(self.vol)
        changed = np.append(self.slots.changed, False)
        self.weights[-1] += self.weights[changed].sum()  # a delisted position goes to cash
        self.weights[changed] = 0.0
        self.returns[self.slots.changed] = 0.0  # a new listing has no last return yet

        self.t += 1
        truncated = self.t >= self.episode_length
        return sekai.StepResult(self._obs(), reward, False, truncated, {"assets": self.slots.assets})

    def _list(self):
        self.vol[f"asset_{self.next_id}"] = self.rng.uniform(0.01, 0.03)
        self.next_id += 1

    def _obs(self):
        listed = list(self.vol)
        slots = [self.slots.slot_of(a) for a in listed]
        features = np.array([[self.returns[s], self.vol[a]] for a, s in zip(listed, slots)])
        assets = self.observation_space["assets"].pack(features.reshape(-1, 2), slots=slots)
        return {"assets": assets, "weights": self.weights.astype(np.float32)}
```

It passes the checker, which samples its random actions with the mask:

```python
env = ChangingUniverseEnv()
print(sekai.check_env(env, check_determinism=True))  # []
```

A random rollout. `asset_2` keeps slot 2 while the assets around it are replaced:

```python
env.action_space.seed(0)
result = env.reset(seed=0)
print(result.info["assets"])  # ('asset_0', 'asset_1', 'asset_2', None)
for _ in range(40):
    result = env.step(env.action_space.sample(mask=env.action_mask()))
print(result.info["assets"])  # ('asset_4', 'asset_5', 'asset_2', 'asset_6')
print(result.observation["assets"]["mask"], env.action_mask())  # [1 1 1 1] [1 1 1 1 1]
```

The usual wrappers apply without changes. `NormalizeObservation` keeps statistics of shape
`(2,)` for the assets (one mean and variance per feature, pooled over occupied slots), and
`FlattenObservation` gives an MLP-ready vector of values, mask and weights:

```python
norm = sekai.NormalizeObservation(ChangingUniverseEnv())
print(norm.obs_rms["assets"])  # RunningMeanStd(shape=(2,), count=0.0001)
print(sekai.FlattenObservation(ChangingUniverseEnv()).observation_space.shape)  # (17,)
```

In a vector env, stack the sub-envs' masks into one mask for the batched action space:

```python
vec = sekai.SyncVecEnv([ChangingUniverseEnv for _ in range(3)])
vec.reset(seed=0)
masks = np.stack(vec.call("action_mask"))
print(masks.shape, masks[:, 3])  # (3, 5) [0 0 0]

actions = vec.batch_action_space
actions.seed(0)
print(actions.sample(mask=masks)[:, 3])  # [0. 0. 0.]
vec.close()
```

For a real dynamic universe on historical or simulated data, with features, costs and
financing, see `Market-v0` in the [finance toolkit](../finance/index.md).
