# Training agents

sekai environments train with any gymnasium-based library through the
[gymnasium bridge](gymnasium.md#using-sekai-envs-with-gymnasium-libraries). Most need nothing
more; the finance environments whose observations hold a padded set (`Market-v0`'s assets,
`LimitOrderBook-v0`'s trades) need one wrapper, and `sekai.nn` gives them a network that
treats the set as a set. This page goes from the bridge to a trained Stable-Baselines3 policy.

```bash
pip install "sekai[gymnasium,torch]" stable-baselines3
```

| Observation | Example | Wrap with | Stable-Baselines3 policy |
|---|---|---|---|
| `Box` | `CartPole-v1`, `Trading-v0` | nothing | `"MlpPolicy"` |
| flat `Dict` | `Portfolio-v0` | nothing | `"MultiInputPolicy"` |
| `Dict` holding a `MaskedBox` (or any nested `Dict`/`Tuple`) | `Market-v0`, `LimitOrderBook-v0` | `UnnestObservation` | `"MultiInputPolicy"` with `MarketFeaturesExtractor`, or `SlotActorCriticPolicy` |
| anything, as one vector | | `FlattenObservation` | `"MlpPolicy"` |

## The gymnasium bridge

`sekai.compat.to_gymnasium(env)` wraps a sekai env, or a wrapper stack, as a `gymnasium.Env`:
spaces are converted one to one, results become tuples, seeds pass through, and an env's
[`action_mask()`](dynamic.md#action-masks) becomes the `action_masks()` method sb3-contrib's
`MaskablePPO` reads.

```python
import numpy as np
import sekai
from sekai.compat import to_gymnasium

gym_env = to_gymnasium(sekai.make("Market-v0"))
print(gym_env.observation_space["assets"])
# Dict('values': Box(-inf, inf, (10, 20, 2), float32), 'mask': MultiBinary((10,)))
```

Conversion keeps the nesting: `Market-v0`'s `"assets"` entry is a
[`MaskedBox`](dynamic.md#sets-maskedbox), a `Dict` of `values` and `mask`, so the gymnasium
space is a `Dict` inside a `Dict`. Stable-Baselines3 does not support that:
`MultiInputPolicy` raises `NotImplementedError: Nested observation spaces are not supported`.

## Nested observations: UnnestObservation

`sekai.UnnestObservation(env)` turns a nested `Dict`/`Tuple` observation into one `Dict` with a
key per leaf, joined with `/` (tuple positions as `0`, `1`, ...). Unlike `FlattenObservation`,
which makes one vector, every entry keeps its own shape, so a network can still tell the set
from the rest:

```python
env = sekai.UnnestObservation(sekai.make("Market-v0"))
print(env.observation_space)
# Dict({assets/values: Box(-inf, inf, (10, 20, 2), float32), assets/mask: MultiBinary(10),
#       portfolio: Box(-inf, inf, (11,), float32), account: Box(-inf, inf, (2,), float32)})

obs = env.reset(seed=0).observation
print(obs["assets/values"].shape, obs["assets/mask"])  # (10, 20, 2) [1 1 0 1 1 1 1 0 1 1]
nested = env.nest(obs)  # the exact inverse: the wrapped env's observation
print(sorted(nested), sorted(nested["assets"]))  # ['account', 'assets', 'portfolio'] ['mask', 'values']
```

The leaves are the same arrays, not copies, and `action_mask()` passes through, so
`MaskablePPO` still sees the mask. The structural maps are also available as functions,
for code that holds the nested space (a policy that wants the structure back, on tensors as
well as arrays):

```python
from sekai.wrappers.unnest_obs import nest, unnest, unnest_space

space = sekai.make("LimitOrderBook-v0").observation_space
print(list(unnest_space(space).keys()))  # ['book', 'trades/values', 'trades/mask', 'agent']
x = space.sample()
back = nest(space, unnest(space, x))
print(sorted(back["trades"]), back["trades"]["mask"] is x["trades"]["mask"])  # ['mask', 'values'] True
```

`unnest` and `nest` work on batches too (a vector env's dicts of stacked arrays have the same
nesting), and `sekai.VecUnnestObservation(vec)` applies them to a vector env whose sub-envs
cannot be wrapped one by one, such as a [native vector env](vector.md).

!!! note "Why `/` and not `.`"
    PyTorch module names cannot contain `.`, and Stable-Baselines3's default `Dict`
    extractor builds one module per key, so keys like `assets.values` fail there with a
    `KeyError`. Pass `sep="."` (or anything else) only for libraries that accept it.

## Set encoders: sekai.nn

`sekai.nn` holds reference PyTorch networks for sekai observations
(`pip install "sekai[torch]"`; `import sekai` itself never imports PyTorch).
`MaskedSetEncoder` encodes a batch of `MaskedBox` samples, `values` `(B, S, *element_shape)`
and `mask` `(B, S)`, into an embedding per slot and one per set:

```python
import torch
import sekai.nn

encoder = sekai.nn.MaskedSetEncoder((20, 2), embed_dim=32, element_encoder="conv")
values = torch.as_tensor(obs["assets/values"])[None]  # a batch of one set
mask = torch.as_tensor(obs["assets/mask"])[None]
enc = encoder(values, mask)
print(enc.slots.shape, enc.pooled.shape)  # torch.Size([1, 10, 32]) torch.Size([1, 64])
```

Each occupied slot goes through the same element encoder, optional transformer layers mix the
slots, and the occupied slots are pooled. Whatever the empty slots hold (even NaN):

- the per-slot embeddings are permutation-equivariant over the slots and the pooled
  embedding is invariant: moving the assets to other slots moves their embeddings with them
  and leaves the set's embedding unchanged;
- empty slots have no effect: their embeddings are zero, and they reach neither an occupied
  slot, nor the pooled embedding, nor the gradients;
- an empty set encodes to zeros, without NaN.

| Option | Values |
|---|---|
| `element_encoder` | `"mlp"` (default; an MLP over the flattened element), `"conv"` (1-D convolutions over the rows of an `(L, F)` window), `"transformer"` (one transformer layer over the rows), or your own `nn.Module` |
| `hidden_sizes` | widths of the `"mlp"` hidden layers, channels of the `"conv"` layers |
| `attention_layers` | transformer layers across the slots, attending to occupied slots only (default 0: a pure Deep Sets encoder) |
| `pooling` | one or more of `"mean"`, `"max"`, `"sum"` over the occupied slots, concatenated (default mean and max) |

`masked_mean(x, mask)` and `masked_max(x, mask)` are the pooling functions on their own.

## Stable-Baselines3

`sekai.nn.sb3.MarketFeaturesExtractor` is a features extractor for `MultiInputPolicy` on
unnested observations. It finds every `<name>/values` entry with a `<name>/mask` beside it,
encodes each such set with a `MaskedSetEncoder`, flattens the other entries through an MLP, and
concatenates the sets' pooled embeddings with it. `SlotActorCriticPolicy` builds on it: the
mean of each slot's action is the same MLP applied to that slot's embedding and the pooled
context, so an asset's weight follows its own features whichever slot it is in, while the cash
weight and the value read the context.

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from sekai.nn.sb3 import MarketFeaturesExtractor, SlotActorCriticPolicy


def make_env():
    return to_gymnasium(sekai.UnnestObservation(sekai.make("Market-v0", episode_length=63)))


vec_env = make_vec_env(make_env, n_envs=4, seed=0)
set_kwargs = {"element_encoder": "conv", "attention_layers": 1}

pooled = PPO(
    "MultiInputPolicy", vec_env, n_steps=256, batch_size=256, seed=0,
    policy_kwargs={
        "features_extractor_class": MarketFeaturesExtractor,
        "features_extractor_kwargs": {"set_kwargs": set_kwargs},
    },
)
pooled.learn(total_timesteps=20_000)

per_asset = PPO(
    SlotActorCriticPolicy, vec_env, n_steps=256, batch_size=256, seed=0,
    policy_kwargs={"slot_set": "assets", "features_extractor_kwargs": {"set_kwargs": set_kwargs}},
)
per_asset.learn(total_timesteps=20_000)


def evaluate(policy, episodes=50):
    """Mean log growth of equity per episode (the sum of LogReturn rewards)."""
    env = make_env()
    totals = []
    for seed in range(1000, 1000 + episodes):
        obs, _ = env.reset(seed=seed)
        done, total = False, 0.0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(policy(obs))
            total += float(reward)
            done = terminated or truncated
        totals.append(total)
    return float(np.mean(totals))


equal_weight = np.append(np.ones(10), 0.0).astype(np.float32)  # every listed asset, no cash
print(f"{evaluate(lambda o: pooled.predict(o, deterministic=True)[0]):+.4f}")  # ~ +0.0075
print(f"{evaluate(lambda o: per_asset.predict(o, deterministic=True)[0]):+.4f}")  # ~ +0.0061
print(f"{evaluate(lambda o: equal_weight):+.4f}")  # +0.0115
```

The two 20k-step runs take under a minute on a laptop CPU and are far from converged: the
assets' drift is small next to their volatility, and equal weight is the bar a learned
allocation has to clear (see the [finance baselines](../environments/finance.md)). The script is
[`examples/integrations/sb3_market.py`](https://github.com/QianCapital/sekai/blob/master/examples/integrations/sb3_market.py).

### Per-slot outputs in your own policy

`MarketFeaturesExtractor.encode(observations)` returns more than the features `forward()`
gives SB3: a `MarketEncoding` with the `context` (pooled embeddings and the other entries) and,
for each set by name, its `SetEncoding` (`slots` `(B, S, embed_dim)`, `pooled`, `mask`). With
`slot_set="assets"`, `forward()` appends that set's slot embeddings, flattened, to the features,
which is how a policy's action head reads them through SB3's pipeline. A scoring head is then a
few lines; `SlotActorCriticPolicy` is this head with SB3's plumbing, and its source is the
template for variants (another distribution, a softmax over the occupied slots):

```python
extractor = MarketFeaturesExtractor(make_env().observation_space, 32, slot_set="assets")
head = torch.nn.Linear(32 + extractor.context_dim, 1)  # the same for every slot


def slot_scores(observations):
    """One score per asset slot, from that slot's embedding and the context."""
    out = extractor.encode(observations)
    slots = out.sets["assets"].slots  # (B, 10, 32), zero in empty slots
    context = out.context[:, None].expand(-1, slots.shape[1], -1)
    return head(torch.cat([slots, context], dim=-1)).squeeze(-1)  # (B, 10)


batch = {key: torch.as_tensor(value, dtype=torch.float32)[None] for key, value in obs.items()}
print(slot_scores(batch).shape)  # torch.Size([1, 10])
```

## Action masks: MaskablePPO

`LimitOrderBook-v0` masks the orders that would breach its inventory limit, and the mask
reaches sb3-contrib's `MaskablePPO` through both wrappers (`pip install sb3-contrib`). Its
`MultiInputPolicy` takes the same features extractor:

```python
from sb3_contrib import MaskablePPO

lob = to_gymnasium(sekai.UnnestObservation(sekai.make("LimitOrderBook-v0")))
model = MaskablePPO(
    "MultiInputPolicy", lob, seed=0,
    policy_kwargs={"features_extractor_class": MarketFeaturesExtractor},
)
model.learn(total_timesteps=2_048)  # every action it samples is allowed by the mask
```

## CleanRL and other libraries

[CleanRL](https://docs.cleanrl.dev/)'s single-file scripts build their environments with a
`make_env()` function and `gymnasium.vector.SyncVectorEnv`: return a converted sekai env from
it (`to_gymnasium(sekai.FlattenObservation(...))` for the MLP agents, such as
`ppo_continuous_action.py`). For a set-aware agent, return the unnested env and give the
agent's network a `MaskedSetEncoder` over the batched `values` and `mask` entries:

```python
import gymnasium

envs = gymnasium.vector.SyncVectorEnv([make_env for _ in range(4)])
space = envs.single_observation_space
assets = sekai.nn.MaskedSetEncoder(space["assets/values"].shape[1:], 64, element_encoder="conv")
next_obs, _ = envs.reset(seed=0)
enc = assets(torch.as_tensor(next_obs["assets/values"]), torch.as_tensor(next_obs["assets/mask"]))
print(enc.pooled.shape)  # torch.Size([4, 128])
envs.close()
```

RLlib and other libraries that take gymnasium envs work the same way; for those that accept
nested `Dict` observations, `to_gymnasium(env)` is enough.
