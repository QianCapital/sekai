# Batched and differentiable environments

`BatchMarketEnv` is the general financial markets environment, written once and run
three ways:

- **one environment**: `sekai.make("Market-v1")`, a standard `sekai.Env`;
- **a native vector environment**: `sekai.make_vec("Market-v1", num_envs=1024)`, every
  sub-env stepped with array operations (no Python loop over sub-envs);
- **a batch on a GPU**: `BatchMarketEnv(4096, ..., backend="torch", device="cuda")`,
  observations, rewards and infos as tensors on the device, and, with
  `differentiable = True`, rewards differentiable in the actions, so a policy can be
  trained by backpropagating through the market (direct policy optimisation).

Each of the `num_envs` *lanes* is an independent account trading its own episode: a
window of your history, or a fresh scenario from a market model or a generator of your
own. It is assembled from the [financial markets toolkit](index.md), and each part can be
swapped:

| Part | Argument | Default |
|---|---|---|
| where the data comes from | `data=` (a `MarketData`), `model=` (a `MarketModel`), or `scenarios=` (a `ScenarioGenerator`) | the regime-switching market of `Market-v0` |
| when orders fill | `execution=` | `"next_close"` |
| what the agent controls | `action=` (an `ActionMap`) | long-only scores, or target weights with `allow_short` |
| what it observes | `observation=` (an `ObservationBuilder`), or `features=` / `exogenous=` | the observation of `Market-v0`, plus pending orders |
| trading costs | `transaction_cost=` (a `CostModel`) | 10 bp of turnover |
| holding costs | `holding_cost=` (a `HoldingCost`) | none |
| risk model | `risk=` (a `FactorRiskModel`) | none |
| reward | `reward=` (a `RewardFunction`) | log growth of equity |

```python
import sekai
from sekai.envs.finance import BatchMarketEnv

env = sekai.make("Market-v1")                       # a one-lane BatchMarketEnv as an Env
vec = sekai.make_vec("Market-v1", num_envs=256)     # a MarketVecEnv (NumPy, native)
batch = BatchMarketEnv(256, backend="torch")        # tensors, for training on a device
```

## Execution timing

When does a decision fill? On real data the answer decides whether a backtest means
anything. A decision made on the close of day $t$ cannot fill at that same close: the
close is only known once trading has ended. Filling there anyway (what `Market-v0` does)
lets a policy trade on information it could not have acted on, and reinforcement learning
is very good at finding such gifts. `execution=` makes the choice explicit:

| `execution` | Fills at | Data needed | Use |
|---|---|---|---|
| `"close"` | the close of $t$, the price just observed | | simulated MDPs, where it is the definition; `Market-v0` parity. Optimistic on real data |
| `"next_close"` (default) | the close of $t + 1$ | | conservative and always available: the order is pending for a period while the previous book is held, and the observation includes it (`"pending"`), so it stays Markov |
| `"next_open"` | the open of $t + 1$ | an `"open"` field | daily data with opening prices: the old book earns the overnight move, the new one the move from the open to the close |
| `"next_vwap"` | the VWAP of $t + 1$ | a `"vwap"` field | as `"next_open"`, at a volume-weighted price |

With next-close execution, decision $k$ fills at the close of $k + 1$: exactly the account
that same-close execution would give with every action arriving one decision later
(transaction costs included). With next-open execution the period splits at the open:
the equity grows by the old book's overnight move, pays the trading cost, then grows by
the new book's move from the open to the close.

[`examples/execution_timing.py`](https://github.com/QianCapital/sekai/blob/master/examples/execution_timing.py)
shows what is at stake. Closing prices with bid-ask bounce reverse slightly from one
close to the next, so a short-term reversal strategy seems to work when it fills at the
close it observed, and stops working when orders fill at the next close:

```
execution='close'       annual return   +9.5%
execution='next_close'  annual return   -0.2%
```

Execution happens in the costs too: trading costs read the data (spread, volume) of the
period in which the order fills, and the volatility of the window the decision observed.

## Where the data comes from

**Your history.** `data=` takes a `MarketData` (see [Market data](data.md)), with every
feature, cost and risk input as a field. Raw prices with `split` and `dividend` fields
and `corporate_actions=True` trade on total returns
([corporate actions](data.md#corporate-actions)), and a
[memory-mapped](data.md#saving-and-memory-mapping) `MarketData` is shared by every worker
process instead of copied. Episodes are windows of it, placed by `sampler=`
(each lane gets `sampler.spawn()`) or by `reset(options={"start": ...})`. sekai never
fetches or sells market data: you load yours into a `MarketData` from files or databases
you have access to.

**A market model.** `model=` takes any `sekai.finance.MarketModel` (see
[Market simulators](simulators.md)); each episode is a freshly simulated path, and the
model's state series (such as the regime) can be observed with `exogenous=` or passed to
the policy as privileged information with `privileged_state=True`.

**Your generator.** `scenarios=` takes anything with
`sample(n_paths, *, seed) -> MarketPanel`: a generative model of markets (a GAN, a
diffusion model), a factor simulator, a bootstrap. A `MarketPanel` holds `P` paths of
`T` times and `N` assets on one backend: `returns` `(P, T, N)` over `(t - 1, t]`, a
`valid` mask, per-asset `fields` `(P, T, N, ...)` and market-wide `exogenous` series
`(P, T, ...)`. Each episode uses the first `lookback + episode_length + 1` times of a
fresh path, and `sample()` must be deterministic given the seed:

```python
import torch
from sekai.finance import MarketPanel


class AlphaScenarios:
    """Returns driven by a persistent alpha signal (an AR(1) per asset) plus noise."""

    def __init__(self, n_assets=50, n_steps=81, ic=0.1, vol=0.02, persistence=0.9):
        self.n_assets, self.n_steps = n_assets, n_steps
        self.ic, self.vol, self.rho = ic, vol, persistence

    def sample(self, n_paths, *, seed):
        g = torch.Generator().manual_seed(seed)
        shape = (n_paths, self.n_steps, self.n_assets)
        shocks = torch.randn(shape, generator=g)
        alpha = torch.empty(shape)
        alpha[:, 0] = shocks[:, 0]
        for t in range(1, self.n_steps):
            alpha[:, t] = self.rho * alpha[:, t - 1] + (1 - self.rho**2) ** 0.5 * shocks[:, t]
        noise = torch.randn(shape, generator=g)
        returns = torch.zeros(shape)
        returns[:, 1:] = self.vol * (self.ic * alpha[:, :-1] + (1 - self.ic**2) ** 0.5 * noise[:, 1:])
        return MarketPanel.from_returns(
            returns,
            fields={"alpha": alpha, "dollar_volume": torch.full(shape, 2e7), "spread": torch.full(shape, 1e-3)},
        )
```

The environment checks at construction that the panels have every field its components
read (observation, costs, risk model, constraints, execution prices), and how many times
an episode needs.

## The universe

Assets list and delist: a NaN price in a `MarketData`, or `valid = False` in a panel,
means the asset cannot be traded at that time. Each lane maps the tradable assets onto
`max_assets` slots (default: one per asset): an asset keeps its slot while it trades, and
newcomers take the lowest free slots, in column order. `env.slot_assets` names the asset in
each slot, and `env.action_mask()` marks the slots that can be traded now. An asset that
delists earns `delisting_return` over the period and its position becomes cash.

## What the agent controls

An `ActionMap` defines the action space and turns each action into target weights per
slot, given the account (`ActionContext`: what is tradable, the current weights, what can
be shorted, exposures). Every map is differentiable in the action:

| Map | Action | Weights |
|---|---|---|
| `LongOnlyScores(max_leverage, cash=True)` | `(S + 1,)` in `[0, 1]`, cash last | scores normalised into weights (`Market-v0`'s long-only action) |
| `TargetWeights(max_leverage)` | `(S,)` in `[-1, 1]` | scaled down to the gross cap (`Market-v0`'s long/short action) |
| `ScoresLeverage(constraints)` | `(S + 1,)`, unbounded | scores projected onto `constraints` at a gross of `max_gross * sigmoid(last)` |
| `TanhWeights(constraints)` | `(S,)`, unbounded | `cap * tanh(a)`, projected onto `constraints` |
| `TradeDelta(constraints)` | `(S + 1,)`, unbounded | the current book plus the action (zero holds), then as `ScoresLeverage` |

The last three project onto hard [constraints](execution.md#constraints): gross and per-name
caps, dollar and factor neutrality, names that cannot be shorted. The policy cannot propose
a book that breaks them, so constraints are never a matter of penalties the policy may
ignore. Each also has `from_target(scores, gross_fraction, ctx)`, the action that gives a
chosen book, so baselines can be written once for every parameterisation.

## Costs, risk and rewards

Trading costs are any [`CostModel`](execution.md#cost-models); its inputs come from the
data: `volume` is the field `"dollar_volume"` (traded value), or `"volume"` (shares) times
the close; `spread` is the field `"spread"`; `volatility` is estimated over the observed
window; `equity` is `capital` times the account's equity. Holding costs
([`HoldingCost`](execution.md#holding-costs)) charge each period for the positions held:
`BorrowFee()` reads each name's borrow fee from a `"borrow_rate"` field, and
`GrossFinancing(rate)` charges for the whole book. `shortable="shortable"` names a field
(nonzero where shorting is allowed) that every action map respects.

`risk=FactorRiskModel()` reads a factor model from the data (loadings `(T, N, k)` and
idiosyncratic variances `(T, N)` as fields, the factor covariance `(T, k, k)` as a
market-wide series) and adds the ex-ante `variance` of each step's book to the info and
to the reward's `StepOutcome`, so `SimpleReturn() - 2.0 * PortfolioVariance()` is a
mean-variance objective on the risk the book takes, not on the noise of its returns.

## Account rules

Besides costs and financing, a lane's account can follow the rules a broker enforces:

- `initial_margin=0.5`: equity needed per unit of gross exposure to open positions, so
  orders above `1 / initial_margin` gross (2x here) are scaled down;
- `maintenance_margin=0.25`: equity needed to keep them. When price moves push gross
  exposure above `1 / maintenance_margin`, every position is cut in proportion at that
  close, paying the cost of the trade, back to `1 / initial_margin`; `info["margin_call"]`
  flags it. The initial margin defaults to twice the maintenance margin (at most 1);
- `max_drawdown=0.2`: a risk stop that ends a lane's episode at a 20% drawdown;
- `flat_overnight=True`: no position from one trading session into the next. The book is
  sold to cash at the last decision of each session ([`MarketData.sessions()`](data.md#trading-sessions)),
  and `info["session_end"]` flags it;
- `shortable="shortable"` and `holding_cost=BorrowFee()`: names that cannot be shorted, and
  each name's borrow fee, from the data.

## What the agent observes

By default (`MarketObservation`, configured with `features=` and `exogenous=`), the
observation of `Market-v0`: a padded set of assets (`MaskedBox`) with a window of returns
and features each, the portfolio, the account and market-wide series, plus `"pending"`
(the order waiting to fill) with next-close execution.

To observe anything else, write an `ObservationBuilder`: declare the data it reads
(`fields`, `market_fields`), its space, and build the batched observation from a
`MarketView`, which gives windows and current values of returns, fields and market series
per slot, and the state of every lane's account (`weights`, `equity`, `peak`, `pending`,
`tradable`):

```python
import numpy as np
from sekai.finance import ObservationBuilder
from sekai.spaces import Box


class AlphaAndBook(ObservationBuilder):
    """Per slot: the alpha signal and the current weight (scaled to unit mean size)."""

    fields = ("alpha",)

    def space(self, env):
        return Box(-np.inf, np.inf, shape=(env.n_slots, 2), dtype=np.float32)

    def __call__(self, view):
        w = view.weights
        book = w / (w.abs().mean(-1, keepdim=True) + 1e-8)
        return torch.stack([view.field("alpha"), book], dim=-1)
```

This one is written for the torch backend; a builder for both backends uses the
operations of `view.backend` (`B.abs`, `B.mean`, `B.stack`...) instead.

## Backends and gradients

`backend="numpy"` (the default) computes in float64, like every other sekai environment.
`backend="torch"` keeps the data and the state on `device=` in `dtype=` (default: torch's
default dtype, usually float32). The accounting, costs, constraint projection, risk model
and rewards are the same code on both (see [`sekai.backend`](../api/backend.md)), and in
float64 the two give the same results.

With `env.differentiable = True`, `step()` builds the autograd graph from the actions to
the rewards; otherwise it runs without one. Prices do not react to the agent (impact is a
cost, not a price change), so the gradient of an episode's reward with respect to a
deterministic policy's parameters is exact and can be followed by backpropagation through
time: direct policy optimisation, whose low-variance gradients can find in a few dozen
iterations what a likelihood-ratio method (PPO) has to estimate from noisy rewards. Call
`env.detach_state()` (and detach the last observation) every `k` steps to truncate the
graph. A minimal trainer:

```python
from sekai.finance import Constraints, ScoresLeverage, SimpleReturn, SpreadCost, SquareRootImpact

env = BatchMarketEnv(
    128,
    scenarios=AlphaScenarios(),
    lookback=20,
    episode_length=60,
    execution="close",  # the MDP of the simulation: decide after the close, earn the next period
    observation=AlphaAndBook(),
    action=ScoresLeverage(Constraints(max_gross=2.0, max_position=0.1, dollar_neutral=True)),
    transaction_cost=SpreadCost() + SquareRootImpact(0.5),
    reward=SimpleReturn(),
    capital=1e8,
    backend="torch",
    autoreset=False,
)
env.differentiable = True
params = torch.tensor([1.0, 0.0, 0.0], requires_grad=True)  # alpha gain, book gain, leverage
optimiser = torch.optim.Adam([params], lr=0.05)

for iteration in range(60):
    result, rewards = env.reset(seed=iteration), []
    while not bool(result.dones.all()):
        obs = result.observations
        scores = params[0] * obs[..., 0] + params[1] * obs[..., 1]
        result = env.step(torch.cat([scores, params[2].expand(len(obs), 1)], dim=-1))
        rewards.append(result.rewards)
    loss = -torch.stack(rewards).mean() * 1e4
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
```

The full script,
[`examples/integrations/direct_policy_torch.py`](https://github.com/QianCapital/sekai/blob/master/examples/integrations/direct_policy_torch.py),
learns to lean on the current book, which cuts turnover and costs: on held-out scenarios
the Sharpe ratio rises from 4.4 to 7.2 and daily turnover falls from 46% to 34%.

**On a GPU**, the host queues each step's work and runs ahead while the device computes,
except where it has to read a value back: then it waits until the device has finished
everything queued. A step reads back once: which lanes finished, together with its
checks of the actions, of their projection and of the cost models' inputs (it raises
their errors, as on the CPU, before it changes any state). Keep your loop free of reads
too: `.item()`, `bool(...)`, `.cpu()` or an `if` on a tensor each add a wait. The loop
above tests `result.dones.all()` at every step; on a GPU, looping
`range(env.episode_length)` instead avoids that read (a finished lane is frozen, with
reward 0, until it is reset).

## The batched API

A `BatchMarketEnv` is a `sekai.vector.BatchEnv`. `reset(seed=None, options=None,
indices=None)` resets every lane, or the lanes in `indices`, and returns a `BatchReset`
(`observations`, `info` of the lanes reset, `indices`, `dones`). `step(actions)` takes a
`(num_envs, ...)` batch and returns a `BatchStep`: `observations`, `rewards`, `terminated`,
`truncated`, `info` (dicts of `(num_envs, ...)` arrays) and, with autoreset, the
`final_observations` of the lanes that finished and their `reset_info`. Without autoreset a
finished lane is frozen until it is reset: reward 0, its final flags, its state unchanged.

Two adapters connect it to the rest of sekai: `BatchVecEnv` is a `NativeVecEnv` with NumPy
at the boundary (what `make_vec` returns, and what Stable-Baselines3 sees), and
`UnbatchedEnv` is a single `Env` of a one-lane batch (what `make` returns; its attributes
are the lane's, so `env.action_mask()` and renderers work unchanged). Any environment that
keeps its state in arrays can subclass `BatchEnv` the same way.

**Speed.** On 8 CPU cores, `make_vec("Market-v1", num_envs=1024)` steps about 170,000
sub-envs per second with the default observation (10 assets, a 20-day window), against
about 11,000 for `Market-v0` in a `SyncVecEnv`.

## Market-v0 and Market-v1

`Market-v1` is `BatchMarketEnv` with one lane. With `execution="close"` it reproduces
`Market-v0` step for step (same slots, costs, financing, delistings, rewards and
observations, to rounding), which is how it is tested. What changes by default:

- orders fill at the next close, and the observation includes the pending order;
- infos add `holding_cost`, `net_exposure` and `drawdown` (and `variance` and `exposures`
  with a risk model);
- simulated episodes are drawn from the generator with a seed taken from the env's random
  generator, so a given seed gives different paths than `Market-v0`.

`Market-v0` keeps its behaviour unchanged.

## See also

- [Costs, accounts and rewards](execution.md): cost models, holding costs, constraints,
  risk models and reward functions.
- [Financial Markets environments](../environments/finance.md#market-v1).
- API: [`sekai.envs.finance.BatchMarketEnv`](../api/envs.md#sekai.envs.finance.BatchMarketEnv),
  [`sekai.vector.BatchEnv`](../api/vector.md#batched-environments),
  [`sekai.backend`](../api/backend.md), [`sekai.finance`](../api/finance.md).
