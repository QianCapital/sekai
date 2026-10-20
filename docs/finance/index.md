# Finance toolkit

`sekai.finance` holds the parts the finance environments are built from: market data and
simulators, causal features, transaction costs, account rules, reward functions,
performance metrics and an order book. Use them to configure the
[built-in environments](../environments/finance.md) (which accept most of them as
arguments) or to assemble your own.

| Module | For | Main names |
|---|---|---|
| `data` | historical panels with listings and delistings; leak-free splits | `MarketData`, `Time` |
| `sampling` | where each episode starts in a history | `RandomStart`, `SequentialStart`, `FixedStart` |
| `simulators` | synthetic markets, batch-first | `GBM`, `Heston`, `RegimeSwitching`, `ListingProcess`, `Randomized`, ... |
| `features` | point-in-time transforms and a look-ahead detector | `momentum`, `rolling_zscore`, `asof_join`, `check_causal`, ... |
| `costs` | transaction costs and market impact | `ProportionalCost`, `SpreadCost`, `SquareRootImpact`, ... |
| `account` | target weights, leverage, financing, drift | `AccountConfig`, `project_weights`, `settle`, `Settlement` |
| `rewards` | what the agent optimises | `LogReturn`, `DifferentialSharpe`, `DrawdownPenalty`, ... |
| `metrics` | how policies compare | `performance_summary`, `RecordPerformance`, ... |
| `universe` | stable slots for a changing set of instruments | `SlotMap` |
| `lob` | a matching engine and background order flow | `OrderBook`, `ZeroIntelligenceFlow` |

The classes are importable from `sekai.finance` directly; the functions live in their
modules (`sekai.finance.features.momentum`, `sekai.finance.metrics.sharpe_ratio`, ...).
NumPy is the only dependency (pandas is optional, for `MarketData.from_pandas`).

## How the environments are assembled

`Market-v0`, the most general of them, runs one decision like this (`Trading-v0` and
`Portfolio-v0` are special cases with one asset, or a fixed universe):

```text
 MarketData (history)          or      MarketModel.simulate()                data, simulators
   prices, listing mask, fields, exogenous series / latent state
        │
        │  features computed once over the history (causal)                 features
        ▼
 episode window   ◄── sampler, or reset(options={"start": ...})             sampling
        │
        ▼
 observation: assets in stable slots (SlotMap) as a MaskedBox,              universe
              portfolio weights, account state, market series
        │
 action ──► target weights within the account's rules (project_weights)     account
        │
        ▼
 trading cost of the rebalance (CostModel)                                  costs
        │
        ▼
 hold over the period: returns, financing, drift (settle)                   account
        │
        ▼
 reward = RewardFunction(StepOutcome)                                       rewards
 info:   equity, turnover, cost, timestamp, dt  ──►  RecordPerformance      metrics
```

Every stage after the data is vectorised over a leading batch axis, which is how the
native vector environments step a thousand sub-envs at once with the same code.

## End to end

A historical panel with listings and a volume field, a causal feature, a leak-free split,
`Market-v0` with spread and impact costs and a drawdown-aware reward, and a performance
summary of an equal-weight baseline on the test period:

```python
import numpy as np
import sekai
from sekai.finance import (GBM, DrawdownPenalty, ListingProcess, LogReturn, MarketData,
                           RecordPerformance, SpreadCost, SquareRootImpact, features)

# A synthetic five-year panel of 12 stocks that list and delist (load real data the same way).
rng = np.random.default_rng(0)
prices = ListingProcess(GBM(mu=0.06, sigma=rng.uniform(0.15, 0.4, 12)), min_listed=4).simulate(rng, 1259).prices
dates = np.busday_offset("2020-01-01", np.arange(1260), roll="forward")
data = MarketData(prices, timestamps=dates, fields={"volume": np.full(prices.shape, 5e5)})
data = data.with_fields(momentum=np.tanh(10 * features.momentum(data.prices, 60)))
train, test = data.split(0.75, embargo=5, history=20)
print(test)  # MarketData(T=330, N=12, fields=['close', 'momentum', 'volume'], warmup=20, 2023-07-26 .. 2024-10-29)

env = sekai.make(
    "Market-v0",
    data=test,
    features=["momentum"],
    lookback=20,
    episode_length=252,
    transaction_cost=SpreadCost(half_spread=5e-4) + SquareRootImpact(0.5),
    reward=LogReturn() - 0.5 * DrawdownPenalty(),
    sampler="sequential",
)
env = RecordPerformance(env, periods_per_year=data.periods_per_year)

result = env.reset(seed=0)
print(result.info["timestamp"])  # 2023-08-23
while not result.done:
    mask = sekai.get_action_mask(env)  # 1 for tradable slots, then cash
    result = env.step(np.append(mask[:-1], 0.0).astype(np.float32))  # equal weight, no cash
print(result.info["performance"])
```

```text
periods                       252
total return               14.52%
annualised return          15.06%
annualised volatility       9.61%
Sharpe ratio                1.508
Sortino ratio               2.355
max drawdown                8.66%
Calmar ratio                1.740
value at risk               0.83%
conditional VaR             1.11%
hit rate                   51.98%
mean turnover           0.0205383
total cost             0.00349695
```

The first decision falls on the first test day after the embargo, with the 20 rows of
history before it filling the lookback window. A learning agent would be trained on an
environment built the same way from `train`.

## Pages

| Page | Covers |
|---|---|
| [Market data and backtesting](data.md) | `MarketData`, pandas import, slicing, `split` and `walk_forward` with embargo and history, episode samplers and the `start` option, point-in-time features, `asof_join`, `check_causal`, a walk-forward evaluation |
| [Market simulators](simulators.md) | the `MarketModel` interface, each model's dynamics, dynamic universes, domain randomisation, latent state, custom models |
| [Costs, accounts and rewards](execution.md) | cost models and their inputs, account rules and settlement, reward functions and their composition |
| [Performance metrics](metrics.md) | metric definitions, summaries, `RecordPerformance` |
| [Order books](order-book.md) | `OrderBook`, zero-intelligence order flow and its parameters |
| [Finance environments](../environments/finance.md) | `Trading-v0`, `Portfolio-v0`, `Market-v0`, `MarketMaking-v0`, `OptimalExecution-v0`, `LimitOrderBook-v0`, `LimitOrderBookMarket`, native vector versions, baselines |
| [API reference](../api/finance.md) | every class and function in `sekai.finance` |

For variable-size observations and action masks in general, see the
[dynamic observations guide](../guide/dynamic.md); for timestamps, latency and event-driven
stepping, the [time guide](../guide/time.md).
