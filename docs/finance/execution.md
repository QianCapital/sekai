# Costs, accounts and rewards

The finance environments keep the account in *weight space*: positions are fractions of
equity, the rest is cash, and equity starts at 1. One decision runs through three
modules:

1. `sekai.finance.account`: the action becomes target weights that satisfy the account's
   rules (long-only or long/short, a leverage cap, nothing in untradable assets);
2. `sekai.finance.costs`: moving from the current (drifted) weights to the targets costs a
   fraction of equity;
3. `sekai.finance.account` again: over the period the positions earn the assets' returns
   and cash earns (or pays) interest; the equity grows by a factor $g_t$ and the weights
   drift;
4. `sekai.finance.rewards`: a reward function turns the step's outcome into the reward.

Everything here broadcasts over leading batch dimensions, so the same objects serve a
single environment and every sub-env of a native vector env at once.

## Cost models

A `CostModel` is called with `trades`, the signed change in each asset's weight (shape
`(..., n)`), and returns the cost as a fraction of equity (shape `(...)`):

| Model | Cost as a fraction of equity | `requires` |
|---|---|---|
| `NoCost()` | $0$ | |
| `ProportionalCost(rate)` | $\sum_i \kappa_i\,\lvert\Delta w_i\rvert$, one rate or one per asset | |
| `SpreadCost(half_spread)` | $\sum_i h_i\,\lvert\Delta w_i\rvert$: crossing half the bid-ask spread | |
| `SpreadCost(fallback_spread=None)` | the same with $h_i$ = half the data's quoted `spread` (the cross-sectional median where an asset's is missing or negative; a zero spread costs nothing) | `spread` |
| `FixedCost(amount, threshold=1e-6)` | `amount` per asset traded (weight change above `threshold`), divided by equity in currency | `equity` |
| `SquareRootImpact(coefficient=0.5, fallback_volatility=None)` | square-root market impact (below) | `equity`, `volatility`, `volume` |

**Missing inputs are never free.** Only a non-finite value (NaN) counts as missing; a
finite zero is a valid input, so a zero quoted spread costs nothing and a zero volatility
(a flat window) gives zero impact. A negative value is invalid and treated like a missing
one. A quoted spread or a volatility that is missing for one asset is replaced by the
cross-sectional median of the others. When *no* asset on a row has one (no spreads quoted
that day, no return history yet), there is nothing to take a median of, and trading on
that row raises a `ValueError` rather than costing nothing. To price such trades anyway,
give the model an explicit value: `fallback_spread` (a full spread, as a fraction of the
price) or `fallback_volatility` (per step). Untraded rows never need one.

A plain number passed as `transaction_cost=` to an environment means
`ProportionalCost(number)`; the default `1e-3` is 10 bp of every unit traded.

**Square-root impact.** Empirically, trading $q$ in currency against a traded value $V$
per period moves the price by about $c\,\sigma\sqrt{q/V}$, where $\sigma$ is the
per-period volatility and $c$ a coefficient of order 0.1–1 (Almgren et al. 2005; Tóth
et al. 2011). The trade pays that move on its whole size, so with equity $E$ in currency
and $q_i = \lvert\Delta w_i\rvert E$,

$$
\text{cost} = \sum_i c\,\sigma_i \sqrt{\frac{\lvert\Delta w_i\rvert\,E}{V_i}}\;\lvert\Delta w_i\rvert ,
$$

which grows as $\lvert\Delta w\rvert^{3/2}$: large rebalances in illiquid names get
expensive, which a proportional cost cannot express. The impact per unit traded,
$c\,\sigma_i\sqrt{q_i/V_i}$, is capped at 1 (a trade never costs more than its value); an
asset with zero or missing volume has no liquidity, so trading it costs the full traded
value; a missing volatility is replaced by the median volatility of the other assets (or
`fallback_volatility` when no asset has one); and a zero volatility gives zero impact.

Any trading cost of 100% of equity or more wipes the account out: the environments
terminate the episode rather than let equity change sign.

```python
import numpy as np
import sekai
from sekai.finance import FixedCost, ProportionalCost, SpreadCost, SquareRootImpact

trades = np.array([0.10, -0.05, 0.0])  # buy 10% of equity in asset 0, sell 5% of asset 1
context = dict(equity=1e6,  # account value in currency
               volatility=np.array([0.02, 0.01, 0.015]),  # per-step
               volume=np.array([5e7, 2e7, 1e7]))  # traded value per step, in currency

for model in (ProportionalCost(1e-3), SpreadCost(half_spread=2e-4), FixedCost(5.0),
              SquareRootImpact(0.5)):
    print(f"{model!r:31s} {float(model(trades, **context)) * 1e4:.3f} bp")
```

```text
ProportionalCost(0.001)         1.500 bp
SpreadCost(half_spread=0.0002)  0.300 bp
FixedCost(5.0)                  0.100 bp
SquareRootImpact(0.5)           0.572 bp
```

Models compose with `+` into a `CompositeCost` whose `requires` is the union of its
parts', so a realistic schedule reads like its definition:

```python
costs = SpreadCost(half_spread=2e-4) + SquareRootImpact(0.5) + FixedCost(5.0)
print(costs)  # SpreadCost(half_spread=0.0002) + SquareRootImpact(0.5) + FixedCost(5.0)
print(sorted(costs.requires))  # ['equity', 'volatility', 'volume']
print(f"{float(costs(trades, **context)) * 1e4:.3f} bp")  # 0.972 bp
```

Batches work the same way, one row per sub-env:

```python
batch = np.array([[0.1, -0.05, 0.0], [0.5, 0.5, 0.0]])
cost = costs(batch, equity=np.array([1e6, 1e6]), volatility=np.full((2, 3), 0.02),
             volume=np.full((2, 3), 5e7))
print((cost * 1e4).round(3))  # [ 1.005 12.1  ]
```

### Which inputs each model needs

Environments compute only the inputs named in the cost model's `requires`:

| Input | What `Trading-v0`, `Portfolio-v0` and `Market-v0` pass |
|---|---|
| `equity` | account value in currency: `capital=` (default 1,000,000) times the current relative equity |
| `volatility` | per-step standard deviation of each asset's log returns over the observation's `lookback` window |
| `volume` | the data field `"volume"` (units traded per step) times the current price, i.e. traded value |
| `spread` | the data field `"spread"`: the quoted bid-ask spread as a fraction of the price |

`volume` and `spread` therefore need historical `MarketData` with those fields; asking for a
model whose inputs the data cannot supply raises a `ValueError` naming the missing field
when the environment is built. So does a volatility-based model (`SquareRootImpact`) with
`lookback=1`: one return gives no volatility estimate, so every trade would be unpriced;
use `lookback >= 2` (a fallback does not help, since a one-return window can look flat).
Here a four-asset history with volumes and spreads, and the cost of the same first
rebalance (cash to 20% in each asset) for accounts of different sizes:

```python
from sekai.envs.finance import simulate_gbm
from sekai.finance import MarketData

prices = simulate_gbm(np.random.default_rng(0), 1000, s0=[50, 120, 30, 80],
                      mu=0.05, sigma=[0.25, 0.18, 0.4, 0.08])
data = MarketData(prices, fields={"volume": np.full(prices.shape, 2e6),
                                  "spread": np.full(prices.shape, 5e-4)})

for capital in (1e6, 1e8, 1e9):
    env = sekai.make("Portfolio-v0", data=data, lookback=60, episode_length=252,
                     transaction_cost=SpreadCost() + SquareRootImpact(0.5), capital=capital)
    env.reset(seed=0)
    info = env.step(np.ones(5, np.float32)).info
    print(f"capital {capital:>13,.0f}: turnover {info['turnover']:.1f}, cost {info['cost'] * 1e4:6.1f} bp")
```

```text
capital     1,000,000: turnover 0.8, cost    5.3 bp
capital   100,000,000: turnover 0.8, cost   34.9 bp
capital 1,000,000,000: turnover 0.8, cost  106.1 bp
```

The spread costs the same 2 bp at every size; impact grows with the square root of size.

```python
try:
    sekai.make("Portfolio-v0", transaction_cost=SpreadCost())  # simulated: no spread data
except ValueError as error:
    print(error)
```

```text
SpreadCost(half_spread=from data) needs a 'spread' data field (fraction of the price), which a simulated market does not provide.
```

### Writing a cost model

Subclass `CostModel`, implement `__call__(trades, *, equity=None, volatility=None,
volume=None, spread=None)` returning an array of the batch shape, and set `requires` to the
inputs you read. A broker commission of 1 bp with a minimum ticket of 10 currency units:

```python
from sekai.finance import CostModel


class MinimumCommission(CostModel):
    requires = frozenset({"equity"})

    def __init__(self, rate=1e-4, minimum=10.0):
        self.rate, self.minimum = rate, minimum

    def __call__(self, trades, *, equity=None, **context):
        notional = np.abs(trades) * np.asarray(equity)[..., None]
        fees = np.where(notional > 0, np.maximum(self.rate * notional, self.minimum), 0.0)
        return fees.sum(axis=-1) / equity


print(f"{float(MinimumCommission()(trades, equity=1e6)) * 1e4:.2f} bp")  # 0.20 bp
print(f"{float(MinimumCommission()(trades, equity=1e5)) * 1e4:.2f} bp")  # 2.00 bp
```

## Account rules

`AccountConfig` holds the constraints and financing terms:

| Field | Default | Meaning |
|---|---|---|
| `allow_short` | `False` | allow negative weights |
| `max_leverage` | `1.0` | cap on gross exposure $\sum_i \lvert w_i\rvert$ |
| `cash_rate` | `0.0` | annual interest earned on positive cash |
| `margin_rate` | `None` | annual interest paid on borrowed (negative) cash; `None` means `cash_rate` |
| `borrow_rate` | `0.0` | annual fee on the value of short positions |

`Market-v0` takes these as constructor arguments of the same names (its long/short action
space also caps each weight at ±1, whatever `max_leverage`); `Portfolio-v0` has
`allow_short` with a leverage of 1 and no financing.

**Projection.** `project_weights(scores, tradable, config, *, cash_score=None)` maps raw
action values to admissible target weights. Long-only, scores (and an optional cash score)
are clipped at zero and normalised to sum to `max_leverage`; all zeros means all cash.
Long/short, values are clipped to $\pm$`max_leverage` and scaled down if gross exposure
exceeds it. Untradable assets get weight 0 either way:

```python
from sekai.finance import AccountConfig
from sekai.finance.account import project_weights, settle

long_only = AccountConfig()
everything = np.array([True, True, True])
print(project_weights([2.0, 1.0, 1.0], everything, long_only))  # [0.5  0.25 0.25]
print(project_weights([2.0, 1.0, 1.0], everything, long_only, cash_score=4.0))  # [0.25  0.125 0.125]
print(project_weights([2.0, 1.0, 1.0], [True, False, True], long_only))  # [0.66666667 0.         0.33333333]

levered = AccountConfig(allow_short=True, max_leverage=2.0, cash_rate=0.04,
                        margin_rate=0.06, borrow_rate=0.01)
w = project_weights([1.5, -1.0, 0.5], everything, levered)
print(w.round(4), np.abs(w).sum().round(4))  # [ 1.     -0.6667  0.3333] 2.0
```

**Settlement.** `settle(weights, returns, *, cost, dt, config)` holds the weights over one
period of length `dt` years, after paying `cost`. With cash weight
$c = 1 - \sum_i w_i$ and asset returns $r_i$:

$$
f = c\,\rho\,\Delta t \;-\; b\,\Delta t\sum_i \max(-w_i, 0), \qquad
\rho = \begin{cases}\text{cash\_rate} & c \ge 0\\ \text{margin\_rate} & c < 0\end{cases}
$$

$$
g = (1 - \text{cost})\Big(1 + \sum_i w_i r_i + f\Big), \qquad
\tilde w_i = \frac{w_i\,(1 + r_i)}{1 + \sum_j w_j r_j + f},
$$

where $b$ is `borrow_rate`, $g$ the equity growth factor (`growth`), $f$ the financing
(`financing`) and $\tilde w$ the drifted weights the next rebalance starts from
(`drifted`). If $g \le 0$ the account is wiped out (`wiped_out`) and the drifted weights are
zero. Financing accrues once per step of `dt` years: on daily rows, weekends and holidays
accrue nothing extra. The cash weight includes the proceeds of short sales, which therefore
earn the full `cash_rate`.

```python
s = settle(w, [0.01, -0.02, 0.0], cost=5e-4, dt=1 / 252, config=levered)
print(f"{float(s.growth):.6f} {float(s.financing):.2e}", s.drifted.round(4), bool(s.wiped_out))
```

```text
1.022848 2.65e-05 [ 0.9869 -0.6384  0.3257] False
```

The net exposure is $1 - 2/3 + 1/3 = 2/3$, so a third of equity is cash earning 4%,
less the 1% fee on the 2/3 short. `turnover(target, current)` gives
$\sum_i \lvert w_i - \tilde w_i\rvert$, the quantity proportional costs charge.

**Delisting.** In `Market-v0`, an asset whose price disappears during a period earns
`delisting_return` (default 0) over that period and its position becomes cash. Here `BBB`
delists after the second decision and recovers half its value:

```python
prices = np.array([[100.0, 50.0], [100.0, 50.0], [101.0, 50.0], [102.0, np.nan]])
env = sekai.make("Market-v0", data=MarketData(prices, assets=["AAA", "BBB"]), lookback=1,
                 episode_length=2, transaction_cost=0.0, delisting_return=-0.5)
result = env.reset(seed=0)
for _ in range(2):
    result = env.step(np.array([1.0, 1.0, 0.0], np.float32))  # half in each, no cash
    print(result.observation["portfolio"].round(4), round(result.info["equity"], 4))
```

```text
[0.5025 0.4975 0.    ] 1.005
[0.6689 0.     0.3311] 0.7587
```

The portfolio observation lists the weights per slot with cash last; `BBB`'s slot is
empty after the delisting, and it is masked out of the next action.

## Reward functions

A `RewardFunction` maps the step's `StepOutcome` to a reward. The outcome holds
`log_return` ($\log g_t$, net of all costs), `simple_return` ($g_t - 1$), `equity` (after the
step, relative to the start of the episode), `turnover`, `cost` and `pnl` (the change in
equity, $E_t - E_{t-1}$); build one with
`StepOutcome.from_growth(growth, equity=..., turnover=..., cost=..., previous_equity=...)`.
The built-in environments always pass `previous_equity`. Without it `pnl` is `None`, and
`ProfitAndLoss` recovers the change from the equity after the step, which is impossible
on a wipe-out (equity 0), so it raises there instead of rewarding the loss with 0.

| Reward | Per step | Summed over an episode |
|---|---|---|
| `LogReturn()` (default) | $\log g_t$ | $\log(E_T / E_0)$, the Kelly objective |
| `SimpleReturn()` | $g_t - 1$ | |
| `ProfitAndLoss()` | $E_t - E_{t-1}$, exact (a wipe-out step loses all of $E_{t-1}$) | $E_T - 1$, P&L in units of initial equity |
| `MeanVariance(risk_aversion=1.0)` | $r_t - \tfrac{\lambda}{2} r_t^2$ on the log return | |
| `TurnoverPenalty()` | $\tau_t$ (subtract it) | total turnover |
| `DrawdownPenalty()` | how much the drawdown deepened (subtract it) | total deepening of drawdowns |
| `DifferentialSharpe(eta=0.01)` | the differential Sharpe ratio | approximately maximises the Sharpe ratio |

**Drawdown penalty.** With running peak $P_t = \max(1, \max_{s \le t} E_s)$ and drawdown
$D_t = 1 - E_t / P_t$, the penalty is $\max(D_t - D_{t-1}, 0)$: zero while equity is at a
peak or recovering, positive only when the drawdown gets deeper.

**Differential Sharpe ratio** (Moody & Saffell 2001). Exponential moving estimates of the
first and second moments of the step returns $R_t$ (simple returns),

$$
A_t = A_{t-1} + \eta\,(R_t - A_{t-1}), \qquad B_t = B_{t-1} + \eta\,(R_t^2 - B_{t-1}),
$$

define a Sharpe ratio $S_t = A_t / \sqrt{B_t - A_t^2}$; the reward is its sensitivity to
the latest return,

$$
D_t = \frac{B_{t-1}\,\Delta A_t - \tfrac12 A_{t-1}\,\Delta B_t}{\big(B_{t-1} - A_{t-1}^2\big)^{3/2}},
\qquad \Delta A_t = R_t - A_{t-1},\quad \Delta B_t = R_t^2 - B_{t-1},
$$

an online, per-step signal whose sum approximately maximises the Sharpe ratio. It is 0
until the variance estimate is positive; $\eta$ is about one over the averaging window.

Rewards compose with `+`, `-`, `*` and constants, and `reset()` reaches every stateful
part:

```python
from sekai.finance import DrawdownPenalty, LogReturn, StepOutcome, TurnoverPenalty

reward = LogReturn() - 0.5 * DrawdownPenalty() - 1e-3 * TurnoverPenalty()
print(reward)  # LogReturn() - 0.5 * DrawdownPenalty() - 0.001 * TurnoverPenalty()

drawdown, equity = DrawdownPenalty(), 1.0
for growth in (1.02, 0.97, 0.99, 1.05, 0.98):
    equity *= growth
    outcome = StepOutcome.from_growth(growth, equity=equity, turnover=0.1, cost=1e-4)
    print(f"equity {equity:.4f}  drawdown penalty {float(drawdown(outcome)):.4f}  "
          f"reward {float(reward(outcome)):+.5f}")
```

```text
equity 1.0200  drawdown penalty 0.0000  reward +0.01970
equity 0.9894  drawdown penalty 0.0300  reward -0.04556
equity 0.9795  drawdown penalty 0.0097  reward -0.01500
equity 1.0285  drawdown penalty 0.0000  reward +0.04869
equity 1.0079  drawdown penalty 0.0200  reward -0.03030
```

Pass a reward to any of `Trading-v0`, `Portfolio-v0` or `Market-v0` (and their native vector
versions) with `reward=`. Each environment keeps its own copy (a deep copy, so one
stateful reward passed to several environments, or to `make_vec`, never shares state
between them) and resets it at every episode start. Setting `env.reward_fn` later stores
a copy too, so `vec.set_attr("reward_fn", DifferentialSharpe(0.1))`, which hands every
sub-env the same object, still gives each its own state:

```python
from sekai.finance import DifferentialSharpe

env = sekai.make("Portfolio-v0", reward=DifferentialSharpe(eta=0.01))
result, total = env.reset(seed=0), 0.0
while not result.done:
    result = env.step(np.ones(6, np.float32))
    total += result.reward
print(f"{total:.2f}, final equity {result.info['equity']:.4f}")  # 37.52, final equity 1.0203
```

The scale of a differential-Sharpe episode return is not that of a log return; compare
policies on the metrics of the next page, not on rewards of different kinds.

**Batches and partial resets.** Outcome fields may be arrays with one entry per sub-env.
Stateful rewards (`DrawdownPenalty`, `DifferentialSharpe`) then keep per-env state, and
`reset(mask)` restarts only the sub-envs where `mask` is True, which is what a native
vector env does when some of its sub-envs finish:

```python
sharpe = DifferentialSharpe(eta=0.1)
rng = np.random.default_rng(0)
for _ in range(5):
    growth = 1.0 + rng.normal(0.001, 0.01, size=4)
    rewards = sharpe(StepOutcome.from_growth(growth, equity=np.ones(4), turnover=np.zeros(4),
                                             cost=np.zeros(4)))
print(rewards.round(3))  # [-0.427 -0.464  0.849  2.467]
sharpe.reset(np.array([True, False, False, False]))  # sub-env 0 starts a new episode
```

**Writing a reward.** Subclass `RewardFunction` and implement `__call__(outcome)` (with
NumPy operations, so it batches); override `reset(mask=None)` if it keeps state. A
downside-only penalty:

```python
from sekai.finance import RewardFunction


class Downside(RewardFunction):
    """The step's log return when negative, else 0."""

    def __call__(self, outcome):
        return np.minimum(outcome.log_return, 0.0)


reward = LogReturn() + 0.5 * Downside()  # losses count 1.5 times
outcome = StepOutcome.from_growth(0.98, equity=0.98, turnover=0.0, cost=0.0)
print(round(float(reward(outcome)), 5))  # -0.0303
```

## See also

- [Finance environments](../environments/finance.md) for `transaction_cost=`, `reward=`
  and the account options in each environment.
- [Performance metrics](metrics.md) to compare policies whatever they were trained on.
- [`sekai.finance` API](../api/finance.md#costs).
