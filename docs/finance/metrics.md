# Performance metrics

Rewards are what an agent optimises; metrics are how you compare agents. Two policies
trained on different rewards (a log return and a differential Sharpe ratio, say) have
incomparable episode returns but comparable Sharpe ratios and drawdowns.
`sekai.finance.metrics` has the standard measures as plain functions, summaries that
compute them all at once, and `RecordPerformance`, a wrapper that attaches a summary to the
last step of every episode.

## Metric functions

Each function takes a 1-D series of **simple** per-period returns (`0.01` is +1% over the
period), raises `ValueError` if it is empty or not finite, and returns a float.
Annualisation uses `periods_per_year`: 252 for daily bars (the default), 52 for weekly,
12 for monthly, `252 * 390` for one-minute bars of a US equity session.

| Function | Definition |
|---|---|
| `total_return(r)` | $\prod_t (1 + r_t) - 1$ |
| `annualized_return(r, periods_per_year=252)` | $\prod_t (1 + r_t)^{p/n} - 1$, the compound annual growth rate |
| `annualized_volatility(r, periods_per_year=252)` | $\operatorname{std}(r)\,\sqrt p$ (sample standard deviation) |
| `sharpe_ratio(r, periods_per_year=252, risk_free=0.0)` | $\overline{e} / \operatorname{std}(e)\,\sqrt p$ with $e_t = r_t - r_f / p$; 0 if $e$ is constant |
| `sortino_ratio(r, periods_per_year=252, target=0.0)` | $\overline{e} \big/ \sqrt{\overline{\min(e, 0)^2}}\,\sqrt p$ with $e_t = r_t - \text{target}/p$; `inf` if no period is below target and $\overline e > 0$ |
| `drawdown_series(r)` | $1 - E_t / \max_{s \le t} E_s$ for the equity path $E$ starting at 1, length $n + 1$ |
| `max_drawdown(r)` | the largest drawdown, in $[0, 1]$ |
| `calmar_ratio(r, periods_per_year=252)` | annualised return / maximum drawdown |
| `value_at_risk(r, level=0.95)` | $-q_{1 - \text{level}}(r)$, the historical per-period loss exceeded with probability $1 - \text{level}$ |
| `conditional_value_at_risk(r, level=0.95)` | $-\,\mathbb E[r \mid r \le q_{1-\text{level}}]$, the expected shortfall |
| `hit_rate(r)` | the fraction of non-zero periods with a positive return |
| `equity_curve(r, initial=1.0)` | `initial` followed by $\text{initial}\cdot\prod_{s \le t}(1 + r_s)$ |
| `returns_from_equity(equity)` | the inverse of `equity_curve` |

Here $n$ is the number of periods, $p$ = `periods_per_year`, and $r_f$ and the target are
annual simple rates. Losses are limited, as in the environments: a return of -100% or worse
wipes the account out, equity stays at zero from then on, so `total_return` is at least -1,
drawdowns are at most 1 and `annualized_return` is -1.

**Rounding.** Returns computed from equities ($E_t / E_{t-1} - 1$) carry floating-point
rounding of a few $10^{-16}$. Excess returns within that of zero count as exactly zero,
and a series constant up to that rounding has zero volatility. So cash earning exactly the
risk-free rate has a Sharpe and a Sortino ratio of 0 at every rate, rather than a Sortino of
$-\sqrt p$ or `inf` decided by the sign of the rounding error.

```python
import numpy as np
import sekai
from sekai.finance import metrics

rng = np.random.default_rng(0)
returns = rng.normal(0.0004, 0.01, size=504)  # two years of daily returns

print(f"{metrics.total_return(returns):.4f}")  # 0.0351
print(f"{metrics.annualized_return(returns):.4f}")  # 0.0174
print(f"{metrics.annualized_volatility(returns):.4f}")  # 0.1612
print(f"{metrics.sharpe_ratio(returns):.3f}")  # 0.187
print(f"{metrics.sharpe_ratio(returns, risk_free=0.04):.3f}")  # -0.061
print(f"{metrics.max_drawdown(returns):.4f}")  # 0.1905
print(f"{metrics.value_at_risk(returns):.4f} {metrics.conditional_value_at_risk(returns):.4f}")  # 0.0145 0.0214
print(metrics.equity_curve(returns, initial=100.0)[:3].round(2))  # [100.   100.17 100.07]
print(metrics.total_return([0.5, -1.2, 0.3]), metrics.max_drawdown([0.5, -1.2, 0.3]))  # -1.0 1.0
```

The annualised Sharpe ratio assumes independent periods: it scales the per-period ratio
by $\sqrt p$. Pass the `periods_per_year` that matches the data, not 252 by habit: the
weekday history of [Market data](data.md) has 261 rows a year.

## Summaries

`performance_summary(returns, periods_per_year=252, *, risk_free=0.0, level=0.95,
turnover=None, costs=None)` computes everything at once into a frozen
`PerformanceSummary`, which prints as a table and converts with `as_dict()`:

```python
from sekai.finance import performance_summary

summary = performance_summary(returns)
print(summary)
print(round(summary.sharpe_ratio, 3), summary.as_dict()["n_periods"])  # 0.187 504
```

```text
periods                   504
total return            3.51%
annualised return       1.74%
annualised volatility  16.12%
Sharpe ratio            0.187
Sortino ratio           0.265
max drawdown           19.05%
Calmar ratio            0.091
value at risk           1.45%
conditional VaR         2.14%
hit rate               49.60%
```

`turnover` and `costs` (per-period series) add the mean turnover and the total cost.

Returns are undefined for a process that starts from zero wealth, such as a market maker's
mark-to-market P&L. `pnl_summary(pnl, periods_per_year=252)` summarises per-period P&L
increments in absolute units into a `PnLSummary`: total, mean and volatility of P&L, a
Sharpe ratio of the increments, the maximum drawdown of cumulative P&L from its running
peak (which starts at 0), and the hit rate.

```python
from sekai.finance import pnl_summary

print(pnl_summary(rng.normal(0.2, 1.0, size=200), periods_per_year=200))
```

```text
periods              200
total P&L        39.9801
mean P&L          0.1999
P&L volatility  0.963544
Sharpe ratio       2.934
max drawdown     7.87073
hit rate          59.50%
```

With `periods_per_year` set to the number of periods in a session, the Sharpe ratio is per
session rather than per year.

## RecordPerformance

`RecordPerformance(env, *, periods_per_year=252, equity_key="equity", pnl_key=None,
turnover_key="turnover", cost_key="cost", risk_free=0.0, deque_size=100)` reads the
account from each step's `info` and, on the final step of an episode, attaches a summary
as `info["performance"]`. It also appends it to `summaries` (the last `deque_size`
episodes), keeps it as `last_summary`, and `summary()` averages every field over the stored
episodes.

| Mode | Selected by | Reads | Summary |
|---|---|---|---|
| equity | default | `info[equity_key]`, relative account value; returns are $E_t / E_{t-1} - 1$ (the reset info gives $E_0$, else 1) | `PerformanceSummary`, with mean turnover and total cost from `turnover_key` / `cost_key` when the env reports them |
| P&L | `pnl_key="..."` | `info[pnl_key]`, cumulative wealth or P&L in absolute units, possibly zero or negative | `PnLSummary` of its increments |

What the built-in environments report:

| Environment | Mode | Keys |
|---|---|---|
| `Trading-v0`, `Portfolio-v0`, `Market-v0` | equity (the defaults work) | `equity`, `turnover`, `cost` |
| `MarketMaking-v0`, `LimitOrderBook-v0` | P&L: `pnl_key="wealth"` | `wealth` (cash plus inventory at the mid) |

```python
from sekai.finance import RecordPerformance

env = RecordPerformance(sekai.make("Portfolio-v0"))
for seed in range(20):
    result = env.reset(seed=seed)
    while not result.done:
        result = env.step(np.ones(6, np.float32))  # equal weight, rebalanced daily
print(result.info["performance"])
print({key: round(value, 3) for key, value in env.summary().items()})
```

```text
periods                      252
total return               0.71%
annualised return          0.71%
annualised volatility      8.37%
Sharpe ratio               0.127
Sortino ratio              0.177
max drawdown              10.80%
Calmar ratio               0.066
value at risk              0.84%
conditional VaR            1.19%
hit rate                  48.81%
mean turnover          0.0124452
total cost             0.0031362
{'n_periods': 252.0, 'total_return': 0.023, 'annualized_return': 0.023, 'annualized_volatility': 0.083, 'sharpe_ratio': 0.276, 'sortino_ratio': 0.492, 'max_drawdown': 0.088, 'calmar_ratio': 0.782, 'value_at_risk': 0.008, 'conditional_value_at_risk': 0.01, 'hit_rate': 0.5, 'mean_turnover': 0.013, 'total_cost': 0.003}
```

A market maker's account starts at zero wealth, so use P&L mode. With 200 steps a session,
`periods_per_year=200` makes the Sharpe ratio a per-session one:

```python
mm = RecordPerformance(sekai.make("MarketMaking-v0"), pnl_key="wealth", periods_per_year=200)
for seed in range(20):
    result = mm.reset(seed=seed)
    while not result.done:
        result = mm.step(mm.unwrapped.avellaneda_stoikov_action())
print({key: round(value, 2) for key, value in mm.summary().items()})
```

```text
{'n_periods': 200.0, 'total_pnl': 59.0, 'mean_pnl': 0.29, 'pnl_volatility': 0.46, 'sharpe_ratio': 9.07, 'max_drawdown': 1.06, 'hit_rate': 0.74}
```

**With AutoReset.** Around an inner `AutoReset`, `reset()` is never called between
episodes. `RecordPerformance` then reads the finished episode's last value from
`info["final_info"]` and the next episode's starting value from `info["reset_info"]`, so
back-to-back episodes are still summarised separately:

```python
from sekai.wrappers import AutoReset

LONG = 2
env = RecordPerformance(AutoReset(sekai.make("Trading-v0", episode_length=63)))
env.reset(seed=0)
for _ in range(252):
    env.step(LONG)
print([f"{s.total_return:+.2%}" for s in env.summaries])  # ['+20.64%', '-7.36%', '-9.32%', '+0.50%']
```

A `reset()` in the middle of an episode discards the unfinished one.

**With vector environments.** `RecordPerformance` wraps single environments; pass it in
`wrappers=` to get per-sub-env summaries (`make_vec` then uses `SyncVecEnv`, since a native
vector env has no per-env objects to wrap):

```python
vec = sekai.make_vec("Portfolio-v0", 4, episode_length=50, wrappers=[RecordPerformance])
result = vec.reset(seed=0)
for _ in range(50):
    result = vec.step(np.ones((4, 6), np.float32))
print(type(vec).__name__, [round(info["performance"].sharpe_ratio, 2) for info in result.infos])
```

```text
SyncVecEnv [1.88, -0.3, 0.02, -1.54]
```

The summaries ride on the final step's info, inside `final_info` as well.

## See also

- [Market data and backtesting](data.md#walk-forward-evaluation): summaries per walk-forward fold.
- [Statistics guide](../guide/statistics.md): episode returns and lengths for any environment.
- [`sekai.finance.metrics` API](../api/finance.md#metrics).
