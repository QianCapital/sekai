# Performance metrics

Rewards are what an agent optimises; metrics are how you compare agents. Two policies
trained on different rewards (a log return and a differential Sharpe ratio, say) have
incomparable episode returns but comparable Sharpe ratios and drawdowns.
`sekai.finance.metrics` has the standard measures as plain functions, summaries that
compute them all at once, `RecordPerformance`, a wrapper that attaches a summary to the
last step of every episode, and [statistics that correct for selection
bias](#overfitting-and-multiple-testing) when the best of many backtests is chosen.

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
| `returns_from_equity(equity)` | the inverse of `equity_curve`: $E_t / E_{t-1} - 1$, with -1 for a wipe-out (the first $E_t \le 0$) and 0 after it, where equity must stay 0 |

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
mark-to-market P&L. `pnl_summary(pnl, periods_per_year=252, *, scale=0.0)` summarises
per-period P&L increments in absolute units into a `PnLSummary`: total, mean and volatility
of P&L, a Sharpe ratio of the increments, the maximum drawdown of cumulative P&L from its
running peak (which starts at 0), and the hit rate. Increments computed as differences of a
cumulative wealth $W$ carry rounding of order $10^{-16}\,\lvert W\rvert$; pass the wealth's
size (its largest absolute value) as `scale`, and P&L that is constant up to that rounding
has zero volatility and a Sharpe ratio of 0, as for returns. `RecordPerformance` does so.

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

## Overfitting and multiple testing

Train twenty seeds, sweep a grid of hyperparameters, keep the best backtest: its Sharpe
ratio is the largest of many noisy estimates and overstates what the strategy will do next.
Even a single Sharpe ratio is less conclusive than it looks over a short record or with
negatively skewed, fat-tailed returns. These functions correct for both. Like the metric
functions they take simple per-period returns, and Sharpe ratios are annualised with
`periods_per_year`, as `sharpe_ratio` returns them.

| Function | Question it answers |
|---|---|
| `probabilistic_sharpe_ratio(r, benchmark_sharpe=0.0, periods_per_year=252)` | How likely is the true Sharpe ratio above the benchmark, given the record's length, skewness and kurtosis? (PSR) |
| `minimum_track_record_length(r, benchmark_sharpe=0.0, confidence=0.95, periods_per_year=252)` | How many periods would this Sharpe ratio, skewness and kurtosis need for a PSR of `confidence`? (MinTRL) |
| `expected_maximum_sharpe(n_trials, sharpe_std)` | What Sharpe ratio does the best of `n_trials` skill-less trials reach by luck alone? |
| `deflated_sharpe_ratio(r, trial_sharpes, periods_per_year=252)` | Is the selected strategy better than the best of luck? The PSR against that expected maximum (DSR) |
| `probability_of_backtest_overfitting(returns, n_partitions=16, metric=None)` | From the returns of every trial: how often does the in-sample best land at or below the out-of-sample median? (PBO) |

With $\widehat{SR}$ the observed and $SR^*$ the benchmark Sharpe ratio **per period**
(`sharpe_ratio(r, p) / sqrt(p)` and `benchmark_sharpe / sqrt(p)`), $n$ the number of
returns and $\hat\gamma_3$, $\hat\gamma_4$ their skewness and kurtosis (3 for normal
returns; the moment estimators $m_3 / m_2^{3/2}$ and $m_4 / m_2^2$), Bailey and López de
Prado (2012) give

$$
\mathrm{PSR}(SR^*) = \Phi\!\left(\frac{(\widehat{SR} - SR^*)\sqrt{n - 1}}
{\sqrt{1 - \hat\gamma_3 \widehat{SR} + \frac{\hat\gamma_4 - 1}{4}\widehat{SR}^2}}\right),
\qquad
\mathrm{MinTRL} = 1 + \left(1 - \hat\gamma_3 \widehat{SR} + \frac{\hat\gamma_4 - 1}{4}
\widehat{SR}^2\right)\left(\frac{\Phi^{-1}(\text{confidence})}{\widehat{SR} - SR^*}\right)^2,
$$

where $\Phi$ is the standard normal CDF. MinTRL is infinite if $\widehat{SR} \le SR^*$.
The best of $N$ independent trials whose Sharpe ratios have no skill and a standard deviation
$\sigma$ across trials is expected to reach (Bailey and López de Prado, 2014)

$$
SR_0^* \approx \sigma\left((1 - \gamma)\,\Phi^{-1}\!\left(1 - \tfrac{1}{N}\right)
+ \gamma\,\Phi^{-1}\!\left(1 - \tfrac{1}{N e}\right)\right),
$$

with $\gamma \approx 0.5772$ the Euler-Mascheroni constant (0 for a single trial). The
deflated Sharpe ratio is $\mathrm{PSR}(SR_0^*)$. Pass `trial_sharpes`, the annualised Sharpe
ratios of **every** configuration tried, including the selected one and the ones you
discarded, and their number and standard deviation (ddof=1) are used; or pass
`n_trials=` and `trial_sharpe_std=` directly, for instance an effective number of
independent trials when many configurations are near copies of each other. The normal CDF
and its inverse are implemented in the module (Acklam's approximation refined by Newton
steps), with no SciPy dependency.

**The probability of backtest overfitting** (Bailey, Borwein, López de Prado and Zhu, 2015)
needs the return series of every trial over the same periods, as columns of a
`(n_periods, n_strategies)` array. Combinatorially symmetric cross-validation (CSCV) splits
the periods into $S$ = `n_partitions` equal consecutive blocks and, for each of the
$\binom{S}{S/2}$ ways to pick half of them as the in-sample set (12,870 for the default 16),
selects the strategy with the best in-sample `metric` (by default the per-period Sharpe
ratio) and finds its relative rank $\bar\omega$ among the out-of-sample results (rank over
$N + 1$, so the worst is near 0). The PBO is the fraction of combinations whose logit
$\lambda = \ln(\bar\omega / (1 - \bar\omega))$ is at most 0, i.e. in which the in-sample
winner is at or below the out-of-sample median. In-sample and out-of-sample sets always have
the same length: when the number of periods is not a multiple of $S$, the oldest
`n_periods % n_partitions` periods are left out. Strategies tied for the in-sample best each
count as selected with equal probability, so the result does not depend on the order of the
columns. `n_partitions` must be even and at most 20 (184,756 combinations).

The `OverfittingReport` it returns has the `pbo`, the per-combination `logits` and the
in-sample best's `is_performance` and `oos_performance`, the `degradation_slope` and
`degradation_intercept` of the least-squares line of out-of-sample on in-sample performance
(a slope near or below zero means a better in-sample result does not predict a better
out-of-sample one), and the `probability_of_loss`, the fraction of combinations in which the
in-sample best loses out of sample.

**When to use which.**

- One strategy fixed in advance: `probabilistic_sharpe_ratio`, and
  `minimum_track_record_length` to ask how long a record must be before you trust it.
- The best of many trials: `deflated_sharpe_ratio`. It needs only the trials' Sharpe ratios,
  so it also works when the other runs' returns were not kept. Count everything you tried.
- The return series of all trials: `probability_of_backtest_overfitting` judges the
  selection procedure itself, without distributional assumptions. A low PBO together with a
  high DSR is the evidence you want; a PBO near or above 0.5 says that picking the backtest
  winner is no better than picking at random. With RL agents, the columns are the test-period
  returns of each seed or hyperparameter setting on the same window, for instance
  `returns_from_equity` of each run's equity. Each block must hold enough periods for the
  metric to mean something; use fewer partitions for a short record.

Thirty configurations without skill, three years of daily returns each:

```python
from sekai.finance import metrics

rng = np.random.default_rng(0)
trials = rng.normal(0.0, 0.01, size=(756, 30))  # 30 configurations, 3 years, no skill
sharpes = np.array([metrics.sharpe_ratio(trials[:, j]) for j in range(30)])
best = trials[:, sharpes.argmax()]

print(f"{sharpes.max():.3f}")  # 1.115
print(f"{metrics.probabilistic_sharpe_ratio(best):.3f}")  # 0.973
print(f"{metrics.minimum_track_record_length(best) / 252:.2f}")  # 2.20 (years)
print(f"{metrics.expected_maximum_sharpe(30, sharpes.std(ddof=1)):.3f}")  # 1.121
print(f"{metrics.deflated_sharpe_ratio(best, sharpes):.3f}")  # 0.496
print(metrics.probability_of_backtest_overfitting(trials))
```

```text
combinations                 12870
probability of overfitting  62.88%
probability of loss         60.82%
degradation slope           -0.408
degradation intercept        0.030
partitions                      16
periods per partition           47
```

On its own the winner looks skilful: a 97% PSR and a track record already longer than the
2.2 years it needs. But 1.115 is just what the best of 30 skill-less trials reaches
(1.121), so the DSR is a coin flip, and the in-sample winner falls below the out-of-sample
median in 63% of the splits. When the configurations differ in true skill, both measures say
so:

```python
skilled = trials + np.linspace(0.0, 0.0008, 30)  # true Sharpe ratios from 0 to 1.27
sharpes = np.array([metrics.sharpe_ratio(skilled[:, j]) for j in range(30)])
best = skilled[:, sharpes.argmax()]
print(f"{metrics.deflated_sharpe_ratio(best, sharpes):.3f}")  # 0.951
print(f"{metrics.probability_of_backtest_overfitting(skilled).pbo:.3f}")  # 0.155
```

The PBO of a single panel is itself noisy (the splits share their data), so read it as
evidence rather than a precise probability. `metric` can be any callable taking the chosen
blocks' returns in time order, shape `(t, n_strategies)`, and returning one finite score per
strategy, such as `lambda r: r.mean(axis=0)`; the default Sharpe ratio is computed from
per-block statistics and is much faster.

## See also

- [Market data and backtesting](data.md#walk-forward-evaluation): summaries per walk-forward fold.
- [Statistics guide](../guide/statistics.md): episode returns and lengths for any environment.
- [`sekai.finance.metrics` API](../api/finance.md#metrics).
