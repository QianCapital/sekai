# Benchmarks

A learned trading policy means little until it beats simple rules on the same episodes,
and a backtest means little until the number of things tried is accounted for.
`sekai.finance.benchmark` makes both routine:

- a `Task` is an environment and a fixed set of evaluation episodes: episode `k` resets
  with seed `seed + k` (10,000 and up by default), so every policy faces the same
  scenarios, none of which a training run that keeps its seeds below them has seen;
- `evaluate(policy, task)` runs a policy on them and returns an `Evaluation` (every
  episode's equity curve, pooled per-step returns, Sharpe ratio, probabilistic Sharpe
  ratio, drawdowns, turnover);
- `compare(policies, task)` evaluates several side by side and reports each one's
  [deflated Sharpe ratio](metrics.md#overfitting-and-multiple-testing), counting every
  policy compared as a trial: include the variants you tried, not only the winner;
- the baselines `Cash`, `EqualWeight`, `InverseVolatility` and `Momentum` (long-only, or
  long/short) express a target book that `target_action` turns into an action of
  `Market-v0`, `Market-v1` (whatever its action map) and `Portfolio-v0` / `-v1`.

A policy is any callable `policy(observation, env) -> action`, where `env` is the
unwrapped environment (for `env.action_mask()` and the like). A trained model is one
lambda away:

```python
from sekai.finance.benchmark import Momentum, Task, compare, standard_baselines

task = Task("market", "Market-v1", {"episode_length": 252}, episodes=50)
policies = {**standard_baselines(), "long/short momentum": Momentum(20, long_short=True)}
print(compare(policies, task))
```

```text
market: 50 episodes of Market-v1
policy                   Sharpe         PSR         DSR    mean ret     std ret  avg max DD    turnover
cash                       0.00       0.500       0.036      +0.00%       0.00%       0.00%       0.000
equal weight               0.03       0.575       0.054      +1.33%      18.94%      15.89%       0.015
inverse vol               -0.04       0.398       0.020      +0.25%      18.69%      16.32%       0.056
momentum                  -0.39       0.003       0.000      -6.38%      19.76%      21.48%       0.345
long/short momentum       -0.39       0.003       0.000      -3.45%      10.84%      11.62%       0.173
```

None of these rules has an edge here: within a regime, which asset does better is
unpredictable, and none of them times the hidden regimes. Their deflated Sharpe ratios are
all near 0, and the best probabilistic Sharpe ratio, equal weight's 0.58, is what luck
alone gives the best of five candidates.

For your own data, evaluate on the test period of a [split](data.md#chronological-splits):
`Task("test", "Market-v1", {"data": test, "sampler": "sequential", ...}, episodes=n)`
replays its windows one after another (`n` windows cover it once).

## Reference results

`Market-v1`'s default market switches between a calm and a stressed regime, and its
assets list and delist. Over 200 held-out one-year episodes, orders filling at the next
close and 10 bp of turnover in costs:

```text
market: 200 episodes of Market-v1
policy                Sharpe         PSR         DSR    mean ret     std ret  avg max DD    turnover
cash                    0.00       0.500       0.000      +0.00%       0.00%       0.00%       0.000
equal weight            0.15       0.981       0.001      +3.22%      20.10%      15.44%       0.015
inverse vol             0.08       0.875       0.000      +2.09%      19.88%      15.83%       0.055
momentum               -0.36       0.000       0.000      -6.33%      18.94%      21.51%       0.349
PPO (200k steps)       -0.57       0.000       0.000      -6.93%      14.68%      16.05%       0.372
```

PPO loses to holding everything equally. It learned to take less risk (a lower spread of
episode returns) but turns over 37% of the book a day, which at 10 bp is about 9% a year in
costs, more than this market pays. That is the typical failure of policy-gradient agents
in markets with costs, and why a [cost-aware parameterisation](batched.md#what-the-agent-controls)
(`TradeDelta`, where holding still is the zero action) or a
[pathwise trainer](batched.md#backends-and-gradients) are worth trying.

PPO is Stable-Baselines3's, with `sekai.nn.sb3.MarketFeaturesExtractor` encoding the
assets as a set, trained for 200,000 steps on 8 environments (`n_steps=256`,
`batch_size=256`, otherwise SB3's defaults), and evaluated deterministically:
[`examples/integrations/benchmark_reference.py`](https://github.com/QianCapital/sekai/blob/master/examples/integrations/benchmark_reference.py)
reproduces the table. It is a reference point, not a tuned result: no hyperparameter was
searched, and a longer run, a recurrent policy or the regime as an input
(`exogenous=("regime",)`) are the obvious next steps.

## See also

- [Performance metrics](metrics.md), including the overfitting statistics.
- [Training agents](../guide/training.md) and
  [batched environments](batched.md#backends-and-gradients) for training.
- API: [`sekai.finance.benchmark`](../api/finance.md#benchmarks).
