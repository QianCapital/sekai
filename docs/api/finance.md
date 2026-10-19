# Finance

The `sekai.finance` toolkit. Classes are exported from `sekai.finance`; functions from
their modules (`sekai.finance.features`, `sekai.finance.metrics`, ...). See the
[Finance toolkit](../finance/index.md) pages for explanations and examples.

## Market data

`sekai.finance.data`. See [Market data and backtesting](../finance/data.md).

::: sekai.finance.data.MarketData

## Sampling

`sekai.finance.sampling`. See [Episode samplers](../finance/data.md#episode-samplers).

::: sekai.finance.sampling.EpisodeSampler

::: sekai.finance.sampling.RandomStart

::: sekai.finance.sampling.SequentialStart

::: sekai.finance.sampling.FixedStart

::: sekai.finance.sampling.as_sampler

::: sekai.finance.sampling.sample_start

::: sekai.finance.sampling.first_start

::: sekai.finance.sampling.resolve_start

## Simulators

`sekai.finance.simulators`. See [Market simulators](../finance/simulators.md).

::: sekai.finance.simulators.MarketModel

::: sekai.finance.simulators.Simulation

::: sekai.finance.simulators.GBM

::: sekai.finance.simulators.Heston

::: sekai.finance.simulators.MertonJumpDiffusion

::: sekai.finance.simulators.RegimeSwitching

::: sekai.finance.simulators.GARCH

::: sekai.finance.simulators.MeanReverting

::: sekai.finance.simulators.CointegratedPair

::: sekai.finance.simulators.BlockBootstrap

::: sekai.finance.simulators.ListingProcess

::: sekai.finance.simulators.Randomized

## Features

`sekai.finance.features`. See [Point-in-time features](../finance/data.md#point-in-time-features).

::: sekai.finance.features
    options:
      show_root_heading: false
      members:
        - log_returns
        - simple_returns
        - momentum
        - lag
        - forward_fill
        - rolling_mean
        - rolling_std
        - rolling_zscore
        - ewm_mean
        - ewm_std
        - realized_volatility
        - cross_sectional_zscore
        - cross_sectional_rank
        - asof_join
        - check_causal

## Costs

`sekai.finance.costs`. See [Cost models](../finance/execution.md#cost-models).

::: sekai.finance.costs.CostModel

::: sekai.finance.costs.NoCost

::: sekai.finance.costs.ProportionalCost

::: sekai.finance.costs.SpreadCost

::: sekai.finance.costs.FixedCost

::: sekai.finance.costs.SquareRootImpact

::: sekai.finance.costs.CompositeCost

::: sekai.finance.costs.as_cost_model

## Account

`sekai.finance.account`. See [Account rules](../finance/execution.md#account-rules).

::: sekai.finance.account.AccountConfig

::: sekai.finance.account.project_weights

::: sekai.finance.account.Settlement

::: sekai.finance.account.settle

::: sekai.finance.account.turnover

## Rewards

`sekai.finance.rewards`. See [Reward functions](../finance/execution.md#reward-functions).

::: sekai.finance.rewards.StepOutcome

::: sekai.finance.rewards.RewardFunction

::: sekai.finance.rewards.LogReturn

::: sekai.finance.rewards.SimpleReturn

::: sekai.finance.rewards.ProfitAndLoss

::: sekai.finance.rewards.MeanVariance

::: sekai.finance.rewards.TurnoverPenalty

::: sekai.finance.rewards.DrawdownPenalty

::: sekai.finance.rewards.DifferentialSharpe

## Metrics

`sekai.finance.metrics`. See [Performance metrics](../finance/metrics.md).

::: sekai.finance.metrics
    options:
      show_root_heading: false
      members:
        - total_return
        - annualized_return
        - annualized_volatility
        - sharpe_ratio
        - sortino_ratio
        - drawdown_series
        - max_drawdown
        - calmar_ratio
        - value_at_risk
        - conditional_value_at_risk
        - hit_rate
        - equity_curve
        - returns_from_equity
        - PerformanceSummary
        - performance_summary
        - PnLSummary
        - pnl_summary
        - RecordPerformance

## Universe

`sekai.finance.universe`. See [Slots and dynamic universes](../environments/finance.md#slots-and-dynamic-universes).

::: sekai.finance.universe.SlotMap

## Order book

`sekai.finance.lob`. See [Order books](../finance/order-book.md).

::: sekai.finance.lob.OrderBook

::: sekai.finance.lob.Side

::: sekai.finance.lob.Order

::: sekai.finance.lob.Fill

::: sekai.finance.lob.BACKGROUND

::: sekai.finance.lob.ZeroIntelligenceFlow
