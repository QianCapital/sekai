# Finance

The `sekai.finance` toolkit. Classes are exported from `sekai.finance`; functions from
their modules (`sekai.finance.features`, `sekai.finance.metrics`, ...). See the
[Finance toolkit](../finance/index.md) pages for explanations and examples.

## Market data

`sekai.finance.data`. See [Market data and backtesting](../finance/data.md).

::: sekai.finance.data.MarketData

::: sekai.finance.data.Time

## Panels and scenarios

`sekai.finance.panel`. See [Where the data comes from](../finance/batched.md#where-the-data-comes-from).

::: sekai.finance.panel.MarketPanel

::: sekai.finance.panel.ScenarioGenerator

::: sekai.finance.panel.ModelScenarios

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

### Holding costs

See [Holding costs](../finance/execution.md#holding-costs).

::: sekai.finance.costs.HoldingCost

::: sekai.finance.costs.BorrowFee

::: sekai.finance.costs.GrossFinancing

::: sekai.finance.costs.CompositeHoldingCost

## Account

`sekai.finance.account`. See [Account rules](../finance/execution.md#account-rules).

::: sekai.finance.account.AccountConfig

::: sekai.finance.account.project_weights

::: sekai.finance.account.Settlement

::: sekai.finance.account.settle

::: sekai.finance.account.turnover

## Constraints

`sekai.finance.constraints`. See [Constraints](../finance/execution.md#constraints).

::: sekai.finance.constraints.Constraints

::: sekai.finance.constraints.project

::: sekai.finance.constraints.neutralize

::: sekai.finance.constraints.constraint_columns

## Risk models

`sekai.finance.risk`. See [Risk models](../finance/execution.md#risk-models).

::: sekai.finance.risk.FactorRiskModel

::: sekai.finance.risk.portfolio_variance

::: sekai.finance.risk.factor_exposures

## Action maps

`sekai.finance.actions`. See [What the agent controls](../finance/batched.md#what-the-agent-controls).

::: sekai.finance.actions.ActionMap

::: sekai.finance.actions.ActionContext

::: sekai.finance.actions.LongOnlyScores

::: sekai.finance.actions.TargetWeights

::: sekai.finance.actions.ScoresLeverage

::: sekai.finance.actions.TanhWeights

::: sekai.finance.actions.TradeDelta

## Observations

`sekai.finance.observations`. See [What the agent observes](../finance/batched.md#what-the-agent-observes).

::: sekai.finance.observations.ObservationBuilder

::: sekai.finance.observations.MarketObservation

::: sekai.finance.observations.MarketView

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

::: sekai.finance.rewards.PortfolioVariance

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
        - probabilistic_sharpe_ratio
        - minimum_track_record_length
        - expected_maximum_sharpe
        - deflated_sharpe_ratio
        - OverfittingReport
        - probability_of_backtest_overfitting
        - RecordPerformance

## Paper and live trading

`sekai.finance.live`. See [Paper and live trading](../finance/live.md).

::: sekai.finance.live.Bar

::: sekai.finance.live.DataFeed

::: sekai.finance.live.ReplayFeed

::: sekai.finance.live.Broker

::: sekai.finance.live.PaperBroker

::: sekai.finance.live.AccountState

::: sekai.finance.live.ExecutionReport

::: sekai.finance.live.orders_for

::: sekai.finance.live.RiskLimits

::: sekai.finance.live.RiskEvent

::: sekai.finance.live.GuardedBroker

## Benchmarks

`sekai.finance.benchmark`. See [Benchmarks](../finance/benchmarks.md).

::: sekai.finance.benchmark.Task

::: sekai.finance.benchmark.evaluate

::: sekai.finance.benchmark.Evaluation

::: sekai.finance.benchmark.compare

::: sekai.finance.benchmark.Comparison

::: sekai.finance.benchmark.target_action

::: sekai.finance.benchmark.Cash

::: sekai.finance.benchmark.EqualWeight

::: sekai.finance.benchmark.InverseVolatility

::: sekai.finance.benchmark.Momentum

::: sekai.finance.benchmark.standard_baselines

::: sekai.finance.benchmark.standard_tasks

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

### Historical replay

`sekai.finance.lob_replay` (re-exported from `sekai.finance.lob` and `sekai.finance`). See
[Historical replay](../finance/order-book.md#historical-replay).

::: sekai.finance.lob_replay.MessageReplay

::: sekai.finance.lob_replay.OrderBookMessages

::: sekai.finance.lob_replay.MessageType

::: sekai.finance.lob_replay.BookSnapshot

::: sekai.finance.lob_replay.BookSnapshots

::: sekai.finance.lob_replay.read_lobster

## Rendering

`sekai.finance.rendering`: charts for [dashboards](../guide/rendering.md#dashboards-for-your-own-environments)
of market environments. The classes are also exported from `sekai.finance`.

::: sekai.finance.rendering.PriceChart

::: sekai.finance.rendering.OrderBookChart

::: sekai.finance.rendering.UniverseChart
