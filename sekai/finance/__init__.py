"""Building blocks for financial-market environments.

- `data`: `MarketData`, point-in-time panels with listings and delistings,
  chronological splits and walk-forward folds; `Time`, a timestamp to look up
  by value where an integer would be a row position.
- `sampling`: episode samplers (random, sequential and fixed windows).
- `simulators`: market models (GBM, Heston, jumps, regime switching, GARCH, mean
  reversion, cointegration, block bootstrap), dynamic universes and domain
  randomisation.
- `features`: causal feature transforms, point-in-time `asof_join`, and the
  `check_causal` look-ahead detector.
- `costs`: transaction-cost and market-impact models, and holding costs (per-name
  borrow fees, financing on gross exposure).
- `account`: weight-space accounting (constraints, leverage, financing, drift).
- `constraints`: hard portfolio constraints by projection (gross and per-name caps,
  dollar and factor neutrality, unshortable names).
- `risk`: factor risk models (ex-ante variance, factor exposures).
- `actions`: action maps, from a policy's raw action to target weights.
- `observations`: observation builders for batched market environments.
- `panel`: `MarketPanel`, batched market data on NumPy or torch, and scenario
  generators (`ScenarioGenerator`, `ModelScenarios`).
- `rewards`: reward functions (log return, P&L, mean-variance, differential
  Sharpe, drawdown and turnover penalties, ex-ante variance).
- `metrics`: performance metrics, statistics for selection bias and backtest
  overfitting, and the `RecordPerformance` wrapper.
- `benchmark`: reproducible evaluation tasks, classical baselines and comparisons.
- `live`: data feeds, brokers (a paper broker with sekai's accounting) and risk limits
  for paper and live trading with `sekai.envs.finance.LiveMarketEnv`.
- `universe`: `SlotMap`, stable slots for a changing set of instruments.
- `lob`: a limit order book matching engine, zero-intelligence order flow, and the
  replay of historical message streams (LOBSTER files) as background flow.
- `rendering`: charts for dashboards of market environments (prices with positions
  and trades, order-book depth, changing universes).

The built-in finance environments (`sekai.envs.finance`) are assembled from these
parts, and so can yours. The accounting, costs, constraints, risk model, action maps
and rewards run on any `sekai.backend` (NumPy, or torch with autograd on any device).
"""

from sekai.finance.data import MarketData, Time
from sekai.finance.sampling import (
    EpisodeSampler,
    FixedStart,
    RandomStart,
    SequentialStart,
)
from sekai.finance.simulators import (
    GARCH,
    GBM,
    BlockBootstrap,
    CointegratedPair,
    Heston,
    ListingProcess,
    MarketModel,
    MeanReverting,
    MertonJumpDiffusion,
    Randomized,
    RegimeSwitching,
    Simulation,
)
from sekai.finance.costs import (
    CompositeCost,
    CostModel,
    FixedCost,
    NoCost,
    ProportionalCost,
    SpreadCost,
    SquareRootImpact,
    HoldingCost,
    BorrowFee,
    GrossFinancing,
    CompositeHoldingCost,
)
from sekai.finance.account import AccountConfig, Settlement
from sekai.finance.constraints import Constraints, project
from sekai.finance.risk import FactorRiskModel, factor_exposures, portfolio_variance
from sekai.finance.actions import (
    ActionContext,
    ActionMap,
    LongOnlyScores,
    ScoresLeverage,
    TanhWeights,
    TargetWeights,
    TradeDelta,
)
from sekai.finance.observations import MarketObservation, MarketView, ObservationBuilder
from sekai.finance.panel import MarketPanel, ModelScenarios, ScenarioGenerator
from sekai.finance.live import (
    AccountState,
    Bar,
    Broker,
    DataFeed,
    ExecutionReport,
    GuardedBroker,
    PaperBroker,
    ReplayFeed,
    RiskEvent,
    RiskLimits,
    orders_for,
)
from sekai.finance.rewards import (
    DifferentialSharpe,
    DrawdownPenalty,
    LogReturn,
    MeanVariance,
    ProfitAndLoss,
    RewardFunction,
    SimpleReturn,
    StepOutcome,
    TurnoverPenalty,
    PortfolioVariance,
)
from sekai.finance.metrics import (
    OverfittingReport,
    PerformanceSummary,
    PnLSummary,
    RecordPerformance,
    deflated_sharpe_ratio,
    minimum_track_record_length,
    performance_summary,
    pnl_summary,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from sekai.finance.universe import SlotMap
from sekai.finance.lob import (
    BACKGROUND,
    BookSnapshot,
    BookSnapshots,
    Fill,
    MessageReplay,
    MessageType,
    Order,
    OrderBook,
    OrderBookMessages,
    Side,
    ZeroIntelligenceFlow,
    read_lobster,
)
from sekai.finance.rendering import OrderBookChart, PriceChart, UniverseChart
from sekai.finance import account, costs, data, features, lob, metrics, rewards, sampling
from sekai.finance import rendering, simulators, universe
from sekai.finance import actions, benchmark, constraints, live, observations, panel, risk

__all__ = [
    # Data
    "MarketData",
    "Time",
    "MarketPanel",
    "ScenarioGenerator",
    "ModelScenarios",
    # Paper and live trading
    "Bar",
    "DataFeed",
    "ReplayFeed",
    "Broker",
    "PaperBroker",
    "AccountState",
    "ExecutionReport",
    "RiskLimits",
    "RiskEvent",
    "GuardedBroker",
    "orders_for",
    "EpisodeSampler",
    "RandomStart",
    "SequentialStart",
    "FixedStart",
    # Simulators
    "MarketModel",
    "Simulation",
    "GBM",
    "Heston",
    "MertonJumpDiffusion",
    "RegimeSwitching",
    "GARCH",
    "MeanReverting",
    "CointegratedPair",
    "BlockBootstrap",
    "Randomized",
    "ListingProcess",
    # Costs and accounting
    "CostModel",
    "NoCost",
    "ProportionalCost",
    "SpreadCost",
    "FixedCost",
    "SquareRootImpact",
    "CompositeCost",
    "HoldingCost",
    "BorrowFee",
    "GrossFinancing",
    "CompositeHoldingCost",
    "AccountConfig",
    "Settlement",
    # Constraints, risk, actions and observations
    "Constraints",
    "project",
    "FactorRiskModel",
    "factor_exposures",
    "portfolio_variance",
    "ActionMap",
    "ActionContext",
    "LongOnlyScores",
    "TargetWeights",
    "ScoresLeverage",
    "TanhWeights",
    "TradeDelta",
    "ObservationBuilder",
    "MarketObservation",
    "MarketView",
    # Rewards
    "RewardFunction",
    "StepOutcome",
    "LogReturn",
    "SimpleReturn",
    "ProfitAndLoss",
    "MeanVariance",
    "TurnoverPenalty",
    "DrawdownPenalty",
    "DifferentialSharpe",
    "PortfolioVariance",
    # Metrics
    "PerformanceSummary",
    "PnLSummary",
    "performance_summary",
    "pnl_summary",
    "probabilistic_sharpe_ratio",
    "minimum_track_record_length",
    "deflated_sharpe_ratio",
    "OverfittingReport",
    "probability_of_backtest_overfitting",
    "RecordPerformance",
    # Universe and order book
    "SlotMap",
    "OrderBook",
    "Order",
    "Fill",
    "Side",
    "BACKGROUND",
    "ZeroIntelligenceFlow",
    "MessageReplay",
    "MessageType",
    "OrderBookMessages",
    "BookSnapshot",
    "BookSnapshots",
    "read_lobster",
    # Rendering
    "PriceChart",
    "OrderBookChart",
    "UniverseChart",
    # Submodules
    "account",
    "costs",
    "data",
    "features",
    "lob",
    "metrics",
    "rendering",
    "rewards",
    "sampling",
    "simulators",
    "universe",
    "actions",
    "benchmark",
    "constraints",
    "live",
    "observations",
    "panel",
    "risk",
]
