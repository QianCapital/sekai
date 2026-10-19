"""Building blocks for financial-market environments.

- `data`: `MarketData`, point-in-time panels with listings and delistings,
  chronological splits and walk-forward folds.
- `sampling`: episode samplers (random, sequential and fixed windows).
- `simulators`: market models (GBM, Heston, jumps, regime switching, GARCH, mean
  reversion, cointegration, block bootstrap), dynamic universes and domain
  randomisation.
- `features`: causal feature transforms, point-in-time `asof_join`, and the
  `check_causal` look-ahead detector.
- `costs`: transaction-cost and market-impact models.
- `account`: weight-space accounting (constraints, leverage, financing, drift).
- `rewards`: reward functions (log return, P&L, mean-variance, differential
  Sharpe, drawdown and turnover penalties).
- `metrics`: performance metrics and the `RecordPerformance` wrapper.
- `universe`: `SlotMap`, stable slots for a changing set of instruments.
- `lob`: a limit order book matching engine and zero-intelligence order flow.

The built-in finance environments (`sekai.envs.finance`) are assembled from these
parts, and so can yours.
"""

from sekai.finance.data import MarketData
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
)
from sekai.finance.account import AccountConfig
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
)
from sekai.finance.metrics import (
    PerformanceSummary,
    PnLSummary,
    RecordPerformance,
    performance_summary,
    pnl_summary,
)
from sekai.finance.universe import SlotMap
from sekai.finance.lob import BACKGROUND, Fill, Order, OrderBook, Side, ZeroIntelligenceFlow
from sekai.finance import account, costs, data, features, lob, metrics, rewards, sampling
from sekai.finance import simulators, universe

__all__ = [
    # Data
    "MarketData",
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
    "AccountConfig",
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
    # Metrics
    "PerformanceSummary",
    "PnLSummary",
    "performance_summary",
    "pnl_summary",
    "RecordPerformance",
    # Universe and order book
    "SlotMap",
    "OrderBook",
    "Order",
    "Fill",
    "Side",
    "BACKGROUND",
    "ZeroIntelligenceFlow",
    # Submodules
    "account",
    "costs",
    "data",
    "features",
    "lob",
    "metrics",
    "rewards",
    "sampling",
    "simulators",
    "universe",
]
