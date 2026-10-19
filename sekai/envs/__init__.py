"""Built-in environments, registered with the global registry on import.

    sekai.make("CartPole-v1")
    sekai.make("Portfolio-v0", n_assets=10)

Classic control envs match gymnasium's dynamics exactly; the finance envs are
sekai's own. See `sekai list` or sekai.pprint_registry() for every ID.
"""

from sekai.envs.classic_control import (
    AcrobotEnv,
    CartPoleEnv,
    ContinuousMountainCarEnv,
    MountainCarEnv,
    PendulumEnv,
)
from sekai.envs.finance import (
    LimitOrderBookEnv,
    LimitOrderBookMarket,
    MarketEnv,
    MarketMakingEnv,
    OptimalExecutionEnv,
    PortfolioEnv,
    TradingEnv,
)
from sekai.registry.registry import register

# ---------------------------------------------------------------------------
# Classic control — same IDs, time limits and reward thresholds as gymnasium
# ---------------------------------------------------------------------------

register(
    id="CartPole-v1",
    entry_point="sekai.envs.classic_control.cartpole:CartPoleEnv",
    max_episode_steps=500,
    reward_threshold=475.0,
)
register(
    id="MountainCar-v0",
    entry_point="sekai.envs.classic_control.mountain_car:MountainCarEnv",
    max_episode_steps=200,
    reward_threshold=-110.0,
)
register(
    id="MountainCarContinuous-v0",
    entry_point="sekai.envs.classic_control.continuous_mountain_car:ContinuousMountainCarEnv",
    max_episode_steps=999,
    reward_threshold=90.0,
)
register(
    id="Pendulum-v1",
    entry_point="sekai.envs.classic_control.pendulum:PendulumEnv",
    max_episode_steps=200,
)
register(
    id="Acrobot-v1",
    entry_point="sekai.envs.classic_control.acrobot:AcrobotEnv",
    max_episode_steps=500,
    reward_threshold=-100.0,
)

# ---------------------------------------------------------------------------
# Finance — horizons are part of each env, so no TimeLimit is added
# ---------------------------------------------------------------------------

register(
    id="Trading-v0",
    entry_point="sekai.envs.finance.trading:TradingEnv",
    vector_entry_point="sekai.envs.finance.vector:TradingVecEnv",
)
register(
    id="Portfolio-v0",
    entry_point="sekai.envs.finance.portfolio:PortfolioEnv",
    vector_entry_point="sekai.envs.finance.vector:PortfolioVecEnv",
)
register(
    id="MarketMaking-v0",
    entry_point="sekai.envs.finance.market_making:MarketMakingEnv",
    vector_entry_point="sekai.envs.finance.vector:MarketMakingVecEnv",
)
register(
    id="OptimalExecution-v0",
    entry_point="sekai.envs.finance.optimal_execution:OptimalExecutionEnv",
    vector_entry_point="sekai.envs.finance.vector:OptimalExecutionVecEnv",
)
register(id="Market-v0", entry_point="sekai.envs.finance.market_env:MarketEnv")
register(id="LimitOrderBook-v0", entry_point="sekai.envs.finance.lob_env:LimitOrderBookEnv")

__all__ = [
    "AcrobotEnv",
    "CartPoleEnv",
    "ContinuousMountainCarEnv",
    "MountainCarEnv",
    "PendulumEnv",
    "LimitOrderBookEnv",
    "LimitOrderBookMarket",
    "MarketEnv",
    "MarketMakingEnv",
    "OptimalExecutionEnv",
    "PortfolioEnv",
    "TradingEnv",
]
