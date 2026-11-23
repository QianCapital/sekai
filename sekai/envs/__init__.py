"""Built-in environments, registered with the global registry on import.

    sekai.make("CartPole-v1")
    sekai.make("Portfolio-v0", n_assets=10)
    sekai.make("Portfolio-v0", prices=closes)  # a (T, n) array or a DataFrame of closes
    sekai.make("Market-v0", data=market_data)  # a MarketData: so too in Trading/Portfolio
    sekai.make("Chess-v0")  # also Go9x9-v0, Xiangqi-v0, HeadsUpHoldem-v0, Blackjack-v0

Classic control envs match gymnasium's dynamics exactly; the finance envs are
sekai's own. The strategy games (`sekai.envs.games`) are imported only when one is
made or accessed. See `sekai list` or sekai.pprint_registry() for every ID.
"""

import importlib
from typing import Any

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
from sekai.registry.spec import WrapperSpec

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
# v1: orders fill at the next close, not at the close the agent just observed.
register(
    id="Trading-v1",
    entry_point="sekai.envs.finance.trading:TradingEnv",
    vector_entry_point="sekai.envs.finance.vector:TradingVecEnv",
    kwargs={"execution": "next_close"},
)
register(
    id="Portfolio-v1",
    entry_point="sekai.envs.finance.portfolio:PortfolioEnv",
    vector_entry_point="sekai.envs.finance.vector:PortfolioVecEnv",
    kwargs={"execution": "next_close"},
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
register(
    id="Market-v1",
    entry_point="sekai.envs.finance.batch_market:make_market",
    vector_entry_point="sekai.envs.finance.batch_market:MarketVecEnv",
)
register(id="LimitOrderBook-v0", entry_point="sekai.envs.finance.lob_env:LimitOrderBookEnv")

# ---------------------------------------------------------------------------
# Strategy games — two-player games are played from both sides by the caller
# ---------------------------------------------------------------------------

register(
    id="Chess-v0",
    entry_point="sekai.envs.games.chess.env:ChessEnv",
    max_episode_steps=512,
)
register(
    id="ChessVsRandom-v0",
    entry_point="sekai.envs.games.chess.env:ChessEnv",
    max_episode_steps=512,
    additional_wrappers=(
        WrapperSpec(
            name="OpponentWrapper",
            entry_point="sekai.envs.games.wrappers:OpponentWrapper",
            kwargs={"agent_color": "random"},
        ),
    ),
)
for _size in (9, 13, 19):
    # Go ends itself (two passes, or its own move limit), so no TimeLimit is added.
    register(
        id=f"Go{_size}x{_size}-v0",
        entry_point="sekai.envs.games.go.env:GoEnv",
        kwargs={"size": _size},
    )
del _size
register(
    id="Xiangqi-v0",
    entry_point="sekai.envs.games.xiangqi.env:XiangqiEnv",
    max_episode_steps=400,
)
register(id="HeadsUpHoldem-v0", entry_point="sekai.envs.games.poker.env:HoldemEnv")
register(id="Blackjack-v0", entry_point="sekai.envs.games.blackjack.env:BlackjackEnv")

_GAME_EXPORTS = (
    "ChessEnv",
    "GoEnv",
    "XiangqiEnv",
    "HoldemEnv",
    "BlackjackEnv",
)


def __getattr__(name: str) -> Any:
    # The strategy games load on first use, so `import sekai` does not build their tables.
    if name == "games":
        return importlib.import_module("sekai.envs.games")
    if name in _GAME_EXPORTS:
        return getattr(importlib.import_module("sekai.envs.games"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    "BlackjackEnv",
    "ChessEnv",
    "GoEnv",
    "HoldemEnv",
    "XiangqiEnv",
]
