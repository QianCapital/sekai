# Environments

The built-in environment classes, exported from `sekai.envs` and registered under the IDs in
the [catalog](../environments/index.md). Create them with `sekai.make(id)`, or instantiate
the classes directly (no time limit is applied then).

## Classic Control

See [Classic Control](../environments/classic-control.md).

::: sekai.envs.CartPoleEnv

::: sekai.envs.MountainCarEnv

::: sekai.envs.ContinuousMountainCarEnv

::: sekai.envs.PendulumEnv

::: sekai.envs.AcrobotEnv

### Renderers

Exported from `sekai.envs.classic_control`. Each `render(env)` returns an `(H, W, 3)` `uint8`
frame.

::: sekai.envs.classic_control.CartPoleRenderer

::: sekai.envs.classic_control.MountainCarRenderer

::: sekai.envs.classic_control.PendulumRenderer

::: sekai.envs.classic_control.AcrobotRenderer

## Financial Markets

See [Financial Markets](../environments/finance.md).

::: sekai.envs.finance.TradingEnv

::: sekai.envs.finance.PortfolioEnv

::: sekai.envs.finance.MarketMakingEnv

::: sekai.envs.finance.OptimalExecutionEnv

::: sekai.envs.finance.MarketEnv

::: sekai.envs.finance.default_market

::: sekai.envs.finance.BatchMarketEnv

::: sekai.envs.finance.make_market

::: sekai.envs.finance.LiveMarketEnv

::: sekai.envs.finance.batch_market.cost_context

::: sekai.envs.finance.LimitOrderBookEnv

::: sekai.envs.finance.LimitOrderBookMarket

### Financial Markets renderers

Exported from `sekai.envs.finance`. Each is a `sekai.rendering.Dashboard`; `render(env)`
returns an `(H, W, 3)` `uint8` frame of the episode so far.

::: sekai.envs.finance.TradingRenderer

::: sekai.envs.finance.PortfolioRenderer

::: sekai.envs.finance.MarketRenderer

::: sekai.envs.finance.MarketMakingRenderer

::: sekai.envs.finance.OptimalExecutionRenderer

::: sekai.envs.finance.LimitOrderBookRenderer

::: sekai.envs.finance.LimitOrderBookMarketRenderer

### Native vector environments

Registered as the `vector_entry_point` of their environments, so `sekai.make_vec(id, n)`
returns them; see [Native vector environments](../environments/finance.md#native-vector-environments).

::: sekai.envs.finance.TradingVecEnv

::: sekai.envs.finance.PortfolioVecEnv

::: sekai.envs.finance.MarketVecEnv

::: sekai.envs.finance.MarketMakingVecEnv

::: sekai.envs.finance.OptimalExecutionVecEnv

### Price simulation

::: sekai.envs.finance.simulate_gbm

::: sekai.envs.finance.random_correlation

::: sekai.envs.finance.sample_window

## Strategy Games

See [Strategy Games](../environments/games.md) and [Self-play and search](../guide/games.md).
Everything below is in `sekai.envs.games`, which imports each game's subpackage the first
time it is used.

::: sekai.envs.games.chess.ChessEnv

::: sekai.envs.games.go.GoEnv

::: sekai.envs.games.xiangqi.XiangqiEnv

::: sekai.envs.games.poker.HoldemEnv

::: sekai.envs.games.blackjack.BlackjackEnv

### Playing

::: sekai.envs.games.OpponentWrapper

::: sekai.envs.games.MultiAgentGame

::: sekai.envs.games.RandomPolicy

::: sekai.envs.games.Policy

::: sekai.envs.games.GameError

::: sekai.envs.games.IllegalMove

::: sekai.envs.games.GameOver

### Game renderers

Each `render(env)` returns an `(H, W, 3)` `uint8` frame of the current position; the text
renderers return a string.

::: sekai.envs.games.chess.ChessRenderer

::: sekai.envs.games.go.GoRenderer

::: sekai.envs.games.xiangqi.XiangqiRenderer

::: sekai.envs.games.poker.HoldemRenderer

::: sekai.envs.games.blackjack.BlackjackRenderer

::: sekai.envs.games.GameTheme

### Chess toolkit

::: sekai.envs.games.chess.Board

::: sekai.envs.games.chess.Move

::: sekai.envs.games.chess.ObservationEncoder

::: sekai.envs.games.chess.to_pgn

::: sekai.envs.games.chess.env_pgn

### Go toolkit

::: sekai.envs.games.go.Board

::: sekai.envs.games.go.final_score

::: sekai.envs.games.go.area_score

::: sekai.envs.games.go.to_sgf

### Xiangqi toolkit

::: sekai.envs.games.xiangqi.Position

::: sekai.envs.games.xiangqi.to_wxf

::: sekai.envs.games.xiangqi.to_chinese

### Hold'em toolkit

::: sekai.envs.games.poker.HoldemHand

::: sekai.envs.games.poker.SeatView

::: sekai.envs.games.poker.evaluate

::: sekai.envs.games.poker.format_hand_history

### Blackjack toolkit

::: sekai.envs.games.blackjack.Rules

::: sekai.envs.games.blackjack.BlackjackTable
