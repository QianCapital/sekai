# Environments

The built-in environment classes, exported from `sekai.envs` and registered under the IDs in
the [catalog](../environments/index.md). Create them with `sekai.make(id)`, or instantiate
the classes directly (no time limit is applied then).

## Classic control

See [Classic control](../environments/classic-control.md).

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

## Finance

See [Finance](../environments/finance.md).

::: sekai.envs.finance.TradingEnv

::: sekai.envs.finance.PortfolioEnv

::: sekai.envs.finance.MarketMakingEnv

::: sekai.envs.finance.OptimalExecutionEnv

::: sekai.envs.finance.MarketEnv

::: sekai.envs.finance.default_market

::: sekai.envs.finance.LimitOrderBookEnv

::: sekai.envs.finance.LimitOrderBookMarket

### Finance renderers

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

::: sekai.envs.finance.MarketMakingVecEnv

::: sekai.envs.finance.OptimalExecutionVecEnv

### Price simulation

::: sekai.envs.finance.simulate_gbm

::: sekai.envs.finance.random_correlation

::: sekai.envs.finance.sample_window
