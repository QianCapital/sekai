# Paper and live trading

A policy trained on `Market-v1` trades a paper or live account through
`sekai.envs.finance.LiveMarketEnv`, which builds the same observations and accepts the same
actions from bars as they arrive. sekai defines the interfaces and never connects to a market
itself. You supply two objects:

- a **`DataFeed`**: `assets` and `next_bar()`, which returns the next `Bar` (closing prices,
  per-asset fields such as volume and spread, market-wide series) or `None` when the feed
  ends. Wrap your market data service, message queue or daily files in one.
  `ReplayFeed(data, start, stop)` replays a `MarketData` you own, for paper trading and for
  testing the loop;
- a **`Broker`**: `reset(assets)` (a new account, or attach to an existing one),
  `execute(weights, bar, context)` (trade to target weights at the bar's close) and
  `mark(bar)` (the account once the next bar is in). `PaperBroker` simulates the account
  with sekai's own accounting, cost and holding-cost models.

Each `step(action)` turns the action into target weights with the environment's action map,
sends them to the broker (with next-close execution, the default, the targets of the
previous decision; the new ones fill at the next close and are observed as pending), reads
the next bar, and lets the broker mark the account. The reward is the reward function of
the account's growth, as in `Market-v1`.

**Parity.** Replaying a history through `LiveMarketEnv` with a `PaperBroker` reproduces
`Market-v1` on the same history step for step: rewards, equity, costs, financing and
observations agree to rounding (about 1e-16), with spread and impact costs, per-name borrow
fees and listings and delistings, in both execution modes. The test suite checks it, so a
policy meets the same environment in the backtest and in paper trading.

```python
import numpy as np
from sekai.envs.finance import LiveMarketEnv
from sekai.finance import GBM, MarketData, PaperBroker, ReplayFeed, SpreadCost, TargetWeights

rng = np.random.default_rng(0)
data = MarketData(
    GBM(mu=0.05, sigma=0.2, n_assets=5).simulate(rng, 299).prices,
    timestamps=np.busday_offset("2024-01-01", np.arange(300), roll="forward"),
    fields={"spread": np.full((300, 5), 4e-4)},
)
broker = PaperBroker(transaction_cost=SpreadCost(), capital=1e6)
env = LiveMarketEnv(ReplayFeed(data), broker, lookback=20, action=TargetWeights())

result = env.reset()
while not result.done:
    result = env.step(np.full(5, 0.2, np.float32))  # 20% in each asset
print(result.info["timestamp"], f"{result.info['equity']:.4f}", result.info.get("feed_ended"))
# 2025-02-21 0.9754 True
```

## A broker adapter

An adapter to a real broker turns target weights into orders and reads the account back.
`orders_for(weights, account_value, prices, positions, lot_size=...)` computes the share
quantities, rounded to whole lots towards zero:

```python
from sekai.finance import AccountState, ExecutionReport, orders_for


class MyBroker:
    """A sketch: replace the `client` calls with your broker's API."""

    def __init__(self, client):
        self.client = client

    def reset(self, assets):
        self.assets, self.start = assets, self.client.account_value()
        return self._state()

    def execute(self, weights, bar, context):
        positions = np.array([self.client.position(a) for a in self.assets])
        orders = orders_for(weights, self.client.account_value(), bar.prices, positions)
        for asset, quantity in zip(self.assets, orders):
            if quantity:
                self.client.market_on_close(asset, quantity)
        return ExecutionReport(weights=weights, turnover=float(np.abs(weights - self._state().weights).sum()), cost=0.0)

    def mark(self, bar):
        self.client.wait_for_fills()
        return self._state()

    def _state(self):
        value = self.client.account_value()
        prices = np.array([self.client.price(a) for a in self.assets])
        held = np.array([self.client.position(a) for a in self.assets]) * prices
        return AccountState(equity=value / self.start, weights=held / value, value=value)
```

## Risk limits

`GuardedBroker(broker, RiskLimits(...))` enforces hard limits between any policy and any
broker, whatever the policy asks for, and records every intervention in `events`:

| Limit | Effect |
|---|---|
| `max_position` | each `|w_i|` is clipped |
| `max_gross` | the book is scaled down to this gross exposure |
| `max_net` | the larger side is scaled down to this net exposure |
| `max_turnover` | a rebalance goes only this far towards the target |
| `max_drawdown` | kill switch: at this drawdown from the peak everything is sold, and the account stays flat until `reset` |
| `allow_short=False` | no negative weights |

Learned policies make mistakes that a backtest does not show. Limits that live outside the
policy, and that hold however it behaves, are what make trading one acceptable:

```python
from sekai.finance import GuardedBroker, RiskLimits

limits = RiskLimits(max_position=0.25, max_gross=1.0, max_drawdown=0.10)
guarded = GuardedBroker(PaperBroker(transaction_cost=SpreadCost()), limits)
env = LiveMarketEnv(ReplayFeed(data), guarded, lookback=20, action=TargetWeights())
result = env.reset()
while not result.done:
    result = env.step(np.array([1.0, 1.0, 0.0, 0.0, 0.0], np.float32))  # 50% in each of two assets
print(sorted({event.kind for event in guarded.events}))  # ['kill_switch', 'max_position']
```

The policy asked for half the book in each of two assets: the targets were cut to 25% each,
and when the account fell 10% from its peak the kill switch sold everything and kept it flat
to the end.

## See also

- [Batched and differentiable environments](batched.md): `Market-v1`, which `LiveMarketEnv`
  mirrors.
- API: [`sekai.finance.live`](../api/finance.md#paper-and-live-trading),
  [`LiveMarketEnv`](../api/envs.md#sekai.envs.finance.LiveMarketEnv).
