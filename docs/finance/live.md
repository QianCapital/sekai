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
the account's growth, as in `Market-v1`, over the weights the broker reports it executed
(`ExecutionReport.weights`: after any risk limits, not the targets the policy asked for).

**Parity.** Replaying a history through `LiveMarketEnv` with a `PaperBroker` reproduces
`Market-v1` on the same history step for step: rewards, equity, costs, financing and
observations agree to rounding (about 1e-16), with spread and impact costs, per-name borrow
fees, a volatility field, and listings and delistings, in both execution modes; on the test
part of a split, the first decision is `Market-v1`'s too ([warm-up bars](#warm-up-bars)).
The test suite checks it, so a policy meets the same environment in the backtest and in
paper trading.

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

## Bars

Every bar is checked as it arrives, by `LiveMarketEnv` and by `PaperBroker`
(`Bar.validate`), so a bad price never reaches an account. As in `MarketData`, a price is
NaN (the asset cannot be traded now) or finite and positive; with `corporate_actions`, a
`split` is NaN or positive and a `dividend` NaN or non-negative. The fields and market series
the environment reads must be present, without infinities, and of the shapes its
observation space was built for (below), and so must the inputs the broker's cost model
needs and has no fallback for (a `spread` for `SpreadCost()`, say; an input with a fallback
may be missing, and reaches the model as NaN). Anything else raises `ValueError`, naming the
asset (or field) and the bar's timestamp, and the bar is not used; after one in `step()`,
the order is out but the account has no mark, so call `reset()` before stepping on. The same
goes for a `mark()` that raises (a broker outage): the episode ends, so no order goes out on
an account the environment can no longer see, and `reset()` attaches to it again.

### Warm-up bars

A bar with `warmup=True` is history for the lookback window, like the rows of
`MarketData.warmup`: the environment observes it but never makes a decision on it. Warm-up
bars come first: `reset()` reads bars until it holds `lookback + 1` and the newest is not a
warm-up bar, and a warm-up bar after bars to trade on raises `ValueError`. `ReplayFeed` marks
the history rows that `split` and `walk_forward` prepend to a test period, so paper trading
the test part decides on test rows only, from the row `Market-v1` starts at:

```python
train, test = data.split(0.8, embargo=5, history=20)  # 20 rows of history for the lookback
env = LiveMarketEnv(ReplayFeed(test), PaperBroker(), lookback=20)
print(env.reset().info["timestamp"] == test.timestamps[test.warmup])  # True
```

A live feed delivers its history the same way: the last `lookback` bars from your data
service, as warm-up bars, then the live ones.

### Field shapes

The observation space is built before any bar has arrived, so a field with several values
per asset (factor loadings `(N, 3)`, say) or a matrix-valued market series has to be
declared: `LiveMarketEnv(..., field_shapes={"loadings": (3,)}, exogenous_shapes={"cov": (3,
3)})` (the trailing shape per asset, and the shape of the series). By default they are the
feed's attributes of the same names, which `ReplayFeed` has; a field not declared has one
value per asset, a series one value. A bar that disagrees raises `ValueError`, rather than
giving an observation outside the space.

## Episodes

`reset()` asks the broker for the account first (`Broker.reset`), then reads bars. The bars
carry over from one episode to the next: a later reset continues from the newest bar without
waiting for new ones, so the account is never left unwatched between episodes. While
`reset()` waits for bars to trade on, an account that already holds positions (a live account
on a feed without history, say) holds them bar by bar: an order for its current weights, then
a mark, so the broker, and a `GuardedBroker`, mark and check every bar. An episode starts
with the account's book as the pending order, so its first step (with `next_close`) keeps
what is held rather than selling it. Once the feed has ended (`next_bar()` returned
`None`), `reset()` raises `RuntimeError`.

A position whose asset can't be traded at a bar (no price: a halt, on a live account) is held
as it is, and stays in the pending order; if the book then exceeds the action's gross cap,
the rest of it is scaled down to fit. A halt therefore never forces a sale and a buy-back.
(A `PaperBroker` replays history, where a missing price means the asset has delisted.)

## A broker adapter

An adapter to a real broker turns target weights into orders and reads the account back.
`orders_for(weights, account_value, prices, positions, lot_size=...)` computes the share
quantities, in whole lots rounded towards zero (a target worth exactly k lots is k lots,
whatever the float rounding of `weight * value / price`). An asset with a NaN price gets no
order; anything that would make a quantity undefined (a non-finite weight, position or
account value, a `lot_size` that is not positive, a price of 0, below 0 or infinite for an
asset that needs an order) raises `ValueError` instead, so an adapter never sends NaN orders:

```python
from sekai.finance import AccountState, ExecutionReport, orders_for


class MyBroker:
    """A sketch: replace the `client` calls with your broker's API."""

    fresh_account = False  # reset() attaches to the same live account (also the default)

    def __init__(self, client):
        self.client = client

    def reset(self, assets):
        self.assets, self.start = assets, self.client.account_value()
        return self._state()

    def execute(self, weights, bar, context):
        value, before = self.client.account_value(), self._state().weights
        positions = np.array([self.client.position(a) for a in self.assets])
        orders = orders_for(weights, value, bar.prices, positions)
        for asset, quantity in zip(self.assets, orders):
            if quantity:
                self.client.market_on_close(asset, quantity)
        prices = np.array([self.client.price(a) for a in self.assets])
        held = (positions + orders) * prices / value  # what was sent, in whole shares
        return ExecutionReport(weights=held, turnover=float(np.abs(held - before).sum()), cost=0.0)

    def mark(self, bar):
        self.client.wait_for_fills()
        return self._state()

    def _state(self):
        value = self.client.account_value()
        prices = np.array([self.client.price(a) for a in self.assets])
        held = np.array([self.client.position(a) for a in self.assets]) * prices
        return AccountState(equity=value / self.start, weights=held / value, value=value)
```

`ExecutionReport.weights` should be what the account holds after the orders: the reward
sees them. `fresh_account` tells a `GuardedBroker` whether `reset()` opens a new account
(below).

## Risk limits

`GuardedBroker(broker, RiskLimits(...))` enforces hard limits between any policy and any
broker, whatever the policy asks for, and records every intervention in `events`:

| Limit | Effect |
|---|---|
| `max_position` | each `|w_i|` is clipped |
| `max_gross` | the book is scaled down to this gross exposure |
| `max_net` | the larger side is scaled down to this net exposure |
| `max_turnover` | a rebalance goes only this far towards the target (beyond the trades that bring a drifted book back within the caps above) |
| `max_drawdown` | kill switch: at this drawdown of the account from its peak everything is sold, and the account stays flat until `rearm()` |
| `allow_short=False` | no negative weights |

The book that reaches the broker always satisfies `max_position`, `max_gross`, `max_net` and
`allow_short` (exactly, rounding included), even when price moves have pushed the current book over a cap: the trades that
only bring it back within the caps are made whatever the turnover budget, and the budget
limits the rest of the move. A target weight that is not finite (a NaN from a policy bug)
means "do not trade that asset": it keeps its current weight (an event `"non_finite_target"`),
and the book still goes through every cap. So does an asset without a price at the bar,
which cannot be traded: the caps bind on the other assets, so they hold for the book as
executed (only a position that cannot be traded can break one, on its own).

**The kill switch** trips when the account's drawdown reaches `max_drawdown`, and it fails
closed: it also trips when the broker reports an account that is not finite (equity, value or
weights) or has no value left, with or without `max_drawdown`, and on an order after a
`mark()` that raised, until a mark or `reset()` reports the account again. The next order sells everything, and the account
stays flat. It latches: a `reset()` does not clear it, only `rearm()` does, an operator's
decision (recorded as a `"rearm"` event), after which the drawdown is measured from the
account's value at that moment.

**The drawdown is the account's**: its value (`AccountState.value`) against the highest value
reported (`guarded.peak`, `guarded.drawdown`). A broker whose `reset()` opens a fresh account
says so with `fresh_account = True`, as `PaperBroker` does, and each account's peak then
starts at its opening. Any other broker is taken to be one persistent account, a live one,
and the peak carries over from one episode to the next, so a run of episodes cannot escape the
limit by starting afresh. (Deposits and withdrawals move the value too: a withdrawal can trip
the switch, and `rearm()` then measures from the new value.)

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
to the end. It would keep the next episode flat too; to trade again, re-arm it:

```python
guarded.rearm()
print(guarded.killed, guarded.events[-1].kind)  # False rearm
```

## See also

- [Batched and differentiable environments](batched.md): `Market-v1`, which `LiveMarketEnv`
  mirrors.
- API: [`sekai.finance.live`](../api/finance.md#paper-and-live-trading),
  [`LiveMarketEnv`](../api/envs.md#sekai.envs.finance.LiveMarketEnv).
