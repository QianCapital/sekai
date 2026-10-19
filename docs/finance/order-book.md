# Order books

`sekai.finance.lob` is a limit order book for one instrument, in integer price ticks,
plus a stochastic model of the background traders that keep it alive. Together they are
the market behind [`LimitOrderBook-v0`](../environments/finance.md#limitorderbook-v0) and
the multi-agent [`LimitOrderBookMarket`](../environments/finance.md#limitorderbookmarket),
and they work on their own for simulation studies.

## OrderBook

A continuous double auction with **price-time priority**: an incoming order matches the
opposite side while the prices cross, best price first and first in, first out within a
price level, and always executes at the resting (maker) order's price. Whatever a limit
order cannot fill rests in the book; whatever a market order cannot fill is dropped.

```python
import numpy as np
from sekai.finance import BACKGROUND, OrderBook, Side

book = OrderBook()
book.limit(Side.ASK, 101, 5)  # background sellers (owner BACKGROUND by default)
book.limit(Side.ASK, 102, 10)
book.limit(Side.BID, 99, 4)
mine, fills = book.limit(Side.BID, 99, 3, owner=7)  # joins the queue behind 4 lots
print(book)  # OrderBook(time=0, best_bid=99, best_ask=101, orders=4)
print(book.mid(), book.spread(), book.queue_ahead(mine), book.volume_at(Side.BID, 99))  # 100.0 2 4 7
```

A market buy of 8 lots takes the whole level at 101 and part of the next:

```python
order_id, fills = book.market(Side.BID, 8, owner=7)
for fill in fills:
    print(fill.price, fill.quantity, fill.maker_owner, fill.taker_owner)
print(book.best_ask(), book.last_price)  # 102 102
```

```text
101 5 -1 7
102 3 -1 7
```

A sell limit at 99 for 5 lots crosses the bid: it fills the 4 background lots that arrived
first, then 1 of owner 7's 3, and fills completely (so it returns no resting id):

```python
order_id, fills = book.limit(Side.ASK, 99, 5, owner=3)
print(order_id, [(fill.quantity, fill.maker_owner) for fill in fills])  # None [(4, -1), (1, 7)]
print(book.get(mine))  # Order(order_id=4, side=<Side.BID: 1>, price=99, quantity=2, owner=7, time=0.0)
```

| Method | Does |
|---|---|
| `limit(side, price, quantity, owner=BACKGROUND)` | match, then rest the remainder; returns `(order_id or None, fills)` |
| `market(side, quantity, owner=BACKGROUND)` | match and drop the remainder; returns `(order_id, fills)` |
| `cancel(order_id)` | remove a resting order; False if it is no longer resting |
| `reduce(order_id, quantity)` | lower a resting order's quantity, keeping its place in the queue (0 cancels) |
| `best_bid()`, `best_ask()`, `mid()`, `spread()` | top of book in ticks (`None` if a side is empty) |
| `depth(levels)` | `(bid_prices, bid_sizes, ask_prices, ask_sizes)`, best level first, zero-padded |
| `volume_at(side, price)`, `n_levels(side)` | aggregate volume at a price; number of price levels |
| `queue_ahead(order_id)` | volume that must trade at the order's price before it starts to fill |
| `get(order_id)`, `orders_of(owner)`, `n_orders_of(owner)` | resting orders (live records: read, never mutate) |
| `clear()` | empty the book and reset `time`, `last_price` and the id counter |

`Side.BID` is +1 and `Side.ASK` -1 (the sign of the owner's position change on a fill).
A `Fill` records the price, quantity, both order ids and owners, the taker's side and the
time; an `Order` its id, side, price, remaining quantity, owner and arrival time. `owner`
is any integer: `BACKGROUND` (-1) for background flow, agent ids otherwise. Self-trades are
allowed. `book.time` is a plain attribute the caller (or the background flow) advances; it
stamps orders and fills. `book.last_price` is the last fill's price.

```python
print(book.depth(3))  # (array([99,  0,  0]), array([2, 0, 0]), array([102,   0,   0]), array([7, 0, 0]))
print(book.reduce(mine, 1), book.queue_ahead(mine), book.cancel(mine), book.cancel(mine))  # True 0 True False
```

**Complexity.** Prices are kept in sorted lists, so creating or removing a price level is
an $O(\log L)$ search over $L$ levels; matching costs $O(1)$ per fill. Cancelling is $O(1)$
by *lazy deletion*: a cancelled order is zeroed in place and dropped when it reaches the
front of its queue (or when cancelled orders come to outnumber live ones at a level),
while every level's aggregate volume stays exact.

## Zero-intelligence order flow

`ZeroIntelligenceFlow` drives a book with randomly arriving orders in the spirit of Smith,
Farmer, Gillemot & Krishnamurthy (2003) and Cont, Stoikov & Talreja (2010): traders with no
strategy at all, whose aggregate behaviour nevertheless produces a spread, a depth profile
and a diffusing price. Three kinds of event arrive as independent Poisson processes:

- **Limit orders**, at rate $\lambda_L$ (`limit_rate`) per side. A buy is placed $k$ ticks
  below the best ask and a sell $k$ ticks above the best bid, with
  $k \in \{1, \dots, K\}$ ($K$ = `max_offset`) and
  $\Pr(k) \propto e^{-\alpha (k - 1)}$ ($\alpha$ = `offset_decay`; 0 is uniform). So a
  limit order never crosses on arrival, and $k = 1$ improves on, or joins, the best
  quote.
- **Market orders**, at rate $\lambda_M$ (`market_rate`) per side, taking the opposite side.
- **Cancellations**, at rate $\delta$ (`cancel_rate`) *per resting background order*, each
  removing a uniformly random one. Agents' orders are never cancelled by the flow.

Order sizes are $1 + \text{Poisson}(\bar m - 1)$, so their mean is $\bar m$ (`mean_size`).
With $n$ resting background orders the next event comes after an exponential time with
total rate

$$
\Lambda = 2\lambda_L + 2\lambda_M + \delta\, n ,
$$

which is recomputed after every event. If an event empties one side of the book, a
background limit order is placed there at once, so the book stays two-sided.

| Parameter | Default | |
|---|---|---|
| `limit_rate` | `10.0` | $\lambda_L$, limit orders per unit time per side |
| `market_rate` | `2.0` | $\lambda_M$, market orders per unit time per side |
| `cancel_rate` | `0.5` | $\delta$, cancellations per resting order per unit time |
| `max_offset` | `10` | $K$, the farthest limit-order distance in ticks |
| `offset_decay` | `0.5` | $\alpha$, decay of the offset distribution per tick |
| `mean_size` | `3.0` | $\bar m$, mean order size |

`initialise(book, rng, *, mid, levels=10, orders_per_level=3)` clears a book and seeds it
symmetrically around `mid`; `advance(book, rng, until)` simulates every event up to time
`until` and returns the fills, including those against agents' resting orders:

```python
from sekai.finance import ZeroIntelligenceFlow

flow = ZeroIntelligenceFlow()
rng = np.random.default_rng(0)
flow.initialise(book, rng, mid=10_000)
print(book)  # OrderBook(time=0, best_bid=9999, best_ask=10001, orders=60)
fills = flow.advance(book, rng, until=20.0)
print(book, len(fills))  # OrderBook(time=20, best_bid=10000, best_ask=10001, orders=37) 129
```

### What the parameters do

The steady state is set by the balance between orders arriving (limit orders) and
leaving (fills and cancellations). The table measures it over ten sessions of 300 time
units after a warm-up of 20, from the standard deviation of the mid's change over one time
unit (ticks per $\sqrt{\text{time}}$; it scales diffusively), the mean spread, the mean
number of resting orders and the number of fills per unit time:

```python
def measure(flow, sessions=10, length=300, warmup=20):
    moves, spreads, depth, n_fills = [], [], [], 0
    for seed in range(sessions):
        rng = np.random.default_rng(seed)
        book = OrderBook()
        flow.initialise(book, rng, mid=10_000)
        flow.advance(book, rng, warmup)
        mids = [book.mid()]
        for t in range(1, length + 1):
            n_fills += len(flow.advance(book, rng, warmup + t))
            mids.append(book.mid())
            spreads.append(book.spread())
            depth.append(len(book))
        moves.append(np.diff(mids))
    return np.concatenate(moves).std(), np.mean(spreads), np.mean(depth), n_fills / (sessions * length)


print(f"{'flow':20s} {'mid vol':>8s} {'spread':>7s} {'orders':>7s} {'fills':>6s}")
for label, params in {
    "defaults": {},
    "cancel_rate=0.1": {"cancel_rate": 0.1},
    "cancel_rate=1.0": {"cancel_rate": 1.0},
    "cancel_rate=2.0": {"cancel_rate": 2.0},
    "limit_rate=20": {"limit_rate": 20.0},
    "market_rate=4": {"market_rate": 4.0},
    "offset_decay=0.1": {"offset_decay": 0.1},
    "offset_decay=2.0": {"offset_decay": 2.0},
}.items():
    vol, spread, orders, fills = measure(ZeroIntelligenceFlow(**params))
    print(f"{label:20s} {vol:8.2f} {spread:7.2f} {orders:7.1f} {fills:6.1f}")
```

```text
flow                  mid vol  spread  orders  fills
defaults                 0.50    1.08    32.1    6.5
cancel_rate=0.1          0.05    1.00   158.0    6.7
cancel_rate=1.0          1.31    1.21    16.3    6.4
cancel_rate=2.0          2.60    1.54     8.6    6.1
limit_rate=20            0.05    1.00    72.2    6.6
market_rate=4            1.31    1.29    24.8   13.0
offset_decay=0.1         1.45    1.61    32.2    6.6
offset_decay=2.0         0.04    1.00    31.9    6.7
```

Volatility comes from the best quotes being consumed faster than they are replenished:

- **More cancellation** (or more market orders) thins the book, so the best levels empty
  more often and the mid moves more; the spread widens slowly.
- **More limit orders** thicken the queues at the best prices until the mid barely moves.
- **Offset decay** concentrates new orders near the touch (large $\alpha$, deep best
  levels, a frozen mid) or spreads them out (small $\alpha$, thin best levels, more
  volatility and a wider spread).
- The fill rate is driven by `market_rate` (each market order may fill several resting
  orders); `mean_size` scales volumes without changing the price dynamics much.

The defaults give a liquid, one-tick market whose mid diffuses by about half a tick per
unit time: with decisions every time unit, an agent sees the price move on most steps
without being swamped by it.

## In the environments

`LimitOrderBook-v0` (`sekai.envs.finance.LimitOrderBookEnv`) puts one agent (owner 0) in a
book driven by a `ZeroIntelligenceFlow` (pass your own with `flow=`). Each episode
initialises the book at `mid_price` and lets the flow run for `warmup` time units; then, every
`decision_interval`, the agent rests or moves a bid and an ask (or sends a market order)
and the flow advances to the next decision. Quote levels count from the best price among
the *other* traders' orders (level 1 joins it, level $k$ rests $k - 1$ ticks behind), and a
quote already at the requested level is left alone, including a level-1 quote that is itself
the best price, alone there; so repeating an action moves a quote only to follow the other
traders' best price, never off the best. A quote left at the same price keeps its
queue position, so `queue_ahead` matters and appears in the observation; moving it sends it
to the back. The live book is `env.unwrapped.book`.

`LimitOrderBookMarket` puts several agents (owners 0, 1, ...) in the same book, each
deciding on its own clock: an event-driven multi-agent environment in which one agent's
market order can fill another's resting quote. The observations, actions and rewards, the
baselines and the multi-agent stepping rules are on the
[environments page](../environments/finance.md#limitorderbook-v0).

## See also

- [Finance environments](../environments/finance.md#limitorderbook-v0): `LimitOrderBook-v0` and `LimitOrderBookMarket`.
- [`MarketMaking-v0`](../environments/finance.md#marketmaking-v0): the reduced-form
  (Avellaneda–Stoikov) market-making problem, with a closed-form baseline.
- [`sekai.finance.lob` API](../api/finance.md#order-book).
