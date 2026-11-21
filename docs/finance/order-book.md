# Order books

`sekai.finance.lob` is a limit order book for one instrument, in integer price ticks,
plus a stochastic model of the background traders that keep it alive, and a
[replay](#historical-replay) of historical order flow from files you own. Together they are
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
is any integer: `BACKGROUND` (-1) for background flow, agent ids otherwise. The book
itself allows self-trades (an order may fill against a resting order of the same owner);
the environments prevent them for their agents (below). `book.time` is a plain attribute the caller (or the background flow) advances; it
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

## Historical replay

A market-making result only holds up when the agent has traded against real order flow,
with a realistic queue position. `MessageReplay` is background flow that replays a
historical message stream into an `OrderBook`, with the same
`advance(book, rng, until)` interface as `ZeroIntelligenceFlow`, so it plugs into the
environments as `flow=`. sekai never fetches or sells market data: you bring your own
files (`read_lobster` parses the LOBSTER format, below), or build the arrays yourself.

| Class | Holds |
|---|---|
| `OrderBookMessages(time, event, order_id, size, price, side)` | the message stream, one NumPy array per field, validated (sorted times, known events, positive sizes, distinct submitted ids...) and read-only |
| `MessageType` | `SUBMIT` (1), `CANCEL` (2, partial), `DELETE` (3), `EXECUTE` (4, visible), `EXECUTE_HIDDEN` (5), `CROSS` (6, auction), `HALT` (7): LOBSTER's event types |
| `BookSnapshot(bid_prices, bid_sizes, ask_prices, ask_sizes)` | the visible book at one moment, best level first (the layout of `OrderBook.depth`) |
| `BookSnapshots(...)` | the visible book after every message, `(n, levels)` arrays (a LOBSTER orderbook file); `snapshots[i]` is a `BookSnapshot`; rows may be crossed or locked, as around halts (`crossed` marks them) |

Prices are integer ticks and `side` is the side of the order a message is about, so the
execution of a bid (`side=1`) is a sale. `MessageReplay(messages, snapshot=None,
snapshots=None, cap_to_history=False)` starts from `snapshot`, the book before the first
message (empty if `None`), and `initialise(book, start=t)` sets the book to its state after
every message stamped at or before time `t` by replaying them; with `snapshots` it is
seeded from the row of the last of those messages instead, in O(levels). A snapshot
enters the book as one `BACKGROUND` order per price level, ahead of every later order at
that price. A crossed or locked row (best bid at or above the best ask, as a recorded book
can be during a halt) is not the state an `OrderBook` can be in, so a replay starting there
is seeded from the last uncrossed row before it and replays the messages since.

```python
from sekai.finance import BookSnapshot, MessageReplay, MessageType, OrderBookMessages

# The book at 10:00:00 (36,000 seconds after midnight), best level first.
snapshot = BookSnapshot(
    bid_prices=np.array([100, 99]),
    bid_sizes=np.array([5, 8]),
    ask_prices=np.array([102, 103]),
    ask_sizes=np.array([4, 6]),
)
M = MessageType
messages = OrderBookMessages(
    time=np.array([36_000.2, 36_000.5, 36_000.9, 36_001.5]),
    event=np.array([M.SUBMIT, M.EXECUTE, M.CANCEL, M.DELETE]),
    order_id=np.array([41, 900, 41, 41]),  # order 900 rested before 10:00: in the snapshot
    size=np.array([3, 4, 1, 2]),
    price=np.array([99, 102, 99, 99]),
    side=np.array([1, -1, 1, 1]),
)
replay = MessageReplay(messages, snapshot=snapshot)
book = OrderBook()
replay.initialise(book, start=36_000.0)
print(book)  # OrderBook(time=36000, best_bid=100, best_ask=102, orders=4)
mine, _ = book.limit(Side.ASK, 101, 2, owner=7)  # improve the best ask
for fill in replay.advance(book, None, until=36_001.0):
    print(fill.price, fill.quantity, fill.maker_owner, fill.taker_side.name)
print(book.depth(2))  # (array([100,  99]), array([ 5, 10]), array([102, 103]), array([2, 6]))
```

```text
101 2 7 BID
102 2 -1 BID
```

The historical buyer took 4 lots at 102; replayed, it meets the better-priced ask of owner
7 first, then the rest at 102. The replay is deterministic (`rng` is unused and may be
`None`); `replay.step(book)` applies one message at a time, `replay.position` is the next
message, and `replay.halted` says whether trading is halted. A replay keeps the position of
the book it drives, so use one per book (`replay.spawn()` makes another; the environments
take their own copy).

### The replay model

Historical orders rest in the book as `BACKGROUND` orders and keep their price-time
priority, so an agent's order competes with them for its place in the queue. Every message
is applied to the *current* book, which the agent's trades may have changed:

| Message | Replayed as |
|---|---|
| submission | a new background limit order; if an agent order rests at a crossing price, it trades with it first, as it would have |
| partial cancel, delete | reduce, or cancel what is left of, the order if it still rests; the agent may already have filled it, and then nothing happens |
| visible execution | a marketable order of the executed size against the executed order's side, limited at the execution price: whatever has priority in the current book fills first (an agent order at a better price, or ahead in the queue), and an unfilled remainder is dropped, never trading through the historical price |
| hidden execution | the same, so only orders at least as good as the execution price can fill (visible orders have priority over hidden ones at a price): in a book that follows history there are none, but an agent quote there is filled (the rest traded with hidden liquidity, which is not in the book, so it is not among the environments' observed `trades`) |
| halt | sets `halted` until trading resumes (the environments then send no agent orders); cross trades are ignored |

Without agents the replay reproduces the historical book exactly, after every message,
provided the snapshot holds every level the messages touch.

### Biases of the replay

Replaying a fixed history around an agent is an approximation, with known biases:

- **The flow is exogenous.** Historical traders do not react to the agent: nobody cancels,
  re-prices or trades differently because of its quotes or trades, so the agent's impact
  on *later* order flow is not modelled. Results for an agent that trades a large share of
  the volume are optimistic.
- **Consumed liquidity is gone.** An execution of an order that the agent's trades already
  consumed (or moved back in the queue) walks further into the queue at that price,
  filling orders behind it that history did not fill, or goes unfilled; a historical cancel
  of a consumed order does nothing.
- **Displaced quantity stays.** When an agent order ahead in the queue absorbs part of a
  historical execution, the historical order keeps the quantity it would have traded, until
  its own later messages remove it: the counterfactual book that fixed order sizes imply,
  in which the agent's liquidity can hold a level that history saw consumed. Pass
  `cap_to_history=True` to cut every background order to its historical remaining size
  after each message about it instead, so the background book never exceeds history (and
  the agent's passive fills leave it unchanged).
- **Orders from before the start are aggregated.** A snapshot gives levels, not orders:
  each level enters the book as one order, and later messages about the orders in it reduce
  it. Orders deeper than the snapshot's levels are unknown, so the book can lack depth far
  from the touch until those levels refill; an execution of such an order fills only orders
  at strictly better prices.
- **No crossed books.** An `OrderBook` matches at once, so orders submitted during a halt
  that cross trade immediately instead of in the reopening auction; episodes spanning a
  halt are approximate. An episode that starts while the recorded book is crossed starts
  from the last uncrossed snapshot before then, with the messages since replayed onto it:
  the book a replay of the whole history would give.

### Loading LOBSTER files

[LOBSTER](https://lobsterdata.com) reconstructs NASDAQ order books from exchange messages
and delivers, per stock and day, a message file and an orderbook file (CSV, no header).
`read_lobster(message_path, orderbook_path=None, *, tick_size=0.01, levels=None)` parses
files you own with NumPy alone (pandas is not needed) and returns
`(messages, snapshots)` (`snapshots` is `None` without an orderbook file):

- **Message file**: time (seconds after midnight), event type (1 to 7, as `MessageType`),
  order id, size, price (dollars times 10,000) and direction (1 buy, -1 sell order; for an
  execution, the side of the resting order).
- **Orderbook file**: the visible book after each message, `levels` blocks of ask price,
  ask size, bid price, bid size; empty levels hold the dummy prices 9999999999 (ask) and
  -9999999999 (bid) with size 0, and are read as price 0, size 0. Rows around halts and
  auctions can be crossed or locked (the best bid at or above the best ask): they are read
  as they are and marked in `snapshots.crossed`, and a replay never seeds a book from one
  (above).

Prices become integer ticks of `tick_size` dollars, and visible orders must lie on that
grid. Hidden executions may not (at the midpoint, say), so their prices are rounded away
from the book's interior (up for a hidden bid, down for a hidden ask), which lets a replayed
hidden execution fill only orders at least as good as its true price. Halt messages keep
LOBSTER's code in `price`: -1 halt, 0 quoting resumes, 1 trading resumes.

```python
from sekai.finance import MessageReplay, read_lobster
from sekai.envs.finance import LimitOrderBookEnv

messages, snapshots = read_lobster(
    "AAPL_2012-06-21_34200000_57600000_message_10.csv",
    "AAPL_2012-06-21_34200000_57600000_orderbook_10.csv",
    tick_size=0.01,
)
replay = MessageReplay(messages, snapshots=snapshots)
env = LimitOrderBookEnv(flow=replay, session_length=300.0, warmup=0.0, latency=0.002)
result = env.reset(seed=0)  # a random five-minute window of the day
result = env.reset(options={"start": 36_000.0})  # first decision at 10:00:00
print(result.info["time"])  # 36000.0, on the messages' clock
```

### Latency

`LimitOrderBookEnv(latency=...)` delays the agent's orders: an action decided at time $t$
reaches the book at $t + \ell$. The background flow, simulated or replayed, runs to that
time first (messages stamped at or before it come first), then the action's cancels and
orders are applied, and the flow runs on to the next decision. `latency` is a number in
$[0, \Delta]$ ($\Delta$ = `decision_interval`), or a function of the env's random generator
for a random latency drawn per action (a draw outside that range raises `ValueError`):

```python
env = LimitOrderBookEnv(latency=0.05)  # in the flow's time units: seconds for a replay
env = LimitOrderBookEnv(latency=lambda rng: min(rng.exponential(0.2), 1.0))
```

An order message carries a price, so a delayed action is **priced when it is decided**,
from the book the agent observed: a bid sent to join the best bid can arrive behind a new
best bid, or cross a falling ask and trade at once as a taker; a market order executes
against the book it finds on arrival; a cancel can arrive after the fill it was meant to
prevent. A quote the action keeps sends nothing, so it stays as it is (filled or not).
Inventory limits are checked on arrival, and an action arriving after the session's end is
dropped. With `latency=0` (the default) an action is carried out at once, its quotes
priced after its market order.

Observations are taken at decision time: the agent sees the book as it is when it decides.
To delay what it sees as well, wrap the environment in
[`sekai.DelayObservation`](../guide/time.md#latency), which delays observations by whole
steps. The inventory-skew policy of the
[environments page](../environments/finance.md#limitorderbook-v0) loses its edge as its
quotes go stale (the mean session reward over seeds 0 to 19 of `LimitOrderBook-v0`, with
its standard error):

```text
latency 0.0        +82.94 ± 1.73
latency 0.1        +81.82 ± 1.60
latency 0.5        +61.55 ± 1.81
latency exp(0.2)   +76.57 ± 1.67
```

## In the environments

`LimitOrderBook-v0` (`sekai.envs.finance.LimitOrderBookEnv`) puts one agent (owner 0) in a
book driven by a `ZeroIntelligenceFlow` (pass your own with `flow=`). Each episode
initialises the book at `mid_price` and lets the flow run for `warmup` time units; then, every
`decision_interval`, the agent may send a market order, then rests or moves a bid and an
ask, and the flow advances to the next decision. With `flow=MessageReplay(...)` the book
starts in its historical state instead, and each episode is a window of the message
stream: `sampler=` places the windows on a grid of `decision_interval` from the first
message, and `reset(options={"start": t})` makes the first decision at time `t` (a float or
`Time(t)`; an int is a grid position), as with the
[episode samplers](data.md#episode-samplers) of the historical environments;
`info["time"]` is then the book's time on the messages' clock. (Simulated flow has no
history to start in, and ignores `start`, as it did in 0.2.) With `latency=` the agent's
orders reach the book later ([above](#latency)). A market order first cancels the agent's
own resting quote on the opposite side (a buy cancels its ask, a sell its bid), so the agent
never trades with itself: self-trade prevention, as in the cancel-resting mode of most
exchanges; the quote is then placed again if the action asks for it, at the back of its
queue. Quote levels count from the best price among
the *other* traders' orders (level 1 joins it, level $k$ rests $k - 1$ ticks behind), and a
quote already at the requested level is left alone, including a level-1 quote that is itself
the best price, alone there; so repeating an action moves a quote only to follow the other
traders' best price, never off the best. A quote left at the same price keeps its
queue position, so `queue_ahead` matters and appears in the observation; moving it sends it
to the back. The live book is `env.unwrapped.book`.

`LimitOrderBookMarket` puts several agents (owners 0, 1, ...) in the same book, each
deciding on its own clock: an event-driven multi-agent environment in which one agent's
market order can fill another's resting quote (but not its own: self-trade prevention
applies to every agent). All the actions of a step are checked before any is carried out.
It also takes a `MessageReplay` as `flow=` (and a `sampler=`), but has no `latency`: its
traders' orders reach the book at once. The observations, actions and rewards, the
baselines and the multi-agent stepping rules are on the
[environments page](../environments/finance.md#limitorderbook-v0).

## See also

- [Finance environments](../environments/finance.md#limitorderbook-v0): `LimitOrderBook-v0` and `LimitOrderBookMarket`.
- [`MarketMaking-v0`](../environments/finance.md#marketmaking-v0): the reduced-form
  (Avellaneda–Stoikov) market-making problem, with a closed-form baseline.
- [`sekai.finance.lob` API](../api/finance.md#order-book).
