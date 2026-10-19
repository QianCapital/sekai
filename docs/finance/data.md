# Market data and backtesting

Historical environments need three things that are easy to get subtly wrong: a panel of
data in which assets come and go, train/test splits that do not leak information across
their boundary, and features that use only what was known at the time. `sekai.finance`
provides each:

| Module | Contents |
|---|---|
| `sekai.finance.data` | `MarketData`: an immutable, time-aligned panel with chronological `split()` and `walk_forward()` |
| `sekai.finance.sampling` | episode samplers: where in the history each episode starts |
| `sekai.finance.features` | causal feature transforms, point-in-time `asof_join`, the `check_causal` look-ahead detector |

`Trading-v0`, `Portfolio-v0` and `Market-v0` all accept a `MarketData`.

## MarketData

A `MarketData` is `T` timestamps × `N` assets × named fields, plus market-wide
*exogenous* series. A NaN price means the asset is not listed (or not tradable) at that
time, so a universe whose members come and go is represented as it is, without
forward-filling or dropping assets (`Market-v0` treats any gap as a delisting, so fill
short halts with `features.forward_fill(prices, limit=...)` if they should not close
positions):

```python
import numpy as np
import sekai
from sekai.finance import MarketData

closes = np.array([
    [100.0, np.nan, 20.0],
    [101.0, np.nan, 20.5],
    [102.5, 50.0, 20.4],
    [101.8, 50.6, np.nan],
    [103.0, 51.2, np.nan],
])
data = MarketData(
    closes,
    timestamps=["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"],
    assets=["AAA", "BBB", "CCC"],
    fields={"volume": np.full((5, 3), 1e6)},
    exogenous={"rate": np.full(5, 0.05)},
)
print(data)  # MarketData(T=5, N=3, fields=['close', 'volume'], exogenous=['rate'], 2024-01-02 .. 2024-01-08)
print(data.mask.astype(int))
```

```text
[[1 0 1]
 [1 0 1]
 [1 1 1]
 [1 1 0]
 [1 1 0]]
```

`BBB` lists on the third day and `CCC` delists after it.

| Argument | Shape | Rules |
|---|---|---|
| `prices` | `(T, N)`, or `(T,)` for one asset | stored as the field `price_field` (default `"close"`); NaN = not listed; finite prices must be positive |
| `fields` | each `(T, N)` | more per-asset data: volumes, spreads, features; NaN allowed, infinities not |
| `timestamps` | `(T,)` | strictly increasing `datetime64` (date strings and `datetime` objects are converted) or numbers; default `0, 1, ..., T-1`; timezone-aware values raise `ValueError` ([below](#from-pandas)) |
| `assets` | `N` ids | unique; default `"asset_0", "asset_1", ...` |
| `exogenous` | each `(T,)` or `(T, k)` | market-wide series: interest rates, an index level, macro data |
| `warmup` | `int`, default 0 | leading rows that are lookback-only history: observed, never a decision row (set by [splits](#chronological-splits)) |

| Accessor | Returns |
|---|---|
| `prices`, `mask` | `(T, N)` prices (NaN where unlisted) and the boolean listing mask |
| `fields`, `field(name)` | every per-asset field (prices included), or one |
| `exogenous`, `exogenous_series(name)`, `exogenous_width(name)` | market-wide series, one of them, its number of columns |
| `timestamps`, `assets`, `n_steps`, `n_assets`, `is_datetime` | the axes |
| `warmup` | number of leading warm-up rows (0 unless the data came from `split` or `walk_forward`) |
| `periods_per_year` | rows per year implied by datetime timestamps (`None` for numeric ones) |
| `log_returns()` | `(T, N)` log returns, NaN in row 0 and wherever either price is missing |
| `index_of(t)` | row of the first timestamp at or after `t` (integers are row positions) |

`periods_per_year` is what `Market-v0` uses for its step length `dt`: about 252 for
exchange trading days, 261 for every weekday, 365.25 for calendar days. For the five rows
above it is 4 intervals over 6 days, `243.5`.

**Fields or exogenous?** A *field* has a value per asset and time (a volume, a quoted
spread, a momentum score); `Market-v0` observes the fields named in `features=` in each
asset's slot. An *exogenous* series has one value per time for the whole market (the risk-free
rate, a VIX level, a macro release); `Market-v0` observes those named in `exogenous=` as a
separate `"market"` window. Two field names mean something to the environments: `volume`
(units traded per step) and `spread` (quoted bid-ask spread as a fraction of the price) feed
the [cost models](execution.md#which-inputs-each-model-needs).

**Immutability.** Every array is read-only and a `MarketData` never changes after it is
built, so an environment and your feature code can share one object safely. Derived data
is a new object:

```python
try:
    data.prices[0, 0] = 0.0
except ValueError as error:
    print(error)  # assignment destination is read-only

quoted = data.with_fields(spread=np.full((5, 3), 4e-4))
print(sorted(quoted.fields), sorted(data.fields))  # ['close', 'spread', 'volume'] ['close', 'volume']
```

`with_exogenous(**series)` does the same for market-wide series.

### From pandas

With pandas installed (`pip install "sekai[pandas]"`), build from wide DataFrames (index =
timestamps, columns = assets). Every field is aligned to the prices' index and columns, and
anything missing becomes NaN:

```python
import pandas as pd

index = pd.bdate_range("2024-01-01", periods=4)
close = pd.DataFrame({"AAA": [100.0, 101.0, 102.0, 101.5], "BBB": [np.nan, np.nan, 50.0, 50.5]},
                     index=index)
volume = pd.DataFrame({"AAA": [1.0e6, 1.2e6, 0.9e6, 1.1e6]}, index=index)  # no BBB column
wide = MarketData.from_pandas(close, fields={"volume": volume},
                              exogenous={"rate": pd.Series(0.05, index=index)})
print(wide)  # MarketData(T=4, N=2, fields=['close', 'volume'], exogenous=['rate'], 2024-01-01 .. 2024-01-04)
print(wide.field("volume")[:, 1])  # [nan nan nan nan]
```

or from a long ("tidy") DataFrame with one row per (time, asset), as most databases
export it. A missing row is a missing price:

```python
tidy = pd.DataFrame({
    "date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-04"],
    "symbol": ["AAA", "BBB", "AAA", "BBB", "AAA"],
    "close": [100.0, 50.0, 101.0, 50.4, 101.5],
    "volume": [1.0e6, 2.0e5, 1.1e6, 1.9e5, 0.9e6],
})
long = MarketData.from_long(tidy, time="date", asset="symbol", price="close", fields=["volume"])
print(long.prices)
```

```text
[[100.   50. ]
 [101.   50.4]
 [101.5   nan]]
```

**Time zones.** Timestamps are timezone-naive. A timezone-aware index (or `datetime`
objects with a `tzinfo`, or strings with a UTC offset) raises a `ValueError` instead of
being shifted to UTC behind a warning, which would silently move every bar, and with it
every `options={"start": ...}` and `asof_join` alignment, by the UTC offset. Convert
explicitly first: `frame.tz_convert("UTC").tz_localize(None)` to work in UTC, or
`frame.tz_localize(None)` to keep the local wall-clock times. Keep event times for
`asof_join` in the same convention.

## Slicing and selection

The examples below use a synthetic history of five assets over 2,520 weekdays (the
[simulators](simulators.md) make one in a line); load real data the same way.

```python
from sekai.finance import GBM

rng = np.random.default_rng(0)
dates = np.busday_offset("2015-01-01", np.arange(2520), roll="forward")  # weekdays
model = GBM(mu=[0.08, 0.06, 0.10, 0.03, 0.05], sigma=[0.25, 0.18, 0.35, 0.06, 0.20])
history = MarketData(model.simulate(rng, 2519).prices, timestamps=dates,
                     assets=["EQ1", "EQ2", "EQ3", "BOND", "EQ4"])
print(history)  # MarketData(T=2520, N=5, fields=['close'], 2015-01-01 .. 2024-08-28)
print(round(history.periods_per_year, 1))  # 260.9

print(history[252:504])  # MarketData(T=252, N=5, fields=['close'], 2015-12-21 .. 2016-12-06)
print(history.between("2020-01-01", "2021-01-01"))  # MarketData(T=262, N=5, fields=['close'], 2020-01-01 .. 2020-12-31)
print(history.select(["EQ1", "BOND"]))  # MarketData(T=2520, N=2, fields=['close'], 2015-01-01 .. 2024-08-28)
```

`data[a:b]` slices rows by position, `between(start, end)` by time (`start <= t < end`,
either bound optional) and `select(assets)` picks columns by id. All of them carry every
field and exogenous series along.

## Chronological splits

`split(*points, embargo=0, history=0)` cuts the data at fractions of the rows (`0.8`), row
positions, or timestamps:

```python
train, test = history.split(0.8, embargo=5, history=30)
for part in (train, test):
    print(len(part), part.timestamps[0], part.timestamps[-1])
print(test.timestamps[30])  # 2022-09-30
```

```text
2016 2015-01-01 2022-09-22
529 2022-08-19 2024-08-28
```

The split point is row 2016 (80% of 2520). The test part's first *decision* is 5 rows
later (the embargo), at row 2021, and 30 rows before that (the history) are included
so that an environment with `lookback=30` can fill its first observation:

```text
row         0                        1991       2016   2021                     2520
            |---------------------------|----------|------|------------------------|
train       [======================================)
embargo                                            [======)
test                                    [  history: observed, never traded    )[== decisions ==)
```

- **Embargo** keeps the two periods' *decisions* apart. The last training rewards come
  from returns right up to the split point; any feature or reward that looks over a window
  (a 60-day momentum, a forward-looking label, autocorrelated returns) makes the first test
  decisions statistically dependent on the end of training. Skipping `embargo` rows removes
  that overlap, as an embargo does in purged cross-validation. Make it at least
  as long as the longest horizon over which your features or rewards look *forward*.
- **History** lets a test episode see the past without trading on it. A test environment
  with lookback `L` would otherwise spend its first `L` rows warming up, and the tempting fix
  of slicing the test set earlier by hand also lets it *trade* on rows that belong to
  training. With `history=lookback`, the first decision falls exactly at the start of the
  test period, and the extra rows only ever appear in observations. They may overlap the
  embargo and the end of training: observing the past is legitimate, trading on it twice is
  not.

The test part records its history rows as `test.warmup` (here 30), and every finance
environment (`Market-v0`, `Trading-v0`, `Portfolio-v0` and their native vector versions)
places episodes so that no decision falls on a warm-up row, whatever its lookback: a
shorter lookback just observes fewer of them, and `reset(options={"start": t})` on a
warm-up row raises a `ValueError`. Slices of a part keep the warm-up rows they contain
(`test[10:].warmup == 20`); `select`, `with_fields` and `with_exogenous` keep it.

!!! tip "Set `history` to at least the environment's `lookback`"
    With `history` smaller than the lookback, the first `lookback - history` test rows
    are only observed, never traded.

Several points give several parts, each after the first embargoed and given history:

```python
parts = history.split("2021-01-01", "2023-01-01", embargo=5, history=30)
print([len(part) for part in parts])  # [1566, 546, 458]
```

`walk_forward(train_size, test_size, *, step=None, embargo=0, history=0, expanding=False)`
yields rolling `(train, test)` folds: fold `k` trains on `train_size` rows (all rows so far
with `expanding=True`), skips `embargo` rows, and tests on the next `test_size`, with
`history` extra rows before them. Folds advance by `step` rows, by default `test_size`, so
the test periods tile the data:

```python
for train_fold, test_fold in history.walk_forward(3 * 261, 261, embargo=5, history=60):
    print(train_fold.timestamps[0], train_fold.timestamps[-1], "|",
          test_fold.timestamps[60], test_fold.timestamps[-1])
```

```text
2015-01-01 2018-01-01 | 2018-01-09 2019-01-08
2016-01-01 2019-01-01 | 2019-01-09 2020-01-08
2017-01-02 2020-01-01 | 2020-01-09 2021-01-07
2018-01-02 2020-12-31 | 2021-01-08 2022-01-07
2019-01-02 2021-12-31 | 2022-01-10 2023-01-09
2020-01-02 2023-01-02 | 2023-01-10 2024-01-09
```

A test fold of `history + test_size` rows holds exactly one episode of
`episode_length = test_size - 1` decisions (the last decision earns the return to the
fold's final row). See [walk-forward evaluation](#walk-forward-evaluation) below.

## Episode samplers

Every historical environment takes `sampler=`, which decides where each episode's window
of `lookback + episode_length + 1` rows starts:

| `sampler=` | Class | Episodes |
|---|---|---|
| `"random"` (default) | `RandomStart()` | uniformly random windows, for training |
| `"sequential"` | `SequentialStart(start=0, stride=None)` | consecutive windows, `stride` rows apart (default `episode_length`, so they tile the history); `epoch` becomes 1 once the last window has been handed out, then it wraps around |
| an `int` `i` | `FixedStart(i)` | always window `i`, to replay one period |
| an `EpisodeSampler` | your subclass | implement `sample(rng, n_starts, episode_length)`, returning a start in `[0, n_starts)` and drawing any randomness from `rng`, the env's generator |

On data with [warm-up rows](#chronological-splits) the sampler only sees the windows
whose first decision comes after them: `n_starts` counts those, and position `i` is the
`i`-th of them, so `FixedStart(0)` and the first `SequentialStart` window put the first
decision on the first row after the warm-up.

`SequentialStart` walks an evaluation period exactly once: `epoch` increments as soon as
the last window has been handed out, so a loop that resets at the top while `epoch == 0`
runs every window once.

```python
from sekai.finance import SequentialStart

sampler = SequentialStart()
env = sekai.make("Portfolio-v0", prices=test, lookback=30, episode_length=63, sampler=sampler)
first_decisions = []
while sampler.epoch == 0:
    result = env.reset(seed=len(first_decisions))
    first_decisions.append(str(result.info["timestamp"]))
    while not result.done:
        result = env.step(np.ones(6, np.float32))
print(first_decisions)
```

```text
['2022-09-30', '2022-12-28', '2023-03-27', '2023-06-22', '2023-09-19', '2023-12-15', '2024-03-13']
```

Rows after the last full window, too few for another whole episode (here the last 57),
are not visited; pick `episode_length` (or `stride`) to suit the period. A sampler is
stateful. `sekai.make_vec()` gives every sub-env its own copy (on every backend, native
included), so each sub-env behaves like a separate environment: with a sequential sampler
they all replay the same windows. To split a period across sub-envs instead, give each one
its own offset sampler or start (see
[Episode samplers](../guide/vector.md#episode-samplers)). One `EpisodeSampler` instance that
you pass to several environments you build yourself is shared between them.

**Starting at a given date.** `reset(options={"start": t})` overrides the sampler for one
episode. `t` is the timestamp of the episode's **first decision**: for dated data anything
`numpy.datetime64` accepts, or an `int` row position; for numeric timestamps a number,
which is a timestamp (as in `index_of`). The window begins `lookback` rows earlier, and
`info["timestamp"]` after the reset reports the decision time, so
`options={"start": info["timestamp"]}` replays an episode, also on a split whose numeric
timestamps no longer start at 0. (For plain arrays the timestamps are `0, 1, ..., T-1`, so
rows and timestamps coincide.) A start without enough history before it, on a warm-up
row, or without a full episode after it, raises a `ValueError` that names the valid range:

```python
env = sekai.make("Portfolio-v0", prices=test, lookback=30, episode_length=63)
print(env.reset(seed=0, options={"start": "2023-03-01"}).info["timestamp"])  # 2023-03-01
print(env.reset(options={"start": 30}).info["timestamp"])  # 2022-09-30
try:
    env.reset(options={"start": "2024-08-01"})
except ValueError as error:
    print(error)
```

```text
start '2024-08-01' (row 509) is outside the valid range of first decisions: rows 30..465 (2022-09-30 .. 2024-05-31).
```

A timestamp between rows resolves to the next row at or after it. Simulated environments
have no dates, so the option raises there.

## Point-in-time features

`sekai.finance.features` transforms arrays shaped `(T,)` or `(T, N)` along time. Every
transform is **causal**: the output at time `t` depends only on inputs at times `<= t`. So
features can be computed once over the whole history and attached as fields; an
environment then slices them without leaking the future. NaN means missing (not yet
listed, a halted day) and propagates as documented per function.

| Function | Output at time `t` |
|---|---|
| `log_returns(prices)`, `simple_returns(prices)` | one-period returns; NaN in row 0 and where either price is missing |
| `momentum(prices, window)` | $\log(P_t / P_{t-w})$ |
| `lag(x, periods=1)` | $x_{t-k}$ |
| `forward_fill(x, *, limit=None)` | last valid value, at most `limit` rows old |
| `rolling_mean(x, window, *, min_periods=None)`, `rolling_std(..., ddof=1)` | trailing-window statistics over rows $t-w+1..t$, ignoring NaN (as pandas `rolling`) |
| `rolling_zscore(x, window, *, min_periods=None)` | $(x_t - \text{mean}) / \text{std}$ over the trailing window |
| `ewm_mean(x, halflife, *, min_periods=1)`, `ewm_std(...)` | exponentially weighted statistics, $m_t = (1-a)\,m_{t-1} + a\,x_t$ with $a = 1 - 2^{-1/h}$ |
| `realized_volatility(returns, window, *, periods_per_year=None)` | trailing standard deviation of returns, annualised if asked |
| `cross_sectional_zscore(x)`, `cross_sectional_rank(x)` | each row standardised, or ranked into $[0, 1]$, across the assets listed at $t$ |
| `asof_join(target_times, event_times, values, *, lag=None)` | the last event value published by $t$ (below) |
| `check_causal(fn, x)` | raises `sekai.LookAheadError` if `fn` looks ahead (below) |

```python
from sekai.finance import features

returns = features.log_returns(history.prices)
mom = features.momentum(history.prices, 60)
history = history.with_fields(
    momentum=mom,
    momentum_rank=features.cross_sectional_rank(mom),
    volatility=features.realized_volatility(returns, 20, periods_per_year=261),
)
print(history)  # MarketData(T=2520, N=5, fields=['close', 'momentum', 'momentum_rank', 'volatility'], 2015-01-01 .. 2024-08-28)
print(history.field("volatility")[-1].round(3))  # [0.233 0.183 0.367 0.063 0.267]
```

`Market-v0` observes such fields per asset with `features=["momentum", "volatility"]`
(missing values appear as 0, next to an availability flag); see
[Market-v0](../environments/finance.md#market-v0).

### As-of joins

Data that arrives at irregular times (macro releases, earnings, index rebalances) is
aligned to bar timestamps with `asof_join`: each target time gets the value of the last
event with `event_time + lag <= t`, the latest information that was *public* then. A
monthly inflation figure refers to the month's end but is published about two weeks later;
joining on the reference date without the lag would let the agent trade on it early:

```python
bars = np.array(["2024-02-08", "2024-02-09", "2024-02-12", "2024-02-13", "2024-02-14"],
                dtype="datetime64[D]")
month_end = np.array(["2023-12-31", "2024-01-31"], dtype="datetime64[D]")
cpi = np.array([3.4, 3.1])
print(features.asof_join(bars, month_end, cpi))  # [3.1 3.1 3.1 3.1 3.1]
print(features.asof_join(bars, month_end, cpi, lag=np.timedelta64(13, "D")))  # [3.4 3.4 3.4 3.1 3.1]
```

With the publication lag, January's figure is known from 13 February. Attach the result
with `with_exogenous(cpi=...)`; before the first release it is NaN.

## Detecting look-ahead

`check_causal(fn, x)` tests a transform empirically: for a few random cut points `c` it
changes the rows of `x` after `c` and checks that the output for rows `0..c` is unchanged.
Each cut is tried with several perturbations, so that a leak cannot hide in what one of
them preserves:

| Perturbation | Applies to | Catches leaks through |
|---|---|---|
| rescaled: valid entries times $e^{\varepsilon}$, $\varepsilon \sim \mathcal N(0, 0.1^2)$ | all data | future values |
| rows reordered | all data | the order of future values |
| signs flipped at random, additive noise, some entries set to 0 | data with negative values | the sign of future values (`np.sign` of next-period returns) |
| $\lvert x + \text{noise}\rvert$, some entries set to 0 | non-negative data with zeros | future zeros (no volume tomorrow) |
| some valid entries set to NaN, NaNs filled with values from `x` | data with NaNs | the future NaN pattern: survivorship bias, e.g. dropping assets that delist later |

Perturbations stay in the data's domain (strictly positive prices stay positive), and
numpy floating-point warnings are silenced while `fn` runs on perturbed copies. A causal
feature passes silently:

```python
features.check_causal(lambda p: features.rolling_zscore(features.momentum(p, 60), 250),
                      history.prices)
```

Standardising with full-sample statistics, a common slip, uses the future at every row:

```python
def full_sample_zscore(prices):
    r = features.log_returns(prices)
    return (r - np.nanmean(r, axis=0)) / np.nanstd(r, axis=0)


try:
    features.check_causal(full_sample_zscore, history.prices)
except sekai.LookAheadError as error:
    print(error)
```

```text
Look-ahead bias detected: the output at time index 1 changed when only inputs after time index 680 (the cut point) were perturbed (rescaled).
```

`fn` must be deterministic and return an array aligned with the input's time axis.
`n_checks` (default 5) sets the number of cut points, `rng` their source, and `rtol` /
`atol` the comparison tolerance. Run it in your tests for every feature you add.

## Walk-forward evaluation

Putting the pieces together: six one-year folds of the synthetic history, each test year
embargoed from three years of training and given 60 days of history. The baseline is
inverse-volatility weights *estimated on the training fold* and held fixed through the
test year (rebalanced daily), against equal weight; `RecordPerformance` summarises each
test episode ([Performance metrics](metrics.md)).

```python
from sekai.finance import RecordPerformance

LOOKBACK, YEAR = 60, 261


def fit_inverse_volatility(train):
    """Target weights from the training period only: proportional to 1 / volatility."""
    vol = np.nanstd(train.log_returns(), axis=0)
    return np.append(1 / vol, 0.0).astype(np.float32)


equal_weight = np.append(np.ones(5), 0.0).astype(np.float32)
sharpes = {"equal weight": [], "inverse volatility": []}
print(f"{'test from':10s}  {'policy':18s} {'Sharpe':>6s} {'return':>7s} {'max DD':>6s}")
for train, test in history.walk_forward(3 * YEAR, YEAR, embargo=5, history=LOOKBACK):
    env = RecordPerformance(
        sekai.make("Portfolio-v0", prices=test, lookback=LOOKBACK, episode_length=YEAR - 1),
        periods_per_year=YEAR,
    )
    for name, action in {"equal weight": equal_weight,
                         "inverse volatility": fit_inverse_volatility(train)}.items():
        result = env.reset(seed=0)
        start = str(result.info["timestamp"])
        while not result.done:
            result = env.step(action)
        p = result.info["performance"]
        sharpes[name].append(p.sharpe_ratio)
        print(f"{start:10s}  {name:18s} {p.sharpe_ratio:+6.2f} {p.annualized_return:+7.1%} "
              f"{p.max_drawdown:6.1%}")
print({name: round(float(np.mean(s)), 2) for name, s in sharpes.items()})
```

```text
test from   policy             Sharpe  return max DD
2018-01-09  equal weight        +1.67  +18.0%   5.0%
2018-01-09  inverse volatility  +1.79  +12.5%   3.9%
2019-01-09  equal weight        +0.03   -0.2%   8.3%
2019-01-09  inverse volatility  +1.00   +6.4%   4.0%
2020-01-09  equal weight        -0.31   -3.9%  10.2%
2020-01-09  inverse volatility  +0.22   +1.3%   6.8%
2021-01-08  equal weight        +2.81  +34.7%   7.9%
2021-01-08  inverse volatility  +3.19  +23.7%   4.7%
2022-01-10  equal weight        +0.53   +5.3%   8.9%
2022-01-10  inverse volatility  +0.43   +2.7%   5.0%
2023-01-10  equal weight        +0.19   +1.5%  12.3%
2023-01-10  inverse volatility  +1.11   +7.7%   6.8%
{'equal weight': 0.82, 'inverse volatility': 1.29}
```

Each test fold has exactly one window, so the sampler is irrelevant here; with longer test
periods use `sampler="sequential"` to walk through them. A learning agent slots into the
same loop: train it on an environment built from `train` (random windows), then evaluate the
frozen policy on `test`. Because the history rows are never traded, the
summaries cover the test years only.

## See also

- [Market simulators](simulators.md): synthetic histories, dynamic universes and domain randomisation.
- [Costs, accounts and rewards](execution.md): the `volume` and `spread` fields in use.
- [`sekai.finance.data` API](../api/finance.md#market-data).
