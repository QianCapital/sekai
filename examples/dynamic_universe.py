"""Trade a universe whose members list and delist, with point-in-time features.

Builds a synthetic ten-year history of 30 stocks that come and go (most are
alive for only part of the period), adds causal features, checks them for
look-ahead bias, and evaluates two baselines on Market-v0, where the
observation is a padded set of at most 16 assets with a validity mask.
"""

import numpy as np

import sekai
from sekai.finance import GBM, ListingProcess, MarketData, RecordPerformance, SpreadCost
from sekai.finance.features import check_causal, momentum, rolling_zscore

rng = np.random.default_rng(0)
n_assets, n_days = 30, 2520
model = ListingProcess(
    GBM(mu=rng.uniform(-0.05, 0.15, n_assets), sigma=rng.uniform(0.15, 0.45, n_assets)),
    initial_listed=0.4,
    listing_rate=0.3,
    delisting_rate=0.3,
    min_listed=4,
)
sim = model.simulate(rng, n_days - 1)
dates = np.busday_offset("2015-01-01", np.arange(n_days), roll="forward")  # trading days
history = MarketData(sim.prices, timestamps=dates, assets=[f"STK{i:02d}" for i in range(n_assets)])
alive = history.mask.sum(axis=1)
print(f"{history.n_assets} assets, between {alive.min()} and {alive.max()} listed on any day")

# Causal features: momentum and a z-scored version of it. check_causal perturbs the
# future and raises sekai.LookAheadError if a feature's past values change.
mom = momentum(history.prices, 60)
check_causal(lambda p: rolling_zscore(momentum(p, 60), 120), history.prices)
history = history.with_fields(momentum=np.tanh(10 * mom), mom_z=rolling_zscore(mom, 120))

env = sekai.make(
    "Market-v0",
    data=history,
    max_assets=16,  # at most 16 slots; an asset keeps its slot while listed
    features=["momentum", "mom_z"],
    lookback=20,
    episode_length=252,
    transaction_cost=SpreadCost(half_spread=5e-4),
)
print("assets observation:", env.observation_space["assets"])


def equal_weight(env, obs):
    return np.append(env.action_mask()[:-1], 0.0).astype(np.float32)


def momentum_tilt(env, obs):
    values, mask = obs["assets"]["values"], obs["assets"]["mask"]
    score = np.where(mask == 1, np.exp(3 * values[:, -1, 2]), 0.0)  # latest momentum feature
    return np.append(score, 0.0).astype(np.float32)


perf_env = RecordPerformance(env)
for name, policy in {"equal weight": equal_weight, "momentum tilt": momentum_tilt}.items():
    for seed in range(20):
        result = perf_env.reset(seed=seed)
        while not result.done:
            result = perf_env.step(policy(perf_env.unwrapped, result.observation))
    summary = perf_env.summary()
    print(
        f"{name:14s} Sharpe {summary['sharpe_ratio']:+.2f}  "
        f"annual return {summary['annualized_return']:+.1%}  "
        f"max drawdown {summary['max_drawdown']:.1%}"
    )
    perf_env.summaries.clear()

# The padded set normalises per feature over occupied slots only, so padding never biases it.
norm = sekai.NormalizeObservation(env, stats="ewm", halflife=500)
obs = norm.reset(seed=0).observation
print(
    "normalised slots:",
    obs["assets"]["values"].shape,
    "occupied:",
    int(obs["assets"]["mask"].sum()),
)
