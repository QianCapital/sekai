"""Walk-forward evaluation without leakage between training and test periods.

Splits a synthetic 12-year, 6-asset history into rolling folds: train on four
years, skip a short embargo, test on the next year. Each test environment can
see `lookback` days before its period as history, but every decision falls
inside it, and the sequential sampler walks through the test year exactly once.
"""

import numpy as np

import sekai
from sekai.finance import MarketData, RecordPerformance, SequentialStart
from sekai.finance.simulators import GBM, RegimeSwitching

rng = np.random.default_rng(1)
n = 6
calm = GBM(
    mu=0.10,
    sigma=np.linspace(0.08, 0.30, n),
    corr=np.full((n, n), 0.3) + 0.7 * np.eye(n),
    n_assets=n,
)
stress = GBM(
    mu=-0.25,
    sigma=np.linspace(0.20, 0.60, n),
    corr=np.full((n, n), 0.8) + 0.2 * np.eye(n),
    n_assets=n,
)
market = RegimeSwitching([calm, stress], [[0.995, 0.005], [0.02, 0.98]])
sim = market.simulate(rng, 12 * 252 - 1)
dates = np.busday_offset("2012-01-02", np.arange(12 * 252), roll="forward")
data = MarketData(sim.prices, timestamps=dates)

LOOKBACK, YEAR = 60, 252


def inverse_volatility(obs):
    vol = obs["returns"].std(axis=0) + 1e-8
    return np.append(1 / vol, 0.0).astype(np.float32)


def equal_weight(obs):
    return np.ones(n + 1, np.float32)


print(f"{'test from':10s} {'policy':18s} {'Sharpe':>7s} {'return':>8s} {'max DD':>7s}")
for _train, test in data.walk_forward(4 * YEAR, YEAR, embargo=5, history=LOOKBACK):
    # A learner would be trained on Portfolio-v0 built from `_train` here; these
    # baselines need no training, so only the test periods are used.
    for name, policy in {
        "equal weight": equal_weight,
        "inverse volatility": inverse_volatility,
    }.items():
        env = RecordPerformance(
            sekai.make(
                "Portfolio-v0",
                prices=test,
                lookback=LOOKBACK,
                episode_length=YEAR - 1,
                sampler=SequentialStart(),
            )
        )
        result = env.reset(seed=0)
        first_decision = result.info["timestamp"]
        while not result.done:
            result = env.step(policy(result.observation))
        p = env.last_summary
        start = str(first_decision)[:7]
        print(
            f"{start:10s} {name:18s} {p.sharpe_ratio:+7.2f} {p.annualized_return:+8.1%} "
            f"{p.max_drawdown:7.1%}"
        )
