"""Compare classical finance baselines in sekai's finance environments.

- OptimalExecution-v0: the Almgren-Chriss optimal schedule vs. TWAP.
- MarketMaking-v0:     Avellaneda-Stoikov quotes vs. fixed symmetric quotes.
- Portfolio-v0:        equal weight vs. all cash vs. a single asset.

These closed-form policies are the bar a learned policy should clear.
"""

import numpy as np

import sekai


def evaluate(env: sekai.Env, policy, episodes: int = 200) -> tuple[float, float]:
    totals = []
    for seed in range(episodes):
        result = env.reset(seed=seed)
        total, t = 0.0, 0
        while not result.done:
            result = env.step(policy(env, t))
            total += result.reward
            t += 1
        totals.append(total)
    return float(np.mean(totals)), float(np.std(totals))


print("== OptimalExecution-v0 (reward: -(shortfall + risk penalty), in bps) ==")
execution = sekai.make("OptimalExecution-v0")
ac = execution.almgren_chriss_actions()
n = execution.n_steps
for name, policy in {
    "Almgren-Chriss": lambda env, t: ac[t],
    "TWAP": lambda env, t: np.array([1 / (n - t)], np.float32),
    "sell at once": lambda env, t: np.ones(1, np.float32),
}.items():
    mean, std = evaluate(execution, policy)
    print(f"  {name:15s} {mean:8.1f} ± {std:6.1f}")

print("== MarketMaking-v0 (reward: risk-adjusted PnL) ==")
mm = sekai.make("MarketMaking-v0")
for name, policy in {
    "Avellaneda-Stoikov": lambda env, t: env.avellaneda_stoikov_action(),
    "fixed 0.5 / 0.5": lambda env, t: np.full(2, 0.5, np.float32),
    "fixed 2.0 / 2.0": lambda env, t: np.full(2, 2.0, np.float32),
}.items():
    mean, std = evaluate(mm, policy)
    print(f"  {name:19s} {mean:7.2f} ± {std:5.2f}")

print("== Portfolio-v0 (reward: log growth over one year) ==")
portfolio = sekai.make("Portfolio-v0")
k = portfolio.n_assets
for name, weights in {
    "equal weight": np.ones(k + 1, np.float32),
    "all cash": np.eye(k + 1, dtype=np.float32)[k],
    "lowest-vol asset": np.eye(k + 1, dtype=np.float32)[0],
}.items():
    mean, std = evaluate(portfolio, lambda env, t, w=weights: w)
    print(f"  {name:16s} {mean:+.3f} ± {std:.3f}")
