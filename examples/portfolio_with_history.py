"""Run Portfolio-v0 on your own price history, with wrappers for an MLP policy.

Real data would come from a CSV or database; here we synthesise five years of
daily prices for four assets so the example runs offline.
"""

import numpy as np

import sekai
from sekai.envs.finance import simulate_gbm

rng = np.random.default_rng(7)
prices = simulate_gbm(  # shape (1261, 4): ~5 years of daily closes
    rng, 1260, s0=[50, 120, 30, 80], mu=[0.08, 0.05, 0.12, 0.02], sigma=[0.25, 0.15, 0.4, 0.1]
)

env = sekai.make(
    "Portfolio-v0", prices=prices, episode_length=126, lookback=20, transaction_cost=5e-4
)
print("observation space:", env.observation_space)

# Most RL libraries want a flat vector: flatten the Dict and add time-to-go.
flat = sekai.TimeAwareObservation(sekai.FlattenObservation(env), max_episode_steps=126)
print("flattened:", flat.observation_space)


def momentum(obs: np.ndarray) -> np.ndarray:
    """Overweight assets with the best recent returns (obs: 20x4 returns, 5 weights, time)."""
    recent = obs[: 20 * 4].reshape(20, 4).sum(axis=0)
    scores = np.exp(recent * 10)
    return np.append(scores / scores.sum(), 0.0).astype(np.float32)


for name, policy in {
    "momentum": momentum,
    "equal weight": lambda obs: np.ones(5, np.float32),
}.items():
    growth = []
    for seed in range(50):  # each episode starts at a random date in the history
        result = flat.reset(seed=seed)
        total = 0.0
        while not result.done:
            result = flat.step(policy(result.observation))
            total += result.reward
        growth.append(np.expm1(total))
    print(f"{name:13s} mean 6-month return {np.mean(growth):+.2%}")
