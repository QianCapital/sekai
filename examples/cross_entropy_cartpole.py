"""Train a linear CartPole policy with the cross-entropy method, in pure NumPy.

No deep-learning framework needed: sample policy weights, keep the best 20%,
refit the sampling distribution, repeat. Reaches the maximum return of 500
within a handful of iterations.
"""

import numpy as np

import sekai

env = sekai.make("CartPole-v1")
rng = np.random.default_rng(0)
obs_dim = env.observation_space.shape[0]


def episode_return(weights: np.ndarray, seed: int) -> float:
    result = env.reset(seed=seed)
    total = 0.0
    while not result.done:
        action = int(result.observation @ weights[:obs_dim] + weights[obs_dim] > 0)
        result = env.step(action)
        total += result.reward
    return total


mean, std = np.zeros(obs_dim + 1), np.ones(obs_dim + 1)
for iteration in range(20):
    population = rng.normal(mean, std, size=(40, obs_dim + 1))
    returns = np.array(
        [episode_return(w, seed=iteration * 100 + i) for i, w in enumerate(population)]
    )
    elite = population[np.argsort(returns)[-8:]]
    mean, std = elite.mean(axis=0), elite.std(axis=0) + 1e-3
    score = np.mean([episode_return(mean, seed=10_000 + k) for k in range(5)])
    print(f"iteration {iteration:2d}: population best {returns.max():5.0f}, policy {score:5.1f}")
    if score >= env.spec.reward_threshold:
        print(f"solved: mean return {score:.0f} >= threshold {env.spec.reward_threshold:.0f}")
        break
