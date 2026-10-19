"""Synthetic markets: stylised facts, domain randomisation, and hidden regimes.

Compares the return distributions of several market models, trains-in-spirit on a
randomised market (each episode draws new drifts and volatilities, so a policy
cannot overfit one parameter set), and shows the hidden regime of Market-v0's
default market both as privileged information and as an observable feature.
"""

import numpy as np

import sekai
from sekai.finance import (
    GARCH,
    GBM,
    Heston,
    MertonJumpDiffusion,
    Randomized,
    RegimeSwitching,
)

rng = np.random.default_rng(0)
models = {
    "GBM": GBM(sigma=0.2),
    "Heston": Heston(),
    "Merton jumps": MertonJumpDiffusion(jump_intensity=5.0),
    "GARCH(1,1)": GARCH(omega=2e-6, alpha=0.1, beta=0.88),
    "regime switching": RegimeSwitching(
        [GBM(mu=0.1, sigma=0.12), GBM(mu=-0.3, sigma=0.45)], [[0.99, 0.01], [0.04, 0.96]]
    ),
}
print(f"{'model':17s} {'ann. vol':>8s} {'kurtosis':>9s} {'|r| autocorr':>13s}")
for name, model in models.items():
    prices = model.simulate(rng, 5000).prices[:, 0]
    r = np.diff(np.log(prices))
    kurt = ((r - r.mean()) ** 4).mean() / r.var() ** 2 - 3
    a = np.abs(r) - np.abs(r).mean()
    clustering = (a[1:] * a[:-1]).mean() / a.var()  # volatility clustering
    print(f"{name:17s} {r.std() * np.sqrt(252):8.1%} {kurt:9.2f} {clustering:13.3f}")

# Domain randomisation: every episode of Portfolio-v0 gets freshly drawn market parameters.
randomized = Randomized(
    lambda g: GBM(mu=g.uniform(-0.1, 0.2, 4), sigma=g.uniform(0.1, 0.5, 4), n_assets=4)
)
env = sekai.make("Portfolio-v0", model=randomized, episode_length=126)
growth = []
for seed in range(30):
    result, total = env.reset(seed=seed), 0.0
    while not result.done:
        result = env.step(np.ones(5, np.float32))  # equal weight
        total += result.reward
    growth.append(np.expm1(total))
print(f"equal weight over randomised markets: mean {np.mean(growth):+.1%}, sd {np.std(growth):.1%}")

# Market-v0's default market has a hidden calm/stress regime. Expose it to a critic only
# (privileged_state) or to the policy as well (exogenous).
env = sekai.make("Market-v0", privileged_state=True, exogenous=["regime"], episode_length=252)
result, stressed = env.reset(seed=3), 0
while not result.done:
    result = env.step(np.append(env.action_mask()[:-1], 0).astype(np.float32))
    stressed += int(result.info["state"]["regime"])
window = result.observation["market"].shape
print(f"days in the stressed regime: {stressed} of 252; observed regime window: {window}")
