"""Train a policy by backpropagating through a batched market environment (torch).

    pip install "sekai[torch]"

The market is exogenous (prices do not react to the agent), and every part of a
`BatchMarketEnv` step (the constraint projection, spread and impact costs, the
accounting, the reward) is differentiable in the action. So the gradient of the
episode's reward with respect to the policy's parameters can be computed exactly by
backpropagation through time, instead of estimated from sampled actions as policy
gradient methods do: direct (pathwise) policy optimisation, in the spirit of Moody &
Saffell's direct reinforcement.

The scenarios carry an "alpha" signal that predicts the next period's returns a
little (an information coefficient of 0.1). The policy is
`scores = a * alpha + b * current_book`, a leverage logit `c`, projected onto a
dollar-neutral book with per-name caps. Training finds that leaning on the current
book (`b > 0`) cuts turnover, and so costs, without giving up much of the signal.

Not run in CI (it needs PyTorch).
"""

import numpy as np
import torch

from sekai.envs.finance import BatchMarketEnv
from sekai.finance import (
    Constraints,
    MarketPanel,
    ObservationBuilder,
    ScoresLeverage,
    SimpleReturn,
    SpreadCost,
    SquareRootImpact,
)
from sekai.spaces import Box


class AlphaScenarios:
    """Returns driven by a persistent alpha signal (an AR(1) per asset) plus noise."""

    def __init__(self, n_assets=50, n_steps=64, ic=0.1, vol=0.02, persistence=0.9):
        self.n_assets, self.n_steps = n_assets, n_steps
        self.ic, self.vol, self.rho = ic, vol, persistence

    def sample(self, n_paths, *, seed):
        g = torch.Generator().manual_seed(seed)
        shape = (n_paths, self.n_steps, self.n_assets)
        shocks = torch.randn(shape, generator=g)
        alpha = torch.empty(shape)
        alpha[:, 0] = shocks[:, 0]
        for t in range(1, self.n_steps):
            alpha[:, t] = self.rho * alpha[:, t - 1] + (1 - self.rho**2) ** 0.5 * shocks[:, t]
        noise = torch.randn(shape, generator=g)
        returns = torch.zeros(shape)
        returns[:, 1:] = self.vol * (
            self.ic * alpha[:, :-1] + (1 - self.ic**2) ** 0.5 * noise[:, 1:]
        )
        return MarketPanel.from_returns(
            returns,
            fields={
                "alpha": alpha,
                "dollar_volume": torch.full(shape, 2e7),
                "spread": torch.full(shape, 1e-3),
            },
        )


class AlphaAndBook(ObservationBuilder):
    """Per slot: the alpha signal and the current weight (scaled to unit mean size)."""

    fields = ("alpha",)

    def space(self, env):
        return Box(-np.inf, np.inf, shape=(env.n_slots, 2), dtype=np.float32)

    def __call__(self, view):
        w = view.weights
        book = w / (w.abs().mean(-1, keepdim=True) + 1e-8)
        return torch.stack([view.field("alpha"), book], dim=-1)


rules = Constraints(max_gross=2.0, max_position=0.1, dollar_neutral=True)
env = BatchMarketEnv(
    128,
    scenarios=AlphaScenarios(n_steps=81),
    lookback=20,  # the impact model estimates each asset's volatility over the lookback
    episode_length=60,
    execution="close",  # the MDP of the simulation: decide after the close, earn the next period
    observation=AlphaAndBook(),
    action=ScoresLeverage(rules),
    transaction_cost=SpreadCost() + SquareRootImpact(0.5),
    reward=SimpleReturn(),
    capital=1e8,
    backend="torch",
    autoreset=False,
)
env.differentiable = True

params = torch.tensor([1.0, 0.0, 0.0], requires_grad=True)  # a, b, c
optimiser = torch.optim.Adam([params], lr=0.05)


def act(obs, p):
    scores = p[0] * obs[..., 0] + p[1] * obs[..., 1]
    return torch.cat([scores, p[2].expand(len(obs), 1)], dim=-1)


def episode(p, seed):
    """Mean reward per step, annualised Sharpe and mean turnover of one batch of episodes."""
    result = env.reset(seed=seed)
    rewards, turnover = [], []
    while not bool(result.dones.all()):
        result = env.step(act(result.observations, p))
        rewards.append(result.rewards)
        turnover.append(result.info["turnover"])
    r = torch.stack(rewards)
    sharpe = (r.mean(0) / (r.std(0) + 1e-8)).mean() * 252**0.5
    return r.mean(), sharpe, torch.stack(turnover).mean()


with torch.no_grad():
    _, sharpe, turnover = episode(params, seed=10**6)
print(f"before: a, b, c = {params.tolist()}  Sharpe {sharpe:.2f}, turnover {turnover:.2f}/day")
for iteration in range(60):
    loss = -episode(params, seed=iteration)[0] * 1e4  # basis points per step
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
with torch.no_grad():
    _, sharpe, turnover = episode(params, seed=10**6)  # held-out scenarios
learned = [round(v, 2) for v in params.tolist()]
print(f"after:  a, b, c = {learned}  Sharpe {sharpe:.2f}, turnover {turnover:.2f}/day")
