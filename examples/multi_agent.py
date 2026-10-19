"""A two-agent market: a market maker quotes, a trader decides whether to cross the spread."""

from typing import Any

import numpy as np

import sekai


class QuoteAndTradeEnv(sekai.MultiAgentEnv[Any, Any, str]):
    def __init__(self, n_steps: int = 50) -> None:
        self.n_steps = n_steps
        self.possible_agents = ["maker", "trader"]
        self.agents = list(self.possible_agents)
        self.observation_spaces = {
            "maker": sekai.Box(-np.inf, np.inf, shape=(2,)),  # [mid, inventory]
            "trader": sekai.Box(-np.inf, np.inf, shape=(3,)),  # [mid, bid, ask]
        }
        self.action_spaces = {
            "maker": sekai.Box(0.01, 1.0, shape=(1,)),  # half spread
            "trader": sekai.Discrete(3),  # sell / pass / buy
        }

    def _obs(self, half_spread: float) -> dict[str, np.ndarray]:
        return {
            "maker": np.array([self.mid, self.inventory], np.float32),
            "trader": np.array(
                [self.mid, self.mid - half_spread, self.mid + half_spread], np.float32
            ),
        }

    def reset(self, *, seed: Any = None, options: Any = None) -> sekai.MAResetResult:
        self._set_rng(seed)  # same seeding API as sekai.Env
        self.mid, self.inventory, self.t = 100.0, 0, 0
        return sekai.MAResetResult(self._obs(0.1), {a: {} for a in self.agents})

    def step(self, actions: dict[str, Any]) -> sekai.MAStepResult:
        half = float(actions["maker"][0])
        side = int(actions["trader"]) - 1  # -1 sell, 0 pass, +1 buy
        next_mid = self.mid + self.rng.normal(0, 0.3)
        # The trader pays the spread; both mark to the next mid.
        trader_pnl = side * (next_mid - self.mid) - abs(side) * half
        self.inventory -= side
        maker_pnl = abs(side) * half + self.inventory * (next_mid - self.mid)
        self.mid, self.t = next_mid, self.t + 1
        done = {a: self.t >= self.n_steps for a in self.agents}
        return sekai.MAStepResult(
            observations=self._obs(half),
            rewards={"maker": maker_pnl, "trader": trader_pnl},
            terminated=done,
            truncated={a: False for a in self.agents},
            info={a: {} for a in self.agents},
        )


env = QuoteAndTradeEnv()
result = env.reset(seed=0)
totals = {a: 0.0 for a in env.agents}
while True:
    actions = {a: env.action_space(a).sample() for a in env.agents}
    result = env.step(actions)
    for agent, reward in result.rewards.items():
        totals[agent] += reward
    if result.all_done:
        break
print({agent: round(total, 2) for agent, total in totals.items()})
