"""A small simultaneous-move multi-agent env for interop tests."""

from __future__ import annotations

from typing import Any

import numpy as np

import sekai


class QuoteAndTrade(sekai.MultiAgentEnv[Any, Any, str]):
    def __init__(self, n_steps: int = 20) -> None:
        self.n_steps = n_steps
        self.possible_agents = ["maker", "trader"]
        self.agents = list(self.possible_agents)
        self.observation_spaces = {
            "maker": sekai.Box(-np.inf, np.inf, shape=(2,)),
            "trader": sekai.Box(-np.inf, np.inf, shape=(3,)),
        }
        self.action_spaces = {
            "maker": sekai.Box(0.01, 1.0, shape=(1,)),
            "trader": sekai.Discrete(3),
        }
        self.mid, self.inventory, self.t = 100.0, 0, 0

    def _obs(self, half: float) -> dict[str, np.ndarray]:
        return {
            "maker": np.array([self.mid, self.inventory], np.float32),
            "trader": np.array([self.mid, self.mid - half, self.mid + half], np.float32),
        }

    def reset(self, *, seed: Any = None, options: Any = None) -> sekai.MAResetResult:
        self._set_rng(seed)
        self.agents = list(self.possible_agents)
        self.mid, self.inventory, self.t = 100.0, 0, 0
        return sekai.MAResetResult(self._obs(0.1), {a: {} for a in self.agents})

    def step(self, actions: dict[str, Any]) -> sekai.MAStepResult:
        half = float(actions["maker"][0])
        side = int(actions["trader"]) - 1
        self.mid += float(self.rng.normal(0, 0.3))
        self.inventory -= side
        self.t += 1
        done = self.t >= self.n_steps
        result = sekai.MAStepResult(
            observations=self._obs(half),
            rewards={"maker": abs(side) * half, "trader": -abs(side) * half},
            terminated=dict.fromkeys(self.agents, done),
            truncated=dict.fromkeys(self.agents, False),
            info={a: {} for a in self.agents},
        )
        if done:
            self.agents = []
        return result
