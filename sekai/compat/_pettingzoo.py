from __future__ import annotations

from typing import Any

import numpy as np
from pettingzoo.utils.env import ParallelEnv

from sekai.compat._gymnasium import space_from_gymnasium, space_to_gymnasium
from sekai.core.multi_agent import MultiAgentEnv
from sekai.core.result import MAResetResult, MAStepResult


class SekaiToPettingZoo(ParallelEnv):  # type: ignore[misc]
    """Exposes a sekai MultiAgentEnv through PettingZoo's Parallel API.

    Spaces are converted to gymnasium spaces; results become the usual
    `(observations, rewards, terminations, truncations, infos)` dicts.
    Event-driven sekai envs (e.g. LimitOrderBookMarket) return observations
    only for the agents that act next, which a strict Parallel-API consumer
    may not expect; their rewards and flags still cover every agent.
    """

    def __init__(self, env: MultiAgentEnv[Any, Any, Any]) -> None:
        self.sekai_env = env
        self.possible_agents = list(env.possible_agents)
        self.agents = list(env.agents)
        self.observation_spaces = {
            a: space_to_gymnasium(env.observation_space(a)) for a in self.possible_agents
        }
        self.action_spaces = {
            a: space_to_gymnasium(env.action_space(a)) for a in self.possible_agents
        }
        self.metadata = {"name": type(env).__name__, "render_modes": [], "is_parallelizable": True}
        self.render_mode = None

    def observation_space(self, agent: Any) -> Any:
        return self.observation_spaces[agent]

    def action_space(self, agent: Any) -> Any:
        return self.action_spaces[agent]

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[Any, Any], dict[Any, Any]]:
        result = self.sekai_env.reset(seed=seed, options=options)
        self.agents = list(self.sekai_env.agents)
        infos = {a: dict(result.info.get(a, {})) for a in result.observations}
        return dict(result.observations), infos

    def step(self, actions: dict[Any, Any]) -> tuple[dict[Any, Any], ...]:
        r = self.sekai_env.step(actions)
        self.agents = list(self.sekai_env.agents)
        return (
            dict(r.observations),
            {a: float(v) for a, v in r.rewards.items()},
            {a: bool(v) for a, v in r.terminated.items()},
            {a: bool(v) for a, v in r.truncated.items()},
            {a: dict(v) for a, v in r.info.items()},
        )

    def render(self) -> None:
        return None

    def close(self) -> None:
        self.sekai_env.close()

    def __str__(self) -> str:
        return f"<SekaiToPettingZoo{self.sekai_env}>"


class PettingZooToSekai(MultiAgentEnv[Any, Any, Any]):
    """Exposes a PettingZoo ParallelEnv through the sekai MultiAgentEnv API.

    The PettingZoo env keeps its own randomness; seeds passed to reset() go
    straight to it (a NumPy integer as an int). PettingZoo has no standard way
    to hand an env a Generator, so a Generator seed raises TypeError.
    """

    def __init__(self, env: Any) -> None:
        self.pz_env = env
        self.possible_agents = list(env.possible_agents)
        self.agents = list(getattr(env, "agents", self.possible_agents))
        self.observation_spaces = {
            a: space_from_gymnasium(env.observation_space(a)) for a in self.possible_agents
        }
        self.action_spaces = {
            a: space_from_gymnasium(env.action_space(a)) for a in self.possible_agents
        }

    def reset(
        self, *, seed: Any = None, options: dict[str, Any] | None = None
    ) -> MAResetResult[Any, Any]:
        if isinstance(seed, np.random.Generator):
            raise TypeError(
                "PettingZoo envs take an int seed, not a numpy Generator; "
                "pass e.g. seed=int(rng.integers(2**31)) to derive one from it."
            )
        if isinstance(seed, np.integer):
            seed = int(seed)  # gymnasium's seeding, used by many PettingZoo envs, rejects these
        observations, infos = self.pz_env.reset(seed=seed, options=options)
        self.agents = list(self.pz_env.agents)
        if isinstance(seed, int):
            self._rng_seed = seed
        return MAResetResult(dict(observations), {a: dict(i) for a, i in infos.items()})

    def step(self, actions: dict[Any, Any]) -> MAStepResult[Any, Any]:
        observations, rewards, terminations, truncations, infos = self.pz_env.step(actions)
        self.agents = list(self.pz_env.agents)
        return MAStepResult(
            observations=dict(observations),
            rewards={a: float(r) for a, r in rewards.items()},
            terminated={a: bool(t) for a, t in terminations.items()},
            truncated={a: bool(t) for a, t in truncations.items()},
            info={a: dict(i) for a, i in infos.items()},
        )

    def close(self) -> None:
        self.pz_env.close()

    def __str__(self) -> str:
        return f"<PettingZooToSekai {type(self.pz_env).__name__}>"
