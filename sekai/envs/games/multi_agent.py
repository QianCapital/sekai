"""The two-player games as multi-agent environments, with an agent for each side.

`MultiAgentGame` turns a two-player game env, which the caller plays from both sides, into
a `sekai.MultiAgentEnv`. It follows sekai's convention for event-driven multi-agent
environments (see the multi-agent guide):

- `observations` holds only the agent to move, with its own observation: its keys say
  who moves, and `step()` takes exactly that agent's action.
- `rewards`, `terminated`, `truncated` and `info` cover both agents. The reward of a step
  is the mover's, and its negative is the other agent's (the games are zero-sum), so at
  the end the winner gets `+1` and the loser `-1` (in poker, their net in big blinds).
- On the last step both agents get a final observation, and `agents` becomes empty.

    env = MultiAgentGame(sekai.make("Chess-v0"))
    result = env.reset(seed=0)
    while not result.all_done:
        agent, = result.observations
        mask = env.action_mask(agent)
        result = env.step({agent: env.action_space(agent).sample(mask=mask)})
    print(result.rewards)  # e.g. {'white': 1.0, 'black': -1.0}
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from sekai._types import SeedType
from sekai.core.env import Env
from sekai.core.multi_agent import MultiAgentEnv
from sekai.core.result import MAResetResult, MAStepResult, StepResult
from sekai.envs.games.wrappers import check_private_views
from sekai.error import ResetRequired
from sekai.utils.masking import get_action_mask


class MultiAgentGame(MultiAgentEnv[Any, Any, str]):
    """A two-player game (chess, go, xiangqi or heads-up hold'em) as a `MultiAgentEnv`.

    The agents are the game's sides (`env.sides`: `"white"` and `"black"`, `"black"` and
    `"white"`, `"red"` and `"black"`, `"p1"` and `"p2"`), each with the game's observation
    and action spaces. See the module docstring for the protocol.

    An agent's `info` is the game's, as that agent sees it: its `legal_action_mask` (and
    `action_mask(agent)`) marks the legal actions of the agent to move and nothing for the
    other, and in poker the viewer's entries (`observer`, `view`, `result`) are the
    agent's own seat's. Final observations: in poker each agent's own view; in the board
    games the final position as the game encodes it, from the side to move, for both.

    A reset into a position that is already over (a `fen`, or `moves` replayed, that
    reaches the end of a game) returns no observations: nobody is to move.

    Args:
        env: A two-player game env, possibly wrapped (in `TimeLimit`, say). Wrappers that
            change the observations go outside: poker refuses them inside, where they
            would see both seats' private views.
    """

    def __init__(self, env: Env[Any, Any]) -> None:
        base = env.unwrapped
        players = getattr(base, "players", None)
        sides = getattr(base, "sides", None)
        if players is None or sides is None or len(players) != 2:
            raise TypeError(
                f"MultiAgentGame needs a two-player game env, got {type(base).__name__}."
            )
        check_private_views(env, "MultiAgentGame")
        self.env = env
        self.possible_agents = list(sides)
        self.agents: list[str] = []
        self.observation_spaces = dict.fromkeys(sides, env.observation_space)
        self.action_spaces = dict.fromkeys(sides, env.action_space)
        self._side_of = dict(zip(players, sides, strict=True))
        self._player_of = dict(zip(sides, players, strict=True))
        self._to_move: str | None = None

    @property
    def unwrapped(self) -> Any:
        """The game env itself (what the game renderers draw)."""
        return self.env.unwrapped

    def reset(
        self, *, seed: SeedType = None, options: dict[str, Any] | None = None
    ) -> MAResetResult[Any, str]:
        self._set_rng(seed)
        result = self.env.reset(seed=seed, options=options)
        over = bool(getattr(self.env.unwrapped, "done", False))
        self._to_move = None if over else self._side_of[result.info["to_play"]]
        self.agents = [] if over else list(self.possible_agents)
        observations = {} if self._to_move is None else {self._to_move: result.observation}
        info = {agent: self._info_for(agent, result.info) for agent in self.possible_agents}
        return MAResetResult(observations=observations, info=info)

    def step(self, actions: dict[Any, Any]) -> MAStepResult[Any, str]:
        mover = self._to_move
        if mover is None:
            raise ResetRequired()
        if set(actions) != {mover}:
            raise ValueError(
                f"It is {mover}'s turn: step() takes {{{mover!r}: action}}, got actions for "
                f"{sorted(map(str, actions))}."
            )
        result: StepResult[Any] = self.env.step(actions[mover])
        other = self._other(mover)
        reward = float(result.reward)
        if result.done:
            self._to_move = None
            observations = {agent: self._final_observation(agent, result) for agent in self.agents}
            self.agents = []
        else:
            self._to_move = self._side_of[result.info["to_play"]]
            observations = {self._to_move: result.observation}
        return MAStepResult(
            observations=observations,
            rewards={mover: reward, other: -reward if reward else 0.0},
            terminated=dict.fromkeys(self.possible_agents, bool(result.terminated)),
            truncated=dict.fromkeys(self.possible_agents, bool(result.truncated)),
            info={agent: self._info_for(agent, result.info) for agent in self.possible_agents},
        )

    def action_mask(self, agent: str) -> NDArray[np.int8]:
        """The agent's legal actions (sekai's action-mask convention): the game's mask for
        the agent to move, all zeros for the other."""
        if agent not in self._player_of:
            raise KeyError(f"No agent {agent!r}; the agents are {self.possible_agents}.")
        mask = np.asarray(get_action_mask(self.env), dtype=np.int8)
        return mask if agent == self._to_move else np.zeros_like(mask)

    def close(self) -> None:
        self.env.close()

    # ------------------------------------------------------------------

    def _other(self, agent: str) -> str:
        first, second = self.possible_agents
        return str(second if agent == first else first)

    def _seat_view(self, agent: str) -> tuple[Any, dict[str, Any]] | None:
        view = getattr(self.env.unwrapped, "_seat_view", None)
        if view is None:
            return None
        return cast("tuple[Any, dict[str, Any]]", view(self._player_of[agent]))

    def _final_observation(self, agent: str, result: StepResult[Any]) -> Any:
        own = self._seat_view(agent)
        return result.observation if own is None else own[0]

    def _info_for(self, agent: str, info: dict[str, Any]) -> dict[str, Any]:
        mine = dict(info)
        if agent != self._to_move:  # nothing to do until its turn
            mine["legal_action_mask"] = np.zeros_like(info["legal_action_mask"])
            mine["legal_actions"] = np.zeros(0, dtype=np.int64)
        own = self._seat_view(agent)
        if own is not None:  # a game of hidden information: the agent's own view
            mine.update(own[1])
        return mine

    def __str__(self) -> str:
        return f"<MultiAgentGame{self.env}>"


__all__ = ["MultiAgentGame"]
