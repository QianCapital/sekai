"""Wrappers for the two-player games.

The two-player game envs are played from both sides by the caller (self-play): each step
is a move by the player to act, who sees the observation from their own side.
`OpponentWrapper` turns one into a single-agent environment against a fixed opponent.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from sekai._types import ActType, ObsType, SeedType
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.core.wrapper import ObservationWrapper, Wrapper
from sekai.envs.games.policies import Policy, RandomPolicy
from sekai.utils.seeding import make_seed_sequence

MAX_REDEALS = 100
"""How many games in a row the opponent may end before the agent's first move."""


class OpponentWrapper(Wrapper[ObsType, ActType, ObsType, ActType]):
    """A two-player game against a fixed opponent: one agent, the other side automatic.

    After each agent action the opponent plays until it is the agent's turn again (or the
    game ends), so the agent always observes a position in which it is to act. Rewards are
    the agent's: each game reward is for the player who just moved, so the opponent's are
    subtracted (the games are zero-sum). At the end of a game the agent gets `+1` for a
    win and `-1` for a loss in the board games, and its net in big blinds in poker.

    The last observation of a game is the env's: in the board games it is from the side to
    move, which is the opponent's when the agent's own move ended the game (or a
    `TimeLimit` inside stopped it); in poker, where the observation is a seat's private
    view, it is always the agent's own.

    Args:
        env: A two-player game env (chess, go, xiangqi or heads-up hold'em), possibly
            wrapped (e.g. in `TimeLimit`). Wrappers that change the observations, or that
            work across steps (`ActionRepeat`, `StickyAction`, `NormalizeReward`), go
            outside (`FrameStackObservation(OpponentWrapper(env))`): inside, they would
            span both players' turns. Poker refuses observation wrappers inside, where
            they would see both seats' private views.
        opponent: Any policy with `act(observation, info) -> action`; by default
            `RandomPolicy`. If it has a `seed()` method, `reset(seed=...)` re-seeds it
            with a seed derived from that one, so its choices are independent of the
            env's own draws.
        agent_color: The side the agent plays: a side's name (`"white"`, `"black"`,
            `"red"`, `"p1"`, `"p2"`), a value of the env's `info["to_play"]`, `"first"`,
            `"second"`, or `"random"` (drawn from the env's generator at every reset).

    `info` gains `"agent_color"` (the agent's side, as `info["to_play"]` names it) and,
    after the opponent has moved, `"opponent_action"` and `"opponent_move"` (`None` if
    the opponent's action was illegal and forfeited the game).

    A game the opponent ends before the agent has moved (in poker, a fold from the small
    blind) gives the agent nothing to do, so `reset()` deals again; the agent's rewards in
    the games it skipped are in `info["skipped_rewards"]` (present only if it skipped any),
    for an evaluation to count. `reset()` raises `RuntimeError` if the opponent ends
    `MAX_REDEALS` games in a row, as a deterministic one might from a fixed position. A
    game that is already over when the env resets is returned as it is.
    """

    def __init__(
        self,
        env: Env[ObsType, ActType],
        opponent: Policy | None = None,
        agent_color: Any = "first",
    ) -> None:
        super().__init__(env)
        base = env.unwrapped
        players = getattr(base, "players", None)
        sides = getattr(base, "sides", None)
        if players is None or sides is None or len(players) != 2:
            raise TypeError(
                f"OpponentWrapper needs a two-player game env, got {type(base).__name__}."
            )
        if getattr(base, "_seat_view", None) is not None:
            # A game of hidden information: each observation is one seat's private view, and
            # the wrapper hands the agent its own from the env itself.
            layer: Any = env
            while isinstance(layer, Wrapper):
                inner = layer.env
                if isinstance(layer, ObservationWrapper) or (
                    layer.observation_space != inner.observation_space
                ):
                    raise TypeError(
                        f"{type(layer).__name__} inside OpponentWrapper would see both seats' "
                        f"private views of {type(base).__name__}: wrap the OpponentWrapper in "
                        "it instead (an opponent that needs its own can apply it in act())."
                    )
                layer = inner
        self.opponent: Policy = opponent if opponent is not None else RandomPolicy()
        self._players: tuple[Any, Any] = (players[0], players[1])
        self._spec = self._parse(agent_color, tuple(sides))
        self.agent_color: Any = self._players[0]

    def _parse(self, color: Any, sides: tuple[str, ...]) -> Any:
        if isinstance(color, str):
            key = color.lower()
            if key == "random":
                return "random"
            if key == "first":
                return self._players[0]
            if key == "second":
                return self._players[1]
            if key in sides:
                return self._players[sides.index(key)]
        elif any(color is p or (type(color) is type(p) and color == p) for p in self._players):
            return color
        raise ValueError(
            f"agent_color must be one of {list(sides)}, {list(self._players)}, 'first', "
            f"'second' or 'random', got {color!r}."
        )

    def _resolve(self) -> Any:
        if self._spec == "random":
            draw = int(self.env.unwrapped.rng.integers(2))
            return self._players[0] if draw else self._players[1]
        return self._spec

    def reset(
        self, *, seed: SeedType = None, options: dict[str, Any] | None = None
    ) -> ResetResult[ObsType]:
        skipped: list[float] = []
        while True:
            result = self.env.reset(seed=seed, options=options)
            if seed is not None and hasattr(self.opponent, "seed"):
                # Not the env's seed itself: the same seed gives the same stream, and the
                # opponent's first choice would follow the env's first draw (the colours).
                if isinstance(seed, np.random.Generator):
                    self.opponent.seed(seed)
                else:
                    self.opponent.seed(make_seed_sequence(int(seed), 1)[0])
            seed = None  # a game dealt again continues the generators' streams
            self.agent_color = self._resolve()
            observation, info = result.observation, dict(result.info)
            if bool(getattr(self.env.unwrapped, "done", False)):
                break  # over before anyone moved: nothing to deal again for
            reply = None
            reward = 0.0
            while info["to_play"] != self.agent_color:
                reply = self._opponent_reply(observation, info)
                observation, info = reply.observation, reply.info
                reward -= float(reply.reward)
                if reply.done:
                    break
            if reply is None or not reply.done:
                break
            skipped.append(reward)
            if len(skipped) >= MAX_REDEALS:
                raise RuntimeError(
                    f"The opponent ended {MAX_REDEALS} games in a row before the agent's "
                    "first move."
                )
        info["agent_color"] = self.agent_color
        if skipped:
            info["skipped_rewards"] = skipped
        return ResetResult(observation=observation, info=info)

    def step(self, action: ActType) -> StepResult[ObsType]:
        result = self.env.step(action)
        reward = float(result.reward)
        reply = result
        while not reply.done and reply.info["to_play"] != self.agent_color:
            reply = self._opponent_reply(reply.observation, reply.info)
            reward -= float(reply.reward)
        observation, info = reply.observation, {**reply.info, "agent_color": self.agent_color}
        seat_view = getattr(self.env.unwrapped, "_seat_view", None)
        if reply.done and seat_view is not None:
            # The game ended or a TimeLimit stopped it, and the env's last observation is a
            # seat's private view (the last actor's, or the next one's): give the agent its own.
            observation, mine = seat_view(self.agent_color)
            info.update(mine)
        return StepResult(
            observation=observation,
            reward=reward,
            terminated=reply.terminated,
            truncated=reply.truncated,
            info=info,
        )

    def _opponent_reply(self, observation: ObsType, info: dict[str, Any]) -> StepResult[ObsType]:
        opponent_action = int(self.opponent.act(observation, info))
        reply = self.env.step(cast(ActType, opponent_action))  # the games' actions are ints
        if "illegal_action" in reply.info:  # a forfeit: no move was made
            move = None
        else:
            move = reply.info.get("last_move", reply.info.get("last_action"))
        return StepResult(
            observation=reply.observation,
            reward=reply.reward,
            terminated=reply.terminated,
            truncated=reply.truncated,
            info={**reply.info, "opponent_action": opponent_action, "opponent_move": move},
        )

    def __str__(self) -> str:
        spec = f"opponent={self.opponent!r}, agent_color={self._spec!r}"
        return f"<OpponentWrapper({spec}){self.env}>"


__all__ = ["OpponentWrapper"]
