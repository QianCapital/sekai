"""Heads-up no-limit hold'em as a sekai environment: one hand per episode.

`HoldemEnv` is a turn-based two-player game behind the single-agent `sekai.Env`
interface, like the other two-player games: the caller plays both seats (self-play), or
`OpponentWrapper` plays one of them.

- The observation is always for **the seat to act** (see `observations`; it is built from
  that seat's `SeatView` only, so it never contains the opponent's cards). On the
  terminal step it is for the seat that just acted.
- Actions are the seven-action abstraction of `actions`: `0` fold, `1` check/call, `2`
  min-raise, `3` half pot, `4` pot, `5` two pots, `6` all-in. Illegal actions raise
  `IllegalAction` (or, with `illegal_action="lose"`, fold the hand, even where checking
  was free); `info["legal_action_mask"]` (or `action_mask()`) says which are legal.
- `reward` is 0 until the hand ends; on the terminal step it is the net chips won by the
  seat that just acted, in big blinds. `info["payoffs"]` holds both seats' net in big
  blinds (zeros until the end), indexed by seat.

`info` keys: `legal_action_mask` (int8, 7), `legal_actions` (int64), `to_play` (seat to
act, `None` once the hand is over), `observer` (whose perspective the observation is),
`street`, `payoffs`, `view` (the observer's `SeatView`, for policies that want the
structured state), `last_action` (display text or `None`) and `result` (`HandResult` or
`None`); after an illegal action that folded the hand, also `illegal_action`.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from sekai._types import SeedType
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.envs.games.poker.actions import ACTION_NAMES, NUM_ABSTRACT_ACTIONS, abstract_actions
from sekai.envs.games.poker.errors import HandComplete, IllegalAction
from sekai.envs.games.poker.holdem import Action, HoldemHand, SeatView, format_action
from sekai.envs.games.poker.observations import encode_observation, observation_space
from sekai.error import InvalidAction, ResetRequired
from sekai.spaces.discrete import Discrete

ENV_ID = "HeadsUpHoldem-v0"

Observation = NDArray[np.float32]
IllegalActionMode = Literal["raise", "lose"]


class HoldemEnv(Env[Observation, np.intp]):
    """One hand of heads-up no-limit Texas hold'em per episode.

    Args:
        small_blind: The small blind in chips.
        big_blind: The big blind in chips.
        stack: Both seats' starting stack (`reset(options={"stacks": ...})` overrides it).
        history_length: The number of most recent actions appended to the observation (6
            slots each); `0` keeps the compact 125-slot observation.
        illegal_action: `"raise"` (default) raises `IllegalAction` on an illegal action;
            `"lose"` folds the hand for the seat instead (it loses what it has put in),
            even where checking was free.

    `reset` options: `button` (0 or 1; drawn from the env's generator by default),
    `stacks` (two ints, each larger than the big blind) and `deck` (a fixed card order,
    see `HoldemHand`).

    Raises:
        ValueError: Unless `0 < small_blind <= big_blind < stack` and
            `history_length >= 0`.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_fps": 2}
    #: The seats' names, in seat order (as `OpponentWrapper`'s `agent_color` takes them).
    sides: ClassVar[tuple[str, str]] = ("p1", "p2")
    #: The seats as `info["to_play"]` gives them, in the same order.
    players: ClassVar[tuple[int, int]] = (0, 1)

    def __init__(
        self,
        small_blind: int = 50,
        big_blind: int = 100,
        stack: int = 10000,
        history_length: int = 0,
        illegal_action: IllegalActionMode = "raise",
    ) -> None:
        if not 0 < small_blind <= big_blind < stack:
            raise ValueError("Need 0 < small_blind <= big_blind < stack.")
        if history_length < 0:
            raise ValueError("history_length must be >= 0.")
        if illegal_action not in ("raise", "lose"):
            raise ValueError(f"illegal_action must be 'raise' or 'lose', got {illegal_action!r}.")
        self.illegal_action: IllegalActionMode = illegal_action
        self.small_blind = int(small_blind)
        self.big_blind = int(big_blind)
        self.stack = int(stack)
        self.history_length = int(history_length)
        self.observation_space = observation_space(self.history_length)
        self.action_space = Discrete(NUM_ABSTRACT_ACTIONS)
        self._hand: HoldemHand | None = None
        self._last_actor: int = 0
        # The abstract actions of a hand, and the hand and the length of its log they were
        # computed at: every change to a hand logs an event.
        self._actions: list[Action | None] = []
        self._actions_of: tuple[HoldemHand | None, int] = (None, -1)

    # ------------------------------------------------------------------
    # sekai.Env API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[Observation]:
        self._set_rng(seed)
        opts = options or {}
        button = opts.get("button")
        if button is None:
            button = int(self.rng.integers(2))
        stacks = tuple(opts.get("stacks", (self.stack, self.stack)))
        if len(stacks) != 2 or min(stacks) <= self.big_blind:
            raise ValueError(f"stacks must be two amounts above the big blind, got {stacks!r}.")
        self._hand = HoldemHand(
            stacks=stacks,
            button=int(button),
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            rng=self.rng,
            deck=opts.get("deck"),
        )
        assert self._hand.to_act is not None
        self._last_actor = self._hand.to_act
        view = self._hand.view(self._observer())
        return ResetResult(observation=self._encode(view), info=self._info(view, None, [0.0, 0.0]))

    def step(self, action: int | np.integer[Any]) -> StepResult[Observation]:
        hand = self._hand
        if hand is None:
            raise ResetRequired()
        if hand.is_complete:
            raise HandComplete("The hand is over; call reset() to deal the next one.")
        if not self.action_space.contains(action):
            raise InvalidAction(action, self.action_space)
        index = int(action)
        actions = self._abstract_actions()
        concrete = actions[index]
        if concrete is None and self.illegal_action == "raise":
            legal = ", ".join(ACTION_NAMES[i] for i, a in enumerate(actions) if a is not None)
            raise IllegalAction(
                f"Action {index} ({ACTION_NAMES[index]}) is not legal here; legal: {legal}. "
                "Mask your policy with info['legal_action_mask'], or construct the env with "
                "illegal_action='lose' to fold the hand instead of raising.",
                action=index,
                legal=hand.legal(),
            )
        actor = hand.to_act
        assert actor is not None
        if concrete is None:  # illegal_action="lose": the seat gives the hand up
            text = format_action(Action.fold(), hand)
            hand.forfeit()
        else:
            text = format_action(concrete, hand)
            hand.act(concrete)
        self._last_actor = actor

        payoffs = self._payoffs()
        complete = hand.is_complete
        view = hand.view(self._observer())
        info = self._info(view, text, payoffs)
        if concrete is None:
            info["illegal_action"] = index
        return StepResult(
            observation=self._encode(view),
            reward=payoffs[actor] if complete else 0.0,
            terminated=complete,
            truncated=False,
            info=info,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def hand(self) -> HoldemHand:
        """The underlying hand (trusted: it knows both hands and the deck)."""
        if self._hand is None:
            raise ResetRequired()
        return self._hand

    @property
    def to_play(self) -> int | None:
        """The seat to act, `None` once the hand is over."""
        return self.hand.to_act

    @property
    def done(self) -> bool:
        """Whether the hand is over."""
        return self._hand is not None and self._hand.is_complete

    def legal_action_mask(self) -> NDArray[np.int8]:
        """The `int8` mask of the legal abstract actions (all zeros once the hand is over)."""
        mask: NDArray[np.int8] = _masks(self._abstract_actions())[0].copy()
        return mask

    def action_mask(self) -> NDArray[np.int8]:
        """`legal_action_mask()`, under the name sekai's action-mask convention uses."""
        return self.legal_action_mask()

    def legal_actions(self) -> NDArray[np.int64]:
        """The indices of the legal abstract actions."""
        legal: NDArray[np.int64] = _masks(self._abstract_actions())[1].copy()
        return legal

    def observe(self, seat: int | None = None) -> Observation:
        """The observation for `seat` (default: the seat to act, or the last actor at the end)."""
        return self._encode(self.hand.view(self._observer() if seat is None else seat))

    def _observer(self) -> int:
        to_act = self.hand.to_act
        return self._last_actor if to_act is None else to_act

    def _seat_view(self, seat: int) -> tuple[Observation, dict[str, Any]]:
        """The observation for `seat` and the `info` entries that depend on the viewer:
        what `OpponentWrapper` gives the agent when the opponent's action ends the hand
        (the env's own last observation is the last actor's)."""
        view = self.hand.view(seat)
        return self._encode(view), self._viewer_info(view)

    def _seat_info(self, seat: int) -> dict[str, Any]:
        """The `info` entries that depend on the viewer, as `seat` sees them."""
        return self._viewer_info(self.hand.view(seat))

    @staticmethod
    def _viewer_info(view: SeatView) -> dict[str, Any]:
        return {"observer": view.seat, "view": view, "result": view.result}

    def _abstract_actions(self) -> list[Action | None]:
        """`abstract_actions(self.hand)`, computed once per state of the hand."""
        hand = self.hand
        cached, length = self._actions_of
        if cached is not hand or length != len(hand._events):
            self._actions = abstract_actions(hand)
            self._actions_of = (hand, len(hand._events))
        return self._actions

    def _encode(self, view: SeatView) -> Observation:
        return encode_observation(view, self.history_length)

    def _payoffs(self) -> list[float]:
        result = self.hand._result
        if result is None:
            return [0.0, 0.0]
        return [d / self.big_blind for d in result.deltas]

    def _info(
        self, view: SeatView, last_action: str | None, payoffs: list[float]
    ) -> dict[str, Any]:
        hand = self.hand
        mask, legal = _masks(self._abstract_actions())
        return {
            "legal_action_mask": mask.copy(),
            "legal_actions": legal.copy(),
            "to_play": hand.to_act,
            "observer": view.seat,
            "street": hand.street,
            "payoffs": payoffs,
            "view": view,
            "last_action": last_action,
            "result": view.result,
        }

    def __str__(self) -> str:
        return f"<{ENV_ID} blinds={self.small_blind}/{self.big_blind} stack={self.stack}>"


_MASKS: dict[tuple[bool, ...], tuple[NDArray[np.int8], NDArray[np.int64]]] = {}


def _masks(actions: list[Action | None]) -> tuple[NDArray[np.int8], NDArray[np.int64]]:
    """The legal-action mask and indices of `actions` (shared: copy them to hand them out)."""
    key = tuple(a is not None for a in actions)
    masks = _MASKS.get(key)
    if masks is None:
        mask = np.array(key, dtype=np.int8)
        legal = np.flatnonzero(mask).astype(np.int64)
        mask.flags.writeable = False
        legal.flags.writeable = False
        masks = _MASKS[key] = (mask, legal)
    return masks


__all__ = ["HoldemEnv"]
