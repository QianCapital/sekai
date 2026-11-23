"""The blackjack environment: one seat against the dealer, one round per episode.

`BlackjackEnv` seats a single agent at a `BlackjackTable`. **One episode is one round**:
the agent bets, plays its hands, and the episode terminates when the round is settled.

- **The shoe persists across resets**, reshuffling by penetration as at a real table, so
  the count carries from one episode to the next and counting can be learned.
  `reset(seed=...)` (or `options={"new_shoe": True}`) starts a fresh shoe. A round still
  in play when `reset()` is called (a `TimeLimit` stopped the episode) is first finished,
  standing on its remaining hands (its result, in no episode's rewards, is in
  `env.table.last_round`).
- The **bankroll resets** to `bankroll` every episode unless
  `options={"keep_bankroll": True}` (then it carries over; if it no longer covers the
  smallest bet level it is restored and `info["rebuy"]` is `True`), or
  `options={"bankroll": amount}` sets it.
- Actions: `0` stand, `1` hit, `2` double, `3` split, `4` surrender, `5 + k` bet
  `bet_levels[k] * min_bet` (see `sekai.envs.games.blackjack.actions`).
- Observations: a 43-slot `float32` vector of public information only (see
  `sekai.envs.games.blackjack.observations`).
- `reward` is `0` until the round settles, then the round's net chips / `min_bet` (a
  10-chip natural at a 10-chip minimum is `+1.5`).
- `info`: `legal_action_mask` (`int8`; all zeros once the round is over; also
  `env.action_mask()`), `legal_actions` (`int64`), `to_play` (always `0`), `phase`,
  `view` (the public table view), `round_no`, `bankroll`, `rebuy` after a reset, and once
  settled `net` (chips) and `round` (the round's summary).

An illegal action raises `IllegalAction`; stepping a finished round raises `RoundOver`.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray

from sekai._types import SeedType
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.envs.games.blackjack.actions import (
    BET_OFFSET,
    DEFAULT_BET_LEVELS,
    NUM_PLAY_ACTIONS,
    PLAY_ACTION_NAMES,
    action_to_move,
    legal_action_mask,
    num_actions,
    validate_bet_levels,
)
from sekai.envs.games.blackjack.errors import IllegalAction, RoundOver
from sekai.envs.games.blackjack.observations import Observation, encode_view, observation_space
from sekai.envs.games.blackjack.rules import Rules
from sekai.envs.games.blackjack.table import BlackjackTable
from sekai.error import InvalidAction, ResetRequired
from sekai.spaces.discrete import Discrete

ENV_ID = "Blackjack-v0"


@functools.lru_cache(maxsize=512)
def _mask_template(
    legal: tuple[Any, ...], min_bet: int, bet_levels: tuple[int, ...]
) -> tuple[NDArray[np.int8], NDArray[np.int64]]:
    """The mask and the legal indices for a table's legal actions, made once (read-only).

    `legal` is `legal_actions()` as a tuple: its moves, or `("bet", min, max, step)`.
    """
    moves: dict[str, Any] | list[str] = (
        {"min": legal[1], "max": legal[2], "step": legal[3]}
        if legal[:1] == ("bet",)
        else list(legal)
    )
    mask = legal_action_mask(moves, min_bet=min_bet, bet_levels=bet_levels)
    indices = np.flatnonzero(mask).astype(np.int64)
    mask.flags.writeable = False
    indices.flags.writeable = False
    return mask, indices


class BlackjackEnv(Env[Observation, np.intp]):
    """Single-seat blackjack against the dealer, one round per episode.

    See the module docstring for the episode, the reset options, rewards and `info`.

    Args:
        rules: Table rules.
        bankroll: The seat's bankroll at the start of every episode.
        bet_levels: Bet sizes in units of `rules.min_bet`, one action each.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_fps": 2}

    def __init__(
        self,
        rules: Rules = Rules(),  # noqa: B008 - a frozen dataclass
        bankroll: float = 1000,
        bet_levels: Sequence[int] = DEFAULT_BET_LEVELS,
    ) -> None:
        self.rules = rules
        self.bet_levels = validate_bet_levels(bet_levels)
        if self.bet_levels[0] * rules.min_bet > rules.max_bet:
            raise ValueError(
                f"The smallest bet level ({self.bet_levels[0]} x {rules.min_bet}) is above the "
                f"table maximum ({rules.max_bet}), so no bet could be placed."
            )
        if bankroll < self.bet_levels[0] * rules.min_bet:
            raise ValueError(
                f"bankroll {bankroll} does not cover the smallest bet level "
                f"({self.bet_levels[0] * rules.min_bet})."
            )
        self.bankroll = float(bankroll)
        self.observation_space = observation_space()
        self.action_space = Discrete(num_actions(self.bet_levels))
        self.table: BlackjackTable | None = None
        self._done = False
        self._start_bankroll = self.bankroll

    # ------------------------------------------------------------------
    # sekai.Env API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[Observation]:
        """Start the next round (a new shoe with a seed or `new_shoe`); see the module docs."""
        self._set_rng(seed)
        opts = options or {}
        if self.table is not None:
            # A round still in play (a TimeLimit stopped the episode, say) is finished first,
            # standing on its remaining hands, so the shoe and the count go on as at a table.
            while self.table.phase == "playing":
                self.table.act("stand")
        carried = self.table.bankrolls[0] if self.table is not None else self.bankroll
        if self.table is None or seed is not None or opts.get("new_shoe"):
            self.table = BlackjackTable(
                self.rules,
                seats=1,
                bankroll=self.bankroll,
                rng=self.rng,
                max_events=1000,
                max_history=100,
            )
        target = self.bankroll
        if "bankroll" in opts:
            target = float(opts["bankroll"])
        elif opts.get("keep_bankroll"):
            target = float(carried)
        rebuy = False
        if target < self.bet_levels[0] * self.rules.min_bet:
            target, rebuy = self.bankroll, True
        if self.table.bankrolls[0] != target:
            self.table.set_bankroll(0, target)
        self._start_bankroll = float(self.table.bankrolls[0])
        self._done = False
        view = self.table.view(0)
        info = self._info(view)
        info["rebuy"] = rebuy
        return ResetResult(observation=encode_view(view, 0, self.bankroll), info=info)

    def step(self, action: int | np.integer[Any]) -> StepResult[Observation]:
        """Bet (in the betting phase) or play the active hand."""
        table = self.table
        if table is None:
            raise ResetRequired()
        if self._done:
            raise RoundOver(table.rounds_played)
        if not self.action_space.contains(action):
            raise InvalidAction(action, self.action_space)
        a = int(action)
        if not self._is_legal(table, a):
            raise IllegalAction(a, self._illegal_reason(a), self.legal_actions().tolist())
        played = table.rounds_played
        table.act(action_to_move(a, min_bet=self.rules.min_bet, bet_levels=self.bet_levels))
        self._done = table.rounds_played > played
        reward = 0.0
        info_extra: dict[str, Any] = {}
        if self._done:
            last = table.last_round
            assert last is not None
            net = float(last["seats"][0]["net"])
            reward = net / self.rules.min_bet
            info_extra = {"net": last["seats"][0]["net"], "round": last}
        view = table.view(0)
        info = self._info(view)
        info.update(info_extra)
        return StepResult(
            observation=encode_view(view, 0, self.bankroll),
            reward=reward,
            terminated=self._done,
            truncated=False,
            info=info,
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def done(self) -> bool:
        """Whether the episode's round has been settled."""
        return self._done

    def legal_action_mask(self) -> NDArray[np.int8]:
        """`int8` mask of the legal actions (all zeros before a reset and once settled)."""
        if self.table is None or self._done:
            return np.zeros(num_actions(self.bet_levels), dtype=np.int8)
        return legal_action_mask(
            self.table.legal_actions(), min_bet=self.rules.min_bet, bet_levels=self.bet_levels
        )

    def action_mask(self) -> NDArray[np.int8]:
        """The legal actions, as sekai's action-mask convention has it: `legal_action_mask()`."""
        return self.legal_action_mask()

    def legal_actions(self) -> NDArray[np.int64]:
        """Indices of the legal actions (`int64`)."""
        return np.flatnonzero(self.legal_action_mask()).astype(np.int64)

    def view(self) -> dict[str, Any]:
        """The public table view for the agent's seat."""
        if self.table is None:
            raise ResetRequired()
        return self.table.view(0)

    def observe(self) -> Observation:
        """The observation of the current state."""
        if self.table is None:
            raise ResetRequired()
        return encode_view(self.table.view(0), 0, self.bankroll)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_legal(self, table: BlackjackTable, a: int) -> bool:
        """`legal_action_mask()[a]`, for an action in the space, without building the mask."""
        legal = table.legal_actions()
        if isinstance(legal, dict):
            if a < BET_OFFSET:
                return False
            amount = self.bet_levels[a - BET_OFFSET] * self.rules.min_bet
            return bool(legal["min"] <= amount <= legal["max"] and amount % legal["step"] == 0)
        return a < NUM_PLAY_ACTIONS and PLAY_ACTION_NAMES[a] in legal

    def _info(self, view: dict[str, Any]) -> dict[str, Any]:
        table = self.table
        assert table is not None
        # The view's legal actions are the table's: nothing has changed since.
        legal = view["legal_actions"]
        if self._done:
            key: tuple[Any, ...] = ()
        elif isinstance(legal, dict):
            key = ("bet", legal["min"], legal["max"], legal["step"])
        else:
            key = tuple(legal)
        mask, indices = _mask_template(key, self.rules.min_bet, tuple(self.bet_levels))
        return {
            "legal_action_mask": mask.copy(),
            "legal_actions": indices.copy(),
            "to_play": 0,
            "phase": table.phase,
            "view": view,
            "round_no": table.round_no,
            "bankroll": table.bankrolls[0],
        }

    def _illegal_reason(self, a: int) -> str:
        assert self.table is not None
        if self.table.phase == "betting":
            if a < len(PLAY_ACTION_NAMES):
                return f"{PLAY_ACTION_NAMES[a]} is not possible before the bet is placed"
            return "that bet level is outside the table limits or the bankroll"
        if a >= len(PLAY_ACTION_NAMES):
            return "bets are only taken before the deal"
        return f"{PLAY_ACTION_NAMES[a]} is not allowed on this hand"

    def __str__(self) -> str:
        return f"<{ENV_ID} {self.rules.label} bet_levels={self.bet_levels}>"


__all__ = ["BlackjackEnv", "ENV_ID"]
