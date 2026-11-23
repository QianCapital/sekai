"""The Xiangqi environment.

`XiangqiEnv` is a *turn-based two-player* environment exposed through the single-agent
`sekai.Env` interface, which is exactly what AlphaZero-style self-play needs:

- The acting agent is always **the side to move**. Observations are oriented to that side
  (see `sekai.envs.games.xiangqi.observations`) and actions are encoded relative to it
  (see `sekai.envs.games.xiangqi.actions`), so one network plays both colours.
- `reward` is from the perspective of **the player who just moved**: `+1` if that move
  won the game, `-1` if it lost it (completing a perpetual-check cycle, or an illegal move
  with `illegal_action="lose"`), `0` otherwise, including every non-terminal step.
- `info["legal_action_mask"]` (and `env.action_mask()`) is the `int8` legal-move mask over
  all 8100 actions. Policies must respect it; by default an illegal action raises
  `IllegalMove`.

## Game end

Checked after every move, in this order:

1. `CHECKMATE` / `STALEMATE`: the side to move has no legal move and **loses** (in
   check or not; stalemate is a loss in Xiangqi).
2. `PERPETUAL_CHECK` / `REPETITION`: the position (placement + side to move) occurs for
   the third time. Over the moves since its first occurrence, if one side gave check with
   every one of its moves and the other side did not, the checking side loses; otherwise
   the game is drawn. This deliberately simplifies the Asian rules (WXF/CXA): chasing
   (perpetual attack on an unprotected piece) is not penalised, and the verdict is not
   refined by "idle" or mixed check/chase cycles.
3. `NO_CAPTURE`: 120 plies without a capture (the FEN half-move clock) is a draw.
4. `INSUFFICIENT_MATERIAL`: neither side has a chariot, horse, cannon or soldier: draw.

Episodes are bounded by the registry (`max_episode_steps=400` truncates the episode).
"""

from __future__ import annotations

import copy
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from sekai._types import SeedType
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.envs.games.xiangqi.actions import (
    NUM_ACTIONS,
    action_to_move,
    move_to_action,
    sorted_actions,
)
from sekai.envs.games.xiangqi.engine import (
    NO_CAPTURE_LIMIT,
    RED,
    Color,
    Position,
    color_name,
    move_from_uci,
    move_to_uci,
)
from sekai.envs.games.xiangqi.errors import GameOver, IllegalMove
from sekai.envs.games.xiangqi.notation import to_chinese, to_wxf
from sekai.envs.games.xiangqi.observations import (
    DEFAULT_HISTORY_LENGTH,
    Frames,
    ObservationEncoder,
    advance_frames,
    position_frames,
)
from sekai.error import InvalidAction, ResetRequired
from sekai.spaces.discrete import Discrete

ENV_ID = "Xiangqi-v0"

IllegalActionMode = Literal["raise", "lose"]
Observation = NDArray[np.float32]

TERMINATIONS = (
    "CHECKMATE",
    "STALEMATE",
    "PERPETUAL_CHECK",
    "REPETITION",
    "NO_CAPTURE",
    "INSUFFICIENT_MATERIAL",
    "ILLEGAL_MOVE",
)

_LAST_ACTION = NUM_ACTIONS - 1  # Black's action of a move is _LAST_ACTION - move


@dataclass(frozen=True, slots=True)
class XiangqiResult:
    """How a finished game ended."""

    winner: Color | None
    """`"red"`, `"black"` or `None` for a draw."""

    termination: str
    """One of `TERMINATIONS`, e.g. `"CHECKMATE"` or `"PERPETUAL_CHECK"`."""

    def result(self) -> str:
        """Result string: `"1-0"` (Red won), `"0-1"` (Black won) or `"1/2-1/2"`."""
        if self.winner is None:
            return "1/2-1/2"
        return "1-0" if self.winner == "red" else "0-1"


def game_outcome(position: Position, *, has_legal_move: bool | None = None) -> XiangqiResult | None:
    """Apply the game-ending rules (see the module docstring) to `position`; `None` if ongoing."""
    if has_legal_move is None:
        has_legal_move = position.has_legal_move()
    if not has_legal_move:
        termination = "CHECKMATE" if position.checked else "STALEMATE"
        return XiangqiResult(winner=color_name(-position.side), termination=termination)
    start = position.repetition_start(3)
    if start >= 0:
        loser = position.perpetual_checker(start)
        if loser:
            return XiangqiResult(winner=color_name(-loser), termination="PERPETUAL_CHECK")
        return XiangqiResult(winner=None, termination="REPETITION")
    if position.halfmove >= NO_CAPTURE_LIMIT:
        return XiangqiResult(winner=None, termination="NO_CAPTURE")
    if position.insufficient_material():
        return XiangqiResult(winner=None, termination="INSUFFICIENT_MATERIAL")
    return None


def _frozen(array: NDArray[Any]) -> NDArray[Any]:
    array.flags.writeable = False
    return array


class XiangqiEnv(Env[Observation, np.intp]):
    """Xiangqi (Chinese chess): two-player self-play from the side to move's point of view.

    - **Observation**: `Box(0, 1, (16 * history_length + 3, 10, 9), float32)`, board
      planes in the mover's frame (see `sekai.envs.games.xiangqi.observations`).
    - **Action**: `Discrete(8100)`, `from * 90 + to` in the mover's frame (see
      `sekai.envs.games.xiangqi.actions`); `encode_move("h2e2")` / `decode_action()`
      convert UCCI moves.
    - **Reward**: for the player who just moved: `+1` for a winning move, `-1` for a
      losing one, else `0`.
    - **info**: `legal_action_mask` (int8, 8100), `legal_actions` (sorted int64),
      `to_play` (`"red"`/`"black"`), `fen`, `last_move` (UCCI or `None`), `ply`,
      `is_check`, `result` (`"*"`, `"1-0"`, `"0-1"`, `"1/2-1/2"`), `winner` and
      `termination` (one of `TERMINATIONS`, or `None` while the game goes on); after a
      forfeit, also `illegal_action`.

    Args:
        history_length: number of past positions stacked into the observation. `8` gives
            131 planes; `1` gives 19.
        starting_fen: position to start every episode from (UCCI FEN). `None` is the
            standard initial position. `reset(options={"fen": ...})` overrides it.
        illegal_action: `"raise"` (default) raises `IllegalMove` on an illegal
            action; `"lose"` ends the game as a loss for the mover (reward `-1`,
            termination `"ILLEGAL_MOVE"`) instead.

    `reset` options: `fen` (starting position) and `moves` (a list of UCCI moves to
    replay from it, e.g. to restore a game; the observation history and repetition counts
    are rebuilt exactly as if the moves had been stepped). A position that is already
    over is returned as it is, with no legal moves: check `env.done` (or
    `info["termination"]`) before stepping.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_fps": 2}
    sides: ClassVar[tuple[str, str]] = ("red", "black")
    """The sides, Red first (Red moves first)."""
    players: ClassVar[tuple[str, str]] = ("red", "black")
    """The values of `info["to_play"]`, in the order of `sides`."""

    def __init__(
        self,
        history_length: int = DEFAULT_HISTORY_LENGTH,
        starting_fen: str | None = None,
        illegal_action: IllegalActionMode = "raise",
    ) -> None:
        if illegal_action not in ("raise", "lose"):
            raise ValueError(f"illegal_action must be 'raise' or 'lose', got {illegal_action!r}.")
        self.history_length = history_length
        self.starting_fen = starting_fen
        self.illegal_action: IllegalActionMode = illegal_action

        self.encoder = ObservationEncoder(history_length)
        self.observation_space = self.encoder.space()
        self.action_space = Discrete(NUM_ACTIONS)

        self._history: deque[Frames] = deque(maxlen=history_length)
        self._position = Position(starting_fen)
        self._initial_fen = self._position.fen()
        self._moves: list[str] = []
        # The legal moves of the current position (computed once per position, read-only).
        self._legal: tuple[int, ...] = ()
        self._actions: NDArray[np.int64] = np.zeros(0, dtype=np.int64)
        self._mask: NDArray[np.int8] = np.zeros(NUM_ACTIONS, dtype=np.int8)
        self._outcome: XiangqiResult | None = None
        self._ply = 0
        self._has_reset = False
        self._start(self._position, ())

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
        options = options or {}
        fen = options.get("fen")
        moves = options.get("moves")
        if moves is None:  # (not `or ()`: an array of actions has no truth value)
            moves = ()
        elif isinstance(moves, str):
            moves = moves.split()
        saved = dict(self.__dict__)
        try:
            self._start(Position(self.starting_fen if fen is None else fen), moves)
        except Exception:
            # A bad FEN or replay leaves the environment exactly as it was.
            self.__dict__.clear()
            self.__dict__.update(saved)
            raise
        self._has_reset = True
        return ResetResult(observation=self._observe(), info=self._info())

    def step(self, action: int | np.integer[Any]) -> StepResult[Observation]:
        if not self._has_reset:
            raise ResetRequired()
        if self._outcome is not None:
            raise GameOver(self._outcome.result(), self._position.fen())
        if not self.action_space.contains(action):
            raise InvalidAction(action, self.action_space)

        index = int(action)
        mover = self._position.side
        move = index if mover == RED else _LAST_ACTION - index

        if not self._mask[index]:
            if self.illegal_action == "raise":
                raise IllegalMove(move_to_uci(move), self._position.fen(), action=index)
            self._outcome = XiangqiResult(winner=color_name(-mover), termination="ILLEGAL_MOVE")
            info = self._info()
            info["illegal_action"] = index
            return StepResult(self._observe(), -1.0, True, False, info)

        self._play(move)
        reward = 0.0
        if self._outcome is not None and self._outcome.winner is not None:
            reward = 1.0 if self._outcome.winner == color_name(mover) else -1.0
        return StepResult(
            observation=self._observe(),
            reward=reward,
            terminated=self._outcome is not None,
            truncated=False,
            info=self._info(),
        )

    # ------------------------------------------------------------------
    # Public helpers (for agents, search, tests and apps)
    # ------------------------------------------------------------------

    @property
    def position(self) -> Position:
        """The live engine position. Treat it as read-only; `position.copy()` to search."""
        return self._position

    @property
    def turn(self) -> Color:
        """Side to move: `"red"` or `"black"`."""
        return self._position.turn

    @property
    def fen(self) -> str:
        return self._position.fen()

    @property
    def initial_fen(self) -> str:
        """FEN of the episode's starting position (before any `moves` replay)."""
        return self._initial_fen

    @property
    def moves(self) -> list[str]:
        """UCCI moves played since the episode's starting position (replayed moves included)."""
        return list(self._moves)

    @property
    def last_move(self) -> str | None:
        return self._moves[-1] if self._moves else None

    @property
    def ply(self) -> int:
        """Half-moves played since `reset` (replayed `moves` included)."""
        return self._ply

    @property
    def outcome(self) -> XiangqiResult | None:
        """`None` while the game is in progress."""
        return self._outcome

    @property
    def done(self) -> bool:
        return self._outcome is not None

    def is_check(self) -> bool:
        """Whether the side to move is in check (facing generals count as check)."""
        return self._position.checked

    def legal_moves(self) -> list[str]:
        """Sorted UCCI strings of all legal moves; empty once the game is over."""
        if self._outcome is not None:
            return []
        return sorted(move_to_uci(m) for m in self._legal)

    def is_legal(self, uci: str) -> bool:
        """Whether `uci` is a well-formed, legal move here (`False` once the game is over)."""
        if self._outcome is not None:
            return False
        try:
            move = move_from_uci(uci)
        except ValueError:
            return False
        return bool(self._mask[self._action_of(move)])

    def encode_move(self, uci: str) -> int:
        """Action index of the UCCI move `uci` in the current position's frame (not checked)."""
        return move_to_action(move_from_uci(uci), self._position.side)

    def decode_action(self, action: int) -> str:
        """UCCI string of `action` in the current position's frame (no legality check)."""
        return move_to_uci(action_to_move(action, self._position.side))

    def legal_action_mask(self) -> NDArray[np.int8]:
        """`int8` mask over all 8100 actions; all zeros once the game is over."""
        if self._outcome is not None:
            return np.zeros(NUM_ACTIONS, dtype=np.int8)
        mask: NDArray[np.int8] = self._mask.copy()
        return mask

    def action_mask(self) -> NDArray[np.int8]:
        """The legal-action mask (sekai's action-mask convention): `legal_action_mask()`."""
        return self.legal_action_mask()

    def legal_actions(self) -> NDArray[np.int64]:
        """Sorted action indices of the legal moves; empty once the game is over."""
        if self._outcome is not None:
            return np.zeros(0, dtype=np.int64)
        actions: NDArray[np.int64] = self._actions.copy()
        return actions

    def observe(self) -> Observation:
        """Observation of the current position, as last returned by `reset`/`step`."""
        return self._observe()

    def move_wxf(self, uci: str) -> str:
        """WXF notation (e.g. `"C2=5"`) of a legal move in the current position."""
        return to_wxf(self._position, uci)

    def move_chinese(self, uci: str) -> str:
        """Chinese notation (e.g. `"炮二平五"`) of a legal move in the current position."""
        return to_chinese(self._position, uci)

    def clone(self) -> XiangqiEnv:
        """Independent copy of the environment, including history and repetition state.

        Intended for tree search: step the clone freely without disturbing the original.
        Costs a few microseconds (the position, a short move list and a deque of references
        to immutable planes are copied). The clone gets its own lazily-seeded RNG.
        """
        new = copy.copy(self)
        new._position = self._position.copy()
        new._history = deque(self._history, maxlen=self.history_length)
        new._moves = list(self._moves)
        new._rng = None
        return new

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _action_of(self, move: int) -> int:
        return move if self._position.side == RED else _LAST_ACTION - move

    def _start(self, position: Position, moves: Sequence[str]) -> None:
        # Rebind (never mutate) the per-episode containers so that reset() can roll back.
        self._position = position
        self._initial_fen = position.fen()
        self._history = deque(maxlen=self.history_length)
        self._moves = []
        self._ply = 0
        self._record_position()
        self._refresh()
        for i, uci in enumerate(moves):
            if self._outcome is not None:
                raise GameOver(self._outcome.result(), position.fen())
            move = move_from_uci(uci)
            if not self._mask[self._action_of(move)]:
                raise IllegalMove(
                    move_to_uci(move), position.fen(), reason=f"replaying options['moves'][{i}]"
                )
            self._play(move)

    def _play(self, move: int) -> None:
        position = self._position
        fr, to = divmod(move, 90)
        piece, captured = position.board[fr], position.board[to]
        position.push(move)
        self._moves.append(move_to_uci(move))
        self._ply += 1
        frames = advance_frames(
            self._history[-1], piece, fr, to, captured, position.repetition_count()
        )
        self._history.append(frames)
        self._refresh()

    def _refresh(self) -> None:
        """The current position's legal moves (as actions and a mask) and outcome."""
        position = self._position
        legal = position.legal_moves()
        actions = sorted_actions(legal, position.side)
        mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
        mask[actions] = 1
        self._legal = tuple(legal)
        self._actions = _frozen(actions)
        self._mask = _frozen(mask)
        self._outcome = game_outcome(position, has_legal_move=bool(legal))

    def _record_position(self) -> None:
        pos = self._position
        self._history.append(position_frames(pos, repetitions=pos.repetition_count()))

    def _observe(self) -> Observation:
        return self.encoder.encode_frames(self._position, reversed(self._history))

    def _info(self) -> dict[str, Any]:
        position = self._position
        outcome = self._outcome
        if outcome is None:
            mask, legal = self._mask.copy(), self._actions.copy()
        else:
            mask, legal = np.zeros(NUM_ACTIONS, dtype=np.int8), np.zeros(0, dtype=np.int64)
        return {
            "legal_action_mask": mask,
            "legal_actions": legal,
            "to_play": position.turn,
            "fen": position.fen(),
            "last_move": self._moves[-1] if self._moves else None,
            "ply": self._ply,
            "is_check": position.checked,
            "result": outcome.result() if outcome is not None else "*",
            "winner": outcome.winner if outcome is not None else None,
            "termination": outcome.termination if outcome is not None else None,
        }

    def __str__(self) -> str:
        return f"<{ENV_ID} history_length={self.history_length} fen='{self._position.fen()}'>"


__all__ = [
    "XiangqiEnv",
    "XiangqiResult",
    "IllegalActionMode",
    "TERMINATIONS",
    "game_outcome",
]
