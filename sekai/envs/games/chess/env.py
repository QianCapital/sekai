"""The chess environment.

`ChessEnv` is a *turn-based two-player* environment exposed through the single-agent
`sekai.Env` interface, which is exactly what AlphaZero-style self-play needs:

* The acting agent is always **the side to move**. Observations are oriented to that side
  (see `sekai.envs.games.chess.observations`) and actions are encoded relative to it (see
  `sekai.envs.games.chess.actions`), so one network plays both colours.
* `reward` is from the perspective of **the player who just moved**: `+1` for delivering
  checkmate, `-1` for losing (only possible via an illegal move with
  `illegal_action="lose"`), `0` otherwise, including every non-terminal step.
* `info["legal_action_mask"]` (also `action_mask()`) is the `int8` legal-move mask over
  all 4672 actions. By default an illegal action raises `IllegalMove`.

Games end (`terminated=True`) on checkmate, stalemate, insufficient material, the
seventy-five-move rule and fivefold repetition. With `claim_draw=True` (the default, as
in AlphaZero) threefold repetition and the fifty-move rule also end the game as a draw.
`Chess-v0` truncates games after 512 plies.

To train a single agent against a fixed opponent instead, wrap the env in
`sekai.envs.games.OpponentWrapper` (`ChessVsRandom-v0` does, with a random opponent).
"""

from __future__ import annotations

import copy
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from sekai._types import SeedType
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.envs.games.chess.actions import (
    ACTION_OF_CODE,
    NUM_ACTIONS,
    decode_action,
    encode_move,
)
from sekai.envs.games.chess.board import (
    BLACK,
    WHITE,
    Board,
    Color,
    Move,
    code_to_move,
)
from sekai.envs.games.chess.errors import GameOver, IllegalMove
from sekai.envs.games.chess.observations import (
    DEFAULT_HISTORY_LENGTH,
    ObservationEncoder,
    frames,
    position_planes,
)
from sekai.error import InvalidAction, ResetRequired
from sekai.spaces.discrete import Discrete

IllegalActionMode = Literal["raise", "lose"]
Observation = NDArray[np.float32]
Planes = NDArray[np.float32]

TERMINATIONS = (
    "CHECKMATE",
    "STALEMATE",
    "INSUFFICIENT_MATERIAL",
    "SEVENTYFIVE_MOVES",
    "FIVEFOLD_REPETITION",
    "FIFTY_MOVES",
    "THREEFOLD_REPETITION",
    "ILLEGAL_MOVE",
)
"""How a game can end (`GameResult.termination`)."""

# A position's identity for repetitions: placement, side to move, castling rights and
# the en passant square when a capture there is legal.
TranspositionKey = tuple[bytes, bool, int, int | None]


@dataclass(frozen=True, slots=True)
class GameResult:
    """How a finished game ended."""

    termination: str
    """One of `TERMINATIONS`: `"CHECKMATE"`, `"STALEMATE"`, ..., `"ILLEGAL_MOVE"`."""

    winner: Color | None
    """`True` (White), `False` (Black) or `None` for a draw."""

    def result(self) -> str:
        """PGN-style result string: `"1-0"`, `"0-1"` or `"1/2-1/2"`."""
        if self.winner is None:
            return "1/2-1/2"
        return "1-0" if self.winner == WHITE else "0-1"


class ChessEnv(Env[Observation, np.intp]):
    """Standard chess as a sekai environment.

    Args:
        history_length: number of past positions stacked into the observation. `8`
            reproduces AlphaZero's 119-plane input; `1` gives 21 planes.
        illegal_action: `"raise"` raises `IllegalMove` on an illegal action; `"lose"`
            ends the game as a loss for the mover (reward `-1`) instead.
        claim_draw: when `True` (default) threefold repetition and the fifty-move rule
            end the game as a draw automatically, as in AlphaZero. When `False` only
            forced draws apply (stalemate, insufficient material, fivefold repetition,
            seventy-five-move rule).
        starting_fen: position to start every episode from. `None` is the standard
            initial position. `reset(options={"fen": ...})` overrides it per episode.

    `info` holds `legal_action_mask` (`int8`), `legal_actions` (`int64`), `to_play`
    (`True` for White), `fen`, `last_move` (UCI or `None`), `ply`, `is_check`, `result`
    (`"*"` while running, else `"1-0"`, `"0-1"`, `"1/2-1/2"`), `winner` and `termination`.
    """

    metadata: ClassVar[dict[str, Any]] = {"name": "Chess-v0"}
    sides: ClassVar[tuple[str, str]] = ("white", "black")
    """The two sides, White first."""
    players: ClassVar[tuple[bool, bool]] = (WHITE, BLACK)
    """The values of `info["to_play"]` for the two sides."""

    def __init__(
        self,
        *,
        history_length: int = DEFAULT_HISTORY_LENGTH,
        illegal_action: IllegalActionMode = "raise",
        claim_draw: bool = True,
        starting_fen: str | None = None,
    ) -> None:
        if illegal_action not in ("raise", "lose"):
            raise ValueError(f"illegal_action must be 'raise' or 'lose', got {illegal_action!r}.")
        self.history_length = history_length
        self.illegal_action: IllegalActionMode = illegal_action
        self.claim_draw = claim_draw
        self.starting_fen = starting_fen

        self.encoder = ObservationEncoder(history_length)
        self.observation_space = self.encoder.space()
        self.action_space = Discrete(NUM_ACTIONS)

        self.board: Board = self._new_board(starting_fen)
        self._start_fen = self.board.fen()
        # Each position's planes in White's and in Black's frame, oldest first.
        self._history: deque[tuple[Planes, Planes]] = deque(maxlen=history_length)
        self._position_counts: Counter[TranspositionKey] = Counter()
        self._outcome: GameResult | None = None
        self._actions: dict[int, int] | None = None  # legal action -> move code
        self._ply = 0
        self._has_reset = False

    # ------------------------------------------------------------------
    # sekai.Env API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: SeedType = None, options: dict[str, Any] | None = None
    ) -> ResetResult[Observation]:
        self._set_rng(seed)
        fen = (options or {}).get("fen", self.starting_fen)
        self.board = self._new_board(fen)
        self._actions = None
        self._start_fen = self.board.fen()
        self._history.clear()
        self._position_counts.clear()
        self._ply = 0
        self._record_position()
        self._outcome = self._compute_outcome()
        self._has_reset = True
        return ResetResult(observation=self._observe(), info=self._info(last_move=None))

    def step(self, action: int | np.integer[Any]) -> StepResult[Observation]:
        if not self._has_reset:
            raise ResetRequired()
        if self._outcome is not None:
            raise GameOver(self._outcome.result(), self.board.fen())
        if not self.action_space.contains(action):
            raise InvalidAction(action, self.action_space)

        action = int(action)
        board = self.board
        mover = board.turn
        code = self._legal_actions_map().get(action)
        if code is None:
            move = decode_action(action, board)
            if self.illegal_action == "raise":
                raise IllegalMove(action, move, board.fen())
            self._outcome = GameResult(termination="ILLEGAL_MOVE", winner=not mover)
            info = self._info(last_move=None)
            info["illegal_action"] = action
            return StepResult(self._observe(), -1.0, True, False, info)

        move = code_to_move(code)
        board.push(move)
        self._actions = None
        self._ply += 1
        self._record_position()
        self._outcome = self._compute_outcome()

        reward = 0.0
        if self._outcome is not None and self._outcome.winner is not None:
            reward = 1.0 if self._outcome.winner == mover else -1.0

        return StepResult(
            observation=self._observe(),
            reward=reward,
            terminated=self._outcome is not None,
            truncated=False,
            info=self._info(last_move=move),
        )

    # ------------------------------------------------------------------
    # Public helpers (useful for agents, search and tests)
    # ------------------------------------------------------------------

    @property
    def turn(self) -> Color:
        """Side to move: `True` for White, `False` for Black."""
        return self.board.turn

    @property
    def fen(self) -> str:
        return self.board.fen()

    @property
    def start_fen(self) -> str:
        """FEN of the position the episode started from."""
        return self._start_fen

    @property
    def outcome(self) -> GameResult | None:
        """`None` while the game is in progress."""
        return self._outcome

    @property
    def done(self) -> bool:
        return self._outcome is not None

    @property
    def ply(self) -> int:
        """Half-moves played since `reset`."""
        return self._ply

    def legal_action_mask(self) -> NDArray[np.int8]:
        """`int8` mask over all 4672 actions; all zeros once the game is over."""
        mask = np.zeros(NUM_ACTIONS, dtype=np.int8)
        if self._outcome is None:
            actions = self._legal_actions_map()
            if actions:
                mask[list(actions)] = 1
        return mask

    def action_mask(self) -> NDArray[np.int8]:
        """The legal actions as a mask (sekai's action-mask convention); see
        `legal_action_mask`."""
        return self.legal_action_mask()

    def legal_actions(self) -> NDArray[np.int64]:
        """The legal actions, sorted; empty once the game is over."""
        if self._outcome is not None:
            return np.zeros(0, dtype=np.int64)
        return np.array(sorted(self._legal_actions_map()), dtype=np.int64)

    def observe(self) -> Observation:
        """Observation of the current position, as last returned by `reset`/`step`."""
        return self._observe()

    def encode_move(self, move: Move | str | Any) -> int:
        """Action index for `move` (a `Move`, a UCI or SAN string, or any object with a
        `uci()` method) in the current position."""
        if isinstance(move, Move):
            parsed = move
        elif isinstance(move, str):
            try:
                parsed = Move.from_uci(move)
            except ValueError:
                parsed = self.board.parse_san(move)
        else:
            parsed = Move.from_uci(move.uci())
        return encode_move(parsed, self.board.turn)

    def decode_action(self, action: int) -> Move | None:
        """Move for `action` in the current position (`None` if off-board; legality not checked)."""
        return decode_action(action, self.board)

    def clone(self, *, full_stack: bool = True) -> ChessEnv:
        """Independent copy of the environment, including move history and repetition counts.

        Intended for tree search: step the clone freely without disturbing the original.
        The clone gets its own lazily-seeded RNG.

        `full_stack=False` keeps only the moves since the last capture or pawn move in the
        clone's `board.move_stack`, which is all that repetitions need, so cloning stays
        cheap deep into a game; `ply` is unaffected.
        """
        new = copy.copy(self)
        new.board = self.board.copy(stack=True if full_stack else max(self.board.halfmove_clock, 1))
        new._history = deque(self._history, maxlen=self.history_length)
        new._position_counts = Counter(self._position_counts)
        new._rng = None
        return new

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _new_board(fen: str | None) -> Board:
        board = Board(fen) if fen else Board()
        errors = board.status_errors()
        if errors:
            raise ValueError(f"Invalid starting position {fen!r}: {', '.join(errors)}.")
        return board

    def _legal_actions_map(self) -> dict[int, int]:
        if self._actions is None:
            table = ACTION_OF_CODE[self.board.turn]
            self._actions = {table[m]: m for m in self.board.legal_move_codes()}
        return self._actions

    def _transposition_key(self) -> TranspositionKey:
        b = self.board
        return (
            bytes(b.squares),
            b.turn,
            b.castling,
            b.ep_square if b.has_legal_en_passant() else None,
        )

    def _record_position(self) -> None:
        key = self._transposition_key()
        self._position_counts[key] += 1
        self._history.append(frames(position_planes(self.board, self._position_counts[key])))

    def _compute_outcome(self) -> GameResult | None:
        board = self.board
        legal = board.legal_move_codes()
        if not legal and board.is_check():
            return GameResult(termination="CHECKMATE", winner=not board.turn)
        if board.is_insufficient_material():
            return GameResult(termination="INSUFFICIENT_MATERIAL", winner=None)
        if not legal:
            return GameResult(termination="STALEMATE", winner=None)
        if board.halfmove_clock >= 150:
            return GameResult(termination="SEVENTYFIVE_MOVES", winner=None)
        count = self._position_counts[self._transposition_key()]
        if count >= 5:
            return GameResult(termination="FIVEFOLD_REPETITION", winner=None)
        if self.claim_draw:
            if board.halfmove_clock >= 100:
                return GameResult(termination="FIFTY_MOVES", winner=None)
            if count >= 3:
                return GameResult(termination="THREEFOLD_REPETITION", winner=None)
        return None

    def _observe(self) -> Observation:
        return self.encoder.encode_frames(self.board, self._history)

    def _info(self, *, last_move: Move | None) -> dict[str, Any]:
        outcome = self._outcome
        return {
            "legal_action_mask": self.legal_action_mask(),
            "legal_actions": self.legal_actions(),
            "to_play": self.board.turn,
            "fen": self.board.fen(),
            "last_move": last_move.uci() if last_move is not None else None,
            "ply": self._ply,
            "is_check": self.board.is_check(),
            "result": outcome.result() if outcome is not None else "*",
            "winner": outcome.winner if outcome is not None else None,
            "termination": outcome.termination if outcome is not None else None,
        }

    def __str__(self) -> str:
        return f"<Chess-v0 history_length={self.history_length} fen='{self.board.fen()}'>"


__all__ = ["ChessEnv", "GameResult", "IllegalActionMode", "TERMINATIONS"]
