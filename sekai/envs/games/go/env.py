"""The Go environment.

`GoEnv` is a *turn-based two-player* game exposed through the single-agent `sekai.Env`
interface, which is what AlphaZero-style self-play needs:

- The acting agent is always **the player to move** (`info["to_play"]`, `"black"` or
  `"white"`). Observations are AlphaGo Zero planes split by player to move versus
  opponent (see `sekai.envs.games.go.observations`), so one network plays both colours.
- Actions are `r * size + c` (row 0 at the top) plus pass `= size * size`; see
  `sekai.envs.games.go.actions` for GTP vertices such as `"D4"`.
- `reward` is **terminal only** and from the perspective of **the player who just moved**:
  `+1` if that player won, `-1` if they lost, `0` for a draw and for every non-terminal
  step.
- `info["legal_action_mask"]` (also `env.action_mask()`) is the `int8` legal-move mask
  over all `size * size + 1` actions (pass is always legal while the game is on). An
  illegal action raises `sekai.envs.games.go.IllegalMove`, or with
  `illegal_action="lose"` forfeits the game: reward `-1`, result `"W+F"` or `"B+F"`.

Rules
-----
Black moves first (White after a handicap). Stones without liberties are captured;
suicide is illegal. With `superko=True` (default) **positional superko** applies: a move
may not recreate any earlier stone configuration of the game, including the starting one
(whose turn it is does not matter). With `superko=False` only simple ko applies.
Two consecutive passes end the game (`terminated=True`); reaching `max_moves` plies
(default `3 * size * size`, passes included) ends it with `truncated=True`. Either way
the game is scored by **Tromp-Taylor area scoring**: a point counts for a colour if it holds
that colour's stone, or is empty and reaches only that colour's stones; White adds
`komi`. No dead-stone judgement happens inside the environment (the zero-knowledge RL
contract); `sekai.envs.games.go.scoring.final_score` estimates dead stones for humans.
A truncated game is scored and rewarded exactly like a finished one.

Reset options
-------------
`reset(options=...)` accepts `komi` (float), `handicap` (`0` or `2..9` stones on the
standard star points of 9x9, 13x13 and 19x19 boards; White then moves first; komi is
not changed automatically), `moves` (GTP vertices, `"pass"` included, or action ints,
replayed from the start, e.g. to restore a game) and, for puzzles and tests, `board` (a
board string as returned by `GoEnv.board_string()`) with `to_play` (`"black"` or
`"white"`). Options apply to that episode only. If `reset` raises (a bad option, an
illegal move in `moves`) the environment must be reset again before stepping.

Info
----
Every `reset`/`step` info dict holds `legal_action_mask` (`int8`), `legal_actions`
(`int64`), `to_play` (`"black"`/`"white"`), `last_move` (GTP vertex, `"pass"` or
`None`), `ply`, `captures` (`{"black": n, "white": n}`, stones captured **by** each
colour), `passes` (consecutive), `komi`, `result` (`"*"` while ongoing, else `"B+3.5"`,
`"W+0.5"` or `"Draw"`) and `winner` (`"black"`, `"white"` or `None`); once the game is
over also `score` (`{"black": points, "white": points including komi}`) and
`termination` (`"two passes"`, `"max moves"` or, for a forfeit, `"illegal move"`, when
`illegal_action` holds the action).
"""

from __future__ import annotations

import copy
import functools
import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from sekai._types import SeedType
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.envs.games.go.actions import action_to_vertex, check_size, vertex_to_action
from sekai.envs.games.go.board import BLACK, NO_POINT, WHITE, Board, geometry, handicap_points
from sekai.envs.games.go.errors import GameOver, IllegalMove
from sekai.envs.games.go.observations import DEFAULT_HISTORY_LENGTH, ObservationEncoder
from sekai.error import InvalidAction, ResetRequired
from sekai.spaces.discrete import Discrete

Observation = NDArray[np.float32]

DEFAULT_KOMI = 7.5
REASON_TWO_PASSES = "two passes"
REASON_MAX_MOVES = "max moves"
REASON_ILLEGAL_MOVE = "illegal move"
IllegalActionMode = Literal["raise", "lose"]
_COLOR_NAME = {BLACK: "black", WHITE: "white"}
_COLOR_CODE = {"black": BLACK, "white": WHITE, "b": BLACK, "w": WHITE}
_RESET_OPTIONS = frozenset({"komi", "handicap", "moves", "board", "to_play"})


def format_result(margin: float) -> str:
    """`"B+3.5"`, `"W+0.5"` or `"Draw"` for a Black-minus-White margin."""
    if margin > 0:
        return f"B+{margin:g}"
    if margin < 0:
        return f"W+{-margin:g}"
    return "Draw"


def winner_of(margin: float) -> str | None:
    """`"black"`, `"white"` or `None` (a draw) for a Black-minus-White margin."""
    if margin > 0:
        return "black"
    if margin < 0:
        return "white"
    return None


@dataclass(frozen=True, slots=True)
class GoResult:
    """How a finished game was scored (Tromp-Taylor area scoring).

    Attributes:
        winner: `"black"`, `"white"` or `None` for a draw.
        black: Black's area points.
        white: White's area points including komi.
        margin: `black - white`.
        result: `"B+3.5"`, `"W+0.5"` or `"Draw"`; `"W+F"` or `"B+F"` for a forfeit, whose
            points are the board's area score when the illegal move was played.
        reason: `"two passes"`, `"max moves"` or `"illegal move"` (a forfeit).
    """

    winner: str | None
    black: float
    white: float
    margin: float
    result: str
    reason: str


@functools.cache
def _vertex_table(size: int) -> tuple[str, ...]:
    """The GTP vertex of every action on a `size x size` board, pass last."""
    return tuple(action_to_vertex(a, size) for a in range(size * size + 1))


class GoEnv(Env[Observation, np.intp]):
    """Go on a `size x size` board as a sekai environment.

    Registered as `Go9x9-v0`, `Go13x13-v0` and `Go19x19-v0`; see the module docstring for
    the rules, observations, rewards, reset options and info.

    Args:
        size: Board size, `2..19` (the registered environments use 9, 13 and 19). Fixed
            per instance because it determines the spaces.
        komi: Points added to White's score; per-episode override via
            `reset(options={"komi": ...})`. An integer komi makes draws possible.
        history_length: Positions stacked into the observation (`T`); the observation
            has `2 * T + 1` planes. `8` reproduces AlphaGo Zero's 17 planes.
        max_moves: Plies (passes included) after which the game is truncated and scored;
            default `3 * size * size`.
        superko: Positional superko (default) or, if `False`, simple ko only.
        illegal_action: `"raise"` (default) raises `IllegalMove` on an illegal action;
            `"lose"` ends the game as a loss for the mover (reward `-1`) instead.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_fps": 4}
    #: The sides, in the order they move (Black first).
    sides: ClassVar[tuple[str, str]] = ("black", "white")
    #: The values of `info["to_play"]`, in the same order.
    players: ClassVar[tuple[str, str]] = ("black", "white")

    def __init__(
        self,
        size: int = 9,
        komi: float = DEFAULT_KOMI,
        history_length: int = DEFAULT_HISTORY_LENGTH,
        max_moves: int | None = None,
        superko: bool = True,
        illegal_action: IllegalActionMode = "raise",
    ) -> None:
        if illegal_action not in ("raise", "lose"):
            raise ValueError(f"illegal_action must be 'raise' or 'lose', got {illegal_action!r}.")
        self.illegal_action: IllegalActionMode = illegal_action
        self.size = check_size(size)
        self._default_komi = _check_komi(komi)
        self.komi = self._default_komi
        if max_moves is None:
            max_moves = 3 * self.size * self.size
        if int(max_moves) < 1:
            raise ValueError(f"max_moves must be positive, got {max_moves!r}.")
        self.max_moves = int(max_moves)
        self.superko = bool(superko)
        self.history_length = int(history_length)

        self.geometry = geometry(self.size)
        self.num_points = self.size * self.size
        self.pass_action = self.num_points
        self.encoder = ObservationEncoder(self.size, self.history_length)
        self.observation_space = self.encoder.space()
        self.action_space = Discrete(self.num_points + 1)
        self._vertices = _vertex_table(self.size)

        self._board = Board(self.size)
        # Positions as bytes (one colour per point), oldest first.
        self._history: deque[bytes] = deque(maxlen=self.history_length)
        self._seen: dict[int, bytes] = {}
        # The positions of `_seen` by number of stones, as (black, white) bitsets.
        self._by_stones: dict[int, tuple[tuple[int, int], ...]] = {}
        self._moves: list[str] = []
        self._captures = [0, 0, 0]
        self._passes = 0
        self._ply = 0
        self._last_action: int | None = None
        self._handicap = 0
        self._handicap_points: list[int] = []
        self._outcome: GoResult | None = None
        self._mask: NDArray[np.int8] | None = None
        self._has_reset = False

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
        try:
            return self._reset(dict(options or {}))
        except BaseException:
            # A failed reset (bad option, illegal move in `moves`) leaves no usable game.
            self._has_reset = False
            raise

    def _reset(self, opts: dict[str, Any]) -> ResetResult[Observation]:
        unknown = set(opts) - _RESET_OPTIONS
        if unknown:
            raise ValueError(
                f"Unknown reset option(s) {sorted(unknown)}; expected {sorted(_RESET_OPTIONS)}."
            )
        self.komi = _check_komi(opts.get("komi", self._default_komi))
        handicap = opts.get("handicap", 0) or 0
        if isinstance(handicap, bool) or not isinstance(handicap, (int, np.integer)):
            raise ValueError(f"handicap must be an integer, got {handicap!r}.")
        handicap = int(handicap)

        if opts.get("board") is not None:
            if handicap:
                raise ValueError("Options 'board' and 'handicap' cannot be combined.")
            to_play = _color_code(opts.get("to_play", "black"))
            board = Board.from_string(str(opts["board"]), to_play)
            if board.size != self.size:
                raise ValueError(f"Board string is {board.size}x{board.size}, env is {self.size}.")
            self._handicap_points = []
        elif handicap:
            self._handicap_points = handicap_points(self.size, handicap)
            colors = [0] * self.num_points
            for p in self._handicap_points:
                colors[p] = BLACK
            board = Board.from_colors(colors, self.size, WHITE)
        else:
            if "to_play" in opts:
                raise ValueError("Option 'to_play' requires option 'board'.")
            board = Board(self.size)
            self._handicap_points = []

        self._handicap = handicap
        self._board = board
        self._history.clear()
        self._seen = {}
        self._by_stones = {}
        self._moves = []
        self._captures = [0, 0, 0]
        self._passes = 0
        self._ply = 0
        self._last_action = None
        self._outcome = None
        self._mask = None
        self._record_position()
        self._has_reset = True

        moves = opts.get("moves")
        for move in () if moves is None else moves:  # (a list, or an array of actions)
            if self._outcome is not None:
                raise GameOver(self._outcome.result)
            self._apply(self._to_action(move))

        return ResetResult(observation=self._observe(), info=self._info())

    def step(self, action: int | np.integer[Any]) -> StepResult[Observation]:
        if not self._has_reset:
            raise ResetRequired()
        if self._outcome is not None:
            raise GameOver(self._outcome.result)
        if not self.action_space.contains(action):
            raise InvalidAction(action, self.action_space)

        mover = _COLOR_NAME[self._board.to_play]
        if self.illegal_action == "lose" and not self.is_legal(int(action)):
            self._outcome = self._forfeit()
            info = self._info()
            info["illegal_action"] = int(action)
            return StepResult(self._observe(), -1.0, True, False, info)
        self._apply(int(action))

        reward = 0.0
        outcome = self._outcome
        if outcome is not None and outcome.winner is not None:
            reward = 1.0 if outcome.winner == mover else -1.0
        return StepResult(
            observation=self._observe(),
            reward=reward,
            terminated=outcome is not None and outcome.reason == REASON_TWO_PASSES,
            truncated=outcome is not None and outcome.reason == REASON_MAX_MOVES,
            info=self._info(),
        )

    # ------------------------------------------------------------------
    # Public helpers (agents, search, renderers and tests)
    # ------------------------------------------------------------------

    @property
    def board(self) -> Board:
        """The engine board. Treat as read-only; use `board.copy()` or `clone()`."""
        return self._board

    @property
    def to_play(self) -> str:
        """`"black"` or `"white"`."""
        return _COLOR_NAME[self._board.to_play]

    @property
    def captures(self) -> dict[str, int]:
        """Stones captured **by** each colour so far."""
        return {"black": self._captures[BLACK], "white": self._captures[WHITE]}

    @property
    def last_move(self) -> str | None:
        """GTP vertex (or `"pass"`) of the last move, `None` before the first move."""
        if self._last_action is None:
            return None
        return self._vertices[self._last_action]

    @property
    def moves(self) -> list[str]:
        """Every move played since the start (after handicap), as GTP vertices / `"pass"`."""
        return list(self._moves)

    @property
    def ply(self) -> int:
        """Moves (passes included) played since the start."""
        return self._ply

    @property
    def consecutive_passes(self) -> int:
        """Passes in a row just played."""
        return self._passes

    @property
    def handicap(self) -> int:
        """Handicap stones placed at reset (0 for none)."""
        return self._handicap

    @property
    def handicap_stones(self) -> list[str]:
        """GTP vertices of the handicap stones placed at reset."""
        return [action_to_vertex(p, self.size) for p in self._handicap_points]

    @property
    def ko_point(self) -> str | None:
        """Vertex the player to move may not play right now because of simple ko (display)."""
        if self._outcome is not None or self._board.ko == NO_POINT:
            return None
        return action_to_vertex(self._board.ko, self.size)

    @property
    def done(self) -> bool:
        """Whether the game is over."""
        return self._outcome is not None

    @property
    def outcome(self) -> GoResult | None:
        """How the game was scored; `None` while the game is in progress."""
        return self._outcome

    def board_string(self) -> str:
        """`size*size` characters row-major from the top-left: `.` empty, `X` black,
        `O` white."""
        return self._board.to_string()

    def legal_action_mask(self) -> NDArray[np.int8]:
        """`int8` mask over all actions (pass last); all zeros once the game is over."""
        if self._outcome is not None:
            return np.zeros(self.num_points + 1, dtype=np.int8)
        if self._mask is None:
            self._mask = self._compute_mask()
        mask: NDArray[np.int8] = self._mask.copy()
        return mask

    def action_mask(self) -> NDArray[np.int8]:
        """The legal-move mask, `legal_action_mask()` (sekai's action-mask convention)."""
        return self.legal_action_mask()

    def legal_actions(self) -> NDArray[np.int64]:
        """The legal actions, ascending (pass last)."""
        return np.flatnonzero(self.legal_action_mask()).astype(np.int64)

    def is_legal(self, action: int) -> bool:
        """Whether `action` is a legal move now."""
        if self._outcome is not None or not 0 <= int(action) <= self.num_points:
            return False
        return self._illegal_reason(int(action)) is None

    def observe(self) -> Observation:
        """Observation of the current position, as last returned by `reset`/`step`."""
        return self._observe()

    def encode_vertex(self, vertex: str) -> int:
        """Action index of a GTP vertex (`"D4"`, `"pass"`)."""
        return vertex_to_action(vertex, self.size)

    def decode_action(self, action: int) -> str:
        """GTP vertex of an action index (`"pass"` for the pass action)."""
        return action_to_vertex(action, self.size)

    def tromp_taylor_margin(self) -> float:
        """Black's Tromp-Taylor area minus White's area plus komi, for the current position."""
        black, white = self._board.area_score()
        return black - white - self.komi

    def clone(self) -> GoEnv:
        """Independent copy of the environment, including history and superko state.

        Intended for search: step the clone freely without disturbing the original. Shares
        the immutable geometry tables and spaces; the clone gets its own lazily-seeded RNG.
        """
        new = copy.copy(self)
        new._board = self._board.copy()
        new._history = deque(self._history, maxlen=self.history_length)
        new._seen = dict(self._seen)
        new._by_stones = dict(self._by_stones)
        new._moves = list(self._moves)
        new._captures = list(self._captures)
        new._handicap_points = list(self._handicap_points)
        new._rng = None
        return new

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _to_action(self, move: Any) -> int:
        if isinstance(move, str):
            return vertex_to_action(move, self.size)
        if isinstance(move, (int, np.integer)) and not isinstance(move, bool):
            if not 0 <= int(move) <= self.num_points:
                raise InvalidAction(move, self.action_space)
            return int(move)
        raise ValueError(f"Moves must be GTP vertices or action indices, got {move!r}.")

    def _repeats(self, p: int) -> bool:
        """Whether playing `p` would recreate an earlier stone configuration."""
        board = self._board
        stored = self._seen.get(board.hash_after(p))
        return stored is not None and stored == board.colors_after(p)

    def _illegal_reason(self, action: int) -> str | None:
        board = self._board
        if action == self.num_points:
            return None
        reason = board.illegal_reason(action, ignore_ko=self.superko)
        if reason is None and self.superko and self._repeats(action):
            reason = "ko" if action == board.ko else "superko"
        return reason

    def _compute_mask(self) -> NDArray[np.int8]:
        board = self._board
        n = self.num_points
        legal, captures = board.legal_bits(ignore_ko=self.superko)
        if self.superko and legal:
            legal &= ~self._repeating(legal, captures)
        mask = np.empty(n + 1, dtype=np.int8)
        raw = np.frombuffer(legal.to_bytes((n + 7) // 8, "little"), dtype=np.uint8)
        mask[:n] = np.unpackbits(raw, bitorder="little")[:n]
        mask[n] = 1
        return mask

    def _repeating(self, legal: int, captures: int) -> int:
        """The legal points (a bitset) whose move would recreate an earlier position.

        A move adds a stone of the mover's and removes the stones it captures, so it can
        only recreate a position with as many stones; those are compared as bitsets.
        """
        board = self._board
        by_stones = self._by_stones
        count = board.stone_count()
        black, white = board.stones[BLACK], board.stones[WHITE]
        to_play = board.to_play
        mine, theirs = (black, white) if to_play == BLACK else (white, black)
        repeats = 0
        earlier = by_stones.get(count + 1)
        if earlier:  # moves that capture nothing: every stone stays
            for seen_black, seen_white in earlier:
                seen_mine, seen_theirs = (
                    (seen_black, seen_white) if to_play == BLACK else (seen_white, seen_black)
                )
                if seen_theirs == theirs and seen_mine & mine == mine:
                    repeats |= seen_mine ^ mine
            repeats &= legal & ~captures
        while captures:
            low = captures & -captures
            captures ^= low
            taken = board.captured_by(low.bit_length() - 1)
            earlier = by_stones.get(count + 1 - taken.bit_count())
            if earlier:
                after = (mine | low, theirs & ~taken)
                if to_play != BLACK:
                    after = (after[1], after[0])
                if after in earlier:
                    repeats |= low
        return repeats

    def _apply(self, action: int) -> None:
        """Play `action` for the side to move, update history and check for the end."""
        board = self._board
        mover = board.to_play
        if action == self.num_points:
            board.pass_move()
            self._passes += 1
        else:
            mask = self._mask
            if mask is None or not mask[action]:  # a computed mask already says it is legal
                reason = self._illegal_reason(action)
                if reason is not None:
                    vertex = self.decode_action(action)
                    raise IllegalMove(action, vertex, reason, board.to_string())
            self._captures[mover] += board.play_unchecked(action)
            self._passes = 0
        self._ply += 1
        self._last_action = action
        self._moves.append(self._vertices[action])
        self._mask = None
        self._record_position()
        if self._passes >= 2:
            self._outcome = self._score_position(REASON_TWO_PASSES)
        elif self._ply >= self.max_moves:
            self._outcome = self._score_position(REASON_MAX_MOVES)

    def _record_position(self) -> None:
        board = self._board
        config = board.config()
        if board.hash not in self._seen:
            self._seen[board.hash] = config
            count = board.stone_count()
            entry = (board.stones[BLACK], board.stones[WHITE])
            self._by_stones[count] = (*self._by_stones.get(count, ()), entry)
        self._history.append(config)

    def _score_position(self, reason: str) -> GoResult:
        black, white = self._board.area_score()
        white_total = white + self.komi
        margin = black - white_total
        return GoResult(
            winner=winner_of(margin),
            black=float(black),
            white=float(white_total),
            margin=float(margin),
            result=format_result(margin),
            reason=reason,
        )

    def _forfeit(self) -> GoResult:
        """The side to move loses by its illegal move (the board is as it was)."""
        black, white = self._board.area_score()
        white_total = white + self.komi
        loser = self._board.to_play
        return GoResult(
            winner=_COLOR_NAME[3 - loser],
            black=float(black),
            white=float(white_total),
            margin=float(black - white_total),
            result="W+F" if loser == BLACK else "B+F",
            reason=REASON_ILLEGAL_MOVE,
        )

    def _observe(self) -> Observation:
        # ObservationEncoder.encode() on the history, without stacking arrays.
        history = self._history
        k = len(history)
        if k == 0:
            raise ValueError("history must contain at least the current position.")
        t = self.history_length
        size = self.size
        to_play = self._board.to_play
        stack = np.frombuffer(b"".join(reversed(history)), dtype=np.int8).reshape(k, size, size)
        obs = np.zeros(self.encoder.shape, dtype=np.float32)
        obs[:k] = stack == to_play
        obs[t : t + k] = stack == 3 - to_play
        if to_play == BLACK:
            obs[2 * t] = 1.0
        return obs

    def _info(self) -> dict[str, Any]:
        mask = self.legal_action_mask()
        outcome = self._outcome
        info: dict[str, Any] = {
            "legal_action_mask": mask,
            "legal_actions": np.flatnonzero(mask).astype(np.int64),
            "to_play": self.to_play,
            "last_move": self.last_move,
            "ply": self._ply,
            "captures": self.captures,
            "passes": self._passes,
            "komi": self.komi,
            "result": outcome.result if outcome is not None else "*",
            "winner": outcome.winner if outcome is not None else None,
        }
        if outcome is not None:
            info["score"] = {"black": outcome.black, "white": outcome.white}
            info["termination"] = outcome.reason
        return info

    def __str__(self) -> str:
        return (
            f"<Go{self.size}x{self.size} komi={self.komi:g} ply={self._ply} "
            f"to_play={self.to_play} board='{self._board.to_string()}'>"
        )


def _check_komi(komi: Any) -> float:
    try:
        value = float(komi)
    except (TypeError, ValueError):
        raise ValueError(f"komi must be a number, got {komi!r}.") from None
    if not math.isfinite(value):
        raise ValueError(f"komi must be finite, got {komi!r}.")
    return value


def _color_code(name: Any) -> int:
    if isinstance(name, str) and name.lower() in _COLOR_CODE:
        return _COLOR_CODE[name.lower()]
    if name in (BLACK, WHITE):
        return int(name)
    raise ValueError(f"to_play must be 'black' or 'white', got {name!r}.")


def replay(size: int, moves: Sequence[str | int], **kwargs: Any) -> GoEnv:
    """A fresh `GoEnv` with `moves` replayed (`kwargs` go to the reset options, e.g.
    `komi` or `handicap`)."""
    env = GoEnv(size)
    env.reset(options={**kwargs, "moves": list(moves)})
    return env


__all__ = [
    "DEFAULT_KOMI",
    "REASON_TWO_PASSES",
    "REASON_MAX_MOVES",
    "REASON_ILLEGAL_MOVE",
    "GoEnv",
    "IllegalActionMode",
    "GoResult",
    "format_result",
    "winner_of",
    "replay",
]
