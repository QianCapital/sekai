"""PGN export of chess games.

`to_pgn` writes a game as PGN, the standard text format of chess games (readable by every
chess program): the Seven Tag Roster, the starting position when it is not the standard
one (`SetUp` and `FEN`), then the moves in standard algebraic notation with move numbers,
wrapped at 80 columns, and the result. `env_pgn` writes a `ChessEnv`'s game so far.

    pgn = env_pgn(env, white="Agent", black="Random")
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Iterable, Mapping
from typing import Any

from sekai.envs.games.chess.board import STARTING_FEN, Board, Move

_ROSTER = ("Event", "Site", "Date", "Round", "White", "Black", "Result")


def _escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def to_pgn(
    moves: Iterable[Move | str],
    *,
    start_fen: str | None = None,
    result: str = "*",
    headers: Mapping[str, Any] | None = None,
    white: str = "?",
    black: str = "?",
    event: str = "?",
    site: str = "?",
    date: str | _dt.date | None = None,
    round: str = "?",
) -> str:
    """The game played from `start_fen` (default: the standard position) as PGN.

    Args:
        moves: The moves in order, as `Move`s or UCI strings.
        start_fen: The starting position, written as `SetUp`/`FEN` tags when it is not
            the standard one.
        result: `"1-0"`, `"0-1"`, `"1/2-1/2"` or `"*"` (unfinished).
        headers: Further tags (`{"Termination": "checkmate"}`), written after the roster.
        white: White's name (`"?"`, as for every roster tag, when unknown).
        black: Black's name.
        event: The event's name.
        site: Where the game was played.
        date: A `datetime.date` or a PGN date string (default: unknown, `"????.??.??"`).
        round: The round.

    Raises:
        ValueError: If a move is not legal in its position.
    """
    board = Board(start_fen or STARTING_FEN)
    if isinstance(date, _dt.date):
        date = date.strftime("%Y.%m.%d")
    tags: dict[str, Any] = {
        "Event": event,
        "Site": site,
        "Date": date or "????.??.??",
        "Round": round,
        "White": white,
        "Black": black,
        "Result": result,
    }
    if start_fen and board.fen() != STARTING_FEN:
        tags["SetUp"] = "1"
        tags["FEN"] = board.fen()
    for key, value in (headers or {}).items():
        if key not in _ROSTER:
            tags[key] = value
    lines = [f'[{key} "{_escape(value)}"]' for key, value in tags.items()]

    tokens: list[str] = []
    first = True
    for item in moves:
        move = Move.from_uci(item) if isinstance(item, str) else item
        if not board.is_legal(move):
            raise ValueError(f"Move {move.uci()} is illegal in position '{board.fen()}'.")
        if board.turn:
            tokens.append(f"{board.fullmove_number}.")
        elif first:
            tokens.append(f"{board.fullmove_number}...")
        tokens.append(board.san(move))
        board.push(move)
        first = False
    tokens.append(result)

    text, line = [], ""
    for token in tokens:
        if line and len(line) + 1 + len(token) > 80:
            text.append(line)
            line = token
        else:
            line = f"{line} {token}" if line else token
    text.append(line)
    return "\n".join(lines) + "\n\n" + "\n".join(text) + "\n"


def env_pgn(env: Any, **kwargs: Any) -> str:
    """The game of a `ChessEnv` (wrapped or not) so far as PGN; `kwargs` go to `to_pgn`.

    The result is the env's (`"*"` while the game goes on); its termination, if any, is
    written as a `Termination` tag. Needs the env's whole move history, so not a clone
    made with `full_stack=False`.
    """
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    board = base.board
    if len(board.move_stack) != base.ply:
        raise ValueError(
            "The env's board does not hold the whole game (a clone with full_stack=False)."
        )
    outcome = base.outcome
    headers = dict(kwargs.pop("headers", None) or {})
    if outcome is not None:
        headers.setdefault("Termination", outcome.termination.lower().replace("_", " "))
    kwargs.setdefault("result", outcome.result() if outcome is not None else "*")
    return to_pgn(board.move_stack, start_fen=base.start_fen, headers=headers, **kwargs)


__all__ = ["env_pgn", "to_pgn"]
