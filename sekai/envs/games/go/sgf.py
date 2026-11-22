"""SGF (FF[4]) export of Go games.

`to_sgf` writes a single-variation game record:

    (;GM[1]FF[4]CA[UTF-8]AP[sekai]SZ[9]KM[7.5]RU[Chinese]PB[Black]PW[White]
    DT[2026-10-18]RE[B+3.5];B[ee];W[cg];B[];W[])

SGF points are two letters, column then row, both counted from the **top-left** corner
(`a` = 0), so action `r * size + c` is `chr(97 + c) + chr(97 + r)`; a pass is `[]`.
Handicap games add `HA[n]` and `AB[..]` with the standard stones and White moves first.
`RU[Chinese]` is the closest standard ruleset name to the environment's area scoring with
positional superko.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Sequence

from sekai.envs.games.go.actions import check_size, pass_action, vertex_to_action
from sekai.envs.games.go.board import handicap_points
from sekai.envs.games.go.errors import InvalidVertex

APPLICATION = "sekai"
"""The `AP` (application) property `to_sgf` writes by default."""


def sgf_point(action: int, size: int) -> str:
    """Two-letter SGF point of a non-pass action (`""` for pass)."""
    if action == pass_action(size):
        return ""
    if not 0 <= action < size * size:
        raise InvalidVertex(f"Action {action} is off a {size}x{size} board.")
    row, col = divmod(int(action), size)
    return chr(ord("a") + col) + chr(ord("a") + row)


def sgf_to_action(point: str, size: int) -> int:
    """Action index of a two-letter SGF point (`""` and `"tt"` on boards <= 19 are pass)."""
    if point == "" or (point == "tt" and size <= 19):
        return pass_action(size)
    if len(point) != 2 or not point.isalpha() or not point.islower():
        raise InvalidVertex(f"{point!r} is not an SGF point.")
    col, row = ord(point[0]) - ord("a"), ord(point[1]) - ord("a")
    if not (0 <= col < size and 0 <= row < size):
        raise InvalidVertex(f"SGF point {point!r} is off a {size}x{size} board.")
    return row * size + col


def _escape(text: str) -> str:
    return str(text).replace("\\", "\\\\").replace("]", "\\]")


def to_sgf(
    size: int,
    komi: float,
    moves: Sequence[str | int],
    *,
    result: str | None = None,
    black_name: str = "Black",
    white_name: str = "White",
    date: str | _dt.date | None = None,
    handicap: int = 0,
    application: str = APPLICATION,
) -> str:
    """SGF FF[4] record of a game.

    Args:
        size: Board size.
        komi: Komi (written as `KM`).
        moves: The moves in order, as GTP vertices (`"D4"`, `"pass"`) or action
            indices, starting with Black (with White if `handicap` is at least 2).
        result: E.g. `"B+3.5"`, `"W+R"`, `"Draw"`; omitted when `None` or `"*"`.
        black_name: Black's name (`PB`).
        white_name: White's name (`PW`).
        date: `DT` value; a `datetime.date` or string, default today's date.
        handicap: Number of fixed handicap stones (`HA` and `AB`), 0 for none.
        application: The `AP` property, the program that wrote the record.
    """
    size = check_size(size)
    if date is None:
        date = _dt.date.today()
    if isinstance(date, _dt.date):
        date = date.isoformat()
    props = [
        "GM[1]",
        "FF[4]",
        "CA[UTF-8]",
        f"AP[{_escape(application)}]",
        f"SZ[{size}]",
        f"KM[{float(komi):g}]",
        "RU[Chinese]",
        f"PB[{_escape(black_name)}]",
        f"PW[{_escape(white_name)}]",
        f"DT[{_escape(date)}]",
    ]
    if result is not None and result != "*":
        props.append(f"RE[{_escape(result)}]")
    first = "B"
    if handicap:
        stones = handicap_points(size, int(handicap))
        props.append(f"HA[{int(handicap)}]")
        props.append("AB" + "".join(f"[{sgf_point(p, size)}]" for p in stones))
        first = "W"

    nodes = []
    colour = first
    for move in moves:
        if isinstance(move, str):
            action = vertex_to_action(move, size)
        else:
            action = int(move)
        nodes.append(f";{colour}[{sgf_point(action, size)}]")
        colour = "W" if colour == "B" else "B"
    return "(;" + "".join(props) + "".join(nodes) + ")\n"


__all__ = ["APPLICATION", "sgf_point", "sgf_to_action", "to_sgf"]
