"""Action indices and GTP vertices for Go.

On a `size x size` board there are `size * size + 1` actions:

    action = r * size + c        # a stone on row r, column c
    action = size * size         # pass

with `r = 0` the **top** row and `c = 0` the **leftmost** column, i.e. row-major from the
top-left corner, the same order as `GoEnv.board_string()`. Engine points
(`sekai.envs.games.go.board`) use exactly the same integers.

GTP vertices (the notation shown to humans) name columns `A`..`T` **skipping** `I` and
count rows `1..size` from the **bottom**. On 9x9 the top-left point is `A9` (action 0),
the bottom-left `A1` (action 72) and the bottom-right `J1` (action 80):

    >>> vertex_to_action("A9", 9), vertex_to_action("J1", 9), vertex_to_action("pass", 9)
    (0, 80, 81)
    >>> action_to_vertex(40, 9)
    'E5'

Parsing is case-insensitive and ignores surrounding whitespace; `"pass"` in any case is
the pass action. Off-board or malformed input raises `InvalidVertex`.
"""

from __future__ import annotations

from sekai.envs.games.go.errors import InvalidVertex

MIN_SIZE = 2
MAX_SIZE = 19
COLUMNS = "ABCDEFGHJKLMNOPQRST"
"""GTP column letters; `I` is skipped."""
PASS_VERTEX = "pass"


def check_size(size: int) -> int:
    """Validate a board size (`2..19`) and return it as an `int`."""
    if isinstance(size, bool) or not isinstance(size, int) or not MIN_SIZE <= size <= MAX_SIZE:
        raise ValueError(f"Board size must be an integer in [{MIN_SIZE}, {MAX_SIZE}], got {size!r}")
    return int(size)


def num_actions(size: int) -> int:
    """`size * size + 1`: every point plus pass."""
    return size * size + 1


def pass_action(size: int) -> int:
    """The pass action, `size * size`."""
    return size * size


def action_to_rc(action: int, size: int) -> tuple[int, int]:
    """`(row, column)` of a non-pass action, row 0 at the top."""
    if not 0 <= action < size * size:
        raise InvalidVertex(f"Action {action} is not a point on a {size}x{size} board.")
    return divmod(int(action), size)


def rc_to_action(row: int, col: int, size: int) -> int:
    """Action index of `(row, column)`, row 0 at the top."""
    if not (0 <= row < size and 0 <= col < size):
        raise InvalidVertex(f"({row}, {col}) is off a {size}x{size} board.")
    return row * size + col


def action_to_vertex(action: int, size: int) -> str:
    """GTP vertex of `action` (`"pass"` for the pass action)."""
    size = check_size(size)
    action = int(action)
    if action == size * size:
        return PASS_VERTEX
    row, col = action_to_rc(action, size)
    return f"{COLUMNS[col]}{size - row}"


def vertex_to_action(vertex: str, size: int) -> int:
    """Action index of a GTP vertex such as `"D4"`, `"q16"` or `"pass"`."""
    size = check_size(size)
    if not isinstance(vertex, str):
        raise InvalidVertex(f"Vertex must be a string, got {vertex!r}.")
    text = vertex.strip().upper()
    if text == "PASS":
        return size * size
    if len(text) < 2 or text[0] not in COLUMNS or not text[1:].isdigit():
        raise InvalidVertex(f"{vertex!r} is not a GTP vertex (expected e.g. 'D4' or 'pass').")
    col = COLUMNS.index(text[0])
    number = int(text[1:])
    if col >= size or not 1 <= number <= size:
        raise InvalidVertex(f"Vertex {vertex!r} is off a {size}x{size} board.")
    return (size - number) * size + col


__all__ = [
    "MIN_SIZE",
    "MAX_SIZE",
    "COLUMNS",
    "PASS_VERTEX",
    "check_size",
    "num_actions",
    "pass_action",
    "action_to_rc",
    "rc_to_action",
    "action_to_vertex",
    "vertex_to_action",
]
