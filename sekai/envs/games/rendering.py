"""What the game renderers share: themes, the frame around a board or table, player strips.

Every strategy game has an RGB renderer (`ChessRenderer`, `GoRenderer`, ...), all in one
look: a header with the game's name and its status (whose turn it is, the result), then
for the board games the board between two player strips (name, colour, captured
material, a dot on the side to move), and for the card games the felt table with its
seats. They draw with NumPy alone and read only the public state of the environment.

There are two themes: `"dark"` (the default, forest ink) and `"light"` (paper). Board
colours (wood, squares, felt) are the same in both.

Every renderer also has a text counterpart (`ChessTextRenderer`, ...) that returns the
position as a string, for terminals and logs.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray

from sekai.envs.games._draw import Painter
from sekai.error import ResetRequired
from sekai.rendering.canvas import Canvas
from sekai.rendering.renderer import BaseRenderer
from sekai.rendering.text import face
from sekai.rendering.theme import Color, ColorLike, rgb

Tone = Literal["idle", "yours", "thinking", "win", "loss", "draw"]


@dataclass(frozen=True)
class GameTheme:
    """The colours of a game frame, by role.

    Attributes:
        name: `"dark"` or `"light"`.
        background: The frame's page colour.
        surface: Raised surfaces (pills, plates).
        border: Hairlines.
        text: Body text.
        text_strong: Names and titles.
        muted: Secondary text (labels, side names).
        faint: The least important text.
        gold: Accents (the house gold).
        good: Good news and the turn dot.
        bad: Bad news.
        warn: A finished game's status dot.
        shadow: The colour of board and table shadows.
        shadow_alpha: Their opacity.
    """

    name: str
    background: Color
    surface: Color
    border: Color
    text: Color
    text_strong: Color
    muted: Color
    faint: Color
    gold: Color
    good: Color
    bad: Color
    warn: Color
    shadow: Color
    shadow_alpha: float

    def tone(self, tone: Tone) -> Color:
        """The colour of a status dot."""
        return {
            "idle": self.faint,
            "yours": self.good,
            "thinking": self.gold,
            "win": self.good,
            "loss": self.bad,
            "draw": self.warn,
        }[tone]


DARK = GameTheme(
    name="dark",
    background=rgb("#121810"),
    surface=rgb("#192116"),
    border=rgb("#2b3626"),
    text=rgb("#e6dfd2"),
    text_strong=rgb("#f8f6f2"),
    muted=rgb("#a39b88"),
    faint=rgb("#6b7461"),
    gold=rgb("#c9b48b"),
    good=rgb("#86ad93"),
    bad=rgb("#d98573"),
    warn=rgb("#d6b46c"),
    shadow=(0, 0, 0),
    shadow_alpha=0.75,
)
"""Forest ink: the default theme."""

LIGHT = GameTheme(
    name="light",
    background=rgb("#f6f3ec"),
    surface=rgb("#fdfcf9"),
    border=rgb("#e3dccd"),
    text=rgb("#2a3324"),
    text_strong=rgb("#1a2117"),
    muted=rgb("#6e6656"),
    faint=rgb("#8d8471"),
    gold=rgb("#8a6d3b"),
    good=rgb("#3f6b52"),
    bad=rgb("#b04e39"),
    warn=rgb("#9a7424"),
    shadow=rgb("#344e41"),
    shadow_alpha=0.38,
)
"""Warm paper: the light theme."""

THEMES = {"dark": DARK, "light": LIGHT}

SIDE_CHIPS: dict[str, tuple[str, str]] = {
    "white": ("#f4f1ea", "#a8a294"),
    "black": ("#1b1f19", "#4a5444"),
    "red": ("#b8352b", "#8f2a22"),
}
"""Fill and edge of the little disc that shows a player's colour."""


def game_theme(theme: GameTheme | str) -> GameTheme:
    """A `GameTheme`, or a built-in one by name (`"dark"`, `"light"`)."""
    if isinstance(theme, GameTheme):
        return theme
    try:
        return THEMES[theme]
    except KeyError:
        raise ValueError(f"Unknown theme {theme!r}; use 'dark', 'light' or a GameTheme.") from None


def unwrap(env: Any, expected: type | tuple[type, ...]) -> Any:
    """The base environment, checked to be of the type a renderer draws."""
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    if not isinstance(base, expected):
        names = (
            expected.__name__
            if isinstance(expected, type)
            else "/".join(t.__name__ for t in expected)
        )
        raise TypeError(f"This renderer draws {names}, got {type(base).__name__}.")
    return base


@dataclass(frozen=True)
class Status:
    """What the header says: a line of text, an optional second line, and a tone."""

    text: str
    sub: str | None = None
    tone: Tone = "idle"


@dataclass(frozen=True)
class Player:
    """One side of a board game, as its strip shows it."""

    side: str
    """The side's id (`"white"`, `"black"`, `"red"`)."""

    name: str
    """The name shown (`names[side]`, or the side's label)."""

    label: str
    """The side's label (`"White"`)."""


class GameRenderer(BaseRenderer, abc.ABC):
    """Base class of the game renderers: theme, header and the frame's layout.

    Subclasses draw the game itself in `draw_surface` and say what the header shows in
    `status`. The static part of a surface (a board without its pieces) can be cached by
    drawing it in `draw_static`, which runs once per layout.

    Args:
        theme: `"dark"` (the default), `"light"` or a `GameTheme`.
        header: Draw the header (the game's name and status).
        names: Display names by side (`{"white": "Agent", "black": "Random"}`); sides
            without a name show their label.
    """

    title: ClassVar[str] = "Game"
    env_types: ClassVar[tuple[type, ...]] = ()

    def __init__(
        self,
        *,
        theme: GameTheme | str = "dark",
        header: bool = True,
        names: Mapping[str, str] | None = None,
    ) -> None:
        self.theme = game_theme(theme)
        self.header = bool(header)
        self.names = dict(names or {})
        self._static: dict[Any, tuple[NDArray[np.uint8], tuple[int, int]]] = {}

    # -- what subclasses provide -------------------------------------------------------

    @abc.abstractmethod
    def frame_size(self) -> tuple[int, int]:
        """The frame's `(width, height)` in pixels."""

    @abc.abstractmethod
    def status(self, env: Any) -> Status:
        """What the header says about the game (the env is the unwrapped one)."""

    @abc.abstractmethod
    def draw_surface(self, painter: Painter, env: Any) -> None:
        """Draw the game (the env is the unwrapped one)."""

    def static_key(self, env: Any) -> Any:
        """What the cached static layer depends on besides the renderer's settings."""
        return None

    def draw_static(self, painter: Painter, env: Any) -> None:
        """Draw the parts of the frame that do not change during a game (cached)."""

    def check_ready(self, env: Any) -> None:
        """Raise `ResetRequired` if the env has no game to draw yet."""

    # -- shared ---------------------------------------------------------------------

    @property
    def scale(self) -> float:
        """Pixels per unit of the layout (1.0 at the default size)."""
        return 1.0

    def side_name(self, side: str, label: str) -> str:
        return self.names.get(side, label)

    def render(self, env: Any) -> NDArray[np.uint8]:
        base = unwrap(env, self.env_types)
        self.check_ready(base)
        width, height = self.frame_size()
        key = (width, height, self.theme.name, self.header, self.static_key(base))
        cached = self._static.get(key)
        if cached is None:
            canvas = Canvas(width, height, self.theme.background, antialias=True)
            self.draw_static(Painter(canvas), base)
            if len(self._static) > 8:
                self._static.clear()
            self._static[key] = (canvas.pixels.copy(), (width, height))
            cached = self._static[key]
        canvas = Canvas(width, height, self.theme.background, antialias=True)
        canvas.pixels[...] = cached[0]
        painter = Painter(canvas)
        if self.header:
            self.draw_header(painter, base)
        self.draw_surface(painter, base)
        return canvas.to_array()

    def header_height(self) -> float:
        return 46 * self.scale if self.header else 0.0

    def margin(self) -> float:
        return 16 * self.scale

    def draw_header(self, painter: Painter, env: Any) -> None:
        """The game's name on the left; its status (dot, text, second line) on the right."""
        s, theme = self.scale, self.theme
        width, _ = self.frame_size()
        left, top = self.margin(), self.margin() * 0.75
        mid = top + 15 * s
        painter.text(
            left, mid, self.title, theme.text_strong, size=20 * s, serif=True, valign="middle"
        )
        status = self.status(env)
        right = width - self.margin()
        sub_size, text_size = 11.5 * s, 13 * s
        text_w = face(text_size, True).width(status.text)
        sub_w = face(sub_size).width(status.sub) if status.sub else 0.0
        block = max(text_w, sub_w)
        text_left = right - block
        line = mid - (7 * s if status.sub else 0)
        painter.text(
            text_left,
            line,
            status.text,
            theme.text_strong,
            size=text_size,
            bold=True,
            valign="middle",
        )
        if status.sub:
            painter.text(
                text_left, line + 15 * s, status.sub, theme.muted, size=sub_size, valign="middle"
            )
        dot = theme.tone(status.tone)
        dx = text_left - 12 * s
        if status.tone == "yours":
            painter.circle(dx, line, 9 * s, dot, alpha=0.22)
        painter.circle(dx, line, 5 * s, dot)
        rule = top + 36 * s
        painter.rect(left, rule, right, rule + max(1.0, s), theme.border)


class BoardRenderer(GameRenderer):
    """A two-player board game: header, then the board between the two player strips.

    The bottom strip is the side whose point of view the board takes (`perspective`).

    Args:
        size: The board's width in pixels.
        theme: `"dark"`, `"light"` or a `GameTheme`.
        header: Draw the header.
        strips: Draw the player strips above and below the board.
        names: Display names by side.
    """

    #: The sides, in the order the game lists them (the first moves first).
    sides: ClassVar[tuple[str, str]] = ("white", "black")
    #: Their labels.
    labels: ClassVar[dict[str, str]] = {"white": "White", "black": "Black"}

    def __init__(
        self,
        size: int = 480,
        *,
        theme: GameTheme | str = "dark",
        header: bool = True,
        strips: bool = True,
        names: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(theme=theme, header=header, names=names)
        if size < 64:
            raise ValueError(f"size must be at least 64 pixels, got {size}.")
        self.size = int(size)
        self.strips = bool(strips)

    @property
    def scale(self) -> float:
        return self.size / 480

    # -- layout ---------------------------------------------------------------------

    def board_height(self) -> int:
        """The board's height in pixels (square boards: its width)."""
        return self.size

    def strip_height(self) -> float:
        return 50 * self.scale if self.strips else 0.0

    def frame_size(self) -> tuple[int, int]:
        m = self.margin()
        height = (
            2 * m
            + self.header_height()
            + 2 * self.strip_height()
            + self.board_height()
            + (0 if self.strips else 8 * self.scale)
        )
        return self.size + 2 * round(m), round(height)

    def board_box(self) -> tuple[float, float, float, float]:
        """`(left, top, right, bottom)` of the board."""
        left = round(self.margin())
        top = round(self.margin() + self.header_height() + self.strip_height())
        return left, top, left + self.size, top + self.board_height()

    # -- strips -----------------------------------------------------------------------

    @abc.abstractmethod
    def bottom_side(self, env: Any) -> str:
        """The side drawn at the bottom of the board."""

    def turn(self, env: Any) -> str | None:
        """The side to move, or `None` once the game is over."""
        return None

    def strip_info(self, painter: Painter, env: Any, side: str, right: float, cy: float) -> None:
        """Draw what a strip shows at its right end (captured material), right-aligned."""

    def player(self, side: str) -> Player:
        label = self.labels.get(side, side.title())
        return Player(side, self.side_name(side, label), label)

    def draw_strips(self, painter: Painter, env: Any) -> None:
        if not self.strips:
            return
        bottom = self.bottom_side(env)
        top = self.sides[1] if bottom == self.sides[0] else self.sides[0]
        left, board_top, right, board_bottom = self.board_box()
        h = self.strip_height()
        turn = self.turn(env)
        self._strip(painter, env, top, left, board_top - h, right, h, turn == top)
        self._strip(painter, env, bottom, left, board_bottom, right, h, turn == bottom)

    def _strip(
        self,
        painter: Painter,
        env: Any,
        side: str,
        left: float,
        top: float,
        right: float,
        height: float,
        is_turn: bool,
    ) -> None:
        s, theme = self.scale, self.theme
        player = self.player(side)
        cy = top + height / 2
        name_y, meta_y = cy - 8 * s, cy + 9 * s
        x = left + 2 * s
        width = painter.text(
            x, name_y, player.name, theme.text_strong, size=14 * s, bold=True, valign="middle"
        )
        if is_turn:
            dot_x = x + width + 9 * s
            painter.circle(dot_x, name_y, 6.5 * s, theme.good, alpha=0.22)
            painter.circle(dot_x, name_y, 3.5 * s, theme.good)
        fill, edge = SIDE_CHIPS.get(side, ("#888888", "#555555"))
        chip_r = 7 * s
        painter.circle(x + chip_r, meta_y, chip_r, edge)
        painter.circle(x + chip_r, meta_y, chip_r - max(s, 1.0), fill)
        painter.text(
            x + 2 * chip_r + 7 * s,
            meta_y,
            player.label,
            theme.muted,
            size=12 * s,
            valign="middle",
        )
        self.strip_info(painter, env, side, right - 2 * s, cy)

    def draw_surface(self, painter: Painter, env: Any) -> None:
        self.draw_strips(painter, env)
        self.draw_board(painter, env)

    @abc.abstractmethod
    def draw_board(self, painter: Painter, env: Any) -> None:
        """Draw the pieces and highlights on the (cached) board."""

    def draw_board_shadow(self, painter: Painter, radius: float) -> None:
        """The board's drop shadow, into the static layer."""
        s, theme = self.scale, self.theme
        left, top, right, bottom = self.board_box()
        spread = (-24 if theme.name == "dark" else -28) * s
        painter.shadow(
            left,
            top,
            right,
            bottom,
            blur=80 * s * 0.6,
            offset=(0, 30 * s * 0.6),
            spread=spread * 0.6,
            radius=radius,
            color=theme.shadow,
            alpha=theme.shadow_alpha,
        )


def tone_for_result(winner: str | None) -> Tone:
    """The tone of a finished game's status, as a spectator sees it."""
    return "draw"


def mix_alpha(color: ColorLike, alpha: float, background: ColorLike) -> Color:
    """`color` at opacity `alpha` over `background`, as one opaque colour."""
    c, b = rgb(color), rgb(background)
    return (
        round(c[0] * alpha + b[0] * (1 - alpha)),
        round(c[1] * alpha + b[1] * (1 - alpha)),
        round(c[2] * alpha + b[2] * (1 - alpha)),
    )


def ready_or_raise(ready: bool) -> None:
    if not ready:
        raise ResetRequired()


__all__ = [
    "BoardRenderer",
    "DARK",
    "GameRenderer",
    "GameTheme",
    "LIGHT",
    "Player",
    "SIDE_CHIPS",
    "Status",
    "THEMES",
    "Tone",
    "game_theme",
    "mix_alpha",
    "ready_or_raise",
    "unwrap",
]
