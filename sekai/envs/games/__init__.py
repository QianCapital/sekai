"""Strategy games: chess, go, xiangqi, heads-up no-limit hold'em and blackjack.

    sekai.make("Chess-v0")          # also ChessVsRandom-v0
    sekai.make("Go9x9-v0")          # also Go13x13-v0, Go19x19-v0
    sekai.make("Xiangqi-v0")
    sekai.make("HeadsUpHoldem-v0")
    sekai.make("Blackjack-v0")

The two-player games follow the AlphaZero convention, so one network plays both sides
and tree search needs nothing else:

- The env is played from both sides by the caller (self-play): each step is a move by
  the player to act, and the observation is always from the point of view of the player
  to act. `OpponentWrapper` plays one side with a fixed policy instead.
- `info["legal_action_mask"]` (an `int8` mask; also `env.action_mask()`) marks the
  legal actions. An illegal action raises `IllegalMove`.
- Rewards come at the end of the game, for the player who just moved: `+1` win, `-1`
  loss, `0` draw (net big blinds in poker), and `0` on every other step.
- The board games have `clone()`, an independent copy to search from.

Each game lives in its own subpackage (`sekai.envs.games.chess`, `.go`, `.xiangqi`,
`.poker`, `.blackjack`) with its rules engine, action and observation encodings, notation
and renderers. Importing this package imports none of them until they are used.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from sekai.envs.games.errors import GameError, GameOver, IllegalMove
from sekai.envs.games.policies import Policy, RandomPolicy
from sekai.envs.games.rendering import DARK, LIGHT, GameRenderer, GameTheme
from sekai.envs.games.wrappers import OpponentWrapper

if TYPE_CHECKING:
    from sekai.envs.games.blackjack import BlackjackEnv, BlackjackRenderer, BlackjackTextRenderer
    from sekai.envs.games.chess import ChessEnv, ChessRenderer, ChessTextRenderer
    from sekai.envs.games.go import GoEnv, GoRenderer, GoTextRenderer
    from sekai.envs.games.poker import HoldemEnv, HoldemRenderer, HoldemTextRenderer
    from sekai.envs.games.xiangqi import XiangqiEnv, XiangqiRenderer, XiangqiTextRenderer

_LAZY = {
    "ChessEnv": "chess",
    "ChessRenderer": "chess",
    "ChessTextRenderer": "chess",
    "GoEnv": "go",
    "GoRenderer": "go",
    "GoTextRenderer": "go",
    "XiangqiEnv": "xiangqi",
    "XiangqiRenderer": "xiangqi",
    "XiangqiTextRenderer": "xiangqi",
    "HoldemEnv": "poker",
    "HoldemRenderer": "poker",
    "HoldemTextRenderer": "poker",
    "BlackjackEnv": "blackjack",
    "BlackjackRenderer": "blackjack",
    "BlackjackTextRenderer": "blackjack",
}
_GAMES = ("chess", "go", "xiangqi", "poker", "blackjack")


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module = importlib.import_module(f"sekai.envs.games.{_LAZY[name]}")
        return getattr(module, name)
    if name in _GAMES:
        return importlib.import_module(f"sekai.envs.games.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Environments
    "ChessEnv",
    "GoEnv",
    "XiangqiEnv",
    "HoldemEnv",
    "BlackjackEnv",
    # Renderers
    "ChessRenderer",
    "GoRenderer",
    "XiangqiRenderer",
    "HoldemRenderer",
    "BlackjackRenderer",
    "ChessTextRenderer",
    "GoTextRenderer",
    "XiangqiTextRenderer",
    "HoldemTextRenderer",
    "BlackjackTextRenderer",
    "GameRenderer",
    "GameTheme",
    "DARK",
    "LIGHT",
    # Playing
    "OpponentWrapper",
    "Policy",
    "RandomPolicy",
    # Errors
    "GameError",
    "GameOver",
    "IllegalMove",
]
