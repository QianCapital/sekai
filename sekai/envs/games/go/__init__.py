"""Go: engine, environment, scoring, SGF records and renderers.

    import sekai

    env = sekai.make("Go9x9-v0")  # also Go13x13-v0 and Go19x19-v0
    result = env.reset(seed=0)
    done = False
    while not done:
        action = env.action_space.sample(mask=result.info["legal_action_mask"])
        result = env.step(action)
        done = result.done
    print(result.info["result"], result.info["score"])

Square boards from 2x2 to 19x19, komi, fixed handicaps of 2 to 9 stones, area scoring
(Tromp-Taylor) and positional superko; two passes end the game. Modules: `board` (the fast
incremental engine and playouts), `env` (`GoEnv`), `actions` (action indices and GTP
vertices), `observations` (AlphaGo Zero planes), `scoring` (dead-stone estimation for
people), `sgf` (game records) and `rendering` (`GoRenderer`, `GoTextRenderer`).
"""

from __future__ import annotations

from sekai.envs.games.go.actions import (
    COLUMNS,
    action_to_vertex,
    num_actions,
    pass_action,
    vertex_to_action,
)
from sekai.envs.games.go.board import BLACK, EMPTY, WHITE, Board, geometry, handicap_points
from sekai.envs.games.go.env import GoEnv, GoResult, format_result
from sekai.envs.games.go.errors import GameOver, GoEnvError, IllegalMove, InvalidVertex
from sekai.envs.games.go.observations import ObservationEncoder, decode_observation
from sekai.envs.games.go.rendering import GoRenderer, GoTextRenderer
from sekai.envs.games.go.scoring import GoScore, area_score, estimate_ownership, final_score
from sekai.envs.games.go.sgf import to_sgf

__all__ = [
    # Environment
    "GoEnv",
    "GoResult",
    "format_result",
    # Renderers
    "GoRenderer",
    "GoTextRenderer",
    # Engine
    "Board",
    "geometry",
    "handicap_points",
    "EMPTY",
    "BLACK",
    "WHITE",
    # Actions / vertices
    "COLUMNS",
    "num_actions",
    "pass_action",
    "action_to_vertex",
    "vertex_to_action",
    # Observations
    "ObservationEncoder",
    "decode_observation",
    # Scoring
    "GoScore",
    "area_score",
    "estimate_ownership",
    "final_score",
    # SGF
    "to_sgf",
    # Errors
    "GoEnvError",
    "IllegalMove",
    "GameOver",
    "InvalidVertex",
]
