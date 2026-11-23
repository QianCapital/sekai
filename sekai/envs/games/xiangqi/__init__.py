"""Xiangqi (Chinese chess): rules engine, environment, notation and renderers.

    env = sekai.make("Xiangqi-v0")  # 400-ply TimeLimit
    result = env.reset(seed=0)
    while not result.done:
        action = env.action_space.sample(mask=result.info["legal_action_mask"])
        result = env.step(action)
    print(result.info["result"], result.info["termination"])

Full rules: palace, river, horse-leg and elephant-eye blocks, cannon screens, flying
generals. Having no legal move loses; perpetual check loses; other threefold repetitions,
120 plies without a capture and bare material draw. Perft matches the published counts
(44 / 1,920 / 79,666 / 3,290,240).

Modules: `.engine` (`Position`: board, FEN, move generation, perft), `.actions` (the
8100-action encoding in the mover's frame), `.observations` (board planes), `.notation`
(WXF and Chinese move notation), `.env` (`XiangqiEnv`) and `.rendering`
(`XiangqiRenderer`, `XiangqiTextRenderer`).
"""

from __future__ import annotations

from sekai.envs.games.xiangqi.actions import (
    NUM_ACTIONS,
    NUM_VALID_ACTIONS,
    action_to_move,
    decode_action,
    encode_move,
    legal_action_mask,
    legal_actions,
    move_to_action,
)
from sekai.envs.games.xiangqi.engine import (
    BLACK,
    RED,
    START_FEN,
    Position,
    mirror_fen,
    move_from_uci,
    move_to_uci,
)
from sekai.envs.games.xiangqi.env import TERMINATIONS, XiangqiEnv, XiangqiResult, game_outcome
from sekai.envs.games.xiangqi.errors import (
    GameOver,
    IllegalMove,
    InvalidFen,
    InvalidMoveFormat,
    XiangqiError,
)
from sekai.envs.games.xiangqi.notation import to_chinese, to_wxf
from sekai.envs.games.xiangqi.observations import (
    NUM_CONSTANT_PLANES,
    PLANES_PER_POSITION,
    ObservationEncoder,
)
from sekai.envs.games.xiangqi.rendering import XiangqiRenderer, XiangqiTextRenderer

__all__ = [
    # Environment
    "XiangqiEnv",
    "XiangqiResult",
    "TERMINATIONS",
    "game_outcome",
    # Renderers
    "XiangqiRenderer",
    "XiangqiTextRenderer",
    # Engine
    "Position",
    "RED",
    "BLACK",
    "START_FEN",
    "mirror_fen",
    "move_from_uci",
    "move_to_uci",
    # Actions
    "NUM_ACTIONS",
    "NUM_VALID_ACTIONS",
    "encode_move",
    "decode_action",
    "move_to_action",
    "action_to_move",
    "legal_actions",
    "legal_action_mask",
    # Observations
    "ObservationEncoder",
    "PLANES_PER_POSITION",
    "NUM_CONSTANT_PLANES",
    # Notation
    "to_wxf",
    "to_chinese",
    # Errors
    "XiangqiError",
    "InvalidFen",
    "InvalidMoveFormat",
    "IllegalMove",
    "GameOver",
]
