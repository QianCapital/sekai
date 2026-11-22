"""Chess: a rules engine, the AlphaZero encodings, the environment, PGN and renderers.

    env = sekai.make("Chess-v0")  # self-play; 512-ply limit
    result = env.reset(seed=0)
    while not result.done:
        action = env.action_space.sample(mask=result.info["legal_action_mask"])
        result = env.step(action)
    print(result.info["result"], result.info["termination"])

- `ChessEnv`: the side to move acts; 119 x 8 x 8 observations from its point of view;
  4672 actions (`actions`); rewards +1/-1/0 at the end for the player who just moved.
- `Board` and `Move`: a dependency-free rules engine (legal moves, FEN, SAN, draws).
- `ObservationEncoder` and the encodings in `observations` and `actions`.
- `ChessRenderer` (RGB frames) and `ChessTextRenderer` (text), `to_pgn` / `env_pgn`.
"""

from sekai.envs.games.chess.actions import (
    NUM_ACTIONS,
    NUM_MOVE_PLANES,
    NUM_VALID_ACTIONS,
    action_to_uci,
    decode_action,
    encode_move,
    legal_action_mask,
    legal_actions,
)
from sekai.envs.games.chess.board import (
    BISHOP,
    BLACK,
    KING,
    KNIGHT,
    PAWN,
    QUEEN,
    ROOK,
    STARTING_FEN,
    WHITE,
    AmbiguousMoveError,
    Board,
    IllegalMoveError,
    InvalidMoveError,
    Move,
)
from sekai.envs.games.chess.env import TERMINATIONS, ChessEnv, GameResult
from sekai.envs.games.chess.errors import ChessEnvError, GameOver, IllegalMove
from sekai.envs.games.chess.observations import (
    NUM_CONSTANT_PLANES,
    PLANES_PER_POSITION,
    ObservationEncoder,
)
from sekai.envs.games.chess.pgn import env_pgn, to_pgn
from sekai.envs.games.chess.rendering import ChessRenderer, ChessTextRenderer

__all__ = [
    # Environment
    "ChessEnv",
    "GameResult",
    "TERMINATIONS",
    # Engine
    "Board",
    "Move",
    "WHITE",
    "BLACK",
    "PAWN",
    "KNIGHT",
    "BISHOP",
    "ROOK",
    "QUEEN",
    "KING",
    "STARTING_FEN",
    # Actions
    "NUM_ACTIONS",
    "NUM_MOVE_PLANES",
    "NUM_VALID_ACTIONS",
    "encode_move",
    "decode_action",
    "legal_actions",
    "legal_action_mask",
    "action_to_uci",
    # Observations
    "ObservationEncoder",
    "PLANES_PER_POSITION",
    "NUM_CONSTANT_PLANES",
    # Records and rendering
    "to_pgn",
    "env_pgn",
    "ChessRenderer",
    "ChessTextRenderer",
    # Errors
    "ChessEnvError",
    "IllegalMove",
    "GameOver",
    "InvalidMoveError",
    "IllegalMoveError",
    "AmbiguousMoveError",
]
