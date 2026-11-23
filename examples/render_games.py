"""Draw a game of each strategy game and save each as a GIF, in both themes.

Each game has a renderer in `sekai.envs.games`, recorded here with RecordFrames: chess
replays the Opera Game (Morphy, Paris 1858); go and xiangqi play simple policies to the
end of a game; poker plays a few hands and blackjack a few rounds. Each game is recorded
once in the dark theme, the default (`games_chess.gif`, ...), and in the light one
(`games_chess_light.gif`, ...). Needs Pillow: pip install "sekai[render]".

Pass a directory to keep the images (this is how the documentation's animations are
made):

    python examples/render_games.py docs/assets
"""

import sys
import tempfile
from functools import partial
from pathlib import Path

import numpy as np

import sekai
from sekai.envs.games import (
    BlackjackRenderer,
    ChessRenderer,
    GoRenderer,
    HoldemRenderer,
    XiangqiRenderer,
)
from sekai.rendering import RecordFrames

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)

OPERA_GAME = (
    "e4 e5 Nf3 d6 d4 Bg4 dxe5 Bxf3 Qxf3 dxe5 Bc4 Nf6 Qb3 Qe7 Nc3 c6 Bg5 b5 Nxb5 cxb5 "
    "Bxb5+ Nbd7 O-O-O Rd8 Rxd7 Rxd7 Rd1 Qe6 Bxd7+ Nxd7 Qb8+ Nxb8 Rd8#"
)


def themed(env, renderer):
    """Record `env` in the dark theme (the default) and, around that, in the light one."""
    return RecordFrames(RecordFrames(env, renderer()), renderer(theme="light"))


def hold(env, n):
    """Show the last frame `n` more times, in both recordings."""
    for recording in (env.env, env):
        recording.frames.extend([recording.frames[-1]] * n)


def save(env, name, fps):
    for recording, suffix in ((env.env, ""), (env, "_light")):
        path = recording.save(out / f"games_{name}{suffix}.gif", fps=fps)
        print(f"{name}{suffix}: {len(recording.frames)} frames -> {path}")
    env.close()


# Chess: the Opera Game, from White's side.
names = {"white": "Paul Morphy", "black": "Duke of Brunswick"}
env = themed(sekai.make("Chess-v0"), partial(ChessRenderer, 400, names=names))
env.reset(seed=0)
for san in OPERA_GAME.split():
    env.step(env.unwrapped.encode_move(san))
hold(env, 4)  # the final position
save(env, "chess", fps=1.5)


# Go: random stones that never fill their own eyes, until both players pass.
def go_policy(base):
    board = base.board
    points = [int(a) for a in base.legal_actions() if a < base.pass_action]
    points = [p for p in points if not board.is_eye(p)]
    return int(rng.choice(points)) if points else base.pass_action


env = themed(sekai.make("Go9x9-v0"), partial(GoRenderer, 400))
result = env.reset(seed=0)
while not result.done:
    result = env.step(go_policy(env.unwrapped))
hold(env, 6)
save(env, "go", fps=6)


# Xiangqi: captures first, otherwise a random move.
def xiangqi_policy(base, info):
    legal = np.flatnonzero(info["legal_action_mask"])
    board = base.position.board  # 90 points, rank * 9 + file, Red's side first

    def target(action):
        uci = base.decode_action(int(action))  # e.g. "h2e2"
        return int(uci[3]) * 9 + "abcdefghi".index(uci[2])

    captures = [a for a in legal if board[target(a)]]
    return int(rng.choice(captures if captures else legal))


env = themed(sekai.make("Xiangqi-v0", max_episode_steps=120), partial(XiangqiRenderer, 400))
result = env.reset(seed=3)
while not result.done:
    result = env.step(xiangqi_policy(env.unwrapped, result.info))
hold(env, 4)
save(env, "xiangqi", fps=4)


# Poker: a few hands between two random players.
env = themed(sekai.make("HeadsUpHoldem-v0"), partial(HoldemRenderer, 560))
for hand in range(4):
    result = env.reset(seed=hand)
    while not result.done:
        legal = np.flatnonzero(result.info["legal_action_mask"])
        result = env.step(int(rng.choice(legal)))
    hold(env, 2)
save(env, "poker", fps=1.5)


# Blackjack: bet the minimum, hit below 17.
env = themed(sekai.make("Blackjack-v0"), partial(BlackjackRenderer, 560))
for round_ in range(6):
    result = env.reset(seed=7) if round_ == 0 else env.reset()
    while not result.done:
        mask = result.info["legal_action_mask"]
        if mask[5]:  # betting: the smallest bet
            action = 5
        else:
            view = result.info["view"]
            total = view["seats"][0]["hands"][view["active_hand"]]["total"]
            action = 1 if total < 17 and mask[1] else 0
        result = env.step(action)
    hold(env, 2)
save(env, "blackjack", fps=1.5)
