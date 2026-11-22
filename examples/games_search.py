"""AlphaZero-style tree search on the strategy games, with nothing but the environment.

A two-player game env gives a search everything it needs: `clone()` to branch, the legal
moves in `info["legal_action_mask"]`, and rewards for the player who just moved. This
example's PUCT search (the search of AlphaZero) evaluates a position with a uniform prior
and a random playout. Swap `evaluate` for a network that returns `(logits, value)` and it
is AlphaZero's search; the observation the network would read is always from the side to
move.

The search finds a mate in one in chess, then plays a game of 5x5 go against a random
player.
"""

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import sekai
from sekai.envs.games.go import GoEnv

C_PUCT = 1.5
PLAYOUT_PLIES = 100


@dataclass
class Node:
    prior: float
    visits: int = 0
    value_sum: float = 0.0  # from the point of view of the player who moved into this node
    children: dict[int, "Node"] = field(default_factory=dict)
    terminal: float | None = None

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


def evaluate(env: Any, info: dict[str, Any], rng: np.random.Generator) -> tuple[dict, float]:
    """Priors over the legal actions and the value for the player to move.

    Here: a uniform prior and a random playout of up to PLAYOUT_PLIES moves (worth 0 if the
    game does not end). A network would return softmax(logits[legal]) and its value head.
    """
    legal = np.flatnonzero(info["legal_action_mask"])
    priors = {int(a): 1.0 / len(legal) for a in legal}
    playout, sign = env.clone(), 1.0
    for _ in range(PLAYOUT_PLIES):
        result = playout.step(int(rng.choice(np.flatnonzero(info["legal_action_mask"]))))
        if result.done:
            return priors, sign * float(result.reward)  # the reward is for the last mover
        sign, info = -sign, result.info
    return priors, 0.0


def search(env: Any, info: dict[str, Any], simulations: int, rng: np.random.Generator) -> int:
    root = Node(prior=1.0)
    priors, _ = evaluate(env, info, rng)
    root.children = {a: Node(prior=p) for a, p in priors.items()}
    for _ in range(simulations):
        node, path, branch = root, [root], env.clone()
        value = 0.0  # for the player who moved into path[-1]
        while node.children:
            total = math.sqrt(max(node.visits, 1))
            action, node = max(
                node.children.items(),
                key=lambda kv: kv[1].q + C_PUCT * kv[1].prior * total / (1 + kv[1].visits),
            )
            path.append(node)
            if node.terminal is not None:
                value = node.terminal
                break
            result = branch.step(action)
            if result.done:
                node.terminal = value = float(result.reward)
                break
            priors, leaf_value = evaluate(branch, result.info, rng)
            node.children = {a: Node(prior=p) for a, p in priors.items()}
            value = -leaf_value  # the leaf's value is for the player to move there
            break
        for visited in reversed(path):  # negamax backup
            visited.visits += 1
            visited.value_sum += value
            value = -value
    return max(root.children, key=lambda a: root.children[a].visits)


rng = np.random.default_rng(0)

# Chess: after 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6, White mates with Qxf7#, one of 43 moves.
env = sekai.make("Chess-v0")
fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
result = env.reset(options={"fen": fen})
action = search(env.unwrapped, result.info, simulations=200, rng=rng)
print("White plays", env.unwrapped.board.san(env.unwrapped.decode_action(action)))

# Go on a 5x5 board: the search plays Black, a random player White.
env = GoEnv(size=5)
result = env.reset(seed=0)
while not result.done:
    if result.info["to_play"] == "black":
        action = search(env, result.info, simulations=300, rng=rng)
    else:
        action = int(rng.choice(np.flatnonzero(result.info["legal_action_mask"])))
    result = env.step(action)
print(env.board.pretty())
print("Result:", result.info["result"])  # Black wins, komi 7.5 notwithstanding
