# Self-play and search

The [strategy games](../environments/games.md) are built so that the methods that mastered
them work on them directly: self-play, tree search guided by a network (AlphaZero), and
policy-gradient training with masked actions. This guide shows how each one uses the
environments.

## Self-play

A two-player game env is played from both sides by the caller. Each observation is from
the point of view of the player to act, so one network plays both sides, and each reward
is for the player who just moved. A self-play game records, for every position, the
observation, the legal actions and the player to move; at the end each position's value
target is the final reward seen from its player's side:

```python
import numpy as np
import sekai

env = sekai.make("Chess-v0")


def self_play(policy, seed):
    """One game: [(observation, mask, target value)] for every position played."""
    result, positions, players = env.reset(seed=seed), [], []
    while not result.done:
        mask = result.info["legal_action_mask"]
        positions.append((result.observation, mask))
        players.append(result.info["to_play"])
        result = env.step(policy(result.observation, mask))
    mover = players[-1]  # the reward is for whoever made the last move
    z = result.reward if result.terminated else 0.0  # a truncated game is a draw
    return [
        (obs, mask, z if player == mover else -z)
        for (obs, mask), player in zip(positions, players)
    ]
```

A policy samples only legal actions by masking its logits:

```python
logits = np.where(mask.astype(bool), logits, -np.inf)
probabilities = np.exp(logits - logits.max())
action = rng.choice(len(probabilities), p=probabilities / probabilities.sum())
```

## Tree search

`clone()` gives an independent copy of a board game to try moves on, and terminal rewards
for the mover make the backup a negamax: the value of a node for the player who moved
into it is minus the value for the player to move there. That is all PUCT search (the
search of AlphaZero) needs. A network supplies a prior over the legal moves and a value
for the player to move at each new leaf:

```python
def evaluate(env, observation, info):
    logits, value = network(observation[None])         # (1, n_actions), (1,)
    legal = np.flatnonzero(info["legal_action_mask"])
    priors = softmax(logits[0, legal])
    return dict(zip(legal.tolist(), priors)), float(value[0])
```

[`examples/games_search.py`](https://github.com/QianCapital/sekai/blob/master/examples/games_search.py)
is a complete PUCT search over any of the board games, with a uniform prior and random
playouts in place of the network: it finds a mate in one in chess and beats a random
player at 5×5 go. Clones are cheap: the chess env clones and steps in about 50 µs
(`clone(full_stack=False)` keeps only the moves repetitions need), and
the go and xiangqi engines also offer bare boards (`env.board`, `env.position`) for search
code that wants to skip the observations.

The card games have no `clone()`: their state includes cards that a player may not see.
Search in poker works on information sets instead, for example by sampling the hidden
cards consistently with what a seat has seen (`HoldemHand.from_view` does).

## Many games at once

The game envs vectorise like any other, with autoreset. A game that ends is reset in the
same step: its `info` describes the game that ended (with an all-zero mask), and the new
game's is in `info["reset_info"]`, so take the next masks from there:

```python
vec = sekai.make_vec("Go9x9-v0", num_envs=64, vectorization_mode="proc")
batch = vec.reset(seed=0)
masks = np.stack([info["legal_action_mask"] for info in batch.infos])
for _ in range(1000):
    actions = [np.random.choice(np.flatnonzero(mask)) for mask in masks]  # (or a network's)
    batch = vec.step(np.array(actions))
    masks = np.stack([info.get("reset_info", info)["legal_action_mask"] for info in batch.infos])
```

(`vec.call("action_mask")` returns the same masks, from the envs themselves.)

Batching the network's evaluations across many games in flight (for search: across the
roots of many searches) is where most of the speed of self-play comes from on a GPU.

## Against fixed opponents

`OpponentWrapper` plays one side with a fixed policy, which turns a game into an ordinary
single-agent environment, for training against a baseline or for evaluating one:

```python
from sekai.envs.games import OpponentWrapper, RandomPolicy

env = OpponentWrapper(sekai.make("HeadsUpHoldem-v0"), RandomPolicy(seed=0), agent_color="random")
```

## Masked policy gradients

The gymnasium bridge exposes the mask as `action_masks()`, so sb3-contrib's MaskablePPO
trains directly on blackjack (one player against the dealer), or on a two-player game
against a fixed opponent:

```python
from sb3_contrib import MaskablePPO
from sekai.compat import to_gymnasium

model = MaskablePPO("MlpPolicy", to_gymnasium(sekai.make("Blackjack-v0"))).learn(200_000)

poker = to_gymnasium(OpponentWrapper(sekai.make("HeadsUpHoldem-v0"), agent_color="random"))
model = MaskablePPO("MlpPolicy", poker).learn(200_000)
```

## Deploying a trained network

The encodings are the standard ones (AlphaZero's for chess, AlphaGo Zero's for go), so a
network that maps `observation[None]` to `(logits over all actions, value)` plays from any
position of the environment, in a search or alone, and can be saved with TorchScript for
any program that builds the same observations.
