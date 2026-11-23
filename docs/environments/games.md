# Strategy Games

Five classic games of strategy, built for reinforcement learning the way AlphaZero learned
its games: chess, go, xiangqi (Chinese chess), heads-up no-limit Texas hold'em and
blackjack. Each is a standard `sekai.Env`, so wrappers, vector envs and the gymnasium
bridge apply as usual, and each comes with a rules engine you can use on its own, the
notation its players use, and RGB and text renderers.

| ID | Game | Observation | Actions | Players | Episode |
|---|---|---|---|---|---|
| [`Chess-v0`](#chess) | chess | `Box(0, 1, (119, 8, 8))` | `Discrete(4672)` | two (self-play) | a game; truncated at 512 plies |
| [`ChessVsRandom-v0`](#playing-against-an-opponent) | chess against a random opponent | `Box(0, 1, (119, 8, 8))` | `Discrete(4672)` | one | a game; truncated at 512 plies |
| [`Go9x9-v0`, `Go13x13-v0`, `Go19x19-v0`](#go) | go on an N×N board | `Box(0, 1, (17, N, N))` | `Discrete(N² + 1)` | two (self-play) | a game: two passes, or 3N² moves |
| [`Xiangqi-v0`](#xiangqi) | xiangqi | `Box(0, 1, (131, 10, 9))` | `Discrete(8100)` | two (self-play) | a game; truncated at 400 plies |
| [`HeadsUpHoldem-v0`](#heads-up-no-limit-holdem) | heads-up no-limit hold'em | `Box(125,)` | `Discrete(7)` | two (self-play) | one hand |
| [`Blackjack-v0`](#blackjack) | blackjack against the dealer | `Box(43,)` | `Discrete(9)` | one | one round; the shoe carries over |

Observations are `float32`. The environments live in `sekai.envs.games` (one subpackage
per game: `chess`, `go`, `xiangqi`, `poker`, `blackjack`), which is imported the first
time one of them is made, so `import sekai` stays as fast as before. None needs anything
but NumPy: the chess rules are sekai's own (they match python-chess's exactly), and the
renderers draw with NumPy.

## How the two-player games work

Chess, go, xiangqi and hold'em follow the conventions of AlphaZero, which let one network
play both sides and let tree search work on the environment directly:

- **The caller plays both sides.** Each `step()` is a move by the player to act
  (`info["to_play"]`), and the observation is always from that player's point of view:
  the board is turned (chess, xiangqi) or the planes are split into "mine" and "theirs"
  (go), and in poker it holds only what that seat may know.
- **Only legal moves.** `info["legal_action_mask"]` is an `int8` mask of the legal
  actions (also the env's `action_mask()`, sekai's
  [action-mask convention](../guide/training.md#action-masks-maskableppo), which
  `sekai.check_env`, `action_space.sample(mask=...)` and MaskablePPO understand; through
  wrappers such as `TimeLimit`, `sekai.get_action_mask(env)` finds it), and
  `info["legal_actions"]` lists them. An illegal action raises `IllegalMove` (chess and
  xiangqi can instead forfeit the game with `illegal_action="lose"`), so checkers that
  step with unmasked random actions, such as gymnasium's and Stable-Baselines3's
  `check_env`, stop at the first illegal one: check the games with `sekai.check_env` (or
  `sekai check`), which samples within the mask.
- **A reset can start where the game is over.** A `fen`, or `moves` replayed, may reach
  the end of a game (mate, stalemate, two passes): the reset returns that position with
  no legal moves, so check `env.done` (or `info["termination"]`) before stepping.
- **Rewards at the end, for the player who just moved:** `+1` for a win, `-1` for a loss,
  `0` for a draw (in poker, the net chips won, in big blinds), and `0` on every other
  step. From the other player's side the reward is the negative: the games are zero-sum.
- **Search from any position.** Chess, go and xiangqi have `env.clone()`, an independent
  copy to try moves on; every environment also has sekai's `snapshot()` / `restore()`.

```python
import sekai

env = sekai.make("Go9x9-v0")
result = env.reset(seed=0)
while not result.done:
    action = env.action_space.sample(mask=result.info["legal_action_mask"])
    result = env.step(action)  # the next player to move sees the next observation
print(result.info["result"], result.reward)  # e.g. "W+7.5", the reward of the last mover
```

A value network trained on these games learns the value for the player to move; a tree
search backs values up with a change of sign at each ply (negamax).
[`examples/games_search.py`](https://github.com/QianCapital/sekai/blob/master/examples/games_search.py)
is a complete PUCT search over any of the board games; the
[self-play guide](../guide/games.md) covers training.

### Playing against an opponent

`OpponentWrapper` turns a two-player game into a single-agent environment: a fixed policy
plays the other side, replying until it is the agent's turn again, and rewards are the
agent's (`+1` for a win, `-1` for a loss, its net in poker).

```python
from sekai.envs.games import OpponentWrapper, RandomPolicy

env = OpponentWrapper(sekai.make("Xiangqi-v0"), RandomPolicy(seed=0), agent_color="random")
result = env.reset(seed=1)
print(result.info["agent_color"])  # "red" or "black", drawn at every reset
```

`agent_color` names a side (`"white"`, `"black"`, `"red"`, `"p1"`, `"p2"`), `"first"`,
`"second"`, or `"random"`. The opponent is anything with `act(observation, info)`;
`info` gains `agent_color`, `opponent_action` and `opponent_move`. A game the opponent
ends before the agent has moved (a fold from the small blind) is dealt again, with the
agent's rewards from the games skipped in `info["skipped_rewards"]` for evaluations to
count. `ChessVsRandom-v0` is `Chess-v0` in this wrapper with a random opponent and a
random colour.

### Speed

Time per step of random legal play on one core (lower is better), for the environments
without wrappers, and the time to clone one and step the clone, as a tree search does:

| Environment | Step | Clone + step |
|---|---|---|
| `Chess-v0` | 38 µs | 49 µs |
| `Go9x9-v0` | 13 µs | 19 µs |
| `Go13x13-v0` | 16 µs | 24 µs |
| `Go19x19-v0` | 21 µs | 33 µs |
| `Xiangqi-v0` | 30 µs | 41 µs |
| `HeadsUpHoldem-v0` | 19 µs | |
| `Blackjack-v0` | 22 µs | |

For chess that is `clone(full_stack=False)`; a full clone copies the whole move stack.

## Chess

`sekai.envs.games.chess.ChessEnv`: standard chess, with the encodings of AlphaZero.

### Observation

`Box(0, 1, (14T + 7, 8, 8), float32)`, channel first and indexed `[plane, rank, file]` in
the side to move's frame: for Black the board is mirrored vertically and the colours
swapped, so "our" pieces always start at the bottom. `T` is `history_length` (default
8, 119 planes; 1 gives 21). For each of the last `T` positions, the current one first
(missing history is zero):

| Plane | Meaning |
|---|---|
| 0-5 | our pawns, knights, bishops, rooks, queens, king |
| 6-11 | their pawns, knights, bishops, rooks, queens, king |
| 12 | 1 if that position had occurred at least twice |
| 13 | 1 if that position had occurred at least three times |

then 7 constant planes: the colour to move (1 for White), the move number / 256 (at
most 1), our kingside and queenside castling rights, theirs, and the half-move clock /
100 (the fifty-move rule's progress).

### Actions

`Discrete(4672)`: `action = from_square * 73 + plane`, the from-square in the mover's
frame. Planes 0-55 are queen-like moves (8 directions × 7 distances), 56-63 knight moves
and 64-72 underpromotions (capture left, push, capture right × knight, bishop, rook);
promoting to a queen is the plain pawn move. Only 1858 of the 4672 actions can ever be
legal. `env.encode_move()` takes a `Move`, a UCI string (`"e2e4"`) or SAN (`"Nf3"`), and
`env.decode_action()` gives the move of an action.

### Rewards and termination

`+1` to the player who mates, `-1` for an illegal move with `illegal_action="lose"`, `0`
otherwise. Games end (`terminated`) on checkmate, stalemate, insufficient material, the
seventy-five-move rule and fivefold repetition, and with `claim_draw=True` (the default,
as in AlphaZero) on threefold repetition and the fifty-move rule. `Chess-v0` truncates at
512 plies.

### Info

| Key | Value |
|---|---|
| `legal_action_mask`, `legal_actions` | the legal actions (`int8` mask, sorted `int64` indices); empty once the game is over |
| `to_play` | the side to move: `True` for White, `False` for Black |
| `fen` | the position |
| `last_move` | the last move in UCI, or `None` |
| `ply` | half-moves played since `reset()` |
| `is_check` | whether the side to move is in check |
| `result` | `"*"` while playing, then `"1-0"`, `"0-1"` or `"1/2-1/2"` |
| `winner`, `termination` | once over: `True`/`False`/`None`, and `"CHECKMATE"`, `"STALEMATE"`, `"INSUFFICIENT_MATERIAL"`, `"SEVENTYFIVE_MOVES"`, `"FIVEFOLD_REPETITION"`, `"FIFTY_MOVES"`, `"THREEFOLD_REPETITION"` or `"ILLEGAL_MOVE"` |

### Options

```python
ChessEnv(
    history_length=8,        # positions stacked in the observation
    illegal_action="raise",  # or "lose": forfeit the game instead
    claim_draw=True,         # threefold repetition and the fifty-move rule end the game
    starting_fen=None,       # the standard position, or a FEN
)
env.reset(options={"fen": "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"})
```

### The rules engine

`sekai.envs.games.chess.Board` is a dependency-free chess engine: `legal_moves`,
`push()` / `pop()`, `fen()`, `san()` / `parse_san()`, `is_check()`, `outcome` checks and
`unicode()`. It reproduces python-chess's rules where implementations can differ (FEN
validation, castling rights, which en passant squares a FEN shows, insufficient material,
SAN) and matches the published perft counts. `to_pgn()` and `env_pgn(env)` write games as
PGN.

## Go

`sekai.envs.games.go.GoEnv(size)`: go on a square board of size 2 to 19, with area
(Tromp-Taylor) scoring, komi, handicaps and positional superko. `Go9x9-v0`, `Go13x13-v0`
and `Go19x19-v0` fix the size.

### Observation

`Box(0, 1, (2T + 1, N, N), float32)` with `T = 8`: 17 planes, AlphaGo Zero's input,
indexed `[plane, row, column]` from the top-left. Planes `0..T-1` hold the stones of the
player to move in the current and previous positions, `T..2T-1` the opponent's, and the
last plane is all ones when Black is to move. The board is not turned; the planes are
split by player instead.

### Actions

`Discrete(N² + 1)`: `row * N + column` from the top-left, and `N²` to pass (always legal
while the game is on). `env.encode_vertex("D4")` and `env.decode_action()` convert from
and to GTP vertices (columns `A`-`T` without `I`, rows counted from the bottom).

### Rules, rewards and termination

Captures, then suicide is illegal; positional superko (no earlier stone configuration may
recur; `superko=False` keeps only simple ko). Two consecutive passes end the game
(`terminated`), and `max_moves` plies (default `3 N²`) cut it short (`truncated`); either
way it is scored by Tromp-Taylor area scoring with `komi` (default 7.5) for White, and the
reward is `+1`/`-1`/`0` for the player who just moved. No dead stones are removed inside
the environment, as in zero-knowledge training; `scoring.final_score(env)` estimates
them with playouts for people.

### Info

`legal_action_mask`, `legal_actions`, `to_play` (`"black"` or `"white"`), `last_move`
(a vertex, `"pass"` or `None`), `ply`, `captures` (stones captured by each colour),
`passes` (consecutive), `komi`, `result` (`"*"`, then `"B+3.5"`, `"W+0.5"` or `"Draw"`)
and `winner`; once over also `score` (each side's points) and `termination`
(`"two passes"` or `"max moves"`).

### Options

```python
GoEnv(size=9, komi=7.5, history_length=8, max_moves=None, superko=True)
env.reset(options={"komi": 6.5, "handicap": 4})        # 2-9 stones on the star points
env.reset(options={"moves": ["D4", "F6", "pass"]})      # replay moves (vertices or actions)
env.reset(options={"board": board_string, "to_play": "white"})  # set up a position
```

`sekai.envs.games.go` also has the engine (`Board`, with fast random playouts),
`scoring` (area scores, ownership and dead-stone estimates) and `to_sgf()` for game
records.

## Xiangqi

`sekai.envs.games.xiangqi.XiangqiEnv`: Chinese chess with the full rules (the palace, the
river, blocked horses and elephants, cannon screens, the flying generals).

### Observation

`Box(0, 1, (16T + 3, 10, 9), float32)` with `T = 8`: 131 planes, indexed
`[plane, rank, file]` in the mover's frame (for Black the board is turned 180° and the
colours swapped). For each of the last `T` positions: our general, advisors, elephants,
horses, chariots, cannons and soldiers (planes 0-6), theirs (7-13), and two repetition
planes; then 3 constant planes: Red to move, the game's length in plies / 400, and plies
without a capture / 120.

### Actions

`Discrete(8100)`: `action = from_square * 90 + to_square` in the mover's frame
(`square = rank * 9 + file`, rank 0 the mover's back rank). `env.encode_move("h2e2")` and
`env.decode_action()` convert from and to UCCI coordinates; `env.move_wxf()` and
`env.move_chinese()` give the WXF (`C2=5`) and traditional Chinese (炮二平五) notation.

### Rules, rewards and termination

Having no legal move loses, in check or not. A position repeated three times ends the
game: if one side gave check with every one of its moves since the first occurrence and
the other side did not, the side that checked loses (perpetual check); otherwise it is a
draw. 120 plies without a capture and bare material (no chariot, horse, cannon or
soldier on either side) are draws. Rewards are `+1`/`-1`/`0` for the player who just moved;
`Xiangqi-v0` truncates at 400 plies.

### Info

`legal_action_mask`, `legal_actions`, `to_play` (`"red"` or `"black"`), `fen`,
`last_move` (UCCI), `ply`, `is_check`, `result` (`"1-0"` is a Red win) and `winner`
(`"red"`, `"black"` or `None`), and `termination` (`"CHECKMATE"`, `"STALEMATE"`,
`"PERPETUAL_CHECK"`, `"REPETITION"`, `"NO_CAPTURE"`, `"INSUFFICIENT_MATERIAL"` or
`"ILLEGAL_MOVE"`).

### Options

```python
XiangqiEnv(history_length=8, starting_fen=None, illegal_action="raise")
env.reset(options={"fen": fen, "moves": ["h2e2", "h9g7"]})  # a position, then moves to replay
```

## Heads-up no-limit hold'em

`sekai.envs.games.poker.HoldemEnv`: one hand of heads-up no-limit Texas hold'em per
episode, with blinds of 50/100 and stacks of 10,000 by default. The button posts the small
blind and acts first before the flop; minimum raises, all-ins for less that do not reopen
the betting, and uncalled bets returned follow the rules of the game.

### Observation

`Box(125,)`, built from the view of the seat to act (at the end of the hand, of the seat
that just acted) only, so it never contains the opponent's hidden cards: the seat's hole
cards and the board (one-hot over the 52 cards each), the street, whether the seat has the
button, the pot, both stacks and street bets, the amount to call, the minimum raise and
the big blind (as fractions of the larger starting stack), and how many bets and raises
each seat made on each street. `history_length=k` appends the last `k` actions, 6 slots
each (`Box(125 + 6k,)`).

### Actions

`Discrete(7)`: fold, check or call, a minimum raise, raises of half the pot, the pot and
two pots, and all-in. Sizes that are not legal (or would repeat a smaller one) are masked.

### Rewards

`0` until the hand ends, then the net chips won by the seat that just acted, in big
blinds; `info["payoffs"]` holds both seats'. From the other seat's side the reward is the
negative.

### Info

`legal_action_mask`, `legal_actions`, `to_play` (the seat to act, `None` once the hand
is over), `observer` (whose view the observation is), `street`, `payoffs`, `view` (the
observer's `SeatView`: everything that seat may know), `last_action` and `result`.

### Options

```python
HoldemEnv(small_blind=50, big_blind=100, stack=10_000, history_length=0)
env.reset(options={"button": 0, "stacks": (5_000, 15_000)})  # seat 0 on the button
env.reset(options={"deck": "AsKs QhQd 2c7d9h Th 3s"})  # seat 0, seat 1, flop, turn, river
```

Without `button` the button is drawn at random for each hand, and without `deck` the cards
are shuffled; stacks must be above the big blind.

The engine, `HoldemHand`, is a trusted dealer: give agents its `view(seat)`, never the
hand. `sekai.envs.games.poker` also has a 5-to-7-card hand evaluator and PokerStars-style
hand histories (`format_hand_history`).

## Blackjack

`sekai.envs.games.blackjack.BlackjackEnv`: blackjack as a casino deals it, one round per
episode, with six decks, dealer stands on all 17s, 3:2 naturals, double after split,
resplits to four hands, late surrender and the US hole-card peek by default (all set by
`Rules`). The shoe carries over from one episode to the next and is reshuffled at the
penetration, as at a real table, so card counting can be learned.

### Observation

`Box(43,)` of public information only: the phase (betting or playing), the hand's total,
softness, pair and size, which of double, split and surrender are allowed, the dealer's up
card, the hand's index among split hands, the Hi-Lo true count, how much of the shoe has
been dealt, the composition of the cards not yet seen, and the bankroll.

### Actions

`Discrete(9)`: stand, hit, double, split, surrender, and four bet sizes (1, 2, 4 and 8
times the table minimum, set by `bet_levels`) for the betting phase.

### Rewards

`0` until the round is settled, then the round's net in units of the table minimum (a
natural at the minimum bet is `+1.5`).

### Info

`legal_action_mask`, `legal_actions`, `to_play` (always 0), `phase`, `view` (the public
table), `round_no` and `bankroll`; `rebuy` after a reset, and once the round is settled
`net` and `round` (the round's summary).

### Options

```python
BlackjackEnv(rules=Rules(), bankroll=1000, bet_levels=(1, 2, 4, 8))
env.reset(options={"keep_bankroll": True})  # or {"bankroll": 500}, {"new_shoe": True}
```

`reset()` deals the next round from the same shoe, with the bankroll back at its starting
value unless the options say otherwise: `keep_bankroll` carries it over (rebuying, and
setting `info["rebuy"]`, when it falls below the smallest bet), `bankroll` sets it, and
`new_shoe` (as does a seed) starts a fresh shoe. A round still in play when `reset()` is
called (when a `TimeLimit` stops an episode, say) is finished first, standing on its
remaining hands, so the shoe and the count go on as at a table; with `keep_bankroll`, the
bankroll carries that round's result. `Rules` sets the decks, the dealer's
soft 17, the payout of a natural, doubling, splitting, surrender, the peek, the
penetration and the table limits; `sekai.envs.games.blackjack` also has the multi-seat
`BlackjackTable` that the environment deals from.

## Rendering

Every game has an RGB renderer and a text renderer:

| Game | RGB | Text |
|---|---|---|
| Chess | `ChessRenderer` | `ChessTextRenderer` |
| Go | `GoRenderer` | `GoTextRenderer` |
| Xiangqi | `XiangqiRenderer` | `XiangqiTextRenderer` |
| Hold'em | `HoldemRenderer` | `HoldemTextRenderer` |
| Blackjack | `BlackjackRenderer` | `BlackjackTextRenderer` |

```python
from sekai.envs.games import ChessRenderer
from sekai.rendering import RecordFrames

env = RecordFrames(sekai.make("Chess-v0"), ChessRenderer(orientation="mover"))
```

They take a `theme` (`"dark"`, the default, or `"light"`), `header=False` to leave out the
game's name and status, and `names` to name the players. See
[Rendering](../guide/rendering.md#strategy-games) for each one's options.
