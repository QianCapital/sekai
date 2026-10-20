# Multi-agent environments

`sekai.MultiAgentEnv[ObsType, ActType, AgentID]` is a built-in base class for environments
where each agent has its own observation and action space, e.g. a market maker and a
directional trader, or a portfolio manager and a risk manager. Every dictionary in the API is
keyed by agent ID. Agents act simultaneously, or, in an
[event-driven](#event-driven-environments) env, each on its own clock.

| Member | Notes |
|---|---|
| `possible_agents` | every agent that can ever take part |
| `agents` | the agents currently active (may shrink as agents finish) |
| `observation_spaces`, `action_spaces` | `dict[agent, Space]` |
| `observation_space(agent)`, `action_space(agent)` | lookups into those dicts |
| `num_agents`, `max_num_agents` | `len(agents)`, `len(possible_agents)` |
| `reset(*, seed=None, options=None)` | **abstract**; returns `MAResetResult(observations, info)` |
| `step(actions)` | **abstract**; `actions` is `{agent: action}`; returns `MAStepResult` |
| `async_reset()`, `async_step()` | default to the sync methods, as for `Env` |
| `close()`, `with env:` | as for `Env` |

`MAStepResult` holds `observations`, `rewards`, `terminated`, `truncated` and `info`, each a
dict keyed by agent, plus two properties: `all_done` (every agent is terminated or truncated)
and `any_done`. They are evaluated per agent ID over the agents in `terminated` or `truncated`;
an agent missing from one of the two dicts counts as `False` there, and with no agents in either
both properties are `False`. `MAResetResult` has `observations` and `info`, and `all_done` / `any_done`
are always `False`, so one loop handles both, like the single-agent results.

## Example: a two-agent market

A market maker chooses its half-spread; a trader sees the quotes and decides whether to buy,
sell or pass. The trader pays the spread and both mark to the next mid price.

```python
import numpy as np
import sekai


class QuoteAndTradeEnv(sekai.MultiAgentEnv[np.ndarray, object, str]):
    def __init__(self, n_steps: int = 50):
        self.n_steps = n_steps
        self.possible_agents = ["maker", "trader"]
        self.agents = list(self.possible_agents)
        self.observation_spaces = {
            "maker": sekai.Box(-np.inf, np.inf, shape=(2,)),   # mid, inventory
            "trader": sekai.Box(-np.inf, np.inf, shape=(3,)),  # mid, bid, ask
        }
        self.action_spaces = {
            "maker": sekai.Box(0.01, 1.0, shape=(1,)),        # half-spread
            "trader": sekai.Discrete(3, start=-1),             # sell / pass / buy
        }

    def reset(self, *, seed=None, options=None):
        self._set_rng(seed)  # same seeding contract as sekai.Env
        self.agents = list(self.possible_agents)
        self.mid, self.inventory, self.t = 100.0, 0, 0
        return sekai.MAResetResult(self._obs(0.1), {a: {} for a in self.agents})

    def step(self, actions):
        half = float(actions["maker"][0])
        side = int(actions["trader"])
        next_mid = self.mid + self.rng.normal(0.0, 0.3)
        trader_pnl = side * (next_mid - self.mid) - abs(side) * half
        self.inventory -= side
        maker_pnl = abs(side) * half + self.inventory * (next_mid - self.mid)
        self.mid, self.t = next_mid, self.t + 1
        done = self.t >= self.n_steps
        result = sekai.MAStepResult(
            observations=self._obs(half),
            rewards={"maker": maker_pnl, "trader": trader_pnl},
            terminated={a: done for a in self.agents},
            truncated={a: False for a in self.agents},
            info={a: {} for a in self.agents},
        )
        if done:
            self.agents = []  # nobody acts after the last step
        return result

    def _obs(self, half):
        return {
            "maker": np.array([self.mid, self.inventory], np.float32),
            "trader": np.array([self.mid, self.mid - half, self.mid + half], np.float32),
        }
```

Running it with random policies:

```python
env = QuoteAndTradeEnv()
print(env, env.num_agents)  # <QuoteAndTradeEnv agents=['maker', 'trader']> 2
print(env.action_space("trader"))  # Discrete(3, start=-1)

for agent in env.possible_agents:
    env.action_space(agent).seed(0)

result = env.reset(seed=0)
totals = dict.fromkeys(env.agents, 0.0)
while not result.all_done:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    result = env.step(actions)
    for agent, reward in result.rewards.items():
        totals[agent] += reward
print({agent: round(total, 2) for agent, total in totals.items()})  # {'maker': 25.88, 'trader': -19.65}
```

!!! tip "Seeding"
    `MultiAgentEnv` has the same seeding API as `sekai.Env`: call `self._set_rng(seed)` at the
    start of `reset()` and draw randomness from `self.rng`. An int seed re-seeds, `None`
    continues the current stream.

## Agents that leave early

When an agent is done, remove it from `self.agents` and stop expecting its action. Keep
reporting it in the result dicts on the step it finishes, so learners see its final
transition; `all_done` becomes `True` once every agent reported in the result is done. When
the episode ends, `self.agents` is empty, as in the example above.

## Event-driven environments

In many markets agents do not move in lockstep: a market maker requotes every second, a
slower trader decides every minute, an execution algorithm wakes up on fills. sekai's
convention for such environments:

- `observations` holds only the agents that act next; its keys say who moves;
- `step()` takes actions from exactly those agents, and checks them all before carrying out
  any, so a step that raises (an invalid action, a missing agent) leaves the env unchanged;
- `rewards`, `terminated`, `truncated` and `info` cover every agent, so a reward earned
  between an agent's decisions is not lost.

A learner adds up each agent's rewards until that agent's next observation, which closes one
transition. Steps then have different lengths for different agents; discount by elapsed time
(`info[agent]["timestamp"]`), as described in
[Time and history](time.md#variable-step-lengths).

`sekai.envs.finance.LimitOrderBookMarket` is the built-in example: several traders in one
limit order book, trader `i` deciding every `decision_intervals[i]` units of session time
(the [finance documentation](../finance/index.md) covers the market itself). One trader's
market order can fill another's resting quote but never its own (self-trade prevention:
it first cancels the trader's quote on the opposite side). With intervals of 1, 2 and 5,
all three act first, then only the fast one:

```python
from sekai.envs.finance import LimitOrderBookMarket

env = LimitOrderBookMarket(3, decision_intervals=[1.0, 2.0, 5.0], session_length=60.0)
for agent in env.possible_agents:
    env.action_space(agent).seed(0)

result = env.reset(seed=0)
print(sorted(result.observations))  # ['trader_0', 'trader_1', 'trader_2']
result = env.step({agent: env.action_space(agent).sample() for agent in result.observations})
print(sorted(result.observations), sorted(result.rewards))  # ['trader_0'] ['trader_0', 'trader_1', 'trader_2']
```

The loop takes its actions from `result.observations` and its rewards from `result.rewards`:

```python
result = env.reset(seed=0)
pending = dict.fromkeys(env.possible_agents, 0.0)  # reward since each agent's last decision
decisions = dict.fromkeys(env.possible_agents, 0)
while not result.all_done:
    actions = {
        agent: env.action_space(agent).sample(mask=env.action_mask(agent))
        for agent in result.observations  # only the agents whose turn it is
    }
    result = env.step(actions)
    for agent, reward in result.rewards.items():  # every agent
        pending[agent] += reward
    for agent in result.observations:  # these agents decide next
        # a learner stores (previous observation, action, pending[agent], new observation) here
        decisions[agent] += 1
        pending[agent] = 0.0
print(decisions)  # {'trader_0': 60, 'trader_1': 30, 'trader_2': 12}
```

On the last step every agent receives a final observation; `env.agents` is then empty, and
another `step()` raises `ResetRequired` until the next `reset()`.

## Action masks

The [action-mask convention](dynamic.md#action-masks) carries over with the agent as an
argument: `action_mask(agent)` returns that agent's mask, in the format
`action_space(agent).sample(mask=...)` accepts, as in the loop above. `sekai.get_action_mask()`
and `ActionMaskObservation` are single-agent tools; call the method directly.

```python
print(env.action_space("trader_0"))  # MultiDiscrete([6, 6, 3])
print([mask.shape for mask in env.action_mask("trader_0")])  # [(6,), (6,), (3,)]
```

## What is single-agent only

The wrappers, vector environments and `check_env` operate on `sekai.Env`. A `MultiAgentEnv`
can be registered and built with `sekai.make()`, but register it **without**
`max_episode_steps`, because `TimeLimit` is a single-agent wrapper:

```python
sekai.register("ma/QuoteAndTrade-v0", QuoteAndTradeEnv)
env = sekai.make("ma/QuoteAndTrade-v0", n_steps=10)
print(env.spec, env.n_steps)  # EnvSpec(id='ma/QuoteAndTrade-v0') 10
```

To train a multi-agent environment with single-agent tooling, expose one agent's view
through a small `sekai.Env` adapter that plays the other agents with fixed policies, or use a
multi-agent library through PettingZoo.

## PettingZoo

`sekai.compat.to_pettingzoo(env)` exposes a `MultiAgentEnv` as a PettingZoo `ParallelEnv`:
spaces become gymnasium spaces and results the usual `(observations, rewards, terminations,
truncations, infos)` dicts. `from_pettingzoo(parallel_env)` goes the other way, so any
PettingZoo parallel environment can be driven through the sekai API. Both need the optional
dependency:

```bash
pip install "sekai[pettingzoo]"
```

```python
from pettingzoo.test import parallel_api_test
from sekai.compat import from_pettingzoo, to_pettingzoo

pz_env = to_pettingzoo(QuoteAndTradeEnv())
parallel_api_test(pz_env, num_cycles=100)  # passes
observations, infos = pz_env.reset(seed=0)
print(sorted(observations), pz_env.action_space("trader"))  # ['maker', 'trader'] Discrete(3, start=-1)

env = from_pettingzoo(pz_env)  # any PettingZoo ParallelEnv
print(sorted(env.reset(seed=0).observations), env.action_space("maker"))  # ['maker', 'trader'] Box(0.01, 1.0, (1,), float32)
```

The PettingZoo env keeps its own randomness; seeds passed to `reset()` go straight to it.
An event-driven env keeps its convention through the bridge: observations only for the
agents that act next, rewards and flags for all. That passes PettingZoo's parallel API test,
but a consumer that expects an observation for every live agent at every step needs to
handle it.
