"""Market making in a simulated limit order book, alone and against other agents.

LimitOrderBook-v0 runs a price-time priority matching engine with Poisson
background order flow. A quote left at the same price keeps its queue position.
We compare three simple quoting rules, then run an event-driven multi-agent
market in which traders decide on different clocks.
"""

import numpy as np

import sekai
from sekai.envs.finance import LimitOrderBookMarket

NO_QUOTE, JOIN_BEST = 0, 1
SELL, HOLD, BUY = 0, 1, 2


def skewed(env, obs) -> np.ndarray:
    """Join both best quotes, but back off the side that would grow the inventory."""
    inventory = obs["agent"][0]  # inventory / max_inventory
    bid = JOIN_BEST + (2 if inventory > 0.25 else 0)
    ask = JOIN_BEST + (2 if inventory < -0.25 else 0)
    return np.array([bid, ask, HOLD])


policies = {
    "no quotes": lambda env, obs: np.array([NO_QUOTE, NO_QUOTE, HOLD]),
    "join both sides": lambda env, obs: np.array([JOIN_BEST, JOIN_BEST, HOLD]),
    "inventory-skewed": skewed,
}
env = sekai.make("LimitOrderBook-v0", session_length=200)
print("observation:", env.observation_space)
for name, policy in policies.items():
    totals, fills = [], []
    for seed in range(10):
        result, total, n_fills = env.reset(seed=seed), 0.0, 0
        while not result.done:
            result = env.step(policy(env, result.observation))
            total += result.reward
            n_fills += result.info["fills"]
        totals.append(total)
        fills.append(n_fills)
    mean, sd = np.mean(totals), np.std(totals)
    print(f"{name:17s} reward {mean:+8.1f} ± {sd:6.1f}   fills {np.mean(fills):6.1f}")

# Three traders sharing one book: decisions every 1, 2.5 and 0.5 time units.
market = LimitOrderBookMarket(3, decision_intervals=[1.0, 2.5, 0.5], session_length=100)
result = market.reset(seed=0)
market.action_space("trader_0").seed(0)  # all traders share one action space object
turns = dict.fromkeys(market.possible_agents, 0)
totals = dict.fromkeys(market.possible_agents, 0.0)
while True:
    actions = {}
    for agent in result.observations:  # only the traders whose turn it is
        actions[agent] = market.action_space(agent).sample(mask=market.action_mask(agent))
        turns[agent] += 1
    result = market.step(actions)
    for agent, reward in result.rewards.items():  # rewards accrue to every trader
        totals[agent] += reward
    if result.all_done:
        break
print("decisions per trader:", turns)
print("random traders' P&L:", {agent: round(total, 1) for agent, total in totals.items()})
