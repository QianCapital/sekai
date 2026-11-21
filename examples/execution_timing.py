"""Why execution timing matters: a strategy that only "works" when it fills at the close it saw.

Closing prices carry microstructure noise: a close at the ask is followed, on average,
by one nearer the mid (bid-ask bounce), so returns reverse a little from one close to
the next. A short-term reversal strategy (buy what just fell, sell what just rose) looks
profitable when it fills at the very close it observed, which no real order can do, and
the profit disappears once orders fill at the next close.

Market-v0 fills at the observed close; Market-v1 lets you choose (`execution=`) and
defaults to the next close. Same data, same policy:
"""

import numpy as np

import sekai
from sekai.finance import GBM, MarketData, NoCost, TargetWeights

rng = np.random.default_rng(0)
mid = GBM(mu=0.05, sigma=0.25, n_assets=40).simulate(rng, 2519).prices
bounce = 1 + 0.002 * rng.choice([-1.0, 1.0], size=mid.shape)  # closes at the bid or the ask
data = MarketData(mid * bounce)


def reversal(observation):
    """Long the assets that fell most on the last day, short those that rose: dollar neutral."""
    last = observation["assets"]["values"][:, -1, 0]  # each slot's latest log return
    tilt = -(last - last.mean())
    return (tilt / np.abs(tilt).sum()).astype(np.float32)


for execution in ("close", "next_close"):
    env = sekai.make(
        "Market-v1",
        data=data,
        execution=execution,
        lookback=2,
        episode_length=2500,
        action=TargetWeights(),
        transaction_cost=NoCost(),
    )
    result = env.reset(seed=0, options={"start": 2})
    while not result.done:
        result = env.step(reversal(result.observation))
    growth = result.info["equity"] ** (252 / 2500) - 1
    print(f"execution={execution!r:13} annual return {growth:+7.1%}")
