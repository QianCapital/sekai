"""Make `pickles_0_2_0.pkl`: objects pickled by sekai 0.2.0 and how they continue there.

`tests/test_compat_0_2.py` loads them with the current sekai and checks they continue
the same way. Run with sekai 0.2.0 and NumPy 1.x (so the file also loads on NumPy 2):

    PYTHONPATH=<a sekai 0.2.0 checkout> python tests/data/make_pickles_0_2_0.py
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np

import sekai
from sekai.finance import (
    GBM,
    DifferentialSharpe,
    DrawdownPenalty,
    LogReturn,
    MarketData,
    ProportionalCost,
    SpreadCost,
    SquareRootImpact,
    StepOutcome,
)

warnings.filterwarnings("ignore")
STEPS = 25


def rollout(env: Any, seed: int) -> dict[str, Any]:
    """Continue `env` with seeded actions: the actions and what came back."""
    space = getattr(env, "batch_action_space", None) or env.action_space
    space.seed(seed)
    actions, out = [], []
    for _ in range(STEPS):
        action = space.sample()
        result = env.step(action)
        actions.append(action)
        rewards = getattr(result, "rewards", getattr(result, "reward", None))
        observation = getattr(result, "observations", getattr(result, "observation", None))
        out.append((observation, rewards))
        done = getattr(result, "done", False)
        if done is True:
            env.reset()
    return {"actions": actions, "results": out}


def main() -> None:
    assert sekai.__version__ == "0.2.0", "run this with sekai 0.2.0"
    rng = np.random.default_rng(0)
    prices = np.cumprod(1 + rng.normal(0, 0.01, (120, 3)), axis=0) * 100
    data = MarketData(
        prices,
        fields={
            "volume": rng.lognormal(12, 1, prices.shape),
            "spread": np.full(prices.shape, 5e-4),
        },
        timestamps=np.arange("2020-01-01", 120, dtype="datetime64[D]"),
        exogenous={"rate": np.full(120, 0.01)},
    )
    items: dict[str, tuple[bytes, Any]] = {}

    def env_item(name: str, env: Any, seed: int) -> None:
        blob = pickle.dumps(env)
        items[name] = (blob, rollout(pickle.loads(blob), seed))

    for env_id, kwargs in [
        ("Trading-v0", {"data": data.select(["asset_0"]), "episode_length": 40}),
        ("Portfolio-v0", {"data": data, "episode_length": 40}),
        ("Market-v0", {"data": data, "episode_length": 40}),
        ("CartPole-v1", {}),
        ("MarketMaking-v0", {}),
        ("OptimalExecution-v0", {}),
    ]:
        env = sekai.make(env_id, **kwargs)
        env.reset(seed=3)
        env.action_space.seed(3)
        for _ in range(5):
            env.step(env.action_space.sample())
        env_item(f"env:{env_id}", env, 7)
    one = data.select(["asset_0"])
    wrapped = sekai.NormalizeObservation(sekai.make("Trading-v0", data=one, episode_length=40))
    wrapped.reset(seed=1)
    env_item("wrapped:Trading-v0", wrapped, 8)
    for env_id, source in (("Trading-v0", one), ("Portfolio-v0", data)):
        vec = sekai.make_vec(env_id, num_envs=2, data=source, episode_length=30)
        vec.reset(seed=2)
        env_item(f"vec:{env_id}", vec, 9)

    sliced = data[10:90]
    items["data"] = (
        pickle.dumps(data),
        {"prices": data.prices, "log_returns": data.log_returns(), "slice": sliced.prices},
    )
    items["data:split"] = (pickle.dumps(data.split(0.5)[1]), {"prices": data.split(0.5)[1].prices})

    trades = rng.normal(0, 0.05, (2, 3))
    ctx = {
        "spread": np.full((2, 3), 1e-3),
        "equity": np.array([1e6, 2e6]),
        "volatility": np.full((2, 3), 0.01),
        "volume": np.full((2, 3), 1e7),
    }
    for name, model in [
        ("cost:spread", SpreadCost()),
        ("cost:composite", SpreadCost() + SquareRootImpact(0.2) + ProportionalCost(1e-4)),
    ]:
        items[name] = (
            pickle.dumps(model),
            {"trades": trades, "context": ctx, "cost": model(trades, **ctx)},
        )

    reward = LogReturn() - 0.1 * DifferentialSharpe() - DrawdownPenalty()
    outcomes = [
        StepOutcome.from_growth(
            1 + rng.normal(0, 0.02, 2), equity=np.ones(2), turnover=np.zeros(2), cost=np.zeros(2)
        )
        for _ in range(6)
    ]
    for o in outcomes[:3]:
        reward(o)
    blob = pickle.dumps(reward)
    items["reward"] = (
        blob,
        {"outcomes": outcomes[3:], "values": [reward(o) for o in outcomes[3:]]},
    )
    model = GBM(mu=0.05, sigma=0.2)
    items["model"] = (
        pickle.dumps(model),
        {"paths": model.simulate(np.random.default_rng(5), 10).prices},
    )

    path = Path(__file__).with_name("pickles_0_2_0.pkl")
    path.write_bytes(pickle.dumps({"numpy": np.__version__, "items": items}))
    print(f"wrote {path} ({path.stat().st_size // 1024} KB, {len(items)} objects)")


if __name__ == "__main__":
    main()
