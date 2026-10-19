"""An async environment backed by a (simulated) network feed, run with AsyncVecEnv.

Each step awaits a 20 ms "exchange round trip". Run sequentially, 8 envs would
take 8 x 20 ms per step; AsyncVecEnv overlaps the waits so a batch step takes
about 20 ms.
"""

import asyncio
import time
from typing import Any

import numpy as np

import sekai

LATENCY = 0.02


class LiveFeedEnv(sekai.Env[np.ndarray, np.intp]):
    def __init__(self) -> None:
        self.observation_space = sekai.Box(0, np.inf, shape=(1,), dtype=np.float32)
        self.action_space = sekai.Discrete(3)  # sell / hold / buy
        self.price = 100.0

    async def _next_tick(self) -> float:
        await asyncio.sleep(LATENCY)  # stands in for an await on a websocket / REST call
        self.price *= float(np.exp(self.rng.normal(0, 0.001)))
        return self.price

    async def async_reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> sekai.ResetResult[np.ndarray]:
        self._set_rng(seed)
        self.price = 100.0
        return sekai.ResetResult(np.array([await self._next_tick()], np.float32), {})

    async def async_step(self, action: int | np.integer) -> sekai.StepResult[np.ndarray]:
        before = self.price
        after = await self._next_tick()
        reward = (int(action) - 1) * (after - before)
        return sekai.StepResult(np.array([after], np.float32), reward, False, False, {})

    # The sync API is still required; here it just drives the async one. Note that
    # a wrapper overriding only step() would route async calls through this sync
    # path, so wrappers for async-native envs should implement async_step() too.
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        return asyncio.run(self.async_reset(seed=seed, options=options))

    def step(self, action: int | np.integer):
        return asyncio.run(self.async_step(action))


async def main() -> None:
    vec = sekai.AsyncVecEnv([LiveFeedEnv] * 8)
    await vec.async_reset(seed=0)
    start = time.perf_counter()
    for _ in range(10):
        result = await vec.async_step(vec.batch_action_space.sample())
    elapsed = (time.perf_counter() - start) / 10
    print(f"8 envs, {LATENCY * 1000:.0f} ms latency each: {elapsed * 1000:.0f} ms per batched step")
    print("prices:", np.round(result.observations[:, 0], 3))
    vec.close()


asyncio.run(main())
