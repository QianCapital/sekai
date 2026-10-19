# Async environments

Every sekai environment has two coroutines next to its sync methods:

```py
result = await env.async_reset(seed=0)
result = await env.async_step(action)
```

By default they call `reset()` / `step()`, so any env can be driven from async code. An
environment whose step waits on I/O (an exchange gateway, a market-data websocket, a remote
simulator) overrides them with real `await`s, and `AsyncVecEnv` then runs many such envs
concurrently on one event loop.

## An async-native environment

Override `async_reset()` and `async_step()`. `reset()` and `step()` are still abstract and
must exist; driving the coroutine with `asyncio.run()` is the simplest choice for callers that
are not async themselves.

```python
import asyncio
import time

import numpy as np
import sekai

LATENCY = 0.02  # a 20 ms round trip to the venue


class VenueEnv(sekai.Env[np.ndarray, np.intp]):
    """Each step sends an order and awaits the next quote from a (simulated) venue."""

    def __init__(self):
        self.observation_space = sekai.Box(0.0, np.inf, shape=(1,), dtype=np.float32)
        self.action_space = sekai.Discrete(3, start=-1)  # sell / hold / buy
        self.price = 100.0

    async def _next_quote(self) -> float:
        await asyncio.sleep(LATENCY)  # stands in for `await websocket.recv()`
        self.price *= float(np.exp(self.rng.normal(0.0, 1e-3)))
        return self.price

    async def async_reset(self, *, seed=None, options=None):
        self._set_rng(seed)
        self.price = 100.0
        return sekai.ResetResult(np.array([await self._next_quote()], np.float32), {})

    async def async_step(self, action):
        before = self.price
        after = await self._next_quote()
        reward = float(int(action) * (after - before))
        return sekai.StepResult(np.array([after], np.float32), reward, False, False, {})

    # Required sync API: run the coroutine to completion.
    def reset(self, *, seed=None, options=None):
        return asyncio.run(self.async_reset(seed=seed, options=options))

    def step(self, action):
        return asyncio.run(self.async_step(action))
```

## Running envs concurrently

`AsyncVecEnv` gathers the sub-envs' `async_step()` coroutines, so their waits overlap: a
batched step costs one round trip instead of eight.

```python
async def main():
    vec = sekai.AsyncVecEnv([VenueEnv] * 8)
    await vec.async_reset(seed=0)
    start = time.perf_counter()
    for _ in range(10):
        result = await vec.async_step(vec.batch_action_space.sample())
    per_step = (time.perf_counter() - start) / 10
    vec.close()
    return per_step, result


per_step, result = asyncio.run(main())
print(f"{per_step * 1000:.0f} ms per batched step")  # ~ 21 ms per batched step
print(result.observations.shape)  # (8, 1)

with sekai.SyncVecEnv([VenueEnv] * 8) as vec:  # the same envs, one after another
    vec.reset(seed=0)
    start = time.perf_counter()
    vec.step(vec.batch_action_space.sample())
    print(f"{(time.perf_counter() - start) * 1000:.0f} ms per batched step")  # ~ 170 ms per batched step
```

Autoreset, spaces, results and sub-env access are the same as for every
[vector environment](vector.md).

!!! note "No speed-up for blocking envs"
    `AsyncVecEnv` only helps when `async_step()` actually awaits. Envs that use the default
    async methods (all built-in envs do) run one after another, exactly like `SyncVecEnv`.
    For CPU-bound simulators use `ProcVecEnv`.

## Calling from sync code

`AsyncVecEnv.step()` and `reset()` are sync shims: they run the coroutines on a private event
loop owned by the vector env, so plain scripts and training loops can use it like any other
`VecEnv`:

```python
vec = sekai.AsyncVecEnv([VenueEnv] * 4)
print(vec.reset(seed=0).observations.shape)  # (4, 1)
vec.close()  # also closes the private loop
```

!!! warning "Inside a running event loop, await"
    The shims cannot run while an event loop is already running in the same thread (inside
    `async def` code, or a Jupyter notebook, which runs a loop). There they raise
    `RuntimeError`; use `await vec.async_reset(...)` / `await vec.async_step(...)`
    instead. Top-level `await` works in Jupyter and IPython.

```python
async def wrong():
    vec = sekai.AsyncVecEnv([VenueEnv] * 2)
    try:
        vec.reset(seed=0)
    except RuntimeError as e:
        print(type(e).__name__)  # RuntimeError
    finally:
        vec.close()

asyncio.run(wrong())
```

## Wrappers keep the async path

All built-in wrappers implement both `step()` and `async_step()` (and the `reset` pair), so
`await wrapped.async_step(a)` applies every layer and still awaits the base env:

```python
async def episode():
    env = sekai.RecordEpisodeStatistics(sekai.TimeLimit(VenueEnv(), max_episode_steps=5))
    result = await env.async_reset(seed=0)
    while not result.done:
        result = await env.async_step(1)
    return result


result = asyncio.run(episode())
print(result.truncated, result.info["episode"].episode_length)  # True 5
```

For your own wrappers the rule is: if a class overrides `step()` but not `async_step()`,
sekai routes `async_step()` through your `step()` so your logic is never skipped, but
everything below that wrapper then runs **synchronously**.

!!! danger "Wrappers around async-native envs need `async_step()` too"
    For an async-native env whose sync `step()` drives the coroutine with `asyncio.run()`
    (like `VenueEnv` above), a step-only wrapper does not just lose concurrency: under
    `AsyncVecEnv` (or any `await`), its `async_step()` calls `step()`, which calls
    `asyncio.run()` inside the running loop and fails.

```python
class RewardInBps(sekai.Wrapper):           # overrides step() only
    def step(self, action):
        r = self.env.step(action)
        return sekai.StepResult(r.observation, 1e4 * r.reward, r.terminated, r.truncated, r.info)


async def broken():
    vec = sekai.AsyncVecEnv([lambda: RewardInBps(VenueEnv())] * 2)
    await vec.async_reset(seed=0)
    try:
        await vec.async_step(np.array([1, 1]))
    except RuntimeError as e:
        print(e)  # asyncio.run() cannot be called from a running event loop
    finally:
        vec.close()

asyncio.run(broken())
```

Implement both paths, sharing the logic through a helper, and the wrapper stays async end to
end (see [Writing a wrapper](wrappers.md#stateful-wrappers-and-the-async-rule)):

```python
class AsyncRewardInBps(sekai.Wrapper):
    def _scale(self, r):
        return sekai.StepResult(r.observation, 1e4 * r.reward, r.terminated, r.truncated, r.info)

    def step(self, action):
        return self._scale(self.env.step(action))

    async def async_step(self, action):
        return self._scale(await self.env.async_step(action))
```

## Vector wrappers keep the async path

Every `VecEnv` has `async_step()` / `async_reset()`. For `SyncVecEnv` and `ProcVecEnv` they
run the sync methods; `AsyncVecEnv` runs its sub-envs concurrently. Vector wrappers
(`VecRecordEpisodeStatistics`, `VecNormalizeObservation`, `VecNormalizeReward`,
`VecTransformObservation`, `VecTransformReward` and your `VecWrapper` subclasses) delegate
both paths, so a wrapped `AsyncVecEnv` is still concurrent. The same routing rule applies:
a `VecEnv` or `VecWrapper` subclass that overrides only `step()` / `reset()` gets async
methods that run its overrides.

```python
async def normalised():
    vec = sekai.VecNormalizeReward(
        sekai.VecRecordEpisodeStatistics(sekai.AsyncVecEnv([lambda: AsyncRewardInBps(VenueEnv())] * 8))
    )
    await vec.async_reset(seed=0)
    start = time.perf_counter()
    result = await vec.async_step(np.ones(8, dtype=np.int64))
    elapsed = time.perf_counter() - start
    vec.close()
    return elapsed, result


elapsed, result = asyncio.run(normalised())
print(f"{elapsed * 1000:.0f} ms for 8 sub-envs")  # ~ 21 ms for 8 sub-envs
print(result.rewards.shape)  # (8,)
```

## Errors and closing

If one sub-env raises inside `AsyncVecEnv.async_step()` / `async_reset()`, the other sub-envs
still finish their call before the first error is re-raised, so no sub-env is left halfway
through a step and the vector env can be used again. After `close()`, `vec.closed` is `True`
and further calls raise `sekai.VecEnvClosed`.

## Checking

[`check_env`](checking.md) exercises `async_reset()` and `async_step()` too (with
`check_async=True`, the default) and reports an error if they raise or return something
other than a result object. The check is skipped when `check_env` itself is called from
inside a running event loop.
