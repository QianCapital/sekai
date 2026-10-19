---
title: sekai
hide:
  - navigation
  - toc
---

<div class="qc-hero" markdown>

<p class="qc-hero__eyebrow">Open source from Qian Capital</p>

# sekai · 世界

<p class="qc-hero__lede">A modern, flexible reinforcement-learning environment framework, built for
complex environments and financial markets.</p>

[Get started](getting-started/quickstart.md){ .md-button .md-button--primary }
[Migrating from gymnasium](guide/gymnasium.md){ .md-button }

</div>

<div class="sk-intro" markdown>

**sekai** is a reinforcement-learning environment framework for Python: a modern take on the
gym/gymnasium API with structured results instead of tuples, async-native environments,
observations whose size can change, native and multiprocess vectorisation, renderers kept
outside the environment, and typed episode statistics. Its finance toolkit (point-in-time
market data, simulators, costs, rewards, metrics and a limit order book) powers
environments for trading, portfolio allocation over changing universes, market making,
optimal execution and order-book markets. It ships the classic-control benchmarks too
(dynamics bit-identical to gymnasium), and two-way gymnasium and PettingZoo bridges mean
Stable-Baselines3, CleanRL, RLlib and multi-agent libraries work out of the box.

</div>

## Install

```bash
pip install sekai                 # NumPy is the only hard dependency
pip install "sekai[gymnasium]"    # + gymnasium bridge (SB3, CleanRL, RLlib, ...)
pip install "sekai[all]"          # + PettingZoo bridge, pandas data loading, video export
```

## A first look

=== "Classic control"

    ```python
    import sekai

    env = sekai.make("CartPole-v1")
    env.action_space.seed(0)              # spaces have their own RNG
    result = env.reset(seed=0)            # ResetResult(observation, info)
    total = 0.0
    while not result.done:                # ResetResult.done is always False
        result = env.step(env.action_space.sample())
        total += result.reward            # StepResult: a frozen dataclass, not a tuple
    print(result.terminated, result.truncated, total)  # True False 18.0
    ```

=== "Finance"

    ```python
    import sekai

    env = sekai.make("OptimalExecution-v0")    # Almgren–Chriss liquidation
    schedule = env.almgren_chriss_actions()    # closed-form optimal policy

    result = env.reset(seed=0)
    for action in schedule:
        result = env.step(action)
    print(result.done, round(result.info["shortfall_bps"], 1))  # True 207.1
    ```

## Features

<div class="grid cards" markdown>

-   :material-code-braces:{ .lg .middle } **Structured results**

    ---

    `step()` returns a frozen `StepResult` with a `.done` property; `reset()` returns a
    `ResetResult`. No more unpacking five values in the wrong order.

    [:octicons-arrow-right-24: Environments](guide/environments.md)

-   :material-lightning-bolt:{ .lg .middle } **Async-native**

    ---

    Every environment has `async_step()` / `async_reset()`. `AsyncVecEnv` overlaps the
    I/O of live feeds and remote simulators with `asyncio`.

    [:octicons-arrow-right-24: Async environments](guide/async.md)

-   :material-layers-triple:{ .lg .middle } **Fast vectorisation**

    ---

    Native vector environments step thousands of sub-envs with array operations; plus
    `SyncVecEnv`, `AsyncVecEnv` (asyncio) and `ProcVecEnv` (worker processes, shared
    memory), all with same-step autoreset and normalisation wrappers.

    [:octicons-arrow-right-24: Vector environments](guide/vector.md)

-   :material-shape-plus:{ .lg .middle } **Variable-size observations**

    ---

    `MaskedBox` padded sets and `Sequence` spaces for observations that grow and shrink,
    with action masks that `check_env`, wrappers and masked-policy learners understand.

    [:octicons-arrow-right-24: Variable-size observations](guide/dynamic.md)

-   :material-finance:{ .lg .middle } **Financial markets**

    ---

    Trading, portfolios over changing universes, Avellaneda–Stoikov market making,
    Almgren–Chriss execution and limit order books, on simulated markets or your own history.

    [:octicons-arrow-right-24: Finance environments](environments/finance.md)

-   :material-toolbox:{ .lg .middle } **Finance toolkit**

    ---

    Point-in-time market data and walk-forward splits, a look-ahead detector, market
    simulators, cost and impact models, rewards and performance metrics.

    [:octicons-arrow-right-24: Finance toolkit](finance/index.md)

-   :material-swap-horizontal:{ .lg .middle } **gymnasium both ways**

    ---

    `to_gymnasium()` exposes any sekai env to gymnasium-based libraries;
    `from_gymnasium()` brings any gymnasium env into sekai.

    [:octicons-arrow-right-24: Migrating from gymnasium](guide/gymnasium.md)

-   :material-movie-open:{ .lg .middle } **Renderers, not render modes**

    ---

    Rendering lives in `Renderer` objects. A NumPy `Canvas`, renderers for every classic
    control env and `RecordFrames` for GIF/MP4, with no pygame needed.

    [:octicons-arrow-right-24: Rendering](guide/rendering.md)

-   :material-chart-box:{ .lg .middle } **Typed statistics**

    ---

    `info["episode"]` is an `EpisodeStats` dataclass, reported to pluggable trackers
    (in memory, logging, or your own).

    [:octicons-arrow-right-24: Episode statistics](guide/statistics.md)

-   :material-puzzle:{ .lg .middle } **Registry, plugins and tooling**

    ---

    `make()` / `make_vec()`, lazy plugin namespaces, `check_env()` for your test suite and
    a `sekai` command line (`list`, `info`, `check`, `bench`).

    [:octicons-arrow-right-24: Registry](guide/registry.md)

</div>

## Why sekai?

gym and gymnasium have a set of friction points that compound in large or production
codebases. sekai keeps the familiar shape of the API and fixes these:

| | gymnasium | sekai |
|---|---|---|
| Step / reset results | `obs, rew, term, trunc, info = env.step(a)`: easy to unpack wrong | frozen `StepResult` / `ResetResult` dataclasses with `.done` |
| Async environments | not supported | `async_step()` / `async_reset()` on every env; `AsyncVecEnv` runs them concurrently |
| Vector env naming | `AsyncVectorEnv` means multiprocessing | `SyncVecEnv`, `AsyncVecEnv` (asyncio), `ProcVecEnv` (multiprocessing) |
| Vector autoreset | next-step by default (an extra step returns the reset obs) | same-step: terminal obs in `infos[i]["final_observation"]` |
| Vector infos | dict of arrays plus `_key` masks | `list[dict]`, one per sub-env; `stack_infos()` when you want arrays |
| Rendering | `render()` on the env, `render_mode` fixed at construction | `Renderer` objects, injected and swappable; `RecordFrames` wrapper |
| Episode statistics | `info["episode"]["r"]` in an untyped dict | `info["episode"]` is a typed `EpisodeStats` |
| Multi-agent | a separate library (PettingZoo) | `MultiAgentEnv` with per-agent spaces, event-driven turns, and a PettingZoo bridge |
| Variable-size observations | `Sequence` / `Graph` spaces that do not batch | `MaskedBox` padded sets that batch, flatten and normalise; `Sequence` with padding |
| Fast vector envs | `vector_entry_point` | `NativeVecEnv` base class and native finance envs; shared-memory `ProcVecEnv` |
| Registry plugins | register at import time | `register_namespace()` lazy loaders, `sekai -m plugin` on the CLI |
| Env checking | `PassiveEnvChecker` wrapped into every `make()` | `check_env()` in your test suite; zero overhead in training |

## Where next

- **New here?** Start with [Installation](getting-started/installation.md) and the
  [Quickstart](getting-started/quickstart.md).
- **Coming from gymnasium?** The [migration guide](guide/gymnasium.md) maps every call and
  shows how to train with Stable-Baselines3.
- **Looking for an environment?** See the [catalog](environments/index.md).
- **Need a signature?** The [API reference](api/index.md) is generated from the source.
