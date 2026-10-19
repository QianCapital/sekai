# Changelog

All notable changes to sekai are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/) (pre-1.0, minor versions may change behaviour).

## [0.2.0] - Unreleased

The first release with built-in environments, a finance toolkit, a test suite, CI and a
documentation site.

### Added

**Dynamic dimensionality and action masks**

- `sekai.MaskedBox`: a variable-size set of up to `max_size` elements padded to a fixed shape, with
  an int8 validity mask (`{"values": (max_size, *shape), "mask": (max_size,)}`). It is a `Dict`
  subclass, so flattening, batching, vectorisation and gymnasium conversion work unchanged;
  `pack()` / `unpack()` build and read samples.
- `sekai.Sequence`: variable-length sequences of any feature space (tuples, or stacked batches with
  `stack=True`), with batching, JSON and gymnasium conversion.
- `Box.sample(mask=...)` (masked entries are set to `Box.zero()`) and `sekai.spaces.zero_sample()`.
- An action-mask convention: environments implement `action_mask()` in the format of
  `action_space.sample(mask=...)`. `sekai.get_action_mask(env)` finds it through wrappers,
  `check_env` samples random actions with it (and reports malformed masks),
  `ActionMaskObservation` adds it to the observation, and `to_gymnasium()` exposes it as
  `action_masks()` for sb3-contrib's MaskablePPO.
- Wrappers `PadSequenceObservation` (Sequence → MaskedBox), `ActionMaskObservation`, and
  `DelayAction` (order latency: actions execute `delay` steps after they are chosen).
- `FrameStackObservation` stacks any batchable observation space (Dict, Tuple, MaskedBox,
  Discrete, ...), leaf by leaf, with `padding="reset"` or `"zeros"`.
- `NormalizeObservation` / `VecNormalizeObservation` normalise nested observation spaces (each Box
  leaf, MaskedBox leaves per feature over occupied slots only, other leaves passed through), can
  use exponentially weighted statistics for non-stationary data (`stats="ewm", halflife=...`),
  clip (`clip=`), and save/restore with `state_dict()` / `load_state_dict()`.
  `sekai.ExponentialMovingStats` is the EWM counterpart of `RunningMeanStd`.

**Vectorisation**

- Native vector environments: `sekai.NativeVecEnv` is a base class for environments that advance
  all sub-envs at once with array operations (seeding, per-env options, same-step autoreset and
  per-env infos are handled for you). Register one with
  `register(..., vector_entry_point=...)`; `make_vec()` then uses it by default
  (`vectorization_mode=None` → `"native"` when available, else `"sync"`). Native envs take
  `max_episode_steps` and truncate exactly like `TimeLimit` (`infos[i]["elapsed_steps"]` and
  `get_attr("elapsed_steps")` included), and `set_attr()` raises `AttributeError` for names the
  env doesn't have.
- Episode samplers have a `spawn()` hook: `make_vec()` (every backend) and the native finance
  envs give each sub-env `sampler.spawn()`, so sub-envs behave as independent copies of the env.
  `SequentialStart` spawns a fresh copy, so with `sampler="sequential"` every sub-env replays
  the same windows and the instance you pass in is not advanced; other samplers are shared by
  default, so later changes to them (a curriculum, say) reach every sub-env. Override `spawn()`
  for per-environment state. The vector guide shows how to split a history across sub-envs.
- The native finance envs expose their parameters and state through `get_attr()` / `set_attr()`
  as `SyncVecEnv` does; shape-defining ones (`lookback`, `episode_length`, `n_assets`,
  `allow_short`) and the data are read-only. Likewise `episode_length` and `lookback` are
  read-only on `Trading-v0`, `Portfolio-v0` and `Market-v0`.
- `ProcVecEnv(num_workers=..., shared_memory=...)`: observations travel through shared memory
  (fixed-size spaces, on by default) and several sub-envs can share a worker process.
- `sekai bench --mode auto|native|sync|async|proc --workers N`; `sekai list` marks environments
  with a native vector implementation.

**Finance toolkit (`sekai.finance`)**

- `MarketData`: an immutable, point-in-time panel of prices, per-asset fields and market-wide
  (exogenous) series, where NaN marks an unlisted asset; built from arrays or pandas (wide or long
  format); chronological `split()` and `walk_forward()` with embargo and lookback history.
  The history rows of a split are recorded as `MarketData.warmup`, and no environment ever
  makes a decision on one, whatever its lookback (sampler positions, and their error messages,
  count the windows after the warm-up). Timezone-aware timestamps are rejected with
  a message showing how to convert them, never shifted silently.
- Episode samplers (`RandomStart`, `SequentialStart`, `FixedStart`) and
  `reset(options={"start": date})` for every historical environment (a start on a warm-up row
  raises `ValueError`).
- Market simulators, batch-first with latent state: GBM, Heston, Merton jump-diffusion, regime
  switching, GARCH(1,1), exponential Ornstein-Uhlenbeck, cointegrated pairs, block bootstrap,
  `Randomized` (domain randomisation) and `ListingProcess` (assets list and delist).
- Causal feature transforms (rolling/EWM statistics, momentum, cross-sectional ranks and
  z-scores), point-in-time `asof_join` with publication lags, and `check_causal`, which detects
  look-ahead bias (raising the new `sekai.LookAheadError`) by rescaling, reordering,
  sign-flipping and re-masking the future and checking the past is unaffected.
- Cost models (`ProportionalCost`, `SpreadCost`, `FixedCost`, `SquareRootImpact`, composable with
  `+`), weight-space accounting (`AccountConfig`: leverage, shorting, cash, margin and borrow rates),
  and reward functions (`LogReturn`, `ProfitAndLoss`, `MeanVariance`, `DifferentialSharpe`,
  `DrawdownPenalty`, `TurnoverPenalty`, composable arithmetically). Cost models never price a
  trade at zero because an input is missing: they raise unless given `fallback_spread=` /
  `fallback_volatility=`. Only NaN (or a negative value) counts as missing; a finite zero
  spread or volatility is valid and costs nothing. Environments reject a volatility-based cost
  with `lookback=1`, fallback or not.
  `ProfitAndLoss` uses `StepOutcome.pnl`, the exact change in equity, so its rewards sum to
  final equity − 1 even through a wipe-out.
- Performance metrics (Sharpe, Sortino, Calmar, drawdown, VaR/CVaR, ...), `PerformanceSummary`,
  and the `RecordPerformance` wrapper (`info["performance"]` at the end of each episode).
- `SlotMap` (stable slots for a changing universe) and a limit order book: a price-time priority
  matching engine (`OrderBook`) with zero-intelligence Poisson order flow.

**Built-in environments**

- Environments registered on `import sekai`:
  - Classic control: `CartPole-v1`, `MountainCar-v0`, `MountainCarContinuous-v0`,
    `Pendulum-v1`, `Acrobot-v1`. Dynamics are bit-for-bit identical to gymnasium's
    (verified by tests) and need only NumPy.
  - Finance: `Trading-v0` (single-asset trading), `Portfolio-v0` (multi-asset allocation),
    `MarketMaking-v0` (Avellaneda–Stoikov) and `OptimalExecution-v0` (Almgren–Chriss),
    with simulated markets or your own price history, plus closed-form baselines
    (`avellaneda_stoikov_action()`, `almgren_chriss_actions()`). Trading and Portfolio accept
    `MarketData`, any market `model=`, any cost model, a `reward=` function and a `sampler=`.
    In both, positions drift with prices between steps and costs are charged on the trade back
    to the target, so they agree exactly on one asset.
  - Episode horizons follow one rule: environments whose horizon is part of the task
    (market making, optimal execution, the order book) observe time and terminate at the
    horizon; those whose episode is a window of an ongoing market (`Trading-v0`,
    `Portfolio-v0`, `Market-v0`) truncate and don't observe time (add `TimeAwareObservation`
    to observe it).
  - `Market-v0`: multi-asset trading over a changing universe (padded-set observations, action
    masks, stable slots), with data fields and exogenous series as features, cost models,
    financing, leverage and shorting, pluggable rewards and optional privileged (latent) state;
    simulated by default from a two-regime market whose assets list and delist.
  - `LimitOrderBook-v0`: event-driven market making and trading in a simulated order book,
    and `LimitOrderBookMarket`, a multi-agent market in which traders act on their own clocks.
  - Native vector versions of the four classic finance environments (`TradingVecEnv`,
    `PortfolioVecEnv`, `MarketMakingVecEnv`, `OptimalExecutionVecEnv`), used by `make_vec()`:
    1.3-2.6 million environment steps per second at 1024 sub-envs on an 8-core machine,
    about 15-20x `SyncVecEnv`.
  - `sekai.envs.finance.simulate_gbm()` for correlated geometric Brownian motion paths.
- Finance environments report `info["timestamp"]` and `info["dt"]` (step length), the convention
  for time-aware and event-driven environments.
- **Gymnasium interop** (`pip install "sekai[gymnasium]"`): `sekai.compat.to_gymnasium()`
  exposes any sekai env to gymnasium-based libraries (Stable-Baselines3, CleanRL, RLlib);
  `from_gymnasium()` brings any gymnasium env into sekai; space converters both ways, which keep
  `Dict` key order so both libraries flatten observations identically. Environments with nested
  observations (`Market-v0`, `LimitOrderBook-v0`) train with SB3 through
  `sekai.FlattenObservation` and `MlpPolicy` (MaskablePPO included).
- **PettingZoo interop** (`pip install "sekai[pettingzoo]"`): `sekai.compat.to_pettingzoo()` and
  `from_pettingzoo()` convert between `MultiAgentEnv` and PettingZoo's Parallel API.
- **Rendering**: a dependency-free NumPy `Canvas`, RGB renderers for every classic control
  env (`CartPoleRenderer`, ...), the `RecordFrames` wrapper and `save_gif()` / `save_video()`.
- **Wrappers**: `ClipReward`, `RescaleObservation`, `DtypeObservation`, `FilterObservation`,
  `TimeAwareObservation`, `DelayObservation`, `StickyAction`, `ActionRepeat`.
- **Vector wrappers**: `VecRecordEpisodeStatistics`, `VecNormalizeObservation`,
  `VecNormalizeReward`, `VecTransformObservation`, `VecTransformReward`.
- `sekai.make_vec(id, num_envs, vectorization_mode="sync" | "async" | "proc")`.
- `sekai.make(..., max_episode_steps=N)` overrides the registered time limit (`-1` removes it);
  `make()` also accepts an `EnvSpec`. The spec attached to the env records the kwargs actually used.
- `sekai.deregister()`, `sekai.pprint_registry()`, `EnvSpec.pprint()`, and "did you mean"
  suggestions in `EnvNotFound`.
- Vector envs: autoreset (see Changed), `batch_action_space`, a `closed` flag,
  `async_step()` / `async_reset()` on every VecEnv and vector wrapper (so wrapping an
  `AsyncVecEnv` keeps it concurrent), `call()` / `get_attr()` / `set_attr()` on
  `AsyncVecEnv`, and `ProcVecEnv(close_timeout=...)`.
- `VecObservationWrapper.final_observation()`: vector observation wrappers also transform the
  terminal observations stored by autoreset.
- Space utilities: `batch_space` for every space type, `concatenate()`, `iterate()`,
  `space_from_jsonable()` / `register_space_type()`, and `from_jsonable()` for `Dict`/`Tuple`.
- `check_env`: `check_determinism` is now implemented; new checks for the async API,
  `Space` types, observation dtypes and observations mutated in place after being returned.
- `ResetResult.done` (always `False`), so `result = env.reset()` /
  `while not result.done: result = env.step(...)` works with one variable. Likewise
  `MAResetResult.all_done` / `any_done` and `VecResetResult.dones`.
- `MultiAgentEnv` gets the same seeding API as `Env` (`rng`, `_set_rng(seed)`).
- `Env.set_attr()`; `sekai.rendering.BaseRenderer` (no-op `close()` plus `with` support);
  `elapsed_steps` on the finance envs.
- sekai exceptions can be pickled (e.g. raised inside a `ProcessPoolExecutor` job).
- `sekai.RunningMeanStd`, with `state_dict()` / `load_state_dict()` to save normalisation stats.
- A `sekai` command-line tool (also `python -m sekai`): `list`, `info`, `check`, `bench`.
- `py.typed`: sekai now ships type information, and type-checks clean under `mypy --strict`.
- A test suite, GitHub Actions CI on Linux/macOS/Windows and Python 3.10–3.14,
  trusted-publishing release workflow, MkDocs documentation site (Qian Capital theme),
  runnable examples.

### Changed

These change behaviour for code written against 0.1.x:

- **Vector envs reset finished sub-envs automatically** (`autoreset=True`), AutoReset-style:
  the returned observation starts the new episode, and the terminal observation/info are in
  `infos[i]["final_observation"]` / `["final_info"]`. Pass `autoreset=False` for the old
  behaviour. Sub-envs already wrapped in `AutoReset` are not reset twice.
- **`Env._set_rng(None)` keeps the current RNG** instead of re-seeding from OS entropy, so
  `reset(seed=s)` followed by plain `reset()` calls is reproducible (gymnasium semantics).
  Previously every unseeded reset drew fresh entropy.
- Wrappers share the wrapped env's RNG (`wrapper.rng is env.rng`).
- `NormalizeObservation` / `NormalizeReward` use `RunningMeanStd` (`obs_rms` / `return_rms`)
  and can be frozen with `update_running_mean = False`. `NormalizeReward` now resets its
  discounted return at episode ends, not only on `reset()`.
- `flatten_space()` always returns a `float32` Box, matching what `flatten()` produces;
  for `Dict`/`Tuple` the Box carries the sub-spaces' bounds instead of `(-inf, inf)`.
- Batched observations: `Discrete` observations are stacked into arrays, and all three
  vector backends batch `Dict`/`Tuple` observations the same way (recursively).
- `AutoReset` also puts the reset's info in `info["reset_info"]`.
- `ProcVecEnv` raises `VecEnvWorkerError` (with the worker's traceback) instead of
  `RuntimeError`, keeps workers alive after an error, and reads the spaces from worker 0
  instead of building an extra env in the parent process.
- `MultiDiscrete.sample()` returns the space's `dtype` (default `int64`).
- `sekai.make(id, max_episode_steps=N)` now sets the time limit (as in gymnasium) instead of
  passing `max_episode_steps` to the environment's constructor. An env that takes its own
  `max_episode_steps` argument should be registered with it in `kwargs=`.
- Vector envs raise `VecEnvClosed` when used after `close()`.
- `RescaleObservation` and `TimeAwareObservation` output `float32` for integer observations
  (they produced truncated or overflowing integers).
- `RecordEpisodeStatistics` starts a new episode's counters when it reports one, so it is
  also correct around an inner `AutoReset`.
- `VecEnv.get_attr()` / `set_attr()` / `call()` and `Wrapper.set_attr()` resolve names through
  the wrapper stack, so `make_vec("CartPole-v1").get_attr("x_threshold")` reaches the base env
  and `set_attr` changes the layer that has the attribute (it used to create a new attribute
  on the outermost wrapper).
- The `Renderer` protocol requires only `render()` and `close()`; duck-typed renderers now pass
  `isinstance(r, Renderer)`. Context-manager support moved to `BaseRenderer`.
- `Box` reprs show exact bounds without float32 noise, and per-dimension bounds when they differ.
- `check_env` reports each per-step problem once, with how many steps showed it.
- Wrapper argument validation raises `TypeError` / `ValueError` instead of `AssertionError`
  (asserts disappear under `python -O`). `Wrapper(...)` rejects objects that aren't `sekai.Env`.
- Optional dependency groups: `all` now installs the runtime extras
  (`gymnasium`, `pettingzoo`, `pandas`, `render`); development tools moved to `dev`, tests to `test`.

### Deprecated

- `sekai.backend` (`BackendOps`, `NumpyBackend`) warns on import and will be removed in 0.3. Nothing
  in sekai used it; for fast batched simulation write a native vector environment instead.

### Fixed

- `await env.async_step()` / `async_reset()` skipped the logic of every wrapper (e.g.
  `TimeLimit` never truncated). Built-in wrappers now implement both paths, and any wrapper
  or env that overrides only `step()`/`reset()` has its async methods routed through them.
- `EnvSpec` parsed `"CartPole-v1"` as name `"CartPole-v1"`, version `None`.
- Integer `Box` sampling overflowed at the dtype's maximum (`Box(0, 255, dtype=uint8)`
  raised `ValueError`).
- `AsyncVecEnv.step()` / `reset()` used `asyncio.get_event_loop()`, which fails on
  Python 3.14 when no loop is set; they now run on a private loop and give a clear error
  inside a running loop.
- `ProcVecEnv` could crash with `AttributeError` on `None` when a worker failed, because
  errors travelled through a queue that raced the result pipe.
- Vector envs iterated over the keys of `Dict` action batches instead of the sub-envs.
- `Dict` / `Tuple` spaces crashed when seeded with a `numpy.random.Generator`.
- `MultiDiscrete.sample(mask)` failed for the tuple-of-arrays mask format.
- `Discrete.contains()` rejected 0-d integer arrays; `MultiBinary.contains()` rejected any
  dtype other than `int8`; `MultiDiscrete.contains()` rejected lists.
- `RecordEpisodeStatistics` reported a seed of `None` when the env was wrapped.
- `EnvSpec.to_json()` dropped `additional_wrappers`.
- `make()` failed when an entry point returned an already-wrapped env.
- A namespace loader that raised was discarded, so the namespace could never load afterwards.
- Renderers crashed on a diverged (NaN) state; non-finite shapes are now skipped.
- An integer `Box` with infinite bounds (`Box(-np.inf, np.inf, dtype=np.int64)`) collapsed to a
  single value; infinite and out-of-range bounds now map to the dtype's limits, as in gymnasium.
  NaN bounds raise `ValueError`, and bounds are copied instead of aliasing the caller's array.
- `Box.contains()` rejected Python ints for unsigned or narrower integer Boxes
  (`Box(0, 255, (), np.uint8).contains(3)` was `False`).
- `MAStepResult.all_done` / `any_done` paired `terminated` and `truncated` by position instead
  of by agent; they now look up each agent, and an empty result is not done.
- `RescaleAction` could return actions one float32 step outside the env's bounds; it clips now.
- `RecordEpisodeStatistics` and `NormalizeReward` summed `np.float32` rewards in float32.
- `Tuple.sample(mask=...)` with too few masks returned a shorter, invalid tuple; it raises now.
- `check_env` crashed when `action_mask()` masked every action in a reachable state; that is
  reported as an error now.
- `MultiBinary` rejected NumPy integers for `n`; `Discrete.sample` and `MultiBinary.sample`
  rejected list masks.
- A sub-env whose `close()` raised stopped `VecEnv.close()` from closing the others, and made
  `ProcVecEnv.close()` wait out its whole timeout; every sub-env is now closed, the first error
  is raised afterwards, and workers that ignore termination are killed.

## [0.1.2] - 2026-05-06

### Fixed

- The package metadata version now matches `sekai.__version__`.

## [0.1.1] - 2026-05-06

### Added

- First public release: `Env` / `Wrapper` / `MultiAgentEnv` with `StepResult` / `ResetResult`,
  async methods on every env, spaces (`Box`, `Discrete`, `MultiDiscrete`, `MultiBinary`,
  `Dict`, `Tuple`), registry with lazy namespaces, `SyncVecEnv` / `AsyncVecEnv` / `ProcVecEnv`,
  core wrappers, typed `EpisodeStats` with pluggable trackers, `Renderer` protocol,
  and `check_env`.

[0.2.0]: https://github.com/QianCapital/sekai/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/QianCapital/sekai/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/QianCapital/sekai/releases/tag/v0.1.1
