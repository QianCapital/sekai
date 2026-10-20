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
  `action_masks()` for sb3-contrib's MaskablePPO. `MultiDiscrete.sample(mask=...)` and
  `flatten_action_mask()` reject mask entries with the wrong number of flags, and
  `RescaleAction` passes the env's zero for the components the env masks.
- Wrappers `PadSequenceObservation` (Sequence → MaskedBox), `ActionMaskObservation`, and
  `DelayAction` (order latency: actions execute `delay` steps after they are chosen).
- `FrameStackObservation` stacks any batchable observation space (Dict, Tuple, MaskedBox,
  Discrete, ...), leaf by leaf, with `padding="reset"` or `"zeros"`.
- `NormalizeObservation` / `VecNormalizeObservation` normalise nested observation spaces (each Box
  leaf, MaskedBox leaves per feature over occupied slots only, other leaves passed through), can
  use exponentially weighted statistics for non-stationary data (`stats="ewm", halflife=...`;
  the half-life counts steps, so it needs no scaling with `num_envs`), clip (`clip=`), and
  save/restore with `state_dict()` / `load_state_dict()`.
  `sekai.ExponentialMovingStats` is the EWM counterpart of `RunningMeanStd`.

**Vectorisation**

- Native vector environments: `sekai.NativeVecEnv` is a base class for environments that advance
  all sub-envs at once with array operations (seeding, per-env options, same-step autoreset and
  per-env infos are handled for you). Register one with
  `register(..., vector_entry_point=...)`; `make_vec()` then uses it by default
  (`vectorization_mode=None` → `"native"` when available, else `"sync"`). Native envs take
  `max_episode_steps` and truncate exactly like `TimeLimit` (`infos[i]["elapsed_steps"]` and
  `get_attr("elapsed_steps")` included), and `set_attr()` raises `AttributeError` for names the
  env doesn't have. They raise `ResetRequired` when stepped before `reset()` and accept the
  same seeds as every VecEnv. A member that is the same for every sub-env can be marked
  `@sekai.vector.shared`, so `get_attr()` / `call()` return it whole instead of splitting it
  by sub-env. An ID registered with `additional_wrappers` vectorises with `SyncVecEnv`
  (explicit `"native"` raises), since a native env can't apply them.
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
  (fixed-size spaces, on by default) and several sub-envs can share a worker process. A dead
  worker that hosts several sub-envs is reported as `Worker w (sub-envs a-b) raised ...`.
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
  An integer always means a row position (counted within the object, negative from the end)
  in `index_of()`, `between()`, `split()` and `reset(options={"start": ...})`, on dated and
  numeric time axes alike; `sekai.finance.Time(t)` looks up a timestamp by value on any axis
  (`options={"start": Time(info["timestamp"])}` replays an episode). A date on numeric
  timestamps, or a float on datetime ones, raises `ValueError` instead of picking a row.
- Episode samplers (`RandomStart`, `SequentialStart`, `FixedStart`) and
  `reset(options={"start": ...})` (a row, a date or `Time(t)`) for every historical environment
  (a start on a warm-up row raises `ValueError`).
- Market simulators, batch-first with latent state: GBM, Heston, Merton jump-diffusion, regime
  switching, GARCH(1,1), exponential Ornstein-Uhlenbeck, cointegrated pairs, block bootstrap,
  `Randomized` (domain randomisation) and `ListingProcess` (assets list and delist).
- Causal feature transforms (rolling/EWM statistics, momentum, cross-sectional ranks and
  z-scores), point-in-time `asof_join` with publication lags, and `check_causal`, which detects
  look-ahead bias (raising the new `sekai.LookAheadError`) by rescaling, reordering,
  sign-flipping and re-masking the future and checking the past is unaffected. Every check
  includes the first row as a cut point, and an output whose shape depends on the future also
  raises `LookAheadError`.
- Cost models (`ProportionalCost`, `SpreadCost`, `FixedCost`, `SquareRootImpact`, composable with
  `+`), weight-space accounting (`AccountConfig`: leverage, shorting, cash, margin and borrow rates),
  and reward functions (`LogReturn`, `ProfitAndLoss`, `MeanVariance`, `DifferentialSharpe`,
  `DrawdownPenalty`, `TurnoverPenalty`, composable arithmetically). Cost models never price a
  trade at zero because an input is missing: they raise unless given `fallback_spread=` /
  `fallback_volatility=`. Only NaN (or a negative value) counts as missing; a finite zero
  spread or volatility is valid and costs nothing. Environments reject a volatility-based cost
  with `lookback=1`, fallback or not. Cost models and `AccountConfig` reject NaN, infinite
  and (where meaningless) negative parameters.
  `ProfitAndLoss` uses `StepOutcome.pnl`, the exact change in equity, so its rewards sum to
  final equity − 1 even through a wipe-out.
- Performance metrics (Sharpe, Sortino, Calmar, drawdown, VaR/CVaR, ...), `PerformanceSummary`,
  and the `RecordPerformance` wrapper (`info["performance"]` at the end of each episode).
  `returns_from_equity()` inverts `equity_curve()` also through a wipe-out, and
  `pnl_summary(..., scale=)` (which `RecordPerformance` uses in P&L mode) measures rounding
  against the wealth level, so a constant P&L reports zero volatility, not a huge Sharpe ratio.
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
    (`avellaneda_stoikov_action()`, `almgren_chriss_actions()`). Trading and Portfolio take
    exactly one price source: raw prices (`prices=`, arrays or pandas, where a
    `DatetimeIndex` keeps its dates), a `MarketData` (`data=`, the keyword Market-v0 uses
    too) or any market `model=`; plus any cost model, a `reward=` function and a `sampler=`.
    Portfolio-v0's `n_assets` must agree with the history or model when given.
    In both, positions drift with prices between steps and costs are charged on the trade back
    to the target, so they agree exactly on one asset.
  - Episode horizons follow one rule: environments whose horizon is part of the task
    (market making, optimal execution, the order book) observe time and terminate at the
    horizon; those whose episode is a window of an ongoing market (`Trading-v0`,
    `Portfolio-v0`, `Market-v0`) truncate and don't observe time (add `TimeAwareObservation`
    to observe it).
  - Finance environments raise `ResetRequired` when stepped after the episode ended, check
    their numeric parameters at construction (`ValueError` naming the parameter), and copy
    a `reward_fn` set later (also through `set_attr`), so vector sub-envs never share a
    stateful reward. Read-only accessors expose the current state to renderers and loggers:
    `timestamp` (Trading, Portfolio, Market), `price_history` (Trading, Portfolio),
    `latent_state` (Market) and `MarketMaking-v0`'s last quotes and fills (`bid`, `ask`,
    `bid_filled`, `ask_filled`).
  - `Market-v0`: multi-asset trading over a changing universe (padded-set observations, action
    masks, stable slots), with data fields and exogenous series as features, cost models,
    financing, leverage and shorting, pluggable rewards and optional privileged (latent) state;
    simulated by default from a two-regime market whose assets list and delist.
  - `LimitOrderBook-v0`: event-driven market making and trading in a simulated order book,
    and `LimitOrderBookMarket`, a multi-agent market in which traders act on their own clocks.
    A trader's market order first cancels its own resting quote on the other side
    (self-trade prevention), and `LimitOrderBookMarket` checks every action before carrying
    out any.
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
  env (`CartPoleRenderer`, ...), the `RecordFrames` wrapper and `save_gif()` / `save_video()`,
  which reject float frames, values outside [0, 255] and mixed frame sizes before writing.
- **Finance renderers** (`sekai.envs.finance`): `TradingRenderer`, `PortfolioRenderer`,
  `MarketRenderer`, `MarketMakingRenderer`, `OptimalExecutionRenderer`,
  `LimitOrderBookRenderer` and `LimitOrderBookMarketRenderer` draw each episode as a dashboard:
  prices (with the lookback window), positions shaded and trades marked, equity against a
  benchmark, allocations over time, the changing universe, quotes and fills, the
  Almgren–Chriss schedule, and the order book as a depth ladder with the agent's orders at
  their place in the queue. They take `width`, `height`, `theme` and `window` (a scrolling
  time axis), and draw a frame in 10–20 ms.
- **Dashboards** (`sekai.rendering.Dashboard`): renderers for any environment, assembled from
  charts that read its state through sources (attribute paths or functions) and record it per
  episode. A new episode is detected from `elapsed_steps`, with one history per env. The charts
  are `LineChart` with `Series` (rebased, signed, step or shaded), `BarChart` and
  `StackedArea`; `sekai.finance.rendering` adds `PriceChart`, `OrderBookChart` and
  `UniverseChart`. Dashboards also provide `Row` / `Column` layouts, a header of `Stat`s with
  their change since the start, a time axis shared across charts, and `Panel` for charts of
  your own. `sekai.rendering.charts` holds the drawing helpers (`Plot`, `nice_ticks`,
  `format_number`, ...).
- **Themes**: `sekai.rendering.LIGHT` and `DARK`, in the Qian Capital palette. Colours are named
  by role (the `up` / `down` polarity pair, `accent`, `highlight`, ...), with an eight-colour
  categorical palette checked for colour-vision deficiency. `Theme.with_colors()` adapts a
  theme.
- `Canvas(..., antialias=True)` draws shapes with exact edge coverage, and every shape takes
  `alpha=`. Canvas also gains `clip()`, rounded `rect(radius=)`, dashed lines,
  `fill_between()`, `image()` and `text()`. Text uses BDO Grotesk glyphs rasterised ahead of
  time (9–32 px, regular and bold, kerned, with tabular digits), shipped under the SIL Open
  Font License, so the package's license expression is now `Apache-2.0 AND OFL-1.1`.
  `sekai.rendering.text_width()` measures text. Without `antialias`, existing shapes draw
  exactly as before.
- `save_gif()` gives an RGB GIF's frames one shared palette, so each frame stores only what
  changed (animated charts come out about half the size) and colours can't flicker. Frames
  with 256 colours or fewer are saved exactly.
- `examples/render_finance.py` records every finance environment with a baseline policy; it
  produced the documentation's animations.
- **Wrappers**: `ClipReward` (`None` leaves a side unbounded), `RescaleObservation`,
  `DtypeObservation`, `FilterObservation`, `TimeAwareObservation`, `DelayObservation`,
  `StickyAction`, `ActionRepeat`. Stateful wrappers (`TimeLimit`, `TimeAwareObservation`,
  `FrameStackObservation`, `DelayObservation`, `DelayAction`, `StickyAction`) warn when built
  around an `AutoReset`, which must be the outermost wrapper.
- **Vector wrappers**: `VecRecordEpisodeStatistics` (each episode recorded once, with
  autoreset on or off), `VecNormalizeObservation`, `VecNormalizeReward`,
  `VecTransformObservation` and `VecTransformReward` (which take `func`, like gymnasium's).
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
  `Dict.seed()` / `Tuple.seed()` take per-sub-space seeds (a dict / list), as their
  constructors do.
- `check_env`: `check_determinism` is now implemented; new checks for the async API,
  `Space` types, observation dtypes and observations mutated in place after being returned.
  It checks every `reset()` (between episodes and `reset(seed=0)` too; one that raises is
  reported as an error), warns when `reset()` overwrites the terminal observation in place,
  validates action masks with `flatten_action_mask()`, and records copies for the
  determinism check.
- `ResetResult.done` (always `False`), so `result = env.reset()` /
  `while not result.done: result = env.step(...)` works with one variable. Likewise
  `MAResetResult.all_done` / `any_done` and `VecResetResult.dones`.
- `MultiAgentEnv` gets the same seeding API as `Env` (`rng`, `_set_rng(seed)`).
- `Env.set_attr()`; `sekai.rendering.BaseRenderer` (no-op `close()` plus `with` support);
  `elapsed_steps` on the finance envs.
- sekai exceptions can be pickled (e.g. raised inside a `ProcessPoolExecutor` job).
- `sekai.RunningMeanStd`, with `state_dict()` / `load_state_dict()` to save normalisation stats.
  Its first batch sets the mean (no variance bias for data far from zero), and empty batches
  are ignored.
- A `sekai` command-line tool (also `python -m sekai`): `list`, `info`, `check`, `bench`.
- `py.typed`: sekai now ships type information, and type-checks clean under `mypy --strict`.
- A test suite, GitHub Actions CI (Python 3.10–3.14 on Linux, plus macOS and Windows),
  trusted-publishing release workflow, MkDocs documentation site (Qian Capital theme),
  runnable examples.

### Changed

These change behaviour for code written against 0.1.x:

- **Vector envs reset finished sub-envs automatically** (`autoreset=True`), AutoReset-style:
  the returned observation starts the new episode, and the terminal observation/info are in
  `infos[i]["final_observation"]` / `["final_info"]`. Sub-envs already wrapped in
  `AutoReset` are not reset twice. Pass `autoreset=False` to manage resets yourself: a
  finished sub-env is then frozen until the next `reset()` (where 0.1.x stepped it again).
  It is not stepped, its action is ignored, and each later step returns its final
  observation, reward 0, its final `terminated` / `truncated` flags and an empty info, so
  `while not finished.all()` evaluation loops work with every backend and env.
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
- A custom `Space` subclass without `__eq__` compares by identity (it used to equal every
  instance of its type), and `MultiDiscrete` equality includes the dtype.
- `Dict` / `Tuple` raise `ValueError` for seeds that name unknown keys or extra sub-spaces,
  instead of ignoring them.
- Wrapper argument validation raises `TypeError` / `ValueError` instead of `AssertionError`
  (asserts disappear under `python -O`). `Wrapper(...)` rejects objects that aren't `sekai.Env`.
- Optional dependency groups: `all` now installs the runtime extras
  (`gymnasium`, `pettingzoo`, `pandas`, `render`); development tools moved to `dev`, tests to `test`.

### Deprecated

- `sekai.backend` (`BackendOps`, `NumpyBackend`) warns on import and will be removed in 0.3. Nothing
  in sekai used it; for fast batched simulation write a native vector environment instead.
  `sekai.BackendMismatch` goes with it: it is no longer in `sekai.__all__` and warns when used.
- The `fn=` keyword of `TransformObservation` / `TransformReward`: the argument is now `func`,
  gymnasium's name (positional calls are unaffected). `fn=` warns and will be removed in 0.3.

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
- `SyncVecEnv`, `AsyncVecEnv` and `ProcVecEnv` leaked the sub-envs already built when building
  another one failed; they are closed before the error propagates.
- `ProcVecEnv` left an "Exception ignored ... BufferError" message at exit after an error.
- `unflatten()` could return values outside the space (a float64 `Box` just above its upper
  bound; an int64 `Box` value near the dtype's maximum with its sign flipped). `Box` values
  are now clipped into the bounds, integer ones rounded without overflow, and `MultiBinary`
  values thresholded at 0.5.
- Vector envs dropped every action when the action space contained an empty `Dict` / `Tuple`.
- `MultiBinary.contains()` and `MultiDiscrete.contains()` raised on ragged input instead of
  returning `False`.
- `Box` kept NumPy integers in its shape, so its `to_jsonable()` output could not be
  serialised to JSON.

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
