# API reference

Generated from the docstrings in the source. Everything listed is importable from the
top-level `sekai` package unless the page says otherwise.

| Page | Contents |
|---|---|
| [Core](core.md) | `Env`, `Wrapper` and its semantic subclasses, `MultiAgentEnv`, result dataclasses |
| [Spaces](spaces.md) | `Box`, `Discrete`, `MultiDiscrete`, `MultiBinary`, `Dict`, `Tuple`, `Sequence`, `MaskedBox` and `sekai.spaces` utilities |
| [Wrappers](wrappers.md) | every single-env wrapper in `sekai.wrappers` |
| [Vector](vector.md) | `SyncVecEnv`, `AsyncVecEnv`, `ProcVecEnv`, `NativeVecEnv`, results and vector wrappers |
| [Registry](registry.md) | `register`, `make`, `make_vec`, `EnvSpec`, ... |
| [Environments](envs.md) | built-in environment classes (and native vector versions), renderers and price simulators in `sekai.envs` |
| [Finance](finance.md) | the `sekai.finance` toolkit: market data, samplers, simulators, features, costs, accounts, rewards, metrics, universe slots, order book |
| [Rendering](rendering.md) | `sekai.rendering`: `Renderer`, `Canvas`, `RecordFrames`, `save_gif`, `save_video` |
| [Statistics](stats.md) | `EpisodeStats` and trackers |
| [Utilities](utils.md) | `check_env`, `RunningMeanStd`, `ExponentialMovingStats`, action-mask helpers, seeding helpers |
| [Compatibility](compat.md) | `sekai.compat`: the gymnasium and PettingZoo bridges |
| [Errors](errors.md) | `SekaiError` and its subclasses |
| [CLI](cli.md) | `sekai.cli` |

Names starting with an underscore are private and not covered by semantic-versioning
guarantees. `Env._set_rng()` is the one exception: it is the documented way for
environments to seed themselves.
