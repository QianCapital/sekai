# Command line

Installing sekai adds a `sekai` command; `python -m sekai` is equivalent.

```console
$ sekai --help
usage: sekai [-h] [--version] [-m MODULE] {list,info,check,bench} ...

sekai — RL environment framework command-line tools.

positional arguments:
  {list,info,check,bench}
    list                list registered environments
    info                show an environment's spec and spaces
    check               run sekai.check_env on an environment
    bench               measure random-action steps per second

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
  -m MODULE, --import MODULE
                        import MODULE first so its environments are registered
                        (repeatable)
```

| Command | Does |
|---|---|
| `sekai list [--namespace NS]` | list registered IDs with their time limit and entry point (`--namespace ""` for IDs without one) |
| `sekai info ID` | print the spec, spaces, metadata and the wrapper stack `make()` builds |
| `sekai check ID [--steps N] [--determinism] [--strict]` | run [`check_env`](checking.md) (500 steps by default) |
| `sekai bench ID [--steps N] [--num-envs N] [--mode auto\|native\|sync\|async\|proc] [--workers N] [--seed S]` | time random-action steps |
| `sekai --version` | print the installed version |

## list

```console
$ sekai list
ID                        MAX STEPS  ENTRY POINT
Acrobot-v1                      500  sekai.envs.classic_control.acrobot:AcrobotEnv
CartPole-v1                     500  sekai.envs.classic_control.cartpole:CartPoleEnv
MarketMaking-v0                   -  sekai.envs.finance.market_making:MarketMakingEnv
MountainCar-v0                  200  sekai.envs.classic_control.mountain_car:MountainCarEnv
MountainCarContinuous-v0        999  sekai.envs.classic_control.continuous_mountain_car:ContinuousMountainCarEnv
OptimalExecution-v0               -  sekai.envs.finance.optimal_execution:OptimalExecutionEnv
Pendulum-v1                     200  sekai.envs.classic_control.pendulum:PendulumEnv
Portfolio-v0                      -  sekai.envs.finance.portfolio:PortfolioEnv
Trading-v0                        -  sekai.envs.finance.trading:TradingEnv
```

## info

```console
$ sekai info CartPole-v1
id=CartPole-v1
entry_point=sekai.envs.classic_control.cartpole:CartPoleEnv
max_episode_steps=500
reward_threshold=475.0
observation_space=Box(-inf, inf, (4,), float32)
action_space=Discrete(2)
metadata={'render_fps': 50}
wrappers=TimeLimit -> CartPoleEnv
```

## check

```console
$ sekai check CartPole-v1 --steps 200 --determinism
CartPole-v1: all checks passed.
```

Each issue is printed on its own line. The exit status is suitable for CI:

| Exit code | Meaning |
|---|---|
| `0` | no errors (warnings are printed but tolerated without `--strict`) |
| `1` | at least one `ERROR`, or any issue with `--strict` |
| `2` | a usage error: the ID is not registered, a `-m` module cannot be imported, or an option is invalid |

```console
$ sekai check CartPol-v1
error: Environment 'CartPol-v1' not found in registry. Did you mean: 'CartPole-v1'? Use sekai.all_specs() or `sekai list` to see registered environments.
```

## bench

`bench` times `--steps` calls of `step()` with random actions after a short warm-up, resetting
finished episodes. With `--num-envs` above 1 (or an explicit `--mode` other than `sync`) it
builds a vector env with [`make_vec`](vector.md), and each timed call steps every sub-env, so
the reported total is `steps × num_envs`. The default `--mode auto` uses the environment's
[native vector implementation](vector.md#native-vector-environments) when it has one (marked
`[native vec]` by `sekai list`) and `SyncVecEnv` otherwise; `--workers` sets the number of
worker processes for `--mode proc` and is rejected with any other mode. `--steps`,
`--num-envs` and `--workers` (and `check --steps`) must be positive integers, and `--seed` a
non-negative one. Invalid
combinations, such as `--mode native` for an environment without a native implementation,
print `error: ...` and exit with status 2. Throughput depends on your machine; for example:

```console
$ sekai bench CartPole-v1
CartPole-v1 [single env]: 10,000 env steps in 0.06s = 180,972 steps/s
$ sekai bench CartPole-v1 --num-envs 8 --steps 5000
CartPole-v1 [SyncVecEnv x8]: 40,000 env steps in 0.27s = 149,814 steps/s
$ sekai bench Portfolio-v0 --num-envs 1024 --steps 200
Portfolio-v0 [PortfolioVecEnv x1024]: 204,800 env steps in 0.19s = 1,101,372 steps/s
$ sekai bench Portfolio-v0 --num-envs 1024 --mode sync --steps 20
Portfolio-v0 [SyncVecEnv x1024]: 20,480 env steps in 0.42s = 48,355 steps/s
$ sekai bench Portfolio-v0 --num-envs 16 --mode proc --workers 4 --steps 1000
Portfolio-v0 [ProcVecEnv x16]: 16,000 env steps in 0.70s = 22,769 steps/s
```

For simulators that vectorise, the native implementation is fastest by far. Among the
per-env backends, fast environments are quickest with `sync`: a `proc` step pays an
inter-process round trip, which only pays off when the environment's own step is expensive.

## Plugins

Environments registered by another package are visible once that package is imported.
`-m MODULE` (repeatable) imports modules before the command runs:

```python
# acme_envs.py
import sekai

sekai.register(
    "acme/LongOnlyTrading-v0",
    entry_point="sekai.envs.finance.trading:TradingEnv",
    kwargs={"positions": (0.0, 1.0)},
    max_episode_steps=126,
)
```

```console
$ python -m sekai -m acme_envs list --namespace acme
ID                       MAX STEPS  ENTRY POINT
acme/LongOnlyTrading-v0        126  sekai.envs.finance.trading:TradingEnv
$ python -m sekai -m acme_envs check acme/LongOnlyTrading-v0 --determinism
acme/LongOnlyTrading-v0: all checks passed.
```

!!! note "Where `-m` looks"
    The module must be importable: an installed package, or a module in the current directory
    (which `-m` adds to `sys.path`, as `python -m` does). A module that fails to import is
    reported as `error: could not import ...` with exit code 2.
