# Market simulators

`sekai.finance.simulators` turns a random generator into price paths. Every model is
vectorised over paths and assets, so one call simulates a single market for an
environment or thousands for a native vector env, a Monte Carlo study or a synthetic
training set. Models can also report latent series (the market regime, stochastic
variance, jump counts) and describe a universe whose members list and delist.

## The MarketModel interface

```python
import numpy as np
import sekai
from sekai.finance import GBM

rng = np.random.default_rng(0)
model = GBM(mu=[0.05, 0.08], sigma=[0.2, 0.3], corr=[[1.0, 0.5], [0.5, 1.0]])
print(model)  # GBM(n_assets=2, mu=[0.05, 0.08], sigma=[0.2, 0.3], s0=100.0, corr=[[1.0, 0.5], [0.5, 1.0]], dt=1/252)

one = model.simulate(rng, 252)
print(one.prices.shape, one.mask, dict(one.state))  # (253, 2) None {}
batch = model.simulate(rng, 252, n_paths=1000)
print(batch.prices.shape)  # (1000, 253, 2)
```

`simulate(rng, n_steps, *, n_paths=None)` returns a `Simulation`:

| Attribute | Shape | Meaning |
|---|---|---|
| `prices` | `(n_steps + 1, n_assets)`, or `(n_paths, n_steps + 1, n_assets)` with `n_paths` | row 0 is the initial price; NaN where an asset is not listed |
| `mask` | like `prices`, or `None` | True where the asset is listed and tradable; `None` when every entry is |
| `state` | `{name: array}` | latent series; leading dimensions `prices.shape[:-1]`, then the per-step shape from `state_spec()` |

`n_paths=None` gives one path without a paths axis; an int (even 1) adds one. Every model
also has `n_assets`, `dt` (the step length in years) and `state_spec()`, which maps state
keys to their per-step shapes, e.g. `{"variance": (n_assets,)}`.

Conventions shared by every built-in model:

- Parameters are annualised and `dt` is in years (default `1/252`, one trading day).
- Per-asset parameters take a scalar or a length-`n_assets` array; `n_assets` is inferred
  from them (or from `corr`), and is needed only to broadcast all-scalar parameters to
  several assets.
- `corr` (where offered) correlates the assets' Gaussian shocks.
- Models are immutable; build a new one to change a parameter.
- All randomness comes from the `rng` you pass, so a seeded generator reproduces a path.

## Models

| Model | Dynamics | `state` |
|---|---|---|
| [`GBM`](#gbm) | log-normal random walk | none |
| [`Heston`](#heston) | stochastic variance, leverage effect | `"variance"` `(n_assets,)` |
| [`MertonJumpDiffusion`](#mertonjumpdiffusion) | GBM plus Poisson log-normal jumps | `"jumps"` `(n_assets,)` |
| [`RegimeSwitching`](#regimeswitching) | GBM whose parameters follow a hidden Markov chain | `"regime"` `()` |
| [`GARCH`](#garch) | GARCH(1,1) volatility clustering | `"variance"` `(n_assets,)` |
| [`MeanReverting`](#meanreverting) | exponential Ornstein–Uhlenbeck | none |
| [`CointegratedPair`](#cointegratedpair) | common trend plus a stationary spread | `"spread"` `()` |
| [`BlockBootstrap`](#blockbootstrap) | resampled blocks of historical returns | none |
| [`ListingProcess`](#dynamic-universes) | wraps a model; assets list and delist | the inner model's |
| [`Randomized`](#domain-randomisation) | wraps a factory; new parameters per path | the models' |

The tiny examples below check each model against its defining property.

### GBM

Geometric Brownian motion, simulated with the exact log-normal transition:

$$
\log S_{t+1} = \log S_t + \big(\mu - \tfrac12\sigma^2\big)\Delta t + \sigma\sqrt{\Delta t}\,Z_t,
\qquad Z_t \sim \mathcal N(0, \Sigma).
$$

Without `n_paths` it draws random numbers in the same order as
`sekai.envs.finance.simulate_gbm`, so both give identical paths for the same seed.

```python
returns = np.diff(np.log(batch.prices), axis=1)
print((returns.std(axis=(0, 1)) * np.sqrt(252)).round(3))  # [0.2 0.3]
print(np.corrcoef(returns[..., 0].ravel(), returns[..., 1].ravel())[0, 1].round(2))  # 0.5
```

### Heston

Stochastic variance $v_t$ that mean-reverts to $\theta$, with price and variance shocks
correlated by $\rho$ (negative $\rho$ gives the leverage effect: prices fall as volatility
rises). Simulated with full-truncation Euler, $v^+ = \max(v, 0)$:

$$
\log S_{t+1} = \log S_t + \big(\mu - \tfrac12 v_t^+\big)\Delta t + \sqrt{v_t^+\Delta t}\,Z^1_t, \qquad
v_{t+1} = v_t + \kappa\big(\theta - v_t^+\big)\Delta t + \xi\sqrt{v_t^+\Delta t}\,Z^2_t .
$$

```python
from sekai.finance import Heston

heston = Heston(mu=0.05, kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, n_assets=2)
sim = heston.simulate(rng, 252, n_paths=1000)
variance = sim.state["variance"]  # (1000, 253, 2)
print(variance.mean().round(4))  # 0.0399
returns = np.diff(np.log(sim.prices), axis=1)
print(np.corrcoef(returns.ravel(), np.diff(variance, axis=1).ravel())[0, 1].round(2))  # -0.7
```

### MertonJumpDiffusion

GBM plus $N_t \sim \text{Poisson}(\lambda\Delta t)$ jumps per step, each a log-normal
$J \sim \mathcal N(m, s^2)$ in log price; the drift is compensated by
$k = e^{m + s^2/2} - 1$ so that $\mathbb E[S_t] = S_0 e^{\mu t}$:

$$
\log S_{t+1} = \log S_t + \big(\mu - \tfrac12\sigma^2 - \lambda k\big)\Delta t
+ \sigma\sqrt{\Delta t}\,Z_t + \sum_{j=1}^{N_t} J_j .
$$

```python
from sekai.finance import MertonJumpDiffusion

merton = MertonJumpDiffusion(mu=0.05, sigma=0.2, jump_intensity=1.0, jump_mean=-0.05, jump_std=0.1)
sim = merton.simulate(rng, 252, n_paths=2000)
print(sim.state["jumps"].sum(axis=1).mean().round(2))  # 0.97
r = np.diff(np.log(sim.prices[..., 0]), axis=1).ravel()
print(round(((r - r.mean()) ** 4).mean() / r.var() ** 2, 1))  # 40.1
```

About one jump a year, and daily returns with a kurtosis far above the Gaussian 3.

### RegimeSwitching

A hidden Markov chain $k_t$ with per-step transition matrix $P$ picks which `GBM`'s
$(\mu, \sigma, \Sigma)$ drives the increment from $t$ to $t+1$:

$$
\Pr(k_{t+1} = j \mid k_t = i) = P_{ij}, \qquad
\log S_{t+1} = \log S_t + \big(\mu_{k_t} - \tfrac12\sigma_{k_t}^2\big)\Delta t + \sigma_{k_t}\sqrt{\Delta t}\,Z_t .
$$

The chain starts from `initial` (default: the stationary distribution of $P$). The regimes
must share `n_assets`, `dt` and `s0`.

```python
from sekai.finance import RegimeSwitching

calm = GBM(mu=0.10, sigma=0.12)
stress = GBM(mu=-0.30, sigma=0.45)
regimes = RegimeSwitching([calm, stress], [[0.99, 0.01], [0.04, 0.96]])
print(regimes.initial)  # [0.8 0.2]
sim = regimes.simulate(rng, 25_200)
regime = sim.state["regime"]  # 0.0 or 1.0 per time
r = np.diff(np.log(sim.prices[:, 0]))
print(regime.mean().round(2), [round(float(r[regime[:-1] == k].std() * np.sqrt(252)), 2) for k in (0, 1)])
```

```text
0.21 [0.12, 0.46]
```

### GARCH

GARCH(1,1) on the per-step log return: large moves raise the next step's variance, so
volatility clusters. $\omega$ is a per-step variance, and the variance starts at its
unconditional level $\omega / (1 - \alpha - \beta)$:

$$
\log S_{t+1} = \log S_t + \mu\Delta t + \varepsilon_t, \qquad
\varepsilon_t = \sqrt{h_t}\,Z_t, \qquad
h_{t+1} = \omega + \alpha\,\varepsilon_t^2 + \beta\,h_t .
$$

```python
from sekai.finance import GARCH

garch = GARCH(omega=1e-6, alpha=0.08, beta=0.9)
print(np.sqrt(garch.unconditional_variance * 252).round(3))  # [0.112]
r = np.diff(np.log(garch.simulate(rng, 25_200).prices[:, 0]))
print(np.corrcoef(r[1:], r[:-1])[0, 1].round(2), np.corrcoef(r[1:] ** 2, r[:-1] ** 2)[0, 1].round(2))
```

```text
0.01 0.22
```

Returns are uncorrelated, their squares are not.

### MeanReverting

An exponential Ornstein–Uhlenbeck process: the log price $x = \log S$ reverts to
$m = \log(\text{mean})$ at speed $\theta$, discretised exactly,

$$
x_{t+1} = m + (x_t - m)\,e^{-\theta\Delta t} + \sigma\sqrt{\frac{1 - e^{-2\theta\Delta t}}{2\theta}}\;Z_t ,
$$

so a deviation halves in $\ln 2 / \theta$ years.

```python
from sekai.finance import MeanReverting

reverting = MeanReverting(theta=5.0, mean=100.0, sigma=0.2, s0=120.0)
half_life = round(np.log(2) / 5.0 * 252)
paths = reverting.simulate(rng, 252, n_paths=1000).prices[..., 0]
print(half_life, np.median(paths[:, half_life]).round(1), np.median(paths[:, -1]).round(1))  # 35 109.3 100.1
```

Halfway, in log terms, from 120 to 100 after one half-life.

### CointegratedPair

Two assets share a GBM trend $x_t$ and differ by a zero-mean OU spread $s_t$ (exact
discretisation, speed $\theta$, volatility `spread_sigma`):

$$
\log P^1_t = \log P^1_0 + x_t + \tfrac12 s_t, \qquad
\log P^2_t = \log P^2_0 + x_t - \tfrac12 s_t,
$$

so each price wanders while $\log(P^1_t / P^2_t) - \log(P^1_0 / P^2_0) = s_t$ mean-reverts:
the textbook pairs-trading market.

```python
from sekai.finance import CointegratedPair

pair = CointegratedPair(mu=0.05, sigma=0.2, theta=5.0, spread_sigma=0.1)
sim = pair.simulate(rng, 252)
print(np.allclose(np.log(sim.prices[:, 0] / sim.prices[:, 1]), sim.state["spread"]))  # True
```

### BlockBootstrap

Resamples a history instead of assuming a distribution: each path concatenates blocks of
`block_size` consecutive rows of historical log returns, each block starting at a random
row (wrapping around circularly). Whole rows are drawn, so cross-asset correlation and
short-range dependence survive; dependence longer than a block does not.

```python
from sekai.finance import BlockBootstrap

historical = np.diff(np.log(model.simulate(rng, 1000).prices), axis=0)  # stands in for real returns
bootstrap = BlockBootstrap(historical, block_size=20, s0=100.0)
sim = bootstrap.simulate(rng, 252, n_paths=500)
r = np.diff(np.log(sim.prices), axis=1)
print(sim.prices.shape, np.corrcoef(r[..., 0].ravel(), r[..., 1].ravel())[0, 1].round(2))  # (500, 253, 2) 0.49
```

## Dynamic universes

`ListingProcess(model, *, initial_listed=0.8, listing_rate=1.0, delisting_rate=0.25,
min_listed=1)` lists and delists the assets of an inner model over time. Each asset follows
an independent two-state Markov chain: listed at $t = 0$ with probability `initial_listed`,
then after each step

$$
\Pr(\text{list}) = 1 - e^{-\lambda\Delta t}, \qquad \Pr(\text{delist}) = 1 - e^{-\delta\Delta t},
$$

with $\lambda$ = `listing_rate` and $\delta$ = `delisting_rate` per year. In the long run
an asset is listed a fraction $\lambda / (\lambda + \delta)$ of the time. Whenever fewer
than `min_listed` assets are listed, the lowest-index unlisted ones are listed. Prices come
from the inner path (NaN while unlisted, `mask` False), so an asset that relists resumes
from where its inner path has got to; the inner model's state passes through.

```python
from sekai.finance import ListingProcess

universe = ListingProcess(GBM(mu=0.05, sigma=0.25, n_assets=8), initial_listed=0.5,
                          listing_rate=1.0, delisting_rate=0.5, min_listed=2)
sim = universe.simulate(rng, 2520)
listed = sim.mask.sum(axis=1)
print(sim.mask.mean().round(2), listed.min(), listed.max())  # 0.64 2 7
```

`Portfolio-v0` and `Trading-v0` need assets that are always listed; a changing universe
is what [`Market-v0`](../environments/finance.md#market-v0) is for: its observation is a
padded set of asset slots with a validity mask, and its action mask marks the tradable ones
(see also the [dynamic observations guide](../guide/dynamic.md)).

## Domain randomisation

A policy trained on one parameter set overfits it. `Randomized(factory)` builds a fresh
model for every simulated path from `factory(rng)`, a function that draws parameters:

```python
from sekai.finance import Randomized


def random_heston(rng):
    return Heston(mu=rng.uniform(-0.05, 0.15), theta=rng.uniform(0.01, 0.09),
                  xi=rng.uniform(0.1, 0.6), rho=rng.uniform(-0.9, -0.3), n_assets=3)


market = Randomized(random_heston)
sim = market.simulate(rng, 252, n_paths=4)
v0 = sim.state["variance"][:, 0, 0]  # the initial variance is theta, drawn per path
print(v0.round(3))
```

```text
[0.021 0.064 0.044 0.035]
```

The model built for `np.random.default_rng(0)` is kept as `market.reference` and fixes
`n_assets`, `dt` and `state_spec()`; every later model must match them. In an environment,
each `reset()` simulates one path, so every episode gets new parameters.

## Latent state: observable or privileged

A model's `state` is information an agent would not normally have. `Market-v0` can expose
it in two ways (state keys come from `state_spec()`; the default market's is `"regime"`):

- **Observable:** `exogenous=["regime"]` adds a `"market"` window of the series to the
  observation, as if the agent could see it. Use it to measure the value of information,
  or for state that is genuinely public.
- **Privileged:** `privileged_state=True` adds `info["state"]`, the state at the new
  decision time, without changing the observation. Asymmetric actor-critic methods give it
  to the critic during training while the actor (the deployed policy) never sees it.

```python
observable = sekai.make("Market-v0", exogenous=["regime"])
print(observable.observation_space["market"])  # Box(-inf, inf, (20, 1), float32)

privileged = sekai.make("Market-v0", privileged_state=True)
result = privileged.reset(seed=0)
print(sorted(result.observation), result.info["state"])  # ['account', 'assets', 'portfolio'] {'regime': array(0.)}
```

The [Market-v0 baselines](../environments/finance.md#market-v0) compare a policy that
sees the regime with ones that do not.

## Using a model in an environment

`Trading-v0` takes any one-asset model and `Portfolio-v0` any model whose assets are always
listed, through `model=`; `Market-v0` takes any model at all:

```python
trading = sekai.make("Trading-v0", model=Heston(theta=0.04, xi=0.5))
portfolio = sekai.make("Portfolio-v0", model=market, lookback=20, episode_length=126)
print(portfolio.n_assets, portfolio.observation_space["returns"])  # 3 Box(-inf, inf, (20, 3), float32)

try:
    sekai.make("Portfolio-v0", model=universe).reset(seed=0)
except ValueError as error:
    print(error)  # PortfolioEnv needs a model whose assets are always listed; use Market-v0 for changing universes.

dynamic = sekai.make("Market-v0", model=universe)
print(dynamic.observation_space["assets"])  # MaskedBox(-inf, inf, (20, 2), max_size=8, float32)
```

Every `reset()` simulates a new path of `lookback + episode_length` steps with the env's
generator, so a seed reproduces an episode. The native vector versions of `Trading-v0` and
`Portfolio-v0` simulate all their sub-envs in one `simulate(..., n_paths=num_envs)` call.

To compute features on simulated data, or to split it chronologically, simulate a history
into a `MarketData` first and use it as data (see [Market data](data.md)):

```python
from sekai.finance import MarketData

sim = universe.simulate(np.random.default_rng(1), 1259)
dates = np.busday_offset("2020-01-01", np.arange(1260), roll="forward")
synthetic = MarketData(sim.prices, timestamps=dates)  # NaN prices = unlisted
print(synthetic, synthetic.mask.all())  # MarketData(T=1260, N=8, fields=['close'], 2020-01-01 .. 2024-10-29) False
```

## Writing your own model

Subclass `MarketModel` and implement `n_assets`, `dt` and `simulate()` (and `state_spec()`
if the model reports state). `simulate` must be vectorised over paths, honour the
`n_paths=None` convention and return positive prices wherever an asset is listed. A model
with Student-t returns, for fat tails without jumps:

```python
from sekai.finance import MarketModel, Simulation


class StudentT(MarketModel):
    def __init__(self, sigma=0.2, dof=4.0, s0=100.0, dt=1 / 252):
        self.sigma, self.dof, self.s0, self._dt = sigma, dof, s0, dt

    @property
    def n_assets(self):
        return 1

    @property
    def dt(self):
        return self._dt

    def simulate(self, rng, n_steps, *, n_paths=None):
        lead = () if n_paths is None else (n_paths,)
        scale = self.sigma * np.sqrt(self._dt * (self.dof - 2) / self.dof)  # unit variance
        steps = scale * rng.standard_t(self.dof, size=(*lead, n_steps, 1))
        log_paths = np.concatenate([np.zeros((*lead, 1, 1)), np.cumsum(steps, axis=-2)], axis=-2)
        return Simulation(prices=self.s0 * np.exp(log_paths))


env = sekai.make("Trading-v0", model=StudentT())
print(env.reset(seed=0).observation.shape)  # (31,)
```

## See also

- [Finance environments](../environments/finance.md): `model=` in each environment.
- [`sekai.finance.simulators` API](../api/finance.md#simulators).
