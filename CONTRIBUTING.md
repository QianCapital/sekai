# Contributing to sekai

Thanks for your interest in improving sekai! Bug reports, new environments, wrappers,
documentation fixes and API ideas are all welcome.

## Development setup

```bash
git clone https://github.com/QianCapital/sekai
cd sekai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,gymnasium,render]"
pre-commit install                                  # optional: run checks on every commit
```

## Checks

CI runs all of these on every pull request; run them locally first:

```bash
pytest                         # tests (add --cov for coverage)
ruff check sekai tests examples
ruff format sekai tests examples
mypy                           # strict mode, configured in pyproject.toml
mkdocs serve                   # preview the docs at http://127.0.0.1:8000
```

`pre-commit run --all-files` runs the linters and type checker in one go.

Some tests compare sekai against gymnasium (dynamics and rendering). They are skipped
when gymnasium or pygame is missing; install `gymnasium[classic-control]` to run them all.

## Project layout

| Path | What lives there |
|---|---|
| `sekai/core/` | `Env`, `Wrapper`, `MultiAgentEnv`, result dataclasses |
| `sekai/spaces/` | Spaces and the `flatten` / `batch_space` / `concatenate` utilities |
| `sekai/wrappers/` | Single-env wrappers, one per module |
| `sekai/vector/` | `SyncVecEnv`, `AsyncVecEnv`, `ProcVecEnv`, vector wrappers |
| `sekai/registry/` | `register`, `make`, `make_vec`, `EnvSpec` |
| `sekai/envs/` | Built-in environments (`classic_control/`, `finance/`), their native vector versions and registration |
| `sekai/finance/` | The finance toolkit: market data, samplers, simulators, features, costs, accounts, rewards, metrics, order book |
| `sekai/rendering/` | `Renderer` protocol, `Canvas`, `RecordFrames` |
| `sekai/compat/` | gymnasium interop |
| `sekai/cli.py` | the `sekai` command |
| `tests/` | pytest suite, mirroring the package layout |
| `docs/` | MkDocs site |
| `examples/` | runnable scripts (CI runs every one) |

## Adding things

**An environment.** Subclass `sekai.Env`, call `self._set_rng(seed)` at the top of
`reset()` and draw all randomness from `self.rng`. Register it in `sekai/envs/__init__.py`
with a string entry point (so importing sekai stays cheap). Add a test that runs
`sekai.check_env(env, check_determinism=True)` and asserts there are no issues, plus tests
for the env's own logic, and a page or section in the docs.

**A wrapper.** If it only transforms observations, rewards or actions, subclass
`ObservationWrapper`, `RewardWrapper` or `ActionWrapper`; the async path comes for free.
If it keeps state, implement both `step()` and `async_step()` (and the `reset` pair) so
async envs stay async through it. Export it from `sekai/wrappers/__init__.py` and `sekai/__init__.py`.

**A space.** Subclass `sekai.Space`, then register `flatdim`, `flatten`, `unflatten`,
`flatten_space`, `batch_space`, `concatenate` and `zero_sample` implementations with the
`singledispatch` hooks in `sekai.spaces.utils`, and `register_space_type()` for JSON.

**A native vector environment.** Subclass `sekai.NativeVecEnv` (state arrays with a leading
`num_envs` axis; implement `_reset_envs`, `_observe` and `_step`) and register it as the
env's `vector_entry_point`. Test that it matches the single env under `SyncVecEnv` wherever
the dynamics are deterministic (see `tests/test_envs_finance_market.py`).

**A finance component.** Cost models, reward functions, samplers and market models are small
classes in `sekai/finance/`; keep them vectorised over a leading batch axis so native vector
environments can use them, and check any data transform with `features.check_causal`.

## Pull requests

- Keep each PR focused on one change, with tests.
- Public API changes need docstrings, a docs update and a `CHANGELOG.md` entry.
- Behaviour changes that could affect existing users go under "Changed" in the changelog,
  with what users need to do.

## Releasing (maintainers)

1. Update `sekai/_version.py` and move the changelog entries under the new version and date.
2. Merge to `master` and wait for CI to pass.
3. Create a GitHub release with tag `vX.Y.Z`. The `Release` workflow checks the tag matches
   the version, builds the sdist and wheel, publishes to PyPI via trusted publishing and
   attaches the files to the release.
