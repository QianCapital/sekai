# Installation

sekai needs **Python 3.10 or newer**. The core package depends only on NumPy (≥ 1.24) and
cloudpickle (used to ship environment factories to `ProcVecEnv` workers).

```bash
pip install sekai
```

Check the install with the command-line tool:

```bash
sekai --version
sekai list
```

## Optional extras

| Extra | Installs | Needed for |
|---|---|---|
| `sekai[gymnasium]` | `gymnasium>=1.0` | [`sekai.compat`](../guide/gymnasium.md): `to_gymnasium()` / `from_gymnasium()`, Stable-Baselines3, CleanRL, RLlib |
| `sekai[render]` | `pillow`, `imageio`, `imageio-ffmpeg` | [saving frames](../guide/rendering.md#saving-gifs-and-videos): `save_gif()` (Pillow), `save_video()` (imageio + ffmpeg) |
| `sekai[pettingzoo]` | `pettingzoo>=1.24`, `gymnasium` | [`to_pettingzoo()` / `from_pettingzoo()`](../guide/multi-agent.md) for multi-agent libraries |
| `sekai[pandas]` | `pandas>=1.5` | [`MarketData.from_pandas()` / `from_long()`](../finance/data.md) |
| `sekai[all]` | `gymnasium`, `pettingzoo`, `pandas` and `render` extras | everything at runtime |
| `sekai[test]` | `pytest`, `pytest-cov`, `gymnasium`, `pettingzoo`, `pandas`, `pillow` | running the test suite |
| `sekai[docs]` | `mkdocs-material`, `mkdocstrings[python]` | building this site |
| `sekai[dev]` | `test` + `docs` + `mypy`, `ruff`, `pre-commit`, `build`, `twine` | contributing |

```bash
pip install "sekai[gymnasium,render]"
```

!!! note "Rendering needs no extras"
    Drawing frames uses sekai's own NumPy [`Canvas`](../guide/rendering.md#canvas), so
    `CartPoleRenderer().render(env)` works with a bare install. The `render` extra is only for
    *writing* GIF/MP4 files.

## From source

```bash
git clone https://github.com/QianCapital/sekai
cd sekai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev,gymnasium,render]"
pytest
```

See [Contributing](../contributing.md) for the full development workflow.

## Type checking

sekai ships a `py.typed` marker and type-checks clean under `mypy --strict`. Environments,
spaces and results are generic, so annotations like `sekai.Env[np.ndarray, np.intp]` and
`sekai.StepResult[np.ndarray]` flow through to your code.
