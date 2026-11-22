"""A torch BatchMarketEnv on a device reads back from it once per step.

Reading a value on a GPU (`.item()`, `.cpu()`, a Python `if` on a tensor) or copying host
data to it makes the host wait until the device has finished its queued work. So on a
device, a step reads back one thing: its finished lanes, together with its checks of the
action and of the cost models' inputs, and only then changes its state. These tests run
that path on the CPU (`env._defer = True`), count what would wait on a GPU, and check
that it gives exactly what the host path gives, errors included.
"""

from __future__ import annotations

import collections
import copy
import pickle
from typing import Any

import numpy as np
import pytest

from sekai.backend import get_backend
from sekai.core.snapshot import Snapshot
from sekai.envs.finance import BatchMarketEnv
from sekai.error import InvalidAction
from sekai.finance import (
    BorrowFee,
    FactorRiskModel,
    PortfolioVariance,
    ProportionalCost,
    SimpleReturn,
    SpreadCost,
    SquareRootImpact,
)
from sekai.finance.costs import _collect_checks
from tests.test_envs_market_v1 import DATA, GaussianScenarios

torch = pytest.importorskip("torch")
python_dispatch = pytest.importorskip("torch.utils._python_dispatch")

N_ENVS = 7
CONFIGS: dict[str, dict[str, Any]] = {
    "costs": dict(
        data=DATA,
        lookback=10,
        episode_length=30,
        features=("volume",),
        allow_short=True,
        transaction_cost=SpreadCost() + ProportionalCost(5e-5) + SquareRootImpact(0.2),
        holding_cost=BorrowFee(),
        shortable="shortable",
        max_drawdown=0.05,
    ),
    "dynamic": dict(
        data=DATA,
        lookback=5,
        episode_length=20,
        max_assets=6,
        allow_short=True,
        risk=FactorRiskModel(),
        execution="close",
        reward=SimpleReturn() - 2.0 * PortfolioVariance(),
    ),
    "margin": dict(
        data=DATA,
        lookback=8,
        episode_length=25,
        allow_short=True,
        max_leverage=4.0,
        initial_margin=0.5,
        maintenance_margin=0.48,
        execution="next_open",
    ),
    "scenarios": dict(
        scenarios=GaussianScenarios(),
        lookback=5,
        episode_length=12,
        max_drawdown=0.05,
        transaction_cost=SquareRootImpact(0.1),
    ),
}


def make(name: str, *, device_path: bool, dtype: Any = None, **kwargs: Any) -> BatchMarketEnv:
    kwargs = {**CONFIGS[name], **kwargs}
    env = BatchMarketEnv(N_ENVS, backend="torch", dtype=dtype or torch.float32, **kwargs)
    env._defer = device_path
    return env


def actions(env: BatchMarketEnv, n: int, seed: int = 0) -> list[Any]:
    shape = (N_ENVS, *env.action_space.shape)  # type: ignore[misc]
    if env.backend.name == "numpy":
        rng = np.random.default_rng(seed)
        return [0.6 * rng.normal(size=shape) for _ in range(n)]
    gen = torch.Generator().manual_seed(seed)
    return [0.6 * torch.randn(shape, generator=gen, dtype=env._like.dtype) for _ in range(n)]


def same(a: Any, b: Any) -> None:
    """Exactly equal results (NaN in the same places)."""
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same(a[k], b[k])
    elif a is None or b is None:
        assert a is b
    elif isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
    else:
        np.testing.assert_array_equal(a, b)


def rollout(env: BatchMarketEnv, steps: list[Any]) -> list[dict[str, Any]]:
    out = [vars(env.reset(seed=1))]
    for i, a in enumerate(steps):
        if not env.autoreset and i in (8, 16):  # restart the frozen lanes
            done = np.flatnonzero(env.backend.to_numpy(env.frozen))
            if len(done):
                out.append(vars(env.reset(indices=done)))
        out.append(vars(env.step(a)))
    return out


@pytest.mark.parametrize("autoreset", [True, False])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_the_device_path_gives_what_the_host_path_gives(
    name: str, dtype: str, autoreset: bool
) -> None:
    host = make(name, device_path=False, dtype=getattr(torch, dtype), autoreset=autoreset)
    device = make(name, device_path=True, dtype=getattr(torch, dtype), autoreset=autoreset)
    steps = actions(host, 40)
    for x, y in zip(rollout(host, steps), rollout(device, steps), strict=True):
        same(x, y)
    for env in (host, device):  # the host and backend copies of the lanes' state agree
        assert np.array_equal(env._frozen_host, env.backend.to_numpy(env.frozen))
        assert np.array_equal(env._t, env.backend.to_numpy(env._t_dev))
        assert np.array_equal(env._start, env.backend.to_numpy(env._start_dev))


class Waits(python_dispatch.TorchDispatchMode):  # type: ignore[misc]
    """Counts, while entered, what would make the host wait for a GPU: values read back
    ("read"), ops whose output size depends on the data ("shape") and host data made
    into tensors ("upload")."""

    SHAPE_OPS = frozenset(
        {"nonzero", "masked_select", "_unique2", "unique_dim", "unique_consecutive"}
    )

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        super().__init__()
        self.count: collections.Counter[str] = collections.Counter()
        self.active = False
        for owner, name, kind in (
            (torch.Tensor, "cpu", "read"),
            (torch.Tensor, "tolist", "read"),
            (torch, "as_tensor", "upload"),
            (torch, "tensor", "upload"),
            (torch, "from_numpy", "upload"),
            (torch, "asarray", "upload"),
        ):
            monkeypatch.setattr(owner, name, self._counting(getattr(owner, name), kind))

    def _counting(self, real: Any, kind: str) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if self.active and (kind == "read" or not isinstance(args[0], torch.Tensor)):
                self.count[kind] += 1
            return real(*args, **kwargs)

        return wrapper

    def __enter__(self) -> Waits:
        self.count.clear()
        self.active = True
        super().__enter__()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.active = False
        super().__exit__(*exc)

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        name = func.overloadpacket.__name__
        if name == "_local_scalar_dense":
            self.count["read"] += 1
        elif name in self.SHAPE_OPS:
            self.count["shape"] += 1
        elif name in ("index", "index_put", "index_put_", "_index_put_impl_"):
            index = args[1] if len(args) > 1 else ()
            if any(isinstance(i, torch.Tensor) and i.dtype == torch.bool for i in index or ()):
                self.count["shape"] += 1
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_a_step_on_a_device_reads_back_once(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    env = make(name, device_path=True)
    env.reset(seed=0)
    waits = Waits(monkeypatch)
    resets = 0
    for a in actions(env, 30):
        with waits:
            result = env.step(a)
        expected = {"read": 1}
        if result.reset_indices is not None:  # their indices, and new starts from the host
            resets += 1
            if env.data is not None:
                expected["upload"] = 2
        if env.data is not None or result.reset_indices is None:
            assert dict(waits.count) == expected
        else:  # scenarios drawn on the host are uploaded too
            assert waits.count["read"] == 1 and waits.count["shape"] == 0
    assert resets > 0


@pytest.mark.parametrize("name", ["costs", "dynamic", "margin"])
def test_a_reset_on_a_device_uploads_only_the_new_starts(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = make(name, device_path=True)
    waits = Waits(monkeypatch)
    with waits:
        env.reset(seed=0)
    assert dict(waits.count) == {"upload": 1}


@pytest.mark.parametrize("device_path", [False, True])
def test_a_non_finite_action_raises_before_the_step_changes_anything(device_path: bool) -> None:
    env = make("costs", device_path=device_path)
    env.reset(seed=0)
    first, second = actions(env, 2)
    env.step(first)
    twin = copy.deepcopy(env)
    bad = second.clone()
    bad[3, 2] = float("nan")
    with pytest.raises(InvalidAction):
        env.step(bad)
    same(vars(env.step(second)), vars(twin.step(second)))  # as if it never happened


@pytest.mark.parametrize("device_path", [False, True])
def test_unpriceable_trades_raise_the_cost_models_error(device_path: bool) -> None:
    no_spreads = DATA.with_fields(spread=np.full(np.shape(DATA.field("spread")), np.nan))
    env = make("costs", device_path=device_path, data=no_spreads, execution="close")
    env.reset(seed=0)
    twin = copy.deepcopy(env)
    a, zero = actions(env, 1)[0], torch.zeros((N_ENVS, *env.action_space.shape))
    with pytest.raises(ValueError, match="SpreadCost has no valid 'spread'"):
        env.step(a)
    same(vars(env.step(zero)), vars(twin.step(zero)))  # (no trades: nothing to price)


# What sekai 0.3.0 did not keep: an env it pickled or snapshotted has none of these.
NEW_IN_0_3_1 = frozenset(
    {
        "_t_dev",
        "_start_dev",
        "_paths_host",
        "_frozen_host",
        "_resetting",
        "_finished_host",
        "_defer",
    }
)


def as_saved_by_0_3_0(state: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in state.items() if k not in NEW_IN_0_3_1}


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_envs_pickled_or_snapshotted_by_0_3_0_carry_on(backend: str) -> None:
    """Unpickling or restoring 0.3.0's state rebuilds the copies it lacks; a restore into
    an env elsewhere in its episode must not keep that env's copies."""
    kw = dict(CONFIGS["costs"], autoreset=False)
    env = BatchMarketEnv(N_ENVS, backend=backend, **kw)
    env.reset(seed=2)
    steps = actions(env, 30)
    for a in steps[:9]:
        env.step(a)
    saved = BatchMarketEnv.__new__(BatchMarketEnv)
    saved.__dict__.update(as_saved_by_0_3_0(copy.deepcopy(vars(env))))
    unpickled = pickle.loads(pickle.dumps(saved))
    snapshot = env.snapshot()
    old_snapshot = Snapshot(snapshot.kinds, tuple(as_saved_by_0_3_0(s) for s in snapshot.layers))
    restored = BatchMarketEnv(N_ENVS, backend=backend, **kw)
    restored.reset(seed=7)
    restored.step(steps[0])
    restored.restore(old_snapshot)
    for a in steps[9:]:
        expected = vars(env.step(a))
        same(vars(unpickled.step(a)), expected)
        same(vars(restored.step(a)), expected)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_collected_cost_checks_give_the_same_costs_and_errors(backend: str) -> None:
    B = get_backend(backend)
    like = torch.zeros((), dtype=torch.float64) if backend == "torch" else None
    spread = B.asfloat([[np.nan, 1e-3, 2e-3], [np.nan, np.nan, np.nan]], like)
    trades = B.asfloat([[0.1, -0.2, 0.0], [0.0, 0.0, 0.0]], like)
    model = SpreadCost()
    eager = model(trades, spread=spread)
    with _collect_checks() as checks:
        collected = model(trades, spread=spread)
    same(collected, eager)
    assert len(checks) == 1 and not bool(checks[0][0])
    unpriced = B.asfloat([[0.1, -0.2, 0.0], [0.3, 0.0, 0.0]], like)  # trades on the empty row
    with pytest.raises(ValueError, match="no valid 'spread'") as error:
        model(unpriced, spread=spread)
    with _collect_checks() as checks:
        model(unpriced, spread=spread)
    assert bool(checks[0][0]) and checks[0][1] == str(error.value)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
def test_scalars_become_tensors_without_host_data(
    dtype: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Host scalars become 0-d tensors made on the device, bit for bit as copying them
    over would make them (rounding and overflow included)."""
    B, like = get_backend("torch"), torch.zeros((), dtype=getattr(torch, dtype))
    floats = [0.0, -0.0, 0.1, 1 / 3, 2 / 252, 1e-320, 1e39, -1e39, float("inf"), float("nan")]
    values = [*floats, 5, True, np.float32(0.1), np.float64(1e5), np.array(0.25), np.int64(3)]
    expected = [torch.as_tensor(np.asarray(v, dtype=np.float64), dtype=like.dtype) for v in values]
    waits = Waits(monkeypatch)
    with waits:
        got = [B.asfloat(v, like) for v in values]
        flags = [B.asbool(v, like) for v in (0, 1, True, np.bool_(False), np.array(2.0))]
        ints = [B.asint(v, like) for v in (0, -7, np.int64(2**40), np.array(9))]
    assert dict(waits.count) == {}
    bits = {2: torch.int16, 4: torch.int32, 8: torch.int64}[like.element_size()]
    for g, e in zip(got, expected, strict=True):
        assert g.dtype == like.dtype and g.shape == ()
        assert bool(torch.isnan(g)) == bool(torch.isnan(e))
        assert bool(torch.isnan(g)) or torch.equal(g.view(bits), e.view(bits))
    assert [bool(f) for f in flags] == [False, True, True, False, True]
    assert all(f.dtype == torch.bool for f in flags)
    assert [int(i) for i in ints] == [0, -7, 2**40, 9] and all(i.dtype == torch.int64 for i in ints)
    assert B.on_host(like) and get_backend("numpy").on_host(np.zeros(3))
