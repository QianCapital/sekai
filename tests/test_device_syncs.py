"""A torch BatchMarketEnv on a device reads back from it once per step.

Reading a value on a GPU (`.item()`, `.cpu()`, a Python `if` on a tensor, the check of
`torch.linalg.solve`) or copying host data to it makes the host wait until the device has
finished its queued work. So on a device, a step reads back one thing: its finished
lanes, together with its checks of the action, of its projection and of the cost models'
inputs, and only then changes its state. These tests run that path on the CPU
(`env._defer = True`), count what would wait on a GPU, and check that it gives exactly
what the host path gives, errors included.
"""

from __future__ import annotations

import collections
import copy
import math
import pickle
import warnings
from typing import Any

import numpy as np
import pytest

from sekai.backend import get_backend
from sekai.backend._base import _collect_checks
from sekai.core.snapshot import Snapshot
from sekai.envs.finance import BatchMarketEnv
from sekai.error import InvalidAction
from sekai.finance import (
    BorrowFee,
    Constraints,
    CostModel,
    FactorRiskModel,
    PortfolioVariance,
    ProportionalCost,
    ScoresLeverage,
    SimpleReturn,
    SpreadCost,
    SquareRootImpact,
)
from sekai.finance.constraints import neutralize
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
    "neutral": dict(  # every step solves for the neutral book 21 times
        data=DATA,
        lookback=6,
        episode_length=20,
        allow_short=True,
        action=ScoresLeverage(
            Constraints(
                max_gross=2.0, max_position=0.3, dollar_neutral=True, neutral_to=("loadings",)
            )
        ),
        transaction_cost=SpreadCost(),
        max_drawdown=0.08,
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
    ("read", `torch.linalg.solve`'s check of its matrices included), ops whose output size
    depends on the data ("shape") and host data made into tensors ("upload")."""

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
        if name in ("_local_scalar_dense", "_linalg_check_errors"):
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
    check_one_read_per_step(env, actions(env, 30), monkeypatch)


def check_one_read_per_step(
    env: BatchMarketEnv, steps: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    waits = Waits(monkeypatch)
    resets = 0
    for a in steps:
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


@pytest.mark.parametrize("name", ["costs", "dynamic", "margin", "neutral"])
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


class HostSpread(CostModel):
    """A cost model of your own that prices on the host, with sekai's `SpreadCost`."""

    requires = frozenset({"spread"})

    def __call__(self, trades: Any, *, spread: Any = None, **context: Any) -> Any:
        cost = SpreadCost()(trades.detach().numpy(), spread=spread.detach().numpy())
        return torch.as_tensor(cost, dtype=trades.dtype)


def test_a_model_called_on_host_arrays_inside_a_step_is_checked_on_the_host() -> None:
    results = []
    for device_path in (False, True):
        env = make("costs", device_path=device_path, transaction_cost=HostSpread())
        results.append(rollout(env, actions(env, 12)))
    for x, y in zip(*results, strict=True):
        same(x, y)
    no_spreads = DATA.with_fields(spread=np.full(np.shape(DATA.field("spread")), np.nan))
    for device_path in (False, True):
        env = make(
            "costs",
            device_path=device_path,
            data=no_spreads,
            execution="close",
            transaction_cost=HostSpread(),
        )
        env.reset(seed=0)
        before = lanes_state(env)
        with pytest.raises(ValueError, match="SpreadCost has no valid 'spread'"):
            env.step(actions(env, 1)[0])
        same(lanes_state(env), before)


@pytest.mark.parametrize("autoreset", [True, False])
def test_the_lanes_a_subclass_ends_end_on_the_device_path_too(autoreset: bool) -> None:
    """`step()` reuses the finished lanes read back with the checks only for the flags
    they were read from, not for those a subclass's `_step` makes of them."""

    class EndsAtFive(BatchMarketEnv):
        def _step(self, actions: Any) -> Any:
            rewards, terminated, truncated, info = super()._step(actions)
            fifth = self.backend.asbool(self._t >= 5, self._like)
            return rewards, terminated, truncated | fifth, info

    envs = []
    for device_path in (False, True):
        env = EndsAtFive(
            N_ENVS, backend="torch", dtype=torch.float32, autoreset=autoreset, **CONFIGS["costs"]
        )
        env._defer = device_path
        envs.append(env)
    steps = actions(envs[0], 20)
    host, device = (rollout(env, steps) for env in envs)
    for x, y in zip(host, device, strict=True):
        same(x, y)
    assert sum(bool(r["truncated"].any()) for r in host if "truncated" in r) >= 2


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_a_partial_reset_that_raises_leaves_the_lanes_rows_consistent(backend: str) -> None:
    """A start that fails to resolve leaves those resolved before it set (as in 0.3.0),
    in every copy of the starts and in what is derived from them."""
    env = BatchMarketEnv(3, data=DATA, backend=backend, lookback=5, episode_length=20)
    env._defer = backend == "torch"
    B = env.backend
    env.reset(seed=0)
    env.step(B.zeros((3, *env.action_space.shape), like=env._like))
    rows = B.to_numpy(env._rows())
    with pytest.raises(ValueError, match="outside"):
        env.reset(indices=[0, 1], options=[{"start": 100}, {"start": 10**6}])
    assert np.array_equal(env._start, B.to_numpy(env._start_dev))
    moved = B.to_numpy(env._rows())
    env._memo.clear()
    assert np.array_equal(moved, B.to_numpy(env._rows()))
    assert moved[0] != rows[0] and np.array_equal(moved[1:], rows[1:])


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
        "_memo",
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
    raised = checks[0][1]()
    assert bool(checks[0][0]) and type(raised) is ValueError and str(raised) == str(error.value)


def lanes_state(env: BatchMarketEnv) -> dict[str, Any]:
    """Copies of the env's arrays: what a step that raises must leave as it was."""
    return {
        k: v.clone() if isinstance(v, torch.Tensor) else v.copy()
        for k, v in vars(env).items()
        if isinstance(v, (torch.Tensor, np.ndarray))
    }


def test_a_singular_projection_raises_solves_error_before_the_step_changes_anything() -> None:
    # Two copies of one large exposure: the ridge is lost to rounding, so the
    # neutralisation's least-squares system is singular.
    twins = DATA.with_fields(twins=np.full((*DATA.prices.shape, 2), 2.0**40))
    rules = Constraints(max_gross=2.0, neutral_to=("twins",))
    errors = []
    for device_path in (False, True):
        env = make("neutral", device_path=device_path, data=twins, action=ScoresLeverage(rules))
        env.reset(seed=0)
        before = lanes_state(env)
        with pytest.raises(torch.linalg.LinAlgError, match="singular") as error:
            env.step(actions(env, 1)[0])
        same(lanes_state(env), before)
        errors.append(str(error.value))
    assert errors[0] == errors[1]


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_collected_solve_checks_give_the_same_book_and_error(backend: str) -> None:
    B = get_backend(backend)
    like = torch.zeros((), dtype=torch.float64) if backend == "torch" else None
    rng = np.random.default_rng(0)
    x = B.asfloat(rng.normal(size=(4, 6)), like)
    columns = B.asfloat(rng.normal(size=(4, 6, 2)), like)
    eager = neutralize(x, columns)
    with _collect_checks() as checks:
        collected = neutralize(x, columns)
    same(collected, eager)
    assert len(checks) == 1 and not bool(checks[0][0])
    twins = B.asfloat(np.full((4, 6, 2), 2.0**40), like)
    with pytest.raises(Exception, match=r"(?i)singular") as error:
        neutralize(x, twins)
    if backend == "numpy":  # on the host, reading the check costs nothing: it raises
        with _collect_checks(), pytest.raises(type(error.value)):
            neutralize(x, twins)
        return
    with _collect_checks() as checks:
        neutralize(x, twins)
    raised = checks[0][1]()
    assert bool(checks[0][0]) and type(raised) is type(error.value)
    assert str(raised) == str(error.value)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64"])
def test_scalars_become_tensors_made_on_the_device(
    dtype: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Host scalars become 0-d tensors made on the device, in the env's dtype (MPS has no
    float64), bit for bit as copying them over makes them, rounding included. Only a
    finite value beyond the dtype's range is converted on the host (overflowing to inf)."""
    B, like = get_backend("torch"), torch.zeros((), dtype=getattr(torch, dtype))
    floats = [0.0, -0.0, 0.1, 1 / 3, 2 / 252, 1e-320, 6e4, 1e39, -1e39, float("inf"), float("nan")]
    values = [*floats, 5, True, np.float32(0.1), np.float64(1e5), np.array(0.25), np.int64(3)]
    as_float = [float(np.asarray(v, dtype=np.float64)) for v in values]
    expected = [torch.as_tensor(np.float64(v), dtype=like.dtype) for v in as_float]
    top = torch.finfo(like.dtype).max
    beyond = sum(math.isfinite(v) and abs(v) > top for v in as_float)
    filled: list[Any] = []
    full = torch.full
    monkeypatch.setattr(torch, "full", lambda *a, **k: filled.append(k["dtype"]) or full(*a, **k))
    waits = Waits(monkeypatch)
    with waits:
        got = [B.asfloat(v, like) for v in values]
        flags = [B.asbool(v, like) for v in (0, 1, True, np.bool_(False), np.array(2.0))]
        ints = [B.asint(v, like) for v in (0, -7, np.int64(2**40), np.array(9))]
    assert dict(waits.count) == ({"upload": beyond} if beyond else {})
    assert like.dtype == torch.float64 or torch.float64 not in filled
    bits = {2: torch.int16, 4: torch.int32, 8: torch.int64}[like.element_size()]
    for g, e in zip(got, expected, strict=True):
        assert g.dtype == like.dtype and g.shape == ()
        assert bool(torch.isnan(g)) == bool(torch.isnan(e))
        assert bool(torch.isnan(g)) or torch.equal(g.view(bits), e.view(bits))
    assert [bool(f) for f in flags] == [False, True, True, False, True]
    assert all(f.dtype == torch.bool for f in flags)
    assert [int(i) for i in ints] == [0, -7, 2**40, 9] and all(i.dtype == torch.int64 for i in ints)
    assert B.on_host(like) and get_backend("numpy").on_host(np.zeros(3))


# On a real accelerator (skipped without one): the device path, chosen by the env itself.

ACCELERATORS = [
    device
    for device, available in (
        ("cuda", torch.cuda.is_available()),
        ("mps", torch.backends.mps.is_available()),
    )
    if available
]
on_accelerators = pytest.mark.parametrize(
    "device", ACCELERATORS or [pytest.param("none", marks=pytest.mark.skip("no CUDA or MPS"))]
)


def on(device: str, name: str, **kwargs: Any) -> BatchMarketEnv:
    """The env `name` on `device`, in float64 where the device has it (MPS has not)."""
    if name == "neutral":
        try:  # (a device without torch.linalg.solve can't neutralise a book, in 0.3.0 either)
            torch.linalg.solve_ex(torch.eye(2, device=device), torch.ones(2, 1, device=device))
        except (NotImplementedError, RuntimeError) as e:
            pytest.skip(f"torch.linalg.solve_ex on {device}: {e}")
    dtype = torch.float32 if device == "mps" else torch.float64
    kwargs = {**CONFIGS[name], **kwargs}
    env = BatchMarketEnv(N_ENVS, backend="torch", device=device, dtype=dtype, **kwargs)
    assert env._deferring()
    return env


def moved(device: str, steps: list[Any]) -> list[Any]:
    return [a.to(device) for a in steps]


def close(a: Any, b: Any, **tolerance: float) -> None:
    """`b` (on a device) is `a` (on the CPU) to rounding."""
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            close(a[k], b[k], **tolerance)
    elif a is None or b is None:
        assert a is b
    elif isinstance(a, torch.Tensor):
        b = b.cpu()
        if a.is_floating_point():
            torch.testing.assert_close(b, a, equal_nan=True, **tolerance)
        else:
            assert torch.equal(a, b)
    else:
        np.testing.assert_array_equal(a, b)


@on_accelerators
@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_on_an_accelerator_a_step_reads_back_once(
    device: str, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = on(device, name)
    env.reset(seed=0)
    check_one_read_per_step(env, moved(device, actions(env, 30)), monkeypatch)
    if device != "cuda":
        return
    # CUDA's own count of the operations that synchronise with the host: one per step.
    env.reset(seed=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            for a in moved(device, actions(env, 12, seed=1)):
                caught.clear()
                result = env.step(a)
                syncs = [w for w in caught if "synchronizing CUDA operation" in str(w.message)]
                if result.reset_indices is None:
                    assert len(syncs) == 1
        finally:
            torch.cuda.set_sync_debug_mode(0)


@on_accelerators
@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_on_an_accelerator_results_are_the_cpus_to_rounding(device: str, name: str) -> None:
    env = on(device, name)
    cpu = BatchMarketEnv(N_ENVS, backend="torch", dtype=env._like.dtype, **CONFIGS[name])
    float32 = env._like.dtype == torch.float32
    tolerance = dict(rtol=1e-3, atol=1e-5) if float32 else dict(rtol=1e-7, atol=1e-10)
    steps = actions(cpu, 25)
    for x, y in zip(rollout(cpu, steps), rollout(env, moved(device, steps)), strict=True):
        close(x, y, **tolerance)


@on_accelerators
def test_on_an_accelerator_errors_come_before_the_step_changes_anything(device: str) -> None:
    env = on(device, "costs")
    env.reset(seed=0)
    first, second = moved(device, actions(env, 2))
    env.step(first)
    twin = copy.deepcopy(env)
    bad = second.clone()
    bad[3, 2] = float("nan")
    with pytest.raises(InvalidAction):
        env.step(bad)
    same(vars(env.step(second)), vars(twin.step(second)))
    no_spreads = DATA.with_fields(spread=np.full(np.shape(DATA.field("spread")), np.nan))
    env = on(device, "costs", data=no_spreads, execution="close")
    env.reset(seed=0)
    before = lanes_state(env)
    with pytest.raises(ValueError, match="SpreadCost has no valid 'spread'"):
        env.step(moved(device, actions(env, 1))[0])
    same(lanes_state(env), before)


@on_accelerators
def test_on_an_accelerator_a_singular_projection_raises_solves_error(device: str) -> None:
    twins = DATA.with_fields(twins=np.full((*DATA.prices.shape, 2), 2.0**40))
    rules = Constraints(max_gross=2.0, neutral_to=("twins",))
    env = on(device, "neutral", data=twins, action=ScoresLeverage(rules))
    env.reset(seed=0)
    before = lanes_state(env)
    with pytest.raises(torch.linalg.LinAlgError, match="singular"):
        env.step(moved(device, actions(env, 1))[0])
    same(lanes_state(env), before)


@on_accelerators
def test_on_an_accelerator_rewards_are_differentiable(device: str) -> None:
    env = on(device, "costs")
    env.differentiable = True
    env.reset(seed=0)
    shape, dtype = env.action_space.shape, env._like.dtype
    theta = torch.zeros(shape, dtype=dtype, device=device, requires_grad=True)
    total = sum(env.step(theta + a).rewards.sum() for a in moved(device, actions(env, 8)))
    (grad,) = torch.autograd.grad(total, theta)
    assert grad.device.type == device
    assert bool(torch.isfinite(grad).all()) and bool(grad.abs().sum() > 0)
