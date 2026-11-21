"""sekai.backend: the NumPy and torch backends agree, and torch stays optional."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from sekai.backend import (
    Backend,
    NumpyBackend,
    backend_and_like,
    backend_of,
    get_backend,
    is_tensor,
    to_numpy,
)

torch = pytest.importorskip("torch")

NP = get_backend("numpy")
TH = get_backend("torch")


def t64(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float64)


def test_dispatch_and_lookup() -> None:
    assert isinstance(NP, NumpyBackend)
    assert get_backend("numpy") is NP and get_backend(NP) is NP
    assert get_backend("torch") is TH and TH.name == "torch"
    assert backend_of(np.ones(2), 3.0) is NP
    assert backend_of(np.ones(2), torch.ones(2)) is TH
    B, like = backend_and_like(None, np.ones(1), torch.ones(1, dtype=torch.float64))
    assert B is TH and like.dtype == torch.float64
    assert backend_and_like(np.ones(1)) == (NP, None)
    assert is_tensor(torch.ones(1)) and not is_tensor(np.ones(1))
    with pytest.raises(ValueError, match="Unknown backend"):
        get_backend("jax")
    assert isinstance(to_numpy(torch.ones(2, requires_grad=True)), np.ndarray)


def test_importing_sekai_does_not_import_torch() -> None:
    code = "import sys, sekai; sekai.make('Market-v1').reset(seed=0); print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_dtype_and_device_rules() -> None:
    f32 = torch.zeros(3)
    assert TH.asfloat(np.ones(2), like=f32).dtype == torch.float32
    assert TH.asfloat(torch.ones(2, dtype=torch.float64), like=f32).dtype == torch.float64
    assert TH.asfloat(torch.ones(2, dtype=torch.int64)).dtype == torch.get_default_dtype()
    assert TH.cast(torch.ones(2, dtype=torch.float64), f32).dtype == torch.float32
    assert TH.zeros((2,), like=torch.zeros(1, dtype=torch.float64)).dtype == torch.float64
    assert TH.zeros((2,), like=torch.zeros(1, dtype=torch.bool)).dtype == torch.get_default_dtype()
    assert NP.asfloat([1, 2]).dtype == np.float64
    assert NP.tiny() == np.finfo(np.float64).tiny
    assert TH.tiny(f32) == torch.finfo(torch.float32).tiny


@pytest.mark.parametrize("n_valid", [0, 1, 2, 3, 4, 7])
def test_nanmedian_matches_numpy(n_valid: int) -> None:
    rng = np.random.default_rng(n_valid)
    x = rng.normal(size=(5, 7))
    x[:, n_valid:] = np.nan
    rng.shuffle(x, axis=1)
    want = NP.nanmedian(x, axis=-1)
    got = TH.nanmedian(t64(x), axis=-1).numpy()
    np.testing.assert_array_equal(np.isnan(want), np.isnan(got))
    np.testing.assert_allclose(got[~np.isnan(got)], want[~np.isnan(want)], rtol=0, atol=1e-15)


def _same(np_out: np.ndarray, th_out: torch.Tensor) -> None:
    np.testing.assert_allclose(TH.to_numpy(th_out), np.asarray(np_out), rtol=1e-13, atol=1e-15)


def test_elementwise_and_reductions_agree() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 6))
    y = rng.normal(size=(4, 6))
    tx, ty = t64(x), t64(y)
    pairs: list[tuple[str, tuple[object, ...]]] = [
        ("abs", ()),
        ("sign", ()),
        ("exp", ()),
        ("tanh", ()),
        ("sigmoid", ()),
    ]
    for name, extra in pairs:
        _same(getattr(NP, name)(x, *extra), getattr(TH, name)(tx, *extra))
    _same(NP.sqrt(np.abs(x)), TH.sqrt(torch.abs(tx)))
    _same(NP.log(np.abs(x)), TH.log(torch.abs(tx)))
    _same(NP.log1p(np.abs(x)), TH.log1p(torch.abs(tx)))
    _same(NP.clip(x, -0.5, None), TH.clip(tx, -0.5, None))
    _same(NP.clip(x, y, None), TH.clip(tx, ty, None))
    _same(NP.maximum(x, 0.1), TH.maximum(tx, 0.1))
    _same(NP.maximum(0.1, x), TH.maximum(0.1, tx))
    _same(NP.minimum(x, y), TH.minimum(tx, ty))
    _same(NP.where(x > 0, x, 0.0), TH.where(tx > 0, tx, 0.0))
    _same(NP.where(x > 0, 1.0, 2.0), TH.where(tx > 0, 1.0, 2.0))
    for axis in (None, 0, -1):
        _same(NP.sum(x, axis=axis), TH.sum(tx, axis=axis))
        _same(NP.mean(x, axis=axis), TH.mean(tx, axis=axis))
        _same(NP.max(x, axis=axis), TH.max(tx, axis=axis))
    _same(NP.sum(x, axis=-1, keepdims=True), TH.sum(tx, axis=-1, keepdims=True))
    _same(NP.cumsum(x, axis=1), TH.cumsum(tx, axis=1))
    assert bool(NP.any(x > 2)) == bool(TH.any(tx > 2))
    np.testing.assert_array_equal(NP.all(x > -9, axis=0), TH.all(tx > -9, axis=0).numpy())
    z = x.copy()
    z[0, 0], z[1, 1], z[2, 2] = np.nan, np.inf, -np.inf
    _same(NP.nan_to_num(z), TH.nan_to_num(t64(z)))
    np.testing.assert_array_equal(NP.isfinite(z), TH.isfinite(t64(z)).numpy())


def test_structure_and_linear_algebra_agree() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 5))
    tx = t64(x)
    idx = rng.integers(0, 5, size=(3, 5))
    _same(NP.take_along_axis(x, idx, axis=1), TH.take_along_axis(tx, torch.as_tensor(idx), axis=1))
    perm = np.argsort(rng.random((3, 5)), axis=1)
    vals = rng.normal(size=(3, 5))
    _same(
        NP.put_along_axis(x, perm, vals, axis=1),
        TH.put_along_axis(tx, torch.as_tensor(perm), t64(vals), axis=1),
    )
    ties = np.array([[1, 0, 1, 0, 1], [0, 0, 0, 1, 1]])
    np.testing.assert_array_equal(
        NP.argsort(ties, axis=1), TH.argsort(torch.as_tensor(ties), axis=1).numpy()
    )
    _same(NP.stack([x, x], axis=1), TH.stack([tx, tx], axis=1))
    _same(NP.concat([x, x], axis=0), TH.concat([tx, tx], axis=0))
    _same(NP.moveaxis(x, 0, 1), TH.moveaxis(tx, 0, 1))
    _same(NP.broadcast_to(x[:1], (3, 5)), TH.broadcast_to(tx[:1], (3, 5)))
    a = rng.normal(size=(2, 4, 4)) + 4 * np.eye(4)
    b = rng.normal(size=(2, 4, 3))
    _same(NP.solve(a, b), TH.solve(t64(a), t64(b)))
    _same(NP.einsum("bij,bjk->bik", a, b), TH.einsum("bij,bjk->bik", t64(a), t64(b)))
    assert NP.item(np.array([3.5])) == TH.item(torch.tensor([3.5])) == 3.5


def test_copy_put_along_axis_does_not_alias() -> None:
    x = torch.zeros(2, 3)
    y = TH.put_along_axis(x, torch.tensor([[0], [2]]), torch.ones(2, 1), axis=1)
    assert float(x.sum()) == 0.0 and float(y.sum()) == 2.0
    a = np.zeros((2, 3))
    NP.put_along_axis(a, np.array([[0], [2]]), np.ones((2, 1)), axis=1)
    assert a.sum() == 0.0


def test_backend_is_abstract() -> None:
    with pytest.raises(TypeError):
        Backend()  # type: ignore[abstract]
