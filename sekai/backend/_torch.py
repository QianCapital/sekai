"""The torch backend: tensors on any device, in their own floating dtype, with autograd."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import torch
from numpy.typing import NDArray

from sekai.backend._base import Array, Backend


def _device(like: Any) -> Any:
    return like.device if isinstance(like, torch.Tensor) else None


def _float_dtype(like: Any) -> torch.dtype:
    if isinstance(like, torch.Tensor) and like.is_floating_point():
        return like.dtype
    return torch.get_default_dtype()


def _scalar(x: Any) -> bool:
    return isinstance(x, (bool, int, float, np.number, np.bool_))


def _full_float(value: float, dtype: torch.dtype, device: Any) -> torch.Tensor:
    """A 0-d float tensor of `value`, made on the device: copying a host scalar there
    would wait for the device. Filled in float64 and cast, it rounds (and overflows to
    inf) exactly as converting the host float64 does."""
    out = torch.full((), value, dtype=torch.float64, device=device)
    return out if dtype == torch.float64 else out.to(dtype)


class TorchBackend(Backend):
    """torch tensors. New tensors take the dtype and device of `like`."""

    name: ClassVar[str] = "torch"

    def is_array(self, x: Any) -> bool:
        return isinstance(x, torch.Tensor)

    def _tensor(self, x: Any, like: Any) -> torch.Tensor:
        """`x` as a tensor on `like`'s device; floats (from NumPy too) in `like`'s float dtype."""
        if isinstance(x, torch.Tensor):
            return x
        arr = np.asarray(x)
        if arr.dtype.kind == "f":
            if arr.ndim == 0:
                return _full_float(float(arr), _float_dtype(like), _device(like))
            return torch.as_tensor(arr, dtype=_float_dtype(like), device=_device(like))
        return torch.as_tensor(arr, device=_device(like))

    def asfloat(self, x: Any, like: Array | None = None) -> Array:
        if isinstance(x, torch.Tensor):
            out = x if x.is_floating_point() else x.to(_float_dtype(like))
            device = _device(like)
            return out if device is None or out.device == device else out.to(device)
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 0:
            return _full_float(float(arr), _float_dtype(like), _device(like))
        return torch.as_tensor(arr, dtype=_float_dtype(like), device=_device(like))

    def asbool(self, x: Any, like: Array | None = None) -> Array:
        if isinstance(x, torch.Tensor):
            return x.to(torch.bool)
        arr = np.asarray(x, dtype=bool)
        if arr.ndim == 0:
            return torch.full((), bool(arr), dtype=torch.bool, device=_device(like))
        return torch.as_tensor(arr, device=_device(like))

    def asint(self, x: Any, like: Array | None = None) -> Array:
        if isinstance(x, torch.Tensor):
            return x.to(torch.int64)
        arr = np.asarray(x, dtype=np.int64)
        if arr.ndim == 0:
            return torch.full((), int(arr), dtype=torch.int64, device=_device(like))
        return torch.as_tensor(arr, device=_device(like))

    def cast(self, x: Any, like: Array) -> Array:
        return self._tensor(x, like).to(dtype=like.dtype, device=like.device)

    def asint8(self, x: Any) -> Array:
        return self._tensor(x, None).to(torch.int8)

    def to_numpy(self, x: Any) -> NDArray[Any]:
        if isinstance(x, torch.Tensor):
            return np.asarray(x.detach().cpu().numpy())
        return np.asarray(x)

    def zeros(self, shape: Sequence[int], like: Array | None = None) -> Array:
        return torch.zeros(tuple(shape), dtype=_float_dtype(like), device=_device(like))

    def full(self, shape: Sequence[int], value: float, like: Array | None = None) -> Array:
        return torch.full(
            tuple(shape), float(value), dtype=_float_dtype(like), device=_device(like)
        )

    def bools(self, shape: Sequence[int], value: bool, like: Array | None = None) -> Array:
        return torch.full(tuple(shape), bool(value), dtype=torch.bool, device=_device(like))

    def ints(self, shape: Sequence[int], value: int, like: Array | None = None) -> Array:
        return torch.full(tuple(shape), int(value), dtype=torch.int64, device=_device(like))

    def arange(self, n: int, like: Array | None = None) -> Array:
        return torch.arange(n, dtype=torch.int64, device=_device(like))

    def copy(self, x: Array) -> Array:
        return x.clone()

    def detach(self, x: Array) -> Array:
        return x.detach()

    def item(self, x: Array) -> Any:
        return x.item() if isinstance(x, torch.Tensor) else np.asarray(x).item()

    def on_host(self, x: Array) -> bool:
        return not isinstance(x, torch.Tensor) or x.device.type == "cpu"

    def where(self, condition: Array, x: Any, y: Any) -> Array:
        like = next((v for v in (x, y, condition) if isinstance(v, torch.Tensor)), None)
        cond = self._tensor(condition, like).to(torch.bool)
        a = x if isinstance(x, torch.Tensor) or _scalar(x) else self._tensor(x, like)
        b = y if isinstance(y, torch.Tensor) or _scalar(y) else self._tensor(y, like)
        if _scalar(a) and _scalar(b):
            a = torch.full((), float(a), dtype=_float_dtype(like), device=cond.device)
        return torch.where(cond, a, b)

    def abs(self, x: Array) -> Array:
        return torch.abs(x)

    def sign(self, x: Array) -> Array:
        return torch.sign(x)

    def sqrt(self, x: Array) -> Array:
        return torch.sqrt(x)

    def exp(self, x: Array) -> Array:
        return torch.exp(x)

    def log(self, x: Array) -> Array:
        return torch.log(x)

    def log1p(self, x: Array) -> Array:
        return torch.log1p(x)

    def tanh(self, x: Array) -> Array:
        return torch.tanh(x)

    def sigmoid(self, x: Array) -> Array:
        return torch.sigmoid(x)

    def clip(self, x: Array, lo: Any, hi: Any) -> Array:
        lo = lo if lo is None or _scalar(lo) else self._tensor(lo, x)
        hi = hi if hi is None or _scalar(hi) else self._tensor(hi, x)
        return torch.clamp(x, min=lo, max=hi)

    def maximum(self, x: Any, y: Any) -> Array:
        if _scalar(y):
            return torch.clamp(self._tensor(x, None), min=float(y))
        if _scalar(x):
            return torch.clamp(self._tensor(y, None), min=float(x))
        like = x if isinstance(x, torch.Tensor) else y
        return torch.maximum(self._tensor(x, like), self._tensor(y, like))

    def minimum(self, x: Any, y: Any) -> Array:
        if _scalar(y):
            return torch.clamp(self._tensor(x, None), max=float(y))
        if _scalar(x):
            return torch.clamp(self._tensor(y, None), max=float(x))
        like = x if isinstance(x, torch.Tensor) else y
        return torch.minimum(self._tensor(x, like), self._tensor(y, like))

    def isfinite(self, x: Array) -> Array:
        return torch.isfinite(x)

    def isnan(self, x: Array) -> Array:
        return torch.isnan(x)

    def nan_to_num(self, x: Array, nan: float = 0.0) -> Array:
        return torch.nan_to_num(x, nan=nan)

    def sum(
        self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array:
        if axis is None:
            out = torch.sum(x)
            return out.reshape((1,) * x.ndim) if keepdims else out
        return torch.sum(x, dim=axis, keepdim=keepdims)

    def mean(self, x: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        if axis is None:
            out = torch.mean(x)
            return out.reshape((1,) * x.ndim) if keepdims else out
        return torch.mean(x, dim=axis, keepdim=keepdims)

    def max(self, x: Array, axis: int | None = None, keepdims: bool = False) -> Array:
        if axis is None:
            out = torch.amax(x)
            return out.reshape((1,) * x.ndim) if keepdims else out
        return torch.amax(x, dim=axis, keepdim=keepdims)

    def any(self, x: Array, axis: int | None = None) -> Array:
        return torch.any(x) if axis is None else torch.any(x, dim=axis)

    def all(self, x: Array, axis: int | None = None) -> Array:
        return torch.all(x) if axis is None else torch.all(x, dim=axis)

    def cumsum(self, x: Array, axis: int) -> Array:
        return torch.cumsum(x, dim=axis)

    def nanmedian(self, x: Array, axis: int = -1) -> Array:
        # torch.nanmedian returns the lower middle value for an even count; NumPy averages
        # the two middle values. Sort with NaN last and average them, as NumPy does.
        valid = ~torch.isnan(x)
        n = valid.sum(dim=axis, keepdim=True)
        ordered = torch.sort(torch.where(valid, x, torch.inf), dim=axis).values
        last = x.shape[axis] - 1
        lo = torch.clamp(torch.div(n - 1, 2, rounding_mode="floor"), 0, max(last, 0))
        hi = torch.clamp(torch.div(n, 2, rounding_mode="floor"), 0, max(last, 0))
        median = 0.5 * (
            torch.take_along_dim(ordered, lo, dim=axis)
            + torch.take_along_dim(ordered, hi, dim=axis)
        )
        return torch.where(n > 0, median, torch.nan).squeeze(axis)

    def stack(self, arrays: Sequence[Array], axis: int = 0) -> Array:
        return torch.stack(list(arrays), dim=axis)

    def concat(self, arrays: Sequence[Array], axis: int = 0) -> Array:
        return torch.cat(list(arrays), dim=axis)

    def broadcast_to(self, x: Array, shape: Sequence[int]) -> Array:
        return torch.broadcast_to(self._tensor(x, None), tuple(shape))

    def moveaxis(self, x: Array, source: int, destination: int) -> Array:
        return torch.movedim(x, source, destination)

    def take_along_axis(self, x: Array, indices: Array, axis: int) -> Array:
        return torch.take_along_dim(x, self.asint(indices, x), dim=axis)

    def put_along_axis(self, x: Array, indices: Array, values: Array, axis: int) -> Array:
        idx = self.asint(indices, x)
        vals = self._tensor(values, x).to(x.dtype)
        return x.scatter(axis, idx, torch.broadcast_to(vals, idx.shape))

    def argsort(self, x: Array, axis: int = -1) -> Array:
        return torch.argsort(self._tensor(x, None), dim=axis, stable=True)

    def einsum(self, subscripts: str, *operands: Array) -> Array:
        return torch.einsum(subscripts, *operands)

    def solve(self, a: Array, b: Array) -> Array:
        return torch.linalg.solve(a, b)

    def eye(self, n: int, like: Array | None = None) -> Array:
        return torch.eye(n, dtype=_float_dtype(like), device=_device(like))

    def tiny(self, like: Array | None = None) -> float:
        return float(torch.finfo(_float_dtype(like)).tiny)


__all__ = ["TorchBackend"]
