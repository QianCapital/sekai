"""The torch backend: tensors on any device, in their own floating dtype, with autograd."""

from __future__ import annotations

import math
import struct
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

import numpy as np
import torch
import torch.backends.opt_einsum
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
    """A 0-d float tensor of `value`, made on the device in `dtype` (copying a host scalar
    there would wait for the device; some devices have no float64). It rounds as converting
    the host float64 does. A finite value beyond the dtype's range, which `torch.full`
    refuses, is converted on the host, overflowing to inf as before."""
    if math.isfinite(value) and abs(value) > torch.finfo(dtype).max:
        return torch.as_tensor(np.float64(value), dtype=dtype, device=device)
    return torch.full((), value, dtype=dtype, device=device)


# 0-d tensors of the Python scalars passed to `where`, by value, dtype and device: torch
# would make one at every call. (False: a pair that keeps the scalar.)
_CONSTANTS: dict[tuple[Any, ...], torch.Tensor | bool] = {}
# Under a mode (FakeTensorMode, a TorchFunctionMode) a new tensor need not be an ordinary
# one, and under a CUDA graph capture its value is not computed yet: none is cached then.
_dispatch_modes: Callable[[], int] = getattr(torch._C, "_len_torch_dispatch_stack", lambda: 0)
_function_modes: Callable[[], bool] = getattr(
    torch._C, "_is_torch_function_mode_enabled", lambda: False
)


def _constant(value: Any, other: torch.Tensor) -> Any:
    """`value`, a Python scalar beside the tensor `other` in `torch.where`, as a 0-d tensor
    made once that gives the same result: in `other`'s dtype, for a float or int beside a
    floating tensor (within its range), an int beside an int64 tensor and a bool beside a
    bool tensor. Any other pair keeps the scalar, for torch to promote as it does, as do a
    tensor subclass, an active mode and a CUDA graph capture."""
    kind = type(value)
    if kind is not float and kind is not int and kind is not bool:
        return value
    if (
        type(other) is not torch.Tensor
        or _dispatch_modes()
        or _function_modes()
        or (other.is_cuda and torch.cuda.is_current_stream_capturing())
    ):
        return value
    dtype, device = other.dtype, other.device
    key = (kind, struct.pack("<d", value) if kind is float else value, dtype, device)
    out = _CONSTANTS.get(key)
    if out is not None:
        return out if isinstance(out, torch.Tensor) else value
    if kind is bool:
        ok = dtype == torch.bool
    elif kind is int:
        ok = dtype == torch.int64 or (dtype.is_floating_point and abs(value) <= 2**53)
    else:
        ok = dtype.is_floating_point and not (
            math.isfinite(value) and abs(value) > torch.finfo(dtype).max
        )
    if len(_CONSTANTS) > 4096:  # values computed at run time: keep the cache small
        _CONSTANTS.clear()
    out = _CONSTANTS[key] = torch.full((), value, dtype=dtype, device=device) if ok else False
    return out if ok else value


# torch.einsum's contraction orders, by equation, strategy and operand shapes.
_PATHS: dict[tuple[Any, ...], list[int]] = {}


def _einsum(subscripts: str, operands: tuple[Any, ...]) -> torch.Tensor:
    """`torch.einsum(subscripts, *operands)`, which asks opt_einsum (when it is installed)
    for the order of a contraction of three or more operands at every call; that takes
    longer than sekai's small contractions themselves. The order is asked for once per
    shape here and the contraction is run as torch.einsum runs it, so the result is the
    same, bit for bit."""
    oe = torch.backends.opt_einsum
    if (
        len(operands) <= 2
        or not (oe.enabled and oe.is_available())
        or any(type(o) is not torch.Tensor for o in operands)
        or torch.overrides.has_torch_function(operands)
    ):
        return torch.einsum(subscripts, *operands)
    key = (subscripts, oe.strategy, *(tuple(o.shape) for o in operands))
    path = _PATHS.get(key)
    if path is None:
        tupled = oe.get_opt_einsum().contract_path(subscripts, *operands, optimize=oe.strategy)
        if len(_PATHS) > 1024:
            _PATHS.clear()
        path = _PATHS[key] = [i for pair in tupled[0] for i in pair]
    return torch._VF.einsum(subscripts, operands, path=path)


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
        # The common calls (a bool mask, then tensors or a Python scalar) first.
        if type(condition) is torch.Tensor and condition.dtype is torch.bool:
            kx, ky = type(x), type(y)
            if kx is torch.Tensor:
                if ky is torch.Tensor:
                    return torch.where(condition, x, y)
                if ky is float or ky is int or ky is bool:
                    return torch.where(condition, x, _constant(y, x))
            elif ky is torch.Tensor and (kx is float or kx is int or kx is bool):
                return torch.where(condition, _constant(x, y), y)
        x_tensor, y_tensor = isinstance(x, torch.Tensor), isinstance(y, torch.Tensor)
        if x_tensor and y_tensor:
            like = x
        else:
            like = next((v for v in (x, y, condition) if isinstance(v, torch.Tensor)), None)
        if isinstance(condition, torch.Tensor) and condition.dtype == torch.bool:
            cond = condition
        else:
            cond = self._tensor(condition, like).to(torch.bool)
        a = x if x_tensor or _scalar(x) else self._tensor(x, like)
        b = y if y_tensor or _scalar(y) else self._tensor(y, like)
        if _scalar(a) and _scalar(b):
            a = torch.full((), float(a), dtype=_float_dtype(like), device=cond.device)
        elif isinstance(a, torch.Tensor) and not isinstance(b, torch.Tensor):
            b = _constant(b, a)
        elif isinstance(b, torch.Tensor) and not isinstance(a, torch.Tensor):
            a = _constant(a, b)
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
        return _einsum(subscripts, operands)

    def solve(self, a: Array, b: Array) -> Array:
        return torch.linalg.solve(a, b)

    def _solve_unchecked(self, a: Array, b: Array) -> tuple[Array, Array]:
        # `solve` is `solve_ex` (the same solution and gradient), then a read of `info`.
        x, info = torch.linalg.solve_ex(a, b)
        return x, torch.any(info != 0)

    def eye(self, n: int, like: Array | None = None) -> Array:
        return torch.eye(n, dtype=_float_dtype(like), device=_device(like))

    def tiny(self, like: Array | None = None) -> float:
        return float(torch.finfo(_float_dtype(like)).tiny)


__all__ = ["TorchBackend"]
