"""The `Backend` interface: the array operations sekai's batched finance core is written in."""

from __future__ import annotations

import abc
import contextlib
from collections.abc import Sequence
from typing import Any, ClassVar

from numpy.typing import NDArray

#: An array of the backend at hand: a `numpy.ndarray` or a `torch.Tensor`.
Array = Any


class Backend(abc.ABC):
    """Array operations with NumPy semantics, implemented for one array library.

    Functions written against a `Backend` (`B.where(...)`, `B.sum(x, axis=-1)`) run
    unchanged on NumPy arrays and on torch tensors. Floating-point arrays keep their
    precision: NumPy code computes in float64 (as the rest of sekai does), torch code
    in the dtype of the tensors it is given (usually float32), on their device.

    `like` arguments name an existing array whose dtype and device a new array takes;
    with NumPy they only matter for the dtype.
    """

    #: "numpy" or "torch".
    name: ClassVar[str]

    # ------------------------------------------------------------------
    # Arrays
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def is_array(self, x: Any) -> bool:
        """Whether `x` is an array of this backend."""

    @abc.abstractmethod
    def asfloat(self, x: Any, like: Array | None = None) -> Array:
        """`x` as a floating-point array: float64 with NumPy; with torch, the floating
        dtype of `x` if it is a float tensor, else that of `like`, else the default dtype."""

    @abc.abstractmethod
    def asbool(self, x: Any, like: Array | None = None) -> Array:
        """`x` as a boolean array (on `like`'s device)."""

    @abc.abstractmethod
    def asint(self, x: Any, like: Array | None = None) -> Array:
        """`x` as an int64 index array (on `like`'s device)."""

    @abc.abstractmethod
    def cast(self, x: Any, like: Array) -> Array:
        """`x` as an array of `like`'s dtype (and device)."""

    @abc.abstractmethod
    def asint8(self, x: Any) -> Array:
        """`x` as int8 (e.g. a 0/1 mask in an observation)."""

    @abc.abstractmethod
    def to_numpy(self, x: Any) -> NDArray[Any]:
        """A NumPy array with the values of `x` (detached and on the CPU for torch)."""

    @abc.abstractmethod
    def zeros(self, shape: Sequence[int], like: Array | None = None) -> Array:
        """A float array of zeros."""

    @abc.abstractmethod
    def full(self, shape: Sequence[int], value: float, like: Array | None = None) -> Array:
        """A float array filled with `value`."""

    @abc.abstractmethod
    def bools(self, shape: Sequence[int], value: bool, like: Array | None = None) -> Array:
        """A boolean array filled with `value`."""

    @abc.abstractmethod
    def ints(self, shape: Sequence[int], value: int, like: Array | None = None) -> Array:
        """An int64 array filled with `value`."""

    @abc.abstractmethod
    def arange(self, n: int, like: Array | None = None) -> Array:
        """`0, 1, ..., n - 1` as int64."""

    def zeros_like(self, x: Array) -> Array:
        """A float array of zeros shaped like `x`."""
        return self.zeros(tuple(x.shape), like=x)

    @abc.abstractmethod
    def copy(self, x: Array) -> Array:
        """A copy of `x` (with torch, still part of the autograd graph)."""

    @abc.abstractmethod
    def detach(self, x: Array) -> Array:
        """`x` cut from the autograd graph (NumPy: `x` itself)."""

    @abc.abstractmethod
    def item(self, x: Array) -> Any:
        """The Python scalar held by a one-element array."""

    def on_host(self, x: Array) -> bool:
        """Whether `x` is in host memory, so reading its values (`item`, `to_numpy`, a
        Python `if` on it) costs no wait for a device. False on a GPU, where each such
        read blocks until the device finishes its queued work."""
        return True

    # ------------------------------------------------------------------
    # Elementwise
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def where(self, condition: Array, x: Any, y: Any) -> Array:
        """`x` where `condition`, else `y`. With torch, give at least one branch as an
        array: two Python scalars make a tensor of torch's default dtype."""

    @abc.abstractmethod
    def abs(self, x: Array) -> Array: ...

    @abc.abstractmethod
    def sign(self, x: Array) -> Array: ...

    @abc.abstractmethod
    def sqrt(self, x: Array) -> Array: ...

    @abc.abstractmethod
    def exp(self, x: Array) -> Array: ...

    @abc.abstractmethod
    def log(self, x: Array) -> Array: ...

    @abc.abstractmethod
    def log1p(self, x: Array) -> Array: ...

    @abc.abstractmethod
    def tanh(self, x: Array) -> Array: ...

    @abc.abstractmethod
    def sigmoid(self, x: Array) -> Array: ...

    @abc.abstractmethod
    def clip(self, x: Array, lo: Any, hi: Any) -> Array:
        """`x` clipped to `[lo, hi]`; either bound may be None."""

    @abc.abstractmethod
    def maximum(self, x: Any, y: Any) -> Array: ...

    @abc.abstractmethod
    def minimum(self, x: Any, y: Any) -> Array: ...

    @abc.abstractmethod
    def isfinite(self, x: Array) -> Array: ...

    @abc.abstractmethod
    def isnan(self, x: Array) -> Array: ...

    @abc.abstractmethod
    def nan_to_num(self, x: Array, nan: float = 0.0) -> Array:
        """NaN replaced by `nan` and infinities by the largest finite values (NumPy's rule)."""

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def sum(
        self, x: Array, axis: int | tuple[int, ...] | None = None, keepdims: bool = False
    ) -> Array: ...

    @abc.abstractmethod
    def mean(self, x: Array, axis: int | None = None, keepdims: bool = False) -> Array: ...

    @abc.abstractmethod
    def max(self, x: Array, axis: int | None = None, keepdims: bool = False) -> Array: ...

    @abc.abstractmethod
    def any(self, x: Array, axis: int | None = None) -> Array: ...

    @abc.abstractmethod
    def all(self, x: Array, axis: int | None = None) -> Array: ...

    @abc.abstractmethod
    def cumsum(self, x: Array, axis: int) -> Array: ...

    @abc.abstractmethod
    def nanmedian(self, x: Array, axis: int = -1) -> Array:
        """Median over `axis` ignoring NaN (the mean of the two middle values for an even
        count, as NumPy computes it); NaN where every value is NaN."""

    # ------------------------------------------------------------------
    # Structure and indexing
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def stack(self, arrays: Sequence[Array], axis: int = 0) -> Array: ...

    @abc.abstractmethod
    def concat(self, arrays: Sequence[Array], axis: int = 0) -> Array: ...

    @abc.abstractmethod
    def broadcast_to(self, x: Array, shape: Sequence[int]) -> Array: ...

    @abc.abstractmethod
    def moveaxis(self, x: Array, source: int, destination: int) -> Array: ...

    @abc.abstractmethod
    def take_along_axis(self, x: Array, indices: Array, axis: int) -> Array: ...

    @abc.abstractmethod
    def put_along_axis(self, x: Array, indices: Array, values: Array, axis: int) -> Array:
        """A copy of `x` with `values` written at `indices` along `axis` (no aliasing)."""

    @abc.abstractmethod
    def argsort(self, x: Array, axis: int = -1) -> Array:
        """Indices that sort `x` along `axis`, stably."""

    # ------------------------------------------------------------------
    # Linear algebra
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def einsum(self, subscripts: str, *operands: Array) -> Array: ...

    @abc.abstractmethod
    def solve(self, a: Array, b: Array) -> Array:
        """`x` with `a @ x = b`, batched over leading dimensions (`b` is `(..., n, k)`)."""

    @abc.abstractmethod
    def eye(self, n: int, like: Array | None = None) -> Array: ...

    # ------------------------------------------------------------------
    # Numerics
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def tiny(self, like: Array | None = None) -> float:
        """The smallest positive normal number of the float dtype at hand."""

    def errstate(self) -> contextlib.AbstractContextManager[Any]:
        """Silence floating-point warnings (NumPy's `errstate`; a no-op for torch)."""
        return contextlib.nullcontext()

    def __repr__(self) -> str:
        return f"<sekai backend {self.name!r}>"


__all__ = ["Array", "Backend"]
