"""Array backends: the operations sekai's batched finance core is written in.

sekai's finance components (accounting, costs, constraint projection, rewards, the
risk model) and its batched market environment are written once against a
`Backend` and run on NumPy arrays or on torch tensors, on any device and with
autograd, so a policy can be trained by backpropagating through the environment.

`backend_of(*arrays)` picks the backend of the arrays it is given: torch if any of
them is a `torch.Tensor`, NumPy otherwise. NumPy is always available. torch is
optional (`pip install "sekai[torch]"`) and is imported only when a tensor is passed
in or `get_backend("torch")` is called, so `import sekai` never imports it.

Example:
    ```python
    from sekai.backend import backend_of

    def gross_exposure(weights):
        B = backend_of(weights)
        return B.sum(B.abs(weights), axis=-1)

    gross_exposure(np.ones((4, 3)))      # NumPy, float64
    gross_exposure(torch.ones(4, 3))     # torch, float32, differentiable
    ```

The `BackendOps` protocol and `default_backend` that `sekai.backend` held in 0.2 (and
the `sekai.backend.protocol` and `sekai.backend.numpy_backend` modules) are deprecated
but still work, with a `DeprecationWarning`; `NumpyBackend` keeps 0.2's methods.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sekai.backend._base import Array, Backend
from sekai.backend._numpy import NumpyBackend

# The NumPy backend (a shared instance, also returned by `get_backend("numpy")`).
_NUMPY: Backend = NumpyBackend()

_torch_backend: Backend | None = None

_TORCH_HINT = 'The torch backend needs PyTorch: pip install "sekai[torch]" (or torch itself).'

# Types that are never tensors. They are checked first because `isinstance` against
# `torch.Tensor` is slow (it has a metaclass check), and NumPy code calls the dispatch
# below several times a step: with these, it costs NumPy code next to nothing, also
# when torch is imported for a policy.
_NOT_TENSORS = frozenset({np.ndarray, float, int, bool, type(None), np.float64, np.bool_})


def _load_torch() -> Backend:
    global _torch_backend
    if _torch_backend is None:
        try:
            from sekai.backend._torch import TorchBackend
        except ImportError as e:  # pragma: no cover - torch is installed where tests run
            raise ImportError(_TORCH_HINT) from e
        _torch_backend = TorchBackend()
    return _torch_backend


def is_tensor(x: Any) -> bool:
    """Whether `x` is a torch tensor (without importing torch)."""
    if type(x) in _NOT_TENSORS:
        return False
    torch = sys.modules.get("torch")
    return torch is not None and isinstance(x, torch.Tensor)


def get_backend(name: str | Backend) -> Backend:
    """The backend called `name` ("numpy" or "torch"), or `name` itself if it is one.

    Raises:
        ValueError: For an unknown name.
        ImportError: For "torch" when PyTorch is not installed.
    """
    if isinstance(name, Backend):
        return name
    if name == "numpy":
        return _NUMPY
    if name == "torch":
        return _load_torch()
    raise ValueError(f"Unknown backend {name!r}; expected 'numpy' or 'torch'.")


def backend_of(*arrays: Any) -> Backend:
    """The backend of `arrays`: torch if any of them is a tensor, else NumPy."""
    return backend_and_like(*arrays)[0]


def backend_and_like(*arrays: Any) -> tuple[Backend, Any]:
    """`backend_of(arrays)` and the first tensor among them (None with NumPy).

    New arrays and converted inputs take that tensor's dtype and device.
    """
    torch = sys.modules.get("torch")
    if torch is not None:
        tensor = torch.Tensor
        for a in arrays:
            if type(a) not in _NOT_TENSORS and isinstance(a, tensor):
                return _load_torch(), a
    return _NUMPY, None


def to_numpy(x: Any) -> NDArray[Any]:
    """`x` as a NumPy array, whatever its backend (torch tensors are detached and copied
    to the CPU)."""
    return backend_of(x).to_numpy(x)


_DEPRECATED = (
    "is deprecated: it was to be removed in sekai 0.3 with the 0.2 backend protocol, and is "
    "kept for compatibility only. Use the sekai.backend.Backend interface "
    '(get_backend("numpy"), backend_of) instead.'
)


def __getattr__(name: str) -> Any:
    # sekai 0.2's backend API: still importable, with a DeprecationWarning.
    if name == "BackendOps":
        warnings.warn(f"sekai.backend.BackendOps {_DEPRECATED}", DeprecationWarning, stacklevel=2)
        from sekai.backend._protocol import BackendOps

        return BackendOps
    if name == "default_backend":
        warnings.warn(
            f"sekai.backend.default_backend {_DEPRECATED}", DeprecationWarning, stacklevel=2
        )
        return _NUMPY
    if name in ("protocol", "numpy_backend"):  # the 0.2 modules (they warn on import)
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    # The 0.2 names are listed, as they were, though served (with a warning) on access.
    return sorted({*globals(), "BackendOps", "default_backend", "protocol", "numpy_backend"})


__all__ = [
    "Array",
    "Backend",
    "NumpyBackend",
    "get_backend",
    "backend_of",
    "backend_and_like",
    "is_tensor",
    "to_numpy",
    # Deprecated (sekai 0.2)
    "BackendOps",
    "default_backend",
]
