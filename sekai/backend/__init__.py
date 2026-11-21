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

(The `BackendOps` protocol that `sekai.backend` held until 0.2 is removed, as its
deprecation announced; nothing in sekai used it.)
"""

from __future__ import annotations

import sys
from typing import Any

from numpy.typing import NDArray

from sekai.backend._base import Array, Backend
from sekai.backend._numpy import NumpyBackend

#: The NumPy backend (a shared instance).
numpy_backend: Backend = NumpyBackend()

_torch_backend: Backend | None = None

_TORCH_HINT = 'The torch backend needs PyTorch: pip install "sekai[torch]" (or torch itself).'


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
        return numpy_backend
    if name == "torch":
        return _load_torch()
    raise ValueError(f"Unknown backend {name!r}; expected 'numpy' or 'torch'.")


def backend_of(*arrays: Any) -> Backend:
    """The backend of `arrays`: torch if any of them is a tensor, else NumPy."""
    if any(is_tensor(a) for a in arrays):
        return _load_torch()
    return numpy_backend


def backend_and_like(*arrays: Any) -> tuple[Backend, Any]:
    """`backend_of(arrays)` and the first tensor among them (None with NumPy).

    New arrays and converted inputs take that tensor's dtype and device.
    """
    for a in arrays:
        if is_tensor(a):
            return _load_torch(), a
    return numpy_backend, None


def to_numpy(x: Any) -> NDArray[Any]:
    """`x` as a NumPy array, whatever its backend (torch tensors are detached and copied
    to the CPU)."""
    return backend_of(x).to_numpy(x)


__all__ = [
    "Array",
    "Backend",
    "NumpyBackend",
    "numpy_backend",
    "get_backend",
    "backend_of",
    "backend_and_like",
    "is_tensor",
    "to_numpy",
]
