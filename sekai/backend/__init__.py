"""Deprecated: array-backend protocol that no part of sekai uses.

`sekai.backend` was meant for non-NumPy (torch, JAX) array backends but was never
wired into environments, spaces or vector envs. It is kept for code that imports
it and will be removed in sekai 0.3. For fast batched simulation use a native
vector environment (`sekai.vector.NativeVecEnv`, `sekai.make_vec`).
"""

import warnings

from sekai.backend.protocol import BackendOps
from sekai.backend.numpy_backend import NumpyBackend

warnings.warn(
    "sekai.backend is deprecated and will be removed in sekai 0.3; nothing in sekai uses it. "
    "For fast batched simulation, see sekai.vector.NativeVecEnv.",
    DeprecationWarning,
    stacklevel=2,
)

default_backend: BackendOps = NumpyBackend()

__all__ = ["BackendOps", "NumpyBackend", "default_backend"]
