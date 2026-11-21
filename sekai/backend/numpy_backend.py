"""Deprecated: sekai 0.2's home of `NumpyBackend`, kept for compatibility.

Import `NumpyBackend` from `sekai.backend` instead.
"""

import warnings

from sekai.backend._numpy import NumpyBackend

warnings.warn(
    "sekai.backend.numpy_backend is deprecated: it was to be removed in sekai 0.3, and is "
    "kept for compatibility only. Import NumpyBackend from sekai.backend instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["NumpyBackend"]
