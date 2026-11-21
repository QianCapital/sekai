"""Deprecated: sekai 0.2's home of `BackendOps`, kept for compatibility.

Use the `sekai.backend.Backend` interface instead.
"""

import warnings

from sekai.backend._protocol import BackendOps

warnings.warn(
    "sekai.backend.protocol is deprecated: it was to be removed in sekai 0.3 with the 0.2 "
    "backend protocol, and is kept for compatibility only. Use the sekai.backend.Backend "
    "interface instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["BackendOps"]
