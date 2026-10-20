# sekai/sekai.py — convenience re-export of the full public API.
#
# Both of the following import styles work:
#   import sekai; sekai.Env
#   from sekai.sekai import Env, Box, StepResult
from typing import Any as _Any

from sekai import *
from sekai import __version__


def __getattr__(name: str) -> _Any:
    # Deprecated names left out of `import *`: still importable, with a warning, until sekai 0.3.
    if name == "BackendMismatch":
        import warnings

        from sekai.error import _BACKEND_MISMATCH_DEPRECATION, _BackendMismatch

        warnings.warn(f"sekai.{_BACKEND_MISMATCH_DEPRECATION}", DeprecationWarning, stacklevel=2)
        return _BackendMismatch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
