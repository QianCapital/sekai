from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any


def _rebuild(cls: type[SekaiError], args: tuple[Any, ...], kwargs: dict[str, Any]) -> SekaiError:
    return cls(*args, **kwargs)


class SekaiError(Exception):
    """Base exception for all sekai errors.

    Subclasses build their message from structured arguments, so the default
    exception pickling (which replays self.args, i.e. just the message) would
    fail. The constructor arguments are recorded instead, which keeps sekai
    errors picklable across processes (ProcVecEnv, ProcessPoolExecutor, ...).
    """

    _init_args: tuple[Any, ...]
    _init_kwargs: dict[str, Any]

    def __new__(cls, *args: Any, **kwargs: Any) -> SekaiError:
        self = super().__new__(cls, *args)
        self._init_args = args
        self._init_kwargs = kwargs
        return self

    def __reduce__(self) -> tuple[Any, ...]:
        return (_rebuild, (type(self), self._init_args, self._init_kwargs))


class EnvNotFound(SekaiError):
    def __init__(self, id: str, suggestions: Sequence[str] = ()) -> None:
        hint = f" Did you mean: {', '.join(repr(s) for s in suggestions)}?" if suggestions else ""
        super().__init__(
            f"Environment '{id}' not found in registry.{hint} "
            "Use sekai.all_specs() or `sekai list` to see registered environments."
        )
        self.id = id
        self.suggestions = tuple(suggestions)


class EnvAlreadyRegistered(SekaiError):
    def __init__(self, id: str) -> None:
        super().__init__(f"Environment '{id}' is already registered.")


class InvalidAction(SekaiError):
    def __init__(self, action: object, space: object) -> None:
        super().__init__(f"Action {action!r} is not valid for space {space!r}.")


class InvalidObservation(SekaiError):
    def __init__(self, obs: object, space: object) -> None:
        super().__init__(f"Observation {obs!r} is not valid for space {space!r}.")


class ResetRequired(SekaiError):
    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message
            or "reset() must be called before the first call to step(). See Env.reset() for usage."
        )


class AsyncCallPending(SekaiError):
    def __init__(self) -> None:
        super().__init__(
            "A previous async call is still pending. "
            "Await the previous result before making another call."
        )


class BackendMismatch(SekaiError):
    """Deprecated with `sekai.backend`; removed in sekai 0.3. Nothing in sekai raises it."""

    def __init__(self, expected: str, got: str) -> None:
        super().__init__(f"Backend mismatch: expected '{expected}', got '{got}'.")


# Reached only through the module __getattr__ below, which warns. (Pickling looks the
# class up by name, so it goes through __getattr__ too.)
_BackendMismatch = BackendMismatch
del BackendMismatch

_BACKEND_MISMATCH_DEPRECATION = (
    "BackendMismatch is deprecated and will be removed in sekai 0.3, with sekai.backend; "
    "nothing in sekai raises it."
)


def __getattr__(name: str) -> Any:
    if name == "BackendMismatch":
        warnings.warn(
            f"sekai.error.{_BACKEND_MISMATCH_DEPRECATION}", DeprecationWarning, stacklevel=2
        )
        return _BackendMismatch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class SpaceMismatch(SekaiError):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class DeprecatedAPI(SekaiError):
    def __init__(self, old: str, new: str) -> None:
        super().__init__(f"'{old}' is deprecated. Use '{new}' instead.")


class VecEnvWorkerError(SekaiError):
    """An exception raised inside a ProcVecEnv worker process."""

    def __init__(self, rank: int, exc_type: str, message: str, remote_traceback: str) -> None:
        detail = f"\n\nWorker traceback:\n{remote_traceback}" if remote_traceback else ""
        super().__init__(f"Sub-env {rank} raised {exc_type}: {message}{detail}")
        self.rank = rank
        self.exc_type = exc_type
        self.remote_traceback = remote_traceback


class VecEnvClosed(SekaiError):
    def __init__(self) -> None:
        super().__init__("This VecEnv has been closed.")


class LookAheadError(SekaiError):
    """A feature or data transform used information from the future (look-ahead bias)."""

    def __init__(self, msg: str) -> None:
        super().__init__(msg)
