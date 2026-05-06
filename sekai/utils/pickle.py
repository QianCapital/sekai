from __future__ import annotations

from typing import Any, Callable


class CloudpickleWrapper:
    """Wraps a callable so it can be sent across process boundaries via cloudpickle."""

    def __init__(self, fn: Callable[[], Any]) -> None:
        self.fn = fn

    def __call__(self) -> Any:
        return self.fn()

    def __getstate__(self) -> bytes:
        import cloudpickle
        return cloudpickle.dumps(self.fn)

    def __setstate__(self, state: bytes) -> None:
        import pickle
        self.fn = pickle.loads(state)
