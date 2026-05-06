from __future__ import annotations


class SekaiError(Exception):
    """Base exception for all sekai errors."""


class EnvNotFound(SekaiError):
    def __init__(self, id: str) -> None:
        super().__init__(
            f"Environment '{id}' not found in registry. "
            "Use sekai.registry.all_specs() to see registered environments."
        )


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
    def __init__(self) -> None:
        super().__init__(
            "reset() must be called before the first call to step(). "
            "See Env.reset() for usage."
        )


class AsyncCallPending(SekaiError):
    def __init__(self) -> None:
        super().__init__(
            "A previous async call is still pending. "
            "Await the previous result before making another call."
        )


class BackendMismatch(SekaiError):
    def __init__(self, expected: str, got: str) -> None:
        super().__init__(f"Backend mismatch: expected '{expected}', got '{got}'.")


class SpaceMismatch(SekaiError):
    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class DeprecatedAPI(SekaiError):
    def __init__(self, old: str, new: str) -> None:
        super().__init__(f"'{old}' is deprecated. Use '{new}' instead.")
