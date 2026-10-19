# Errors

Every exception sekai raises on purpose derives from `SekaiError`, so
`except sekai.SekaiError` catches them all. They are exported from `sekai` and `sekai.error`,
and they pickle cleanly, so they survive the trip from a worker process (`ProcVecEnv`,
`concurrent.futures.ProcessPoolExecutor`, ...).

Argument validation (bad shapes, out-of-range parameters, wrong space types) raises the
built-in `ValueError` / `TypeError` instead.

| Exception | Raised when |
|---|---|
| `SekaiError` | base class |
| `EnvNotFound` | an ID is not registered (with close-match `suggestions`) |
| `EnvAlreadyRegistered` | an ID is registered twice |
| `InvalidAction` | an environment rejects an action (built-in envs raise it; use it in yours) |
| `ResetRequired` | `step()` (or a renderer) is used before `reset()` |
| `VecEnvWorkerError` | a `ProcVecEnv` worker raised or died (carries `rank`, `exc_type`, `remote_traceback`) |
| `VecEnvClosed` | a vector env is used after `close()` |
| `LookAheadError` | a feature transform used information from the future (raised by `sekai.finance.features.check_causal`) |
| `InvalidObservation`, `SpaceMismatch`, `AsyncCallPending`, `BackendMismatch` | not raised by sekai itself; available for your environments and tools |

::: sekai.error
    options:
      show_root_heading: false
      show_if_no_docstring: true
      members:
        - SekaiError
        - EnvNotFound
        - EnvAlreadyRegistered
        - InvalidAction
        - InvalidObservation
        - ResetRequired
        - AsyncCallPending
        - BackendMismatch
        - SpaceMismatch
        - VecEnvWorkerError
        - VecEnvClosed
        - LookAheadError
