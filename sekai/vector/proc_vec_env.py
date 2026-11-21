from __future__ import annotations

import contextlib
import dataclasses
import functools
import multiprocessing as mp
import signal
import sys
import time
import traceback
import warnings
from collections.abc import Callable, Sequence
from enum import Enum, auto
from multiprocessing import shared_memory as shm_module
from multiprocessing.connection import Connection
from multiprocessing.reduction import ForkingPickler
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sekai._types import ActType, ObsType, SeedType
from sekai.core.env import Env
from sekai.core.result import StepResult
from sekai.error import VecEnvClosed, VecEnvWorkerError
from sekai.spaces.box import Box
from sekai.spaces.dict import Dict
from sekai.spaces.discrete import Discrete
from sekai.spaces.multi_binary import MultiBinary
from sekai.spaces.multi_discrete import MultiDiscrete
from sekai.spaces.space import Space
from sekai.spaces.tuple import Tuple
from sekai.utils.pickle import CloudpickleWrapper
from sekai.vector.result import VecResetResult, VecStepResult
from sekai.vector.vec_env import (
    VecEnv,
    close_envs,
    env_call,
    env_get_attr,
    env_set_attr,
    frozen_step,
    merge_autoreset,
    needs_autoreset,
)


class _Cmd(Enum):
    STEP = auto()
    RESET = auto()
    CLOSE = auto()
    CALL = auto()
    GETATTR = auto()
    SETATTR = auto()
    SPACES = auto()
    SHARE = auto()
    RESTART = auto()


# ---------------------------------------------------------------------------
# Shared-memory observation buffers
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Leaf:
    path: tuple[str | int, ...]
    dtype: str
    shape: tuple[int, ...]  # per-env shape
    offset: int


def _plan_leaves(
    space: Space[Any],
) -> list[tuple[tuple[str | int, ...], np.dtype[Any], tuple[int, ...]]] | None:
    """(path, dtype, shape) of every array leaf of space, or None if any leaf is not fixed-size."""
    out: list[tuple[tuple[str | int, ...], np.dtype[Any], tuple[int, ...]]] = []

    def visit(sub: Space[Any], path: tuple[str | int, ...]) -> bool:
        if isinstance(sub, Dict):
            return all(visit(s, (*path, key)) for key, s in sub.items())
        if isinstance(sub, Tuple):
            return all(visit(s, (*path, i)) for i, s in enumerate(sub.spaces))
        if isinstance(sub, Box):
            out.append((path, sub.dtype, sub.shape))
        elif isinstance(sub, Discrete):
            out.append((path, np.dtype(np.intp), ()))
        elif isinstance(sub, MultiDiscrete):
            out.append((path, sub.dtype, tuple(sub.nvec.shape)))
        elif isinstance(sub, MultiBinary):
            out.append((path, np.dtype(np.int8), sub.n))
        else:
            return False
        return True

    return out if visit(space, ()) else None


class _SharedObservations:
    """One shared-memory block holding a batched observation, written by workers."""

    def __init__(
        self,
        block: shm_module.SharedMemory,
        space: Space[Any],
        leaves: list[_Leaf],
        num_envs: int,
        owner: bool,
    ) -> None:
        self.block = block
        self.space = space
        self.leaves = leaves
        self.num_envs = num_envs
        self.owner = owner
        self._views: list[NDArray[Any]] = [
            np.ndarray(
                (num_envs, *leaf.shape),
                dtype=np.dtype(leaf.dtype),
                buffer=block.buf,
                offset=leaf.offset,
            )
            for leaf in leaves
        ]

    @classmethod
    def create(cls, space: Space[Any], num_envs: int) -> _SharedObservations | None:
        plan = _plan_leaves(space)
        if plan is None:
            return None
        leaves: list[_Leaf] = []
        offset = 0
        for path, dtype, shape in plan:
            leaves.append(_Leaf(path, dtype.str, tuple(shape), offset))
            nbytes = int(np.prod((num_envs, *shape))) * dtype.itemsize
            offset += -(-nbytes // 64) * 64  # 64-byte aligned
        block = shm_module.SharedMemory(create=True, size=max(offset, 1))
        return cls(block, space, leaves, num_envs, owner=True)

    def handle(self) -> tuple[str, Space[Any], list[_Leaf], int]:
        """What a worker needs to attach (sent once over the command pipe)."""
        return (self.block.name, self.space, self.leaves, self.num_envs)

    @classmethod
    def attach(
        cls, name: str, space: Space[Any], leaves: list[_Leaf], num_envs: int
    ) -> _SharedObservations:
        # Only the parent owns (and unlinks) the block. Workers share the parent's
        # resource tracker; from Python 3.13 they can opt out of tracking entirely.
        if sys.version_info >= (3, 13):
            block = shm_module.SharedMemory(name=name, track=False)
        else:
            block = shm_module.SharedMemory(name=name)
        return cls(block, space, leaves, num_envs, owner=False)

    @staticmethod
    def _get(obs: Any, path: tuple[str | int, ...]) -> Any:
        for key in path:
            obs = obs[key]
        return obs

    def write(self, index: int, obs: Any) -> None:
        for leaf, view in zip(self.leaves, self._views):
            view[index] = self._get(obs, leaf.path)

    def read(self) -> Any:
        """A copy of the current batch, structured like the observation space."""
        return self._build([view.copy() for view in self._views])

    def read_one(self, index: int) -> Any:
        """A copy of one sub-env's current observation."""
        return self._build([view[index].copy() for view in self._views])

    def _build(self, leaves: list[Any]) -> Any:
        arrays = iter(leaves)

        def build(sub: Space[Any]) -> Any:
            if isinstance(sub, Dict):
                return {key: build(s) for key, s in sub.items()}
            if isinstance(sub, Tuple):
                return tuple(build(s) for s in sub.spaces)
            return next(arrays)

        return build(self.space)

    def close(self) -> None:
        self._views = []
        with contextlib.suppress(Exception):
            self.block.close()
        if self.owner:
            with contextlib.suppress(Exception):
                self.block.unlink()


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _identity(x: Any) -> Any:
    return x


def _dumps(command: Any) -> bytes:
    """Pickle a command for the pipe.

    As bytes, not the memoryview ForkingPickler returns: a memoryview kept alive by a
    traceback (e.g. of a VecEnvWorkerError) pins its BytesIO, which then reports
    "BufferError: Existing exports of data" when the two are garbage-collected.
    """
    return bytes(ForkingPickler.dumps(command))


# Seconds a terminated (then killed) worker gets to exit before close() moves on.
_TERMINATE_GRACE = 1.0


class _WorkerLost(Exception):
    """A worker that died, whose pipe broke, or that did not reply within step_timeout.

    Internal: ProcVecEnv restarts the worker, or raises VecEnvWorkerError for it.
    """

    def __init__(self, worker: int, exc_type: str, cause: str) -> None:
        super().__init__(worker, exc_type, cause)
        self.worker = worker
        self.exc_type = exc_type
        self.cause = cause


def _worker(
    env_indices: list[int],
    env_fns: CloudpickleWrapper,
    pipe: Connection,
    parent_pipe: Connection,
    autoreset: bool,
) -> None:
    """Serve commands for a group of sub-envs. Every reply is (ok, payload).

    Commands carry one payload per sub-env of the group, and replies one result
    per sub-env. On failure the payload is (env index, exception type name,
    message, formatted traceback) rather than the exception itself, which may
    not be picklable. The worker keeps serving after an error so the parent can
    still close it cleanly.
    """
    parent_pipe.close()
    # Ctrl-C in a terminal reaches the whole process group; let the parent handle it
    # (it drains or closes the workers) instead of killing every worker mid-call.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    envs: list[Env[Any, Any]] = []
    try:
        for fn in env_fns():
            envs.append(fn())
    except Exception as e:
        index = env_indices[len(envs)]
        pipe.send((False, (index, type(e).__name__, str(e), traceback.format_exc())))
        with contextlib.suppress(Exception):  # the construction error is the one reported
            close_envs(envs)
        pipe.close()
        return
    pipe.send((True, None))
    shared: _SharedObservations | None = None
    # The step that ended each frozen sub-env's episode (autoreset off; see VecEnv).
    finals: list[StepResult[Any] | None] = [None] * len(envs)

    while True:
        try:
            cmd, data = pipe.recv()
        except EOFError:
            break
        current = env_indices[0]
        try:
            payload: Any
            if cmd is _Cmd.STEP:
                payload = []
                for k, (index, env, action) in enumerate(zip(env_indices, envs, data)):
                    current = index
                    final = finals[k]
                    if final is not None:
                        result = frozen_step(final)
                    else:
                        result = env.step(action)
                        if needs_autoreset(result):
                            if autoreset:
                                result = merge_autoreset(result, env.reset())
                            else:
                                finals[k] = result
                    if shared is not None:
                        shared.write(index, result.observation)
                        result = dataclasses.replace(result, observation=None)
                    payload.append(result)
            elif cmd is _Cmd.RESET:
                finals = [None] * len(envs)
                payload = []
                for index, env, (seed, options) in zip(env_indices, envs, data):
                    current = index
                    reset = env.reset(seed=seed, options=options)
                    if shared is not None:
                        shared.write(index, reset.observation)
                        reset = dataclasses.replace(reset, observation=None)
                    payload.append(reset)
            elif cmd is _Cmd.CLOSE:
                # Close every env even if one fails, then always reply and stop serving,
                # so the parent never waits for a worker that is still running.
                failure: tuple[int, str, str, str] | None = None
                for index, env in zip(env_indices, envs):
                    try:
                        env.close()
                    except Exception as e:
                        if failure is None:
                            failure = (index, type(e).__name__, str(e), traceback.format_exc())
                if shared is not None:
                    shared.close()
                    shared = None
                with contextlib.suppress(OSError):
                    pipe.send((True, None) if failure is None else (False, failure))
                break
            elif cmd is _Cmd.CALL:
                name, args, kwargs = data
                payload = []
                for index, env in zip(env_indices, envs):
                    current = index
                    payload.append(env_call(env, name, args, kwargs))
            elif cmd is _Cmd.GETATTR:
                payload = []
                for index, env in zip(env_indices, envs):
                    current = index
                    payload.append(env_get_attr(env, data))
            elif cmd is _Cmd.SETATTR:
                name, values = data
                for index, env, value in zip(env_indices, envs, values):
                    current = index
                    env_set_attr(env, name, value)
                payload = None
            elif cmd is _Cmd.SPACES:
                payload = (envs[0].observation_space, envs[0].action_space)
            elif cmd is _Cmd.SHARE:
                shared = _SharedObservations.attach(*data)
                payload = None
            elif cmd is _Cmd.RESTART:
                # This worker replaces a lost one, in the middle of a step (see ProcVecEnv).
                # data: per sub-env, the seed to reset it with and, with autoreset off, the
                # final step it was frozen on (None if it was still playing).
                payload = []
                for k, (index, env, (seed, final)) in enumerate(zip(env_indices, envs, data)):
                    current = index
                    reset = env.reset(seed=seed)
                    if final is None:
                        result = StepResult(
                            observation=reset.observation,
                            reward=0.0,
                            terminated=False,
                            truncated=True,
                            info={"worker_restarted": True, "reset_info": reset.info},
                        )
                        finals[k] = None if autoreset else result  # it finished: frozen
                    else:
                        finals[k] = final  # still frozen, on the same final step
                        result = frozen_step(final)
                    if shared is not None:
                        shared.write(index, result.observation)
                        result = dataclasses.replace(result, observation=None)
                    payload.append(result)
            else:
                raise RuntimeError(f"Unknown command: {cmd}")
            pipe.send((True, payload))
        except Exception as e:
            pipe.send((False, (current, type(e).__name__, str(e), traceback.format_exc())))
    if shared is not None:
        shared.close()
    pipe.close()


# ---------------------------------------------------------------------------
# ProcVecEnv
# ---------------------------------------------------------------------------


class ProcVecEnv(VecEnv[ObsType, ActType]):
    _snapshot_unsupported = (
        "ProcVecEnv's sub-envs live in worker processes, so it cannot be snapshotted; "
        "use a SyncVecEnv or a native vector env, or snapshot the sub-envs inside the "
        "workers with vec.call('snapshot')."
    )

    """Vectorised env using worker processes for true CPU-bound parallelism.

    Sub-environments run in worker processes and communicate via
    multiprocessing Pipes. Use this for CPU-intensive simulators; for I/O-bound
    or async environments use AsyncVecEnv, and for simulators that vectorise
    with array operations prefer a native vector env (see `sekai.make_vec`).

    Two things keep the inter-process overhead low:

    - **Shared memory** (on by default when the observation space has fixed
      shapes: Box, Discrete, MultiDiscrete, MultiBinary and Dict/Tuple/MaskedBox
      of them). Workers write observations straight into one shared block, so
      only rewards, flags and infos are pickled. Large observations (order
      books, multi-asset windows) no longer pay a serialisation cost per step.
    - **Several sub-envs per worker** (`num_workers`): with more sub-envs than
      CPU cores, grouping them avoids oversubscription and amortises each
      round trip over several envs.

    The default start method is "forkserver" ("spawn" on Windows), so the
    script that creates a ProcVecEnv must guard its entry point with
    `if __name__ == "__main__":`. env_fns are sent with cloudpickle, so
    lambdas and locally defined classes work.

    An exception inside a sub-env is re-raised in the parent as
    VecEnvWorkerError, carrying the sub-env index and the worker's traceback.
    The other workers' replies are still collected, so the VecEnv stays usable
    afterwards (the step results of that call are discarded, including those of
    sub-envs that did advance); replies left unread by an interrupted call (e.g.
    Ctrl-C, which workers ignore and leave to the parent) are discarded before
    the next command. A worker that dies also raises VecEnvWorkerError, unless it
    is restarted (see Fault tolerance): its `rank` is the sub-env it ran, or, for a
    worker that ran several (so which one failed is unknown), the first of them,
    and the message names the worker and all its sub-envs.

    **Fault tolerance.** With `max_restarts > 0`, a worker lost during step() or
    reset() (its process died, e.g. killed for using too much memory or crashed in
    native code, or it did not reply within `step_timeout` and was terminated) is
    replaced by a new worker that builds the same sub-envs from the same env_fns,
    and attaches to the same shared memory. Each replacement emits a RuntimeWarning
    naming the worker and the cause, and counts towards `restarts`. The state of
    the lost sub-envs is gone (as are attributes set on them with set_attr()), so:

    - In step(), every sub-env of the lost worker is reset (its action is dropped)
      and returns the first observation of its new episode, reward 0,
      `terminated=False`, `truncated=True` and an info holding only
      `"worker_restarted": True` and `"reset_info"`. There is no
      `"final_observation"`: treat it as a truncation without a final observation
      (its action was not played, and its observation belongs to the new episode,
      so do not bootstrap from it). With autoreset off it is then frozen, as any
      finished sub-env (see VecEnv), on that observation; a sub-env that was
      already frozen stays frozen on its own final step, and its result does not
      change.
    - In reset(), the reset is redone on the new worker with the same seeds and
      options, and the infos of its sub-envs gain `"worker_restarted": True`.
    - call(), get_attr() and set_attr() never restart a worker: one lost during
      them raises VecEnvWorkerError (what it was doing is lost), and the next
      step() or reset() restarts it, reporting the restart as above.

    Restarts are reproducible: a restarted sub-env is reset with a seed derived
    from the one that last seeded it. On its k-th restart (k = 0, 1, ...) since then,
    sub-env i gets the seed `root.spawn(k + 1)[k].generate_state(1)[0]`, where `root`
    is the SeedSequence its seed came from: `SeedSequence(s).spawn(num_envs)[i]`
    after `reset(seed=s)`, `SeedSequence(s_i)` after `reset(seed=[..., s_i, ...])`
    (a None seed keeps the root). A sub-env never seeded with an int is restarted
    unseeded. A run with a given seed and the same failures therefore replays
    exactly.

    A lost worker that cannot be restarted (max_restarts is 0 or used up), or whose
    replacement fails to start or to reset, raises VecEnvWorkerError naming the
    worker, the cause and the restarts used. The worker stays lost: every later call
    that needs it raises too (step() and reset() try another replacement if restarts
    remain), and close() still releases everything.

    Args:
        env_fns: Callables that each build one sub-environment (in its worker). If
            one raises (or anything else in the constructor fails), the sub-envs
            already built are closed and the workers stopped before the error
            propagates.
        autoreset: Reset sub-envs automatically when their episode ends (see VecEnv).
        context: multiprocessing start method; defaults to "forkserver" ("spawn" on Windows).
        close_timeout: Seconds close() waits for workers before terminating them.
        num_workers: Worker processes; sub-envs are split evenly between them.
            Defaults to one process per sub-env.
        shared_memory: Pass observations through shared memory. None (default)
            enables it when the observation space allows; True requires it.
        max_restarts: How many lost workers this vec env may replace over its lifetime
            (see Fault tolerance above). 0 (default) replaces none: a lost worker
            raises VecEnvWorkerError.
        step_timeout: Seconds step() and reset() wait for each worker's reply,
            including a replacement's reset but not the building of its sub-envs.
            A worker that does not reply in time is terminated and counts as lost.
            None (default) waits for ever. call(), get_attr() and set_attr() always
            wait.
    """

    # The constructor runs with these; the arguments apply once every worker is up.
    max_restarts: int = 0
    step_timeout: float | None = None

    def __init__(
        self,
        env_fns: Sequence[Callable[[], Env[ObsType, ActType]]],
        *,
        autoreset: bool = True,
        context: str | None = None,
        close_timeout: float = 5.0,
        num_workers: int | None = None,
        shared_memory: bool | None = None,
        max_restarts: int = 0,
        step_timeout: float | None = None,
    ) -> None:
        if len(env_fns) == 0:
            raise ValueError("ProcVecEnv needs at least one env_fn.")
        if num_workers is not None and num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}.")
        if max_restarts < 0:
            raise ValueError(f"max_restarts must be >= 0, got {max_restarts}.")
        if step_timeout is not None and not step_timeout > 0:
            raise ValueError(f"step_timeout must be > 0 or None, got {step_timeout}.")
        if context is None:
            context = "spawn" if sys.platform == "win32" else "forkserver"
        ctx: Any = mp.get_context(context)
        if sys.platform != "win32":
            # Start the resource tracker before any worker, so that forked workers share
            # it and attaching to the shared-memory block never registers it twice.
            from multiprocessing import resource_tracker

            resource_tracker.ensure_running()
        self.num_envs = len(env_fns)
        self.num_workers = min(num_workers or self.num_envs, self.num_envs)
        self.autoreset = autoreset
        self.close_timeout = close_timeout
        self.closed = False
        self._ctx = ctx
        self._groups: list[list[int]] = [
            [int(i) for i in group]
            for group in np.array_split(np.arange(self.num_envs), self.num_workers)
        ]
        self._pipes: list[Connection] = []
        self._processes: list[Any] = []
        self._shared: _SharedObservations | None = None
        # Replies each worker owes us. Anything left over from an interrupted or
        # partially sent command is drained before the next one, so a reply can
        # never be read as the answer to a later command.
        self._pending: list[int] = []
        # Each worker's env_fns, kept to start a replacement.
        self._worker_fns: list[CloudpickleWrapper] = []
        self._worker_restarts = [0] * self.num_workers
        # Per sub-env: the SeedSequence (entropy, spawn key) that last seeded it, and
        # its restarts since then; its restart seeds derive from them.
        self._seed_roots: list[tuple[int, tuple[int, ...]] | None] = [None] * self.num_envs
        self._restarts_since_seeding = [0] * self.num_envs
        # With autoreset off and restarts on: the final step each frozen sub-env is
        # frozen on, for a replacement worker to freeze it on again.
        self._frozen: list[StepResult[Any] | None] = [None] * self.num_envs

        try:
            for worker, group in enumerate(self._groups):
                fns = [env_fns[i] for i in group]
                self._worker_fns.append(CloudpickleWrapper(functools.partial(_identity, fns)))
                self._start(worker)

            self._gather()
            self.observation_space, self.action_space = self._request(0, _Cmd.SPACES, None)
            if shared_memory is not False:
                self._shared = _SharedObservations.create(self.observation_space, self.num_envs)
                if self._shared is None and shared_memory:
                    raise TypeError(
                        "shared_memory=True needs fixed-size observations (Box, Discrete, "
                        "MultiDiscrete, MultiBinary, or Dict/Tuple of them); got "
                        f"{self.observation_space!r}."
                    )
                if self._shared is not None:
                    self._send_all(_Cmd.SHARE, [self._shared.handle()] * self.num_workers)
                    self._gather()
        except BaseException:
            # Close the sub-envs the healthy workers built (a worker whose env_fn failed
            # closes its own), then make sure every worker has stopped.
            try:
                with contextlib.suppress(Exception):  # the original error is the one to report
                    self.close()
            finally:
                self._terminate()
            raise
        self.max_restarts = max_restarts
        self.step_timeout = step_timeout

    @property
    def shared_memory(self) -> bool:
        """Whether observations travel through shared memory."""
        return self._shared is not None

    @property
    def restarts(self) -> int:
        """Workers replaced so far, out of max_restarts (failed replacements included)."""
        return sum(self._worker_restarts)

    @property
    def worker_restarts(self) -> tuple[int, ...]:
        """How many times each worker has been replaced so far (see restarts)."""
        return tuple(self._worker_restarts)

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------

    def _start(self, worker: int) -> None:
        """Start the process for one worker's sub-envs: the first one, or a replacement."""
        parent_pipe, child_pipe = self._ctx.Pipe()
        if worker < len(self._pipes):
            self._pipes[worker] = parent_pipe
        else:
            self._pipes.append(parent_pipe)
            self._pending.append(0)
        proc = self._ctx.Process(
            target=_worker,
            args=(
                self._groups[worker],
                self._worker_fns[worker],
                child_pipe,
                parent_pipe,
                self.autoreset,
            ),
            daemon=True,
            name=f"sekai-ProcVecEnv-worker-{worker}",
        )
        try:
            proc.start()  # e.g. env_fns that cloudpickle cannot pickle
        finally:
            child_pipe.close()
        if worker < len(self._processes):
            self._processes[worker] = proc
        else:
            self._processes.append(proc)
        self._pending[worker] = 1  # the worker's "env_fns succeeded" handshake

    def _stop_worker(self, worker: int) -> None:
        """Close a lost worker's pipe and make sure its process has exited (or kill it)."""
        self._pending[worker] = 0
        self._pipes[worker].close()  # a late reply can never be read
        proc = self._processes[worker]
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=_TERMINATE_GRACE)
        if proc.is_alive():
            proc.kill()
        proc.join(timeout=_TERMINATE_GRACE)

    def _restart(self, lost: _WorkerLost, cmd: _Cmd, payload: list[Any]) -> Any:
        """Replace a lost worker by a new one for the same sub-envs; return its reply to cmd."""
        worker = lost.worker
        self._stop_worker(worker)
        self._worker_restarts[worker] += 1
        count = f"restart {self.restarts} of max_restarts={self.max_restarts}"
        warnings.warn(
            f"ProcVecEnv worker {worker} ({self._describe(worker)}) was lost "
            f"({lost.exc_type}: {lost.cause}); restarting it ({count}).",
            RuntimeWarning,
            stacklevel=4,  # the caller of step() or reset(), through _recover()
        )
        try:
            self._start(worker)
            self._recv(worker)  # its sub-envs are built (no timeout, as in the constructor)
            if self._shared is not None:
                self._send(worker, _dumps((_Cmd.SHARE, self._shared.handle())))
                self._recv(worker)
            self._send(worker, _dumps((cmd, payload)))
            return self._recv(worker, self._deadline())
        except BaseException as e:
            # Never leave a half-started replacement (e.g. after Ctrl-C): stopped, the
            # worker is lost again, and the next step() or reset() can replace it.
            self._stop_worker(worker)
            if not isinstance(e, Exception):
                raise
            raise self._restart_failed(lost, e, count) from e

    def _recover(
        self,
        lost: dict[int, _WorkerLost],
        replies: list[Any],
        cmd: _Cmd,
        payload: Callable[[int], list[Any]],
    ) -> None:
        """Restart the lost workers in order, putting each one's reply to cmd in replies."""
        for worker in sorted(lost):
            replies[worker] = self._restart(lost[worker], cmd, payload(worker))

    def _deadline(self) -> float | None:
        """When a reply due in step() or reset() must have arrived, if there is a step_timeout."""
        return None if self.step_timeout is None else time.monotonic() + self.step_timeout

    def _timed_out(self, worker: int) -> _WorkerLost:
        """Terminate a worker that did not reply within step_timeout: it is lost."""
        self._stop_worker(worker)
        return _WorkerLost(
            worker,
            "TimeoutError",
            f"no reply within step_timeout={self.step_timeout} s; the worker was terminated",
        )

    # ------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------

    def _describe(self, worker: int) -> str:
        group = self._groups[worker]
        if len(group) == 1:
            return f"sub-env {group[0]}"
        return f"sub-envs {group[0]}-{group[-1]}"  # the groups are contiguous

    def _lost_error(self, lost: _WorkerLost, note: str = "") -> VecEnvWorkerError:
        """The error for a lost worker (one that died, broke its pipe or timed out).

        It names the sub-env when the worker ran only one. Otherwise which of its
        sub-envs failed is unknown, so the message names the worker and all of them,
        and `rank` is the first. A note (on restarts) also names the worker.
        """
        group = self._groups[lost.worker]
        error = VecEnvWorkerError(group[0], lost.exc_type, lost.cause + note, "")
        if len(group) > 1 or note:
            error.args = (
                f"Worker {lost.worker} ({self._describe(lost.worker)}) raised "
                f"{lost.exc_type}: {lost.cause}{note}",
            )
        error.__cause__ = lost.__cause__
        return error

    def _not_restarted(self, num_lost: int = 1) -> str:
        """Why lost workers are not restarted, for their error ("" without restarts)."""
        if self.max_restarts == 0:
            return ""
        lost = "1 worker lost" if num_lost == 1 else f"{num_lost} workers lost"
        return (
            f"; not restarted ({lost}, {self.restarts} of max_restarts="
            f"{self.max_restarts} restarts used)"
        )

    def _restarted_later(self) -> str:
        """For a worker lost in call(), get_attr() or set_attr(): what happens next."""
        if self.restarts < self.max_restarts:
            return "; the next step() or reset() restarts it"
        return self._not_restarted()

    def _restart_failed(self, lost: _WorkerLost, error: Exception, count: str) -> VecEnvWorkerError:
        """The error for a lost worker whose replacement failed to start or to reset."""
        worker = lost.worker
        rank, remote_traceback = self._groups[worker][0], ""
        if isinstance(error, VecEnvWorkerError):
            rank, exc_type, remote_traceback = error.rank, error.exc_type, error.remote_traceback
            reason = str(error).partition("\n\nWorker traceback:")[0]
        elif isinstance(error, _WorkerLost):
            exc_type, reason = error.exc_type, f"{error.exc_type}: {error.cause}"
        else:
            exc_type, reason = type(error).__name__, f"{type(error).__name__}: {error}"
        message = (
            f"Worker {worker} ({self._describe(worker)}) was lost ({lost.exc_type}: "
            f"{lost.cause}) and could not be restarted ({count}): {reason}"
        )
        failed = VecEnvWorkerError(rank, exc_type, message, remote_traceback)
        detail = f"\n\nWorker traceback:\n{remote_traceback}" if remote_traceback else ""
        failed.args = (message + detail,)
        return failed

    # ------------------------------------------------------------------
    # Pipe protocol
    # ------------------------------------------------------------------

    def _recv(self, worker: int, deadline: float | None = None) -> Any:
        """One reply from a worker; with a deadline (time.monotonic()), a lost one after it."""
        pipe = self._pipes[worker]
        try:
            if deadline is not None and not pipe.poll(max(deadline - time.monotonic(), 0.0)):
                timed_out = True
            else:
                timed_out = False
                ok, payload = pipe.recv()
        except (EOFError, OSError) as e:
            self._pending[worker] = 0
            raise _WorkerLost(worker, type(e).__name__, "worker process exited unexpectedly") from e
        if timed_out:
            raise self._timed_out(worker)
        self._pending[worker] -= 1
        if not ok:
            raise VecEnvWorkerError(*payload)
        return payload

    def _send(self, worker: int, message: bytes) -> None:
        """Send one pre-pickled command; count the reply it will produce."""
        try:
            self._pipes[worker].send_bytes(message)
        except OSError as e:
            raise _WorkerLost(worker, type(e).__name__, "worker process is not running") from e
        self._pending[worker] += 1

    def _request(self, worker: int, cmd: _Cmd, data: Any) -> Any:
        """Send one command to one worker and return its reply."""
        try:
            self._send(worker, _dumps((cmd, data)))
            return self._recv(worker)
        except _WorkerLost as e:
            raise self._lost_error(e) from e.__cause__

    def _drain(self, deadline: float | None = None) -> dict[int, _WorkerLost]:
        """Discard replies left unread by an interrupted or failed earlier call.

        With a deadline, a worker whose replies have not all arrived by then is
        terminated and returned as lost.
        """
        lost: dict[int, _WorkerLost] = {}
        for worker, pipe in enumerate(self._pipes):
            while self._pending[worker] > 0:
                try:
                    if deadline is not None and not pipe.poll(
                        max(deadline - time.monotonic(), 0.0)
                    ):
                        lost[worker] = self._timed_out(worker)
                        break
                    pipe.recv()
                except (EOFError, OSError):
                    self._pending[worker] = 0
                    break
                self._pending[worker] -= 1
        return lost

    def _gather(self, note: str = "") -> list[Any]:
        """Receive one reply from every worker, then raise the first error, if any."""
        results: list[Any] = []
        first_error: VecEnvWorkerError | None = None
        for worker in range(self.num_workers):
            try:
                results.append(self._recv(worker))
            except _WorkerLost as e:
                results.append(None)
                first_error = first_error or self._lost_error(e, note)
            except VecEnvWorkerError as e:
                results.append(None)
                first_error = first_error or e
        if first_error is not None:
            raise first_error
        return results

    def _gather_envs(self, note: str = "") -> list[Any]:
        """Gather per-worker reply lists and flatten them into per-env order."""
        return [item for reply in self._gather(note) for item in reply]

    def _send_all(
        self, cmd: _Cmd, payloads: Sequence[Any], *, recover: bool = False, note: str = ""
    ) -> dict[int, _WorkerLost]:
        """Send one payload to each worker.

        A lost worker raises VecEnvWorkerError at once (with `note`). With `recover`
        (step() and reset()) and restarts left, the lost workers are returned instead,
        for _collect() and _recover().
        """
        if self.closed:
            raise VecEnvClosed()
        lost = self._drain(self._deadline() if recover else None)
        # Pickle everything first: if any payload is unpicklable, nothing is sent.
        messages = [_dumps((cmd, payload)) for payload in payloads]
        for worker, message in enumerate(messages):
            if worker not in lost:
                try:
                    self._send(worker, message)
                except _WorkerLost as e:
                    lost[worker] = e
            if worker in lost and not (recover and self.restarts < self.max_restarts):
                error = self._lost_error(lost[worker], self._not_restarted() if recover else note)
                raise error from lost[worker].__cause__
        return lost

    def _collect(self, lost: dict[int, _WorkerLost], deadline: float | None) -> list[Any]:
        """The reply of every worker not in `lost`, adding the workers lost meanwhile to it.

        Raises the first error in worker order: a sub-env's exception, or a lost worker
        if there are not enough restarts left for all of them. The lost workers are
        then left as they are (the next step() or reset() finds them lost).
        """
        replies: list[Any] = [None] * self.num_workers
        errors: dict[int, VecEnvWorkerError] = {}
        for worker in range(self.num_workers):
            if worker in lost:
                continue
            try:
                replies[worker] = self._recv(worker, deadline)
            except _WorkerLost as e:
                lost[worker] = e
            except VecEnvWorkerError as e:
                errors[worker] = e
        if len(lost) > self.max_restarts - self.restarts:
            note = self._not_restarted(len(lost))
            errors.update({worker: self._lost_error(e, note) for worker, e in lost.items()})
        if errors:
            raise errors[min(errors)]
        return replies

    def _per_worker(self, per_env: Sequence[Any]) -> list[list[Any]]:
        return [[per_env[i] for i in group] for group in self._groups]

    # ------------------------------------------------------------------
    # Seeding restarts
    # ------------------------------------------------------------------

    def _record_seeds(self, seed: list[SeedType] | SeedType) -> None:
        """Note the SeedSequence behind each sub-env's new seed (see the class docstring)."""
        if isinstance(seed, list):
            for i, s in enumerate(seed):
                if s is None:
                    continue  # the sub-env continues its stream, from the same root
                if isinstance(s, (int, np.integer)) and s >= 0:
                    self._seed_roots[i] = (int(s), ())
                else:
                    self._seed_roots[i] = None  # a Generator (or a seed reset() rejects)
                self._restarts_since_seeding[i] = 0
        elif isinstance(seed, (int, np.integer)):
            # reset() seeds sub-env i from SeedSequence(seed).spawn(num_envs)[i].
            for i in range(self.num_envs):
                self._seed_roots[i] = (int(seed), (i,))
                self._restarts_since_seeding[i] = 0

    def _restart_seed(self, index: int) -> int | None:
        """The seed for sub-env `index`'s next restart (None if it was never seeded)."""
        k = self._restarts_since_seeding[index]
        self._restarts_since_seeding[index] += 1
        root = self._seed_roots[index]
        if root is None:
            return None
        entropy, key = root
        # SeedSequence(entropy, spawn_key=(*key, k)) is the k-th child of the one that
        # seeded the sub-env, i.e. its .spawn(k + 1)[k].
        return int(np.random.SeedSequence(entropy, spawn_key=(*key, k)).generate_state(1)[0])

    def _track_frozen(self, results: Sequence[StepResult[Any]]) -> None:
        """Note the final step of each sub-env that has just been frozen (autoreset off)."""
        for i, result in enumerate(results):
            if self._frozen[i] is None and needs_autoreset(result):
                obs = result.observation if self._shared is None else self._shared.read_one(i)
                self._frozen[i] = StepResult(obs, 0.0, result.terminated, result.truncated, {})

    # ------------------------------------------------------------------
    # VecEnv API
    # ------------------------------------------------------------------

    def step(self, actions: Any) -> VecStepResult[Any]:
        per_worker = self._per_worker(self._split_actions(actions))
        lost = self._send_all(_Cmd.STEP, per_worker, recover=True)
        replies = self._collect(lost, self._deadline())

        def restart(worker: int) -> list[Any]:
            return [(self._restart_seed(i), self._frozen[i]) for i in self._groups[worker]]

        self._recover(lost, replies, _Cmd.RESTART, restart)
        results = [result for reply in replies for result in reply]
        observations = self._shared.read() if self._shared is not None else None
        if not self.autoreset and self.max_restarts:
            self._track_frozen(results)
        return self._batch_step(results, observations)

    def reset(
        self,
        *,
        seed: list[SeedType] | SeedType = None,
        options: list[dict[str, Any] | None] | dict[str, Any] | None = None,
    ) -> VecResetResult[Any]:
        seeds = self._expand_seeds(seed)
        opts = self._expand_options(options)
        self._record_seeds(seed)
        self._frozen = [None] * self.num_envs
        lost = self._send_all(_Cmd.RESET, self._per_worker(list(zip(seeds, opts))), recover=True)
        replies = self._collect(lost, self._deadline())

        def redo(worker: int) -> list[Any]:
            # The same reset; a sub-env given no seed gets its restart seed.
            payload = []
            for i in self._groups[worker]:
                restart_seed = self._restart_seed(i)
                payload.append((restart_seed if seeds[i] is None else seeds[i], opts[i]))
            return payload

        restarted = sorted(lost)
        self._recover(lost, replies, _Cmd.RESET, redo)
        for worker in restarted:
            replies[worker] = [
                dataclasses.replace(reset, info={**reset.info, "worker_restarted": True})
                for reset in replies[worker]
            ]
        results = [reset for reply in replies for reset in reply]
        observations = self._shared.read() if self._shared is not None else None
        return self._batch_reset(results, observations)

    def call(self, method: str, *args: Any, **kwargs: Any) -> list[Any]:
        note = self._restarted_later()
        self._send_all(_Cmd.CALL, [(method, args, kwargs)] * self.num_workers, note=note)
        return self._gather_envs(note)

    def get_attr(self, name: str) -> list[Any]:
        note = self._restarted_later()
        self._send_all(_Cmd.GETATTR, [name] * self.num_workers, note=note)
        return self._gather_envs(note)

    def set_attr(self, name: str, values: list[Any] | Any) -> None:
        if not isinstance(values, list):
            values = [values] * self.num_envs
        if len(values) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} values, got {len(values)}.")
        note = self._restarted_later()
        payloads = [(name, group) for group in self._per_worker(values)]
        self._send_all(_Cmd.SETATTR, payloads, note=note)
        self._gather(note)

    def close(self) -> None:
        """Close every sub-env, waiting up to close_timeout seconds before terminating workers.

        Workers close all their sub-envs even if one fails; the first failure is re-raised
        as VecEnvWorkerError once every worker has stopped. Replacement workers are
        closed like the others; a lost worker that was not replaced is skipped.
        """
        if self.closed:
            return
        self.closed = True
        close_msg = _dumps((_Cmd.CLOSE, None))
        # Every pipe, not range(num_workers): a failed __init__ may not have started them all.
        close_sent = [False] * len(self._pipes)
        for worker in range(len(self._pipes)):
            with contextlib.suppress(_WorkerLost):
                self._send(worker, close_msg)
                close_sent[worker] = True
        deadline = time.monotonic() + self.close_timeout
        error: VecEnvWorkerError | None = None
        for worker, pipe in enumerate(self._pipes):
            # Read (and discard) any outstanding replies, ending with the close reply.
            while self._pending[worker] > 0:
                remaining = max(deadline - time.monotonic(), 0.0)
                try:
                    if not pipe.poll(remaining):
                        break
                    ok, payload = pipe.recv()
                except (EOFError, OSError):
                    break
                self._pending[worker] -= 1
                if close_sent[worker] and self._pending[worker] == 0 and not ok:
                    error = error or VecEnvWorkerError(*payload)
        for proc in self._processes:
            proc.join(timeout=max(deadline - time.monotonic(), 0.0))
        self._terminate()
        if error is not None:
            raise error

    def _terminate(self) -> None:
        self.closed = True
        for proc in self._processes:
            if proc.is_alive():
                proc.terminate()
        # One grace period for all workers (not one each), then kill the survivors:
        # e.g. workers whose env code ignores or blocks SIGTERM.
        deadline = time.monotonic() + _TERMINATE_GRACE
        for proc in self._processes:
            proc.join(timeout=max(deadline - time.monotonic(), 0.0))
        survivors = [proc for proc in self._processes if proc.is_alive()]
        for proc in survivors:
            proc.kill()
        deadline = time.monotonic() + _TERMINATE_GRACE
        for proc in survivors:
            proc.join(timeout=max(deadline - time.monotonic(), 0.0))
        for pipe in self._pipes:
            pipe.close()
        if self._shared is not None:
            self._shared.close()
            self._shared = None

    def __del__(self) -> None:
        # __init__ may have failed before any worker was started.
        if "_processes" in self.__dict__ and not self.closed:
            with contextlib.suppress(Exception):  # e.g. at interpreter shutdown
                self._terminate()
