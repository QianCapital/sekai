from __future__ import annotations

import contextlib
import dataclasses
import functools
import multiprocessing as mp
import signal
import sys
import time
import traceback
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
    env_call,
    env_get_attr,
    env_set_attr,
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
        arrays = iter([view.copy() for view in self._views])

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


# Seconds a terminated (then killed) worker gets to exit before close() moves on.
_TERMINATE_GRACE = 1.0


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
        pipe.close()
        return
    pipe.send((True, None))
    shared: _SharedObservations | None = None

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
                for index, env, action in zip(env_indices, envs, data):
                    current = index
                    result = env.step(action)
                    if autoreset and needs_autoreset(result):
                        result = merge_autoreset(result, env.reset())
                    if shared is not None:
                        shared.write(index, result.observation)
                        result = dataclasses.replace(result, observation=None)
                    payload.append(result)
            elif cmd is _Cmd.RESET:
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
    the next command.

    Args:
        env_fns: Callables that each build one sub-environment (in its worker).
        autoreset: Reset sub-envs automatically when their episode ends (see VecEnv).
        context: multiprocessing start method; defaults to "forkserver" ("spawn" on Windows).
        close_timeout: Seconds close() waits for workers before terminating them.
        num_workers: Worker processes; sub-envs are split evenly between them.
            Defaults to one process per sub-env.
        shared_memory: Pass observations through shared memory. None (default)
            enables it when the observation space allows; True requires it.
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], Env[ObsType, ActType]]],
        *,
        autoreset: bool = True,
        context: str | None = None,
        close_timeout: float = 5.0,
        num_workers: int | None = None,
        shared_memory: bool | None = None,
    ) -> None:
        if len(env_fns) == 0:
            raise ValueError("ProcVecEnv needs at least one env_fn.")
        if num_workers is not None and num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}.")
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

        for worker, group in enumerate(self._groups):
            parent_pipe, child_pipe = ctx.Pipe()
            fns = [env_fns[i] for i in group]
            proc = ctx.Process(
                target=_worker,
                args=(
                    group,
                    CloudpickleWrapper(functools.partial(_identity, fns)),
                    child_pipe,
                    parent_pipe,
                    autoreset,
                ),
                daemon=True,
                name=f"sekai-ProcVecEnv-worker-{worker}",
            )
            proc.start()
            child_pipe.close()
            self._pipes.append(parent_pipe)
            self._processes.append(proc)
            self._pending.append(1)  # the worker's "env_fns succeeded" handshake

        try:
            self._gather()
            self._send(0, ForkingPickler.dumps((_Cmd.SPACES, None)))
            self.observation_space, self.action_space = self._recv(0)
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
            self._terminate()
            raise

    @property
    def shared_memory(self) -> bool:
        """Whether observations travel through shared memory."""
        return self._shared is not None

    # ------------------------------------------------------------------
    # Pipe protocol
    # ------------------------------------------------------------------

    def _recv(self, worker: int) -> Any:
        try:
            ok, payload = self._pipes[worker].recv()
        except (EOFError, OSError) as e:
            self._pending[worker] = 0
            raise VecEnvWorkerError(
                self._groups[worker][0], type(e).__name__, "worker process exited unexpectedly", ""
            ) from e
        self._pending[worker] -= 1
        if not ok:
            raise VecEnvWorkerError(*payload)
        return payload

    def _send(self, worker: int, message: Any) -> None:
        """Send one pre-pickled command; count the reply it will produce."""
        try:
            self._pipes[worker].send_bytes(message)
        except OSError as e:
            raise VecEnvWorkerError(
                self._groups[worker][0], type(e).__name__, "worker process is not running", ""
            ) from e
        self._pending[worker] += 1

    def _drain(self) -> None:
        """Discard replies left unread by an interrupted or failed earlier call."""
        for worker, pipe in enumerate(self._pipes):
            while self._pending[worker] > 0:
                try:
                    pipe.recv()
                except (EOFError, OSError):
                    self._pending[worker] = 0
                    break
                self._pending[worker] -= 1

    def _gather(self) -> list[Any]:
        """Receive one reply from every worker, then raise the first error, if any."""
        results: list[Any] = []
        first_error: VecEnvWorkerError | None = None
        for worker in range(self.num_workers):
            try:
                results.append(self._recv(worker))
            except VecEnvWorkerError as e:
                results.append(None)
                first_error = first_error or e
        if first_error is not None:
            raise first_error
        return results

    def _gather_envs(self) -> list[Any]:
        """Gather per-worker reply lists and flatten them into per-env order."""
        return [item for reply in self._gather() for item in reply]

    def _send_all(self, cmd: _Cmd, payloads: Sequence[Any]) -> None:
        """Send one payload to each worker."""
        if self.closed:
            raise VecEnvClosed()
        self._drain()
        # Pickle everything first: if any payload is unpicklable, nothing is sent.
        messages = [ForkingPickler.dumps((cmd, payload)) for payload in payloads]
        for worker, message in enumerate(messages):
            self._send(worker, message)

    def _per_worker(self, per_env: Sequence[Any]) -> list[list[Any]]:
        return [[per_env[i] for i in group] for group in self._groups]

    # ------------------------------------------------------------------
    # VecEnv API
    # ------------------------------------------------------------------

    def step(self, actions: Any) -> VecStepResult[Any]:
        self._send_all(_Cmd.STEP, self._per_worker(self._split_actions(actions)))
        results = self._gather_envs()
        observations = self._shared.read() if self._shared is not None else None
        return self._batch_step(results, observations)

    def reset(
        self,
        *,
        seed: list[SeedType] | SeedType = None,
        options: list[dict[str, Any] | None] | dict[str, Any] | None = None,
    ) -> VecResetResult[Any]:
        seeds = self._expand_seeds(seed)
        opts = self._expand_options(options)
        self._send_all(_Cmd.RESET, self._per_worker(list(zip(seeds, opts))))
        results = self._gather_envs()
        observations = self._shared.read() if self._shared is not None else None
        return self._batch_reset(results, observations)

    def call(self, method: str, *args: Any, **kwargs: Any) -> list[Any]:
        self._send_all(_Cmd.CALL, [(method, args, kwargs)] * self.num_workers)
        return self._gather_envs()

    def get_attr(self, name: str) -> list[Any]:
        self._send_all(_Cmd.GETATTR, [name] * self.num_workers)
        return self._gather_envs()

    def set_attr(self, name: str, values: list[Any] | Any) -> None:
        if not isinstance(values, list):
            values = [values] * self.num_envs
        if len(values) != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} values, got {len(values)}.")
        self._send_all(_Cmd.SETATTR, [(name, group) for group in self._per_worker(values)])
        self._gather()

    def close(self) -> None:
        """Close every sub-env, waiting up to close_timeout seconds before terminating workers.

        Workers close all their sub-envs even if one fails; the first failure is re-raised
        as VecEnvWorkerError once every worker has stopped.
        """
        if self.closed:
            return
        self.closed = True
        close_msg = ForkingPickler.dumps((_Cmd.CLOSE, None))
        close_sent = [False] * self.num_workers
        for worker in range(self.num_workers):
            with contextlib.suppress(VecEnvWorkerError):
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
