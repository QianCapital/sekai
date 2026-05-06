from __future__ import annotations

import multiprocessing as mp
import sys
from enum import Enum, auto
from typing import Any, Callable, Sequence

import numpy as np

from sekai._types import ActType, ObsType, SeedType
from sekai.core.env import Env
from sekai.utils.pickle import CloudpickleWrapper
from sekai.utils.seeding import make_seed_sequence
from sekai.vector.result import VecResetResult, VecStepResult
from sekai.vector.vec_env import VecEnv


class _Cmd(Enum):
    STEP = auto()
    RESET = auto()
    CLOSE = auto()
    CALL = auto()
    GETATTR = auto()
    SETATTR = auto()


def _worker(
    rank: int,
    env_fn: CloudpickleWrapper,
    pipe: "mp.connection.Connection",
    parent_pipe: "mp.connection.Connection",
    error_queue: "mp.Queue[tuple[int, BaseException]]",
) -> None:
    parent_pipe.close()
    env = env_fn()
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd is _Cmd.STEP:
                result = env.step(data)
                pipe.send(result)
            elif cmd is _Cmd.RESET:
                seed, options = data
                result = env.reset(seed=seed, options=options)
                pipe.send(result)
            elif cmd is _Cmd.CLOSE:
                env.close()
                pipe.send(None)
                break
            elif cmd is _Cmd.CALL:
                method, args, kwargs = data
                pipe.send(getattr(env, method)(*args, **kwargs))
            elif cmd is _Cmd.GETATTR:
                pipe.send(getattr(env, data))
            elif cmd is _Cmd.SETATTR:
                name, value = data
                setattr(env, name, value)
                pipe.send(None)
            else:
                raise RuntimeError(f"Unknown command: {cmd}")
    except Exception as e:
        error_queue.put((rank, e))
        pipe.send(None)
        env.close()


class ProcVecEnv(VecEnv[ObsType, ActType]):
    """Vectorised env using multiprocessing for true CPU-bound parallelism.

    Each sub-environment runs in its own process and communicates via
    multiprocessing Pipes. Use this for CPU-intensive simulators.

    For I/O-bound or async environments, use AsyncVecEnv instead.
    """

    def __init__(
        self,
        env_fns: Sequence[Callable[[], Env[ObsType, ActType]]],
        *,
        context: str = "forkserver" if sys.platform != "win32" else "spawn",
    ) -> None:
        ctx = mp.get_context(context)
        self.num_envs = len(env_fns)
        self._error_queue: mp.Queue[tuple[int, BaseException]] = ctx.Queue()
        self._pipes: list[mp.connection.Connection] = []
        self._processes: list[mp.Process] = []

        for rank, fn in enumerate(env_fns):
            parent_pipe, child_pipe = ctx.Pipe()
            proc = ctx.Process(
                target=_worker,
                args=(rank, CloudpickleWrapper(fn), child_pipe, parent_pipe, self._error_queue),
                daemon=True,
            )
            proc.start()
            child_pipe.close()
            self._pipes.append(parent_pipe)
            self._processes.append(proc)

        dummy_env = env_fns[0]()
        self.observation_space = dummy_env.observation_space
        self.action_space = dummy_env.action_space
        dummy_env.close()

    def _check_errors(self) -> None:
        if not self._error_queue.empty():
            rank, exc = self._error_queue.get()
            raise RuntimeError(f"Error in worker {rank}") from exc

    def step(self, actions: Any) -> VecStepResult[ObsType]:
        for pipe, action in zip(self._pipes, actions):
            pipe.send((_Cmd.STEP, action))
        results = [pipe.recv() for pipe in self._pipes]
        self._check_errors()
        return VecStepResult(
            observations=self._stack_obs([r.observation for r in results]),
            rewards=np.array([r.reward for r in results], dtype=np.float64),
            terminated=np.array([r.terminated for r in results], dtype=np.bool_),
            truncated=np.array([r.truncated for r in results], dtype=np.bool_),
            infos=[r.info for r in results],
        )

    def reset(
        self,
        *,
        seed: list[SeedType] | SeedType = None,
        options: list[dict[str, Any]] | dict[str, Any] | None = None,
    ) -> VecResetResult[ObsType]:
        seeds = self._expand_seeds(seed)
        opts = options if isinstance(options, list) else [options] * self.num_envs
        for pipe, s, o in zip(self._pipes, seeds, opts):
            pipe.send((_Cmd.RESET, (s, o)))
        results = [pipe.recv() for pipe in self._pipes]
        self._check_errors()
        return VecResetResult(
            observations=self._stack_obs([r.observation for r in results]),
            infos=[r.info for r in results],
        )

    def close(self) -> None:
        for pipe in self._pipes:
            try:
                pipe.send((_Cmd.CLOSE, None))
            except Exception:
                pass
        for pipe in self._pipes:
            try:
                pipe.recv()
            except Exception:
                pass
        for proc in self._processes:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
        for pipe in self._pipes:
            pipe.close()

    def call(self, method: str, *args: Any, **kwargs: Any) -> list[Any]:
        for pipe in self._pipes:
            pipe.send((_Cmd.CALL, (method, args, kwargs)))
        return [pipe.recv() for pipe in self._pipes]

    def get_attr(self, name: str) -> list[Any]:
        for pipe in self._pipes:
            pipe.send((_Cmd.GETATTR, name))
        return [pipe.recv() for pipe in self._pipes]

    def set_attr(self, name: str, values: list[Any] | Any) -> None:
        if not isinstance(values, list):
            values = [values] * self.num_envs
        for pipe, v in zip(self._pipes, values):
            pipe.send((_Cmd.SETATTR, (name, v)))
        for pipe in self._pipes:
            pipe.recv()

    def _stack_obs(self, obs_list: list[ObsType]) -> Any:
        first = obs_list[0]
        if isinstance(first, np.ndarray):
            return np.stack(obs_list)
        if isinstance(first, dict):
            return {key: np.stack([o[key] for o in obs_list]) for key in first}  # type: ignore[index]
        return obs_list

    def _expand_seeds(self, seed: list[SeedType] | SeedType) -> list[SeedType]:
        if isinstance(seed, list):
            return seed
        if seed is None:
            return [None] * self.num_envs
        return make_seed_sequence(int(seed), self.num_envs)  # type: ignore[return-value]

    def __del__(self) -> None:
        for proc in self._processes:
            if proc.is_alive():
                proc.terminate()
