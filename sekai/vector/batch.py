"""Batched environments on an array backend, and adapters to the VecEnv and Env APIs.

A `BatchEnv` simulates `num_envs` copies of an environment ("lanes") at once, like a
`NativeVecEnv`, but keeps everything on its `sekai.backend`: observations, rewards,
flags and infos are NumPy arrays or torch tensors with a leading `num_envs` axis, on
the env's device. With torch, `differentiable = True` lets gradients flow from the
rewards back to the actions through the whole rollout, which pathwise (direct
policy optimisation) trainers need, and `detach_state()` truncates that graph.

Two adapters connect a `BatchEnv` to the rest of sekai and its ecosystem:

- `BatchVecEnv(env)` is a `NativeVecEnv` (NumPy at the boundary): what `make_vec()`
  returns, and what Stable-Baselines3 and other `VecEnv` users see.
- `UnbatchedEnv(env)` is a single `Env` of a one-lane `BatchEnv`: what `make()`
  returns.

Example:
    ```python
    env = BatchMarketEnv(256, scenarios=gan, backend="torch", device="cuda", autoreset=False)
    env.differentiable = True
    result = env.reset(seed=0)
    total = 0.0
    while not bool(result.dones.all()):
        result = env.step(policy(result.observations))
        total = total + result.rewards.mean()
    (-total).backward()  # the pathwise gradient of the episode's reward
    ```
"""

from __future__ import annotations

import abc
import contextlib
import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Generic

import numpy as np
from numpy.typing import NDArray

from sekai._types import ActType, InfoDict, ObsType, SeedType
from sekai.backend import Array, Backend, backend_of, get_backend, to_numpy
from sekai.core._rng import SeededRNG
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.error import ResetRequired
from sekai.spaces.space import Space
from sekai.vector.native import InfoArrays, NativeVecEnv, split_infos, take_index

if TYPE_CHECKING:
    from sekai.core.snapshot import Snapshot

_SHARED = "__sekai_shared__"


def map_tree(fn: Any, tree: Any) -> Any:
    """`fn` applied to every leaf of a nested dict / tuple / list of arrays."""
    if isinstance(tree, Mapping):
        return {k: map_tree(fn, v) for k, v in tree.items()}
    if isinstance(tree, tuple):
        return tuple(map_tree(fn, v) for v in tree)
    if isinstance(tree, list):
        return [map_tree(fn, v) for v in tree]
    return fn(tree)


def tree_to_numpy(tree: Any) -> Any:
    """Every array of a nested observation or info as a NumPy array."""
    return map_tree(lambda x: to_numpy(x) if hasattr(x, "shape") else x, tree)


@dataclass(frozen=True)
class BatchReset(Generic[ObsType]):
    """What `BatchEnv.reset()` returns.

    Attributes:
        observations: The observation of every lane, `(num_envs, ...)` arrays.
        info: Info of the lanes that were reset, arrays with a leading axis of their
            number (`num_envs` for a full reset).
        indices: The lanes that were reset, in the order of `info`'s rows.
        dones: `(num_envs,)` bool, all False (so a loop can test `result.dones` from
            the start, as with `BatchStep`).
    """

    observations: ObsType
    info: dict[str, Any]
    indices: Array
    dones: Array


@dataclass(frozen=True)
class BatchStep(Generic[ObsType]):
    """What `BatchEnv.step()` returns; every array has a leading `num_envs` axis.

    Attributes:
        observations: The observation of every lane after the step (with autoreset,
            the first observation of the new episode for the lanes that finished).
        rewards: `(num_envs,)` rewards (0 for frozen lanes).
        terminated: `(num_envs,)` bool.
        truncated: `(num_envs,)` bool.
        info: Info arrays of the step, `(num_envs, ...)` each; for lanes that finished,
            the info of their last step.
        final_observations: With autoreset, when some lane finished: the observation
            of every lane before the resets (only the finished lanes' rows are
            meaningful); else None.
        reset_info: With autoreset, the reset info of the lanes in `reset_indices`
            (arrays with a leading axis of their number); else None.
        reset_indices: The lanes reset by autoreset in this step, or None.
    """

    observations: ObsType
    rewards: Array
    terminated: Array
    truncated: Array
    info: dict[str, Any]
    final_observations: ObsType | None = None
    reset_info: dict[str, Any] | None = None
    reset_indices: Array | None = None

    @property
    def dones(self) -> Array:
        return self.terminated | self.truncated


class BatchEnv(SeededRNG, abc.ABC, Generic[ObsType, ActType]):
    """Base class for environments that simulate `num_envs` lanes at once on one backend.

    Subclasses keep their state in arrays with a leading `num_envs` axis and
    implement:

    - `_reset_lanes(indices, options)`: (re)start the lanes in `indices` (a backend
      int array), with one options dict (or None) per lane; return their reset info
      (arrays with a leading axis of `len(indices)`).
    - `_observe()`: the observation of every lane.
    - `_step(actions)`: advance every lane; return `(rewards, terminated, truncated,
      info)`, `(num_envs,)` arrays and a dict of `(num_envs, ...)` arrays.
    - optionally `_detach()`: cut the state from the autograd graph (torch).

    The base class provides seeding (`self.rng`, one NumPy generator shared by every
    lane, for any randomness drawn on the host), per-lane options, same-step autoreset,
    `ResetRequired` before the first reset, and frozen lanes: without autoreset, a
    lane that finishes is frozen until it is reset, and every later step returns, for
    it, reward 0 and its final terminated / truncated flags. `_step()` still runs for
    every lane (actions are a batch); it must leave frozen lanes' state as it is
    (`self.frozen` is the mask), so their observation stays their final one.

    With the torch backend, `step()` runs under `torch.enable_grad()` only when
    `differentiable` is True (and grad mode is on), else without building a graph.

    Args:
        num_envs: Number of lanes (>= 1).
        observation_space: The observation space of one lane.
        action_space: The action space of one lane.
        backend: "numpy" or "torch" (or a `sekai.backend.Backend`).
        device: torch device (None: the CPU).
        autoreset: Reset finished lanes within `step()`.
    """

    metadata: ClassVar[dict[str, Any]] = {}
    spec: Any = None
    device: Any = None  # the torch device (None for NumPy)

    def __init__(
        self,
        num_envs: int,
        observation_space: Space[ObsType],
        action_space: Space[ActType],
        *,
        backend: str | Backend = "numpy",
        device: Any = None,
        autoreset: bool = True,
    ) -> None:
        if int(num_envs) < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}.")
        self.num_envs = int(num_envs)
        self.observation_space = observation_space
        self.action_space = action_space
        self.backend = get_backend(backend)
        if self.backend.name == "numpy" and device is not None:
            raise ValueError("device= is for the torch backend; the NumPy backend runs on the CPU.")
        if self.backend.name == "torch":
            import torch

            self.device = torch.device("cpu" if device is None else device)
        else:
            self.device = None
        self.autoreset = autoreset
        self.differentiable = False
        self._needs_reset = True
        self._like = self._anchor()
        self._frozen = self.backend.bools((self.num_envs,), False, like=self._like)
        # `_frozen` on the host, and the lanes being reset (a backend array and its host
        # values), so that subclasses need not read either back from a device.
        self._frozen_host = np.zeros(self.num_envs, dtype=bool)
        self._resetting: tuple[Array, NDArray[np.int64]] | None = None
        # `_step` may set this to the host values of `(terminated | truncated) & ~frozen`
        # when it read them back anyway (with its own checks), so `step()` doesn't again.
        self._finished_host: NDArray[np.bool_] | None = None
        self._final_terminated = self._frozen
        self._final_truncated = self._frozen
        self.closed = False

    def _anchor(self) -> Array:
        """A scalar float array on the env's device, the `like` of new arrays."""
        if self.backend.name == "torch":
            import torch

            return torch.zeros((), dtype=torch.get_default_dtype(), device=self.device)
        return np.zeros(())

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _reset_lanes(
        self, indices: Array, options: Sequence[dict[str, Any] | None]
    ) -> dict[str, Any]: ...

    @abc.abstractmethod
    def _observe(self) -> ObsType: ...

    @abc.abstractmethod
    def _step(self, actions: Any) -> tuple[Array, Array, Array, dict[str, Any]]: ...

    def _detach(self) -> None:
        """Cut the lanes' state from the autograd graph; override for stateful envs."""

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    @property
    def frozen(self) -> Array:
        """`(num_envs,)` bool: lanes that finished without autoreset and await a reset."""
        return self._frozen

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._fill_derived(state)

    def _fill_derived(self, state: Mapping[str, Any]) -> None:
        """Rebuild the copies of the state that `state` lacks (as one pickled or
        snapshotted by sekai 0.3.0 does), from the state they copy."""
        if "_frozen_host" not in state:
            self._frozen_host = np.array(self.backend.to_numpy(self._frozen), dtype=bool)
        self.__dict__.setdefault("_resetting", None)
        self.__dict__.setdefault("_finished_host", None)

    def _lanes_on_host(self, indices: Array) -> NDArray[np.int64]:
        """The values of `indices` (as `_reset_lanes` receives them) as a NumPy array,
        without reading them back from the device when the base class made them."""
        if self._resetting is not None and indices is self._resetting[0]:
            return self._resetting[1]
        return np.asarray(self.backend.to_numpy(indices), dtype=np.int64).reshape(-1)

    def _reset_some(
        self, idx: Array, idx_host: NDArray[np.int64], options: Sequence[dict[str, Any] | None]
    ) -> dict[str, Any]:
        """`_reset_lanes(idx, options)` without autograd, with `idx`'s host values known."""
        self._resetting = (idx, idx_host)
        try:
            with contextlib.ExitStack() as stack:
                if self.backend.name == "torch":
                    import torch

                    stack.enter_context(torch.no_grad())
                return self._reset_lanes(idx, options)
        finally:
            self._resetting = None

    def _grad(self) -> contextlib.AbstractContextManager[Any]:
        if self.backend.name != "torch":
            return contextlib.nullcontext()
        import torch

        grad: contextlib.AbstractContextManager[Any] = torch.set_grad_enabled(
            self.differentiable and torch.is_grad_enabled()
        )
        return grad

    def _lane_options(
        self, options: list[dict[str, Any] | None] | dict[str, Any] | None, n: int
    ) -> list[dict[str, Any] | None]:
        if options is None or isinstance(options, dict):
            return [options] * n
        opts = list(options)
        if len(opts) != n:
            raise ValueError(f"Expected {n} options (one per lane reset), got {len(opts)}.")
        return opts

    def reset(
        self,
        *,
        seed: SeedType = None,
        options: list[dict[str, Any] | None] | dict[str, Any] | None = None,
        indices: Any = None,
    ) -> BatchReset[ObsType]:
        """Reset every lane, or the lanes in `indices`.

        Args:
            seed: An int seeds `self.rng`; None continues its stream.
            options: One dict for every lane reset, or a list with one per lane reset.
            indices: Lanes to reset (default: all). The others are left as they are.
        """
        if self.closed:
            from sekai.error import VecEnvClosed

            raise VecEnvClosed()
        if isinstance(seed, (list, tuple, np.random.Generator)):
            raise TypeError(
                "BatchEnv.reset() takes an int seed (or None), not a list or Generator."
            )
        B = self.backend
        if indices is None:
            idx_host = np.arange(self.num_envs, dtype=np.int64)
            idx = B.arange(self.num_envs, like=self._like)
        else:
            idx_host = np.asarray(np.asarray(to_numpy(indices)).reshape(-1), dtype=np.int64)
            idx = B.asint(idx_host, self._like)
        if indices is None:
            self._needs_reset = False
        elif self._needs_reset:
            raise ResetRequired("Reset every lane (indices=None) before resetting some of them.")
        opts = self._lane_options(options, int(idx.shape[0]))
        self._set_rng(seed)
        info = self._reset_some(idx, idx_host, opts)
        lanes = B.bools((self.num_envs,), False, like=self._like)
        lanes = B.put_along_axis(
            lanes, idx, B.bools(tuple(idx.shape), True, like=self._like), axis=0
        )
        self._frozen = self._frozen & ~lanes
        frozen_host = self._frozen_host.copy()
        frozen_host[idx_host] = False
        self._frozen_host = frozen_host
        dones = B.bools((self.num_envs,), False, like=self._like)
        return BatchReset(observations=self._observe(), info=info, indices=idx, dones=dones)

    def step(self, actions: Any) -> BatchStep[ObsType]:
        """Advance every lane with a batch of actions (`(num_envs, ...)`)."""
        if self._needs_reset:
            raise ResetRequired()
        B = self.backend
        frozen = self._frozen
        self._finished_host = None
        with self._grad():
            rewards, terminated, truncated, info = self._step(actions)
        rewards = B.where(frozen, 0.0, B.asfloat(rewards, self._like))
        terminated = B.where(frozen, self._final_terminated, B.asbool(terminated, self._like))
        truncated = B.where(frozen, self._final_truncated, B.asbool(truncated, self._like))
        finished = (terminated | truncated) & ~frozen
        final_obs = reset_info = reset_idx = None
        finished_host = self._finished_host
        self._finished_host = None
        if finished_host is None:
            finished_host = B.to_numpy(finished)  # the step's one read back from a device
        if finished_host.any():
            if self.autoreset:
                final_obs = self._observe()
                reset_host = np.flatnonzero(finished_host).astype(np.int64)
                reset_idx = B.asint(reset_host, self._like)
                reset_info = self._reset_some(reset_idx, reset_host, [None] * len(reset_host))
            else:
                self._final_terminated = B.where(finished, terminated, self._final_terminated)
                self._final_truncated = B.where(finished, truncated, self._final_truncated)
                self._frozen = frozen | finished
                self._frozen_host = self._frozen_host | finished_host
        return BatchStep(
            observations=self._observe(),
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            info=info,
            final_observations=final_obs,
            reset_info=reset_info,
            reset_indices=reset_idx,
        )

    def observe(self) -> ObsType:
        """The current observation of every lane."""
        if self._needs_reset:
            raise ResetRequired()
        return self._observe()

    def detach_state(self) -> None:
        """Cut the state carried between steps from the autograd graph (torch).

        A trainer that backpropagates through rollouts calls it every `k` steps to
        truncate backpropagation through time; observations already returned still
        hold the graph, so detach those too.
        """
        self._detach()

    def snapshot(self) -> Snapshot:
        """A copy of the batched environment's state (random generators included).

        `restore(snapshot)` puts it back, so the same actions then give the same results:
        for planning from a state, "what if" analysis, or checkpoints (a snapshot pickles).
        Data that never changes (spaces, `MarketData`, market models, read-only arrays) is
        shared with the snapshot, not copied. See `sekai.core.snapshot`.
        """
        from sekai.core.snapshot import take_snapshot

        return take_snapshot(self)

    def restore(self, snapshot: Snapshot) -> None:
        """Return to the state of `snapshot` (in place; a snapshot can be restored many times).

        Raises:
            TypeError: If the snapshot was taken of another kind of environment.
        """
        from sekai.core.snapshot import restore_snapshot

        restore_snapshot(self, snapshot)

    def close(self) -> None:
        self.closed = True

    def __repr__(self) -> str:
        return f"<{type(self).__name__}(num_envs={self.num_envs}, backend={self.backend.name!r})>"


def _is_shared(obj: Any, name: str) -> bool:
    member = getattr(type(obj), name, None)
    if isinstance(member, property):
        member = member.fget
    return getattr(member, _SHARED, False) is True


class BatchVecEnv(NativeVecEnv[Any, Any]):
    """A `BatchEnv` as a sekai `VecEnv`, with NumPy observations, rewards and infos.

    Seeding, autoreset (with `final_observation` / `final_info` / `reset_info` in the
    per-env infos), frozen sub-envs, the time limit and `get_attr` / `set_attr` /
    `call` behave as for every `NativeVecEnv`; attributes and methods not defined here
    are those of `batch_env` (arrays converted to NumPy and split per sub-env, except
    members marked `@shared`).

    Args:
        batch_env: The batched env (its own autoreset is turned off; this adapter resets
            finished sub-envs).
        autoreset: Reset finished sub-envs in the same step.
        max_episode_steps: Truncate every sub-env after this many steps.
    """

    def __init__(
        self,
        batch_env: BatchEnv[Any, Any],
        *,
        autoreset: bool = True,
        max_episode_steps: int | None = None,
    ) -> None:
        batch_env.autoreset = False
        self.batch_env = batch_env
        super().__init__(
            batch_env.num_envs,
            batch_env.observation_space,
            batch_env.action_space,
            autoreset=autoreset,
            max_episode_steps=max_episode_steps,
        )
        self._last_reset_info: InfoArrays = {}

    def _reset_envs(
        self, indices: NDArray[np.intp], options: Sequence[dict[str, Any] | None]
    ) -> None:
        env = self.batch_env
        env.rng = self.rng
        every = len(indices) == self.num_envs
        result = env.reset(indices=None if every else indices, options=list(options))
        self._last_reset_info = tree_to_numpy(result.info)

    def _reset_infos(self, indices: NDArray[np.intp]) -> InfoArrays:
        return self._last_reset_info

    def _observe(self) -> Any:
        return tree_to_numpy(self.batch_env.observe())

    def _step(self, actions: Any) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], InfoArrays]:
        env = self.batch_env
        env.rng = self.rng
        result = env.step(actions)
        return (
            to_numpy(result.rewards),
            to_numpy(result.terminated),
            to_numpy(result.truncated),
            tree_to_numpy(result.info),
        )

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name == "batch_env":
            raise AttributeError(name)
        value = getattr(self.batch_env, name)
        if inspect.ismethod(value):  # a method: its results are converted too
            return lambda *args, **kwargs: tree_to_numpy(value(*args, **kwargs))
        return tree_to_numpy(value)

    def _is_shared(self, name: str) -> bool:
        return super()._is_shared(name) or _is_shared(self.batch_env, name)

    def set_attr(self, name: str, values: list[Any] | Any) -> None:
        if (
            name in type(self).__dict__
            or name in self.__dict__
            or not hasattr(self.batch_env, name)
        ):
            super().set_attr(name, values)
            return
        if isinstance(values, list):
            if len(values) != self.num_envs:
                raise ValueError(f"Expected {self.num_envs} values, got {len(values)}.")
            values = np.asarray(values)
        setattr(self.batch_env, name, values)

    def close(self) -> None:
        self.batch_env.close()
        super().close()


class UnbatchedEnv(Env[Any, Any]):
    """A one-lane `BatchEnv` as a single sekai `Env` (NumPy observations and infos).

    `reset()` / `step()` behave as for the finance environments: stepping after the
    episode has ended raises `ResetRequired`. Attributes and methods not defined here
    are those of `batch_env`, with the lane axis removed from arrays (members marked
    `@shared` are returned as they are), so e.g. `env.action_mask()` and renderers
    work unchanged.

    Args:
        batch_env: A `BatchEnv` with `num_envs == 1` (its autoreset is turned off).
    """

    def __init__(self, batch_env: BatchEnv[Any, Any]) -> None:
        if batch_env.num_envs != 1:
            raise ValueError(f"UnbatchedEnv needs a one-lane BatchEnv, got {batch_env.num_envs}.")
        batch_env.autoreset = False
        self.batch_env = batch_env
        self.observation_space = batch_env.observation_space
        self.action_space = batch_env.action_space
        # Per instance (Env.metadata is a class attribute): renderers read the batch env's.
        self.__dict__["metadata"] = dict(type(batch_env).metadata)
        self._started = False
        self._over = False
        self._built = True  # from here on, the batch env's attributes are set on it

    def reset(
        self, *, seed: SeedType = None, options: dict[str, Any] | None = None
    ) -> ResetResult[Any]:
        self._set_rng(seed)
        self.batch_env.rng = self.rng
        result = self.batch_env.reset(options=options)
        self._started, self._over = True, False
        return ResetResult(
            observation=self._lane(result.observations), info=self._lane_info(result.info)
        )

    def step(self, action: Any) -> StepResult[Any]:
        if not self._started:
            raise ResetRequired()
        if self._over:
            raise ResetRequired("The episode has ended; call reset() before step().")
        result = self.batch_env.step(np.asarray(action)[None])
        terminated = bool(to_numpy(result.terminated)[0])
        truncated = bool(to_numpy(result.truncated)[0])
        self._over = terminated or truncated
        return StepResult(
            self._lane(result.observations),
            float(to_numpy(result.rewards)[0]),
            terminated,
            truncated,
            self._lane_info(result.info),
        )

    def _lane(self, observations: Any) -> Any:
        return take_index(self.observation_space, tree_to_numpy(observations), 0)

    @staticmethod
    def _lane_info(info: dict[str, Any]) -> InfoDict:
        return split_infos(tree_to_numpy(info), 1)[0]

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_") or name == "batch_env":
            raise AttributeError(name)
        value = getattr(self.batch_env, name)
        shared = _is_shared(self.batch_env, name)

        def lane(x: Any) -> Any:
            if shared:
                return x
            if isinstance(x, (tuple, list)) and len(x) == 1:
                return x[0]
            arr = to_numpy(x) if hasattr(x, "shape") else x
            if isinstance(arr, np.ndarray) and arr.shape[:1] == (1,):
                return arr[0]
            return arr

        if inspect.ismethod(value):  # a method: its results are converted too
            return lambda *args, **kwargs: lane(value(*args, **kwargs))
        return lane(value)

    def __setattr__(self, name: str, value: Any) -> None:
        # The batch env's attributes (reward_fn, cost_model, ...) are set on it, as on a
        # single environment; per-lane ones take `value` as the lane's.
        batch = self.__dict__.get("batch_env")
        if (
            not self.__dict__.get("_built")
            or name.startswith("_")
            or name in self.__dict__
            or hasattr(type(self), name)
            or not hasattr(batch, name)
        ):
            object.__setattr__(self, name, value)
            return
        current = getattr(batch, name)
        shape = getattr(current, "shape", None)
        if not _is_shared(batch, name) and shape is not None and tuple(shape)[:1] == (1,):
            value = backend_of(current).cast(value, current)[None]
        setattr(batch, name, value)

    def close(self) -> None:
        self.batch_env.close()

    def __repr__(self) -> str:
        return f"<UnbatchedEnv({type(self.batch_env).__name__})>"


__all__ = [
    "BatchEnv",
    "BatchReset",
    "BatchStep",
    "BatchVecEnv",
    "UnbatchedEnv",
    "map_tree",
    "tree_to_numpy",
]
