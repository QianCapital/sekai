from __future__ import annotations

import abc
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from sekai._types import ActType, InfoDict, ObsType, SeedType
from sekai.error import ResetRequired
from sekai.spaces.dict import Dict
from sekai.spaces.space import Space
from sekai.spaces.tuple import Tuple
from sekai.vector.result import VecResetResult, VecStepResult
from sekai.vector.vec_env import GENERATOR_SEED_ERROR, VecEnv

InfoArrays = Mapping[str, NDArray[Any]]

_F = TypeVar("_F", bound=Callable[..., Any])
_SHARED = "__sekai_shared__"


def shared(member: _F) -> _F:
    """Mark a method (or property getter) of a NativeVecEnv as shared by every sub-env.

    `get_attr()` / `call()` then give its value to every sub-env as it is, instead of
    splitting an array with a leading `num_envs` axis, which would mistake e.g. a
    schedule of `n_steps` rows for per-env values whenever `n_steps == num_envs`.

    Example:
        ```python
        class MyVecEnv(NativeVecEnv):
            @shared
            def schedule(self) -> np.ndarray:  # (n_steps, 1), the same for every sub-env
                ...

            @property
            @shared
            def tick_sizes(self) -> np.ndarray: ...
        ```
    """
    setattr(member, _SHARED, True)
    return member


def take_index(space: Space[Any], batch: Any, index: int) -> Any:
    """One sub-env's sample (a copy) from a batched sample of space."""
    if isinstance(space, Dict):
        return {key: take_index(sub, batch[key], index) for key, sub in space.items()}
    if isinstance(space, Tuple):
        return tuple(take_index(sub, b, index) for sub, b in zip(space.spaces, batch))
    item = batch[index]
    return item.copy() if isinstance(item, np.ndarray) else item


def _put_rows(space: Space[Any], dst: Any, src: Any, indices: NDArray[np.intp]) -> Any:
    """`dst` (a batched sample of space) with the rows in `indices` copied from `src`.

    Arrays are updated in place; a tuple of per-env samples (e.g. of a Sequence) is rebuilt.
    """
    if isinstance(space, Dict):
        return {key: _put_rows(sub, dst[key], src[key], indices) for key, sub in space.items()}
    if isinstance(space, Tuple):
        return tuple(_put_rows(sub, d, s, indices) for sub, d, s in zip(space.spaces, dst, src))
    if isinstance(dst, np.ndarray):
        dst[indices] = src[indices]
        return dst
    items = list(dst)
    for i in indices.tolist():
        items[i] = src[i]
    return tuple(items)


def _numeric_batch(actions: Any, n: int) -> NDArray[Any] | None:
    """actions as a numeric array with a leading axis of n, or None if they are not one."""
    if isinstance(actions, dict):
        return None
    try:
        batch = np.asarray(actions)
    except (TypeError, ValueError):  # e.g. ragged per-env actions
        return None
    return batch if batch.dtype.kind in "biuf" and batch.shape[:1] == (n,) else None


def split_infos(infos: InfoArrays, n: int) -> list[InfoDict]:
    """Split {key: array of n values} into n per-env info dicts.

    1-D arrays become Python scalars; arrays with more dimensions give each env
    its own row (copied, so later in-place updates of the source cannot leak).
    """
    out: list[InfoDict] = [{} for _ in range(n)]
    for key, value in infos.items():
        arr = np.asarray(value)
        if arr.shape[:1] != (n,):
            raise ValueError(f"info {key!r} has shape {arr.shape}; expected a leading axis of {n}.")
        if arr.ndim == 1:  # scalars per env; datetimes stay numpy scalars, not integers
            rows: list[Any] = list(arr.copy()) if arr.dtype.kind in "Mm" else arr.tolist()
        else:
            rows = list(arr.copy())
        for info, row in zip(out, rows):
            info[key] = row
    return out


class NativeVecEnv(VecEnv[ObsType, ActType]):
    """Base class for environments that simulate every sub-env at once with array operations.

    SyncVecEnv steps N separate Python environments one after another; a native
    vector env keeps the state of all N in arrays with a leading `num_envs` axis
    and advances them together, which is typically 10-100x faster for
    simulators that vectorise naturally (most market models do). Register one
    with `sekai.register(..., vector_entry_point=...)` and `sekai.make_vec()`
    uses it automatically.

    Subclasses implement:

    - `_reset_envs(indices, options)`: (re)initialise the sub-envs in `indices`
      (an int array), with one options dict (or None) per index.
    - `_observe()`: the batched observation of all sub-envs.
    - `_step(actions)`: advance every sub-env; return `(rewards, terminated,
      truncated, infos)` where the first three have shape `(num_envs,)` and
      infos maps keys to arrays with a leading `num_envs` axis.
    - optionally `_reset_infos(indices)`: info arrays for freshly reset sub-envs.

    The base class provides seeding (one `numpy.random.Generator`, `self.rng`,
    shared by all sub-envs), per-env options, same-step autoreset with
    `final_observation` / `final_info` / `reset_info` in the infos (the
    contract of every sekai VecEnv), conversion of info arrays to per-env
    dicts, ResetRequired from a step() before the first reset(), and an
    optional time limit: with `max_episode_steps=n` every sub-env is truncated
    after n steps and its infos carry `elapsed_steps`, exactly as `TimeLimit`
    does for each sub-env of a SyncVecEnv. The `elapsed_steps` property counts
    each sub-env's steps, limit or not.

    With `autoreset=False`, a sub-env that finishes is frozen until the next
    reset(), as in every VecEnv: step() returns its final observation, reward 0,
    its final terminated / truncated flags and an empty info, whatever `_step()`
    computes for it. `_step()` still advances every sub-env, so it is passed a
    frozen sub-env's last action again (when actions are a numeric array), and
    `self.frozen` tells which sub-envs are frozen: a simulation that would read
    past its data, or that should keep a finished sub-env's state (e.g. for
    get_attr), must leave those sub-envs as they are.

    Because all sub-envs share one generator, a native env is not
    draw-for-draw identical to the same env under SyncVecEnv; it is identical
    in distribution, and exactly identical whenever the dynamics are
    deterministic given the options (e.g. historical data with fixed starts).
    `reset(seed=...)` takes what every VecEnv takes: an int seeds the
    generator; a list of `num_envs` seeds is combined into one seed, where a
    None entry contributes a draw from the current stream (so a list of only
    None continues it, like seed=None); a Generator raises TypeError.

    `get_attr(name)` / `call(name)` read attributes and methods of this object:
    an array result with a leading `num_envs` axis is split per env, anything
    else is repeated for every env, and so is everything a method or property
    marked with `@shared` returns. `set_attr(name, values)` changes an existing
    attribute (a list of `num_envs` values is stored as one array) and raises
    AttributeError for a name this object does not have, since a new attribute
    would have no effect on the simulation.
    """

    def __init__(
        self,
        num_envs: int,
        observation_space: Space[ObsType],
        action_space: Space[ActType],
        *,
        autoreset: bool = True,
        max_episode_steps: int | None = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}.")
        if max_episode_steps is not None and max_episode_steps < 1:
            raise ValueError(f"max_episode_steps must be >= 1, got {max_episode_steps}.")
        self.num_envs = int(num_envs)
        self.observation_space = observation_space
        self.action_space = action_space
        self.autoreset = autoreset
        self.max_episode_steps = max_episode_steps
        self._elapsed = np.zeros(self.num_envs, dtype=np.int64)
        self._rng: np.random.Generator | None = None
        self._rng_seed: int | None = None
        # The state arrays mean nothing until the first reset(); step() refuses to run on them.
        self._needs_reset = True
        # Frozen sub-envs (autoreset off; see VecEnv): what step() returns for them.
        self._frozen = np.zeros(self.num_envs, dtype=np.bool_)
        self._final_obs: Any = None
        self._final_terminated = np.zeros(self.num_envs, dtype=np.bool_)
        self._final_truncated = np.zeros(self.num_envs, dtype=np.bool_)
        self._held_actions: NDArray[Any] | None = None

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _reset_envs(
        self, indices: NDArray[np.intp], options: Sequence[dict[str, Any] | None]
    ) -> None: ...

    @abc.abstractmethod
    def _observe(self) -> Any: ...

    @abc.abstractmethod
    def _step(
        self, actions: Any
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], InfoArrays]: ...

    def _reset_infos(self, indices: NDArray[np.intp]) -> InfoArrays:
        return {}

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    @property
    def rng(self) -> np.random.Generator:
        """The generator shared by all sub-envs (created from OS entropy if unseeded)."""
        if self._rng is None:
            from sekai.utils.seeding import np_random

            self._rng, self._rng_seed = np_random()
        return self._rng

    def _seed(self, seed: list[SeedType] | SeedType) -> None:
        """Seed the shared generator; accepts exactly what SyncVecEnv.reset() accepts.

        A list's entries are what a SyncVecEnv would give each sub-env's reset(): an
        int, None (that sub-env continues its stream: here, a draw from the current
        one), or a Generator (here, a draw from it). They are combined into one seed.
        """
        from sekai.utils.seeding import np_random

        if seed is None:
            return
        if isinstance(seed, np.random.Generator):
            raise TypeError(GENERATOR_SEED_ERROR)
        if not isinstance(seed, list):
            self._rng, self._rng_seed = np_random(int(seed))
            return
        seeds = self._expand_seeds(seed)  # checks the length
        if all(s is None for s in seeds):
            return

        def entropy(s: SeedType) -> int:
            if s is None:
                return int(self.rng.integers(2**63))
            if isinstance(s, np.random.Generator):
                return int(s.integers(2**63))
            return int(s)

        self._rng = np.random.default_rng([entropy(s) for s in seeds])
        self._rng_seed = -1

    # ------------------------------------------------------------------
    # VecEnv API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: list[SeedType] | SeedType = None,
        options: list[dict[str, Any] | None] | dict[str, Any] | None = None,
    ) -> VecResetResult[Any]:
        self._check_open()
        opts = self._expand_options(options)  # validated before the generator is touched
        self._seed(seed)
        indices = np.arange(self.num_envs)
        self._frozen[:] = False
        self._reset_envs(indices, opts)
        self._elapsed[:] = 0
        self._needs_reset = False
        infos = split_infos(self._reset_infos(indices), self.num_envs)
        return VecResetResult(observations=self._observe(), infos=infos)

    def step(self, actions: Any) -> VecStepResult[Any]:
        self._check_open()
        if self._needs_reset:
            raise ResetRequired()
        frozen = self._frozen.copy()  # sub-envs that finished on an earlier step
        if frozen.any():
            actions = self._hold_actions(actions, frozen)
        rewards, terminated, truncated, info_arrays = self._step(actions)
        # Copies: a subclass may return its own state arrays, which it mutates next step.
        rewards = np.array(rewards, dtype=np.float64)
        terminated = np.array(terminated, dtype=np.bool_)
        truncated = np.array(truncated, dtype=np.bool_)
        infos = split_infos(info_arrays, self.num_envs)
        self._elapsed[~frozen] += 1
        if self.max_episode_steps is not None:
            # What TimeLimit does for each sub-env of a SyncVecEnv.
            truncated |= self._elapsed >= self.max_episode_steps
            for info, elapsed in zip(infos, self._elapsed.tolist()):
                info["elapsed_steps"] = elapsed
        dones = (terminated | truncated) & ~frozen
        if self.autoreset and dones.any():
            done_idx = np.flatnonzero(dones)
            terminal = self._observe()
            finals = [take_index(self.observation_space, terminal, int(i)) for i in done_idx]
            self._reset_envs(done_idx, [None] * len(done_idx))
            self._elapsed[done_idx] = 0
            reset_info_arrays = self._reset_infos(done_idx)
            reset_infos = split_infos(reset_info_arrays, len(done_idx))
            for i, final, reset_info in zip(done_idx, finals, reset_infos):
                infos[i] = {
                    **infos[i],
                    "final_observation": final,
                    "final_info": infos[i],
                    "reset_info": reset_info,
                }
        elif dones.any():
            self._freeze(np.flatnonzero(dones), terminated, truncated, actions)
        observations = self._observe()
        if frozen.any():
            idx = np.flatnonzero(frozen)
            observations = _put_rows(self.observation_space, observations, self._final_obs, idx)
            rewards[idx] = 0.0
            terminated[idx] = self._final_terminated[idx]
            truncated[idx] = self._final_truncated[idx]
            for i in idx.tolist():
                infos[i] = {}
        return VecStepResult(
            observations=observations,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
        )

    def _freeze(
        self,
        indices: NDArray[np.intp],
        terminated: NDArray[np.bool_],
        truncated: NDArray[np.bool_],
        actions: Any,
    ) -> None:
        """Record what step() returns for sub-envs that just finished, until reset()."""
        terminal = self._observe()  # a new batch, so these rows stay as they are
        if self._final_obs is None:
            self._final_obs = terminal
        else:
            self._final_obs = _put_rows(self.observation_space, self._final_obs, terminal, indices)
        self._final_terminated[indices] = terminated[indices]
        self._final_truncated[indices] = truncated[indices]
        self._frozen[indices] = True
        batch = _numeric_batch(actions, self.num_envs)
        if batch is not None:
            if self._held_actions is None or self._held_actions.shape != batch.shape:
                self._held_actions = np.array(batch)
            else:
                self._held_actions[indices] = batch[indices]

    def _hold_actions(self, actions: Any, frozen: NDArray[np.bool_]) -> Any:
        """actions with each frozen sub-env's replaced by its last one (a numeric array only)."""
        held = self._held_actions
        batch = _numeric_batch(actions, self.num_envs)
        if held is None or batch is None or batch.shape != held.shape:
            return actions
        batch = np.array(batch)
        batch[frozen] = held[frozen]
        return batch

    @property
    def frozen(self) -> NDArray[np.bool_]:
        """Which sub-envs are frozen (a copy): finished with autoreset off, not reset since.

        step() masks what `_step()` computes for them (see the class docstring), but
        `_step()` still runs for every sub-env; check this mask to leave them as they are.
        """
        out: NDArray[np.bool_] = self._frozen.copy()
        return out

    def close(self) -> None:
        self.closed = True

    @property
    def elapsed_steps(self) -> NDArray[np.int64]:
        """Steps taken in each sub-env's current episode (a copy), as TimeLimit counts them."""
        out: NDArray[np.int64] = self._elapsed.copy()
        return out

    def _is_shared(self, name: str) -> bool:
        """Whether `name` is a method or property marked with `@shared`."""
        member = getattr(type(self), name, None)
        if isinstance(member, property):
            member = member.fget
        return getattr(member, _SHARED, False) is True

    def _per_env(self, name: str, value: Any) -> list[Any]:
        if (
            isinstance(value, np.ndarray)
            and value.shape[:1] == (self.num_envs,)
            and not self._is_shared(name)
        ):
            return list(value)
        return [value] * self.num_envs

    def call(self, method: str, *args: Any, **kwargs: Any) -> list[Any]:
        attr = getattr(self, method)
        return self._per_env(method, attr(*args, **kwargs) if callable(attr) else attr)

    def get_attr(self, name: str) -> list[Any]:
        return self._per_env(name, getattr(self, name))

    def set_attr(self, name: str, values: list[Any] | Any) -> None:
        if not hasattr(self, name):
            raise AttributeError(
                f"{type(self).__name__} has no attribute {name!r}. set_attr() on a native "
                "VecEnv changes existing attributes only: a new one would have no effect on "
                "the simulation."
            )
        if isinstance(values, list):
            if len(values) != self.num_envs:
                raise ValueError(f"Expected {self.num_envs} values, got {len(values)}.")
            values = np.asarray(values)
        setattr(self, name, values)
