from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sekai._types import ActType, InfoDict, ObsType, SeedType
from sekai.spaces.dict import Dict
from sekai.spaces.space import Space
from sekai.spaces.tuple import Tuple
from sekai.vector.result import VecResetResult, VecStepResult
from sekai.vector.vec_env import VecEnv

InfoArrays = Mapping[str, NDArray[Any]]


def take_index(space: Space[Any], batch: Any, index: int) -> Any:
    """One sub-env's sample (a copy) from a batched sample of space."""
    if isinstance(space, Dict):
        return {key: take_index(sub, batch[key], index) for key, sub in space.items()}
    if isinstance(space, Tuple):
        return tuple(take_index(sub, b, index) for sub, b in zip(space.spaces, batch))
    item = batch[index]
    return item.copy() if isinstance(item, np.ndarray) else item


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
    dicts, and an optional time limit: with `max_episode_steps=n` every
    sub-env is truncated after n steps and its infos carry `elapsed_steps`,
    exactly as `TimeLimit` does for each sub-env of a SyncVecEnv. The
    `elapsed_steps` property counts each sub-env's steps, limit or not.

    Because all sub-envs share one generator, a native env is not
    draw-for-draw identical to the same env under SyncVecEnv; it is identical
    in distribution, and exactly identical whenever the dynamics are
    deterministic given the options (e.g. historical data with fixed starts).

    `get_attr(name)` / `call(name)` read attributes and methods of this object:
    an array result with a leading `num_envs` axis is split per env, anything
    else is repeated for every env. `set_attr(name, values)` changes an existing
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
        from sekai.utils.seeding import np_random

        if isinstance(seed, list):
            if len(seed) != self.num_envs:
                raise ValueError(f"Expected {self.num_envs} seeds, got {len(seed)}.")
            ints = [int(s) for s in seed if isinstance(s, (int, np.integer))]
            if len(ints) != len(seed):
                raise TypeError("A native VecEnv takes an int seed or a list of int seeds.")
            self._rng = np.random.default_rng(ints)
            self._rng_seed = -1
        else:
            self._rng, self._rng_seed = np_random(seed)

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
        if seed is not None:
            self._seed(seed)
        indices = np.arange(self.num_envs)
        self._reset_envs(indices, self._expand_options(options))
        self._elapsed[:] = 0
        infos = split_infos(self._reset_infos(indices), self.num_envs)
        return VecResetResult(observations=self._observe(), infos=infos)

    def step(self, actions: Any) -> VecStepResult[Any]:
        self._check_open()
        rewards, terminated, truncated, info_arrays = self._step(actions)
        # Copies: a subclass may return its own state arrays, which it mutates next step.
        terminated = np.array(terminated, dtype=np.bool_)
        truncated = np.array(truncated, dtype=np.bool_)
        infos = split_infos(info_arrays, self.num_envs)
        self._elapsed += 1
        if self.max_episode_steps is not None:
            # What TimeLimit does for each sub-env of a SyncVecEnv.
            truncated |= self._elapsed >= self.max_episode_steps
            for info, elapsed in zip(infos, self._elapsed.tolist()):
                info["elapsed_steps"] = elapsed
        dones = terminated | truncated
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
        return VecStepResult(
            observations=self._observe(),
            rewards=np.array(rewards, dtype=np.float64),
            terminated=terminated,
            truncated=truncated,
            infos=infos,
        )

    def close(self) -> None:
        self.closed = True

    @property
    def elapsed_steps(self) -> NDArray[np.int64]:
        """Steps taken in each sub-env's current episode (a copy), as TimeLimit counts them."""
        out: NDArray[np.int64] = self._elapsed.copy()
        return out

    def _per_env(self, value: Any) -> list[Any]:
        if isinstance(value, np.ndarray) and value.shape[:1] == (self.num_envs,):
            return list(value)
        return [value] * self.num_envs

    def call(self, method: str, *args: Any, **kwargs: Any) -> list[Any]:
        attr = getattr(self, method)
        return self._per_env(attr(*args, **kwargs) if callable(attr) else attr)

    def get_attr(self, name: str) -> list[Any]:
        return self._per_env(getattr(self, name))

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
