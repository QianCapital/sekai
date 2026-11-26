from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.typing import NDArray
from stable_baselines3.common.vec_env.base_vec_env import VecEnv as SB3VecEnv
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)

from sekai._types import SeedType
from sekai.compat._gymnasium import space_to_gymnasium
from sekai.core.wrapper import Wrapper
from sekai.stats.tracker import EpisodeStats
from sekai.vector.native import NativeVecEnv
from sekai.vector.vec_env import VecEnv, env_call


class SekaiToSB3VecEnv(SB3VecEnv):
    """Exposes a sekai VecEnv through Stable-Baselines3's VecEnv API.

    Use it to train on a sekai vector env, a native one included, with Stable-Baselines3
    (PPO, A2C, SAC, ...) instead of rebuilding it from single envs with `make_vec_env`.
    Both libraries reset a finished sub-env inside the same `step()`, so results map one
    to one:

    - observations are the sekai VecEnv's batched observations (an array, or a dict of
      arrays for a `Dict` space), and the spaces are its single-env spaces, converted with
      `space_to_gymnasium()`;
    - rewards are float32, and dones are `terminated | truncated`;
    - infos are the sekai infos plus SB3's keys: `terminal_observation` (sekai's
      `final_observation`) when a sub-env finishes, and `TimeLimit.truncated` (truncated
      and not terminated) on every step, which SB3 uses to bootstrap the value of a
      truncated episode;
    - an `EpisodeStats` in `info["episode"]` (from `VecRecordEpisodeStatistics`) becomes
      the `{"r", "l", "t"}` dict that SB3's logger reads, with `"t"` counted from when
      this wrapper was made, as SB3's `Monitor` counts it. The typed statistics stay in
      that wrapper's tracker.

    Seeds and options work as in SB3: `seed(s)` and `set_options(...)` take effect at the
    next `reset()`, and sub-env `i` gets the seed `s + i`. (A native VecEnv combines a list
    of seeds into one seed for its shared generator, as it does for any list.) The sekai
    VecEnv must autoreset, as it does by default, since SB3 never resets sub-envs itself.
    sekai envs render with separate Renderer objects, so `render_mode` is None.

    `get_attr`, `set_attr` and `env_method` go through the sekai VecEnv's `get_attr`,
    `set_attr` and `call`. With `indices` that pick only some sub-envs, `set_attr` writes
    the others' current values back, and `env_method` calls the chosen sub-envs one by one,
    which needs a VecEnv that holds them (`SyncVecEnv`, `AsyncVecEnv`): a native VecEnv
    has no sub-env objects and a `ProcVecEnv`'s live in its workers, so for those it raises
    NotImplementedError rather than call every sub-env. `env_is_wrapped` is False for any
    gymnasium wrapper (SB3 asks about its `Monitor`), which cannot wrap a sekai env, and
    looks through each sub-env's wrapper stack for a sekai wrapper.
    """

    def __init__(self, venv: VecEnv[Any, Any]) -> None:
        if not isinstance(venv, VecEnv):
            raise TypeError(
                f"to_sb3_vec_env() takes a sekai VecEnv, got {type(venv).__name__}. For a single "
                "env, convert it with to_gymnasium() and vectorise it with SB3's make_vec_env()."
            )
        if not venv.autoreset:
            raise ValueError(
                "to_sb3_vec_env() needs a VecEnv with autoreset=True: Stable-Baselines3 never "
                "resets a finished sub-env itself."
            )
        self.sekai_venv = venv
        self._actions: Any = None
        self._start_time = time.monotonic()  # episode "t" counts from here, as Monitor's does
        super().__init__(
            venv.num_envs,
            space_to_gymnasium(venv.observation_space),
            space_to_gymnasium(venv.action_space),
        )

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def reset(self) -> VecEnvObs:
        seeds: list[SeedType] | None = None
        if any(s is not None for s in self._seeds):
            seeds = [s for s in self._seeds]
        options = [o or None for o in self._options] if any(self._options) else None
        result = self.sekai_venv.reset(seed=seeds, options=options)
        self.reset_infos = list(result.infos)
        self._reset_seeds()  # seeds and options apply to one reset, as in SB3
        self._reset_options()
        observations: VecEnvObs = result.observations
        return observations

    def step_async(self, actions: NDArray[Any]) -> None:
        self._actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        result = self.sekai_venv.step(self._actions)
        infos: list[dict[str, Any]] = []
        for i, info in enumerate(result.infos):
            out = dict(info)
            terminated, truncated = bool(result.terminated[i]), bool(result.truncated[i])
            out["TimeLimit.truncated"] = truncated and not terminated
            if terminated or truncated:
                if "final_observation" in info:
                    out["terminal_observation"] = info["final_observation"]
                if "reset_info" in info:
                    self.reset_infos[i] = info["reset_info"]
            episode = info.get("episode")
            if isinstance(episode, EpisodeStats):
                out["episode"] = {
                    "r": round(episode.episode_return, 6),
                    "l": episode.episode_length,
                    "t": round(episode.end_time - self._start_time, 6),
                }
            infos.append(out)
        rewards = result.rewards.astype(np.float32)
        return result.observations, rewards, result.dones, infos

    def close(self) -> None:
        self.sekai_venv.close()

    # ------------------------------------------------------------------
    # Reaching the sub-envs
    # ------------------------------------------------------------------

    def _targets(self, indices: VecEnvIndices) -> tuple[list[int], bool]:
        """The sub-env indices `indices` picks, and whether they are all of them."""
        targets = [int(i) for i in self._get_indices(indices)]
        return targets, set(targets) == set(range(self.num_envs))

    def _sub_envs(self) -> list[Any] | None:
        """The sub-env objects of a SyncVecEnv or AsyncVecEnv, under any vector wrappers."""
        envs = getattr(getattr(self.sekai_venv, "unwrapped", self.sekai_venv), "envs", None)
        return envs if isinstance(envs, list) else None

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        targets, _ = self._targets(indices)
        if attr_name == "render_mode":  # read by SB3's VecEnv.__init__; sekai envs have none
            return [None] * len(targets)
        values = self.sekai_venv.get_attr(attr_name)
        return [values[i] for i in targets]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        targets, everyone = self._targets(indices)
        if everyone:
            # sekai's set_attr reads a list as one value per sub-env; SB3's value is one value.
            self.sekai_venv.set_attr(
                attr_name, [value] * self.num_envs if isinstance(value, list) else value
            )
            return
        chosen = set(targets)
        current = self.sekai_venv.get_attr(attr_name)
        self.sekai_venv.set_attr(
            attr_name, [value if i in chosen else v for i, v in enumerate(current)]
        )

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: VecEnvIndices = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        targets, everyone = self._targets(indices)
        if everyone:
            results = self.sekai_venv.call(method_name, *method_args, **method_kwargs)
            return [results[i] for i in targets]
        envs = self._sub_envs()
        if envs is None:
            raise NotImplementedError(
                f"env_method() with indices needs the sub-env objects, which "
                f"{type(self.sekai_venv).__name__} does not hold, and its call() reaches every "
                "sub-env. Call env_method() without indices, or use a SyncVecEnv."
            )
        return [env_call(envs[i], method_name, method_args, method_kwargs) for i in targets]

    def env_is_wrapped(self, wrapper_class: type[Any], indices: VecEnvIndices = None) -> list[bool]:
        targets, _ = self._targets(indices)
        unwrapped = getattr(self.sekai_venv, "unwrapped", self.sekai_venv)
        if not issubclass(wrapper_class, Wrapper) or isinstance(unwrapped, NativeVecEnv):
            # A gymnasium wrapper cannot wrap a sekai env, and a native VecEnv's sub-envs
            # are rows of its arrays, which nothing wraps.
            return [False] * len(targets)
        envs = self._sub_envs()
        if envs is None:
            raise NotImplementedError(
                f"env_is_wrapped() needs the sub-env objects, which "
                f"{type(self.sekai_venv).__name__} does not hold."
            )
        return [
            isinstance(envs[i], Wrapper) and any(isinstance(e, wrapper_class) for e in envs[i])
            for i in targets
        ]

    def __repr__(self) -> str:
        return f"SekaiToSB3VecEnv({self.sekai_venv!r})"
