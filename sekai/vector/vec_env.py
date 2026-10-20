from __future__ import annotations

import abc
import contextlib
from collections.abc import Callable, Sequence
from typing import Any, Generic, TypeVar

import numpy as np

from sekai._types import ActType, InfoDict, ObsType, SeedType
from sekai.core._async_fallback import install_async_fallbacks
from sekai.core.result import ResetResult, StepResult
from sekai.error import VecEnvClosed
from sekai.spaces.space import Space
from sekai.vector.result import VecResetResult, VecStepResult

_E = TypeVar("_E")

# One generator cannot seed num_envs sub-envs independently.
GENERATOR_SEED_ERROR = "VecEnv.reset() takes an int seed or a list of seeds, not a Generator."


class VecEnv(abc.ABC, Generic[ObsType, ActType]):
    """Abstract base for vectorised environments.

    Runs N environments in parallel and returns batched results.

    Key differences from gymnasium's VectorEnv:
      - observation_space is always the *single-env* space
      - batch_observation_space is always the *batched* space
      - step/reset return structured dataclasses, not tuples
      - infos is list[dict], not dict[str, ndarray]

    Autoreset (on by default): when a sub-env finishes an episode it is reset
    in the same step() call. The returned observation for that env is the
    first observation of the new episode; the terminal observation and info
    are in `infos[i]["final_observation"]` and `infos[i]["final_info"]`, and the
    reset info in `infos[i]["reset_info"]`. This matches the AutoReset wrapper.

    Without autoreset (`autoreset=False`), a sub-env that finishes an episode is
    frozen until the next reset(): it is not stepped again and its action is
    ignored, and every later step() returns, for it, its final observation,
    reward 0, its final terminated / truncated flags (so `dones` stays True) and
    an empty info. A sub-env that resets itself (e.g. wrapped in AutoReset, so
    its info carries final_observation) is not frozen.

    Every VecEnv has async_step()/async_reset(); by default they run the sync
    methods (AsyncVecEnv overrides them to run sub-envs concurrently). As with
    Env, a subclass that overrides only step()/reset() gets async methods that
    run its overrides.
    """

    num_envs: int
    observation_space: Space[ObsType]
    action_space: Space[ActType]
    autoreset: bool = True
    closed: bool = False
    spec: Any = None  # the EnvSpec when created by sekai.make_vec()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        install_async_fallbacks(cls)

    def _batched(self, attr: str, space: Space[Any]) -> Space[Any]:
        # Cached so that e.g. vec.batch_action_space.seed(0) affects later samples;
        # rebuilt only if the single-env space object is replaced.
        cached: tuple[Space[Any], Space[Any]] | None = self.__dict__.get(attr)
        if cached is None or cached[0] is not space:
            from sekai.spaces.utils import batch_space

            cached = (space, batch_space(space, self.num_envs))
            self.__dict__[attr] = cached
        return cached[1]

    @property
    def batch_observation_space(self) -> Space[Any]:
        """The space of a batched observation, with a leading num_envs axis."""
        return self._batched("_batch_obs_space_cache", self.observation_space)

    @property
    def batch_action_space(self) -> Space[Any]:
        """The space of a batched action, with a leading num_envs axis."""
        return self._batched("_batch_act_space_cache", self.action_space)

    @abc.abstractmethod
    def step(self, actions: Any) -> VecStepResult[Any]:
        """Step all environments with a batch of actions."""
        ...

    @abc.abstractmethod
    def reset(
        self,
        *,
        seed: list[SeedType] | SeedType = None,
        options: list[dict[str, Any] | None] | dict[str, Any] | None = None,
    ) -> VecResetResult[Any]:
        """Reset all environments.

        seed: an int derives one independent seed per sub-env; a list gives
        each sub-env its own seed; None continues each sub-env's RNG stream.
        options: one dict for all sub-envs, or a list with one per sub-env.
        """
        ...

    @abc.abstractmethod
    def close(self) -> None:
        """Close all sub-environments and release resources.

        Every sub-env is closed even if closing another one raises; the first such
        error is re-raised once everything is released.
        """
        ...

    async def async_step(self, actions: Any) -> VecStepResult[Any]:
        """Async step; the default runs step()."""
        return self.step(actions)

    async def async_reset(
        self,
        *,
        seed: list[SeedType] | SeedType = None,
        options: list[dict[str, Any] | None] | dict[str, Any] | None = None,
    ) -> VecResetResult[Any]:
        """Async reset; the default runs reset()."""
        return self.reset(seed=seed, options=options)

    def call(self, method: str, *args: Any, **kwargs: Any) -> list[Any]:
        """Call a method (or read an attribute) on each sub-environment.

        Like get_attr(), the name is looked up through each sub-env's wrapper stack.
        """
        raise NotImplementedError

    def get_attr(self, name: str) -> list[Any]:
        """Get an attribute from each sub-environment, searching its wrapper stack."""
        raise NotImplementedError

    def set_attr(self, name: str, values: list[Any] | Any) -> None:
        """Set an attribute on each sub-environment (one value, or a list of num_envs).

        The value goes to the outermost layer of each sub-env's wrapper stack that
        has the attribute, as with Wrapper.set_attr().
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared helpers for implementations
    # ------------------------------------------------------------------

    def _check_open(self) -> None:
        if self.closed:
            raise VecEnvClosed()

    def _expand_seeds(self, seed: list[SeedType] | SeedType) -> list[SeedType]:
        if isinstance(seed, list):
            if len(seed) != self.num_envs:
                raise ValueError(f"Expected {self.num_envs} seeds, got {len(seed)}.")
            return seed
        if seed is None:
            return [None] * self.num_envs
        if isinstance(seed, np.random.Generator):
            raise TypeError(GENERATOR_SEED_ERROR)
        from sekai.utils.seeding import make_seed_sequence

        return list(make_seed_sequence(int(seed), self.num_envs))

    def _expand_options(
        self, options: list[dict[str, Any] | None] | dict[str, Any] | None
    ) -> list[dict[str, Any] | None]:
        if isinstance(options, list):
            if len(options) != self.num_envs:
                raise ValueError(f"Expected {self.num_envs} options dicts, got {len(options)}.")
            return options
        return [options] * self.num_envs

    def _split_actions(self, actions: Any) -> list[Any]:
        from sekai.spaces.utils import iterate

        per_env = list(iterate(self.action_space, actions))
        if len(per_env) != self.num_envs:
            raise ValueError(f"Expected actions for {self.num_envs} envs, got {len(per_env)}.")
        return per_env

    def _batch_obs(self, observations: Sequence[Any]) -> Any:
        from sekai.spaces.utils import concatenate

        return concatenate(self.observation_space, observations)

    def _batch_step(
        self, results: Sequence[StepResult[Any]], observations: Any = None
    ) -> VecStepResult[Any]:
        if observations is None:
            observations = self._batch_obs([r.observation for r in results])
        return VecStepResult(
            observations=observations,
            rewards=np.array([r.reward for r in results], dtype=np.float64),
            terminated=np.array([r.terminated for r in results], dtype=np.bool_),
            truncated=np.array([r.truncated for r in results], dtype=np.bool_),
            infos=[r.info for r in results],
        )

    def _batch_reset(
        self, results: Sequence[ResetResult[Any]], observations: Any = None
    ) -> VecResetResult[Any]:
        if observations is None:
            observations = self._batch_obs([r.observation for r in results])
        return VecResetResult(
            observations=observations,
            infos=[r.info for r in results],
        )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self.num_envs

    def __enter__(self) -> VecEnv[ObsType, ActType]:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(num_envs={self.num_envs})"


def env_get_attr(env: Any, name: str) -> Any:
    """Read an attribute through a sub-env's wrapper stack (plain getattr for non-sekai envs)."""
    return env.get_attr(name) if hasattr(env, "get_attr") else getattr(env, name)


def env_set_attr(env: Any, name: str, value: Any) -> None:
    if hasattr(env, "set_attr"):
        env.set_attr(name, value)
    else:
        setattr(env, name, value)


def env_call(env: Any, name: str, args: Any, kwargs: Any) -> Any:
    """Call a method found through the wrapper stack; non-callable attributes are returned."""
    attr = env_get_attr(env, name)
    return attr(*args, **kwargs) if callable(attr) else attr


def build_envs(env_fns: Sequence[Callable[[], _E]]) -> list[_E]:
    """Call every env_fn; if one raises, close the envs already built, then re-raise."""
    envs: list[_E] = []
    try:
        for fn in env_fns:
            envs.append(fn())
    except BaseException:
        with contextlib.suppress(Exception):  # the construction error is the one to report
            close_envs(envs)
        raise
    return envs


def close_envs(envs: Sequence[Any]) -> None:
    """Close every env, then re-raise the first error, so one failure leaks no other env."""
    error: Exception | None = None
    for env in envs:
        try:
            env.close()
        except Exception as e:
            error = error or e
    if error is not None:
        raise error


def needs_autoreset(result: StepResult[Any]) -> bool:
    """Whether a sub-env step result should trigger a vec-level reset.

    A sub-env that already auto-resets itself (e.g. wrapped in AutoReset)
    reports final_observation in its info; resetting again would discard the
    new episode's first observation.
    """
    return result.done and "final_observation" not in result.info


def frozen_step(final: StepResult[Any]) -> StepResult[Any]:
    """What a finished sub-env returns, without being stepped, while autoreset is off.

    Its final observation and terminated / truncated flags, reward 0 and a new empty
    info (see VecEnv); `final` is the step that finished its episode.
    """
    return StepResult(
        observation=final.observation,
        reward=0.0,
        terminated=final.terminated,
        truncated=final.truncated,
        info={},
    )


def merge_autoreset(step: StepResult[Any], reset: ResetResult[Any]) -> StepResult[Any]:
    """Combine a terminal step with the following reset, AutoReset-style."""
    info: InfoDict = {
        **step.info,
        "final_observation": step.observation,
        "final_info": step.info,
        "reset_info": reset.info,
    }
    return StepResult(
        observation=reset.observation,
        reward=step.reward,
        terminated=step.terminated,
        truncated=step.truncated,
        info=info,
    )
