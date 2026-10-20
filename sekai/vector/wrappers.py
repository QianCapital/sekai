"""Ready-made VecEnv wrappers for training loops."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sekai._types import ActType, InfoDict, ObsType, SeedType
from sekai.spaces.space import Space
from sekai.stats.collectors import InMemoryTracker
from sekai.stats.tracker import EpisodeStats, StatsTracker
from sekai.utils.normalization import ObservationNormalizer, StatsKind
from sekai.utils.running_stats import RunningMeanStd
from sekai.vector.result import VecResetResult, VecStepResult
from sekai.vector.vec_env import VecEnv
from sekai.vector.vec_wrapper import VecObservationWrapper, VecRewardWrapper, VecWrapper


class VecRecordEpisodeStatistics(VecWrapper[ObsType, ActType, ObsType, ActType]):
    """Per-sub-env episode statistics for a VecEnv.

    When sub-env i finishes an episode, `infos[i]["episode"]` is an EpisodeStats,
    exactly as with the single-env RecordEpisodeStatistics, and the stats are
    reported to the tracker. Works with autoreset on or off: without autoreset,
    a finished sub-env is frozen (see VecEnv) and keeps reporting done, so it is
    ignored until the next reset() and stepping the others on (e.g.
    `while not finished.all()`) records each episode once.
    """

    def __init__(
        self,
        venv: VecEnv[ObsType, ActType],
        tracker: StatsTracker | None = None,
        deque_size: int = 100,
    ) -> None:
        super().__init__(venv)
        self.tracker: StatsTracker = tracker or InMemoryTracker(max_episodes=deque_size)
        self._returns = np.zeros(self.num_envs, dtype=np.float64)
        self._lengths = np.zeros(self.num_envs, dtype=np.int64)
        self._start_times = np.full(self.num_envs, time.monotonic())
        # Sub-envs whose episode has ended and that nothing has reset yet (autoreset off).
        self._finished = np.zeros(self.num_envs, dtype=np.bool_)

    def _on_reset(self, result: VecResetResult[ObsType]) -> VecResetResult[ObsType]:
        self._returns[:] = 0.0
        self._lengths[:] = 0
        self._start_times[:] = time.monotonic()
        self._finished[:] = False
        return result

    def _on_step(self, result: VecStepResult[ObsType]) -> VecStepResult[ObsType]:
        active = ~self._finished
        self._returns[active] += result.rewards[active]
        self._lengths[active] += 1
        dones = result.dones & active
        if not dones.any():
            return result
        if not self.venv.autoreset:
            # A sub-env that resets itself (e.g. wrapped in AutoReset) reports
            # final_observation and plays on; any other stays finished until reset().
            self._finished |= dones & np.array(
                ["final_observation" not in info for info in result.infos], dtype=np.bool_
            )

        now = time.monotonic()
        infos: list[InfoDict] = list(result.infos)
        for i in np.flatnonzero(dones):
            stats = EpisodeStats(
                episode_return=float(self._returns[i]),
                episode_length=int(self._lengths[i]),
                elapsed_time=now - float(self._start_times[i]),
                start_time=float(self._start_times[i]),
                end_time=now,
                terminated=bool(result.terminated[i]),
                truncated=bool(result.truncated[i]),
                extra={"env_index": int(i)},
            )
            self.tracker.on_episode_end(stats)
            infos[i] = {**infos[i], "episode": stats}
            self._returns[i] = 0.0
            self._lengths[i] = 0
            self._start_times[i] = now
        return VecStepResult(
            observations=result.observations,
            rewards=result.rewards,
            terminated=result.terminated,
            truncated=result.truncated,
            infos=infos,
        )

    def reset(
        self,
        *,
        seed: list[SeedType] | SeedType = None,
        options: list[dict[str, Any] | None] | dict[str, Any] | None = None,
    ) -> VecResetResult[ObsType]:
        return self._on_reset(self.venv.reset(seed=seed, options=options))

    async def async_reset(
        self,
        *,
        seed: list[SeedType] | SeedType = None,
        options: list[dict[str, Any] | None] | dict[str, Any] | None = None,
    ) -> VecResetResult[ObsType]:
        return self._on_reset(await self.venv.async_reset(seed=seed, options=options))

    def step(self, actions: Any) -> VecStepResult[ObsType]:
        return self._on_step(self.venv.step(actions))

    async def async_step(self, actions: Any) -> VecStepResult[ObsType]:
        return self._on_step(await self.venv.async_step(actions))


class VecNormalizeObservation(VecObservationWrapper[Any, ActType, Any]):
    """Normalises observations with running statistics shared across all sub-envs.

    The vectorised counterpart of NormalizeObservation, with the same options:
    any observation space (Box leaves normalised, MaskedBox leaves over occupied
    slots only, other leaves passed through), `stats="running"` or `"ewm"` (the
    halflife counts vector steps), and optional clipping.
    `infos[i]["final_observation"]` from autoreset is normalised too, with the
    current statistics but without updating them.
    Set update_running_mean = False to freeze the statistics.
    """

    def __init__(
        self,
        venv: VecEnv[Any, ActType],
        epsilon: float = 1e-8,
        *,
        stats: StatsKind = "running",
        halflife: float = 1000.0,
        clip: float | None = None,
    ) -> None:
        super().__init__(venv)
        self.normalizer = ObservationNormalizer(
            venv.observation_space, stats=stats, halflife=halflife, epsilon=epsilon, clip=clip
        )
        self.epsilon = epsilon
        self.update_running_mean: bool = True
        self.observation_space = self.normalizer.output_space

    @property
    def obs_rms(self) -> Any:
        """The running statistics (nested like the observation space)."""
        return self.normalizer.stats

    def observation(self, obs: Any) -> Any:
        return self.normalizer(obs, update=self.update_running_mean)

    def final_observation(self, obs: Any) -> Any:
        # Normalise with the current statistics without updating them.
        from sekai.spaces.utils import concatenate, iterate

        batch = concatenate(self.venv.observation_space, [obs])
        out = self.normalizer(batch, update=False)
        return next(iter(iterate(self.observation_space, out)))

    def state_dict(self) -> Any:
        """The normalisation statistics as plain data."""
        return self.normalizer.state_dict()

    def load_state_dict(self, state: Any) -> None:
        """Restore statistics saved by state_dict()."""
        self.normalizer.load_state_dict(state)


class VecNormalizeReward(VecWrapper[ObsType, ActType, ObsType, ActType]):
    """Scales rewards by the running std of each sub-env's discounted return.

    The vectorised counterpart of NormalizeReward (OpenAI baselines'
    VecNormalize reward scaling). Returns are reset per sub-env when its
    episode ends. Set update_running_mean = False to freeze the statistics.
    """

    def __init__(
        self,
        venv: VecEnv[ObsType, ActType],
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(venv)
        self.gamma = gamma
        self.epsilon = epsilon
        self.return_rms = RunningMeanStd(shape=())
        self.update_running_mean: bool = True
        self._returns = np.zeros(self.num_envs, dtype=np.float64)

    def reset(
        self,
        *,
        seed: list[SeedType] | SeedType = None,
        options: list[dict[str, Any] | None] | dict[str, Any] | None = None,
    ) -> VecResetResult[ObsType]:
        self._returns[:] = 0.0
        return self.venv.reset(seed=seed, options=options)

    async def async_reset(
        self,
        *,
        seed: list[SeedType] | SeedType = None,
        options: list[dict[str, Any] | None] | dict[str, Any] | None = None,
    ) -> VecResetResult[ObsType]:
        self._returns[:] = 0.0
        return await self.venv.async_reset(seed=seed, options=options)

    def step(self, actions: Any) -> VecStepResult[ObsType]:
        return self._on_step(self.venv.step(actions))

    async def async_step(self, actions: Any) -> VecStepResult[ObsType]:
        return self._on_step(await self.venv.async_step(actions))

    def _on_step(self, result: VecStepResult[ObsType]) -> VecStepResult[ObsType]:
        self._returns = self._returns * self.gamma + result.rewards
        if self.update_running_mean:
            self.return_rms.update(self._returns)
        self._returns[result.dones] = 0.0
        scale = np.sqrt(self.return_rms.var + self.epsilon)
        return VecStepResult(
            observations=result.observations,
            rewards=result.rewards / scale,
            terminated=result.terminated,
            truncated=result.truncated,
            infos=result.infos,
        )


class VecTransformObservation(VecObservationWrapper[Any, ActType, Any]):
    """Applies func to every batch of observations.

    Example:
        ```python
        vec = VecTransformObservation(vec, lambda obs: obs / 255.0)
        ```
    """

    def __init__(
        self,
        venv: VecEnv[Any, ActType],
        func: Callable[[Any], Any],
        new_observation_space: Space[Any] | None = None,
    ) -> None:
        super().__init__(venv)
        self.func = func
        if new_observation_space is not None:
            self.observation_space = new_observation_space

    def observation(self, obs: Any) -> Any:
        return self.func(obs)


class VecTransformReward(VecRewardWrapper[ObsType, ActType]):
    """Applies func to every batch of rewards (an (N,) float64 array).

    Example:
        ```python
        vec = VecTransformReward(vec, np.sign)
        ```
    """

    def __init__(
        self,
        venv: VecEnv[ObsType, ActType],
        func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    ) -> None:
        super().__init__(venv)
        self.func = func

    def reward(self, rewards: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(self.func(rewards), dtype=np.float64)
