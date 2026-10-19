from __future__ import annotations

from typing import Any

from sekai._types import ActType
from sekai.core.env import Env
from sekai.core.wrapper import ObservationWrapper
from sekai.spaces.utils import concatenate, iterate
from sekai.utils.normalization import ObservationNormalizer, StatsKind


class NormalizeObservation(ObservationWrapper[Any, ActType, Any]):
    """Normalises observations to have approximately zero mean and unit variance.

    Works on any observation space: every Box gets its own running statistics,
    a MaskedBox gets per-feature statistics over its occupied slots only (so
    padding does not bias them), and other parts of a Dict/Tuple observation
    (Discrete, MultiBinary, ...) pass through unchanged. Normalised parts are
    float64.

    `stats="running"` (the default) weights every observation ever seen
    equally. On non-stationary data, such as prices whose volatility changes
    by regime, use `stats="ewm"` with a `halflife` (in steps), which tracks
    the recent distribution instead.

    Set `update_running_mean = False` to freeze the statistics, e.g. for
    evaluation, and use `state_dict()` / `load_state_dict()` to save and restore
    them. For a Box observation `obs_rms` is the statistics object itself; for
    nested spaces it mirrors the nesting (a dict for a Dict, ...).

    Args:
        env: The environment to wrap.
        epsilon: Added to the variance before dividing.
        stats: "running" or "ewm".
        halflife: Halflife in steps for stats="ewm".
        clip: If given, normalised values are clipped to [-clip, clip].
    """

    def __init__(
        self,
        env: Env[Any, ActType],
        epsilon: float = 1e-8,
        *,
        stats: StatsKind = "running",
        halflife: float = 1000.0,
        clip: float | None = None,
    ) -> None:
        super().__init__(env)
        self.normalizer = ObservationNormalizer(
            env.observation_space, stats=stats, halflife=halflife, epsilon=epsilon, clip=clip
        )
        self.epsilon = epsilon
        self.update_running_mean: bool = True
        self._inner_space = env.observation_space
        self.observation_space = self.normalizer.output_space

    @property
    def obs_rms(self) -> Any:
        """The running statistics (nested like the observation space)."""
        return self.normalizer.stats

    def observation(self, obs: Any) -> Any:
        batch = concatenate(self._inner_space, [obs])
        out = self.normalizer(batch, update=self.update_running_mean)
        return next(iter(iterate(self.observation_space, out)))

    def state_dict(self) -> Any:
        """The normalisation statistics as plain data."""
        return self.normalizer.state_dict()

    def load_state_dict(self, state: Any) -> None:
        """Restore statistics saved by state_dict()."""
        self.normalizer.load_state_dict(state)
