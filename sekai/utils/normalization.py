"""Observation normalisation over arbitrary (nested) observation spaces.

Shared by `NormalizeObservation` and `VecNormalizeObservation`. Every Box leaf
of the observation space gets its own running statistics; a `MaskedBox` leaf
gets per-feature statistics computed over its occupied slots only, so padding
never biases them; other leaves (Discrete, MultiBinary, Sequence, ...) pass
through unchanged.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from sekai.spaces.box import Box
from sekai.spaces.dict import Dict
from sekai.spaces.masked_box import MaskedBox
from sekai.spaces.space import Space
from sekai.spaces.tuple import Tuple
from sekai.utils.running_stats import ExponentialMovingStats, RunningMeanStd

StatsKind = Literal["running", "ewm"]
Stats = RunningMeanStd | ExponentialMovingStats


class ObservationNormalizer:
    """Normalises batches of observations of `space` (leading batch axis on every leaf).

    Args:
        space: The (unbatched) observation space.
        stats: "running" for equal-weight statistics over all samples (RunningMeanStd),
            "ewm" for exponentially weighted ones that follow non-stationary data
            (ExponentialMovingStats).
        halflife: Halflife for stats="ewm", in calls: each normalised batch (one env
            step, or one step of a vector env) counts once.
        epsilon: Added to the variance before dividing.
        clip: If given, normalised values are clipped to [-clip, clip].
    """

    def __init__(
        self,
        space: Space[Any],
        *,
        stats: StatsKind = "running",
        halflife: float = 1000.0,
        epsilon: float = 1e-8,
        clip: float | None = None,
    ) -> None:
        if stats not in ("running", "ewm"):
            raise ValueError(f"stats must be 'running' or 'ewm', got {stats!r}.")
        if clip is not None and clip <= 0:
            raise ValueError(f"clip must be positive, got {clip}.")
        self.kind: StatsKind = stats
        self.halflife = halflife
        self.epsilon = epsilon
        self.clip = clip
        self.space = space
        self.stats: Any = self._build(space)
        if not self._has_stats(self.stats):
            raise TypeError(
                "Observation normalisation needs at least one Box or MaskedBox in the "
                f"observation space, got {space!r}."
            )
        self.output_space: Space[Any] = self._output_space(space)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _new_stats(self, shape: tuple[int, ...]) -> Stats:
        if self.kind == "ewm":
            return ExponentialMovingStats(shape=shape, halflife=self.halflife)
        return RunningMeanStd(shape=shape)

    def _build(self, space: Space[Any]) -> Any:
        if isinstance(space, MaskedBox):
            return self._new_stats(space.element_shape)
        if isinstance(space, Dict):
            return {key: self._build(sub) for key, sub in space.items()}
        if isinstance(space, Tuple):
            return tuple(self._build(sub) for sub in space.spaces)
        if isinstance(space, Box):
            return self._new_stats(space.shape)
        return None

    def _has_stats(self, node: Any) -> bool:
        if isinstance(node, dict):
            return any(self._has_stats(sub) for sub in node.values())
        if isinstance(node, tuple):
            return any(self._has_stats(sub) for sub in node)
        return node is not None

    def _output_space(self, space: Space[Any]) -> Space[Any]:
        lo, hi = (-np.inf, np.inf) if self.clip is None else (-self.clip, self.clip)
        if isinstance(space, MaskedBox):
            return MaskedBox(
                lo, hi, shape=space.element_shape, max_size=space.max_size, dtype=np.float64
            )
        if isinstance(space, Dict):
            return Dict({key: self._output_space(sub) for key, sub in space.items()})
        if isinstance(space, Tuple):
            return Tuple([self._output_space(sub) for sub in space.spaces])
        if isinstance(space, Box):
            return Box(lo, hi, shape=space.shape, dtype=np.float64)
        return space

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def __call__(self, batch: Any, *, update: bool = True) -> Any:
        """Normalise a batch, first updating the statistics with it if `update`."""
        return self._apply(self.space, self.stats, batch, update)

    @staticmethod
    def _update(node: Stats, rows: NDArray[np.float64]) -> None:
        if isinstance(node, ExponentialMovingStats):
            # The halflife is in steps: a call (one env step, or one vector step over all
            # sub-envs and occupied slots) counts once, however many rows it carries.
            node.update_from_moments(rows.mean(axis=0), rows.var(axis=0), 1)
        else:
            node.update(rows)

    def _finish(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.clip is not None:
            np.clip(x, -self.clip, self.clip, out=x)
        return x

    def _apply(self, space: Space[Any], node: Any, batch: Any, update: bool) -> Any:
        if node is None:
            return batch
        if isinstance(space, MaskedBox):
            values = np.asarray(batch["values"], dtype=np.float64)
            mask = np.asarray(batch["mask"])
            occupied = mask.astype(bool)
            if update and occupied.any():
                self._update(node, values[occupied])
            out = node.normalize(values, self.epsilon)
            out[~occupied] = 0.0
            return {"values": self._finish(out), "mask": mask}
        if isinstance(space, Dict):
            return {
                key: self._apply(sub, node[key], batch[key], update) for key, sub in space.items()
            }
        if isinstance(space, Tuple):
            return tuple(
                self._apply(sub, n, b, update) for sub, n, b in zip(space.spaces, node, batch)
            )
        x = np.asarray(batch, dtype=np.float64)
        if update:
            self._update(node, x)
        return self._finish(node.normalize(x, self.epsilon))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state_dict(self) -> Any:
        """The statistics of every leaf, as nested plain data (same nesting as the space)."""

        def dump(node: Any) -> Any:
            if isinstance(node, dict):
                return {key: dump(sub) for key, sub in node.items()}
            if isinstance(node, tuple):
                return [dump(sub) for sub in node]
            return None if node is None else node.state_dict()

        return dump(self.stats)

    def load_state_dict(self, state: Any) -> None:
        """Restore statistics saved by state_dict()."""

        def load(node: Any, data: Any) -> None:
            if isinstance(node, dict):
                for key, sub in node.items():
                    load(sub, data[key])
            elif isinstance(node, tuple):
                for sub, d in zip(node, data):
                    load(sub, d)
            elif node is not None:
                node.load_state_dict(data)

        load(self.stats, state)


__all__ = ["ObservationNormalizer"]
