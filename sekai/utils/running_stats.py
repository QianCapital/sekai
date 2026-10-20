from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


class RunningMeanStd:
    """Running mean and variance over batches, using the parallel-variance update.

    Shared by the observation/reward normalisation wrappers (single-env and
    vectorised). Starts at mean 0, var 1 with a tiny pseudo-count (`epsilon`) so the
    first update dominates: the first batch sets the mean outright, and the prior's
    unit variance keeps only its pseudo-count's weight (so a single first sample does
    not give a zero variance). Pooling the first batch with the prior mean of 0
    instead would add about `mean**2 * epsilon / count` to the variance for good, a
    large error for data far from zero, such as prices. Empty batches are ignored.
    """

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4) -> None:
        self.mean: NDArray[np.float64] = np.zeros(shape, dtype=np.float64)
        self.var: NDArray[np.float64] = np.ones(shape, dtype=np.float64)
        self.count: float = epsilon
        self.epsilon = epsilon

    def update(self, batch: ArrayLike) -> None:
        """Update with a batch of samples stacked along axis 0."""
        x = np.asarray(batch, dtype=np.float64)
        if x.shape[0] == 0:
            return
        self.update_from_moments(x.mean(axis=0), x.var(axis=0), x.shape[0])

    def update_from_moments(
        self, batch_mean: ArrayLike, batch_var: ArrayLike, batch_count: int
    ) -> None:
        if batch_count <= 0:
            return
        batch_mean = np.asarray(batch_mean, dtype=np.float64)
        batch_var = np.asarray(batch_var, dtype=np.float64)
        if self.count <= self.epsilon:  # only the prior so far: take the first batch's mean
            self.var = (self.var * self.count + batch_var * batch_count) / (
                self.count + batch_count
            )
            self.mean = batch_mean + np.zeros_like(self.mean)
            self.count += batch_count
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        m2 = (
            self.var * self.count
            + batch_var * batch_count
            + np.square(delta) * self.count * batch_count / total
        )
        self.mean = self.mean + delta * batch_count / total
        self.var = m2 / total
        self.count = total

    @property
    def std(self) -> NDArray[np.float64]:
        return np.sqrt(self.var)

    def normalize(self, x: ArrayLike, epsilon: float = 1e-8) -> NDArray[np.float64]:
        return (np.asarray(x, dtype=np.float64) - self.mean) / np.sqrt(self.var + epsilon)

    def state_dict(self) -> dict[str, Any]:
        """The statistics as plain data, e.g. to save next to a trained policy."""
        return {"mean": self.mean.copy(), "var": self.var.copy(), "count": self.count}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64).copy()
        self.var = np.asarray(state["var"], dtype=np.float64).copy()
        self.count = float(state["count"])

    def __repr__(self) -> str:
        return f"RunningMeanStd(shape={self.mean.shape}, count={self.count:.4g})"


class ExponentialMovingStats:
    """Exponentially weighted mean and variance, a drop-in for RunningMeanStd.

    RunningMeanStd weights every sample ever seen equally, so on non-stationary
    data (volatility regimes, trending prices) its statistics drift towards a
    long-run average that no longer describes the current data. This tracks
    the recent distribution instead: a sample's weight halves every `halflife`
    samples. Batches count as `len(batch)` samples arriving together.

    Weights are bias-corrected (like pandas' `ewm(adjust=True)`): early on, before
    a halflife's worth of data has arrived, samples are weighted almost equally,
    so the first batch does not dominate the estimates.
    """

    def __init__(self, shape: tuple[int, ...] = (), halflife: float = 1000.0) -> None:
        if halflife <= 0:
            raise ValueError(f"halflife must be positive, got {halflife}.")
        self.halflife = float(halflife)
        self.mean: NDArray[np.float64] = np.zeros(shape, dtype=np.float64)
        self.var: NDArray[np.float64] = np.ones(shape, dtype=np.float64)
        self.count: float = 0.0
        self.weight: float = 0.0  # decayed total weight of all samples seen

    @property
    def decay(self) -> float:
        """Per-sample weight retained by the existing statistics."""
        return float(0.5 ** (1.0 / self.halflife))

    def update(self, batch: ArrayLike) -> None:
        """Update with a batch of samples stacked along axis 0."""
        x = np.asarray(batch, dtype=np.float64)
        if x.shape[0] == 0:
            return
        self.update_from_moments(x.mean(axis=0), x.var(axis=0), x.shape[0])

    def update_from_moments(
        self, batch_mean: ArrayLike, batch_var: ArrayLike, batch_count: int
    ) -> None:
        if batch_count <= 0:
            return
        batch_mean = np.asarray(batch_mean, dtype=np.float64)
        batch_var = np.asarray(batch_var, dtype=np.float64)
        self.weight = self.weight * self.decay**batch_count + batch_count
        w = batch_count / self.weight  # share of the new batch in the decayed total
        if self.count == 0:
            self.mean = np.array(batch_mean, dtype=np.float64)
            self.var = np.array(batch_var, dtype=np.float64)
        else:
            delta = batch_mean - self.mean
            self.mean = self.mean + w * delta
            self.var = (1.0 - w) * self.var + w * batch_var + w * (1.0 - w) * np.square(delta)
        self.count += batch_count

    @property
    def std(self) -> NDArray[np.float64]:
        out: NDArray[np.float64] = np.sqrt(self.var)
        return out

    def normalize(self, x: ArrayLike, epsilon: float = 1e-8) -> NDArray[np.float64]:
        out: NDArray[np.float64] = (np.asarray(x, dtype=np.float64) - self.mean) / np.sqrt(
            self.var + epsilon
        )
        return out

    def state_dict(self) -> dict[str, Any]:
        """The statistics as plain data, e.g. to save next to a trained policy."""
        return {
            "mean": self.mean.copy(),
            "var": self.var.copy(),
            "count": self.count,
            "halflife": self.halflife,
            "weight": self.weight,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.mean = np.asarray(state["mean"], dtype=np.float64).copy()
        self.var = np.asarray(state["var"], dtype=np.float64).copy()
        self.count = float(state["count"])
        self.halflife = float(state.get("halflife", self.halflife))
        self.weight = float(state.get("weight", self.count))

    def __repr__(self) -> str:
        return (
            f"ExponentialMovingStats(shape={self.mean.shape}, halflife={self.halflife:g}, "
            f"count={self.count:.4g})"
        )
