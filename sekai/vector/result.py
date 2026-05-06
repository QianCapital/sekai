from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray

from sekai._types import InfoDict, ObsType


@dataclass(frozen=True, slots=True)
class VecStepResult(Generic[ObsType]):
    """Structured result from VecEnv.step().

    observations is the batch obs with shape (N, *obs_shape).
    infos is a list of per-env InfoDicts (not stacked numpy arrays).
    """

    observations: ObsType
    rewards: NDArray[np.float64]
    terminated: NDArray[np.bool_]
    truncated: NDArray[np.bool_]
    infos: list[InfoDict]

    @property
    def dones(self) -> NDArray[np.bool_]:
        return self.terminated | self.truncated


@dataclass(frozen=True, slots=True)
class VecResetResult(Generic[ObsType]):
    """Structured result from VecEnv.reset()."""

    observations: ObsType
    infos: list[InfoDict]


def stack_infos(infos: list[InfoDict]) -> dict[str, Any]:
    """Stack a list of per-env info dicts into {key: np.array}.

    Keys that appear in all dicts are stacked; missing keys are skipped.
    Useful when a downstream algorithm wants numpy-array info values.
    """
    if not infos:
        return {}
    keys = set(infos[0].keys())
    for info in infos[1:]:
        keys &= set(info.keys())
    return {k: np.array([info[k] for info in infos]) for k in keys}
