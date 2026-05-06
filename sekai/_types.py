from __future__ import annotations

from typing import Any, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
WObsType = TypeVar("WObsType")
WActType = TypeVar("WActType")
AgentID = TypeVar("AgentID")
T = TypeVar("T")

SeedType: TypeAlias = int | np.random.Generator | None
InfoDict: TypeAlias = dict[str, Any]
RewardType: TypeAlias = float
RNGType: TypeAlias = np.random.Generator
RenderFrame: TypeAlias = NDArray[np.uint8] | str
