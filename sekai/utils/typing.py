"""Shared type aliases used across the sekai package."""

from __future__ import annotations

from typing import Any

import numpy as np

# A generic observation — can be an ndarray, int, dict, tuple, etc.
ObsType = Any

# A generic action
ActType = Any

# Step return type: (observation, reward, terminated, truncated, info)
StepResult = tuple[ObsType, float, bool, bool, dict[str, Any]]

# Reset return type: (observation, info)
ResetResult = tuple[ObsType, dict[str, Any]]

# Render return type (array or None)
RenderFrame = np.ndarray | None
