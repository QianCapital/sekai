from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Any, SupportsFloat

import gymnasium
import gymnasium.spaces as gspaces
import numpy as np
from numpy.typing import NDArray

from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.spaces.box import Box
from sekai.spaces.dict import Dict
from sekai.spaces.discrete import Discrete
from sekai.spaces.multi_binary import MultiBinary
from sekai.spaces.multi_discrete import MultiDiscrete
from sekai.spaces.sequence import Sequence
from sekai.spaces.space import Space
from sekai.spaces.tuple import Tuple
from sekai.utils.masking import flatten_action_mask, get_action_mask, supports_action_mask

# ---------------------------------------------------------------------------
# Spaces
# ---------------------------------------------------------------------------


def space_to_gymnasium(space: Space[Any]) -> gspaces.Space[Any]:
    """Convert a sekai space to the equivalent gymnasium space."""
    if isinstance(space, Box):
        return gspaces.Box(low=space.low, high=space.high, dtype=space.dtype)
    if isinstance(space, Discrete):
        return gspaces.Discrete(space.n, start=space.start)
    if isinstance(space, MultiDiscrete):
        return gspaces.MultiDiscrete(space.nvec, dtype=space.dtype, start=space.start)
    if isinstance(space, MultiBinary):
        return gspaces.MultiBinary(list(space.n))
    if isinstance(space, Dict):
        # A list of pairs keeps sekai's key order: gymnasium sorts the keys of a mapping
        # (gymnasium 1.0 even an OrderedDict's), which would reorder flattened observations.
        return gspaces.Dict([(k, space_to_gymnasium(s)) for k, s in space.items()])
    if isinstance(space, Tuple):
        return gspaces.Tuple([space_to_gymnasium(s) for s in space.spaces])
    if isinstance(space, Sequence):
        # gymnasium's Sequence has no max_length; the bound is dropped.
        return gspaces.Sequence(space_to_gymnasium(space.feature_space), stack=space.stack)
    raise TypeError(f"No gymnasium equivalent for sekai space {type(space).__name__}.")


def space_from_gymnasium(space: gspaces.Space[Any]) -> Space[Any]:
    """Convert a gymnasium space to the equivalent sekai space."""
    if isinstance(space, gspaces.Box):
        return Box(low=space.low, high=space.high, dtype=space.dtype)
    if isinstance(space, gspaces.Discrete):
        return Discrete(int(space.n), start=int(space.start))
    if isinstance(space, gspaces.MultiDiscrete):
        return MultiDiscrete(space.nvec, dtype=space.dtype.type, start=space.start)
    if isinstance(space, gspaces.MultiBinary):
        return MultiBinary(tuple(np.atleast_1d(space.n).tolist()))
    if isinstance(space, gspaces.Dict):
        return Dict(OrderedDict((k, space_from_gymnasium(s)) for k, s in space.spaces.items()))
    if isinstance(space, gspaces.Tuple):
        return Tuple([space_from_gymnasium(s) for s in space.spaces])
    if isinstance(space, gspaces.Sequence):
        return Sequence(space_from_gymnasium(space.feature_space), stack=space.stack)
    raise TypeError(f"No sekai equivalent for gymnasium space {type(space).__name__}.")


# ---------------------------------------------------------------------------
# sekai -> gymnasium
# ---------------------------------------------------------------------------


class SekaiToGymnasium(gymnasium.Env[Any, Any]):
    """Exposes a sekai Env through the gymnasium API.

    Use it to train sekai environments with gymnasium-based libraries
    (Stable-Baselines3, CleanRL, RLlib, ...). Pass a renderer to support
    `render()`: render_mode defaults to "rgb_array"; pass render_mode="ansi"
    explicitly for a renderer that returns text. "human" is not supported, as
    sekai renderers return frames instead of opening a window; wrap the result
    in gymnasium.wrappers.HumanRendering to display "rgb_array" frames.
    """

    RENDER_MODES = ("rgb_array", "ansi")

    def __init__(
        self,
        env: Env[Any, Any],
        renderer: Any = None,
        render_mode: str | None = None,
    ) -> None:
        self.sekai_env = env
        self.renderer = renderer
        if render_mode is not None and renderer is None:
            raise ValueError("render_mode needs a renderer.")
        if render_mode == "human":
            raise ValueError(
                "render_mode='human' is not supported: sekai renderers return frames instead "
                "of opening a window. Use render_mode='rgb_array' and wrap the env in "
                "gymnasium.wrappers.HumanRendering to display them."
            )
        if render_mode is not None and render_mode not in self.RENDER_MODES:
            raise ValueError(
                f"Unsupported render_mode {render_mode!r}; expected one of {self.RENDER_MODES}."
            )
        self.render_mode = render_mode or ("rgb_array" if renderer is not None else None)
        self.observation_space = space_to_gymnasium(env.observation_space)
        self.action_space = space_to_gymnasium(env.action_space)
        fps = env.metadata.get("render_fps")
        self.metadata = {
            "render_modes": [self.render_mode] if self.render_mode else [],
            **({"render_fps": fps} if fps is not None else {}),
        }
        if supports_action_mask(env):
            # sb3-contrib's MaskablePPO (and similar) look for an action_masks() method.
            self.action_masks: Callable[[], NDArray[np.bool_]] = self._action_masks

    def _action_masks(self) -> NDArray[np.bool_]:
        """The sekai env's action_mask(), flattened to one boolean array."""
        return flatten_action_mask(self.sekai_env.action_space, get_action_mask(self.sekai_env))

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        super().reset(seed=seed)
        result = self.sekai_env.reset(seed=seed, options=options)
        return result.observation, result.info

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        r = self.sekai_env.step(action)
        return r.observation, r.reward, bool(r.terminated), bool(r.truncated), r.info

    def render(self) -> Any:
        if self.renderer is None:
            return None
        return self.renderer.render(self.sekai_env)

    def close(self) -> None:
        if self.renderer is not None and hasattr(self.renderer, "close"):
            self.renderer.close()
        self.sekai_env.close()

    def __str__(self) -> str:
        return f"<SekaiToGymnasium{self.sekai_env}>"


# ---------------------------------------------------------------------------
# gymnasium -> sekai
# ---------------------------------------------------------------------------


class GymnasiumToSekai(Env[Any, Any]):
    """Exposes a gymnasium Env through the sekai API.

    The gymnasium env keeps its own RNG; seeds passed to reset() go straight
    to it, and self.rng is the gymnasium env's np_random. A NumPy integer seed
    is passed on as an int; a Generator becomes the gymnasium env's np_random
    (gymnasium's reset() itself only takes int seeds).
    """

    def __init__(self, env: gymnasium.Env[Any, Any]) -> None:
        self.gym_env = env
        self.observation_space = space_from_gymnasium(env.observation_space)
        self.action_space = space_from_gymnasium(env.action_space)
        self.metadata = dict(getattr(env, "metadata", {}) or {})  # type: ignore[misc]

    @property
    def rng(self) -> np.random.Generator:
        return self.gym_env.np_random

    @rng.setter
    def rng(self, generator: np.random.Generator) -> None:
        self.gym_env.np_random = generator

    def reset(self, *, seed: Any = None, options: dict[str, Any] | None = None) -> ResetResult[Any]:
        if isinstance(seed, np.random.Generator):
            # sekai's meaning: draw from this generator from now on. reset(seed=None) keeps it.
            self.gym_env.np_random = seed
            obs, info = self.gym_env.reset(seed=None, options=options)
            self._rng_seed = -1
        else:
            if isinstance(seed, np.integer):
                seed = int(seed)  # gymnasium rejects NumPy integers
            obs, info = self.gym_env.reset(seed=seed, options=options)
            if isinstance(seed, int):
                self._rng_seed = seed
        return ResetResult(observation=obs, info=dict(info))

    def step(self, action: Any) -> StepResult[Any]:
        obs, reward, terminated, truncated, info = self.gym_env.step(action)
        return StepResult(
            observation=obs,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info),
        )

    def close(self) -> None:
        self.gym_env.close()

    def __str__(self) -> str:
        return f"<GymnasiumToSekai{self.gym_env}>"


class GymnasiumRenderer:
    """A sekai Renderer that calls a wrapped gymnasium env's own render()."""

    def render(self, env: Any) -> Any:
        base = env.unwrapped if hasattr(env, "unwrapped") else env
        if not isinstance(base, GymnasiumToSekai):
            raise TypeError("GymnasiumRenderer only renders envs created by from_gymnasium().")
        return base.gym_env.render()

    def close(self) -> None:
        pass

    def __enter__(self) -> GymnasiumRenderer:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
