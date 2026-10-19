"""Interoperability with gymnasium and PettingZoo (optional dependencies).

    import sekai
    from sekai.compat import from_gymnasium, from_pettingzoo, to_gymnasium, to_pettingzoo

    gym_env = to_gymnasium(sekai.make("Portfolio-v0"))  # train with SB3, CleanRL, ...
    env = from_gymnasium(gymnasium.make("LunarLander-v3"))  # use sekai wrappers/VecEnvs
    pz_env = to_pettingzoo(LimitOrderBookMarket(3))  # multi-agent libraries

Install with `pip install "sekai[gymnasium]"` / `pip install "sekai[pettingzoo]"`.
"""

from __future__ import annotations

from typing import Any


def _gym() -> Any:
    try:
        from sekai.compat import _gymnasium
    except ImportError as e:  # pragma: no cover - exercised only without gymnasium
        raise ImportError(
            "sekai.compat needs gymnasium. Install it with: pip install 'sekai[gymnasium]'"
        ) from e
    return _gymnasium


def to_gymnasium(env: Any, renderer: Any = None, render_mode: str | None = None) -> Any:
    """Wrap a sekai Env as a gymnasium.Env (spaces converted, tuples returned)."""
    return _gym().SekaiToGymnasium(env, renderer=renderer, render_mode=render_mode)


def from_gymnasium(env: Any) -> Any:
    """Wrap a gymnasium.Env as a sekai Env (StepResult/ResetResult returned)."""
    return _gym().GymnasiumToSekai(env)


def _pz() -> Any:
    try:
        from sekai.compat import _pettingzoo
    except ImportError as e:  # pragma: no cover - exercised only without pettingzoo
        raise ImportError(
            "This needs pettingzoo (and gymnasium). Install them with: "
            "pip install 'sekai[pettingzoo]'"
        ) from e
    return _pettingzoo


def to_pettingzoo(env: Any) -> Any:
    """Wrap a sekai MultiAgentEnv as a PettingZoo ParallelEnv."""
    return _pz().SekaiToPettingZoo(env)


def from_pettingzoo(env: Any) -> Any:
    """Wrap a PettingZoo ParallelEnv as a sekai MultiAgentEnv."""
    return _pz().PettingZooToSekai(env)


def space_to_gymnasium(space: Any) -> Any:
    """Convert a sekai space to the equivalent gymnasium space."""
    return _gym().space_to_gymnasium(space)


def space_from_gymnasium(space: Any) -> Any:
    """Convert a gymnasium space to the equivalent sekai space."""
    return _gym().space_from_gymnasium(space)


def __getattr__(name: str) -> Any:
    if name in ("SekaiToGymnasium", "GymnasiumToSekai", "GymnasiumRenderer"):
        return getattr(_gym(), name)
    if name in ("SekaiToPettingZoo", "PettingZooToSekai"):
        return getattr(_pz(), name)
    raise AttributeError(f"module 'sekai.compat' has no attribute {name!r}")


__all__ = [
    "to_gymnasium",
    "from_gymnasium",
    "to_pettingzoo",
    "from_pettingzoo",
    "space_to_gymnasium",
    "space_from_gymnasium",
    "SekaiToGymnasium",
    "GymnasiumToSekai",
    "GymnasiumRenderer",
    "SekaiToPettingZoo",
    "PettingZooToSekai",
]
