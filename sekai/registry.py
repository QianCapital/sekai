"""Environment registry — register and create environments by string ID."""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass, field
from typing import Any

from sekai.core.env import Env

# Regex that environment IDs must match: Name-v<integer>
_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]+-v\d+$")


@dataclass
class EnvSpec:
    """Specification for a registered environment.

    Attributes
    ----------
    id:
        Unique identifier for the environment (e.g. ``"CartPole-v1"``).
    entry_point:
        Either a callable that constructs the environment, or a dotted
        import path of the form ``"module.path:ClassName"``.
    max_episode_steps:
        Optional step limit applied by a :class:`~sekai.wrappers.TimeLimit`
        wrapper when creating via :func:`make`.
    reward_threshold:
        Optional reward threshold considered "solved" for this task.
    kwargs:
        Default keyword arguments forwarded to the env constructor.
    """

    id: str
    entry_point: str | type[Env[Any, Any]]
    max_episode_steps: int | None = None
    reward_threshold: float | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


class Registry:
    """A global registry of environment specifications.

    Users interact with the module-level helpers :func:`register`,
    :func:`make`, and :func:`spec` rather than this class directly.
    """

    def __init__(self) -> None:
        self._specs: dict[str, EnvSpec] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        id: str,  # noqa: A002
        entry_point: str | type[Env[Any, Any]],
        *,
        max_episode_steps: int | None = None,
        reward_threshold: float | None = None,
        kwargs: dict[str, Any] | None = None,
    ) -> EnvSpec:
        """Register an environment.

        Parameters
        ----------
        id:
            Unique string ID (must match ``[A-Za-z][A-Za-z0-9_.-]+-v\\d+``).
        entry_point:
            Class or import path ``"package.module:ClassName"``.
        max_episode_steps:
            Step limit for the :class:`~sekai.wrappers.TimeLimit` wrapper.
        reward_threshold:
            Reward threshold (informational only).
        kwargs:
            Default kwargs forwarded to the constructor.

        Returns
        -------
        EnvSpec
            The created spec.
        """
        if not _ID_RE.match(id):
            raise ValueError(
                f"Invalid environment ID {id!r}. "
                "IDs must match '[A-Za-z][A-Za-z0-9_.-]+-v<int>'."
            )
        if id in self._specs:
            raise ValueError(f"Environment with id {id!r} is already registered.")
        spec = EnvSpec(
            id=id,
            entry_point=entry_point,
            max_episode_steps=max_episode_steps,
            reward_threshold=reward_threshold,
            kwargs=kwargs or {},
        )
        self._specs[id] = spec
        return spec

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    def make(
        self,
        id: str,  # noqa: A002
        *,
        max_episode_steps: int | None = None,
        **kwargs: Any,
    ) -> Env[Any, Any]:
        """Create an environment from its registered ID.

        Parameters
        ----------
        id:
            Registered environment ID.
        max_episode_steps:
            Override the spec's ``max_episode_steps``.  Pass ``0`` to
            disable the time limit entirely.
        **kwargs:
            Additional keyword arguments forwarded to the env constructor
            (merged with, and overriding, the spec's defaults).

        Returns
        -------
        Env
            A new environment instance.
        """
        if id not in self._specs:
            available = ", ".join(sorted(self._specs))
            raise KeyError(
                f"No environment registered with id {id!r}. "
                f"Available environments: [{available}]"
            )
        spec = self._specs[id]
        env_cls = _load_entry_point(spec.entry_point)
        merged_kwargs = {**spec.kwargs, **kwargs}
        env = env_cls(**merged_kwargs)

        # Apply TimeLimit wrapper
        step_limit = max_episode_steps if max_episode_steps is not None else spec.max_episode_steps
        if step_limit and step_limit > 0:
            from sekai.wrappers.time_limit import TimeLimit

            env = TimeLimit(env, max_episode_steps=step_limit)

        return env

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def spec(self, id: str) -> EnvSpec:  # noqa: A002
        """Return the :class:`EnvSpec` for a registered environment."""
        if id not in self._specs:
            raise KeyError(f"No environment registered with id {id!r}")
        return self._specs[id]

    def all_specs(self) -> list[EnvSpec]:
        """Return all registered environment specs."""
        return list(self._specs.values())

    def __contains__(self, id: str) -> bool:  # noqa: A002
        return id in self._specs

    def __repr__(self) -> str:
        ids = ", ".join(sorted(self._specs))
        return f"Registry([{ids}])"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _load_entry_point(
    entry_point: str | type[Env[Any, Any]],
) -> type[Env[Any, Any]]:
    """Load an entry point class, importing it if given as a dotted string."""
    if callable(entry_point):
        return entry_point  # type: ignore[return-value]
    if isinstance(entry_point, str):
        if ":" not in entry_point:
            raise ValueError(
                f"entry_point string must be 'module.path:ClassName', got {entry_point!r}"
            )
        module_path, class_name = entry_point.rsplit(":", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls  # type: ignore[return-value]
    raise TypeError(f"entry_point must be a string or callable, got {type(entry_point)!r}")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

registry = Registry()


def register(
    id: str,  # noqa: A002
    entry_point: str | type[Env[Any, Any]],
    *,
    max_episode_steps: int | None = None,
    reward_threshold: float | None = None,
    kwargs: dict[str, Any] | None = None,
) -> EnvSpec:
    """Register an environment in the global registry.

    See :meth:`Registry.register` for parameter documentation.
    """
    return registry.register(
        id,
        entry_point,
        max_episode_steps=max_episode_steps,
        reward_threshold=reward_threshold,
        kwargs=kwargs,
    )


def make(
    id: str,  # noqa: A002
    *,
    max_episode_steps: int | None = None,
    **kwargs: Any,
) -> Env[Any, Any]:
    """Create an environment from the global registry.

    See :meth:`Registry.make` for parameter documentation.
    """
    return registry.make(id, max_episode_steps=max_episode_steps, **kwargs)


def spec(id: str) -> EnvSpec:  # noqa: A002
    """Return the :class:`EnvSpec` for a registered environment."""
    return registry.spec(id)
