from __future__ import annotations

import importlib
from typing import Any, Callable

from sekai.error import EnvAlreadyRegistered, EnvNotFound
from sekai.registry.spec import EnvSpec, WrapperSpec


def _load_entry_point(entry_point: str) -> Callable[..., Any]:
    """Load a callable from a dotted entry point string 'module.path:ClassName'."""
    if ":" not in entry_point:
        raise ValueError(
            f"Invalid entry_point '{entry_point}'. "
            "Must be 'module.path:ClassName'."
        )
    module_path, cls_name = entry_point.rsplit(":", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


def _make_from_spec(spec: EnvSpec, **override_kwargs: Any) -> Any:
    """Instantiate an environment from its spec."""
    if spec.entry_point is None:
        raise EnvNotFound(spec.id)

    kwargs = {**spec.kwargs, **override_kwargs}

    if callable(spec.entry_point) and not isinstance(spec.entry_point, str):
        env = spec.entry_point(**kwargs)
    else:
        cls = _load_entry_point(str(spec.entry_point))
        env = cls(**kwargs)

    env.spec = spec

    if spec.max_episode_steps is not None:
        from sekai.wrappers.time_limit import TimeLimit
        env = TimeLimit(env, max_episode_steps=spec.max_episode_steps)

    for wrapper_spec in spec.additional_wrappers:
        wrapper_cls = _load_entry_point(wrapper_spec.entry_point)
        env = wrapper_cls(env, **wrapper_spec.kwargs)

    return env


class EnvRegistry:
    """Registry mapping env IDs to EnvSpec objects.

    Supports lazy namespace loading via register_namespace() — useful for
    plugin packages that ship environments without importing at startup.
    """

    def __init__(self) -> None:
        self._specs: dict[str, EnvSpec] = {}
        self._namespace_loaders: dict[str, Callable[[], None]] = {}

    def register(
        self,
        id: str,
        entry_point: str | Callable[..., Any],
        *,
        reward_threshold: float | None = None,
        nondeterministic: bool = False,
        max_episode_steps: int | None = None,
        kwargs: dict[str, Any] | None = None,
        additional_wrappers: tuple[WrapperSpec, ...] = (),
    ) -> EnvSpec:
        if id in self._specs:
            raise EnvAlreadyRegistered(id)
        spec = EnvSpec(
            id=id,
            entry_point=entry_point,
            reward_threshold=reward_threshold,
            nondeterministic=nondeterministic,
            max_episode_steps=max_episode_steps,
            kwargs=kwargs or {},
            additional_wrappers=additional_wrappers,
        )
        self._specs[id] = spec
        return spec

    def make(self, id: str, **kwargs: Any) -> Any:
        spec = self.spec(id)
        return _make_from_spec(spec, **kwargs)

    def spec(self, id: str) -> EnvSpec:
        self._maybe_load_namespace(id)
        if id not in self._specs:
            raise EnvNotFound(id)
        return self._specs[id]

    def register_namespace(self, namespace: str, loader: Callable[[], None]) -> None:
        """Register a lazy loader for an entire namespace.

        The loader is called once when any env from that namespace is first requested.
        This enables plugin packages to defer imports until needed.

        Example:
            sekai.register_namespace("qc", lambda: import_module("qiancapital.envs"))
        """
        self._namespace_loaders[namespace] = loader

    def _maybe_load_namespace(self, id: str) -> None:
        if "/" in id:
            namespace = id.split("/")[0]
            if namespace in self._namespace_loaders and id not in self._specs:
                loader = self._namespace_loaders.pop(namespace)
                loader()

    def all_specs(self) -> list[EnvSpec]:
        return list(self._specs.values())

    def pprint(self) -> str:
        lines = ["Registered environments:"]
        by_ns: dict[str | None, list[EnvSpec]] = {}
        for s in sorted(self._specs.values(), key=lambda s: s.id):
            by_ns.setdefault(s.namespace, []).append(s)
        for ns, specs in sorted(by_ns.items(), key=lambda kv: kv[0] or ""):
            header = f"  [{ns or 'no namespace'}]"
            lines.append(header)
            for s in specs:
                lines.append(f"    {s.id}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"EnvRegistry({len(self._specs)} environments)"


# ---------------------------------------------------------------------------
# Global registry instance and convenience functions
# ---------------------------------------------------------------------------

_registry = EnvRegistry()


def register(
    id: str,
    entry_point: str | Callable[..., Any],
    **kwargs: Any,
) -> EnvSpec:
    """Register an environment with the global registry."""
    return _registry.register(id, entry_point, **kwargs)


def make(id: str, **kwargs: Any) -> Any:
    """Create an environment by ID from the global registry."""
    return _registry.make(id, **kwargs)


def spec(id: str) -> EnvSpec:
    """Retrieve an EnvSpec by ID from the global registry."""
    return _registry.spec(id)


def all_specs() -> list[EnvSpec]:
    """Return all registered environment specs."""
    return _registry.all_specs()


def register_namespace(namespace: str, loader: Callable[[], None]) -> None:
    """Register a lazy namespace loader on the global registry."""
    _registry.register_namespace(namespace, loader)
