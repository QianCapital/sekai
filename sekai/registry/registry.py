from __future__ import annotations

import dataclasses
import difflib
import importlib
from collections.abc import Callable, Sequence
from typing import Any, Literal

from sekai.error import EnvAlreadyRegistered, EnvNotFound
from sekai.registry.spec import EnvSpec, WrapperSpec

VectorizationMode = Literal["native", "sync", "async", "proc", "vector_entry_point"]


def _load_entry_point(entry_point: str) -> Callable[..., Any]:
    """Load a callable from a dotted entry point string 'module.path:ClassName'."""
    if ":" not in entry_point:
        raise ValueError(f"Invalid entry_point '{entry_point}'. Must be 'module.path:ClassName'.")
    module_path, attr = entry_point.rsplit(":", 1)
    module = importlib.import_module(module_path)
    fn: Callable[..., Any] = getattr(module, attr)
    return fn


def _make_from_spec(
    spec: EnvSpec,
    *,
    max_episode_steps: int | None = None,
    **override_kwargs: Any,
) -> Any:
    """Instantiate an environment from its spec, applying registered wrappers."""
    if spec.entry_point is None:
        raise EnvNotFound(spec.id)

    kwargs = {**spec.kwargs, **override_kwargs}
    if max_episode_steps is None:
        steps = spec.max_episode_steps
    else:
        steps = None if max_episode_steps == -1 else max_episode_steps

    if callable(spec.entry_point):
        env = spec.entry_point(**kwargs)
    else:
        env = _load_entry_point(spec.entry_point)(**kwargs)

    # The attached spec describes this exact env, so spec.make() rebuilds it faithfully.
    env.spec = dataclasses.replace(spec, kwargs=kwargs, max_episode_steps=steps)

    if steps is not None:
        from sekai.wrappers.time_limit import TimeLimit

        env = TimeLimit(env, max_episode_steps=steps)

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
        self._namespace_loaders: dict[str, Callable[[], Any]] = {}

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
        vector_entry_point: str | Callable[..., Any] | None = None,
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
            vector_entry_point=vector_entry_point,
        )
        self._specs[id] = spec
        return spec

    def deregister(self, id: str) -> None:
        """Remove an environment from the registry."""
        if id not in self._specs:
            raise EnvNotFound(id, self._suggest(id))
        del self._specs[id]

    def make(
        self,
        id: str | EnvSpec,
        *,
        max_episode_steps: int | None = None,
        **kwargs: Any,
    ) -> Any:
        spec = id if isinstance(id, EnvSpec) else self.spec(id)
        return _make_from_spec(spec, max_episode_steps=max_episode_steps, **kwargs)

    def spec(self, id: str) -> EnvSpec:
        self._maybe_load_namespace(id)
        if id not in self._specs:
            raise EnvNotFound(id, self._suggest(id))
        return self._specs[id]

    def register_namespace(self, namespace: str, loader: Callable[[], Any]) -> None:
        """Register a lazy loader for an entire namespace.

        The loader is called once when any env from that namespace is first requested.
        This enables plugin packages to defer imports until needed.

        Example:
            ```python
            sekai.register_namespace("qc", lambda: import_module("qiancapital.envs"))
            ```
        """
        self._namespace_loaders[namespace] = loader

    def _maybe_load_namespace(self, id: str) -> None:
        if "/" in id:
            namespace = id.split("/")[0]
            if namespace in self._namespace_loaders and id not in self._specs:
                # Remove the loader only once it succeeds, so a failed import can be retried.
                self._namespace_loaders[namespace]()
                del self._namespace_loaders[namespace]

    def _suggest(self, id: str) -> list[str]:
        return difflib.get_close_matches(id, self._specs.keys(), n=3, cutoff=0.6)

    def all_specs(self) -> list[EnvSpec]:
        return list(self._specs.values())

    def __contains__(self, id: object) -> bool:
        return id in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def pprint(self) -> str:
        lines = ["Registered environments:"]
        by_ns: dict[str | None, list[EnvSpec]] = {}
        for s in sorted(self._specs.values(), key=lambda s: s.id):
            by_ns.setdefault(s.namespace, []).append(s)
        for ns, specs in sorted(by_ns.items(), key=lambda kv: kv[0] or ""):
            lines.append(f"  [{ns or 'no namespace'}]")
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


def deregister(id: str) -> None:
    """Remove an environment from the global registry."""
    _registry.deregister(id)


def make(id: str | EnvSpec, *, max_episode_steps: int | None = None, **kwargs: Any) -> Any:
    """Create an environment by ID (or EnvSpec) from the global registry.

    Keyword arguments are forwarded to the environment's constructor, overriding
    the kwargs it was registered with. max_episode_steps overrides the
    registered time limit (it is not passed to the constructor); -1 removes it.
    """
    return _registry.make(id, max_episode_steps=max_episode_steps, **kwargs)


def _spawn_samplers(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keyword arguments for one sub-env, with `sampler.spawn()` for any episode sampler.

    A sampler with per-env state (a sequential one's position) spawns a fresh copy per
    sub-env, so every backend behaves as num_envs independent copies of the env (shared,
    the sub-envs of a SyncVecEnv would split the windows between them, while those of a
    ProcVecEnv, which pickle it, would each replay them). Other samplers are shared.
    """
    from sekai.finance.sampling import EpisodeSampler

    return {
        key: value.spawn() if isinstance(value, EpisodeSampler) else value
        for key, value in kwargs.items()
    }


def make_vec(
    id: str | EnvSpec,
    num_envs: int = 1,
    *,
    vectorization_mode: VectorizationMode | None = None,
    vector_kwargs: dict[str, Any] | None = None,
    wrappers: Sequence[Callable[[Any], Any]] = (),
    **env_kwargs: Any,
) -> Any:
    """Create a vectorised environment of num_envs copies of a registered env.

    Args:
        id: Registered environment ID or EnvSpec.
        num_envs: Number of sub-environments.
        vectorization_mode: "native" (the env's own batched implementation, its
            registered vector_entry_point; "vector_entry_point" is an alias),
            "sync" (SyncVecEnv), "async" (AsyncVecEnv, asyncio) or "proc"
            (ProcVecEnv, worker processes). None (the default) picks "native"
            when the env has one, else "sync".
        vector_kwargs: Extra keyword arguments for the VecEnv constructor
            (e.g. autoreset, or num_workers / shared_memory for "proc").
        wrappers: Callables applied in order to each sub-env, e.g.
            [lambda env: sekai.FlattenObservation(env)]. Not available with
            "native", which has no per-env objects; use Vec* wrappers instead.
        **env_kwargs: Forwarded to make() for every sub-env (or to the native
            implementation, which is called as `factory(num_envs=..., **kwargs)` and
            also receives `max_episode_steps` when there is a time limit). Each
            sub-env gets `sampler.spawn()` of an episode sampler (`sampler=`), so
            the sub-envs are independent copies of the env on every backend.
    """
    if num_envs < 1:
        raise ValueError(f"num_envs must be >= 1, got {num_envs}.")
    spec_ = id if isinstance(id, EnvSpec) else _registry.spec(id)

    steps_override = env_kwargs.pop("max_episode_steps", None)
    if steps_override is None:
        steps = spec_.max_episode_steps
    else:
        steps = None if steps_override == -1 else int(steps_override)

    mode = vectorization_mode
    if mode is None:
        mode = "native" if spec_.vector_entry_point is not None and not wrappers else "sync"
    if mode == "vector_entry_point":
        mode = "native"

    if mode == "native":
        if spec_.vector_entry_point is None:
            raise ValueError(
                f"{spec_.id} has no native vector implementation (vector_entry_point); "
                "use vectorization_mode='sync', 'async' or 'proc'."
            )
        if wrappers:
            raise ValueError(
                "wrappers= applies to individual sub-envs, which a native vector env does not "
                "have. Wrap the VecEnv with Vec* wrappers, or use vectorization_mode='sync'."
            )
        kwargs = {**spec_.kwargs, **env_kwargs}
        limit = {} if steps is None else {"max_episode_steps": steps}
        vec_entry = spec_.vector_entry_point
        factory = vec_entry if callable(vec_entry) else _load_entry_point(vec_entry)
        venv = factory(num_envs=num_envs, **kwargs, **limit, **(vector_kwargs or {}))
        venv.spec = dataclasses.replace(spec_, kwargs=kwargs, max_episode_steps=steps)
        return venv

    def env_fn() -> Any:
        kwargs = _spawn_samplers({**spec_.kwargs, **env_kwargs})
        env = _make_from_spec(spec_, max_episode_steps=steps_override, **kwargs)
        for wrapper in wrappers:
            env = wrapper(env)
        return env

    from sekai.vector.async_vec_env import AsyncVecEnv
    from sekai.vector.proc_vec_env import ProcVecEnv
    from sekai.vector.sync_vec_env import SyncVecEnv

    vec_classes: dict[str, Any] = {"sync": SyncVecEnv, "async": AsyncVecEnv, "proc": ProcVecEnv}
    if mode not in vec_classes:
        raise ValueError(
            f"Unknown vectorization_mode {mode!r}; expected one of {['native', *vec_classes]}."
        )
    venv = vec_classes[mode]([env_fn] * num_envs, **(vector_kwargs or {}))
    venv.spec = dataclasses.replace(
        spec_, kwargs={**spec_.kwargs, **env_kwargs}, max_episode_steps=steps
    )
    return venv


def spec(id: str) -> EnvSpec:
    """Retrieve an EnvSpec by ID from the global registry."""
    return _registry.spec(id)


def all_specs() -> list[EnvSpec]:
    """Return all registered environment specs."""
    return _registry.all_specs()


def register_namespace(namespace: str, loader: Callable[[], Any]) -> None:
    """Register a lazy namespace loader on the global registry."""
    _registry.register_namespace(namespace, loader)


def pprint_registry() -> None:
    """Print all registered environments, grouped by namespace."""
    print(_registry.pprint())
