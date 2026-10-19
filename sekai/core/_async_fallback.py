"""Keep async methods consistent with sync overrides in subclasses.

If a subclass overrides step() (or reset()) but not async_step() (or
async_reset()), the async method it would inherit skips the new logic. The
helper below installs an async method on that subclass that runs the
overriding sync method instead.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _owner(cls: type, name: str) -> type:
    """The class in cls's MRO whose own namespace defines `name`."""
    for klass in cls.__mro__:
        if name in klass.__dict__:
            return klass
    raise AttributeError(name)


def _make_async_step(step: Callable[..., Any]) -> Callable[..., Any]:
    async def async_step(self: Any, action: Any) -> Any:
        return step(self, action)

    return async_step


def _make_async_reset(reset: Callable[..., Any]) -> Callable[..., Any]:
    async def async_reset(self: Any, *, seed: Any = None, options: Any = None) -> Any:
        return reset(self, seed=seed, options=options)

    return async_reset


def install_async_fallbacks(cls: type) -> None:
    """Route cls.async_step/async_reset through step/reset where those are more derived.

    "More derived" is decided by MRO position, so a sync method contributed by
    a mixin is honoured too. The installed method calls the sync method found
    at that position (not self.step), so a subclass that later overrides both
    methods and chains to super() does not run its own step logic twice.
    """
    mro = cls.__mro__
    for sync_name, async_name, factory in (
        ("step", "async_step", _make_async_step),
        ("reset", "async_reset", _make_async_reset),
    ):
        sync_owner = _owner(cls, sync_name)
        async_owner = _owner(cls, async_name)
        if mro.index(sync_owner) < mro.index(async_owner):
            fallback = factory(sync_owner.__dict__[sync_name])
            fallback.__name__ = async_name
            fallback.__qualname__ = f"{cls.__qualname__}.{async_name}"
            fallback.__doc__ = (
                f"Runs {sync_owner.__qualname__}.{sync_name}() (installed automatically)."
            )
            setattr(cls, async_name, fallback)
