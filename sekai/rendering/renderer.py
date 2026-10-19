from __future__ import annotations

from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import NDArray

RGBArray: TypeAlias = NDArray[np.uint8]
ANSIString: TypeAlias = str
RenderFrame: TypeAlias = RGBArray | ANSIString


@runtime_checkable
class Renderer(Protocol):
    """Protocol for rendering backends: any object with `render(env)` and `close()`.

    Renderers are separate from Env — they are injected and managed
    independently, so you can swap, compose, or record without touching
    the environment. Subclass `BaseRenderer` to also get a no-op `close()`
    and context-manager support.

    Example:
        ```python
        renderer = CartPoleRenderer()
        result = env.reset(seed=0)
        frames = [renderer.render(env)]
        while not result.done:
            result = env.step(env.action_space.sample())
            frames.append(renderer.render(env))
        ```
    """

    def render(self, env: Any) -> RenderFrame:
        """Render the current state of env."""
        ...

    def close(self) -> None:
        """Release renderer resources."""
        ...


class BaseRenderer:
    """Convenience base class for renderers: no-op `close()` and `with` support.

    Subclasses implement `render(env)`; override `close()` if they hold
    resources such as windows or file handles.
    """

    def render(self, env: Any) -> RenderFrame:
        raise NotImplementedError

    def close(self) -> None:
        """Release renderer resources (nothing by default)."""

    def __enter__(self) -> BaseRenderer:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
