from __future__ import annotations

from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import NDArray

RGBArray: TypeAlias = NDArray[np.uint8]
ANSIString: TypeAlias = str
RenderFrame: TypeAlias = RGBArray | ANSIString


@runtime_checkable
class Renderer(Protocol):
    """Protocol for rendering backends.

    Renderers are separate from Env — they are injected and managed
    independently, so you can swap, compose, or record without touching
    the environment.

    Example usage:
        env = MyEnv()
        renderer = RGBArrayRenderer(width=800, height=600)
        with renderer:
            result = env.reset()
            for _ in range(100):
                frame = renderer.render(env)
                result = env.step(policy(result.observation))
    """

    def render(self, env: Any) -> RenderFrame:
        """Render the current state of env."""
        ...

    def close(self) -> None:
        """Release renderer resources."""
        ...

    def __enter__(self) -> Renderer:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
