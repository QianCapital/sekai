from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sekai._types import ActType, ObsType, SeedType
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.core.wrapper import Wrapper
from sekai.rendering.renderer import Renderer


def _image_frames(frames: Sequence[Any], fps: float, caller: str) -> list[NDArray[np.uint8]]:
    """Check fps and frames for save_gif/save_video; return the frames as uint8 arrays."""
    if not fps > 0:  # also rejects NaN
        raise ValueError(f"{caller}() needs fps > 0, got {fps!r}.")
    if isinstance(frames, np.ndarray) and (
        frames.ndim == 2 or (frames.ndim == 3 and frames.shape[-1] in (3, 4))
    ):
        # Iterating one image would save each of its rows as a frame.
        raise ValueError(
            f"{caller}() got a single frame of shape {frames.shape}; pass a list of frames "
            "or a stacked (T, H, W, C) array (or (T, H, W) for greyscale)."
        )
    if len(frames) == 0:
        raise ValueError("No frames to save.")
    images: list[NDArray[np.uint8]] = []
    for i, frame in enumerate(frames):
        if isinstance(frame, str):
            raise TypeError(
                f"Frame {i} is text, not an image (from an 'ansi' renderer?); "
                f"{caller}() needs RGB arrays of shape (H, W, 3)."
            )
        arr = np.asarray(frame)
        if not np.issubdtype(arr.dtype, np.number):
            raise TypeError(
                f"Frame {i} has dtype {arr.dtype}; {caller}() needs RGB arrays of shape (H, W, 3)."
            )
        if not (arr.ndim == 2 or (arr.ndim == 3 and arr.shape[-1] in (3, 4))):
            raise ValueError(
                f"Frame {i} has shape {arr.shape}; {caller}() needs RGB arrays of shape (H, W, 3) "
                "(or (H, W) greyscale, (H, W, 4) RGBA)."
            )
        images.append(arr.astype(np.uint8, copy=False))
    return images


def save_gif(frames: Sequence[NDArray[np.uint8]], path: str | Path, fps: float = 30) -> Path:
    """Write RGB frames to an animated GIF. Needs Pillow (pip install 'sekai[render]').

    frames: a list of (H, W, 3) arrays or one (T, H, W, 3) array; fps must be > 0.
    """
    arrays = _image_frames(frames, fps, "save_gif")
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("save_gif needs Pillow: pip install 'sekai[render]'") from e
    images = [Image.fromarray(a) for a in arrays]
    out = Path(path)
    images[0].save(
        out, save_all=True, append_images=images[1:], duration=max(int(1000 / fps), 1), loop=0
    )
    return out


def save_video(frames: Sequence[NDArray[np.uint8]], path: str | Path, fps: float = 30) -> Path:
    """Write RGB frames to a video file (e.g. .mp4). Needs imageio with ffmpeg.

    frames: a list of (H, W, 3) arrays or one (T, H, W, 3) array; fps must be > 0.
    """
    arrays = _image_frames(frames, fps, "save_video")
    try:
        import imageio.v2 as imageio
    except ImportError as e:
        raise ImportError("save_video needs imageio: pip install 'sekai[render]'") from e
    out = Path(path)
    imageio.mimsave(out, arrays, fps=fps)
    return out


class RecordFrames(Wrapper[ObsType, ActType, ObsType, ActType]):
    """Captures a frame from `renderer` after every reset() and step().

    Rendering stays outside the env: any Renderer works, and recording can be
    switched off (enabled = False) without touching the environment.

    Example:
        ```python
        env = RecordFrames(sekai.make("CartPole-v1"), CartPoleRenderer())
        ... run an episode ...
        env.save("cartpole.gif")
        ```
    """

    def __init__(
        self,
        env: Env[ObsType, ActType],
        renderer: Renderer,
        *,
        enabled: bool = True,
        max_frames: int | None = None,
    ) -> None:
        super().__init__(env)
        self.renderer = renderer
        self.enabled = enabled
        self.max_frames = max_frames
        self.frames: list[Any] = []

    def _capture(self) -> None:
        if self.enabled and (self.max_frames is None or len(self.frames) < self.max_frames):
            self.frames.append(self.renderer.render(self.env))

    def clear(self) -> None:
        self.frames = []

    def save(self, path: str | Path, fps: float | None = None) -> Path:
        """Save the captured frames as .gif (Pillow) or any video format imageio supports.

        fps defaults to the env's metadata["render_fps"] (30 if it has none). The frames
        must be RGB arrays; frames from a text ("ansi") renderer cannot be saved.
        """
        if fps is None:
            fps = self.env.metadata.get("render_fps") or 30
        if Path(path).suffix.lower() == ".gif":
            return save_gif(self.frames, path, fps)
        return save_video(self.frames, path, fps)

    def reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[ObsType]:
        result = self.env.reset(seed=seed, options=options)
        self._capture()
        return result

    async def async_reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[ObsType]:
        result = await self.env.async_reset(seed=seed, options=options)
        self._capture()
        return result

    def step(self, action: ActType) -> StepResult[ObsType]:
        result = self.env.step(action)
        self._capture()
        return result

    async def async_step(self, action: ActType) -> StepResult[ObsType]:
        result = await self.env.async_step(action)
        self._capture()
        return result

    def close(self) -> None:
        self.renderer.close()
        super().close()
