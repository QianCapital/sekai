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
    """Check fps and frames for save_gif/save_video; return the frames as uint8 arrays.

    Everything is checked before anything is written: frames must be integer images with
    values in [0, 255], all of the same shape (a cast would wrap 300 to 44 and turn float
    images in [0, 1] black; Pillow would crop or pad frames of other sizes).
    """
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
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(
                f"Frame {i} has dtype {arr.dtype}; {caller}() needs integer pixel values in "
                "[0, 255], e.g. uint8. Convert float images in [0, 1] with "
                "(frame * 255).round().astype(np.uint8)."
            )
        if arr.dtype != np.uint8 and arr.size and (np.min(arr) < 0 or np.max(arr) > 255):
            raise ValueError(
                f"Frame {i} has values in [{np.min(arr)}, {np.max(arr)}]; {caller}() needs pixel "
                "values in [0, 255]."
            )
        if images and arr.shape != images[0].shape:
            raise ValueError(
                f"Frame {i} has shape {arr.shape} but frame 0 has {images[0].shape}; "
                f"{caller}() needs frames of one size."
            )
        images.append(arr.astype(np.uint8, copy=False))
    return images


def save_gif(frames: Sequence[NDArray[np.uint8]], path: str | Path, fps: float = 30) -> Path:
    """Write RGB frames to an animated GIF. Needs Pillow (pip install 'sekai[render]').

    frames: a list of (H, W, 3) arrays or one (T, H, W, 3) array, all of one shape, with
    integer values in [0, 255]; fps must be > 0. Bad frames raise before anything is written.
    """
    arrays = _image_frames(frames, fps, "save_gif")
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("save_gif needs Pillow: pip install 'sekai[render]'") from e
    out = Path(path)
    duration = max(int(1000 / fps), 1)
    if all(a.ndim == 3 and a.shape[-1] == 3 for a in arrays):
        images = _shared_palette(arrays, Image)
        images[0].save(
            out,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=0,
            optimize=True,
        )
        return out
    images = [Image.fromarray(a) for a in arrays]
    images[0].save(out, save_all=True, append_images=images[1:], duration=duration, loop=0)
    return out


def _shared_palette(arrays: Sequence[NDArray[np.uint8]], image_module: Any) -> list[Any]:
    """RGB frames as palette images that share one palette.

    With one palette, a pixel that does not change keeps its index from frame to frame, so
    Pillow stores only the part of each frame that changed (a GIF of a chart that grows step
    by step comes out about half the size), and colours cannot flicker between frames. The
    palette comes from frames sampled across the animation; when they hold at most 256
    colours, each gets its own entry and the frames are indexed exactly (Pillow's own
    mapping onto a palette rounds near-identical colours).
    """
    sample = np.concatenate(arrays[:: max(len(arrays) // 16, 1)], axis=0)
    colors = np.unique(_pack(sample))
    if colors.size <= 256:
        rgb = np.stack([(colors >> 16) & 255, (colors >> 8) & 255, colors & 255], axis=1)
        flat = [int(v) for v in rgb.ravel()] + [0] * (768 - rgb.size)
        images = []
        for a in arrays:
            packed = _pack(a)
            index = np.minimum(np.searchsorted(colors, packed), colors.size - 1)
            if not np.array_equal(colors[index], packed):
                break  # a colour the sampled frames lack: quantise every frame instead
            image = image_module.frombytes("P", (a.shape[1], a.shape[0]), index.astype(np.uint8))
            image.putpalette(flat)
            images.append(image)
        else:
            return images
    # method 0 is median cut and dither 0 none (integers work in every Pillow >= 9).
    palette = image_module.fromarray(sample).quantize(colors=256, method=0, dither=0)
    return [image_module.fromarray(a).quantize(palette=palette, dither=0) for a in arrays]


def _pack(rgb: NDArray[np.uint8]) -> NDArray[np.uint32]:
    """Each RGB pixel as one 24-bit integer."""
    r, g, b = (rgb[..., i].astype(np.uint32) for i in range(3))
    packed: NDArray[np.uint32] = np.asarray(r * 65536 + g * 256 + b, dtype=np.uint32)
    return packed


def save_video(frames: Sequence[NDArray[np.uint8]], path: str | Path, fps: float = 30) -> Path:
    """Write RGB frames to a video file (e.g. .mp4). Needs imageio with ffmpeg.

    frames: a list of (H, W, 3) arrays or one (T, H, W, 3) array, all of one shape, with
    integer values in [0, 255]; fps must be > 0. Bad frames raise before anything is written.
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
