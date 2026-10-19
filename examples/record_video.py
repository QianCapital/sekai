"""Render frames with a Renderer (no pygame needed) and save them as a GIF.

Needs Pillow: pip install "sekai[render]".
"""

import tempfile
from pathlib import Path

import sekai
from sekai.envs.classic_control import PendulumRenderer
from sekai.rendering import RecordFrames

env = RecordFrames(sekai.make("Pendulum-v1"), PendulumRenderer(size=200))
result = env.reset(seed=0)
while not result.done:
    result = env.step(env.action_space.sample())

path = env.save(Path(tempfile.mkdtemp()) / "pendulum.gif")
print(f"saved {len(env.frames)} frames ({env.frames[0].shape}) to {path}")
env.close()
