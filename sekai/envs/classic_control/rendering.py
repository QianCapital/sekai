"""RGB renderers for the classic control envs, drawn with sekai.rendering.Canvas.

Each renderer takes the env (wrapped or not) and returns an (H, W, 3) uint8
frame, mirroring gymnasium's rgb_array output:

    renderer = CartPoleRenderer()
    frame = renderer.render(env)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sekai.envs.classic_control.acrobot import AcrobotEnv
from sekai.envs.classic_control.cartpole import CartPoleEnv
from sekai.envs.classic_control.continuous_mountain_car import ContinuousMountainCarEnv
from sekai.envs.classic_control.mountain_car import MountainCarEnv
from sekai.envs.classic_control.pendulum import PendulumEnv
from sekai.error import ResetRequired
from sekai.rendering.canvas import Canvas, rotate
from sekai.rendering.renderer import BaseRenderer


def _base(env: Any, expected: type | tuple[type, ...]) -> Any:
    base = env.unwrapped if hasattr(env, "unwrapped") else env
    if not isinstance(base, expected):
        names = (
            expected.__name__
            if isinstance(expected, type)
            else "/".join(t.__name__ for t in expected)
        )
        raise TypeError(f"This renderer draws {names}, got {type(base).__name__}.")
    if getattr(base, "state", None) is None:
        raise ResetRequired()
    return base


class CartPoleRenderer(BaseRenderer):
    def __init__(self, width: int = 600, height: int = 400) -> None:
        self.width = width
        self.height = height

    def render(self, env: Any) -> NDArray[np.uint8]:
        cp: CartPoleEnv = _base(env, CartPoleEnv)
        x, _, theta, _ = cp.state  # type: ignore[misc]
        canvas = Canvas(self.width, self.height)
        scale = self.width / (cp.x_threshold * 2)
        pole_w, pole_len = 10.0, scale * (2 * cp.length)
        cart_w, cart_h = 50.0, 30.0
        cart_x = x * scale + self.width / 2.0
        cart_y = 100.0

        canvas.line((0, cart_y), (self.width, cart_y), (0, 0, 0))
        canvas.rect(
            cart_x - cart_w / 2,
            cart_y - cart_h / 2,
            cart_x + cart_w / 2,
            cart_y + cart_h / 2,
            (0, 0, 0),
        )
        axle_y = cart_y + cart_h / 4
        pole = [
            (-pole_w / 2, -pole_w / 2),
            (-pole_w / 2, pole_len - pole_w / 2),
            (pole_w / 2, pole_len - pole_w / 2),
            (pole_w / 2, -pole_w / 2),
        ]
        canvas.polygon(rotate(pole, -theta, (cart_x, axle_y)), (202, 152, 101))
        canvas.circle(cart_x, axle_y, pole_w / 2, (129, 132, 203))
        return canvas.to_array()


class MountainCarRenderer(BaseRenderer):
    """Draws MountainCar-v0 and MountainCarContinuous-v0."""

    def __init__(self, width: int = 600, height: int = 400) -> None:
        self.width = width
        self.height = height

    def render(self, env: Any) -> NDArray[np.uint8]:
        mc = _base(env, (MountainCarEnv, ContinuousMountainCarEnv))
        pos = float(mc.state[0])
        canvas = Canvas(self.width, self.height)
        scale = self.width / (mc.max_position - mc.min_position)

        xs = np.linspace(mc.min_position, mc.max_position, 100)
        track = np.column_stack([(xs - mc.min_position) * scale, mc.height(xs) * scale])
        canvas.polyline(track, (0, 0, 0), width=2)

        clearance = 10.0
        car_w, car_h = 40.0, 20.0
        origin = ((pos - mc.min_position) * scale, mc.height(pos) * scale + clearance)
        angle = math.cos(3 * pos)
        body = [(-car_w / 2, 0), (-car_w / 2, car_h), (car_w / 2, car_h), (car_w / 2, 0)]
        canvas.polygon(rotate(body, angle, origin), (0, 0, 0))
        for wx in (car_w / 4, -car_w / 4):
            cx, cy = rotate([(wx, 0.0)], angle, origin)[0]
            canvas.circle(cx, cy, car_h / 2.5, (128, 128, 128))

        flag_x = (mc.goal_position - mc.min_position) * scale
        flag_y = mc.height(mc.goal_position) * scale
        canvas.line((flag_x, flag_y), (flag_x, flag_y + 50), (0, 0, 0), width=2)
        canvas.polygon(
            [(flag_x, flag_y + 50), (flag_x, flag_y + 40), (flag_x + 25, flag_y + 45)],
            (204, 204, 0),
        )
        return canvas.to_array()


class PendulumRenderer(BaseRenderer):
    def __init__(self, size: int = 500) -> None:
        self.size = size

    def render(self, env: Any) -> NDArray[np.uint8]:
        pd: PendulumEnv = _base(env, PendulumEnv)
        theta = float(pd.state[0])  # type: ignore[index]
        canvas = Canvas(self.size, self.size)
        scale = self.size / (2 * 2.2)
        center = (self.size / 2, self.size / 2)
        rod_len, rod_w = 1.0 * scale, 0.2 * scale
        rod = [(0, -rod_w / 2), (0, rod_w / 2), (rod_len, rod_w / 2), (rod_len, -rod_w / 2)]
        canvas.polygon(rotate(rod, theta + np.pi / 2, center), (204, 77, 77))
        canvas.circle(center[0], center[1], rod_w / 2, (204, 77, 77))
        tip = rotate([(rod_len, 0.0)], theta + np.pi / 2, center)[0]
        canvas.circle(tip[0], tip[1], rod_w / 2, (204, 77, 77))
        canvas.circle(center[0], center[1], 0.05 * scale, (0, 0, 0))
        return canvas.to_array()


class AcrobotRenderer(BaseRenderer):
    def __init__(self, size: int = 500) -> None:
        self.size = size

    def render(self, env: Any) -> NDArray[np.uint8]:
        ab: AcrobotEnv = _base(env, AcrobotEnv)
        s = ab.state
        assert s is not None
        canvas = Canvas(self.size, self.size)
        bound = ab.LINK_LENGTH_1 + ab.LINK_LENGTH_2 + 0.2
        scale = self.size / (bound * 2)
        offset = self.size / 2

        # Same construction as gymnasium: joint positions, then a rectangle per link.
        p1 = (-ab.LINK_LENGTH_1 * math.cos(s[0]) * scale, ab.LINK_LENGTH_1 * math.sin(s[0]) * scale)
        p2 = (
            p1[0] - ab.LINK_LENGTH_2 * math.cos(s[0] + s[1]) * scale,
            p1[1] + ab.LINK_LENGTH_2 * math.sin(s[0] + s[1]) * scale,
        )
        joints = np.array([[0.0, 0.0], p1, p2])[:, ::-1]
        thetas = [s[0] - math.pi / 2, s[0] + s[1] - math.pi / 2]
        lengths = [ab.LINK_LENGTH_1 * scale, ab.LINK_LENGTH_2 * scale]

        canvas.line(
            (offset - bound * scale, offset + scale),
            (offset + bound * scale, offset + scale),
            (0, 0, 0),
        )
        half = 0.1 * scale
        for (jx, jy), th, length in zip(joints[:2], thetas, lengths):
            link = [(0, -half), (0, half), (length, half), (length, -half)]
            canvas.polygon(rotate(link, th, (jx + offset, jy + offset)), (0, 204, 204))
            canvas.circle(jx + offset, jy + offset, half, (204, 204, 0))
        return canvas.to_array()


__all__ = ["AcrobotRenderer", "CartPoleRenderer", "MountainCarRenderer", "PendulumRenderer"]
