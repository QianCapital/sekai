from __future__ import annotations

import numpy as np
import pytest

from sekai.spaces import (
    Box,
    Dict,
    Discrete,
    MultiBinary,
    MultiDiscrete,
    Tuple,
    batch_space,
    concatenate,
    flatdim,
    flatten,
    flatten_space,
    iterate,
    unflatten,
)

SPACES = [
    Box(-1, 1, shape=(2, 3)),
    Discrete(4, start=2),
    MultiDiscrete([2, 3], start=[1, 0]),
    MultiBinary(3),
    Dict({"a": Box(0, 1, shape=(2,)), "b": Discrete(3)}),
    Tuple([Discrete(2), Box(-2, 2, shape=(1,))]),
    Dict({"nested": Dict({"x": MultiBinary(2)}), "t": Tuple([MultiDiscrete([3])])}),
]


def _equal(a: object, b: object) -> bool:
    if isinstance(a, dict):
        return all(_equal(a[k], b[k]) for k in a)  # type: ignore[index]
    if isinstance(a, tuple):
        return all(_equal(x, y) for x, y in zip(a, b))  # type: ignore[call-overload]
    return bool(np.allclose(a, b))  # type: ignore[arg-type]


@pytest.mark.parametrize("space", SPACES, ids=repr)
def test_flatten_round_trip(space: object) -> None:
    space.seed(0)  # type: ignore[attr-defined]
    x = space.sample()  # type: ignore[attr-defined]
    flat = flatten(space, x)
    assert flat.shape == (flatdim(space),) and flat.dtype == np.float32
    assert flatten_space(space).contains(flat)
    assert _equal(unflatten(space, flat), x)


def test_flatten_space_keeps_bounds() -> None:
    space = Dict({"a": Box(-3, 3, shape=(2,)), "b": Discrete(2)})
    box = flatten_space(space)
    np.testing.assert_array_equal(box.low, [-3, -3, 0, 0])
    np.testing.assert_array_equal(box.high, [3, 3, 1, 1])


@pytest.mark.parametrize("space", SPACES, ids=repr)
def test_batch_concatenate_iterate(space: object) -> None:
    space.seed(1)  # type: ignore[attr-defined]
    items = [space.sample() for _ in range(4)]  # type: ignore[attr-defined]
    batch = concatenate(space, items)
    assert batch_space(space, 4).contains(batch)
    back = list(iterate(space, batch))
    assert len(back) == 4
    for original, recovered in zip(items, back):
        assert _equal(original, recovered)


def test_iterate_accepts_list_of_samples() -> None:
    space = Dict({"a": Discrete(2)})
    samples = [{"a": 0}, {"a": 1}]
    assert list(iterate(space, samples)) == samples


def test_batch_discrete_is_multidiscrete() -> None:
    batched = batch_space(Discrete(3, start=1), 5)
    assert isinstance(batched, MultiDiscrete)
    assert batched.nvec.tolist() == [3] * 5 and batched.start.tolist() == [1] * 5
