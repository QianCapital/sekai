from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from sekai.spaces import (
    Box,
    Dict,
    Discrete,
    MultiBinary,
    MultiDiscrete,
    Space,
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


# ---------------------------------------------------------------------------
# unflatten() returns members of the space
# ---------------------------------------------------------------------------


def test_unflatten_float_box_stays_within_bounds() -> None:
    space = Box(0, 0.1, shape=(1,), dtype=np.float64)
    x = np.array([0.1])
    out = unflatten(space, flatten(space, x))  # float32(0.1) is 0.1000000015 as float64
    assert out.dtype == np.float64 and out in space


def test_unflatten_int_box_rounds_and_clips_without_overflow() -> None:
    space = Box(-np.inf, np.inf, shape=(2,), dtype=np.int64)
    x = np.array([2**63 - 1, 2**24 + 1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # used to warn "invalid value encountered in cast"
        out = unflatten(space, flatten(space, x))
    # The int64 max survives (it used to flip sign); 2**24 + 1 is past float32's precision.
    assert out.tolist() == [2**63 - 1, 2**24] and out in space

    small = Box(0, 1000, shape=(3,), dtype=np.int64)
    assert unflatten(small, np.array([2.9999, -4.0, 1e12], np.float32)).tolist() == [3, 0, 1000]

    pixels = Box(0, 255, shape=(3,), dtype=np.uint8)
    out = unflatten(pixels, np.array([-3.4, 300.0, np.nan]))
    assert out.dtype == np.uint8 and out.tolist() == [0, 255, 0]  # NaN -> zero()
    assert unflatten(pixels, np.array([-3, 300, 7])).tolist() == [0, 255, 7]  # int input

    unsigned = Box(-np.inf, np.inf, shape=(1,), dtype=np.uint64)
    assert unflatten(unsigned, flatten(unsigned, np.array([2**64 - 1], np.uint64))) in unsigned


def test_unflatten_multibinary_thresholds() -> None:
    space = MultiBinary(4)
    out = unflatten(space, np.array([0.9, 0.2, 2.0, -1.0]))
    assert out.tolist() == [1, 0, 1, 0] and out in space


# ---------------------------------------------------------------------------
# Empty Dicts and Tuples inside batched spaces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "space",
    [
        Dict(a=Discrete(3), meta=Dict()),
        Dict(meta=Dict(), a=Box(0, 1, shape=(2,))),  # the empty part first
        Tuple([Tuple([]), Discrete(3), Dict(inner=Dict())]),
        Dict(a=Discrete(3), meta=Dict(deeper=Tuple([]))),
    ],
)
def test_iterate_with_empty_sub_composites(space: Space[Any]) -> None:
    items = [space.sample() for _ in range(4)]
    batch = concatenate(space, items)
    assert batch in batch_space(space, 4)
    rows = list(iterate(space, batch))  # used to give 0 rows
    assert len(rows) == 4 and all(row in space for row in rows)


def test_iterate_rejects_a_batch_without_any_data() -> None:
    with pytest.raises(ValueError, match="no sub-space holding data"):
        list(iterate(Dict(meta=Dict()), {"meta": {}}))
    assert list(iterate(Dict(), [{}, {}])) == [{}, {}]  # a list of samples is fine
    with pytest.raises(ValueError, match="different numbers"):
        list(iterate(Dict(a=Discrete(2), b=Discrete(2)), {"a": np.zeros(2), "b": np.zeros(3)}))


def test_sequence_stacks_elements_with_empty_parts() -> None:
    from sekai.spaces import Sequence

    seq = Sequence(Dict(meta=Dict(), a=Box(0, 1, shape=(2,))), stack=True, max_length=3, seed=0)
    sample = seq.sample(mask=(2, None))
    assert sample["a"].shape == (2, 2) and sample in seq  # used not to be contained
    assert len(seq.elements(sample)) == 2
    with pytest.raises(TypeError, match="no sub-space holding data"):
        Sequence(Dict(meta=Dict()), stack=True)  # the length could not be represented


def test_vector_env_splits_actions_with_empty_parts() -> None:
    import sekai

    class EmptyMetaAction(sekai.Env[np.ndarray, dict[str, Any]]):
        def __init__(self) -> None:
            self.observation_space = Box(-1, 1, shape=(1,))
            self.action_space = Dict(a=Discrete(3), meta=Dict())

        def reset(self, *, seed: Any = None, options: Any = None) -> sekai.ResetResult:
            self._set_rng(seed)
            return sekai.ResetResult(np.zeros(1, np.float32), {})

        def step(self, action: dict[str, Any]) -> sekai.StepResult:
            assert action in self.action_space
            return sekai.StepResult(np.zeros(1, np.float32), 0.0, False, False, {})

    vec = sekai.SyncVecEnv([EmptyMetaAction, EmptyMetaAction])
    vec.reset(seed=0)
    vec.step(vec.batch_action_space.sample())  # used to raise "got 0" actions
    vec.close()
