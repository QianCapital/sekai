from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from sekai.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Tuple, space_from_jsonable


class TestBox:
    def test_bounded_sample_in_space(self) -> None:
        box = Box(-1.0, 2.0, shape=(3, 4), seed=0)
        for _ in range(50):
            assert box.sample() in box

    def test_unbounded_and_semi_bounded(self) -> None:
        box = Box(np.array([-np.inf, 0, -np.inf]), np.array([np.inf, np.inf, 0]), seed=1)
        s = box.sample()
        assert s.shape == (3,) and s[1] >= 0 and s[2] <= 0
        assert box.is_bounded == (False, False)

    @pytest.mark.parametrize("dtype", [np.uint8, np.int8, np.int32, np.int64])
    def test_integer_box_full_dtype_range(self, dtype: type) -> None:
        info = np.iinfo(dtype)
        box = Box(info.min, info.max, shape=(64,), dtype=dtype, seed=0)
        sample = box.sample()
        assert sample.dtype == dtype and sample in box

    def test_uint8_image_box(self) -> None:
        box = Box(0, 255, shape=(8, 8, 3), dtype=np.uint8, seed=0)
        samples = np.stack([box.sample() for _ in range(20)])
        assert samples.max() > 200  # high end is reachable

    def test_contains(self) -> None:
        box = Box(0.0, 1.0, shape=(2,))
        assert box.contains(np.array([0.0, 1.0], dtype=np.float32))
        assert box.contains([0.5, 0.5])
        assert not box.contains(np.array([0.5, 1.5]))
        assert not box.contains(np.array([0.5]))
        assert not box.contains(np.array([np.nan, 0.5]))

    def test_contains_python_ints_for_narrow_integer_boxes(self) -> None:
        # Python ints arrive as int64, which is not same_kind as uint8; they are judged by value.
        u8 = Box(0, 255, shape=(2,), dtype=np.uint8)
        assert u8.contains([1, 2]) and [0, 255] in u8
        assert Box(0, 10, shape=(), dtype=np.uint8).contains(3)
        assert Box(-5, 5, shape=(2,), dtype=np.int8).contains([-5, 5])
        assert not u8.contains([1, 256]) and not u8.contains([-1, 2])
        assert not u8.contains([1.5, 2.0]) and not u8.contains([1.0, 2.0])  # floats never
        # Real arrays keep the dtype rule.
        assert not u8.contains(np.array([1, 2]))
        assert u8.contains(np.array([1, 2], dtype=np.uint8))

    def test_invalid_bounds(self) -> None:
        with pytest.raises(ValueError):
            Box(1.0, 0.0, shape=(2,))
        with pytest.raises(ValueError):
            Box(np.zeros(2), np.ones(3))

    @pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64, np.uint8])
    def test_nan_bounds_rejected(self, dtype: type) -> None:
        with pytest.raises(ValueError, match="NaN"):
            Box(np.nan, 1.0, shape=(2,), dtype=dtype)
        with pytest.raises(ValueError, match="NaN"):
            Box(np.array([0.0, 0.0]), np.array([1.0, np.nan]), dtype=dtype)

    @pytest.mark.parametrize("dtype", [np.int8, np.int32, np.int64, np.uint8, np.uint64])
    def test_infinite_integer_bounds_map_to_dtype_limits(self, dtype: type) -> None:
        # They used to be cast to garbage: Box(-inf, inf, dtype=int64) had low == high == min.
        info = np.iinfo(dtype)
        box = Box(-np.inf, np.inf, shape=(3,), dtype=dtype, seed=0)
        assert box.low.tolist() == [info.min] * 3 and box.high.tolist() == [info.max] * 3
        assert np.zeros(3, dtype=dtype) in box and box.sample() in box
        half = Box(np.array([0.0, -np.inf]), np.array([np.inf, 5.0]), dtype=dtype)
        assert half.low.tolist() == [0, info.min] and half.high.tolist() == [info.max, 5]

    def test_out_of_range_integer_bounds_are_clipped(self) -> None:
        with pytest.warns(UserWarning, match="uint8"):
            box = Box(-1, 300, shape=(2,), dtype=np.uint8)
        assert box.low.tolist() == [0, 0] and box.high.tolist() == [255, 255]
        info = np.iinfo(np.int64)  # in-range extremes are kept exactly
        low, high = np.array([info.min, info.max - 1]), np.array([info.max, info.max])
        exact = Box(low, high, dtype=np.int64)
        assert exact.low.tolist() == [info.min, info.max - 1]

    def test_python_int_bounds_beyond_int64_are_exact(self) -> None:
        # np.asarray([1, 2**63 + 1]) is a lossy float64 array.
        assert Box([0, 0], [1, 2**63 + 1], dtype=np.uint64).high.tolist() == [1, 2**63 + 1]
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no spurious "clipped" warning
            box = Box([0, 0], [1, 2**64 - 1000], dtype=np.uint64)
        assert box.high.tolist() == [1, 2**64 - 1000]

    def test_contains_python_ints_exactly(self) -> None:
        full = Box(0, 2**64 - 1, (3,), np.uint64, seed=0)
        assert full.contains(full.sample().tolist())
        assert full.contains([1, 2**63, 2**64 - 1]) and not full.contains([1, 2**64, 0])
        assert not full.contains([-1, 0, 0])
        # NumPy < 2 compared int64 with uint64 bounds as float64, letting this through.
        assert not Box(0, 2**62, (1,), np.uint64).contains([2**62 + 1])
        assert Box(-(2**63), 2**63 - 1, (2,), np.int64).contains([-(2**63), 2**63 - 1])

    def test_seeding_is_reproducible(self) -> None:
        a, b = Box(-1, 1, shape=(5,), seed=7), Box(-1, 1, shape=(5,), seed=7)
        np.testing.assert_array_equal(a.sample(), b.sample())

    def test_equality_and_repr(self) -> None:
        assert Box(0, 1, shape=(2,)) == Box(0, 1, shape=(2,))
        assert Box(0, 1, shape=(2,)) != Box(0, 2, shape=(2,))
        assert "Box" in repr(Box(0, 1, shape=(2,)))


class TestDiscrete:
    def test_sample_and_contains(self) -> None:
        d = Discrete(5, start=-2, seed=0)
        for _ in range(50):
            assert d.sample() in d
        assert d.contains(-2) and d.contains(2) and not d.contains(3)

    def test_accepts_numpy_scalars_and_0d_arrays(self) -> None:
        d = Discrete(3)
        assert d.contains(np.int64(2))
        assert d.contains(np.array(1))
        assert not d.contains(np.array([1]))
        assert not d.contains(1.0)
        assert not d.contains(True)

    def test_mask(self) -> None:
        d = Discrete(4, seed=0)
        mask = np.array([0, 1, 0, 1], dtype=np.int8)
        assert {int(d.sample(mask)) for _ in range(50)} == {1, 3}
        with pytest.raises(ValueError):
            d.sample(np.zeros(4, dtype=np.int8))

    def test_invalid_n(self) -> None:
        with pytest.raises(ValueError):
            Discrete(0)

    def test_numpy_integer_n_and_list_mask(self) -> None:
        d = Discrete(np.int64(3), seed=0)
        assert type(d.n) is int and d == Discrete(3)
        assert {int(d.sample([0, 1, 0])) for _ in range(10)} == {1}


class TestMultiDiscrete:
    def test_sample_respects_start_and_dtype(self) -> None:
        md = MultiDiscrete([3, 3], start=[-1, 5], dtype=np.int32, seed=0)
        s = md.sample()
        assert s.dtype == np.int32 and s in md
        assert -1 <= s[0] <= 1 and 5 <= s[1] <= 7

    def test_contains_accepts_lists(self) -> None:
        md = MultiDiscrete([2, 4])
        assert md.contains([1, 3])
        assert not md.contains([2, 0])

    def test_repr_mentions_start(self) -> None:
        assert "start" in repr(MultiDiscrete([2], start=[1]))


class TestMultiBinary:
    def test_contains_any_integer_dtype(self) -> None:
        mb = MultiBinary(3)
        assert mb.contains(np.array([0, 1, 1]))
        assert mb.contains([1, 0, 0])
        assert mb.contains(np.array([True, False, True]))
        assert not mb.contains(np.array([0, 2, 1]))
        assert not mb.contains(np.array([0.0, 1.0, 1.0]))

    def test_mask(self) -> None:
        mb = MultiBinary(4, seed=0)
        s = mb.sample(np.array([1, 0, -1, -1], dtype=np.int8))
        assert s[0] == 1 and s[1] == 0

    def test_numpy_integer_n_and_list_mask(self) -> None:
        mb = MultiBinary(np.int64(3), seed=0)
        assert mb.n == (3,) and mb == MultiBinary(3)
        assert space_from_jsonable(json.loads(json.dumps(mb.to_jsonable()))) == mb
        grid = MultiBinary((np.int64(2), np.int32(3)))
        assert grid.n == (2, 3) and all(type(d) is int for d in grid.n)
        assert mb.sample([1, 0, 1]).tolist() == [1, 0, 1]


class TestComposite:
    def test_dict(self) -> None:
        d = Dict({"a": Box(0, 1, shape=(2,)), "b": Discrete(3)}, seed=0)
        s = d.sample()
        assert set(s) == {"a", "b"} and s in d
        assert not d.contains({"a": s["a"]})
        assert len(d) == 2 and d["b"] == Discrete(3)

    def test_dict_kwargs_constructor(self) -> None:
        d = Dict(x=Discrete(2), y=Discrete(3))
        assert list(d.keys()) == ["x", "y"]

    def test_dict_seed_reproducible(self) -> None:
        a = Dict({"a": Box(0, 1, shape=(2,)), "b": Discrete(9)}, seed=3)
        b = Dict({"a": Box(0, 1, shape=(2,)), "b": Discrete(9)}, seed=3)
        sa, sb = a.sample(), b.sample()
        np.testing.assert_array_equal(sa["a"], sb["a"])
        assert sa["b"] == sb["b"]

    def test_tuple(self) -> None:
        t = Tuple([Box(0, 1, shape=(2,)), MultiBinary(3)], seed=0)
        s = t.sample()
        assert len(s) == 2 and s in t
        assert not t.contains((s[0],))

    def test_tuple_mask_needs_one_entry_per_sub_space(self) -> None:
        t = Tuple([Discrete(2), Discrete(3)], seed=0)
        with pytest.raises(ValueError, match="one mask per sub-space"):
            t.sample(mask=(np.array([0, 1], np.int8),))  # used to return a 1-tuple
        s = t.sample(mask=(np.array([0, 1], np.int8), None))
        assert s[0] == 1 and s in t


@pytest.mark.parametrize(
    "space",
    [
        Box(-1, 1, shape=(2, 3)),
        Box(0, 255, shape=(4,), dtype=np.uint8),
        Discrete(4, start=1),
        MultiDiscrete([2, 3], start=[0, -1]),
        MultiBinary((2, 2)),
        Dict({"a": Box(0, 1, shape=(1,)), "b": Tuple([Discrete(2), MultiBinary(2)])}),
    ],
)
def test_json_round_trip(space: object) -> None:
    rebuilt = space_from_jsonable(space.to_jsonable())  # type: ignore[attr-defined]
    assert rebuilt == space


def test_unknown_json_type() -> None:
    with pytest.raises(ValueError, match="Unknown space type"):
        space_from_jsonable({"type": "Nope"})


@pytest.mark.parametrize(
    "make",
    [lambda s: Dict({"a": Discrete(1000)}, seed=s), lambda s: Tuple([Discrete(1000)], seed=s)],
)
def test_composite_seeded_with_generator(make: object) -> None:
    # Used to crash with int(Generator); sub-spaces now get seeds drawn from the generator.
    a = make(np.random.default_rng(0))  # type: ignore[operator]
    b = make(np.random.default_rng(0))  # type: ignore[operator]
    assert [a.sample() for _ in range(5)] == [b.sample() for _ in range(5)]


class TestMultiDiscreteMask:
    def test_tuple_mask(self) -> None:
        md = MultiDiscrete([3, 2], start=[10, 0], seed=0)
        mask = (np.array([0, 1, 0], np.int8), np.array([1, 0], np.int8))
        assert all(md.sample(mask).tolist() == [11, 0] for _ in range(10))

    def test_padded_array_mask(self) -> None:
        md = MultiDiscrete([3, 2], seed=0)
        mask = np.array([[0, 0, 1], [0, 1, 0]], np.int8)
        assert md.sample(mask).tolist() == [2, 1]

    def test_nested_tuple_mask_for_2d_nvec(self) -> None:
        md = MultiDiscrete([[2, 2], [2, 2]], seed=0)
        one = np.array([0, 1], np.int8)
        assert md.sample(((one, one), (one, one))).tolist() == [[1, 1], [1, 1]]

    def test_fully_masked_dimension(self) -> None:
        with pytest.raises(ValueError, match="All actions masked"):
            MultiDiscrete([2]).sample((np.zeros(2, np.int8),))


def test_box_repr_is_readable() -> None:
    assert repr(Box(-1.2, 0.6, shape=(2,))) == "Box(-1.2, 0.6, (2,), float32)"
    assert repr(Box(0, 255, shape=(3,), dtype=np.uint8)) == "Box(0, 255, (3,), uint8)"
    assert repr(Box(np.array([-1.0, -np.inf]), np.array([1.0, np.inf]))) == (
        "Box([-1.0, -inf], [1.0, inf], (2,), float32)"
    )
    assert "..." in repr(Box(np.zeros(100), np.arange(1, 101)))


# ---------------------------------------------------------------------------
# Regressions found in the 0.2.0 pre-release review
# ---------------------------------------------------------------------------


def test_numpy_ints_in_shapes_become_python_ints() -> None:
    from sekai.spaces import MaskedBox, Sequence

    box = Box(0, 1, shape=(np.int64(3),))
    assert repr(box) == "Box(0.0, 1.0, (3,), float32)"
    assert all(type(d) is int for d in box.shape)
    masked = MaskedBox(0, 1, shape=(np.int64(2),), max_size=np.int64(4))
    assert all(type(d) is int for d in masked.element_shape)
    seq = Sequence(box, max_length=np.int64(5))
    for space in (box, masked, seq):  # json.dumps used to raise on np.int64
        assert space_from_jsonable(json.loads(json.dumps(space.to_jsonable()))) == space


class TestPerSubspaceSeeds:
    def test_dict_seed_accepts_a_dict(self) -> None:
        a = Dict(x=Discrete(100), y=Box(0, 1, shape=(2,)))
        b = Dict(x=Discrete(100), y=Box(0, 1, shape=(2,)), seed={"x": 5})
        assert a.seed({"x": 5}) == -1  # used to raise TypeError
        assert [a["x"].sample() for _ in range(5)] == [b["x"].sample() for _ in range(5)]
        with pytest.raises(ValueError, match="not keys"):
            a.seed({"z": 1})

    def test_tuple_seed_accepts_a_list(self) -> None:
        a = Tuple([Discrete(100), Discrete(100)])
        b = Tuple([Discrete(100), Discrete(100)], seed=[1, 2])
        assert a.seed([1, 2]) == -1  # used to raise TypeError
        assert a.sample() == b.sample()
        a.seed([3, None])  # None leaves sub-space 1's generator where it is
        b[0].seed(3)
        assert a.sample() == b.sample()
        with pytest.raises(ValueError, match="3 seeds"):
            a.seed([1, 2, 3])

    def test_int_seed_still_seeds_everything(self) -> None:
        a, b = Dict(x=Discrete(100)), Dict(x=Discrete(100))
        assert a.seed(7) == b.seed(7) == 7
        assert a.sample() == b.sample()


class TestEquality:
    def test_custom_spaces_compare_by_identity(self) -> None:
        from sekai.spaces import Sequence, Space

        class Level(Space[int]):
            def __init__(self, n: int) -> None:
                self.n = n
                super().__init__()

            def sample(self, mask: object = None) -> int:
                return self.n

            def contains(self, x: object) -> bool:
                return x == self.n

            @property
            def is_flattenable(self) -> bool:
                return False

        one = Level(1)
        assert one == one and Level(1) != Level(2)  # used to be equal (same type)
        assert Sequence(Level(1)) != Sequence(Level(2))
        assert Sequence(one) == Sequence(one)
        assert len({one, one}) == 1  # identity-hashable, consistent with ==

    def test_builtin_spaces_compare_by_value_and_are_unhashable(self) -> None:
        from sekai.spaces import MaskedBox, Sequence

        def make() -> list[object]:
            return [
                Box(0, 1, shape=(2,)),
                Discrete(3),
                MultiDiscrete([2, 3]),
                MultiBinary(4),
                Dict(a=Discrete(2)),
                Tuple([Discrete(2)]),
                Sequence(Discrete(2), max_length=3),
                MaskedBox(0, 1, shape=(2,), max_size=3),
            ]

        for a, b in zip(make(), make()):
            assert a == b and a is not b
            assert type(a).__hash__ is None, type(a).__name__

    def test_multi_discrete_compares_dtype(self) -> None:
        assert MultiDiscrete([2, 3], dtype=np.int32) != MultiDiscrete([2, 3])
        assert MultiDiscrete([2, 3], dtype=np.int32) == MultiDiscrete([2, 3], dtype=np.int32)


@pytest.mark.parametrize(
    ("space", "x"),
    [
        (MultiBinary(2), [[0, 1], [1]]),
        (MultiDiscrete([3, 3]), [[1, 2], [1]]),
        (MultiDiscrete([3, 3]), ([0], [1, 2])),
    ],
)
def test_contains_is_false_for_ragged_input(space: object, x: object) -> None:
    assert space.contains(x) is False  # type: ignore[attr-defined]


def test_masked_box_contains_ragged_values_is_false() -> None:
    from sekai.spaces import MaskedBox

    space = MaskedBox(0, 1, shape=(2,), max_size=2)
    assert not space.contains({"values": [[0, 0], [0]], "mask": np.array([1, 0])})


class TestMultiDiscreteMasks:
    def test_wrong_length_entries_raise(self) -> None:
        space = MultiDiscrete([6, 6, 3], seed=0)
        ok = (np.ones(6, np.int8), np.ones(6, np.int8), np.ones(3, np.int8))
        assert space.sample(mask=ok) in space
        for bad in (
            (np.ones(6, np.int8), np.ones(6, np.int8), np.ones(2, np.int8)),  # too short
            (np.ones(6, np.int8), np.ones(6, np.int8), np.ones(4, np.int8)),  # too long
            ok[:2],  # an entry missing
            (*ok, np.ones(3, np.int8)),  # an extra entry
        ):
            with pytest.raises(ValueError):
                space.sample(mask=bad)

    def test_padded_array_masks(self) -> None:
        space = MultiDiscrete([2, 3], seed=0)
        padded = np.array([[0, 1, 1], [1, 0, 0]], np.int8)  # the 1 past nvec[0] is padding
        for _ in range(10):
            assert space.sample(mask=padded).tolist() == [1, 0]
        with pytest.raises(ValueError, match=r"shape \(2, 3\)"):
            space.sample(mask=np.ones((2, 2), np.int8))  # narrower than max(nvec)

    def test_nested_masks_for_2d_nvec(self) -> None:
        space = MultiDiscrete([[2, 3], [2, 2]], seed=0)
        mask = ((np.array([1, 0]), np.array([0, 0, 1])), (np.array([0, 1]), np.array([1, 0])))
        assert space.sample(mask=mask).tolist() == [[0, 2], [1, 0]]
        with pytest.raises(ValueError, match=r"dimension \(0, 1\)"):
            space.sample(mask=((np.array([1, 0]), np.array([0, 1])), mask[1]))
