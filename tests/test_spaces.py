"""Tests for all space implementations."""

from __future__ import annotations

import numpy as np
import pytest

from sekai.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Space, Tuple


# ---------------------------------------------------------------------------
# Discrete
# ---------------------------------------------------------------------------


class TestDiscrete:
    def test_basic_sample(self):
        space = Discrete(5)
        for _ in range(50):
            s = space.sample()
            assert 0 <= s < 5

    def test_contains(self):
        space = Discrete(5)
        assert space.contains(0)
        assert space.contains(4)
        assert not space.contains(5)
        assert not space.contains(-1)

    def test_in_operator(self):
        space = Discrete(3)
        assert 2 in space
        assert 3 not in space

    def test_start_offset(self):
        space = Discrete(3, start=10)
        for _ in range(30):
            s = space.sample()
            assert 10 <= s < 13
        assert space.contains(10)
        assert space.contains(12)
        assert not space.contains(9)
        assert not space.contains(13)

    def test_repr(self):
        assert repr(Discrete(4)) == "Discrete(4)"
        assert repr(Discrete(4, start=1)) == "Discrete(4, start=1)"

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            Discrete(0)

    def test_equality(self):
        assert Discrete(3) == Discrete(3)
        assert Discrete(3) != Discrete(4)
        assert Discrete(3, start=1) != Discrete(3)

    def test_seed_reproducibility(self):
        s1 = Discrete(100, seed=42)
        s2 = Discrete(100, seed=42)
        assert [s1.sample() for _ in range(10)] == [s2.sample() for _ in range(10)]


# ---------------------------------------------------------------------------
# Box
# ---------------------------------------------------------------------------


class TestBox:
    def test_basic_sample_shape(self):
        space = Box(low=-1.0, high=1.0, shape=(3,))
        s = space.sample()
        assert s.shape == (3,)
        assert s.dtype == np.float32

    def test_contains(self):
        space = Box(low=0.0, high=1.0, shape=(2,))
        assert space.contains(np.array([0.5, 0.5], dtype=np.float32))
        assert not space.contains(np.array([1.5, 0.5], dtype=np.float32))

    def test_inferred_shape(self):
        low = np.array([0.0, -1.0], dtype=np.float32)
        high = np.array([1.0, 1.0], dtype=np.float32)
        space = Box(low, high)
        assert space.shape == (2,)

    def test_invalid_bounds(self):
        with pytest.raises(ValueError):
            Box(low=1.0, high=0.0, shape=(3,))

    def test_unbounded_sample_returns_finite(self):
        space = Box(low=-np.inf, high=np.inf, shape=(4,))
        s = space.sample()
        assert s.shape == (4,)

    def test_repr(self):
        r = repr(Box(0.0, 1.0, shape=(2,)))
        assert "Box" in r

    def test_equality(self):
        b1 = Box(0.0, 1.0, shape=(3,))
        b2 = Box(0.0, 1.0, shape=(3,))
        b3 = Box(0.0, 2.0, shape=(3,))
        assert b1 == b2
        assert b1 != b3

    def test_seed_reproducibility(self):
        s1 = Box(-1.0, 1.0, shape=(5,), seed=42)
        s2 = Box(-1.0, 1.0, shape=(5,), seed=42)
        np.testing.assert_array_equal(s1.sample(), s2.sample())


# ---------------------------------------------------------------------------
# MultiBinary
# ---------------------------------------------------------------------------


class TestMultiBinary:
    def test_sample_shape(self):
        space = MultiBinary(4)
        s = space.sample()
        assert s.shape == (4,)
        assert set(s.tolist()).issubset({0, 1})

    def test_contains(self):
        space = MultiBinary(3)
        assert space.contains(np.array([0, 1, 0], dtype=np.int8))
        assert not space.contains(np.array([0, 2, 0], dtype=np.int8))

    def test_multidim(self):
        space = MultiBinary((2, 3))
        s = space.sample()
        assert s.shape == (2, 3)

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            MultiBinary(0)

    def test_repr(self):
        assert "MultiBinary" in repr(MultiBinary(5))

    def test_equality(self):
        assert MultiBinary(4) == MultiBinary(4)
        assert MultiBinary(4) != MultiBinary(5)


# ---------------------------------------------------------------------------
# MultiDiscrete
# ---------------------------------------------------------------------------


class TestMultiDiscrete:
    def test_sample(self):
        space = MultiDiscrete([3, 4, 2])
        s = space.sample()
        assert s.shape == (3,)
        assert 0 <= s[0] < 3
        assert 0 <= s[1] < 4
        assert 0 <= s[2] < 2

    def test_contains(self):
        space = MultiDiscrete([3, 4])
        assert space.contains(np.array([2, 3], dtype=np.int64))
        assert not space.contains(np.array([3, 3], dtype=np.int64))

    def test_start_offset(self):
        space = MultiDiscrete([3, 4], start=[1, 2])
        for _ in range(20):
            s = space.sample()
            assert 1 <= s[0] < 4
            assert 2 <= s[1] < 6

    def test_invalid_nvec(self):
        with pytest.raises(ValueError):
            MultiDiscrete([3, 0, 2])

    def test_repr(self):
        assert "MultiDiscrete" in repr(MultiDiscrete([2, 3]))

    def test_equality(self):
        assert MultiDiscrete([2, 3]) == MultiDiscrete([2, 3])
        assert MultiDiscrete([2, 3]) != MultiDiscrete([2, 4])


# ---------------------------------------------------------------------------
# Tuple
# ---------------------------------------------------------------------------


class TestTuple:
    def test_sample_structure(self):
        space = Tuple((Discrete(3), Box(0.0, 1.0, shape=(2,))))
        s = space.sample()
        assert isinstance(s, tuple)
        assert len(s) == 2

    def test_contains(self):
        space = Tuple((Discrete(3), Box(0.0, 1.0, shape=(2,))))
        valid = (np.intp(1), np.array([0.5, 0.5], dtype=np.float32))
        assert space.contains(valid)
        invalid = (np.intp(5), np.array([0.5, 0.5], dtype=np.float32))
        assert not space.contains(invalid)

    def test_repr(self):
        assert "Tuple" in repr(Tuple((Discrete(2), Discrete(3))))

    def test_equality(self):
        t1 = Tuple((Discrete(2), Discrete(3)))
        t2 = Tuple((Discrete(2), Discrete(3)))
        t3 = Tuple((Discrete(2), Discrete(4)))
        assert t1 == t2
        assert t1 != t3


# ---------------------------------------------------------------------------
# Dict
# ---------------------------------------------------------------------------


class TestDict:
    def test_sample_keys(self):
        space = Dict({"obs": Box(0.0, 1.0, shape=(4,)), "mask": MultiBinary(2)})
        s = space.sample()
        assert set(s.keys()) == {"obs", "mask"}

    def test_contains(self):
        space = Dict({"a": Discrete(3), "b": Box(-1.0, 1.0, shape=(2,))})
        valid = {"a": np.intp(1), "b": np.array([0.0, 0.0], dtype=np.float32)}
        assert space.contains(valid)
        missing_key = {"a": np.intp(1)}
        assert not space.contains(missing_key)

    def test_repr(self):
        assert "Dict" in repr(Dict({"x": Discrete(2)}))

    def test_equality(self):
        d1 = Dict({"a": Discrete(2)})
        d2 = Dict({"a": Discrete(2)})
        d3 = Dict({"a": Discrete(3)})
        assert d1 == d2
        assert d1 != d3


# ---------------------------------------------------------------------------
# Generic Space seed API
# ---------------------------------------------------------------------------


class TestSeedAPI:
    @pytest.mark.parametrize(
        "space",
        [
            Discrete(5),
            Box(-1.0, 1.0, shape=(3,)),
            MultiBinary(4),
            MultiDiscrete([2, 3]),
        ],
    )
    def test_seed_returns_list_of_ints(self, space: Space):  # type: ignore[type-arg]
        result = space.seed(123)
        assert isinstance(result, list)
        assert all(isinstance(s, int) for s in result)

    def test_tuple_seed(self):
        t = Tuple((Discrete(3), Box(0.0, 1.0, shape=(2,))), seed=0)
        assert isinstance(t.seed(42), list)

    def test_dict_seed(self):
        d = Dict({"a": Discrete(3), "b": Box(0.0, 1.0, shape=(2,))}, seed=0)
        assert isinstance(d.seed(7), list)
