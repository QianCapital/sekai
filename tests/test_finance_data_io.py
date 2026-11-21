"""MarketData.save / load, memory mapping, sharing and pickling."""

from __future__ import annotations

import copy
import pickle
from pathlib import Path

import numpy as np
import pytest

import sekai
from sekai.finance import MarketData


def make(dated: bool = True) -> MarketData:
    rng = np.random.default_rng(0)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (200, 4)), axis=0))
    prices[:15, 3] = np.nan
    stamps = (
        np.busday_offset("2020-01-01", np.arange(200), roll="forward")
        if dated
        else np.arange(200) * 60.0
    )
    return MarketData(
        prices,
        timestamps=stamps,
        assets=["AAA", "BBB", 7, 8.5],
        fields={"volume": rng.lognormal(10, 1, (200, 4)), "loadings": rng.normal(size=(200, 4, 2))},
        exogenous={"rate": np.full(200, 0.01), "cov": np.broadcast_to(np.eye(2), (200, 2, 2))},
        warmup=3,
    )


def same(a: MarketData, b: MarketData) -> None:
    assert a.assets == b.assets and a.price_field == b.price_field and a.warmup == b.warmup
    np.testing.assert_array_equal(a.timestamps, b.timestamps)
    assert set(a.fields) == set(b.fields) and set(a.exogenous) == set(b.exogenous)
    for name in a.fields:
        np.testing.assert_array_equal(a.field(name), b.field(name))
    for name in a.exogenous:
        np.testing.assert_array_equal(a.exogenous_series(name), b.exogenous_series(name))
    np.testing.assert_array_equal(a.mask, b.mask)
    np.testing.assert_array_equal(a.simple_returns(), b.simple_returns())


@pytest.mark.parametrize("dated", [True, False])
@pytest.mark.parametrize("mmap", [True, False])
def test_round_trip(tmp_path: Path, dated: bool, mmap: bool) -> None:
    data = make(dated)
    assert data.save(tmp_path / "d") == tmp_path / "d"
    loaded = MarketData.load(tmp_path / "d", mmap=mmap)
    same(loaded, data)
    assert loaded.memory_mapped is mmap and not data.memory_mapped
    assert isinstance(loaded.prices, np.memmap) is mmap
    assert not loaded.prices.flags.writeable


def test_memory_mapped_splits_pickle_as_a_location(tmp_path: Path) -> None:
    data = MarketData.load(make().save(tmp_path / "d"))
    train, test = data.split(0.7, embargo=3, history=10)
    assert train.memory_mapped and test.memory_mapped and test.warmup == 10
    assert np.shares_memory(train.prices, data.prices)
    for part in (data, train, test, data[20:90], data.between(5, 50)):
        blob = pickle.dumps(part)
        assert len(blob) < 400  # the location, not the arrays
        same(pickle.loads(blob), part)
        same(copy.deepcopy(part), part)
    picked = data.select(["AAA", 7])  # columns: an in-memory copy
    assert not picked.memory_mapped
    same(pickle.loads(pickle.dumps(picked)), picked)
    assert not data.with_fields(extra=np.zeros((200, 4))).memory_mapped


def test_read_only_arrays_are_shared_not_copied() -> None:
    data = make()
    train, _ = data.split(0.5)
    assert np.shares_memory(train.prices, data.prices)
    assert np.shares_memory(train.mask, data.mask)
    owned = np.ones((10, 2))
    fresh = MarketData(owned)
    assert not np.shares_memory(fresh.prices, owned)  # writable input: copied
    owned[0, 0] = 5.0
    assert fresh.prices[0, 0] == 1.0


def test_returns_of_slices() -> None:
    data = make()
    np.testing.assert_allclose(
        data.simple_returns()[1:, 0], data.prices[1:, 0] / data.prices[:-1, 0] - 1
    )
    late = data[50:80]
    assert np.isnan(late.simple_returns()[0]).all()  # no price before the slice's first row
    np.testing.assert_array_equal(late.simple_returns()[1:], data.simple_returns()[51:80])
    np.testing.assert_array_equal(data[:30].simple_returns(), data.simple_returns()[:30])


def test_save_errors(tmp_path: Path) -> None:
    data = make()
    data.save(tmp_path / "d")
    with pytest.raises(FileExistsError, match="overwrite=True"):
        data.save(tmp_path / "d")
    data.save(tmp_path / "d", overwrite=True)
    with pytest.raises(TypeError, match="strings or numbers"):
        MarketData(np.ones((5, 1)), assets=[("a", 1)]).save(tmp_path / "e")
    with pytest.raises(ValueError, match="does not hold a saved MarketData"):
        MarketData.load(tmp_path / "nothing")


def test_validate_false_skips_value_checks() -> None:
    bad = np.ones((5, 2))
    bad[1, 1] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        MarketData(bad)
    assert MarketData(bad, validate=False).n_steps == 5
    with pytest.raises(ValueError, match="increasing"):
        MarketData(np.ones((3, 1)), timestamps=[2, 1, 3])


def test_memory_mapped_data_in_environments(tmp_path: Path) -> None:
    data = MarketData.load(make().save(tmp_path / "d"))
    kw = dict(data=data, lookback=5, episode_length=20, features=("volume",))
    for env_id in ("Market-v0", "Market-v1"):
        assert sekai.check_env(sekai.make(env_id, **kw), check_determinism=True) == []
    vec = sekai.make_vec("Market-v1", num_envs=2, vectorization_mode="proc", **kw)
    vec.reset(seed=0)
    out = vec.step(vec.batch_action_space.sample())
    assert out.observations["assets"]["values"].shape[0] == 2
    vec.close()
