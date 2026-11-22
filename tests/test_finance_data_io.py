"""MarketData.save / load, memory mapping, sharing and pickling."""

from __future__ import annotations

import copy
import os
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


def test_arrays_are_copied_unless_shared_on_purpose(tmp_path: Path) -> None:
    """A MarketData never changes once built: inputs are copied (as in 0.2), whatever
    they are; derived data shares the arrays it can, and copy=False opts in to sharing."""
    owned = np.ones((10, 2))
    fresh = MarketData(owned)
    assert not np.shares_memory(fresh.prices, owned)
    owned[0, 0] = 5.0
    assert fresh.prices[0, 0] == 1.0
    # Read-only inputs too: a pandas copy-on-write view or a memory map of the caller's.
    frozen = np.ones((10, 2))
    frozen.setflags(write=False)
    assert not np.shares_memory(MarketData(frozen).prices, frozen)
    np.save(tmp_path / "p.npy", np.full((10, 2), 100.0))
    mapped = np.load(tmp_path / "p.npy", mmap_mode="r")
    copied = MarketData(mapped, fields={"x": mapped}, exogenous={"e": mapped[:, 0]})
    np.save(tmp_path / "p.npy", np.full((10, 2), 200.0))  # the caller rewrites the file
    assert copied.prices[0, 0] == copied.field("x")[0, 0] == 100.0
    assert type(copied.prices) is np.ndarray
    # copy=False shares the caller's arrays (read-only views of them).
    mine = np.ones((10, 2))
    shared = MarketData(mine, copy=False)
    assert np.shares_memory(shared.prices, mine) and mine.flags.writeable
    assert not shared.prices.flags.writeable
    # Derived data shares this data's arrays; slices of data in memory are copies.
    data = make()
    assert np.shares_memory(data.with_fields(z=data.prices).field("volume"), data.field("volume"))
    assert np.shares_memory(
        data.with_exogenous(z=data.timestamps.astype(float)).prices, data.prices
    )
    train, _ = data.split(0.5)
    assert not np.shares_memory(train.prices, data.prices)
    assert not np.shares_memory(data[10:20].mask, data.mask)


def test_slices_are_checked() -> None:
    data = make()
    for rows in (slice(None, None, -1), slice(40, 10, -1)):
        with pytest.raises(ValueError, match="positive step"):
            data[rows]
    with pytest.raises(ValueError, match="at least 2 rows"):
        data[5:6]
    with pytest.raises(ValueError, match="at least 2 rows"):
        data.select([])
    every_other = data[::2]
    np.testing.assert_array_equal(every_other.prices, data.prices[::2])
    assert np.all(every_other.timestamps[1:] > every_other.timestamps[:-1])


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


def test_save_over_its_own_source(tmp_path: Path) -> None:
    """Saving onto the directory the data is mapped from replaces it safely."""
    root = make().save(tmp_path / "d")
    loaded = MarketData.load(root)
    before = np.array(loaded.prices)
    loaded.save(root, overwrite=True)
    np.testing.assert_array_equal(loaded.prices, before)  # still reads its (old) files
    again = MarketData.load(root)
    same(again, loaded)
    wider = again.with_fields(extra=again.prices * 2)
    wider.save(root, overwrite=True)
    np.testing.assert_array_equal(MarketData.load(root).field("extra"), before * 2)
    assert [p.name for p in tmp_path.iterdir()] == ["d"]  # no temporary directories left


def test_save_never_deletes_other_files(tmp_path: Path) -> None:
    (tmp_path / "mine").mkdir()
    (tmp_path / "mine" / "notes.txt").write_text("keep me")
    with pytest.raises(FileExistsError, match="not empty"):
        make().save(tmp_path / "mine")
    with pytest.raises(FileExistsError, match="not a saved MarketData"):
        make().save(tmp_path / "mine", overwrite=True)
    assert (tmp_path / "mine" / "notes.txt").read_text() == "keep me"
    (tmp_path / "file").write_text("x")
    with pytest.raises(FileExistsError, match="is a file"):
        make().save(tmp_path / "file", overwrite=True)


def test_pickled_slice_refuses_a_replaced_save(tmp_path: Path) -> None:
    root = make().save(tmp_path / "d")
    blob = pickle.dumps(MarketData.load(root)[10:50])
    assert len(blob) < 1000  # by location
    np.testing.assert_array_equal(pickle.loads(blob).prices, make()[10:50].prices)
    make(dated=False).save(root, overwrite=True)
    with pytest.raises(ValueError, match="replaced by another save"):
        pickle.loads(blob)


def test_save_numpy_asset_ids_and_shape_checks(tmp_path: Path) -> None:
    p = np.ones((5, 3))
    root = MarketData(p, assets=np.array([10, 20, 30])).save(tmp_path / "ids")
    assert MarketData.load(root).assets == (10, 20, 30)
    np.save(root / "mask.npy", np.ones((4, 3), dtype=bool))  # a damaged save
    with pytest.raises(ValueError, match="don't agree"):
        MarketData.load(root)


def test_save_keeps_other_files_and_links(tmp_path: Path) -> None:
    root = make().save(tmp_path / "d")
    (root / "notes.txt").write_text("mine")
    make(dated=False).save(root, overwrite=True)
    assert (root / "notes.txt").read_text() == "mine"
    link = tmp_path / "link"
    link.symlink_to(root, target_is_directory=True)
    make().save(link, overwrite=True)
    assert link.is_symlink() and (link / "notes.txt").read_text() == "mine"
    assert MarketData.load(link).n_steps == make().n_steps
    assert sorted(p.name for p in tmp_path.iterdir()) == ["d", "link"]  # nothing left behind


@pytest.mark.skipif(
    not hasattr(os, "geteuid") or os.geteuid() == 0, reason="needs file permissions"
)
def test_save_under_a_read_only_parent(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    root = make().save(parent / "d")
    parent.chmod(0o555)
    try:
        make(dated=False).save(root, overwrite=True)  # in place, file by file
        np.testing.assert_array_equal(MarketData.load(root).prices, make(dated=False).prices)
    finally:
        parent.chmod(0o755)


def test_data_mapped_from_a_replaced_save_still_pickles(tmp_path: Path) -> None:
    root = make().save(tmp_path / "d")
    loaded = MarketData.load(root)
    old = np.array(loaded.prices)
    loaded.save(root, overwrite=True)  # onto its own source
    make(dated=False).save(root, overwrite=True)  # and a different save over that
    for clone in (pickle.loads(pickle.dumps(loaded)), copy.deepcopy(loaded)):
        np.testing.assert_array_equal(clone.prices, old)  # by value now
        assert not clone.memory_mapped
