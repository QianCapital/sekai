"""Variable-size observations, action masks and the wrappers built on them."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import sekai
from sekai.spaces import (
    MaskedBox,
    Sequence,
    batch_space,
    concatenate,
    flatten,
    flatten_space,
    iterate,
    space_from_jsonable,
    zero_sample,
)
from sekai.utils import action_mask_space, flatten_action_mask, get_action_mask
from tests.conftest import CountingEnv, run

# ---------------------------------------------------------------------------
# Toy environments
# ---------------------------------------------------------------------------


class UniverseEnv(sekai.Env[dict[str, Any], np.ndarray]):
    """A changing set of up to 6 assets, each with 2 features; weights are masked by listing."""

    def __init__(self) -> None:
        self.observation_space = sekai.Dict(
            assets=MaskedBox(-np.inf, np.inf, shape=(2,), max_size=6),
            cash=sekai.Box(0, 1, shape=(1,)),
        )
        self.action_space = sekai.Box(0, 1, shape=(6,))
        self.listed = np.zeros(6, dtype=bool)

    def _obs(self) -> dict[str, Any]:
        slots = np.flatnonzero(self.listed)
        features = self.rng.normal(5.0, 2.0, size=(len(slots), 2))
        return {
            "assets": self.observation_space["assets"].pack(features, slots),
            "cash": np.ones(1, np.float32),
        }

    def reset(self, *, seed: Any = None, options: Any = None) -> sekai.ResetResult:
        self._set_rng(seed)
        self.listed = self.rng.random(6) < 0.5
        self.listed[0] = True
        return sekai.ResetResult(self._obs(), {})

    def step(self, action: np.ndarray) -> sekai.StepResult:
        if np.any(np.asarray(action)[~self.listed] != 0):
            raise sekai.InvalidAction(action, self.action_space)
        flip = self.rng.random(6) < 0.2
        self.listed ^= flip
        self.listed[0] = True
        return sekai.StepResult(self._obs(), 0.0, False, False, {})

    def action_mask(self) -> np.ndarray:
        return self.listed.astype(np.int8)


class MaskedDiscreteEnv(sekai.Env[np.ndarray, Any]):
    """Action 0 is only allowed on even steps; the env rejects it otherwise."""

    def __init__(self) -> None:
        self.observation_space = sekai.Box(0, 100, shape=(1,))
        self.action_space = sekai.Discrete(3)
        self.t = 0

    def reset(self, *, seed: Any = None, options: Any = None) -> sekai.ResetResult:
        self._set_rng(seed)
        self.t = 0
        return sekai.ResetResult(np.zeros(1, np.float32), {})

    def step(self, action: Any) -> sekai.StepResult:
        if int(action) == 0 and self.t % 2 == 1:
            raise sekai.InvalidAction(action, self.action_space)
        self.t += 1
        return sekai.StepResult(np.full(1, self.t, np.float32), 1.0, False, self.t >= 20, {})

    def action_mask(self) -> np.ndarray:
        return np.array([self.t % 2 == 0, 1, 1], dtype=np.int8)


class TradesEnv(sekai.Env[dict[str, Any], int]):
    """Each step prints a variable number of trades (price, size)."""

    def __init__(self, stack: bool = False) -> None:
        self.observation_space = sekai.Dict(
            trades=Sequence(sekai.Box(0, np.inf, shape=(2,)), max_length=8, stack=stack),
            position=sekai.Box(-1, 1, shape=(1,)),
        )
        self.action_space = sekai.Discrete(2)

    def _obs(self) -> dict[str, Any]:
        n = int(self.rng.integers(0, 12))
        trades = [np.abs(self.rng.normal(1, 1, 2)).astype(np.float32) for _ in range(n)]
        return {
            "trades": self.observation_space["trades"].pack(trades),
            "position": np.zeros(1, np.float32),
        }

    def reset(self, *, seed: Any = None, options: Any = None) -> sekai.ResetResult:
        self._set_rng(seed)
        return sekai.ResetResult(self._obs(), {})

    def step(self, action: int) -> sekai.StepResult:
        return sekai.StepResult(self._obs(), 0.0, False, False, {})


# ---------------------------------------------------------------------------
# Spaces
# ---------------------------------------------------------------------------


class TestSequence:
    def test_tuple_samples(self) -> None:
        space = Sequence(sekai.Discrete(4), max_length=5, seed=0)
        for _ in range(20):
            x = space.sample()
            assert isinstance(x, tuple) and len(x) <= 5 and space.contains(x)
        assert not space.contains((np.intp(1),) * 6)  # too long
        assert not space.contains((np.intp(9),))  # bad element
        assert not space.contains(np.zeros(2))  # not a tuple

    def test_stacked_samples(self) -> None:
        space = Sequence(sekai.Box(-1, 1, shape=(3,)), stack=True, seed=0)
        x = space.sample(mask=(4, None))
        assert x.shape == (4, 3) and space.contains(x)
        assert space.contains(np.zeros((0, 3), np.float32))
        assert not space.contains(np.full((2, 3), 5.0, np.float32))
        assert space.elements(x)[2].shape == (3,)

    def test_stacked_dict_elements_and_zero(self) -> None:
        space = Sequence(sekai.Dict(a=sekai.Discrete(2), b=sekai.Box(0, 1, (2,))), stack=True)
        x = space.sample(mask=(3, None))
        assert x["a"].shape == (3,) and x["b"].shape == (3, 2) and space.contains(x)
        empty = zero_sample(space)
        assert empty["b"].shape == (0, 2) and space.contains(empty)

    def test_unstackable_feature_space(self) -> None:
        with pytest.raises(TypeError):
            Sequence(Sequence(sekai.Discrete(2)), stack=True)

    def test_unbounded_lengths(self) -> None:
        space = Sequence(sekai.Discrete(2), seed=1)
        lengths = [len(space.sample()) for _ in range(400)]
        assert min(lengths) == 0 and 2.0 < np.mean(lengths) < 4.0

    def test_batching_and_json(self) -> None:
        space = Sequence(sekai.Box(0, 1, shape=(2,)), max_length=3, stack=True, seed=0)
        batched = batch_space(space, 3)
        batch = concatenate(space, [space.sample() for _ in range(3)])
        assert isinstance(batch, tuple) and batched.contains(batch)
        assert len(list(iterate(space, batch))) == 3
        assert space_from_jsonable(space.to_jsonable()) == space
        assert repr(space) == "Sequence(Box(0.0, 1.0, (2,), float32), max_length=3, stack=True)"

    def test_seeding_is_reproducible(self) -> None:
        a = Sequence(sekai.Box(0, 1, shape=(2,)), max_length=4, stack=True, seed=3)
        b = Sequence(sekai.Box(0, 1, shape=(2,)), max_length=4, stack=True, seed=3)
        for _ in range(5):
            np.testing.assert_array_equal(a.sample(), b.sample())


class TestMaskedBox:
    def test_pack_unpack(self) -> None:
        space = MaskedBox(-1, 1, shape=(2,), max_size=5)
        x = space.pack([[0.5, 0.5], [-0.5, 1.0]], slots=[3, 1])
        assert x["mask"].tolist() == [0, 1, 0, 1, 0] and space.contains(x)
        values, slots = space.unpack(x)
        assert slots.tolist() == [1, 3]
        np.testing.assert_array_equal(values, [[-0.5, 1.0], [0.5, 0.5]])

    def test_contains_rules(self) -> None:
        space = MaskedBox(0, 1, shape=(2,), max_size=3, fill_value=0.5)
        x = space.pack(np.ones((1, 2)))
        assert space.contains(x)
        bad_padding = {"values": x["values"].copy(), "mask": x["mask"]}
        bad_padding["values"][2] = 0.0  # empty slot must hold fill_value
        assert not space.contains(bad_padding)
        out_of_bounds = space.pack(np.full((1, 2), 2.0))
        assert not space.contains(out_of_bounds)
        assert not space.contains({"values": x["values"]})

    def test_validation(self) -> None:
        space = MaskedBox(0, 1, shape=(2,), max_size=2)
        with pytest.raises(ValueError):
            space.pack(np.ones((3, 2)))
        with pytest.raises(ValueError):
            space.pack(np.ones((2, 2)), slots=[0, 0])
        with pytest.raises(ValueError):
            MaskedBox(0, 1, shape=(2,), max_size=0)
        with pytest.raises(ValueError):
            MaskedBox(1, 2, shape=(2,), max_size=2, fill_value=0.0)

    def test_sample_respects_mask(self) -> None:
        space = MaskedBox(-np.inf, np.inf, shape=(2, 3), max_size=4, seed=0)
        x = space.sample(mask=np.array([1, 0, 0, 1]))
        assert x["values"].shape == (4, 2, 3) and space.contains(x)
        assert np.all(x["values"][1:3] == 0)

    def test_is_a_dict_for_every_utility(self) -> None:
        space = MaskedBox(-1, 1, shape=(2,), max_size=3, seed=0)
        x = space.sample()
        assert isinstance(space, sekai.Dict)
        assert flatten(space, x).shape == (9,) and flatten_space(space).shape == (9,)
        batch = concatenate(space, [x, space.empty()])
        assert batch_space(space, 2).contains(batch)
        assert space_from_jsonable(space.to_jsonable()) == space
        assert zero_sample(space)["mask"].sum() == 0

    def test_gymnasium_conversion(self) -> None:
        pytest.importorskip("gymnasium")
        from sekai.compat import space_from_gymnasium, space_to_gymnasium

        g = space_to_gymnasium(MaskedBox(-1, 1, shape=(2,), max_size=3))
        assert set(g.spaces) == {"values", "mask"} and g["values"].shape == (3, 2)
        seq = Sequence(sekai.Box(0, 1, shape=(2,)), stack=True)
        assert space_from_gymnasium(space_to_gymnasium(seq)) == seq


def test_box_masked_sample() -> None:
    space = sekai.Box(1, 2, shape=(4,), seed=0)
    x = space.sample(mask=np.array([1, 0, 1, 0]))
    assert x[1] == 1.0 and x[3] == 1.0  # zero clipped into [1, 2]
    assert np.all(x >= 1) and space.contains(x)
    np.testing.assert_array_equal(space.zero(), np.ones(4))


# ---------------------------------------------------------------------------
# Action masks
# ---------------------------------------------------------------------------


class TestActionMasks:
    def test_helpers(self) -> None:
        env = UniverseEnv()
        env.reset(seed=0)
        mask = get_action_mask(sekai.TimeLimit(env, 10))  # found through wrappers
        assert mask is not None and mask.tolist() == env.listed.astype(int).tolist()
        assert get_action_mask(CountingEnv()) is None
        md = sekai.MultiDiscrete([2, 3])
        flat = flatten_action_mask(md, (np.array([1, 0]), np.array([0, 1, 1])))
        assert flat.tolist() == [True, False, False, True, True]
        assert action_mask_space(md) == sekai.MultiBinary(5)

    def test_check_env_samples_with_the_mask(self) -> None:
        # Unmasked random actions would make step() raise.
        assert sekai.check_env(MaskedDiscreteEnv(), n_steps=60, check_determinism=True) == []
        assert sekai.check_env(UniverseEnv(), n_steps=60, check_determinism=True) == []

    def test_check_env_reports_bad_masks(self) -> None:
        class BadMask(MaskedDiscreteEnv):
            def action_mask(self) -> np.ndarray:
                return np.ones(7, np.int8)

        issues = sekai.check_env(BadMask())
        assert any("action_mask" in issue for issue in issues)

    def test_action_mask_observation(self) -> None:
        env = sekai.ActionMaskObservation(MaskedDiscreteEnv())
        obs = env.reset(seed=0).observation
        assert obs["action_mask"].tolist() == [1, 1, 1] and env.observation_space.contains(obs)
        obs = env.step(1).observation
        assert obs["action_mask"].tolist() == [0, 1, 1]
        dict_env = sekai.ActionMaskObservation(UniverseEnv(), key="tradable")
        assert set(dict_env.reset(seed=0).observation) == {"assets", "cash", "tradable"}

    def test_vector_call(self) -> None:
        vec = sekai.SyncVecEnv([MaskedDiscreteEnv] * 2)
        vec.reset(seed=0)
        masks = vec.call("action_mask")
        assert [m.tolist() for m in masks] == [[1, 1, 1], [1, 1, 1]]

    def test_gymnasium_action_masks(self) -> None:
        pytest.importorskip("gymnasium")
        from sekai.compat import to_gymnasium

        env = to_gymnasium(MaskedDiscreteEnv())
        env.reset(seed=0)
        env.step(1)
        assert env.action_masks().tolist() == [False, True, True]
        assert not hasattr(to_gymnasium(CountingEnv()), "action_masks")


# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------


class TestFrameStack:
    def test_dict_observations(self) -> None:
        env = sekai.FrameStackObservation(UniverseEnv(), 3)
        obs = env.reset(seed=0).observation
        assert obs["assets"]["values"].shape == (3, 6, 2) and obs["cash"].shape == (3, 1)
        assert env.observation_space.contains(obs)
        obs = env.step(np.zeros(6, np.float32)).observation
        assert env.observation_space.contains(obs)

    def test_zero_padding(self) -> None:
        env = sekai.FrameStackObservation(CountingEnv(), 3, padding="zeros")
        env.reset(seed=0)
        obs = env.step(0).observation
        assert obs[:, 0].tolist() == [0.0, 0.0, 1.0] and obs[0, 1] == 0.0

    def test_discrete_observations(self) -> None:
        class DiscreteObs(CountingEnv):
            def __init__(self) -> None:
                super().__init__()
                self.observation_space = sekai.Discrete(10)  # type: ignore[assignment]

            def _obs(self) -> Any:
                return np.intp(self.t)

        env = sekai.FrameStackObservation(DiscreteObs(), 2)
        env.reset(seed=0)
        assert env.step(0).observation.tolist() == [0, 1]
        assert env.observation_space == sekai.MultiDiscrete([10, 10])

    def test_validation(self) -> None:
        with pytest.raises(ValueError):
            sekai.FrameStackObservation(CountingEnv(), 0)
        with pytest.raises(ValueError):
            sekai.FrameStackObservation(CountingEnv(), 2, padding="nope")  # type: ignore[arg-type]


class TestNormalize:
    def test_masked_box_ignores_padding(self) -> None:
        env = sekai.NormalizeObservation(UniverseEnv())
        env.reset(seed=0)
        for _ in range(300):
            obs = env.step(np.zeros(6, np.float32)).observation
        stats = env.obs_rms["assets"]
        np.testing.assert_allclose(stats.mean, [5.0, 5.0], atol=0.5)  # zeros would pull it down
        np.testing.assert_allclose(stats.std, [2.0, 2.0], atol=0.5)
        assert env.observation_space.contains(obs)
        assert np.all(obs["assets"]["values"][obs["assets"]["mask"] == 0] == 0.0)
        assert obs["cash"].dtype == np.float64

    def test_passes_through_discrete_leaves_and_clips(self) -> None:
        from tests.conftest import DictObsEnv

        env = sekai.NormalizeObservation(DictObsEnv(), clip=0.5)
        obs = env.reset(seed=0).observation
        assert env.observation_space["b"] == sekai.Discrete(4)
        assert env.observation_space.contains(obs) and np.all(np.abs(obs["a"]) <= 0.5)

    def test_ewm_tracks_regime_changes(self) -> None:
        def final_mean(stats: str) -> float:
            env = sekai.NormalizeObservation(
                sekai.TimeLimit(CountingEnv(horizon=2000), 2000), stats=stats, halflife=50
            )
            env.reset(seed=0)
            for _ in range(1000):
                env.step(0)
            return float(env.obs_rms.mean[0])

        # obs[0] is the step count: an all-history mean lags far behind the recent values.
        assert final_mean("ewm") > 900 > final_mean("running")

    def test_state_dict_round_trip(self) -> None:
        env = sekai.NormalizeObservation(UniverseEnv(), stats="ewm", halflife=10)
        env.reset(seed=0)
        for _ in range(20):
            env.step(np.zeros(6, np.float32))
        state = env.state_dict()
        fresh = sekai.NormalizeObservation(UniverseEnv(), stats="ewm", halflife=10)
        fresh.load_state_dict(state)
        np.testing.assert_array_equal(fresh.obs_rms["assets"].mean, env.obs_rms["assets"].mean)

    def test_vector_dict_and_final_observations(self) -> None:
        vec = sekai.VecNormalizeObservation(
            sekai.SyncVecEnv([lambda: sekai.TimeLimit(UniverseEnv(), 5)] * 3)
        )
        vec.reset(seed=0)
        for _ in range(5):
            r = vec.step(np.zeros((3, 6), np.float32))
        assert r.observations["assets"]["values"].dtype == np.float64
        final = r.infos[0]["final_observation"]
        assert vec.observation_space.contains(final)


def test_delay_action() -> None:
    base = MaskedDiscreteEnv()
    env = sekai.DelayAction(base, 2, noop=np.intp(1))
    env.reset(seed=0)
    seen = []

    class Spy(sekai.ActionWrapper[Any, Any, Any]):
        def action(self, action: Any) -> Any:
            seen.append(int(action))
            return action

    env = sekai.DelayAction(Spy(MaskedDiscreteEnv()), 2, noop=np.intp(1))
    env.reset(seed=0)
    for a in (2, 2, 1, 2):
        env.step(np.intp(a))
    assert seen == [1, 1, 2, 2]
    assert [int(a) for a in env.pending_actions] == [1, 2]
    env.reset()
    assert env.pending_actions == []
    box = sekai.DelayAction(sekai.ClipAction(UniverseEnv()), 1)  # Box: noop defaults to zeros
    np.testing.assert_array_equal(box.noop, np.zeros(6))
    with pytest.raises(ValueError):
        sekai.DelayAction(MaskedDiscreteEnv(), 1)  # Discrete needs an explicit noop
    with pytest.raises(ValueError):
        sekai.DelayAction(MaskedDiscreteEnv(), 1, noop=np.intp(7))


def test_delay_action_async() -> None:
    env = sekai.DelayAction(MaskedDiscreteEnv(), 1, noop=np.intp(1))
    run(env.async_reset(seed=0))
    assert run(env.async_step(np.intp(2))).observation[0] == 1


class TestPadSequence:
    @pytest.mark.parametrize("stack", [False, True])
    def test_pads_to_masked_box(self, stack: bool) -> None:
        env = sekai.PadSequenceObservation(TradesEnv(stack=stack))
        space = env.observation_space
        assert isinstance(space["trades"], MaskedBox) and space["trades"].max_size == 8
        obs = env.reset(seed=0).observation
        for _ in range(30):
            assert space.contains(obs)
            obs = env.step(0).observation

    def test_truncation(self) -> None:
        env = sekai.PadSequenceObservation(TradesEnv(), max_length=2, truncate="tail")
        base = env.unwrapped
        assert isinstance(base, TradesEnv)
        env.reset(seed=0)
        trades = tuple(np.full(2, float(i), np.float32) for i in range(5))
        padded = env.observation({"trades": trades, "position": np.zeros(1, np.float32)})
        np.testing.assert_array_equal(padded["trades"]["values"][:, 0], [3.0, 4.0])
        strict = sekai.PadSequenceObservation(TradesEnv(), max_length=2, truncate="error")
        with pytest.raises(ValueError):
            strict.observation({"trades": trades, "position": np.zeros(1, np.float32)})

    def test_vectorises_and_flattens(self) -> None:
        vec = sekai.SyncVecEnv([lambda: sekai.PadSequenceObservation(TradesEnv())] * 2)
        batch = vec.reset(seed=0).observations
        assert batch["trades"]["values"].shape == (2, 8, 2)
        flat = sekai.FlattenObservation(sekai.PadSequenceObservation(TradesEnv()))
        assert flat.reset(seed=0).observation.shape == (8 * 2 + 8 + 1,)

    def test_requires_a_sequence(self) -> None:
        with pytest.raises(TypeError):
            sekai.PadSequenceObservation(CountingEnv())
        with pytest.raises(ValueError):
            sekai.PadSequenceObservation(
                sekai.TransformObservation(
                    CountingEnv(), lambda o: (o,), Sequence(sekai.Box(0, 1, shape=(2,)))
                )
            )
