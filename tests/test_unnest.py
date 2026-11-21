from __future__ import annotations

import pickle
import warnings
from typing import Any

import numpy as np
import pytest

import sekai
from sekai.spaces import concatenate
from sekai.utils.checker import _equal
from sekai.wrappers.unnest_obs import nest, unnest, unnest_space
from tests.conftest import AsyncOnlyEnv, run

NESTED = sekai.Dict(
    {
        "assets": sekai.MaskedBox(-1.0, 1.0, shape=(4, 2), max_size=3),
        "book": sekai.Tuple([sekai.Box(0, 1, shape=(2,)), sekai.Dict({"side": sekai.Discrete(3)})]),
        "empty": sekai.Dict(),
        "cash": sekai.Box(0, 1, shape=(1,)),
        "flags": sekai.MultiBinary(2),
    }
)
KEYS = [
    "assets/values",
    "assets/mask",
    "book/0",
    "book/1/side",
    "cash",
    "flags",
]


class TestSpaces:
    def test_keys_and_leaves(self) -> None:
        flat = unnest_space(NESTED)
        assert list(flat.keys()) == KEYS
        assert flat["assets/values"] == sekai.Box(-1.0, 1.0, shape=(3, 4, 2))
        assert flat["assets/mask"] == sekai.MultiBinary(3)
        assert flat["book/1/side"] == sekai.Discrete(3)
        assert not any(isinstance(sub, (sekai.Dict, sekai.Tuple)) for sub in flat.values())

    def test_tuple_at_the_top_and_custom_separator(self) -> None:
        space = sekai.Tuple([sekai.Discrete(2), sekai.Dict({"a": sekai.Box(0, 1, shape=(1,))})])
        assert list(unnest_space(space).keys()) == ["0", "1/a"]
        assert list(unnest_space(NESTED, sep=".").keys())[:2] == ["assets.values", "assets.mask"]

    def test_flat_space_keeps_its_keys(self) -> None:
        space = sekai.Dict({"b": sekai.Discrete(2), "a": sekai.Box(0, 1, shape=(2,))})
        assert unnest_space(space) == space

    def test_masked_box_at_the_top(self) -> None:
        space = sekai.MaskedBox(0.0, 1.0, shape=(2,), max_size=4)
        x = space.sample()
        assert list(unnest_space(space).keys()) == ["values", "mask"]
        assert _equal(nest(space, unnest(space, x)), x)

    def test_errors(self) -> None:
        with pytest.raises(TypeError, match="Dict or Tuple"):
            unnest_space(sekai.Box(0, 1, shape=(2,)))
        with pytest.raises(ValueError, match="sep"):
            unnest_space(NESTED, sep="")
        clash = sekai.Dict({"a/b": sekai.Discrete(2), "a": sekai.Dict({"b": sekai.Discrete(3)})})
        with pytest.raises(ValueError, match=r"\['a/b'\]"):
            unnest_space(clash)
        assert list(unnest_space(clash, sep=".").keys()) == ["a/b", "a.b"]


class TestRoundTrip:
    @pytest.mark.parametrize("seed", range(5))
    def test_samples(self, seed: int) -> None:
        NESTED.seed(seed)
        x = NESTED.sample()
        flat = unnest(NESTED, x)
        assert list(flat) == KEYS
        assert unnest_space(NESTED).contains(flat)
        back = nest(NESTED, flat)
        assert _equal(back, x)
        assert list(back) == list(x) and isinstance(back["book"], tuple)
        assert back["empty"] == {}
        assert flat["assets/values"] is x["assets"]["values"]  # leaves are not copied
        assert _equal(unnest(NESTED, nest(NESTED, flat)), flat)

    def test_batches(self) -> None:
        NESTED.seed(0)
        samples = [NESTED.sample() for _ in range(3)]
        batch = concatenate(NESTED, samples)
        flat_batch = concatenate(unnest_space(NESTED), [unnest(NESTED, x) for x in samples])
        assert _equal(unnest(NESTED, batch), flat_batch)
        assert _equal(nest(NESTED, flat_batch), batch)

    def test_nest_ignores_extra_keys_and_needs_every_leaf(self) -> None:
        x = NESTED.sample()
        flat = unnest(NESTED, x)
        assert _equal(nest(NESTED, {**flat, "action_mask": np.ones(2)}), x)
        del flat["cash"]
        with pytest.raises(KeyError, match="cash"):
            nest(NESTED, flat)


class TestWrapper:
    @pytest.mark.parametrize(
        ("env_id", "keys"),
        [
            ("Market-v0", ["assets/values", "assets/mask", "portfolio", "account"]),
            ("LimitOrderBook-v0", ["book", "trades/values", "trades/mask", "agent"]),
        ],
    )
    def test_finance_envs(self, env_id: str, keys: list[str]) -> None:
        env = sekai.UnnestObservation(sekai.make(env_id))
        assert list(env.observation_space.keys()) == keys
        assert sekai.check_env(env, check_determinism=True) == []

    def test_matches_the_wrapped_env(self) -> None:
        inner, env = sekai.make("Market-v0"), sekai.UnnestObservation(sekai.make("Market-v0"))
        a, b = inner.reset(seed=3), env.reset(seed=3)
        assert _equal(env.nest(b.observation), a.observation)
        inner.action_space.seed(0)
        for _ in range(5):
            action = inner.action_space.sample(mask=sekai.get_action_mask(inner))
            a, b = inner.step(action), env.step(action)
            assert _equal(b.observation, unnest(inner.observation_space, a.observation))
            assert (a.reward, a.info["equity"]) == (b.reward, b.info["equity"])
        assert _equal(sekai.get_action_mask(env), sekai.get_action_mask(inner))
        assert env.nested_space == inner.observation_space

    def test_rejects_flat_observations(self, counting_env: Any) -> None:
        with pytest.raises(TypeError, match="Dict or Tuple"):
            sekai.UnnestObservation(counting_env)

    def test_async_path_stays_async(self) -> None:
        base = AsyncOnlyEnv()
        nested = sekai.Dict(
            {"x": sekai.Tuple([base.observation_space, sekai.Dict({"y": base.observation_space})])}
        )
        env = sekai.UnnestObservation(
            sekai.TransformObservation(base, lambda o: {"x": (o, {"y": o + 1})}, nested)
        )
        assert list(run(env.async_reset(seed=0)).observation) == ["x/0", "x/1/y"]
        obs = run(env.async_step(1)).observation
        assert obs["x/0"][0] == 1 and obs["x/1/y"][0] == 2
        assert base.async_calls == 2

    def test_pickles(self) -> None:
        env = sekai.UnnestObservation(sekai.make("Market-v0"))
        env.reset(seed=0)
        clone = pickle.loads(pickle.dumps(env))
        action = np.full(11, 0.5, dtype=np.float32)
        assert _equal(env.step(action).observation, clone.step(action).observation)


class TestVector:
    def test_matches_unnesting_each_sub_env(self) -> None:
        def market() -> Any:
            return sekai.make("Market-v0", episode_length=3)

        vec = sekai.VecUnnestObservation(sekai.SyncVecEnv([market, market]))
        each = sekai.SyncVecEnv([lambda: sekai.UnnestObservation(market())] * 2)
        assert vec.observation_space == each.observation_space
        assert _equal(vec.reset(seed=0).observations, each.reset(seed=0).observations)
        actions = np.full((2, 11), 0.5, dtype=np.float32)
        finals = 0
        for _ in range(4):
            r, s = vec.step(actions), each.step(actions)
            assert _equal(r.observations, s.observations)
            for info, other in zip(r.infos, s.infos):
                if "final_observation" in info:
                    finals += 1
                    assert list(info["final_observation"]) == list(vec.observation_space.keys())
                    assert _equal(info["final_observation"], other["final_observation"])
        assert finals == 2
        nested = vec.nest(r.observations)
        assert nested["assets"]["values"].shape == (2, 10, 20, 2)
        assert _equal(unnest(vec.nested_space, nested), r.observations)
        vec.close()
        each.close()

    def test_async_vec_env(self) -> None:
        vec = sekai.VecUnnestObservation(sekai.AsyncVecEnv([lambda: sekai.make("Market-v0")] * 2))
        result = run(vec.async_reset(seed=0))
        assert result.observations["assets/values"].shape == (2, 10, 20, 2)
        vec.close()


class TestGymnasium:
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.parametrize(
        ("env_id", "mask_size"), [("Market-v0", 11), ("LimitOrderBook-v0", 15)]
    )
    def test_converts_without_nesting(self, env_id: str, mask_size: int) -> None:
        gym = pytest.importorskip("gymnasium")
        from gymnasium.utils.env_checker import check_env as gym_check_env

        from sekai.compat import to_gymnasium

        env = to_gymnasium(sekai.UnnestObservation(sekai.make(env_id)))
        assert isinstance(env.observation_space, gym.spaces.Dict)
        assert not any(
            isinstance(sub, (gym.spaces.Dict, gym.spaces.Tuple))
            for sub in env.observation_space.spaces.values()
        )
        gym_check_env(env, skip_render_check=True)
        env.reset(seed=0)
        assert env.action_masks().shape == (mask_size,)  # still forwarded for MaskablePPO

    def test_sb3_multi_input_policy(self) -> None:
        pytest.importorskip("gymnasium")
        pytest.importorskip("stable_baselines3")
        from stable_baselines3 import PPO
        from stable_baselines3.common.env_checker import check_env as sb3_check_env

        from sekai.compat import to_gymnasium

        def nested_warnings(env: Any) -> list[str]:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                sb3_check_env(env)
            return [str(w.message) for w in caught if "Nested" in str(w.message)]

        assert nested_warnings(to_gymnasium(sekai.make("Market-v0")))
        env = to_gymnasium(sekai.UnnestObservation(sekai.make("Market-v0")))
        assert not nested_warnings(env)
        PPO("MultiInputPolicy", env, n_steps=64, batch_size=32, seed=0).learn(total_timesteps=128)
