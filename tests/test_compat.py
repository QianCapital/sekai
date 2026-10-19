from __future__ import annotations

import numpy as np
import pytest

import sekai

gym = pytest.importorskip("gymnasium")
from gymnasium.utils.env_checker import check_env as gym_check_env  # noqa: E402

from sekai.compat import (  # noqa: E402
    from_gymnasium,
    space_from_gymnasium,
    space_to_gymnasium,
    to_gymnasium,
)

SPACES = [
    sekai.Box(-1, 1, shape=(2, 3)),
    sekai.Box(0, 255, shape=(4,), dtype=np.uint8),
    sekai.Discrete(5, start=-2),
    sekai.MultiDiscrete([2, 3], start=[1, 0]),
    sekai.MultiBinary(4),
    sekai.Dict({"a": sekai.Discrete(2), "b": sekai.Tuple([sekai.Box(0, 1, shape=(1,))])}),
    sekai.Dict(
        {
            "z": sekai.Discrete(2),
            "a": sekai.Dict({"y": sekai.MultiBinary(2), "b": sekai.Discrete(3)}),
        }
    ),
]


@pytest.mark.parametrize("space", SPACES, ids=repr)
def test_space_round_trip(space: sekai.Space) -> None:
    g = space_to_gymnasium(space)
    assert space_from_gymnasium(g) == space
    g.seed(0)
    assert space.contains(g.sample())


def test_dict_key_order_is_kept_both_ways() -> None:
    space = sekai.Dict({"b": sekai.Discrete(3), "a": sekai.Box(-1, 1, shape=(2,))})
    g = space_to_gymnasium(space)
    assert list(g.spaces) == ["b", "a"]  # gymnasium would sort a plain dict's keys
    assert space_from_gymnasium(g) == space
    g2 = gym.spaces.Dict([("y", gym.spaces.Discrete(2)), ("x", gym.spaces.Discrete(3))])
    assert list(space_from_gymnasium(g2).keys()) == ["y", "x"]
    assert list(space_to_gymnasium(space_from_gymnasium(g2)).spaces) == ["y", "x"]
    masked = space_to_gymnasium(sekai.MaskedBox(0.0, 1.0, shape=(2,), max_size=3))
    assert list(masked.spaces) == ["values", "mask"]


@pytest.mark.parametrize("env_id", ["LimitOrderBook-v0", "Market-v0"])
def test_flattening_matches_gymnasium(env_id: str) -> None:
    # Both flatten in the observation space's key order, so the layouts must agree.
    ours = sekai.FlattenObservation(sekai.make(env_id))
    theirs = gym.wrappers.FlattenObservation(to_gymnasium(sekai.make(env_id)))
    assert ours.observation_space.shape == theirs.observation_space.shape
    obs, _ = theirs.reset(seed=0)
    np.testing.assert_array_equal(ours.reset(seed=0).observation, obs)
    ours.action_space.seed(0)
    for _ in range(10):
        action = ours.action_space.sample()
        np.testing.assert_array_equal(ours.step(action).observation, theirs.step(action)[0])


# gymnasium's checker emits advisory warnings (infinite Box bounds, non-normalised
# actions) that its own CartPole and Pendulum trigger too; only errors matter here.
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "env_id", ["CartPole-v1", "Pendulum-v1", "Portfolio-v0", "MarketMaking-v0"]
)
def test_to_gymnasium_passes_gymnasium_checker(env_id: str) -> None:
    gym_check_env(to_gymnasium(sekai.make(env_id)), skip_render_check=True)


def test_to_gymnasium_api() -> None:
    env = to_gymnasium(sekai.make("CartPole-v1"))
    obs, info = env.reset(seed=0)
    assert obs.shape == (4,) and info == {}
    obs, reward, terminated, _truncated, info = env.step(0)
    assert isinstance(reward, float) and isinstance(terminated, bool)
    env.close()


def test_to_gymnasium_rendering() -> None:
    from sekai.envs.classic_control import CartPoleRenderer

    env = to_gymnasium(sekai.make("CartPole-v1"), renderer=CartPoleRenderer())
    assert env.render_mode == "rgb_array"
    env.reset(seed=0)
    assert env.render().shape == (400, 600, 3)


def test_to_gymnasium_validates_render_mode() -> None:
    from sekai.envs.classic_control import CartPoleRenderer

    with pytest.raises(ValueError, match="HumanRendering"):
        to_gymnasium(sekai.make("CartPole-v1"), CartPoleRenderer(), render_mode="human")
    with pytest.raises(ValueError, match="Unsupported render_mode 'rgb'"):
        to_gymnasium(sekai.make("CartPole-v1"), CartPoleRenderer(), render_mode="rgb")
    with pytest.raises(ValueError, match="needs a renderer"):
        to_gymnasium(sekai.make("CartPole-v1"), render_mode="rgb_array")

    class TextRenderer(sekai.rendering.BaseRenderer):
        def render(self, env: object) -> str:
            return "frame"

    env = to_gymnasium(sekai.make("CartPole-v1"), TextRenderer(), render_mode="ansi")
    env.reset(seed=0)
    assert env.render() == "frame" and env.metadata["render_modes"] == ["ansi"]


def test_from_gymnasium() -> None:
    env = from_gymnasium(gym.make("CartPole-v1"))
    assert isinstance(env, sekai.Env)
    assert sekai.check_env(env) == []
    wrapped = sekai.RecordEpisodeStatistics(env)
    wrapped.reset(seed=0)
    while not (r := wrapped.step(1)).done:
        pass
    assert isinstance(r.info["episode"], sekai.EpisodeStats)


def test_from_gymnasium_vectorised() -> None:
    vec = sekai.SyncVecEnv([lambda: from_gymnasium(gym.make("CartPole-v1"))] * 2)
    vec.reset(seed=0)
    assert vec.step([0, 1]).observations.shape == (2, 4)


def test_from_gymnasium_numpy_integer_seed() -> None:
    env = from_gymnasium(gym.make("CartPole-v1"))
    obs = env.reset(seed=np.int64(5)).observation  # gymnasium itself only takes int
    assert env._rng_seed == 5 and type(env._rng_seed) is int
    np.testing.assert_array_equal(obs, gym.make("CartPole-v1").reset(seed=5)[0])
    vec = sekai.SyncVecEnv([lambda: from_gymnasium(gym.make("CartPole-v1"))] * 2)
    vec.reset(seed=list(np.arange(2)))
    vec.close()


def test_from_gymnasium_generator_seed() -> None:
    env = from_gymnasium(gym.make("CartPole-v1"))
    rng = np.random.default_rng(3)
    first = env.reset(seed=rng).observation
    assert env.rng is rng and env._rng_seed == -1  # the env now draws from that generator
    again = from_gymnasium(gym.make("CartPole-v1")).reset(seed=np.random.default_rng(3))
    np.testing.assert_array_equal(first, again.observation)
    env.reset()  # an unseeded reset keeps drawing from it
    assert env.rng is rng


def test_maskable_ppo_on_flattened_order_book() -> None:
    """The documented recipe: FlattenObservation + MlpPolicy, with masks reaching MaskablePPO."""
    pytest.importorskip("sb3_contrib")
    from sb3_contrib import MaskablePPO

    env = to_gymnasium(sekai.FlattenObservation(sekai.make("LimitOrderBook-v0", max_inventory=2)))
    assert env.action_masks().shape == (15,)
    masks_seen: list[bool] = []

    class AuditMasks(gym.Wrapper):  # records whether each action respects the current mask
        def step(self, action: np.ndarray) -> tuple:
            space = self.env.action_space
            parts = np.split(self.env.action_masks(), np.cumsum(space.nvec)[:-1])
            masks_seen.append(all(p[a - s] for p, a, s in zip(parts, action, space.start)))
            return self.env.step(action)

    model = MaskablePPO("MlpPolicy", AuditMasks(env), n_steps=64, batch_size=32, seed=0)
    model.learn(total_timesteps=256)
    assert len(masks_seen) >= 256 and all(masks_seen)


def test_ppo_on_flattened_market() -> None:
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    env = to_gymnasium(sekai.FlattenObservation(sekai.make("Market-v0")))
    PPO("MlpPolicy", env, n_steps=64, batch_size=32, seed=0).learn(total_timesteps=128)
