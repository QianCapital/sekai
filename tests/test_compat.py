from __future__ import annotations

from typing import Any

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


# ---------------------------------------------------------------------------
# to_sb3_vec_env
# ---------------------------------------------------------------------------


def _sb3_vec(venv: Any) -> Any:
    pytest.importorskip("stable_baselines3")
    from sekai.compat import to_sb3_vec_env

    return to_sb3_vec_env(venv)


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return bool(np.array_equal(a, b))


@pytest.mark.parametrize("env_id", ["CartPole-v1", "Pendulum-v1"])  # terminates / truncates
def test_sb3_vec_env_matches_sb3_dummy_vec_env(env_id: str) -> None:
    """A SyncVecEnv through the bridge steps exactly as SB3's DummyVecEnv over to_gymnasium envs."""
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv

    n = 4
    ours = _sb3_vec(sekai.make_vec(env_id, n, vectorization_mode="sync"))
    theirs = DummyVecEnv([lambda: to_gymnasium(sekai.make(env_id))] * n)
    ours.seed(3)
    theirs.seed(3)
    assert _same(ours.reset(), theirs.reset())
    actions = theirs.action_space
    actions.seed(0)
    finished = truncated = 0
    for _ in range(210):
        batch = np.stack([actions.sample() for _ in range(n)])
        obs_a, rew_a, done_a, infos_a = ours.step(batch)
        obs_b, rew_b, done_b, infos_b = theirs.step(batch)
        assert _same(obs_a, obs_b) and _same(rew_a, rew_b) and _same(done_a, done_b)
        assert rew_a.dtype == rew_b.dtype and done_a.dtype == done_b.dtype
        for a, b in zip(infos_a, infos_b):
            assert a["TimeLimit.truncated"] == b["TimeLimit.truncated"]
            assert ("terminal_observation" in a) == ("terminal_observation" in b)
            if "terminal_observation" in b:
                assert _same(a["terminal_observation"], b["terminal_observation"])
                finished += 1
                truncated += b["TimeLimit.truncated"]
    assert finished and (truncated > 0) == (env_id == "Pendulum-v1")
    ours.close()


def test_sb3_vec_env_on_a_native_vec_env() -> None:
    venv = sekai.make_vec("Portfolio-v0", 4, episode_length=5)
    assert isinstance(venv, sekai.NativeVecEnv)
    env = _sb3_vec(venv)
    assert env.observation_space == space_to_gymnasium(venv.observation_space)
    assert env.action_space == space_to_gymnasium(venv.action_space)
    assert env.render_mode is None
    env.seed(1)
    first = env.reset()
    env.seed(1)
    assert _same(env.reset(), first) and first["returns"].shape == (4, 30, 5)
    assert "equity" in env.reset_infos[0]
    for _ in range(5):
        obs, rewards, dones, infos = env.step(np.full((4, 6), 0.5, np.float32))
    assert dones.all() and rewards.dtype == np.float32 and rewards.shape == (4,)
    for i, info in enumerate(infos):  # same-step autoreset: obs already start the next episode
        assert info["TimeLimit.truncated"]
        assert info["terminal_observation"] is info["final_observation"]
        assert not _same(info["terminal_observation"]["returns"], obs["returns"][i])
        assert env.reset_infos[i] is info["reset_info"]


def test_sb3_ppo_on_a_native_vec_env_logs_episodes() -> None:
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO

    stats = sekai.VecRecordEpisodeStatistics(sekai.make_vec("Portfolio-v0", 8, episode_length=16))
    model = PPO("MultiInputPolicy", _sb3_vec(stats), n_steps=32, batch_size=64, seed=0)
    model.learn(total_timesteps=512)
    assert model.num_timesteps == 512
    episodes = list(model.ep_info_buffer)  # SB3's logger reads {"r", "l", "t"} dicts
    assert len(episodes) == 32 and all(e["l"] == 16 and e["t"] >= 0 for e in episodes)
    tracked = stats.tracker.summary()
    assert tracked["num_episodes"] == 32
    assert np.isclose(np.mean([e["r"] for e in episodes]), tracked["mean_return"], atol=1e-5)


def test_sb3_vec_env_attributes_and_methods() -> None:
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import VecMonitor

    sync = _sb3_vec(sekai.make_vec("CartPole-v1", 3, vectorization_mode="sync"))
    assert sync.get_attr("x_threshold") == [2.4] * 3
    sync.set_attr("x_threshold", 3.0, indices=[1])
    assert sync.get_attr("x_threshold") == [2.4, 3.0, 2.4]
    assert sync.get_attr("x_threshold", indices=1) == [3.0]
    sync.set_attr("tags", ["a", "b"])  # one list, for every sub-env
    assert sync.get_attr("tags") == [["a", "b"]] * 3
    sync.reset()
    assert sync.env_method("reset", seed=0, indices=[2])[0].observation.shape == (4,)
    assert sync.env_is_wrapped(sekai.TimeLimit) == [True] * 3
    assert sync.env_is_wrapped(Monitor) == [False] * 3
    VecMonitor(sync)  # asks env_is_wrapped(Monitor); no warning

    native = _sb3_vec(sekai.make_vec("Portfolio-v0", 2))
    native.reset()
    assert native.env_method("elapsed_steps") == [0, 0]
    assert native.env_is_wrapped(sekai.TimeLimit) == [False, False]
    with pytest.raises(NotImplementedError, match="sub-env objects"):
        native.env_method("elapsed_steps", indices=[0])


def test_sb3_vec_env_validates_its_input() -> None:
    with pytest.raises(TypeError, match="to_gymnasium"):
        _sb3_vec(sekai.make("CartPole-v1"))
    with pytest.raises(ValueError, match="autoreset"):
        _sb3_vec(sekai.make_vec("CartPole-v1", 2, vector_kwargs={"autoreset": False}))
