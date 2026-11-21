"""Reference results: PPO with a set encoder against the classical baselines.

    pip install "sekai[gymnasium,torch]" stable-baselines3

Trains PPO (Stable-Baselines3, with `sekai.nn.sb3.MarketFeaturesExtractor`, which encodes
the padded set of assets as a set) on Market-v1's simulated market, then evaluates it
and the baselines on the same held-out episodes with `sekai.finance.benchmark.compare`.
The table in the docs' benchmark page comes from this script. Training seeds are
below the evaluation seeds (10_000 and up), so no evaluation episode was seen in
training. Not run in CI (it needs PyTorch and takes a few minutes).
"""

import sys

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import sekai
from sekai.compat import to_gymnasium
from sekai.finance.benchmark import Task, compare, standard_baselines
from sekai.nn.sb3 import MarketFeaturesExtractor

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
kwargs = {"episode_length": 252}


def make_env():
    return to_gymnasium(sekai.UnnestObservation(sekai.make("Market-v1", **kwargs)))


model = PPO(
    "MultiInputPolicy",
    make_vec_env(make_env, n_envs=8, seed=0),
    n_steps=256,
    batch_size=256,
    learning_rate=3e-4,
    policy_kwargs={"features_extractor_class": MarketFeaturesExtractor},
    seed=0,
    verbose=0,
)
model.learn(total_timesteps=steps)

unnest = sekai.UnnestObservation(sekai.make("Market-v1", **kwargs))


def ppo(observation, env):
    action, _ = model.predict(unnest.observation(observation), deterministic=True)
    return action


policies = {**standard_baselines(), f"PPO ({steps // 1000}k steps)": ppo}
print(compare(policies, Task("market", "Market-v1", kwargs, episodes=200)))
