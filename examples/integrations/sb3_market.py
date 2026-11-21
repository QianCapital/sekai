"""Train Stable-Baselines3 PPO on Market-v0 with sekai.nn's set encoder.

pip install "sekai[gymnasium,torch]" stable-baselines3

Market-v0 observes a padded set of assets (a MaskedBox inside its Dict observation), which
SB3's MultiInputPolicy cannot take. UnnestObservation gives it flat keys instead
("assets/values", "assets/mask", "portfolio", "account"), and MarketFeaturesExtractor
encodes the asset set with a MaskedSetEncoder, ignoring empty slots. SlotActorCriticPolicy
goes further: each asset's weight comes from that asset's own embedding.

Not run in CI (it needs PyTorch); see examples/*.py for dependency-free examples.
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import sekai
from sekai.compat import to_gymnasium
from sekai.nn.sb3 import MarketFeaturesExtractor, SlotActorCriticPolicy


def make_env():
    """Market-v0 (10 asset slots, cash last) with flat-keyed observations, for gymnasium."""
    return to_gymnasium(sekai.UnnestObservation(sekai.make("Market-v0", episode_length=63)))


# Each asset's (20, 2) window of log returns and availability goes through a 1-D convolution.
set_kwargs = {"element_encoder": "conv", "attention_layers": 1}
vec_env = make_vec_env(make_env, n_envs=4, seed=0)

pooled = PPO(
    "MultiInputPolicy",
    vec_env,
    n_steps=256,
    batch_size=256,
    seed=0,
    policy_kwargs={
        "features_extractor_class": MarketFeaturesExtractor,
        "features_extractor_kwargs": {"set_kwargs": set_kwargs},
    },
)
pooled.learn(total_timesteps=20_000)

per_asset = PPO(
    SlotActorCriticPolicy,
    vec_env,
    n_steps=256,
    batch_size=256,
    seed=0,
    policy_kwargs={"slot_set": "assets", "features_extractor_kwargs": {"set_kwargs": set_kwargs}},
)
per_asset.learn(total_timesteps=20_000)


def evaluate(policy, episodes=50):
    """Mean log growth of equity per episode (the sum of LogReturn rewards)."""
    env = make_env()
    totals = []
    for seed in range(1000, 1000 + episodes):
        obs, _ = env.reset(seed=seed)
        done, total = False, 0.0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(policy(obs))
            total += float(reward)
            done = terminated or truncated
        totals.append(total)
    return float(np.mean(totals))


equal_weight = np.append(np.ones(10), 0.0).astype(np.float32)  # every listed asset, no cash
results = {
    "PPO, pooled features": evaluate(lambda obs: pooled.predict(obs, deterministic=True)[0]),
    "PPO, per-asset scores": evaluate(lambda obs: per_asset.predict(obs, deterministic=True)[0]),
    "equal weight": evaluate(lambda obs: equal_weight),
}
for name, growth in results.items():
    print(f"{name:>22}: mean log growth per 63-step episode {growth:+.4f}")
# A 20k-step run is far from converged: the drift is small next to the noise, and the
# equal-weight portfolio is the bar a learned allocation has to clear.
