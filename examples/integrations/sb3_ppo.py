"""Train Stable-Baselines3 PPO on a sekai environment via the gymnasium bridge.

    pip install "sekai[gymnasium]" stable-baselines3

Not run in CI (it needs PyTorch); see examples/*.py for dependency-free examples.
"""

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

import sekai
from sekai.compat import to_gymnasium

# Any sekai env (or wrapper stack) becomes a gymnasium.Env that SB3 accepts.
vec_env = make_vec_env(lambda: to_gymnasium(sekai.make("CartPole-v1")), n_envs=4, seed=0)

model = PPO("MlpPolicy", vec_env, verbose=0, seed=0)
model.learn(total_timesteps=50_000)

env = to_gymnasium(sekai.make("CartPole-v1"))
returns = []
for seed in range(10):
    obs, _ = env.reset(seed=seed)
    done, total = False, 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        done = terminated or truncated
    returns.append(total)
print(f"PPO mean return over 10 episodes: {sum(returns) / len(returns):.1f}")
