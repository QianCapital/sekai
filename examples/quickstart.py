"""Create a built-in environment, run a random policy, and read typed episode stats."""

import sekai

env = sekai.RecordEpisodeStatistics(sekai.make("CartPole-v1"))
print(env)  # <RecordEpisodeStatistics(<TimeLimit(<CartPole-v1>)>)>

result = env.reset(seed=42)
for _ in range(5):
    while not result.done:
        result = env.step(env.action_space.sample())
    stats = result.info["episode"]  # a typed EpisodeStats, not a dict
    print(f"return={stats.episode_return:5.1f}  length={stats.episode_length:3d}")
    result = env.reset()

print("mean return over 5 episodes:", env.tracker.summary()["mean_return"])
env.close()
