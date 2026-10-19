"""Throughput of the vector backends on a finance environment.

SyncVecEnv steps N Python environments in turn; ProcVecEnv spreads them over
worker processes (observations travel through shared memory); a native vector
env advances all N at once with array operations. For simulators that
vectorise, native is typically one to two orders of magnitude faster.
"""

import time

import sekai


def throughput(vec, steps: int = 200) -> float:
    actions = vec.batch_action_space
    actions.seed(0)
    vec.reset(seed=0)
    batch = actions.sample()
    vec.step(batch)  # warm-up
    start = time.perf_counter()
    for _ in range(steps):
        vec.step(batch)
    rate = steps * vec.num_envs / (time.perf_counter() - start)
    vec.close()
    return rate


if __name__ == "__main__":  # ProcVecEnv starts worker processes
    print(f"{'backend':32s} {'env-steps/s':>12s}")
    for label, make in {
        "SyncVecEnv x16": lambda: sekai.make_vec("Portfolio-v0", 16, vectorization_mode="sync"),
        "ProcVecEnv x16 (4 workers)": lambda: sekai.make_vec(
            "Portfolio-v0", 16, vectorization_mode="proc", vector_kwargs={"num_workers": 4}
        ),
        "native x16": lambda: sekai.make_vec("Portfolio-v0", 16),
        "native x1024": lambda: sekai.make_vec("Portfolio-v0", 1024),
    }.items():
        print(f"{label:32s} {throughput(make()):12,.0f}")
