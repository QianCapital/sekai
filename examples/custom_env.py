"""Write your own environment, validate it with check_env, and register it."""

from typing import Any

import numpy as np

import sekai


class GuessNumberEnv(sekai.Env[np.ndarray, np.intp]):
    """Guess a hidden integer in [0, 100); the observation says 'higher' or 'lower'."""

    def __init__(self, max_guesses: int = 10) -> None:
        self.max_guesses = max_guesses
        # obs: [last guess / 100, direction (-1 lower, 0 none, +1 higher)]
        self.observation_space = sekai.Box(
            np.array([0.0, -1.0], np.float32), np.array([1.0, 1.0], np.float32)
        )
        self.action_space = sekai.Discrete(100)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> sekai.ResetResult[np.ndarray]:
        self._set_rng(seed)  # always first: seeds self.rng reproducibly
        self.target = int(self.rng.integers(100))
        self.guesses = 0
        return sekai.ResetResult(np.zeros(2, np.float32), info={})

    def step(self, action: int | np.integer) -> sekai.StepResult[np.ndarray]:
        self.guesses += 1
        guess = int(action)
        direction = float(np.sign(self.target - guess))
        correct = guess == self.target
        return sekai.StepResult(
            observation=np.array([guess / 100, direction], np.float32),
            reward=1.0 if correct else -0.1,
            terminated=correct,
            truncated=self.guesses >= self.max_guesses,
            info={"target": self.target} if correct else {},
        )


issues = sekai.check_env(GuessNumberEnv(), check_determinism=True)
assert not issues, "\n".join(issues)
print("check_env: no issues")

sekai.register("examples/GuessNumber-v0", GuessNumberEnv, kwargs={"max_guesses": 7})
env = sekai.make("examples/GuessNumber-v0")

# A binary-search policy solves it in at most 7 guesses.
result = env.reset(seed=0)
low, high = 0, 100
while not result.done:
    guess = (low + high) // 2
    result = env.step(guess)
    if result.observation[1] > 0:
        low = guess + 1
    elif result.observation[1] < 0:
        high = guess
print(f"found {result.info['target']} in {env.unwrapped.guesses} guesses")
