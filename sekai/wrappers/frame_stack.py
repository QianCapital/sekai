from __future__ import annotations

from collections import deque
from typing import Any, Literal

from sekai._types import ActType, SeedType
from sekai.core.env import Env
from sekai.core.result import ResetResult, StepResult
from sekai.core.wrapper import Wrapper
from sekai.spaces.utils import batch_space, concatenate, zero_sample
from sekai.wrappers.autoreset import _warn_if_outside_autoreset


class FrameStackObservation(Wrapper[Any, ActType, Any, ActType]):
    """Stacks the last N observations along a new leading axis.

    Gives the agent temporal context: recent prices for a policy without
    memory, or motion for pixel observations. Any observation space that can
    be batched works, and nested spaces stack leaf by leaf, so a Dict
    observation of `{"returns": (L, n), "weights": (n + 1,)}` becomes
    `{"returns": (N, L, n), "weights": (N, n + 1)}`; a MaskedBox keeps its
    values/mask layout with a leading stack axis on both.

    padding: what fills the stack right after reset(). "reset" (the default)
    repeats the first observation; "zeros" uses the space's zero sample
    (`sekai.spaces.zero_sample`), which makes the start of an episode
    explicit to the agent.

    Input obs shape: (d,) or (h, w, c)
    Output obs shape: (N, d) or (N, h, w, c)
    """

    def __init__(
        self,
        env: Env[Any, ActType],
        num_stack: int,
        *,
        padding: Literal["reset", "zeros"] = "reset",
    ) -> None:
        if num_stack < 1:
            raise ValueError(f"num_stack must be >= 1, got {num_stack}.")
        if padding not in ("reset", "zeros"):
            raise ValueError(f"padding must be 'reset' or 'zeros', got {padding!r}.")
        space = env.observation_space
        try:
            stacked = batch_space(space, num_stack)
        except NotImplementedError:
            raise TypeError(
                f"FrameStackObservation cannot stack observations of {type(space).__name__}."
            ) from None
        super().__init__(env)
        _warn_if_outside_autoreset(env, "FrameStackObservation")
        self.num_stack = num_stack
        self.padding = padding
        self._inner_space = space
        self._frames: deque[Any] = deque(maxlen=num_stack)
        self.observation_space = stacked

    def _get_obs(self) -> Any:
        return concatenate(self._inner_space, list(self._frames))

    def _on_reset(self, result: ResetResult[Any]) -> ResetResult[Any]:
        pad = result.observation if self.padding == "reset" else zero_sample(self._inner_space)
        for _ in range(self.num_stack - 1):
            self._frames.append(pad)
        self._frames.append(result.observation)
        return ResetResult(observation=self._get_obs(), info=result.info)

    def _on_step(self, result: StepResult[Any]) -> StepResult[Any]:
        self._frames.append(result.observation)
        return StepResult(
            observation=self._get_obs(),
            reward=result.reward,
            terminated=result.terminated,
            truncated=result.truncated,
            info=result.info,
        )

    def reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[Any]:
        return self._on_reset(self.env.reset(seed=seed, options=options))

    async def async_reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[Any]:
        return self._on_reset(await self.env.async_reset(seed=seed, options=options))

    def step(self, action: ActType) -> StepResult[Any]:
        return self._on_step(self.env.step(action))

    async def async_step(self, action: ActType) -> StepResult[Any]:
        return self._on_step(await self.env.async_step(action))
