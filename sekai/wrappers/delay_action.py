from __future__ import annotations

import copy
from collections import deque
from typing import Any

from sekai._types import ActType, ObsType, SeedType
from sekai.core.env import Env
from sekai.core.result import ResetResult
from sekai.core.wrapper import ActionWrapper
from sekai.spaces.box import Box
from sekai.wrappers.autoreset import _warn_if_outside_autoreset


class DelayAction(ActionWrapper[ObsType, ActType, ActType]):
    """Executes each action `delay` steps after the agent chooses it.

    Models order latency: the decision made on bar t reaches the market on bar
    t + delay, so a policy cannot trade on the very price it just observed. For
    the first `delay` steps of an episode the env receives `noop`.

    Args:
        env: The environment to wrap.
        delay: Steps between choosing and executing an action (0 = no delay).
        noop: Action executed while the pipeline fills. Defaults to the zero action
            for Box action spaces (e.g. all-cash or no-trade, depending on the env);
            required for other spaces, because index 0 of a Discrete space is
            rarely "do nothing".
    """

    def __init__(self, env: Env[ObsType, ActType], delay: int, *, noop: Any = None) -> None:
        if delay < 0:
            raise ValueError(f"delay must be >= 0, got {delay}.")
        space = env.action_space
        if noop is None:
            if not isinstance(space, Box):
                raise ValueError(
                    f"DelayAction needs an explicit noop action for {type(space).__name__} "
                    "action spaces."
                )
            noop = space.zero()
        if not space.contains(noop):
            raise ValueError(f"noop {noop!r} is not in the action space {space!r}.")
        super().__init__(env)
        _warn_if_outside_autoreset(env, "DelayAction")
        self.delay = delay
        self.noop: ActType = noop
        self._pending: deque[ActType] = deque()

    @property
    def pending_actions(self) -> list[ActType]:
        """Actions chosen but not yet executed, oldest first."""
        return list(self._pending)

    def action(self, action: ActType) -> ActType:
        if self.delay == 0:
            return action
        # Copy: callers (and VecEnvs, which pass views of one batch array) may reuse buffers.
        self._pending.append(copy.deepcopy(action))
        if len(self._pending) > self.delay:
            return self._pending.popleft()
        return self.noop

    def reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[ObsType]:
        self._pending.clear()
        return self.env.reset(seed=seed, options=options)

    async def async_reset(
        self,
        *,
        seed: SeedType = None,
        options: dict[str, Any] | None = None,
    ) -> ResetResult[ObsType]:
        self._pending.clear()
        return await self.env.async_reset(seed=seed, options=options)
