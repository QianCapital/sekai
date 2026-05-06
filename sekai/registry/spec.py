from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable


ENV_ID_RE = re.compile(r"^(?:(?P<namespace>[a-zA-Z0-9_-]+)\/)?(?P<name>[a-zA-Z0-9_-]+)(?:-v(?P<version>\d+))?$")


@dataclass
class WrapperSpec:
    """Specification for a wrapper to apply on top of an environment."""

    name: str
    entry_point: str  # "module.path:ClassName"
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvSpec:
    """Serialisable environment specification.

    IDs follow the format: [namespace/]Name[-vN]
    Examples: "CartPole-v1", "qc/MarketEnv-v0", "my_ns/CustomEnv"
    """

    id: str
    entry_point: str | Callable[..., Any] | None = None
    reward_threshold: float | None = None
    nondeterministic: bool = False
    max_episode_steps: int | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    additional_wrappers: tuple[WrapperSpec, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        m = ENV_ID_RE.match(self.id)
        if m is None:
            raise ValueError(
                f"Invalid env ID '{self.id}'. "
                "Must match [namespace/]Name[-vN] (e.g. 'CartPole-v1', 'qc/MarketEnv-v0')."
            )
        self.namespace: str | None = m.group("namespace")
        self.name: str = m.group("name")
        version_str = m.group("version")
        self.version: int | None = int(version_str) if version_str is not None else None

    def make(self, **kwargs: Any) -> Any:
        """Instantiate this environment."""
        from sekai.registry.registry import _make_from_spec
        return _make_from_spec(self, **kwargs)

    def to_json(self) -> str:
        data = {
            "id": self.id,
            "reward_threshold": self.reward_threshold,
            "nondeterministic": self.nondeterministic,
            "max_episode_steps": self.max_episode_steps,
            "kwargs": self.kwargs,
        }
        if isinstance(self.entry_point, str):
            data["entry_point"] = self.entry_point
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, s: str) -> EnvSpec:
        data = json.loads(s)
        return cls(**data)

    def __repr__(self) -> str:
        return f"EnvSpec(id='{self.id}')"
