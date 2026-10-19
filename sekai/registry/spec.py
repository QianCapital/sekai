from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# The name group is non-greedy so "CartPole-v1" splits into name="CartPole", version=1.
ENV_ID_RE = re.compile(
    r"^(?:(?P<namespace>[a-zA-Z0-9_-]+)/)?(?P<name>[a-zA-Z0-9_-]+?)(?:-v(?P<version>\d+))?$"
)


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

    vector_entry_point optionally names a natively vectorised implementation
    (a VecEnv taking `num_envs` plus the env's kwargs) that make_vec() uses by default.
    """

    id: str
    entry_point: str | Callable[..., Any] | None = None
    reward_threshold: float | None = None
    nondeterministic: bool = False
    max_episode_steps: int | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    additional_wrappers: tuple[WrapperSpec, ...] = field(default_factory=tuple)
    vector_entry_point: str | Callable[..., Any] | None = None

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
        """Serialise to JSON. Callable entry points cannot be serialised and are dropped."""
        data: dict[str, Any] = {
            "id": self.id,
            "reward_threshold": self.reward_threshold,
            "nondeterministic": self.nondeterministic,
            "max_episode_steps": self.max_episode_steps,
            "kwargs": self.kwargs,
            "additional_wrappers": [
                {"name": w.name, "entry_point": w.entry_point, "kwargs": w.kwargs}
                for w in self.additional_wrappers
            ],
        }
        if isinstance(self.entry_point, str):
            data["entry_point"] = self.entry_point
        if isinstance(self.vector_entry_point, str):
            data["vector_entry_point"] = self.vector_entry_point
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, s: str) -> EnvSpec:
        data = json.loads(s)
        wrappers = tuple(WrapperSpec(**w) for w in data.pop("additional_wrappers", []))
        return cls(**data, additional_wrappers=wrappers)

    def pprint(self) -> str:
        """Human-readable multi-line description of this spec."""
        entry = self.entry_point if isinstance(self.entry_point, str) else repr(self.entry_point)
        lines = [f"id={self.id}", f"entry_point={entry}"]
        if self.vector_entry_point is not None:
            vec = self.vector_entry_point
            lines.append(f"vector_entry_point={vec if isinstance(vec, str) else repr(vec)}")
        if self.max_episode_steps is not None:
            lines.append(f"max_episode_steps={self.max_episode_steps}")
        if self.reward_threshold is not None:
            lines.append(f"reward_threshold={self.reward_threshold}")
        if self.nondeterministic:
            lines.append("nondeterministic=True")
        if self.kwargs:
            lines.append(f"kwargs={self.kwargs}")
        for w in self.additional_wrappers:
            lines.append(f"wrapper={w.name} ({w.entry_point}) {w.kwargs or ''}".rstrip())
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"EnvSpec(id='{self.id}')"
