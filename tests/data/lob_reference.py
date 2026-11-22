"""Reference rollouts of the order-book environments with their default behaviour.

`lob_reference.npz`, next to this file, holds per-step digests of rollouts of
`LimitOrderBook-v0` and `LimitOrderBookMarket` (observations, rewards, terminations and
infos) for a fixed action sequence, recorded with sekai as it was before historical
replay and order latency were added. `tests/test_finance_lob_replay.py` replays the same
rollouts and checks the digests, so a change to the default behaviour shows as a failure,
with the first step that differs.

This module uses only what existed then, so it runs on that version. Regenerate the file
only from a checkout of it (never from later code, which would just record itself), and
only if the reference must change for an outside reason such as a NumPy release that
alters a random stream, and only on x86-64 (arm64 builds of NumPy round `uniform()`
differently; see `uniform_rounds_as_recorded` in the test):

    mkdir -p /tmp/sekai-reference && git archive e0eff20 sekai | tar -x -C /tmp/sekai-reference
    PYTHONPATH=/tmp/sekai-reference python tests/data/lob_reference.py tests/data/lob_reference.npz
"""

from __future__ import annotations

import hashlib
import struct
import sys
from typing import Any

import numpy as np

import sekai
from sekai.envs.finance import LimitOrderBookMarket
from sekai.finance import ZeroIntelligenceFlow

SINGLE_CONFIGS: dict[str, dict[str, Any]] = {
    "default": {},
    "thin": {
        "flow": ZeroIntelligenceFlow(cancel_rate=1.0),
        "decision_interval": 0.3,
        "session_length": 40.0,
        "max_inventory": 3,
    },
}
MULTI_CONFIGS: dict[str, dict[str, Any]] = {
    "default": {},
    "jitter": {
        "n_agents": 3,
        "decision_intervals": [1.0, 2.0, 0.5],
        "jitter": 0.2,
        "session_length": 60.0,
    },
}
SEEDS = (0, 1, 2)


def canonical(value: Any) -> bytes:
    """Platform-independent bytes of an observation, reward or info value."""
    if isinstance(value, dict):
        return b"{" + b"".join(canonical(k) + canonical(value[k]) for k in sorted(value)) + b"}"
    if isinstance(value, list | tuple):
        return b"[" + b"".join(canonical(v) for v in value) + b"]"
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value, dtype=value.dtype.newbyteorder("<"))
        return f"a{array.dtype.str}{array.shape}".encode() + array.tobytes()
    if isinstance(value, bool | np.bool_):
        return b"b1" if value else b"b0"
    if isinstance(value, int | np.integer):
        return b"i" + str(int(value)).encode()
    if isinstance(value, float | np.floating):
        return b"f" + struct.pack("<d", float(value))
    if value is None:
        return b"n"
    if isinstance(value, str):
        return b"s" + value.encode()
    raise TypeError(f"cannot digest {type(value).__name__}")


def digest(*values: Any) -> int:
    """A 64-bit digest of `values`."""
    return int.from_bytes(hashlib.blake2b(canonical(values), digest_size=8).digest(), "little")


def single_rollout(kwargs: dict[str, Any], seed: int, actions: np.ndarray) -> list[int]:
    """Digests of the reset and of each step of `LimitOrderBook-v0`, acting `actions`."""
    env = sekai.make("LimitOrderBook-v0", **kwargs)
    r = env.reset(seed=seed)
    digests = [digest(r.observation, r.info)]
    for action in actions:
        s = env.step(action.astype(np.int64))
        digests.append(digest(s.observation, s.reward, s.terminated, s.truncated, s.info))
        if s.terminated or s.truncated:
            break
    return digests


def multi_rollout(kwargs: dict[str, Any], seed: int, actions: np.ndarray) -> list[int]:
    """Digests of the reset and of each step of `LimitOrderBookMarket`.

    At step k trader i, when it acts, takes `actions[k, i]`.
    """
    env = LimitOrderBookMarket(**kwargs)
    r = env.reset(seed=seed)
    digests = [digest(r.observations, r.info)]
    k = 0
    while env.agents:
        acting = sorted(r.observations)
        step_actions = {a: actions[k, int(a.split("_")[1])].astype(np.int64) for a in acting}
        r = env.step(step_actions)
        digests.append(digest(r.observations, r.rewards, r.terminated, r.truncated, r.info))
        k += 1
    return digests


def record(path: str) -> None:
    """Record every reference rollout into the .npz file at `path`."""
    rng = np.random.default_rng(2026)
    single_actions = rng.integers(0, [6, 6, 3], size=(400, 3)).astype(np.int8)
    multi_actions = rng.integers(0, [6, 6, 3], size=(400, 3, 3)).astype(np.int8)
    out: dict[str, np.ndarray] = {
        "single_actions": single_actions,
        "multi_actions": multi_actions,
    }
    for name, kwargs in SINGLE_CONFIGS.items():
        for seed in SEEDS:
            digests = single_rollout(kwargs, seed, single_actions)
            out[f"single_{name}_{seed}_digests"] = np.array(digests, dtype="<u8")
    for name, kwargs in MULTI_CONFIGS.items():
        for seed in SEEDS:
            digests = multi_rollout(kwargs, seed, multi_actions)
            out[f"multi_{name}_{seed}_digests"] = np.array(digests, dtype="<u8")
    np.savez_compressed(path, **out)


if __name__ == "__main__":
    record(sys.argv[1])
