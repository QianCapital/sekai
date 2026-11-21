"""Reference PyTorch networks for sekai observations (optional dependency).

    import sekai.nn
    encoder = sekai.nn.MaskedSetEncoder(element_shape=(20, 2), embed_dim=64)
    enc = encoder(values, mask)  # values (B, S, 20, 2), mask (B, S)
    enc.slots, enc.pooled        # (B, S, 64) per-slot embeddings, (B, 128) set embedding

`sekai.nn.sb3` adds a Stable-Baselines3 features extractor (imported separately, so this
package works without Stable-Baselines3). `import sekai` never imports PyTorch.

Install with `pip install "sekai[torch]"`.
"""

from __future__ import annotations

try:
    import torch  # noqa: F401
except ImportError as e:  # pragma: no cover - exercised only without torch
    raise ImportError("sekai.nn needs PyTorch. Install it with: pip install 'sekai[torch]'") from e

from sekai.nn.set_encoder import MaskedSetEncoder, SetEncoding, masked_max, masked_mean, mlp

__all__ = [
    "MaskedSetEncoder",
    "SetEncoding",
    "masked_mean",
    "masked_max",
    "mlp",
]
