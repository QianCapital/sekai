"""A permutation-invariant encoder for padded sets (`sekai.MaskedBox` samples)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, NamedTuple

import torch
from torch import nn

ElementEncoder = Literal["mlp", "conv", "transformer"]
Pooling = Literal["mean", "max", "sum"]

_POOLINGS = ("mean", "max", "sum")


def mlp(
    in_dim: int,
    hidden_sizes: Sequence[int],
    out_dim: int,
    activation: type[nn.Module] = nn.ReLU,
) -> nn.Sequential:
    """A fully connected network with `activation` after every layer, the last included.

    Args:
        in_dim: Input features.
        hidden_sizes: Width of each hidden layer (none for one linear layer).
        out_dim: Output features.
        activation: Activation module class.
    """
    layers: list[nn.Module] = []
    for width in (*hidden_sizes, out_dim):
        layers += [nn.Linear(in_dim, width), activation()]
        in_dim = width
    return nn.Sequential(*layers)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over the occupied slots: `x` `(..., S, D)`, boolean `mask` `(..., S)` → `(..., D)`.

    An empty set gives zeros (not NaN), and empty slots contribute nothing, whatever
    they hold.
    """
    total = torch.where(mask.unsqueeze(-1), x, 0.0).sum(dim=-2)
    count = mask.sum(dim=-1, keepdim=True).clamp(min=1)
    return total / count.to(total.dtype)


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Maximum over the occupied slots: `x` `(..., S, D)`, boolean `mask` `(..., S)` → `(..., D)`.

    An empty set gives zeros, and empty slots never win, whatever they hold.
    """
    top = x.masked_fill(~mask.unsqueeze(-1), -math.inf).amax(dim=-2)
    return torch.where(mask.any(dim=-1, keepdim=True), top, 0.0)


def _masked_sum(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask.unsqueeze(-1), x, 0.0).sum(dim=-2)


class SetEncoding(NamedTuple):
    """The output of `MaskedSetEncoder`."""

    slots: torch.Tensor
    """Per-slot embeddings `(B, S, embed_dim)`, zero in empty slots."""
    pooled: torch.Tensor
    """Set embedding `(B, pooled_dim)`: each pooling of the slots, concatenated."""
    mask: torch.Tensor
    """Boolean occupancy `(B, S)`."""


class _TemporalConv(nn.Module):
    """A 1-D convolution over the rows of an `(L, F)` element (a window of F features)."""

    def __init__(
        self,
        length: int,
        features: int,
        channels: Sequence[int],
        out_dim: int,
        kernel_size: int,
        activation: type[nn.Module],
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = features
        for width in channels:
            layers += [nn.Conv1d(in_channels, width, kernel_size, padding="same"), activation()]
            in_channels = width
        self.conv = nn.Sequential(*layers)
        self.out = nn.Sequential(nn.Linear(in_channels * length, out_dim), activation())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor = self.conv(x.transpose(1, 2))  # (N, F, L) -> (N, C, L)
        out: torch.Tensor = self.out(h.flatten(1))
        return out


class _TemporalTransformer(nn.Module):
    """One transformer layer over the rows of an `(L, F)` element, with learned positions."""

    def __init__(
        self,
        length: int,
        features: int,
        out_dim: int,
        num_heads: int,
        dropout: float,
        activation: type[nn.Module],
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(features, out_dim)
        self.position = nn.Parameter(torch.zeros(length, out_dim))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            out_dim, num_heads, 2 * out_dim, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            layer, 1, norm=nn.LayerNorm(out_dim), enable_nested_tensor=False
        )
        self.out = nn.Sequential(nn.Linear(2 * out_dim, out_dim), activation())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h: torch.Tensor = self.encoder(self.embed(x) + self.position)  # (N, L, D)
        out: torch.Tensor = self.out(torch.cat([h[:, -1], h.mean(dim=1)], dim=-1))
        return out


class MaskedSetEncoder(nn.Module):
    """Encodes a batch of padded sets into per-slot embeddings and a set embedding.

    The input is a batch of `sekai.MaskedBox` samples, as tensors: `values` of shape
    `(B, S, *element_shape)` and `mask` of shape `(B, S)` (any dtype; nonzero = occupied).
    Each occupied slot goes through the same element encoder; optional transformer
    layers then mix the slots (attending to occupied slots only); the slots are pooled
    with each of `pooling`, concatenated.

    Guarantees, whatever the empty slots hold (even NaN):

    - per-slot embeddings are permutation-equivariant over slots and the pooled
      embedding permutation-invariant (up to floating-point rounding);
    - empty slots change nothing: their embeddings are zero and they reach neither an
      occupied slot, nor the pooled embedding, nor the gradients;
    - an empty set encodes to zeros, without NaN.

    Element encoders:

    - `"mlp"` (default): an MLP over the flattened element.
    - `"conv"`: for an `(L, F)` element (a window of `L` rows of `F` features, such as
      `Market-v0`'s assets): 1-D convolutions over the rows, one per `hidden_sizes`
      entry, then a linear layer.
    - `"transformer"`: for an `(L, F)` element: one transformer layer over the rows with
      learned positions, reading the last row and the mean.
    - an `nn.Module` mapping `(N, *element_shape)` to `(N, embed_dim)`.

    Example:
        ```python
        env = sekai.make("Market-v0")
        space = env.observation_space["assets"]  # MaskedBox(..., (20, 2), max_size=10)
        assets = env.reset(seed=0).observation["assets"]
        encoder = MaskedSetEncoder(space.element_shape, embed_dim=64, element_encoder="conv")
        values, mask = torch.as_tensor(assets["values"]), torch.as_tensor(assets["mask"])
        enc = encoder(values[None], mask[None])  # a batch of one set
        enc.slots.shape, enc.pooled.shape  # (1, 10, 64), (1, 128)
        ```

    Args:
        element_shape: Shape of one element (`MaskedBox.element_shape`).
        embed_dim: Size of a slot embedding.
        hidden_sizes: Hidden layer widths of the `"mlp"` encoder, channels of the
            `"conv"` one.
        element_encoder: `"mlp"`, `"conv"`, `"transformer"` or a module (see above).
        attention_layers: Transformer encoder layers across slots (0: none, a pure
            Deep Sets encoder).
        num_heads: Attention heads (divides `embed_dim`), for the transformers.
        pooling: One or more of `"mean"`, `"max"`, `"sum"`, over the occupied slots.
        kernel_size: Width of the `"conv"` encoder's kernels.
        dropout: Dropout in the transformer layers.
        activation: Activation module class.
    """

    def __init__(
        self,
        element_shape: int | Sequence[int],
        embed_dim: int = 64,
        *,
        hidden_sizes: Sequence[int] = (64,),
        element_encoder: ElementEncoder | nn.Module = "mlp",
        attention_layers: int = 0,
        num_heads: int = 4,
        pooling: Pooling | Sequence[Pooling] = ("mean", "max"),
        kernel_size: int = 3,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        shape = (element_shape,) if isinstance(element_shape, int) else tuple(element_shape)
        if not shape or min(shape) < 1:
            raise ValueError(f"element_shape must be positive sizes, got {element_shape!r}.")
        if embed_dim < 1:
            raise ValueError(f"embed_dim must be >= 1, got {embed_dim}.")
        if attention_layers < 0:
            raise ValueError(f"attention_layers must be >= 0, got {attention_layers}.")
        poolings = (pooling,) if isinstance(pooling, str) else tuple(pooling)
        unknown = [p for p in poolings if p not in _POOLINGS]
        if not poolings or unknown:
            raise ValueError(f"pooling must be one or more of {_POOLINGS}, got {pooling!r}.")
        if (attention_layers or element_encoder == "transformer") and embed_dim % num_heads:
            raise ValueError(f"num_heads ({num_heads}) must divide embed_dim ({embed_dim}).")
        self.element_shape: tuple[int, ...] = shape
        self.embed_dim = embed_dim
        self.pooling: tuple[Pooling, ...] = poolings

        if isinstance(element_encoder, nn.Module):
            self.element_encoder: nn.Module = element_encoder
        elif element_encoder == "mlp":
            self.element_encoder = nn.Sequential(
                nn.Flatten(), mlp(math.prod(shape), hidden_sizes, embed_dim, activation)
            )
        elif element_encoder in ("conv", "transformer"):
            if len(shape) != 2:
                raise ValueError(
                    f"element_encoder={element_encoder!r} needs (L, F) elements, got {shape}."
                )
            if element_encoder == "conv":
                self.element_encoder = _TemporalConv(
                    *shape, hidden_sizes, embed_dim, kernel_size, activation
                )
            else:
                self.element_encoder = _TemporalTransformer(
                    *shape, embed_dim, num_heads, dropout, activation
                )
        else:
            raise ValueError(
                "element_encoder must be 'mlp', 'conv', 'transformer' or an nn.Module, "
                f"got {element_encoder!r}."
            )

        self.attention: nn.TransformerEncoder | None = None
        if attention_layers:
            layer = nn.TransformerEncoderLayer(
                embed_dim, num_heads, 2 * embed_dim, dropout, batch_first=True, norm_first=True
            )
            self.attention = nn.TransformerEncoder(
                layer, attention_layers, norm=nn.LayerNorm(embed_dim), enable_nested_tensor=False
            )

    @property
    def pooled_dim(self) -> int:
        """Size of the pooled embedding: `embed_dim` per pooling."""
        return self.embed_dim * len(self.pooling)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> SetEncoding:
        """Encode a batch of sets.

        Args:
            values: `(B, S, *element_shape)`; integer values are converted to float.
            mask: `(B, S)`, nonzero (or True) for an occupied slot.

        Returns:
            `SetEncoding(slots, pooled, mask)`.
        """
        ndim = len(self.element_shape)
        if values.dim() != 2 + ndim or tuple(values.shape[2:]) != self.element_shape:
            raise ValueError(
                f"values must have shape (B, S, *{self.element_shape}), got {tuple(values.shape)}."
            )
        if tuple(mask.shape) != tuple(values.shape[:2]):
            raise ValueError(
                f"mask must have shape {tuple(values.shape[:2])}, got {tuple(mask.shape)}."
            )
        occupied = mask if mask.dtype == torch.bool else mask != 0
        if not values.is_floating_point():
            values = values.float()
        batch, size = occupied.shape
        # Empty slots are replaced before any computation: their content cannot leak.
        clean = torch.where(occupied.view(batch, size, *([1] * ndim)), values, 0.0)
        flat = clean.reshape(batch * size, *self.element_shape)
        h: torch.Tensor = self.element_encoder(flat)
        h = h.reshape(batch, size, -1)
        if self.attention is not None:
            # A set with no element attends over its padding (all keys masked would be
            # NaN); those outputs are zeroed below.
            padding = ~occupied & occupied.any(dim=1, keepdim=True)
            h = self.attention(h, src_key_padding_mask=padding)
        slots = torch.where(occupied.unsqueeze(-1), h, 0.0)
        pools = {"mean": masked_mean, "max": masked_max, "sum": _masked_sum}
        pooled = torch.cat([pools[p](slots, occupied) for p in self.pooling], dim=-1)
        return SetEncoding(slots, pooled, occupied)
