"""Stable-Baselines3 networks for flat-keyed sekai observations (`sekai.UnnestObservation`).

    from stable_baselines3 import PPO
    from sekai.nn.sb3 import MarketFeaturesExtractor, SlotActorCriticPolicy

    env = to_gymnasium(sekai.UnnestObservation(sekai.make("Market-v0")))
    kwargs = {"features_extractor_class": MarketFeaturesExtractor}
    PPO("MultiInputPolicy", env, policy_kwargs=kwargs)  # pooled set features
    PPO(SlotActorCriticPolicy, env)  # one score per asset slot, from that slot's embedding

Install with `pip install "sekai[torch]" stable-baselines3`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any, NamedTuple

import numpy as np
import torch
from torch import nn

try:
    from gymnasium import spaces
    from stable_baselines3.common.policies import MultiInputActorCriticPolicy
    from stable_baselines3.common.preprocessing import get_flattened_obs_dim
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor
    from stable_baselines3.common.type_aliases import Schedule
except ImportError as e:  # pragma: no cover - exercised only without stable-baselines3
    raise ImportError(
        "sekai.nn.sb3 needs Stable-Baselines3. Install it with: pip install stable-baselines3"
    ) from e

from sekai.nn.set_encoder import MaskedSetEncoder, SetEncoding, mlp


class MarketEncoding(NamedTuple):
    """The output of `MarketFeaturesExtractor.encode()`."""

    features: torch.Tensor
    """What `forward()` returns: `(B, features_dim)`."""
    context: torch.Tensor
    """The permutation-invariant part `(B, context_dim)`: every set's pooled embedding,
    then the plain entries' MLP output."""
    sets: dict[str, SetEncoding]
    """Each set's `SetEncoding` (per-slot embeddings, pooled embedding, mask), by name."""


def _set_groups(space: spaces.Dict, sep: str) -> list[tuple[str, str, str, tuple[int, ...]]]:
    """(name, values key, mask key, values shape) of each padded set in the entries of `space`."""
    groups = []
    for key, sub in space.spaces.items():
        if not (key == "values" or key.endswith(sep + "values")):
            continue
        prefix = key[: -len("values")]
        mask_key = prefix + "mask"
        mask = space.spaces.get(mask_key)
        if (
            isinstance(sub, spaces.Box)
            and len(sub.shape) >= 2
            and isinstance(mask, (spaces.MultiBinary, spaces.Box))
            and tuple(np.atleast_1d(mask.shape)) == sub.shape[:1]
        ):
            groups.append((prefix[: -len(sep)] if prefix else "", key, mask_key, sub.shape))
    return groups


class MarketFeaturesExtractor(BaseFeaturesExtractor):
    """A features extractor for `Dict` observations holding padded sets.

    For `UnnestObservation`-wrapped envs such as `Market-v0` and `LimitOrderBook-v0`.
    Each `<name>/values` `(S, *shape)` entry with a `<name>/mask` `(S,)` beside it (a
    `sekai.MaskedBox`, unnested) goes through its own `MaskedSetEncoder`; the other
    entries (Box, Discrete, MultiBinary, ..., as Stable-Baselines3 preprocesses them)
    are flattened, concatenated and passed through an MLP. The features are the sets'
    pooled embeddings and the MLP output, concatenated in key order, so they do not
    depend on which slot holds which element.

    `encode(observations)` also returns the per-slot embeddings, for policies that
    score each slot (an asset's weight from that asset's embedding):
    `SlotActorCriticPolicy` is one. With `slot_set`, `forward()` appends that set's
    per-slot embeddings, flattened, to the features, which is how
    `SlotActorCriticPolicy` reads them through Stable-Baselines3's pipeline.

    Example:
        ```python
        policy_kwargs = {
            "features_extractor_class": MarketFeaturesExtractor,
            "features_extractor_kwargs": {"set_kwargs": {"element_encoder": "conv"}},
        }
        model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs)
        ```

    Args:
        observation_space: The gymnasium `Dict` observation space.
        embed_dim: Size of each set's slot embeddings.
        hidden_sizes: Layer widths of the MLP over the plain entries (none: the entries
            are only flattened).
        set_kwargs: Keyword arguments for every `MaskedSetEncoder` (`element_encoder`,
            `attention_layers`, `pooling`, `hidden_sizes`, ...).
        slot_set: Name of the set whose per-slot embeddings `forward()` appends.
        sep: The separator `UnnestObservation` joined the keys with.
        activation: Activation module class of the MLPs.

    Raises:
        ValueError: No entry to encode, or `slot_set` is not a set of the space.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        embed_dim: int = 64,
        *,
        hidden_sizes: Sequence[int] = (64,),
        set_kwargs: Mapping[str, Any] | None = None,
        slot_set: str | None = None,
        sep: str = "/",
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        if not isinstance(observation_space, spaces.Dict):
            raise TypeError(
                "MarketFeaturesExtractor needs a Dict observation space, got "
                f"{type(observation_space).__name__}."
            )
        groups = _set_groups(observation_space, sep)
        names = [name for name, *_ in groups]
        if slot_set is not None and slot_set not in names:
            raise ValueError(f"slot_set {slot_set!r} is not one of the sets {names}.")
        in_sets = {key for _, values, mask, _ in groups for key in (values, mask)}
        plain = [key for key in observation_space.spaces if key not in in_sets]
        if not groups and not plain:
            raise ValueError("The observation space has no entries to encode.")
        kwargs = dict(set_kwargs or {})
        encoders = [MaskedSetEncoder(shape[1:], embed_dim, **kwargs) for *_, shape in groups]
        plain_dim = sum(int(get_flattened_obs_dim(observation_space[key])) for key in plain)
        sizes = tuple(hidden_sizes)
        plain_out = sizes[-1] if plain and sizes else plain_dim
        context_dim = sum(encoder.pooled_dim for encoder in encoders) + plain_out
        num_slots = 0 if slot_set is None else groups[names.index(slot_set)][3][0]
        super().__init__(observation_space, context_dim + num_slots * embed_dim)

        self.sep = sep
        self.set_names: tuple[str, ...] = tuple(names)
        self.set_keys: tuple[tuple[str, str], ...] = tuple((v, m) for _, v, m, _ in groups)
        self.plain_keys: tuple[str, ...] = tuple(plain)
        self.set_encoders = nn.ModuleList(encoders)
        self.plain_net: nn.Module = (
            mlp(plain_dim, sizes[:-1], sizes[-1], activation) if plain and sizes else nn.Flatten()
        )
        self.embed_dim = embed_dim
        self.context_dim = context_dim
        self.slot_set = slot_set
        self.num_slots = num_slots

    def encode(self, observations: Mapping[str, torch.Tensor]) -> MarketEncoding:
        """Encode a batch: the features, their invariant part and each set's encoding."""
        sets = {
            name: encoder(observations[values], observations[mask])
            for name, (values, mask), encoder in zip(
                self.set_names, self.set_keys, self.set_encoders
            )
        }
        parts = [encoding.pooled for encoding in sets.values()]
        if self.plain_keys:
            flat = torch.cat([observations[key].flatten(1).float() for key in self.plain_keys], 1)
            plain: torch.Tensor = self.plain_net(flat)
            parts.append(plain)
        context = torch.cat(parts, dim=1)
        features = context
        if self.slot_set is not None:
            features = torch.cat([context, sets[self.slot_set].slots.flatten(1)], dim=1)
        return MarketEncoding(features, context, sets)

    def forward(self, observations: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.encode(observations).features


class _ContextCritic(MlpExtractor):
    """SlotActorCriticPolicy's mlp_extractor: the actor gets the features as they are,
    the critic an MLP over their permutation-invariant context only."""

    def __init__(
        self,
        features_dim: int,
        context_dim: int,
        sizes: list[int],
        activation: type[nn.Module],
        device: torch.device | str,
    ) -> None:
        super().__init__(context_dim, {"pi": [], "vf": sizes}, activation, device)
        self.latent_dim_pi = features_dim
        self.context_dim = context_dim

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return super().forward_critic(features[:, : self.context_dim])


class _SlotScores(nn.Module):
    """The action means: one score per slot from `[slot embedding, context]` (the same
    MLP for every slot), then the remaining action entries from the context."""

    def __init__(
        self,
        context_dim: int,
        num_slots: int,
        embed_dim: int,
        action_dim: int,
        sizes: Sequence[int],
        activation: type[nn.Module],
    ) -> None:
        super().__init__()
        self.context_dim, self.num_slots, self.embed_dim = context_dim, num_slots, embed_dim
        hidden = [*mlp(embed_dim + context_dim, sizes[:-1], sizes[-1], activation)] if sizes else []
        self.slot_net = nn.Sequential(
            *hidden, nn.Linear(sizes[-1] if sizes else embed_dim + context_dim, 1)
        )
        self.rest = (
            nn.Linear(context_dim, action_dim - num_slots) if action_dim > num_slots else None
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        context = features[:, : self.context_dim]
        slots = features[:, self.context_dim :].reshape(-1, self.num_slots, self.embed_dim)
        joint = torch.cat([slots, context.unsqueeze(1).expand(-1, self.num_slots, -1)], dim=-1)
        scores: torch.Tensor = self.slot_net(joint).squeeze(-1)
        if self.rest is None:
            return scores
        rest: torch.Tensor = self.rest(context)
        return torch.cat([scores, rest], dim=1)


class SlotActorCriticPolicy(MultiInputActorCriticPolicy):
    """An actor-critic policy whose actor scores each slot of a set from its embedding.

    For a `Box` action space with one entry per slot of `slot_set`, first, then possibly
    more (`Market-v0`: a score per asset slot, then cash). The features come from
    `MarketFeaturesExtractor(slot_set=slot_set)`. The mean of slot `i`'s action is the
    same MLP applied to slot `i`'s embedding and the context (every set's pooled
    embedding and the other entries), so the actor is permutation-equivariant over
    slots: an asset's weight follows its own features, whichever slot it is in. The
    other action entries and the critic read the context only (invariant).

    Use it like any SB3 policy class, with `PPO`, `A2C` (or `MaskablePPO`'s
    multi-input policy for discrete actions, which this does not cover):

        ```python
        env = to_gymnasium(sekai.UnnestObservation(sekai.make("Market-v0")))
        model = PPO(SlotActorCriticPolicy, env, policy_kwargs={"slot_set": "assets"})
        ```

    Args:
        observation_space: The gymnasium `Dict` observation space.
        action_space: A 1-D `Box` with at least as many entries as `slot_set` has slots.
        lr_schedule: Learning rate schedule (passed by the algorithm).
        slot_set: Name of the set whose slots are scored (`"assets"` for `Market-v0`).
        head_sizes: Hidden layer widths of the per-slot scoring MLP.
        net_arch: Critic layer widths: a list, or `{"vf": [...]}` (`"pi"`, if given, must
            be empty: the actor is the scoring MLP).
        features_extractor_kwargs: More `MarketFeaturesExtractor` arguments.
        **kwargs: Other `ActorCriticPolicy` arguments (`activation_fn`, `log_std_init`,
            `optimizer_class`, ...).
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space[Any],
        lr_schedule: Schedule,
        *,
        slot_set: str = "assets",
        head_sizes: Sequence[int] = (64,),
        net_arch: list[int] | dict[str, list[int]] | None = None,
        features_extractor_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(action_space, spaces.Box) or len(action_space.shape) != 1:
            raise TypeError(
                f"SlotActorCriticPolicy needs a 1-D Box action space, got {action_space}."
            )
        self.action_dim = action_space.shape[0]
        extractor = kwargs.pop("features_extractor_class", MarketFeaturesExtractor)
        if not issubclass(extractor, MarketFeaturesExtractor):
            raise TypeError(
                "SlotActorCriticPolicy's features extractor is a MarketFeaturesExtractor."
            )
        if isinstance(net_arch, dict):
            if net_arch.get("pi"):
                raise ValueError("net_arch['pi'] must be empty: the actor is the scoring MLP.")
            vf = list(net_arch.get("vf", [64, 64]))
        else:
            vf = [64, 64] if net_arch is None else list(net_arch)
        self.slot_set = slot_set
        self.head_sizes = tuple(head_sizes)
        self._vf_sizes = vf
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch={"pi": [], "vf": vf},
            features_extractor_class=extractor,
            features_extractor_kwargs={**(features_extractor_kwargs or {}), "slot_set": slot_set},
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        extractor = self.features_extractor
        assert isinstance(extractor, MarketFeaturesExtractor)
        if self.action_dim < extractor.num_slots:
            raise ValueError(
                f"The action space {self.action_space} has fewer entries than {self.slot_set!r} "
                f"has slots ({extractor.num_slots})."
            )
        self.mlp_extractor = _ContextCritic(
            self.features_dim,
            extractor.context_dim,
            self._vf_sizes,
            self.activation_fn,
            self.device,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        extractor = self.features_extractor
        assert isinstance(extractor, MarketFeaturesExtractor)
        head = _SlotScores(
            extractor.context_dim,
            extractor.num_slots,
            extractor.embed_dim,
            self.action_dim,
            self.head_sizes,
            self.activation_fn,
        )
        if self.ortho_init:  # as SB3 does: sqrt(2) inside, 0.01 for the action outputs
            head.apply(partial(self.init_weights, gain=np.sqrt(2)))
            for out in (head.slot_net[-1], head.rest):
                if out is not None:
                    out.apply(partial(self.init_weights, gain=0.01))
        self.action_net = head
        # super() built the optimizer around the linear action_net this replaces.
        optimizer_class: Any = self.optimizer_class  # takes lr, unlike Optimizer itself
        self.optimizer = optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data: dict[str, Any] = super()._get_constructor_parameters()
        kwargs = dict(data["features_extractor_kwargs"] or {})
        kwargs.pop("slot_set", None)
        data.update(
            slot_set=self.slot_set,
            head_sizes=self.head_sizes,
            net_arch={"pi": [], "vf": self._vf_sizes},
            features_extractor_kwargs=kwargs,
        )
        return data
