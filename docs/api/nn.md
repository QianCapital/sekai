# Neural networks

`sekai.nn`: reference PyTorch networks for sekai observations. PyTorch is an optional
dependency (`pip install "sekai[torch]"`); `import sekai` never imports it, and
`import sekai.nn` without it raises `ImportError` with the install command. See
[Training agents](../guide/training.md).

## Set encoder

Exported from `sekai.nn`.

::: sekai.nn.MaskedSetEncoder
    options:
      members:
        - pooled_dim
        - forward

::: sekai.nn.SetEncoding

::: sekai.nn.masked_mean

::: sekai.nn.masked_max

::: sekai.nn.mlp

## Stable-Baselines3

`sekai.nn.sb3`, imported on its own (`from sekai.nn.sb3 import ...`) because it also needs
Stable-Baselines3 (`pip install stable-baselines3`). For observations from
`sekai.UnnestObservation` (see [Wrappers](wrappers.md)).

::: sekai.nn.sb3.MarketFeaturesExtractor
    options:
      members:
        - encode

::: sekai.nn.sb3.MarketEncoding

::: sekai.nn.sb3.SlotActorCriticPolicy
    options:
      members: false
