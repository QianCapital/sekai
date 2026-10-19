# Compatibility

`sekai.compat` bridges sekai and gymnasium, and multi-agent sekai envs and PettingZoo. The
bridges need the optional dependencies: `pip install "sekai[gymnasium]"`, or
`pip install "sekai[pettingzoo]"` (which includes gymnasium). See
[Migrating from gymnasium](../guide/gymnasium.md) and [Multi-agent](../guide/multi-agent.md#pettingzoo).

## Functions

::: sekai.compat.to_gymnasium

::: sekai.compat.from_gymnasium

::: sekai.compat.space_to_gymnasium

::: sekai.compat.space_from_gymnasium

::: sekai.compat.to_pettingzoo

::: sekai.compat.from_pettingzoo

## Classes

These are importable from `sekai.compat` (loaded on first access).

::: sekai.compat._gymnasium.SekaiToGymnasium

::: sekai.compat._gymnasium.GymnasiumToSekai

::: sekai.compat._gymnasium.GymnasiumRenderer

::: sekai.compat._pettingzoo.SekaiToPettingZoo

::: sekai.compat._pettingzoo.PettingZooToSekai
