# Compatibility

`sekai.compat` bridges sekai and gymnasium, sekai vector envs and Stable-Baselines3, and
multi-agent sekai envs and PettingZoo. The bridges need the optional dependencies:
`pip install "sekai[gymnasium]"`, plus `pip install stable-baselines3` for `to_sb3_vec_env`, or
`pip install "sekai[pettingzoo]"` (which includes gymnasium). See
[Migrating from gymnasium](../guide/gymnasium.md),
[Training agents](../guide/training.md#vector-environments-to_sb3_vec_env) and
[Multi-agent](../guide/multi-agent.md#pettingzoo).

## Functions

::: sekai.compat.to_gymnasium

::: sekai.compat.from_gymnasium

::: sekai.compat.space_to_gymnasium

::: sekai.compat.space_from_gymnasium

::: sekai.compat.to_sb3_vec_env

::: sekai.compat.to_pettingzoo

::: sekai.compat.from_pettingzoo

## Classes

These are importable from `sekai.compat` (loaded on first access).

::: sekai.compat._gymnasium.SekaiToGymnasium

::: sekai.compat._gymnasium.GymnasiumToSekai

::: sekai.compat._gymnasium.GymnasiumRenderer

::: sekai.compat._stable_baselines3.SekaiToSB3VecEnv

::: sekai.compat._pettingzoo.SekaiToPettingZoo

::: sekai.compat._pettingzoo.PettingZooToSekai
