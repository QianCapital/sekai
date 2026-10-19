# Utilities

## Environment checker

See [Checking environments](../guide/checking.md).

::: sekai.check_env

## Running statistics

Used by the normalisation wrappers; see
[Non-stationary statistics](../guide/time.md#non-stationary-statistics).

::: sekai.RunningMeanStd

::: sekai.ExponentialMovingStats

## Observation normalisation

The engine shared by `NormalizeObservation` and `VecNormalizeObservation`.

::: sekai.utils.normalization.ObservationNormalizer

## Action masks

Exported from `sekai.utils` (and `get_action_mask` from `sekai`). See
[Action masks](../guide/dynamic.md#action-masks).

::: sekai.utils.masking
    options:
      show_root_heading: false
      members:
        - get_action_mask
        - supports_action_mask
        - flatten_action_mask
        - action_mask_space

## Seeding

Exported from `sekai` and `sekai.utils`.

::: sekai.np_random

::: sekai.make_seed_sequence
