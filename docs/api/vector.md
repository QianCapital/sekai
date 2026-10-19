# Vector

Exported from `sekai` and `sekai.vector`. See the
[Vector environments guide](../guide/vector.md).

## Vector environments

::: sekai.VecEnv

::: sekai.SyncVecEnv

::: sekai.AsyncVecEnv

::: sekai.ProcVecEnv

::: sekai.NativeVecEnv
    options:
      members:
        - _reset_envs
        - _observe
        - _step
        - _reset_infos
        - rng
      filters: []
      show_if_no_docstring: true

## Results

::: sekai.VecStepResult

::: sekai.VecResetResult

::: sekai.stack_infos

## Wrapper base classes

::: sekai.VecWrapper

::: sekai.VecObservationWrapper

::: sekai.VecRewardWrapper

::: sekai.VecActionWrapper

## Ready-made wrappers

::: sekai.VecRecordEpisodeStatistics

::: sekai.VecNormalizeObservation

::: sekai.VecNormalizeReward

::: sekai.VecTransformObservation

::: sekai.VecTransformReward
