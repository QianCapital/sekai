# Vector

Exported from `sekai` and `sekai.vector` (`shared` from `sekai.vector` only). See the
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

::: sekai.vector.shared

## Batched environments

`sekai.vector.batch` (also exported from `sekai.vector`). See
[Batched and differentiable environments](../finance/batched.md#the-batched-api).

::: sekai.vector.BatchEnv
    options:
      members:
        - reset
        - step
        - observe
        - detach_state
        - frozen
        - _reset_lanes
        - _observe
        - _step
        - _detach
      filters: []
      show_if_no_docstring: true

::: sekai.vector.BatchReset

::: sekai.vector.BatchStep

::: sekai.vector.BatchVecEnv

::: sekai.vector.UnbatchedEnv

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
