# Wrappers

All wrappers are importable from `sekai.wrappers` and from the top-level `sekai` package.
`RecordFrames` lives in [`sekai.rendering`](rendering.md#sekai.rendering.RecordFrames). See the
[Wrappers guide](../guide/wrappers.md) for an overview table.

## Episode control

::: sekai.wrappers.TimeLimit

::: sekai.wrappers.AutoReset

::: sekai.wrappers.OrderEnforcing

::: sekai.wrappers.RecordEpisodeStatistics

## Observations

::: sekai.wrappers.FlattenObservation

::: sekai.wrappers.NormalizeObservation

::: sekai.wrappers.FrameStackObservation

::: sekai.wrappers.TransformObservation

::: sekai.wrappers.RescaleObservation

::: sekai.wrappers.DtypeObservation

::: sekai.wrappers.FilterObservation

::: sekai.wrappers.TimeAwareObservation

::: sekai.wrappers.DelayObservation

::: sekai.wrappers.PadSequenceObservation

::: sekai.wrappers.ActionMaskObservation

## Actions

::: sekai.wrappers.ClipAction

::: sekai.wrappers.RescaleAction

::: sekai.wrappers.StickyAction

::: sekai.wrappers.ActionRepeat

::: sekai.wrappers.DelayAction

## Rewards

::: sekai.wrappers.NormalizeReward

::: sekai.wrappers.TransformReward

::: sekai.wrappers.ClipReward
