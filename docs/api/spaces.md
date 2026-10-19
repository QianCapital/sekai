# Spaces

Space classes are exported from `sekai` and `sekai.spaces`; the utility functions from
`sekai.spaces` (also available as `sekai.space_utils`). See the [Spaces guide](../guide/spaces.md).

## Base class

::: sekai.Space

## Fundamental spaces

::: sekai.Box

::: sekai.Discrete

::: sekai.MultiDiscrete

::: sekai.MultiBinary

## Composite spaces

::: sekai.Dict

::: sekai.Tuple

## Variable-size spaces

See [Variable-size observations](../guide/dynamic.md).

::: sekai.Sequence

::: sekai.MaskedBox

## Utilities

::: sekai.spaces.utils
    options:
      show_root_heading: false
      members:
        - flatdim
        - flatten
        - unflatten
        - flatten_space
        - batch_space
        - concatenate
        - iterate
        - zero_sample
        - space_from_jsonable
        - register_space_type
