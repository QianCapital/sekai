# Core

The environment base classes and their result types. See the
[Environments](../guide/environments.md), [Wrappers](../guide/wrappers.md) and
[Multi-agent](../guide/multi-agent.md) guides.

## Environments

::: sekai.Env
    options:
      members:
        - reset
        - step
        - async_reset
        - async_step
        - close
        - rng
        - _set_rng
        - unwrapped
      filters: []
      show_if_no_docstring: true

::: sekai.MultiAgentEnv
    options:
      show_if_no_docstring: true
      members:
        - reset
        - step
        - async_reset
        - async_step
        - observation_space
        - action_space
        - num_agents
        - max_num_agents
        - close

## Wrappers

::: sekai.Wrapper
    options:
      members:
        - __iter__
        - __len__
      filters: []

::: sekai.ObservationWrapper

::: sekai.RewardWrapper

::: sekai.ActionWrapper

## Results

::: sekai.StepResult

::: sekai.ResetResult

::: sekai.MAStepResult

::: sekai.MAResetResult
