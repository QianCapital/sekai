# Registry

Exported from `sekai` and `sekai.registry`. See the [Registry guide](../guide/registry.md).

## Functions

::: sekai.register

::: sekai.make

::: sekai.make_vec

::: sekai.spec

::: sekai.all_specs

::: sekai.deregister

::: sekai.register_namespace

::: sekai.pprint_registry

## Specs

::: sekai.EnvSpec

::: sekai.WrapperSpec

## The registry class

The functions above operate on a module-level `EnvRegistry`; create your own instance for an
isolated registry (for example in tests).

::: sekai.registry.EnvRegistry
