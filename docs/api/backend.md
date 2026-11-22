# Backend

`sekai.backend`: the array operations sekai's batched financial markets core is written in,
on NumPy arrays or torch tensors. See
[Batched and differentiable environments](../finance/batched.md#backends-and-gradients).

::: sekai.backend
    options:
      members:
        - backend_of
        - backend_and_like
        - get_backend
        - is_tensor
        - to_numpy

::: sekai.backend.Backend

::: sekai.backend.NumpyBackend

## sekai 0.2's backend protocol

`sekai.backend.BackendOps`, `sekai.backend.default_backend` and the modules
`sekai.backend.protocol` and `sekai.backend.numpy_backend` are sekai 0.2's backend API.
They are deprecated but still work, with a `DeprecationWarning`, and `NumpyBackend` still
implements `BackendOps` (`ones`, `concatenate`, `astype`, `random_uniform`,
`random_integers`, `array_equal`, `zeros(shape, dtype)`). New code uses the `Backend`
interface above.
