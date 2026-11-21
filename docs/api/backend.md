# Backend

`sekai.backend`: the array operations sekai's batched finance core is written in, on NumPy
arrays or torch tensors. See
[Batched and differentiable environments](../finance/batched.md#backends-and-gradients).

::: sekai.backend
    options:
      members:
        - backend_of
        - backend_and_like
        - get_backend
        - is_tensor
        - to_numpy
        - numpy_backend

::: sekai.backend.Backend

::: sekai.backend.NumpyBackend
