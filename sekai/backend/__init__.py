from sekai.backend.protocol import BackendOps
from sekai.backend.numpy_backend import NumpyBackend

default_backend: BackendOps = NumpyBackend()

__all__ = ["BackendOps", "NumpyBackend", "default_backend"]
