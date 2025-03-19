"""Tensor wrapper for RelayX."""
from typing import Any
import jax.numpy as jnp
import torch
import numpy as np

class RelayTensor:
    """A unified tensor wrapper for RelayX.

    Args:
        data: The underlying tensor data (JAX, PyTorch, or NumPy array).
        backend: The backend identifier ('jax', 'torch', 'numpy'). Defaults to 'jax'.
    """
    def __init__(self, data: Any, backend: str = "jax"):
        self._data = data
        self._backend = backend
        self._type = self._infer_type(data)

    def _infer_type(self, data: Any) -> str:
        """Infer the tensor type from the data."""
        if isinstance(data, jnp.ndarray):
            return "jax"
        elif isinstance(data, torch.Tensor):
            return "torch"
        elif isinstance(data, np.ndarray):
            return "numpy"
        raise ValueError(f"Unsupported tensor type: {type(data)}")

    def to_backend(self, backend: str) -> "RelayTensor":
        """Convert the tensor to another backend."""
        if self._backend == backend:
            return self
        if self._type == "torch":
            intermediate = self._data.detach().cpu().numpy()
        elif self._type == "jax":
            intermediate = np.array(self._data)
        else:
            intermediate = self._data

        if backend == "jax":
            data = jnp.array(intermediate)
        elif backend == "torch":
            data = torch.from_numpy(intermediate)
        elif backend == "numpy":
            data = intermediate
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        return RelayTensor(data, backend)

    @property
    def data(self) -> Any:
        """Get the raw tensor data."""
        return self._data

    @property
    def shape(self) -> tuple:
        """Get the tensor shape."""
        return self._data.shape

    def __repr__(self) -> str:
        return f"RelayTensor(backend={self._backend}, shape={self.shape}, type={self._type})"