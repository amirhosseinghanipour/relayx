"""RelayX compute engine."""
from typing import Dict, Callable, Any, Tuple
from .tensor import RelayTensor
from .backends import DEFAULT_BACKENDS, OperationNotSupportedError

class RelayX:
    """Hardware-agnostic compute abstraction.

    Args:
        default_backend: The default backend to use ('jax' or 'torch'). Defaults to 'jax'.
    """
    def __init__(self, default_backend: str = "jax"):
        self.backends: Dict[str, Callable] = DEFAULT_BACKENDS.copy()
        if default_backend not in self.backends:
            raise ValueError(f"Unsupported default backend: {default_backend}. Available: {list(self.backends.keys())}")
        self.current_backend = default_backend

    def register_backend(self, name: str, backend_fn: Callable) -> None:
        """Register a new backend.

        Args:
            name: Backend identifier (e.g., 'cuda').
            backend_fn: Function implementing the backend operations.
        """
        self.backends[name] = backend_fn

    def set_backend(self, backend: str) -> None:
        """Switch the active backend.

        Args:
            backend: Backend identifier to switch to.
        """
        if backend not in self.backends:
            raise ValueError(f"Backend '{backend}' not registered. Available: {list(self.backends.keys())}")
        self.current_backend = backend

    def compute(self, op: str, *args: Any, backend: str = None, **kwargs: Any) -> RelayTensor:
        """Execute a tensor operation on the specified or current backend.

        Args:
            op: Operation name (e.g., 'matmul', 'conv2d', 'add').
            *args: Tensor arguments.
            backend: Optional backend override. Uses current_backend if None.
            **kwargs: Operation-specific parameters.

        Returns:
            RelayTensor with the operation result.

        Raises:
            OperationNotSupportedError: If the operation isn't supported by the backend.
        """
        backend = backend or self.current_backend
        if backend not in self.backends:
            raise ValueError(f"Backend '{backend}' not registered. Available: {list(self.backends.keys())}")
        try:
            result = self.backends[backend](op, *args, **kwargs)
            return RelayTensor(result, backend)
        except OperationNotSupportedError as e:
            raise OperationNotSupportedError(f"{str(e)} Current backend: {backend}")

    def matmul(self, a: RelayTensor, b: RelayTensor, backend: str = None) -> RelayTensor:
        """Matrix multiplication.

        Args:
            a: First tensor.
            b: Second tensor.
            backend: Optional backend override.
        """
        return self.compute("matmul", a, b, backend=backend)

    def conv2d(self, inputs: RelayTensor, kernel: RelayTensor, strides: Tuple[int, int] = (1, 1),
               padding: str = "VALID", backend: str = None) -> RelayTensor:
        """2D convolution.

        Args:
            inputs: Input tensor.
            kernel: Convolution kernel.
            strides: Stride of the convolution.
            padding: Padding mode ('VALID' or 'SAME').
            backend: Optional backend override.
        """
        return self.compute("conv2d", inputs, kernel, strides=strides, padding=padding, backend=backend)

    def add(self, a: RelayTensor, b: RelayTensor, backend: str = None) -> RelayTensor:
        """Element-wise addition.

        Args:
            a: First tensor.
            b: Second tensor.
            backend: Optional backend override.
        """
        return self.compute("add", a, b, backend=backend)