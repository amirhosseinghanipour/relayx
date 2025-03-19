"""Backend implementations for RelayX."""
from typing import Callable, Tuple, Any
import jax
import jax.numpy as jnp
from jax import lax
import torch

class OperationNotSupportedError(Exception):
    """Raised when an operation is not supported by a backend."""
    pass

def jax_backend(op: str, *args: Any, **kwargs: Any) -> Any:
    """JAX/XLA backend implementation.

    Args:
        op: Operation name ('matmul', 'conv2d', 'add').
        *args: Tensor arguments (converted to JAX arrays).
        **kwargs: Additional operation parameters.

    Returns:
        Resulting tensor in JAX format.

    Raises:
        OperationNotSupportedError: If the operation is not supported.
    """
    args = tuple(arg.to_backend("jax").data if isinstance(arg, "RelayTensor") else arg for arg in args)
    if op == "matmul":
        if len(args) != 2:
            raise ValueError(f"matmul requires 2 arguments, got {len(args)}")
        return jax.jit(jnp.matmul)(*args)
    elif op == "conv2d":
        if len(args) != 2:
            raise ValueError(f"conv2d requires 2 arguments (input, kernel), got {len(args)}")
        lhs, rhs = args
        strides = kwargs.get("strides", (1, 1))
        padding = kwargs.get("padding", "VALID")
        dimension_numbers = lax.conv_dimension_numbers(lhs.shape, rhs.shape, ("NHWC", "HWIO", "NHWC"))
        conv_fn = jax.jit(
            lax.conv_general_dilated,
            static_argnames=["window_strides", "padding", "dimension_numbers"]
        )
        return conv_fn(lhs, rhs, window_strides=strides, padding=padding, dimension_numbers=dimension_numbers)
    elif op == "add":
        if len(args) != 2:
            raise ValueError(f"add requires 2 arguments, got {len(args)}")
        return jax.jit(jnp.add)(*args)
    raise OperationNotSupportedError(
        f"Operation '{op}' not supported in JAX backend. Supported ops: matmul, conv2d, add. "
        "Register a custom backend or extend this one."
    )

def torch_backend(op: str, *args: Any, **kwargs: Any) -> Any:
    """PyTorch backend implementation (CPU-only for now).

    Args:
        op: Operation name ('matmul', 'conv2d', 'add').
        *args: Tensor arguments (converted to PyTorch tensors).
        **kwargs: Additional operation parameters.

    Returns:
        Resulting tensor in PyTorch format.

    Raises:
        OperationNotSupportedError: If the operation is not supported.
    """
    args = tuple(arg.to_backend("torch").data if isinstance(arg, "RelayTensor") else arg for arg in args)
    if op == "matmul":
        if len(args) != 2:
            raise ValueError(f"matmul requires 2 arguments, got {len(args)}")
        return torch.matmul(*args)
    elif op == "conv2d":
        if len(args) != 2:
            raise ValueError(f"conv2d requires 2 arguments (input, kernel), got {len(args)}")
        inputs, kernel = args
        # Permute kernel from (H, W, I, O) to (O, I, H, W)
        kernel = kernel.permute(3, 2, 0, 1)
        # Permute input to NCHW if needed
        if inputs.shape[1] != inputs.shape[3] and inputs.shape[1] != 3:
            inputs = inputs.permute(0, 3, 1, 2)
        # Perform convolution
        result = torch.nn.functional.conv2d(
            inputs, kernel, stride=kwargs.get("strides", (1, 1)),
            padding=0 if kwargs.get("padding", "VALID") == "VALID" else "same"
        )
        # Permute result back to NHWC
        if inputs.shape[1] != inputs.shape[3] and inputs.shape[1] != 3:
            result = result.permute(0, 2, 3, 1)
        return result
    elif op == "add":
        if len(args) != 2:
            raise ValueError(f"add requires 2 arguments, got {len(args)}")
        return torch.add(*args)
    raise OperationNotSupportedError(
        f"Operation '{op}' not supported in Torch backend. Supported ops: matmul, conv2d, add. "
        "Register a custom backend or extend this one."
    )

DEFAULT_BACKENDS = {
    "jax": jax_backend,
    "torch": torch_backend
}