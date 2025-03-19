from typing import Any, Dict, Callable, Tuple
import jax
import jax.numpy as jnp
from jax import lax
import torch
import numpy as np

class RelayTensor:
    def __init__(self, data: Any, backend: str = "jax"):
        self._data = data
        self._backend = backend
        self._type = self._infer_type(data)

    def _infer_type(self, data: Any) -> str:
        if isinstance(data, jnp.ndarray):
            return "jax"
        elif isinstance(data, torch.Tensor):
            return "torch"
        elif isinstance(data, np.ndarray):
            return "numpy"
        raise ValueError(f"Unsupported tensor type: {type(data)}")

    def to_backend(self, backend: str) -> "RelayTensor":
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
        return self._data

    @property
    def shape(self) -> tuple:
        return self._data.shape

    def __repr__(self) -> str:
        return f"RelayTensor(backend={self._backend}, shape={self.shape}, type={self._type})"

class OperationNotSupportedError(Exception):
    pass

def jax_backend(op: str, *args: Any, **kwargs: Any) -> Any:
    args = tuple(arg.to_backend("jax").data if isinstance(arg, RelayTensor) else arg for arg in args)
    if op == "matmul":
        if len(args) != 2:
            raise ValueError(f"matmul requires 2 arguments, got {len(args)}")
        return jax.jit(jnp.matmul)(*args)
    elif op == "conv2d":
        if len(args) != 2:
            raise ValueError(f"conv2d requires 2 arguments, got {len(args)}")
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
        f"Operation '{op}' not supported in JAX backend."
    )

def torch_backend(op: str, *args: Any, **kwargs: Any) -> Any:
    args = tuple(arg.to_backend("torch").data if isinstance(arg, RelayTensor) else arg for arg in args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = tuple(arg.to(device) for arg in args)
    if op == "matmul":
        if len(args) != 2:
            raise ValueError(f"matmul requires 2 arguments, got {len(args)}")
        return torch.matmul(*args)
    elif op == "conv2d":
        if len(args) != 2:
            raise ValueError(f"conv2d requires 2 arguments, got {len(args)}")
        inputs, kernel = args
        kernel = kernel.permute(3, 2, 0, 1)
        if inputs.shape[1] != inputs.shape[3] and inputs.shape[1] != 3:
            inputs = inputs.permute(0, 3, 1, 2)
        result = torch.nn.functional.conv2d(
            inputs, kernel, stride=kwargs.get("strides", (1, 1)),
            padding=0 if kwargs.get("padding", "VALID") == "VALID" else "same"
        )
        if inputs.shape[1] != inputs.shape[3] and inputs.shape[1] != 3:
            result = result.permute(0, 2, 3, 1)
        return result
    elif op == "add":
        if len(args) != 2:
            raise ValueError(f"add requires 2 arguments, got {len(args)}")
        return torch.add(*args)
    raise OperationNotSupportedError(
        f"Operation '{op}' not supported in Torch backend."
    )

DEFAULT_BACKENDS = {"jax": jax_backend, "torch": torch_backend}

class RelayX:
    def __init__(self, default_backend: str = "jax"):
        self.backends: Dict[str, Callable] = DEFAULT_BACKENDS.copy()
        if default_backend not in self.backends:
            raise ValueError(f"Unsupported default backend: {default_backend}")
        self.current_backend = default_backend

    def register_backend(self, name: str, backend_fn: Callable) -> None:
        self.backends[name] = backend_fn

    def set_backend(self, backend: str) -> None:
        if backend not in self.backends:
            raise ValueError(f"Backend '{backend}' not registered")
        self.current_backend = backend

    def compute(self, op: str, *args: Any, backend: str = None, **kwargs: Any) -> RelayTensor:
        backend = backend or self.current_backend
        if backend not in self.backends:
            raise ValueError(f"Backend '{backend}' not registered")
        try:
            result = self.backends[backend](op, *args, **kwargs)
            return RelayTensor(result, backend)
        except OperationNotSupportedError as e:
            raise OperationNotSupportedError(f"{str(e)} Current backend: {backend}")

    def matmul(self, a: RelayTensor, b: RelayTensor, backend: str = None) -> RelayTensor:
        return self.compute("matmul", a, b, backend=backend)

    def conv2d(self, inputs: RelayTensor, kernel: RelayTensor, strides: Tuple[int, int] = (1, 1),
               padding: str = "VALID", backend: str = None) -> RelayTensor:
        return self.compute("conv2d", inputs, kernel, strides=strides, padding=padding, backend=backend)

    def add(self, a: RelayTensor, b: RelayTensor, backend: str = None) -> RelayTensor:
        return self.compute("add", a, b, backend=backend)

def test_relayx():
    relayx = RelayX()
    x_jax = RelayTensor(jnp.ones((2, 3)))
    w_jax = RelayTensor(jnp.ones((3, 4)))
    out_jax = relayx.matmul(x_jax, w_jax)
    print("JAX matmul:", out_jax.shape, out_jax.data)
    x_torch = RelayTensor(torch.ones(2, 3))
    w_torch = RelayTensor(torch.ones(3, 4))
    out_torch = relayx.matmul(x_torch, w_torch, backend="torch")
    print("Torch matmul:", out_torch.shape, out_torch.data)
    img_jax = RelayTensor(jnp.ones((1, 28, 28, 1)))
    kernel_jax = RelayTensor(jnp.ones((3, 3, 1, 2)))
    conv_jax = relayx.conv2d(img_jax, kernel_jax, padding="SAME")
    print("JAX conv2d:", conv_jax.shape, conv_jax.data)
    img_torch = RelayTensor(torch.ones(1, 28, 28, 1))
    conv_mixed = relayx.conv2d(img_torch, kernel_jax.to_backend("torch"), backend="torch", padding="SAME")
    print("Torch conv2d (mixed):", conv_mixed.shape, conv_mixed.data)
    a = RelayTensor(jnp.ones((2, 2)))
    b = RelayTensor(jnp.ones((2, 2)))
    add_jax = relayx.add(a, b)
    print("JAX add:", add_jax.shape, add_jax.data)

test_relayx()