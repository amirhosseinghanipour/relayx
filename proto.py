from typing import Any, Dict, Callable, Tuple, Union, Optional
from functools import partial
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
    elif op == "relu":
        if len(args) != 1:
            raise ValueError(f"relu requires 1 argument, got {len(args)}")
        return jax.jit(jnp.maximum)(args[0], 0)
    elif op == "max_pool2d":
        if len(args) != 1:
            raise ValueError(f"max_pool2d requires 1 argument, got {len(args)}")
        window_shape = kwargs.get("kernel_size", (2, 2))
        strides = kwargs.get("strides", window_shape)
        padding = kwargs.get("padding", "VALID")
        def pool_fn(x):
            return lax.reduce_window(x, -jnp.inf, jax.lax.max, (1, *window_shape, 1), (1, *strides, 1), padding)
        return jax.jit(pool_fn)(args[0])
    elif op == "mean":
        if len(args) != 1:
            raise ValueError(f"mean requires 1 argument, got {len(args)}")
        axis = kwargs.get("axis", None)
        return jax.jit(jnp.mean)(args[0], axis=axis)
    elif op == "softmax":
        if len(args) != 1:
            raise ValueError(f"softmax requires 1 argument, got {len(args)}")
        axis = kwargs.get("axis", -1)
        softmax_fn = partial(jax.nn.softmax, axis=axis)
        return jax.jit(softmax_fn)(args[0])
    elif op == "batch_norm2d":
        if len(args) != 1:
            raise ValueError(f"batch_norm2d requires 1 argument, got {len(args)}")
        x = args[0]
        eps = kwargs.get("eps", 1e-5)
        momentum = kwargs.get("momentum", 0.1)
        mean = jnp.mean(x, axis=(0, 1, 2), keepdims=True)
        var = jnp.var(x, axis=(0, 1, 2), keepdims=True)
        return jax.jit(lambda x: (x - mean) / jnp.sqrt(var + eps))(x)
    elif op == "transpose":
        if len(args) != 1:
            raise ValueError(f"transpose requires 1 argument, got {len(args)}")
        dims = kwargs.get("dims")
        if dims is None:
            raise ValueError("transpose requires 'dims' argument")
        return jax.jit(lambda x, dims=dims: jnp.transpose(x, dims))(args[0])
    elif op == "dropout":
        if len(args) != 1:
            raise ValueError(f"dropout requires 1 argument, got {len(args)}")
        x = args[0]
        p = kwargs.get("p", 0.5)
        training = kwargs.get("training", True)
        seed = kwargs.get("seed", 0)
        if not training:
            return x
        key = jax.random.PRNGKey(seed)
        mask = jax.random.bernoulli(key, 1 - p, x.shape)
        return jax.jit(lambda x, m: x * m / (1 - p))(x, mask)
    elif op == "linear":
        if len(args) < 2 or len(args) > 3:
            raise ValueError(f"linear requires 2 or 3 arguments, got {len(args)}")
        x, w = args[0], args[1]
        b = args[2] if len(args) == 3 else None
        out = jax.jit(jnp.dot)(x, w)
        if b is not None:
            out = jax.jit(jnp.add)(out, b)
        return out
    elif op == "sigmoid":
        if len(args) != 1:
            raise ValueError(f"sigmoid requires 1 argument, got {len(args)}")
        return jax.jit(jax.nn.sigmoid)(args[0])
    elif op == "tanh":
        if len(args) != 1:
            raise ValueError(f"tanh requires 1 argument, got {len(args)}")
        return jax.jit(jnp.tanh)(args[0])
    
    raise OperationNotSupportedError(f"Operation '{op}' not supported in JAX backend.")

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
    elif op == "relu":
        if len(args) != 1:
            raise ValueError(f"relu requires 1 argument, got {len(args)}")
        return torch.nn.functional.relu(args[0])
    elif op == "max_pool2d":
        if len(args) != 1:
            raise ValueError(f"max_pool2d requires 1 argument, got {len(args)}")
        inputs = args[0]
        kernel_size = kwargs.get("kernel_size", (2, 2))
        strides = kwargs.get("strides", kernel_size)
        padding = 0 if kwargs.get("padding", "VALID") == "VALID" else "same"
        if inputs.shape[1] != inputs.shape[3] and inputs.shape[1] != 3:
            inputs = inputs.permute(0, 3, 1, 2)
        result = torch.nn.functional.max_pool2d(
            inputs, kernel_size=kernel_size, stride=strides, padding=padding
        )
        if inputs.shape[1] != inputs.shape[3] and inputs.shape[1] != 3:
            result = result.permute(0, 2, 3, 1)
        return result
    elif op == "mean":
        if len(args) != 1:
            raise ValueError(f"mean requires 1 argument, got {len(args)}")
        dim = kwargs.get("axis", None)
        return torch.mean(args[0], dim=dim)
    elif op == "softmax":
        if len(args) != 1:
            raise ValueError(f"softmax requires 1 argument, got {len(args)}")
        dim = kwargs.get("axis", -1)
        return torch.nn.functional.softmax(args[0], dim=dim)
    elif op == "batch_norm2d":
        if len(args) != 1:
            raise ValueError(f"batch_norm2d requires 1 argument, got {len(args)}")
        inputs = args[0]
        eps = kwargs.get("eps", 1e-5)
        momentum = kwargs.get("momentum", 0.1)
        needs_permute = inputs.shape[1] != inputs.shape[3]
        if needs_permute:
            inputs = inputs.permute(0, 3, 1, 2)
        bn = torch.nn.BatchNorm2d(inputs.shape[1], eps=eps, momentum=momentum, affine=False, track_running_stats=False)
        result = bn(inputs)
        if needs_permute:
            result = result.permute(0, 2, 3, 1)
        return result
    elif op == "transpose":
        if len(args) != 1:
            raise ValueError(f"transpose requires 1 argument, got {len(args)}")
        dims = kwargs.get("dims")
        if dims is None:
            raise ValueError("transpose requires 'dims' argument")
        return torch.permute(args[0], dims)
    elif op == "dropout":
        if len(args) != 1:
            raise ValueError(f"dropout requires 1 argument, got {len(args)}")
        x = args[0]
        p = kwargs.get("p", 0.5)
        training = kwargs.get("training", True)
        return torch.nn.functional.dropout(x, p=p, training=training)
    elif op == "linear":
        if len(args) < 2 or len(args) > 3:
            raise ValueError(f"linear requires 2 or 3 arguments, got {len(args)}")
        x, w = args[0], args[1]
        w = w.t()
        b = args[2] if len(args) == 3 else None
        return torch.nn.functional.linear(x, w, b)
    elif op == "sigmoid":
        if len(args) != 1:
            raise ValueError(f"sigmoid requires 1 argument, got {len(args)}")
        return torch.sigmoid(args[0])
    elif op == "tanh":
        if len(args) != 1:
            raise ValueError(f"tanh requires 1 argument, got {len(args)}")
        return torch.tanh(args[0])
    
    raise OperationNotSupportedError(f"Operation '{op}' not supported in Torch backend.")

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

    def relu(self, x: RelayTensor, backend: str = None) -> RelayTensor:
        return self.compute("relu", x, backend=backend)

    def max_pool2d(self, inputs: RelayTensor, kernel_size: Tuple[int, int] = (2, 2),
                   strides: Optional[Tuple[int, int]] = None, padding: str = "VALID",
                   backend: str = None) -> RelayTensor:
        strides = strides or kernel_size
        return self.compute("max_pool2d", inputs, kernel_size=kernel_size, strides=strides,
                            padding=padding, backend=backend)

    def mean(self, x: RelayTensor, axis: Union[int, Tuple[int, ...], None] = None,
             backend: str = None) -> RelayTensor:
        return self.compute("mean", x, axis=axis, backend=backend)

    def softmax(self, x: RelayTensor, axis: int = -1, backend: str = None) -> RelayTensor:
        return self.compute("softmax", x, axis=axis, backend=backend)

    def batch_norm2d(self, inputs: RelayTensor, eps: float = 1e-5, momentum: float = 0.1,
                     backend: str = None) -> RelayTensor:
        return self.compute("batch_norm2d", inputs, eps=eps, momentum=momentum, backend=backend)

    def transpose(self, x: RelayTensor, dims: Tuple[int, ...], backend: str = None) -> RelayTensor:
        return self.compute("transpose", x, dims=dims, backend=backend)

    def dropout(self, x: RelayTensor, p: float = 0.5, training: bool = True, seed: int = 0, backend: str = None) -> RelayTensor:
        return self.compute("dropout", x, p=p, training=training, seed=seed, backend=backend)

    def linear(self, x: RelayTensor, w: RelayTensor, b: Optional[RelayTensor] = None, backend: str = None) -> RelayTensor:
        return self.compute("linear", x, w, b, backend=backend)
    
    def sigmoid(self, x: RelayTensor, backend: str = None) -> RelayTensor:
        return self.compute("sigmoid", x, backend=backend)
    
    def tanh(self, x: RelayTensor, backend: str = None) -> RelayTensor:
        return self.compute("tanh", x, backend=backend)
    
    def conv1d(self, inputs: RelayTensor, kernel: RelayTensor, strides: Tuple[int] = (1,), 
            padding: str = "VALID", backend: str = None) -> RelayTensor:
        return self.compute("conv1d", inputs, kernel, strides=strides, padding=padding, backend=backend)

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
    x_relu = RelayTensor(jnp.array([-1, 2, -3, 4]))
    relu_jax = relayx.relu(x_relu)
    print("JAX relu:", relu_jax.shape, relu_jax.data)
    x_relu_torch = RelayTensor(torch.tensor([-1, 2, -3, 4], dtype=torch.float32))
    relu_torch = relayx.relu(x_relu_torch, backend="torch")
    print("Torch relu:", relu_torch.shape, relu_torch.data)
    pool_input = RelayTensor(jnp.ones((1, 28, 28, 1)))
    pool_jax = relayx.max_pool2d(pool_input, kernel_size=(2, 2), strides=(2, 2))
    print("JAX max_pool2d:", pool_jax.shape, pool_jax.data)
    pool_input_torch = RelayTensor(torch.ones(1, 28, 28, 1))
    pool_torch = relayx.max_pool2d(pool_input_torch, kernel_size=(2, 2), strides=(2, 2), backend="torch")
    print("Torch max_pool2d:", pool_torch.shape, pool_torch.data)
    mean_input = RelayTensor(jnp.array([1, 2, 3, 4]))
    mean_jax = relayx.mean(mean_input)
    print("JAX mean:", mean_jax.shape, mean_jax.data)
    mean_input_torch = RelayTensor(torch.tensor([1, 2, 3, 4], dtype=torch.float32))
    mean_torch = relayx.mean(mean_input_torch, backend="torch")
    print("Torch mean:", mean_torch.shape, mean_torch.data)
    softmax_input = RelayTensor(jnp.array([1, 2, 3, 4], dtype=jnp.float32))
    softmax_jax = relayx.softmax(softmax_input, axis=0)
    print("JAX softmax:", softmax_jax.shape, softmax_jax.data)
    softmax_input_torch = RelayTensor(torch.tensor([1, 2, 3, 4], dtype=torch.float32))
    softmax_torch = relayx.softmax(softmax_input_torch, axis=0, backend="torch")
    print("Torch softmax:", softmax_torch.shape, softmax_torch.data)
    bn_input = RelayTensor(jnp.ones((2, 4, 4, 2), dtype=jnp.float32) * 2 + 1)
    bn_jax = relayx.batch_norm2d(bn_input)
    print("JAX batch_norm2d:", bn_jax.shape, bn_jax.data.mean(), bn_jax.data.std())
    bn_input_torch = RelayTensor(torch.ones(2, 4, 4, 2, dtype=torch.float32) * 2 + 1)
    bn_torch = relayx.batch_norm2d(bn_input_torch, backend="torch")
    print("Torch batch_norm2d:", bn_torch.shape, bn_torch.data.mean(), bn_torch.data.std())
    transpose_input = RelayTensor(jnp.ones((2, 3, 4)))
    transpose_jax = relayx.transpose(transpose_input, dims=(1, 0, 2))
    print("JAX transpose:", transpose_jax.shape, transpose_jax.data)
    transpose_input_torch = RelayTensor(torch.ones(2, 3, 4))
    transpose_torch = relayx.transpose(transpose_input_torch, dims=(1, 0, 2), backend="torch")
    print("Torch transpose:", transpose_torch.shape, transpose_torch.data)
    dropout_input = RelayTensor(jnp.ones((2, 3), dtype=jnp.float32))
    dropout_jax = relayx.dropout(dropout_input, p=0.5, training=True, seed=42)
    print("JAX dropout:", dropout_jax.shape, dropout_jax.data)
    dropout_input_torch = RelayTensor(torch.ones(2, 3, dtype=torch.float32))
    dropout_torch = relayx.dropout(dropout_input_torch, p=0.5, training=True, seed=42, backend="torch")
    print("Torch dropout:", dropout_torch.shape, dropout_torch.data)
    x_linear = RelayTensor(jnp.ones((2, 3)))
    w_linear = RelayTensor(jnp.ones((3, 4)))
    b_linear = RelayTensor(jnp.ones((4,)))
    linear_jax = relayx.linear(x_linear, w_linear, b_linear)
    print("JAX linear:", linear_jax.shape, linear_jax.data)
    x_linear_torch = RelayTensor(torch.ones(2, 3))
    w_linear_torch = RelayTensor(torch.ones(3, 4))
    b_linear_torch = RelayTensor(torch.ones(4))
    linear_torch = relayx.linear(x_linear_torch, w_linear_torch, b_linear_torch, backend="torch")
    print("Torch linear:", linear_torch.shape, linear_torch.data)
    x_sigmoid = RelayTensor(jnp.array([-1, 0, 1], dtype=jnp.float32))
    sigmoid_jax = relayx.sigmoid(x_sigmoid)
    print("JAX sigmoid:", sigmoid_jax.shape, sigmoid_jax.data)
    x_sigmoid_torch = RelayTensor(torch.tensor([-1, 0, 1], dtype=torch.float32))
    sigmoid_torch = relayx.sigmoid(x_sigmoid_torch, backend="torch")
    print("Torch sigmoid:", sigmoid_torch.shape, sigmoid_torch.data)
    x_tanh = RelayTensor(jnp.array([-1, 0, 1], dtype=jnp.float32))
    tanh_jax = relayx.tanh(x_tanh)
    print("JAX tanh:", tanh_jax.shape, tanh_jax.data)
    x_tanh_torch = RelayTensor(torch.tensor([-1, 0, 1], dtype=torch.float32))
    tanh_torch = relayx.tanh(x_tanh_torch, backend="torch")
    print("Torch tanh:", tanh_torch.shape, tanh_torch.data)

test_relayx()