"""Unit tests for RelayX."""
import jax.numpy as jnp
import torch
import numpy as np
from relayx import RelayTensor, RelayX

def test_relayx():
    """Test RelayX operations across backends."""
    relayx = RelayX()

    # JAX matmul
    x_jax = RelayTensor(jnp.ones((2, 3)))
    w_jax = RelayTensor(jnp.ones((3, 4)))
    out_jax = relayx.matmul(x_jax, w_jax)
    print("JAX matmul:", out_jax.shape, out_jax.data)

    # PyTorch matmul
    x_torch = RelayTensor(torch.ones(2, 3))
    w_torch = RelayTensor(torch.ones(3, 4))
    out_torch = relayx.matmul(x_torch, w_torch, backend="torch")
    print("Torch matmul:", out_torch.shape, out_torch.data)

    # JAX conv2d
    img_jax = RelayTensor(jnp.ones((1, 28, 28, 1)))
    kernel_jax = RelayTensor(jnp.ones((3, 3, 1, 2)))
    conv_jax = relayx.conv2d(img_jax, kernel_jax, padding="SAME")
    print("JAX conv2d:", conv_jax.shape, conv_jax.data)

    # Mixed backend conv2d
    img_torch = RelayTensor(torch.ones(1, 28, 28, 1))
    conv_mixed = relayx.conv2d(img_torch, kernel_jax.to_backend("torch"), backend="torch", padding="SAME")
    print("Torch conv2d (mixed):", conv_mixed.shape, conv_mixed.data)

    # JAX add
    a = RelayTensor(jnp.ones((2, 2)))
    b = RelayTensor(jnp.ones((2, 2)))
    add_jax = relayx.add(a, b)
    print("JAX add:", add_jax.shape, add_jax.data)

if __name__ == "__main__":
    test_relayx()