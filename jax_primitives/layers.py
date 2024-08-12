from typing import Literal, Union

import jax
import jax.numpy as jnp

from .base import Dynamic, modelclass


@modelclass
class Linear:

    w: jax.Array
    b: jax.Array

    def __init__(self, in_dim, out_dim, key, bias: bool = True):
        self.w = jnp.sqrt(2 / (in_dim + out_dim)) * jax.random.normal(key, (in_dim, out_dim))
        self.b = jnp.zeros(out_dim) if bias else None

    def __call__(self, x):

        if self.b is not None:
            return x @ self.w + self.b
        else:
            return x @ self.w


@modelclass
class Conv2d:

    """
    Default convention: image batch is of size (B, H, W, C), kernel is (C, C_out, H, W)
    """

    w: jax.Array
    b: jax.Array

    def __init__(self, size, in_channels, out_channels, key, bias: bool = True):
        self.w = jnp.sqrt(2 / (in_channels + out_channels)) * \
            jax.random.normal(key, (size, size, in_channels, out_channels))
        self.b = jnp.zeros(out_channels) if bias else None

    def __call__(self, x):
        d = jax.lax.conv_dimension_numbers(x.shape, self.w.shape, ('NHWC', 'HWIO', 'NHWC'))
        y = jax.lax.conv_general_dilated(x, self.w, (1, 1), 'SAME', (1, 1), (1, 1), d)

        if self.b is not None:
            return y + self.b
        else:
            return y


@modelclass
class Interpolate2d:

    scale: float
    method: Union[Literal['nearest'], Literal['linear'], Literal['cubic']]

    def __init__(self, scale: float, method: str):
        self.scale = scale
        self.method = method

    def __call__(self, x):
        """
        Convention: (B, H, W, C)
        """

        new_shape = (x.shape[0], int(x.shape[1] * self.scale), int(x.shape[2] * self.scale), x.shape[3])

        return jax.image.resize(x, new_shape, self.method)


@modelclass
class BatchNorm:

    w: Dynamic[jax.Array]
    b: Dynamic[jax.Array]
    m_tracked: Dynamic[jax.Array]
    v_tracked: Dynamic[jax.Array]
    eps: float
    momentum: float
    first_call: bool

    def __init__(self, dim: int, eps: float = 1e-05, momentum: float = 0.1):

        self.w = jnp.ones(dim)
        self.b = jnp.zeros(dim)

        self.m_tracked = jnp.zeros(dim)
        self.v_tracked = jnp.ones(dim)

        self.eps = eps
        self.momentum = momentum
        self.first_call = True

    def __call__(self, x, train: bool = True):

        m = jnp.mean(x, range(len(x.shape) - 1))

        if train:
            v = jnp.var(x, range(len(x.shape) - 1), ddof=1)
            y = (x - m) / jnp.sqrt(v + self.eps) * self.w + self.b

            if self.first_call:
                self.m_tracked = m
                self.v_tracked = v
                self.first_call = False
            else:
                self.m_tracked = (1 - self.momentum) * self.m_tracked + self.momentum * m
                self.v_tracked = (1 - self.momentum) * self.v_tracked + self.momentum * v

            return y
        else:
            return (x - self.m_tracked) / jnp.sqrt(self.v_tracked + self.eps) * self.w + self.b
