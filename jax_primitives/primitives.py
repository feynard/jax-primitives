from typing import List, Tuple, Union, Literal

import jax
import jax.numpy as jnp

from .base import Dynamic, Static, modelclass, optimizerclass, pytree


@modelclass
class Linear:
    
    w: Dynamic[jax.Array]
    b: Dynamic[jax.Array]

    @classmethod
    def create(cls, in_dim, out_dim, key, bias: bool = True):

        w = jnp.sqrt(2 / (in_dim + out_dim)) * jax.random.normal(key, (in_dim, out_dim))
        b = jnp.zeros(out_dim) if bias else None

        return cls(w, b)

    def __call__(self, x):
        if self.b is not None:
            return x @ self.w + self.b
        else:
            return x @ self.w


@modelclass
class Conv3x3:

    """
    Default convention: image batch is of size (B, H, W, C), kernel is (C, C_out, H, W)
    """

    w: Dynamic[jax.Array]
    b: Dynamic[jax.Array]

    @classmethod
    def create(cls, in_channels, out_channels, key, bias: bool = True):

        w = jnp.sqrt(2 / (in_channels + out_channels)) * jax.random.normal(key, (3, 3, in_channels, out_channels))
        b = jnp.zeros(out_channels) if bias else None

        return cls(w, b)

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

    @classmethod
    def create(cls, scale: float, method: str):
        return cls(scale, method)

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
    eps: float
    momentum: float

    @classmethod
    def create(cls, dim: int, eps: float = 1e-05, momentum: float = 0.1):

        w = jnp.ones(dim)
        b = jnp.ones(dim)

        return cls(w, b)

    def __call__(self, x, train: bool = True):
        
        m = jnp.mean(x, range(len(x.shape) - 1))
        v = jnp.var(x, range(len(x.shape) - 1))

        y = (x - m) / jnp.sqrt(v + self.eps) * self.w + self.b

        if train:
            return y
        else:



@optimizerclass
class Adam:
    
    t: Dynamic[int]
    alpha: float
    beta_1: float
    beta_2: float
    eps: float
    m: Dynamic[jax.Array]
    v: Dynamic[jax.Array]
    scheduler: Dynamic = None


    @classmethod
    def create(
        cls,
        model: Dynamic,
        alpha: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-8,
        scheduler: Dynamic = None
    ):

        m = jax.tree_map(lambda x: jnp.zeros_like(x), model)
        v = jax.tree_map(lambda x: jnp.zeros_like(x), model)

        return cls(0, alpha, beta_1, beta_2, eps, m, v, scheduler)
    
    @jax.jit
    def step(self, model, grads, alpha):

        t = self.t + 1

        m = self.beta_1 * self.m + (1 - self.beta_1) * grads
        v = self.beta_2 * self.v + (1 - self.beta_2) * grads ** 2

        m_hat = m / (1 - self.beta_1 ** t)
        v_hat = v / (1 - self.beta_2 ** t)

        model = model - alpha * m_hat / (v_hat ** 0.5 + self.eps)

        return Adam(t, self.alpha, self.beta_1, self.beta_2, self.eps, m, v, self.scheduler), model


@optimizerclass
class SGD:
    
    t: Dynamic[int]
    alpha: float
    scheduler: Dynamic = None

    @classmethod
    def create(
        cls,
        alpha: float = 0.001,
        scheduler: Dynamic = None
    ):

        return cls(0, alpha, scheduler)
    
    @jax.jit
    def step(self, model, grads, alpha):

        t = self.t + 1

        model = model - alpha * grads

        return SGD(t, self.alpha, self.scheduler), model


@pytree
class ExponentialAnnealing:

    alpha_sequence: Dynamic[jax.Array]

    @classmethod
    def create(cls, n_steps: int, alpha_start: float, alpha_end: float):
        return cls(jnp.exp(jnp.linspace(jnp.log(alpha_start), jnp.log(alpha_end), n_steps)))
    
    def __getitem__(self, i: int):
        return self.alpha_sequence[i]


@pytree
class CosineAnnealing:

    alpha_sequence: Dynamic[jax.Array]

    @classmethod
    def create(cls, n_steps: int, alpha_start: float, alpha_end: float):
        t = jnp.linspace(0, jnp.pi, n_steps)
        return cls(0.5 * jnp.cos(t) * (alpha_start - alpha_end) + 0.5 * (alpha_start + alpha_end))
    
    def __getitem__(self, i: int):
        return self.alpha_sequence[i]


@modelclass
class MLP:

    layers: Dynamic[List[Linear]]

    @classmethod
    def create(cls, in_dim, out_dim, inner_dim, n_layers, key):
        keys = jax.random.split(key, n_layers + 2)
        layers = []

        layers += [Linear.create(in_dim, inner_dim, keys[0])]
        layers += [Linear.create(inner_dim, inner_dim, keys[i]) for i in range(1, n_layers + 1)]
        layers += [Linear.create(inner_dim, out_dim, keys[n_layers + 1])]

        return cls(layers)
    
    def __call__(self, x):
        y = x
        
        for i, layer in enumerate(self.layers):
            y = layer(y)

            if i != len(self.layers) - 1:
                y = jax.nn.relu(y)

        return y
