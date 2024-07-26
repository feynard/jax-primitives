from typing import Tuple, List, SupportsFloat, Self, Callable, Any, Optional

import jax
import jax.numpy as jnp

from .base import modelclass, optimizerclass, Dynamic


@modelclass
class Linear:

    w: Dynamic[jax.Array]
    b: Dynamic[jax.Array]

    @classmethod
    def create(cls, in_dim, out_dim, key, bias: bool = True):

        w = jnp.sqrt(2 / in_dim) * jax.random.normal(key, (in_dim, out_dim))
        b = jnp.zeros(out_dim) if bias else None

        return cls(w, b)

    def __call__(self, x):
        if self.b is not None:
            return x @ self.w + self.b
        else:
            return x @ self.w


@optimizerclass
class Adam:
    
    t: Dynamic[int]
    alpha: float
    beta_1: float
    beta_2: float
    eps: float
    m: Dynamic[jax.Array]
    v: Dynamic[jax.Array]

    @classmethod
    def create(
        cls,
        model: Dynamic,
        alpha: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-8
    ):

        m = jax.tree_map(lambda x: jnp.zeros_like(x), model)
        v = jax.tree_map(lambda x: jnp.zeros_like(x), model)

        return cls(0, alpha, beta_1, beta_2, eps, m, v)
    
    def step(self, model, grads):
        t = self.t + 1

        m = self.beta_1 * self.m + (1 - self.beta_1) * grads
        v = self.beta_2 * self.v + (1 - self.beta_2) * grads ** 2

        m_hat = m / (1 - self.beta_1 ** t)
        v_hat = v / (1 - self.beta_2 ** t)

        model = model - self.alpha * m_hat / (v_hat ** 0.5 + self.eps)

        return Adam(t, self.alpha, self.beta_1, self.beta_2, self.eps, m, v), model


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
