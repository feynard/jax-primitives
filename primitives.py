from typing import Tuple, List, SupportsFloat, Self, Callable, Any
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from pytree import PyTree
from model import Model, modelclass, forward
from optimizer import optimizerclass, step


@modelclass
class Linear:

    w: jax.Array
    b: jax.Array

    @classmethod
    def create(cls, in_dim, out_dim, key):
        w = jnp.sqrt(2 / in_dim) * jax.random.normal(key, (in_dim, out_dim))
        b = jnp.zeros(out_dim)

        return cls(w, b)

    def __call__(self, x):
        return x @ self.w + self.b

    def tree_flatten(self):
        return (self.w, self.b), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@optimizerclass
class Adam:
    
    t: int
    alpha: float
    beta_1: float
    beta_2: float
    eps: float
    m: Model
    v: Model

    @classmethod
    def create(
        cls,
        model: Model,
        alpha: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-8
    ):

        m = jax.tree_map(lambda x: jnp.zeros_like(x), model)
        v = jax.tree_map(lambda x: jnp.zeros_like(x), model)

        return cls(0, alpha, beta_1, beta_2, eps, m, v)
    
    @staticmethod
    def step(opt: Self, model, grads):
        t = opt.t + 1

        m = opt.beta_1 * opt.m + (1 - opt.beta_1) * grads
        v = opt.beta_2 * opt.v + (1 - opt.beta_2) * (grads ** 2)

        m_hat = m / (1 - opt.beta_1 ** t)
        v_hat = v / (1 - opt.beta_2 ** t)

        model = model - opt.alpha * m_hat / (v_hat ** 0.5 + opt.eps)

        return Adam(t, opt.alpha, opt.beta_1, opt.beta_2, opt.eps, m, v), model


    def tree_flatten(self):
        return (self.t, self.alpha, self.beta_1, self.beta_2, self.eps, self.m, self.v), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@modelclass
class MLP:

    layers: List[Linear]

    @classmethod
    def create(cls, in_dim, out_dim, inner_dim, n_layers, key):
        keys = jax.random.split(key, n_layers + 2)
        
        layers = [Linear.create(in_dim, inner_dim, keys[0])]
        
        for i in range(n_layers):
            layers.append(Linear.create(inner_dim, inner_dim, keys[i + 1]))

        layers.append(Linear.create(inner_dim, out_dim, keys[n_layers + 1]))

        return cls(layers)

    def __call__(self, x):
        y = x
        
        for i, layer in enumerate(self.layers):
            y = layer(y)

            if i != len(self.layers) - 1:
                y = jax.nn.relu(y)

        return y

    def tree_flatten(self):
        return self.layers, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children)

    def __repr__(self):
        result = "MLP:\n"
        
        for i, layer in enumerate(self.layers):
            result += f"Linear [{layer.w.shape[0]:4}, {layer.w.shape[1]:4}]"
            
            if i != len(self.layers) - 1:
                result += "\n"

        return result
