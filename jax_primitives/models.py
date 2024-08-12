from typing import List

import jax

from .base import Dynamic, modelclass
from .layers import Linear


@modelclass
class MLP:

    linear_layers: Dynamic[List[Linear]]

    def __init__(self, in_dim, out_dim, inner_dim, n_layers, key):
        keys = jax.random.split(key, n_layers + 2)
        
        self.linear_layers = []

        self.linear_layers += [Linear(in_dim, inner_dim, keys[0])]
        self.linear_layers += [Linear(inner_dim, inner_dim, keys[i + 1]) for i in range(n_layers)]
        self.linear_layers += [Linear(inner_dim, out_dim, keys[n_layers + 1])]

    def __call__(self, x):
        y = x

        for layer in self.linear_layers[:-1]:
            y = layer(y)
            y = jax.nn.relu(y)

        y = self.linear_layers[-1](y)

        return y
