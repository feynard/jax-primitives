# JAX Primitives

Some useful code for experimenting with ML using `JAX`

## Example

```python
import jax.numpy as jnp
import jax
import jax_primitives as jp

key = jax.random.key(0)
mlp = jp.MLP.create(in_dim=1, out_dim=1, inner_dim=64, n_layers=4, key=key)
opt = jp.Adam.create(mlp, alpha=0.01)

def mse(model, x, y):
    return jnp.mean((model(x) - y) ** 2)

@jax.jit
def update(opt, model, x, y):
    loss, grads = jax.value_and_grad(mse)(model, x, y)
    opt, model = opt.step(model, grads)
    return opt, model

x = jnp.linspace(-1, 1, 100)
y = x ** 3 - x + 0.25 * jnp.sin(x * 16)
```

![image](images/data.svg)

Train the model
```python
x = x.reshape(100, 1)
y = y.reshape(100, 1)

for i in range(2000):
    opt, mlp = update(opt, mlp, x, y)

y_pred = mlp(x.reshape(100, 1))
```

![image](images/predictions.svg)

## Defining Your Model

Just mark existing attributes of your class with `Dynamic` (used as nodes of a coresponing pytree) or `Static` (auxilary data fo the pytree) type hints in either available way:

```python
import jax
from jax_primitives import modelclass, Dynamic, Static

@modelclass
class LinearLayer:

    w: Dynamic[jax.Array]   # dynamic
    b: Dynamic              # dynamic
    in_dim: int             # static
    out_dim: Static         # static

    @classmethod
    def create(cls, in_dim, out_dim, key):
        w = jnp.sqrt(2 / in_dim) * jax.random.normal(key, (in_dim, out_dim))
        b = jnp.zeros(out_dim)

        return cls(w, b, in_dim, out_dim)

    @jax.jit
    def __call__(self, x):
        return x @ self.w + self.b

key = jax.random.key(0)
layer = LinearLayer.create(8, 16, key)
```
