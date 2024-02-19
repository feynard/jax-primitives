# JAX Primitives

Some useful code for running ML using `JAX`

## Example

```python
import jax
import primitives as jp

key = jax.random.PRNGKey(0)

mlp = jp.MLP.create(3, 2, 64, 4, key)
opt = jp.Adam.create(mlp)

def mse(model, x, y):
    return jnp.mean((jp.forward(model, x) - y) ** 2)

@jax.jit
def update(opt, model, x, y):
    loss, grads = jax.value_and_grad(mse)(model, x, y)
    opt, model = jp.step(opt, model, grads)
    return opt, model

for i in range(5000):
    opt, mlp = update(opt, mlp, x, y)
```
