import jax
import jax.numpy as jnp

from .base import Dynamic, Model, optimizerclass, schedulerclass


@optimizerclass
class Adam:
    
    t: Dynamic[int]
    alpha: Dynamic[float]
    beta_1: float
    beta_2: float
    eps: float
    m: Model
    v: Model
    scheduler: Dynamic = None

    def __init__(
        self,
        model: Model,
        alpha: float = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        eps: float = 1e-8,
        scheduler: Dynamic = None
    ):
        self.t = 0
        self.alpha = alpha
        self.m = jax.tree_map(lambda x: jnp.zeros_like(x), model)
        self.v = jax.tree_map(lambda x: jnp.zeros_like(x), model)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.scheduler = scheduler

    def step(self, model, grads):
        self.t = self.t + 1

        self.m = self.beta_1 * self.m + (1 - self.beta_1) * grads
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * grads ** 2

        m_hat = self.m / (1 - self.beta_1 ** self.t)
        v_hat = self.v / (1 - self.beta_2 ** self.t)

        model = model - self.alpha * m_hat / (v_hat ** 0.5 + self.eps)

        return model


@optimizerclass
class SGD:
    
    t: Dynamic[int]
    alpha: Dynamic[float]
    scheduler: Dynamic = None

    def __init__(
        self,
        alpha: float = 0.001,
        scheduler: Dynamic = None
    ):
        self.t = 0
        self.alpha = alpha
        self.scheduler = scheduler
    
    def step(self, model, grads):
        self.t = self.t + 1
        model = model - self.alpha * grads
        return model


@schedulerclass
class ExponentialAnnealing:

    alpha_sequence: Dynamic[jax.Array]

    def __init__(self, n_steps: int, alpha_start: float, alpha_end: float):
        self.alpha_sequence = jnp.exp(jnp.linspace(jnp.log(alpha_start), jnp.log(alpha_end), n_steps))
    
    def __getitem__(self, i: int):
        return self.alpha_sequence[i]


@schedulerclass
class CosineAnnealing:

    alpha_sequence: Dynamic[jax.Array]

    def __init__(self, n_steps: int, alpha_start: float, alpha_end: float):
        t = jnp.linspace(0, jnp.pi, n_steps)
        self.alpha_sequence = 0.5 * jnp.cos(t) * (alpha_start - alpha_end) + 0.5 * (alpha_start + alpha_end)

    def __getitem__(self, i: int):
        return self.alpha_sequence[i]
