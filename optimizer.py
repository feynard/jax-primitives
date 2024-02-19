from typing import Callable, Any, Self
from dataclasses import dataclass

import jax

from pytree import PyTree


class Optimizer(PyTree):

    def _init_not_implemented(cls, *args, **kwargs) -> None:
        raise NotImplementedError(f"init method is not implemented for", cls)

    init: Callable[..., Any] = _init_not_implemented

    def _step_not_implemented(opt: Self, *args, **kwargs) -> None:
        raise NotImplementedError(f"`step` method is not implemented for", type(opt))

    step: Callable[..., Any] = _step_not_implemented


def step(opt: Optimizer, *args, **kwargs):
    return opt.step(opt, *args, **kwargs)


def optimizer_inherit(cls):
    return type(cls.__name__, (cls, Optimizer), {})

def optimizerclass(cls):    
    return jax.tree_util.register_pytree_node_class(optimizer_inherit(dataclass(cls)))
