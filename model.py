from typing import Callable, Any
import dataclasses

import jax

from pytree import PyTree


class Model(PyTree):

    def _create_not_implemented(cls, *args, **kwargs) -> None:
        raise NotImplementedError(f"create method is not implemented for", cls)

    create: Callable[..., Any] = _create_not_implemented

    def _call_not_implemented(self, *args, **kwargs) -> None:
        raise NotImplementedError(f"`__call__` operator is not implemented for", type(self))

    __call__: Callable[..., Any] = _call_not_implemented


def forward(model: Model, *args, **kwargs):
    return model(*args, **kwargs)


def model_inherit(cls):
    return type(cls.__name__, (cls, Model), {})


def modelclass(cls):
    """
    The most important decorator here, that handles all the underlying exercises
    Defining a class in the following way:

    ```
    @modelclass
    class Model:
        ...
    ```
    
    allows you to work with it relatively easy
    """

    return jax.tree_util.register_pytree_node_class(model_inherit(dataclasses.dataclass(cls)))
