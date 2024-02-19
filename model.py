from typing import Callable, Any
import dataclasses
import collections

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

    new_type = type(cls.__name__, (cls, Model), {})
    
    '''
    aux_data = list()
    children = list()

    for f in dataclasses.fields(cls):
        type_name = str(f.type).lower()

        if 'model' in type_name or 'jax.array' in type_name:
            children.append(f.name)
        else:
            aux_data.append(f.name)

    print(aux_data)
    print(children)

    def tree_flatten(self):
        return tuple(self.__dict__[f] for f in children), {f: self.__dict__[f] for f in aux_data}

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
    
    setattr(new_type, 'tree_flatten', tree_flatten)
    setattr(new_type, 'tree_unflatten', tree_unflatten)
    '''

    return new_type


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
