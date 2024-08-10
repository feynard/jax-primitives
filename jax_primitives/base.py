from dataclasses import dataclass
import functools
from typing import List, Self, SupportsFloat, Type, TypeAlias, get_args, get_origin, get_type_hints

import jax


class Dynamic[T]: ...
class Static[T]: ...

Learnable: TypeAlias = Dynamic
Constant: TypeAlias = Static


def __add__(self, t: Self | SupportsFloat) -> Self:
    if isinstance(t, type(self)):
        return jax.tree_util.tree_map(lambda x, y: x + y, self, t)
    else:
        return jax.tree_util.tree_map(lambda x: x + t, self)

def __radd__(self, t: Self | SupportsFloat) -> Self:
    return self.__add__(t)


def __sub__(self, t: Self | SupportsFloat) -> Self:
    if isinstance(t, type(self)):
        return jax.tree_util.tree_map(lambda x, y: x - y, self, t)
    elif isinstance(t, SupportsFloat):
        return jax.tree_util.tree_map(lambda x: x - t, self)

def __neg__(self) -> Self:
    return jax.tree_util.tree_map(lambda x: - x, self)


def __mul__(self, t: Self | SupportsFloat) -> Self:
    if isinstance(t, type(self)):
        return jax.tree_util.tree_map(lambda x, y: x * y, self, t)
    elif isinstance(t, SupportsFloat):
        return jax.tree_util.tree_map(lambda x: x * t, self)

def __rmul__(self, t: Self | SupportsFloat) -> Self:
    return self.__mul__(t)


def __truediv__ (self, t: Self | SupportsFloat) -> Self:
    if isinstance(t, type(self)):
        return jax.tree_util.tree_map(lambda x, y: x / y, self, t)
    elif isinstance(t, SupportsFloat):
        return jax.tree_util.tree_map(lambda x: x / t, self)

def __rtruediv__(self, t: Self | SupportsFloat) -> Self:
    if isinstance(t, type(self)):
        return jax.tree_util.tree_map(lambda x, y: x / y, t, self)
    elif isinstance(t, SupportsFloat):
        return jax.tree_util.tree_map(lambda x: t / x, self)


def __pow__(self, t: Self | SupportsFloat) -> Self:
    if isinstance(t, type(self)):
        return jax.tree_util.tree_map(lambda x, y: x ** y, self, t)
    elif isinstance(t, SupportsFloat):
        return jax.tree_util.tree_map(lambda x: x ** t, self)

def __rpow__(self, t: Self | SupportsFloat) -> Self:
    if isinstance(t, type(self)):
        return jax.tree_util.tree_map(lambda x, y: y ** x, self, t)
    elif isinstance(t, SupportsFloat):
        return jax.tree_util.tree_map(lambda x: t ** x, self)

'''
def apply(self, f: Callable) -> Self:
    return jax.tree_util.tree_map(lambda x: f(x), self)
'''


def create_tree_flatten(dynamic_vars: List[str], static_vars: List[str]):
    dynamic_vars = dynamic_vars
    static_vars = static_vars

    def tree_flatten(self):
        nonlocal dynamic_vars
        nonlocal static_vars

        return [getattr(self, var) for var in dynamic_vars], [getattr(self, var) for var in static_vars]
    
    return tree_flatten


def create_tree_unflatten(dynamic_vars: List[str], static_vars: List[str]):
    dynamic_vars = dynamic_vars
    static_vars = static_vars

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        nonlocal dynamic_vars
        nonlocal static_vars

        kwargs = {
            var: children[i] if i < len(dynamic_vars) else aux_data[i - len(dynamic_vars)]
            for i, var in enumerate(dynamic_vars + static_vars)
        }

        return cls(**kwargs)
    
    return tree_unflatten


def is_dynamic(type_object: Type) -> bool:
    origin = get_origin(type_object)

    if type_object is Dynamic:
        return True

    if origin is not None and origin is Dynamic:
        return True
        
    if origin is None and get_args(type_object) is Dynamic:
        return True
    
    return False


def pytree(cls):
    cls = dataclass(cls, repr=False)
    
    hints = get_type_hints(cls)
    
    dynamic = []
    static = []

    for name in hints:
        if is_dynamic(hints[name]):
            dynamic.append(name)
        else:
            static.append(name)

    cls._tree_flatten = create_tree_flatten(dynamic, static)
    cls._tree_unflatten = create_tree_unflatten(dynamic, static)

    jax.tree_util.register_pytree_node(cls, cls._tree_flatten, cls._tree_unflatten)

    return cls


def modelclass(cls):
    
    if '__call__' not in dir(cls):
        raise NotImplementedError(f"`__call__` method is not implemented for {cls.__name__} class")

    if 'create' not in dir(cls):
        raise NotImplementedError(f"`create` method is not implemented for {cls.__name__} class")
    
    cls.__add__ = __add__
    cls.__radd__ = __radd__
    cls.__sub__ = __sub__
    cls.__neg__ = __neg__
    cls.__mul__ = __mul__
    cls.__rmul__ = __rmul__
    cls.__truediv__ = __truediv__
    cls.__rtruediv__ = __rtruediv__
    cls.__pow__ = __pow__
    cls.__rpow__ = __rpow__

    cls = pytree(cls)
    
    return cls


def optimizerclass(cls):
    
    if 'step' not in dir(cls):
        raise NotImplementedError(f"`step` method is not implemented for {cls.__name__} class")

    if 'create' not in dir(cls):
        raise NotImplementedError(f"`create` method is not implemented for {cls.__name__} class")

    step_func_original = cls.step
    
    @functools.wraps(cls.step)
    def _step(self, model, grads):

        nonlocal step_func_original
        alpha = self.alpha if self.scheduler is None else self.scheduler[self.t]
        return step_func_original(self, model, grads, alpha)

    cls.step = _step
    
    cls = pytree(cls)
    
    return cls


def stateless(method):

    @functools.wraps(method)
    def _method(self, *args, train: bool, **kwargs):
        if train:
            return self, method(self, *args, **kwargs)
        else:
            return method(self, *args, **kwargs)
    
    return _method
