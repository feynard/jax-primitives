import functools
import inspect
import itertools
from typing import List, Self, SupportsFloat, Type, TypeAlias, get_args, get_origin, get_type_hints

import jax


class Dynamic[T]: ...
class Static[T]: ...
Model: TypeAlias = Dynamic


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

        members = {
            var: children[i] if i < len(dynamic_vars) else aux_data[i - len(dynamic_vars)]
            for i, var in enumerate(itertools.chain(dynamic_vars, static_vars))
        }

        obj = object.__new__(cls)

        for k in members:
            setattr(obj, k, members[k])

        return obj
    
    return tree_unflatten


def is_dynamic(type_object: Type) -> bool:

    dynamic_types = (Dynamic, jax.Array)

    def _recursive_helper(t):
        nonlocal dynamic_types

        if t in dynamic_types:
            return True

        origin = get_origin(t)

        if origin in dynamic_types:
            return True

        if origin is None:
            return False
        else:
            for a in get_args(t):
                if _recursive_helper(a):
                    return True

        return False

    return _recursive_helper(type_object)


def pytree(cls):

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

    step_func_original = cls.step
    
    @functools.wraps(cls.step)
    def _step(self, model, grads):

        nonlocal step_func_original
        self.alpha = self.alpha if self.scheduler is None else self.scheduler[self.t]
        return step_func_original(self, model, grads)

    cls.step = _step
    
    cls = pytree(cls)
    
    return cls


def schedulerclass(cls):

    if '__getitem__' not in dir(cls):
        raise NotImplementedError(f"`__getitem__` method is not implemented for {cls.__name__} class")

    cls = pytree(cls)

    return cls


def vmap(f, in_axes):

    parameters = inspect.signature(f).parameters

    @functools.wraps(f)
    def _f(*args, **kwargs):
        nonlocal parameters

        args_list = []
        
        for i, a in enumerate(parameters.keys()):
            if a in kwargs:
                args_list.append(kwargs[a])
            else:
                if i < len(args):
                    args_list.append(args[i])
                elif parameters[a].default is not inspect.Signature.empty:
                    args_list.append(parameters[a].default)
                else:
                    raise ValueError(f"Wrong set of parameters for function {f.__name__}: {f}")

        return jax.vmap(f, in_axes=in_axes)(*args_list)

    return _f