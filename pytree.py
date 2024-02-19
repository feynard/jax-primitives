from typing import SupportsFloat, Self, Callable, Any

import jax


class PyTree:

    def _tree_flatten_not_implemented(self, *args, **kwargs) -> None:
        raise NotImplementedError(f"`tree_flatten` method not implemented for", type(self))

    tree_flatten: Callable[..., Any] = _tree_flatten_not_implemented

    def _tree_unflatten_not_implemented(cls, *args, **kwargs) -> None:
        raise NotImplementedError(f"`tree_unflatten` method not implemented for", cls)

    tree_unflatten: Callable[..., Any] = _tree_unflatten_not_implemented


    def __add__(self, t: Self | SupportsFloat) -> Self:
        if isinstance(t, PyTree):
            return jax.tree_util.tree_map(lambda x, y: x + y, self, t)
        elif isinstance(t, SupportsFloat):
            return jax.tree_util.tree_map(lambda x: x + t, self)

    def __radd__(self, t: Self | SupportsFloat) -> Self:
        return self.__add__(t)


    def __sub__(self, t: Self | SupportsFloat) -> Self:
        if isinstance(t, PyTree):
            return jax.tree_util.tree_map(lambda x, y: x - y, self, t)
        elif isinstance(t, SupportsFloat):
            return jax.tree_util.tree_map(lambda x: x - t, self)

    def __neg__(self) -> Self:
        return jax.tree_util.tree_map(lambda x: - x, self)


    def __mul__(self, t: Self | SupportsFloat) -> Self:
        if isinstance(t, PyTree):
            return jax.tree_util.tree_map(lambda x, y: x * y, self, t)
        elif isinstance(t, SupportsFloat):
            return jax.tree_util.tree_map(lambda x: x * t, self)

    def __rmul__(self, t: Self | SupportsFloat) -> Self:
        return self.__mul__(t)


    def __truediv__ (self, t: Self | SupportsFloat) -> Self:
        if isinstance(t, PyTree):
            return jax.tree_util.tree_map(lambda x, y: x / y, self, t)
        elif isinstance(t, SupportsFloat):
            return jax.tree_util.tree_map(lambda x: x / t, self)

    def __truediv__ (self, t: Self | SupportsFloat) -> Self:
        if isinstance(t, PyTree):
            return jax.tree_util.tree_map(lambda x, y: x / y, self, t)
        elif isinstance(t, SupportsFloat):
            return jax.tree_util.tree_map(lambda x: x / t, self)

    def __rtruediv__(self, t: Self | SupportsFloat) -> Self:
        if isinstance(t, PyTree):
            return jax.tree_util.tree_map(lambda x, y: x / y, t, self)
        elif isinstance(t, SupportsFloat):
            return jax.tree_util.tree_map(lambda x: t / x, self)


    def __pow__(self, t: Self | SupportsFloat) -> Self:
        if isinstance(t, PyTree):
            return jax.tree_util.tree_map(lambda x, y: x ** y, self, t)
        elif isinstance(t, SupportsFloat):
            return jax.tree_util.tree_map(lambda x: x ** t, self)

    def __rpow__(self, t: Self | SupportsFloat) -> Self:
        if isinstance(t, PyTree):
            return jax.tree_util.tree_map(lambda x, y: y ** x, self, t)
        elif isinstance(t, SupportsFloat):
            return jax.tree_util.tree_map(lambda x: t ** x, self)

    def apply(self, f: Callable) -> Self:
        return jax.tree_util.tree_map(lambda x: f(x), self)
