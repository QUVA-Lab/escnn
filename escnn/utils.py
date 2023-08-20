import inspect

from abc import ABCMeta
from functools import partial
from typing import Any

class SingletonMeta(type):
    # There are two ways to implement the singleton pattern in python: with 
    # `__new__()` or with a metaclass.  Normally the former is preferred, 
    # because it's simpler.  (The first rule of metaclasses is: don't use 
    # metaclasses if there's any other alternative.)  However, `__new__()` 
    # doesn't provide any way to control how `__init__()` is called, and we 
    # need that control.  The reason is that there are cases where different 
    # argument values actually mean the same thing.  For example:
    # 
    #   class MyGroup(Group):
    #       
    #       def __init__(self, a, b=None):
    #           self._a = a
    #           self._b = b or a
    #
    # Here, `MyGroup(1)` and `MyGroup(1, 1)` should return the same instance.  
    # But there's no way to do this without knowing something about the  
    # relationship between `a` and `b`.  A metaclass can make this work this by 
    # giving the class a chance to "canonicalize" the arguments that will be 
    # passed to `__init__()`.  Here's what this would look like:
    #
    #   class MyGroup(Group):
    #
    #       def __init__(self, a, b=None):
    #           self._a = a
    #           self._b = b
    #
    #       @staticmethod
    #       def _canonicalize_init_kwargs(kwargs):
    #           kwargs['b'] = kwargs['b'] or kwargs['a']
    #           return kwargs
    #
    # For a real example, see `escnn.group.DirectProductGroup`.

    def __new__(metacls, name, bases, namespace, **kwargs):
        cls = super().__new__(metacls, name, bases, namespace, **kwargs)
        cls._singletons = {}
        return cls

    def __call__(cls, *args, **kwargs):
        # Create one canonical representation of the given arguments, so we 
        # don't get different instances from, for example, ``CyclicGroup(4)`` 
        # and ``CyclicGroup(N=4)``.
        signature_with_self = inspect.signature(cls.__init__)
        params = list(signature_with_self.parameters.values())
        signature = signature_with_self.replace(parameters=params[1:])
        bound_args = signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        init_kwargs = cls._canonicalize_init_kwargs(bound_args.arguments)

        singleton_key = tuple(
                init_kwargs[k]
                for k in bound_args.arguments
                if k in init_kwargs
        )
        if singleton_key in cls._singletons:
            return cls._singletons[singleton_key]

        obj = cls._singletons[singleton_key] = cls.__new__(cls, *args, **kwargs)
        obj._singleton_key = singleton_key  # for pickling, hashing, etc.
        obj.__init__(*singleton_key)
        return obj

class SingletonABCMeta(SingletonMeta, ABCMeta):
    pass



class Singleton(metaclass=SingletonMeta):
    r"""
    Turn a class into a factory for singleton objects.

    Each time you instantiate this class with different arguments, you'll get a 
    different object.  But if you instantiate this class multiple times with 
    the same arguments, you'll get the same object each time.  Each of the 
    objects created by this factory will be comparable, hashable, and 
    picklable.

    This factory is meant to be used for objects whose entire state is 
    specified by the arguments passed to the constructor.  It should not be 
    possible to modify the singleton objects once they've been created.  Doing 
    so could cause the comparing, hashing, and pickling implementations 
    provided by this class to all behave unexpectedly.  That said, it's fine 
    for these objects to treat instance attributes as caches.  As long as the 
    object can be completely recreated by calling the constructor with its 
    original arguments, everything will work.

    Details and caveats:

    - The `__eq__()`, `__hash__()`, and `__reduce__()` implementations provided 
      by this class all make direct use of the arguments provided to 
      `__init__()`.  That means all such argument must be comparable, hashable, 
      and picklable.

    - Currently, ``*args``, ``**kwargs``, and keyword-only arguments are not 
      supported.  Support for these features would be possible to add, though,
      if the need arises.

    - It's possible to customize what exactly it means for two sets of 
      arguments to be "the same" or "different".  This is done by implementing 
      a static method called ``_canonicalize_init_kwargs()``.  This method is 
      given a dictionary of all the arguments that were specified for the 
      constructor, and it should return a dictionary of the arguments to 
      actually pass to the constructor.  Two instances will be considered equal 
      if these returned dictionaries are equal.

    Example::

        >>> class A(Singleton):
        ...
        ...     def __init__(self, x, y=2):
        ...         self._x = x
        ...         self._y = y
        ...
        ...     # Provide read-only access to `x` and `y`, because singleton
        ...     # objects are supposed to be immutable.
        ...
        ...     @property
        ...     def x(self):
        ...         return self.x
        ...
        ...     @property
        ...     def y(self):
        ...         return self.y
        ...
        >>> a = A(1)
        >>> a.x, a.y
        (1, 2)
        >>> a is A(1)
        True
        >>> a is A(1, 2)
        True
        >>> a is A(2)
        False
    """

    def __eq__(self, other: Any) -> bool:
        return (
                self.__class__ is other.__class__ and 
                self._singleton_key == other._singleton_key
        )

    def __hash__(self):
        return hash((self.__class__, self._singleton_key))

    def __reduce__(self):
        return partial(self.__class__, *self._singleton_key), ()

    @staticmethod
    def _canonicalize_init_kwargs(kwargs: dict) -> dict:
        return kwargs


class SingletonABC(Singleton, metaclass=SingletonABCMeta):
    # ABC is also a metaclass, so some care has to be taken to use two 
    # metaclasses at the same time.
    pass
