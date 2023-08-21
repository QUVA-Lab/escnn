import inspect
import warnings

from abc import ABCMeta
from functools import partial
from itertools import chain
from typing import Any

def _get_init_signature(cls):
    # This will happen if the class doesn't define a constructor.  
    # `object.__init__()` doesn't really take any arguments, because it raises 
    # `TypeError` if you try to pass any.  But technically, its signature is 
    # `object.__init__(self, /, *args, **kwargs)`.  This would confuse some 
    # other parts of the code, so here we just replace it with its "effective" 
    # signature.
    if cls.__init__ is object.__init__:
        return inspect.Signature()

    signature_with_self = inspect.signature(cls.__init__)
    params_without_self = list(signature_with_self.parameters.values())[1:]
    return signature_with_self.replace(parameters=params_without_self)

class SingletonMeta(type):
    # See `Singleton` for an overall description of this singleton framework, 
    # but most of the actual logic is implemented by this metaclass.  The 
    # reason we need a metaclass is that we need control over when `__init__()` 
    # is called.  Specifically, we only want to call `__init__()` when we 
    # actually create a new singleton object.  If we're just returning an 
    # existing object, trying to initialize it again could easily put it into 
    # an unexpected and incorrect state.

    def __init__(cls, name, bases, namespace, **kwargs):
        cls._singletons = {}

        # Because we internally store the constructor arguments in a tuple, we 
        # can only handle constructors where every argument can be specified 
        # positionally.  Here we fail with an informative error message if we 
        # find any arguments that would require something more sophisticated.
        init_signature = _get_init_signature(cls)
        for name, param in init_signature.parameters.items():
            if param.kind == param.KEYWORD_ONLY:
                raise SingletonError(f"{cls.__init__.__qualname__}{init_signature}: cannot use keyword-only arguments")
            if param.kind == param.VAR_POSITIONAL:
                raise SingletonError(f"{cls.__init__.__qualname__}{init_signature}: cannot use variable positional arguments")
            if param.kind == param.VAR_KEYWORD:
                raise SingletonError(f"{cls.__init__.__qualname__}{init_signature}: cannot use variable keyword arguments")

    def __call__(cls, *args, **kwargs):
        # Create one canonical representation of the given arguments, so we 
        # don't get different instances from, for example, ``CyclicGroup(4)`` 
        # and ``CyclicGroup(N=4)``.
        init_signature = _get_init_signature(cls)
        bound_args = init_signature.bind(*args, **kwargs)
        bound_args.apply_defaults()
        bound_kwargs = cls._canonicalize_init_kwargs(bound_args.arguments)

        init_args = tuple(
                bound_kwargs[k]
                for k in init_signature.parameters
        )
        # The *name* argument isn't really state, because it doesn't change 
        # anything about how the singleton object behaves.
        singleton_key = tuple(
                bound_kwargs[k]
                for k in init_signature.parameters
                if k != 'name'
        )

        if singleton_key in cls._singletons:
            obj = cls._singletons[singleton_key]
            return obj

        obj = cls._singletons[singleton_key] = cls.__new__(cls, *init_args)
        obj._init_args = init_args
        obj._singleton_key = singleton_key
        obj.__init__(*init_args)
        return obj

class SingletonABCMeta(SingletonMeta, ABCMeta):
    # ABC is also a metaclass, and it's not possible for a class to have more 
    # than one metaclass.  So if we want both ABC and Singleton, we need a new 
    # metaclass that inherits from both of them.
    pass

class SingletonError(Exception):
    pass

class Singleton(metaclass=SingletonMeta):
    r"""
    Turn a class into a factory for singleton objects.

    Each time you instantiate this class with different arguments, you'll get a 
    different object.  But if you instantiate this class multiple times with 
    the same arguments, you'll get the same object each time.  Each of the 
    objects created by this factory will be comparable, hashable, and 
    picklable.

    This factory is meant for objects whose entire state is specified by the 
    arguments passed to the constructor.  It should not be possible to modify 
    the singleton objects once they've been created.  Doing so could cause the 
    comparing, hashing, and pickling implementations provided by this class to 
    all behave unexpectedly.  That said, it's fine for these objects to treat 
    instance attributes as caches.  As long as the object can be completely 
    recreated by calling the constructor with its original arguments, 
    everything will work.

    Details and caveats:

    - The `__eq__()`, `__hash__()`, and `__reduce__()` implementations provided 
      by this class require that all of the arguments provided to `__init__()` 
      be comparable, hashable, and picklable.

    - Currently, keyword-only constructor arguments (including `**kwargs`) are 
      not supported.  Support for such arguments would be possible to add, 
      though, if the need arises.

    - If there is a *name* argument to `__init__()`, it is handled specially.  
      It is not considered to be part of the objects state.  If 

    - Something about _canonicalize

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

    def __repr__(self):
        args_str = ', '.join(map(repr, self._init_args))
        return f'{self.__class__.__name__}({args_str})'

    def __eq__(self, other: Any) -> bool:
        return (
                self.__class__ is other.__class__ and 
                self._singleton_key == other._singleton_key
        )

    def __hash__(self):
        return hash((self.__class__, self._singleton_key))

    def __reduce__(self):
        return partial(self.__class__, *self._init_args), ()

    @staticmethod
    def _canonicalize_init_kwargs(kwargs):
        return kwargs

class SingletonABC(Singleton, metaclass=SingletonABCMeta):
    pass


