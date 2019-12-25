# -*- coding: utf-8 -*-

__all__ = [
    'alias', 'aliased',
    'cachedproperty'
]


###########
# IMPORTS #
###########


from functools import (
    update_wrapper as _update_wrapper,
    wraps as _wraps
)

from threading import (
    RLock as _RLock
)


###########
# CLASSES #
###########


# noinspection PyPep8Naming
class alias(object):

    """
    | A decorator for implementing method aliases.

    | It can be used only inside @aliased-decorated classes.
    """

    def __init__(self, *aliases):

        self.aliases = set(aliases)

    def __call__(self, obj):

        if isinstance(obj, property):
            obj.fget._aliases = self.aliases
        else:
            obj._aliases = self.aliases

        return obj


# noinspection PyPep8Naming, PyUnusedLocal
class cachedproperty(property):

    """
    A decorator for implementing lazy-evaluated read-only properties.
    """

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):

        doc = doc or fget.__doc__
        super(cachedproperty, self).__init__(fget, None, None, doc)

        if fget is None:
            self._func = None
            self._func_name = ''
        else:
            self._func = fget
            self._func_name = self._func.__name__

        _update_wrapper(self, fget)

        self._lock = _RLock()

    def __get__(self, obj, obj_type=None):

        if obj is None:
            return self

        if self._func is None:
            raise AttributeError('This property is unreadable.')

        with self._lock:
            try:
                return obj.__dict__[self._func_name]
            except KeyError:
                return obj.__dict__.setdefault(self._func_name, self._func(obj))

    def __set__(self, obj, value):

        if obj is None:
            raise AttributeError

        raise AttributeError('This property cannot be set.')

    def deleter(self, fdel):

        raise AttributeError('This property cannot implement a deleter.')

    def getter(self, fget):

        return type(self)(fget, None, None, None)

    def setter(self, fset):

        raise AttributeError('This property cannot implement a setter.')


#############
# FUNCTIONS #
#############


# noinspection PyProtectedMember
def aliased(aliased_class):

    """
    A decorator for enabling method aliases.
    """

    def wrapper(func):
        @_wraps(func)
        def inner(*args, **kwargs):
            return func(*args, **kwargs)
        return inner

    aliased_class_dict = aliased_class.__dict__.copy()
    aliased_class_set = set(aliased_class_dict)

    for name, method in aliased_class_dict.items():

        aliases = None

        if isinstance(method, property) and hasattr(method.fget, '_aliases'):
            aliases = method.fget._aliases
        elif hasattr(method, '_aliases'):
            aliases = method._aliases

        if aliases:
            for a in aliases - aliased_class_set:

                doc = method.__doc__
                doc_alias = doc[:len(doc) - len(doc.lstrip())] + 'Alias of **' + name + '**.'

                if isinstance(method, property):
                    wrapped_method = property(method.fget, method.fset, method.fdel, doc_alias)
                else:
                    wrapped_method = wrapper(method)
                    wrapped_method.__doc__ = doc_alias

                setattr(aliased_class, a, wrapped_method)

    return aliased_class
