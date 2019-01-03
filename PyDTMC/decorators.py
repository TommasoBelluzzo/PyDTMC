# -*- coding: utf-8 -*-

__all__ = [
    'alias', 'aliased', 'cachedproperty'
]


###########
# IMPORTS #
###########


from threading import RLock


###########
# CLASSES #
###########


# noinspection PyPep8Naming
class alias(object):

    """
    A decorator for method aliases.
    It can be used only inside a @aliased-decorated classes.
    """

    def __init__(self, *aliases):

        self.aliases = set(aliases)

    def __call__(self, f):

        if isinstance(f, property):
            f.fget._aliases = self.aliases
        else:
            f._aliases = self.aliases

        return f


# noinspection PyPep8Naming
class cachedproperty(object):

    """
    A decorator for lazy-evaluated properties.
    """

    def __init__(self, func):

        self.__doc__ = func.__doc__
        self.func = func
        self.lock = RLock()

    def __get__(self, obj, cls):

        if obj is None:
            return self

        obj_dict = obj.__dict__
        name = self.func.__name__

        with self.lock:
            try:
                return obj_dict[name]
            except KeyError:
                return obj_dict.setdefault(name, self.func(obj))


#############
# FUNCTIONS #
#############


# noinspection PyProtectedMember
# noinspection PyUnresolvedReferences
def aliased(aliased_class):

    """
    A decorator that enables method aliases.
    """

    original_methods = aliased_class.__dict__.copy()

    for name, method in original_methods.items():

        aliases = None

        if isinstance(method, property) and hasattr(method.fget, '_aliases'):
            aliases = method.fget._aliases
        elif hasattr(method, '_aliases'):
            aliases = method._aliases

        if aliases:

            original_methods_set = set(original_methods)

            for a in aliases - original_methods_set:
                setattr(aliased_class, a, method)

    return aliased_class
