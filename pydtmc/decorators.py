# -*- coding: utf-8 -*-

__all__ = [
    'alias',
    'aliased',
    'cached_property',
    'random_output'
]


###########
# IMPORTS #
###########

# Standard

from functools import (
    update_wrapper as _ft_update_wrapper,
    wraps as _ft_wraps
)

from re import (
    search as _re_search
)

from threading import (
    RLock as _RLock
)


###########
# CLASSES #
###########

# noinspection PyPep8Naming
class alias:

    """
    | A class decorator used for implementing property and method aliases.
    | It can be used only inside @aliased-decorated classes.
    """

    def __init__(self, *aliases):

        self.aliases = aliases

    def __call__(self, obj):

        if isinstance(obj, property):
            obj.fget._aliases = self.aliases
        else:
            obj._aliases = self.aliases

        return obj


# noinspection PyPep8Naming
class cached_property(property):

    """
    A class decorator used for implementing cached properties.
    """

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):

        if fset is not None or fdel is not None:
            raise AttributeError('Cached properties cannot implement set and delete methods.')

        if doc is None and fget is not None:
            doc = fget.__doc__

        super().__init__(fget, fset, fdel, doc)

        self._func = fget
        self._func_name = None
        self._lock = _RLock()

        _ft_update_wrapper(self, fget)

    def __set_name__(self, owner, name):

        if self._func_name is None:
            self._func_name = name
        elif name != self._func_name:
            raise AttributeError('Cached properties cannot be shared among different class members.')

    def __get__(self, instance, owner=None):

        if instance is None:
            return self

        instance_dict = instance.__dict__

        with self._lock:
            try:
                return instance_dict[self._func_name]
            except KeyError:
                return instance_dict.setdefault(self._func_name, self._func(instance))

    def __set__(self, instance, value):

        raise AttributeError('Cached properties cannot be altered.')

    def __delete__(self, instance):

        raise AttributeError('Cached properties cannot be altered.')

    def getter(self, fget):

        raise AttributeError('Cached properties cannot be altered.')

    def setter(self, fset):

        raise AttributeError('Cached properties cannot be altered.')

    def deleter(self, fdel):

        raise AttributeError('Cached properties cannot be altered.')


# noinspection PyPep8Naming
class random_output:

    """
    A class decorator used for marking random output methods.
    """

    def __init__(self):

        pass

    def __call__(self, obj):

        if isinstance(obj, property):
            obj.fget._random_output = True
        else:
            obj._random_output = True

        return obj


#############
# FUNCTIONS #
#############

# noinspection PyProtectedMember
def aliased(aliased_class):

    """
    A function decorator used for enabling aliases.
    """

    def wrapper(func):

        @_ft_wraps(func)
        def inner(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        return inner

    member_names = []
    aliases = {}

    for member_name, member in aliased_class.__dict__.items():

        member_names.append(member_name)

        if isinstance(member, property) and hasattr(member.fget, '_aliases'):
            member_aliases = member.fget._aliases
        elif hasattr(member, '_aliases'):
            member_aliases = member._aliases
        else:
            member_aliases = None

        if member_aliases is not None:
            aliases[member] = member_aliases

    if len(aliases) > 0:

        aliases_flat = [a for member_aliases in aliases.values() for a in member_aliases]

        if len(set(aliases_flat)) < len(aliases_flat):
            raise AttributeError('Aliases must be unique and cannot be shared among different class members.')

        if any(not _re_search(r'^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$', a) for a in aliases_flat):
            raise ValueError('Aliases cannot start with an underscore character and must be compliant with PEP8 naming conventions.')

        if any(a in member_names for a in aliases_flat):
            raise ValueError('Aliases cannot be equal to existing class members.')

        for member, member_aliases in aliases.items():
            for a in member_aliases:

                if isinstance(member, property):
                    member_wrapped = property(member.fget, member.fset, member.fdel, member.__doc__)
                else:
                    member_wrapped = wrapper(member)
                    member_wrapped.__doc__ = member.__doc__

                setattr(aliased_class, a, member_wrapped)

    return aliased_class
