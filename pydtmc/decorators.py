# -*- coding: utf-8 -*-

__all__ = [
    'aliased',
    'cached_property',
    'object_mark'
]


###########
# IMPORTS #
###########

# Standard

import functools as _ft
import re as _re
import threading as _th


###########
# CLASSES #
###########

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
        self._lock = _th.RLock()

        _ft.update_wrapper(self, fget)

    def __set_name__(self, owner, name):

        if self._func_name is None:
            self._func_name = name
        elif name != self._func_name:
            raise AttributeError('Cached properties cannot be shared among different class members.')

    def __get__(self, instance, owner=None):

        if instance is None:
            return self

        with self._lock:
            try:
                return instance.__dict__[self._func_name]
            except KeyError:
                return instance.__dict__.setdefault(self._func_name, self._func(instance))

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
class object_mark:

    """
    A class decorator used for marking methods and properties.
    """

    def __init__(self, aliases=None, instance_generator=False, random_output=False):

        if aliases is None and not instance_generator and not random_output:
            raise AttributeError('Object mark decorator must have at least one argument value different than the default one.')

        self.mark_applied = False
        self.aliases = aliases
        self.instance_generator = instance_generator
        self.random_output = random_output

    def __call__(self, obj):

        if self.mark_applied:
            return obj

        target = obj.fget if isinstance(obj, property) else obj

        if self.aliases is not None:
            target._aliases = self.aliases

        if self.instance_generator:
            target._instance_generator = True

        if self.random_output:
            target._random_output = True

        self.mark_applied = True

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

        @_ft.wraps(func)
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

        if any(not _re.search(r'^[a-z][\da-z]*(?:_[\da-z]+)*$', a) for a in aliases_flat):
            raise AttributeError('Aliases cannot start with an underscore character and must be compliant with PEP8 naming conventions.')

        if any(a in member_names for a in aliases_flat):
            raise AttributeError('Aliases cannot be equal to existing class members.')

        for member, member_aliases in aliases.items():

            if isinstance(member, property):
                member_wrapped = property(member.fget, member.fset, member.fdel, member.__doc__)
            else:
                member_wrapped = wrapper(member)
                member_wrapped.__doc__ = member.__doc__

            for member_alias in member_aliases:
                setattr(aliased_class, member_alias, member_wrapped)

    return aliased_class
