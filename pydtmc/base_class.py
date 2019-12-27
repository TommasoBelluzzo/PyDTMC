# -*- coding: utf-8 -*-

__all__ = [
    'BaseClass'
]


###########
# CLASSES #
###########


class BaseClass(type):

    def __new__(self, name, bases, classdict):

        for b in bases:
            if isinstance(b, BaseClass):
                raise TypeError(f"Type '{b.__name__}' is not an acceptable base type.")

        return type.__new__(self, name, bases, dict(classdict))
