# -*- coding: utf-8 -*-

__all__ = [
    'BaseClass'
]


###########
# CLASSES #
###########

class BaseClass(type):

    """
    Defines an abstract base class used for the package classes.
    """

    # noinspection PyMethodParameters
    def __new__(cls, name, bases, classes):

        for b in bases:
            if isinstance(b, BaseClass):
                raise TypeError(f"Type '{b.__name__}' is not an acceptable base type.")

        return type.__new__(cls, name, bases, dict(classes))
