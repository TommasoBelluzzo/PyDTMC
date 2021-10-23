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
    def __new__(cls, class_name, class_bases, class_dict):

        for class_base in class_bases:
            if isinstance(class_base, BaseClass):
                raise TypeError(f"Type '{class_base.__name__}' is not an acceptable base type.")

        return type.__new__(cls, class_name, class_bases, class_dict)
