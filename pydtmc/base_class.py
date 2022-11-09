# -*- coding: utf-8 -*-

__all__ = [
    'BaseClass'
]


###########
# IMPORTS #
###########

# Standard

# noinspection PyPep8Naming
from abc import (
    ABC as _abc_ABC,
    abstractmethod as _abc_abstractmethod
)


###########
# CLASSES #
###########

class BaseClass(_abc_ABC):

    """
    Defines an abstract base class used by package classes.
    """

    # noinspection PyMethodParameters
    def __new__(cls, *args, **kwargs):

        if cls is BaseClass:
            raise TypeError('The base class cannot be instantiated.')

        return super().__new__(cls)

    @property
    @_abc_abstractmethod
    def p(self):
        pass

    @property
    @_abc_abstractmethod
    def size(self):
        pass

    @property
    @_abc_abstractmethod
    def states(self):
        pass

    @_abc_abstractmethod
    def to_dictionary(self):
        pass

    @_abc_abstractmethod
    def to_file(self, file_path):
        pass

    @_abc_abstractmethod
    def to_graph(self):
        pass

    @_abc_abstractmethod
    def to_matrices(self):
        pass

    @staticmethod
    @_abc_abstractmethod
    def from_dictionary(d):
        pass

    @staticmethod
    @_abc_abstractmethod
    def from_file(file_path):
        pass

    @staticmethod
    @_abc_abstractmethod
    def from_graph(graph):
        pass

    @staticmethod
    @_abc_abstractmethod
    def from_matrices(*args):
        pass
