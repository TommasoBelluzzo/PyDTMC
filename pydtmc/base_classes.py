# -*- coding: utf-8 -*-

__all__ = [
    'Model'
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

class Model(_abc_ABC):

    """
    Defines an abstract base class used to implement package models.
    """

    def __new__(cls, *args, **kwargs):  # pylint: disable=W0613

        if cls is Model:
            raise TypeError('The base class cannot be instantiated.')

        return super().__new__(cls)

    @_abc_abstractmethod
    def to_dictionary(self):
        pass

    @_abc_abstractmethod
    def to_file(self, file_path):
        pass

    @_abc_abstractmethod
    def to_graph(self):
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
