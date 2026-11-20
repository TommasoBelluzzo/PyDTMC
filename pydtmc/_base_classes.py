# -*- coding: utf-8 -*-

__all__ = [
    'Model'
]


###########
# IMPORTS #
###########

# Standard

import abc as _abc

# Internal

from . import _custom_types as _ct


###########
# CLASSES #
###########

class Model(_abc.ABC):  # pragma: no cover

    """
    Defines an abstract base class used to implement package models.
    """

    def __new__(cls, *args, **kwargs):  # pylint: disable=W0613

        if cls is Model:
            raise TypeError('The base class cannot be instantiated.')

        return super().__new__(cls)

    @property
    @_abc.abstractmethod
    def n(self) -> int:
        pass

    @property
    @_abc.abstractmethod
    def p(self) -> _ct.tarray:
        pass

    @property
    @_abc.abstractmethod
    def pi(self) -> _ct.tlist_array:
        pass

    @property
    @_abc.abstractmethod
    def states(self) -> _ct.tlist_str:
        pass

    @_abc.abstractmethod
    def to_dictionary(self):
        pass

    @_abc.abstractmethod
    def to_file(self, file_path):
        pass

    @_abc.abstractmethod
    def to_graph(self):
        pass

    @staticmethod
    @_abc.abstractmethod
    def from_dictionary(d):
        pass

    @staticmethod
    @_abc.abstractmethod
    def from_file(file_path):
        pass

    @staticmethod
    @_abc.abstractmethod
    def from_graph(graph):
        pass
