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

from ._custom_types import (
    tarray as _tarray,
    tlist_array as _tlist_array,
    tlist_str as _tlist_str
)


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
    def p(self) -> _tarray:
        pass

    @property
    @_abc.abstractmethod
    def pi(self) -> _tlist_array:
        pass

    @property
    @_abc.abstractmethod
    def states(self) -> _tlist_str:
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
