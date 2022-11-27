# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Standard

import os as _os
import random as _rd
import tempfile as _tf

# Libraries

import networkx as _nx
import numpy.random as _npr
import numpy.testing as _npt
import pytest as _pt

# Internal

from pydtmc import (
    HiddenMarkovModel as _HiddenMarkovModel,
    MarkovChain as _MarkovChain
)


#############
# FUNCTIONS #
#############

def _generate_objects(seed, runs, maximum_size):

    random_state = _rd.getstate()
    _rd.seed(seed)

    rng = _npr.RandomState(seed)  # pylint: disable=no-member

    objects = []

    for _ in range(runs):

        obj_mc = _rd.random() < 0.5
        obj_from_matrix = _rd.random() < 0.5
        size = _rd.randint(2, maximum_size)

        if obj_mc:

            if obj_from_matrix:
                m = rng.randint(101, size=(size, size))
                obj = _MarkovChain.from_matrix(m)
            else:
                zeros = _rd.randint(0, size)
                obj = _MarkovChain.random(size, zeros=zeros, seed=seed)

            objects.append((obj, lambda x: [x.to_matrix()]))

        else:

            size_multiplier = _rd.randint(1, 3)
            n, k = size, size * size_multiplier

            if obj_from_matrix:
                mp, me = rng.randint(101, size=(n, n)), rng.randint(101, size=(n, k))
                obj = _HiddenMarkovModel.from_matrices(mp, me)
            else:
                zeros = _rd.randint(0, size)
                p_zeros, e_zeros = zeros, zeros * size_multiplier
                obj = _HiddenMarkovModel.random(n, k, p_zeros=p_zeros, e_zeros=e_zeros, seed=seed)

            objects.append((obj, lambda x: x.to_matrices()))

    _rd.setstate(random_state)

    return objects


#########
# TESTS #
#########

def test_dictionary(seed, runs, maximum_size):

    objects = _generate_objects(seed, runs, maximum_size)

    for obj, to_matrices in objects:

        obj_matrices = to_matrices(obj)

        d = obj.to_dictionary()

        obj_from = obj.from_dictionary(d)
        obj_from_matrices = to_matrices(obj_from)

        for index, obj_matrix in enumerate(obj_matrices):
            obj_from_matrix = obj_from_matrices[index]
            _npt.assert_allclose(obj_from_matrix, obj_matrix, rtol=1e-5, atol=1e-8)


@_pt.mark.slow
def test_graph(seed, runs, maximum_size):

    objects = _generate_objects(seed, runs, maximum_size)

    for obj, to_matrices in objects:

        obj_matrices = to_matrices(obj)

        graph = obj.to_graph()

        obj_from = obj.from_graph(graph)
        obj_from_matrices = to_matrices(obj_from)

        for index, obj_matrix in enumerate(obj_matrices):
            obj_from_matrix = obj_from_matrices[index]
            _npt.assert_allclose(obj_from_matrix, obj_matrix, rtol=1e-5, atol=1e-8)

        graph = _nx.MultiDiGraph(graph)

        obj_from = obj.from_graph(graph)
        obj_from_matrices = to_matrices(obj_from)

        for index, obj_matrix in enumerate(obj_matrices):
            obj_from_matrix = obj_from_matrices[index]
            _npt.assert_allclose(obj_from_matrix, obj_matrix, rtol=1e-5, atol=1e-8)


# noinspection PyBroadException
@_pt.mark.slow
def test_file(seed, runs, maximum_size, file_extension):

    objects = _generate_objects(seed, runs, maximum_size)

    for obj, to_matrices in objects:

        obj_matrices = to_matrices(obj)

        file_handler, file_path = _tf.mkstemp(suffix=file_extension)
        _os.close(file_handler)

        try:
            obj.to_file(file_path)
            obj_from = obj.from_file(file_path)
            exception = False
        except Exception:
            obj_from = None
            exception = True

        try:
            _os.remove(file_path)
        except Exception:
            pass

        assert exception is False

        obj_from_matrices = to_matrices(obj_from)

        for index, obj_matrix in enumerate(obj_matrices):
            obj_from_matrix = obj_from_matrices[index]
            _npt.assert_allclose(obj_from_matrix, obj_matrix, rtol=1e-5, atol=1e-8)
