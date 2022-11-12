# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Standard

from os import (
    close as _os_close,
    remove as _os_remove
)

from random import (
    getstate as _rd_getstate,
    randint as _rd_randint,
    random as _rd_random,
    seed as _rd_seed,
    setstate as _rd_setstate
)

from tempfile import (
    mkstemp as _tf_mkstemp
)

# Libraries

from networkx import (
    MultiDiGraph as _nx_MultiDiGraph
)

from numpy.random import (
    RandomState as _npr_RandomState
)

from numpy.testing import (
    assert_allclose as _npt_assert_allclose
)

from pytest import (
    mark as _pt_mark
)

# Internal

from pydtmc import (
    HiddenMarkovModel as _HiddenMarkovModel,
    MarkovChain as _MarkovChain
)


#############
# FUNCTIONS #
#############

def _generate_objects(seed, runs, maximum_size):
    random_state = _rd_getstate()
    _rd_seed(seed)

    rng = _npr_RandomState(seed)

    objects = []

    for _ in range(runs):

        obj_mc = _rd_random() < 0.5
        obj_from_matrix = _rd_random() < 0.5
        size = _rd_randint(2, maximum_size)

        if obj_mc:

            if obj_from_matrix:
                m = rng.randint(101, size=(size, size))
                obj = _MarkovChain.from_matrix(m)
            else:
                zeros = _rd_randint(0, size)
                obj = _MarkovChain.random(size, zeros=zeros, seed=seed)

            objects.append((obj, lambda x: [x.to_matrix()]))

        else:

            size_multiplier = _rd_randint(1, 3)
            n, k = size, size * size_multiplier

            if obj_from_matrix:
                mp, me = rng.randint(101, size=(n, n)), rng.randint(101, size=(n, k))
                obj = _HiddenMarkovModel.from_matrices(mp, me)
            else:
                zeros = _rd_randint(0, size)
                p_zeros, e_zeros = zeros, zeros * size_multiplier
                obj = _HiddenMarkovModel.random(n, k, p_zeros=p_zeros, e_zeros=e_zeros, seed=seed)

            objects.append((obj, lambda x: x.to_matrices()))

    _rd_setstate(random_state)

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
            _npt_assert_allclose(obj_from_matrix, obj_matrix, rtol=1e-5, atol=1e-8)


@_pt_mark.slow
def test_graph(seed, runs, maximum_size):
    objects = _generate_objects(seed, runs, maximum_size)

    for obj, to_matrices in objects:

        obj_matrices = to_matrices(obj)

        graph = obj.to_graph()

        obj_from = obj.from_graph(graph)
        obj_from_matrices = to_matrices(obj_from)

        for index, obj_matrix in enumerate(obj_matrices):
            obj_from_matrix = obj_from_matrices[index]
            _npt_assert_allclose(obj_from_matrix, obj_matrix, rtol=1e-5, atol=1e-8)

        graph = _nx_MultiDiGraph(graph)

        obj_from = obj.from_graph(graph)
        obj_from_matrices = to_matrices(obj_from)

        for index, obj_matrix in enumerate(obj_matrices):
            obj_from_matrix = obj_from_matrices[index]
            _npt_assert_allclose(obj_from_matrix, obj_matrix, rtol=1e-5, atol=1e-8)


# noinspection PyBroadException
@_pt_mark.slow
def test_file(seed, runs, maximum_size, file_extension):
    objects = _generate_objects(seed, runs, maximum_size)

    for obj, to_matrices in objects:

        obj_matrices = to_matrices(obj)

        file_handler, file_path = _tf_mkstemp(suffix=file_extension)
        _os_close(file_handler)

        try:
            obj.to_file(file_path)
            obj_from = obj.from_file(file_path)
            exception = False
        except Exception:
            obj_from = None
            exception = True

        try:
            _os_remove(file_path)
        except Exception:
            pass

        assert exception is False

        obj_from_matrices = to_matrices(obj_from)

        for index, obj_matrix in enumerate(obj_matrices):
            obj_from_matrix = obj_from_matrices[index]
            _npt_assert_allclose(obj_from_matrix, obj_matrix, rtol=1e-5, atol=1e-8)
