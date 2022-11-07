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
    randint as _rd_randint
)

from tempfile import (
    mkstemp as _tf_mkstemp
)

# Libraries

from networkx import (
    MultiDiGraph as _nx_MultiDiGraph,
    relabel_nodes as _nx_relabel_nodes
)

from numpy import (
    array as _np_array,
    count_nonzero as _np_count_nonzero,
    isclose as _np_isclose,
    isnan as _np_isnan
)

from numpy.random import (
    randint as _npr_randint,
    seed as _npr_seed
)

from numpy.testing import (
    assert_allclose as _npt_assert_allclose
)

from pytest import (
    mark as _pt_mark
)

# Internal

from pydtmc import (
    HiddenMarkovModel as _HiddenMarkovModel
)


#########
# TESTS #
#########

# noinspection PyBroadException
@_pt_mark.slow
def test_conversions(seed, maximum_n, maximum_k, runs):

    for _ in range(runs):

        n, k = _rd_randint(2, maximum_n), _rd_randint(2, maximum_k)
        p_zeros, e_zeros = _rd_randint(0, n), _rd_randint(0, k)
        hmm = _HiddenMarkovModel.random(n, k, p_zeros=p_zeros, e_zeros=e_zeros, seed=seed)

        d = hmm.to_dictionary()
        hmm_from = _HiddenMarkovModel.from_dictionary(d)
        _npt_assert_allclose(hmm_from.p, hmm.p, rtol=1e-5, atol=1e-8)
        _npt_assert_allclose(hmm_from.e, hmm.e, rtol=1e-5, atol=1e-8)

        graph = hmm.to_graph()
        hmm_from = _HiddenMarkovModel.from_graph(graph)
        _npt_assert_allclose(hmm_from.p, hmm.p, rtol=1e-5, atol=1e-8)
        _npt_assert_allclose(hmm_from.e, hmm.e, rtol=1e-5, atol=1e-8)

        graph = _nx_relabel_nodes(_nx_MultiDiGraph(hmm.p), dict(zip(range(hmm.size), hmm.states)))
        hmm_from = _HiddenMarkovModel.from_graph(graph)
        _npt_assert_allclose(hmm_from.p, hmm.p, rtol=1e-5, atol=1e-8)
        _npt_assert_allclose(hmm_from.e, hmm.e, rtol=1e-5, atol=1e-8)
        #
        # file_handler, file_path = _tf_mkstemp(suffix=file_extension)
        # _os_close(file_handler)
        #
        # try:
        #     hmm.to_file(file_path)
        #     hmm_from = _HiddenMarkovModel.from_file(file_path)
        #     exception = False
        # except Exception:
        #     hmm_from = None
        #     exception = True
        #
        # _os_remove(file_path)
        #
        # assert exception is False
        #
        # _npt_assert_allclose(hmm_from.p, hmm.p, rtol=1e-5, atol=1e-8)
        # _npt_assert_allclose(hmm_from.e, hmm.e, rtol=1e-5, atol=1e-8)
        #
        # mp, me = _npr_randint(101, size=(n, n)), _npr_randint(101, size=(n, k))
        # mc1 = _HiddenMarkovModel.from_matrices(mp, me)
        #
        # mp, me = mc1.to_matrices()
        # mc2 = _HiddenMarkovModel.from_matrices(mp, me)
        #
        # _npt_assert_allclose(mc1.p, mc2.p, rtol=1e-5, atol=1e-8)
        # _npt_assert_allclose(mc1.e, mc2.e, rtol=1e-5, atol=1e-8)


def test_decode(p, e, symbols, use_scaling, value):

    hmm = _HiddenMarkovModel(p, e)
    decoding = hmm.decode(symbols, use_scaling)

    if decoding is None:

        actual = decoding
        expected = value

        assert actual == expected

    else:

        actual = round(decoding[0], 8)
        expected = value[0]

        assert actual == expected

        actual = decoding[1]
        expected = _np_array(value[1])

        print(actual)

        _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        actual = decoding[2]
        expected = _np_array(value[2])

        print(actual)

        _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        actual = decoding[3]
        expected = _np_array(value[3])

        print(actual)

        _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        if use_scaling:

            actual = decoding[4]
            expected = _np_array(value[4])

            print(actual)

            _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


# noinspection DuplicatedCode
def test_estimate(sequence, possible_states, possible_symbols, value):

    hmm = _HiddenMarkovModel.estimate(sequence, possible_states, possible_symbols)

    actual = hmm.p
    expected = _np_array(value[0])

    _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    actual = hmm.e
    expected = _np_array(value[1])

    _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_probabilities(p, e):

    hmm = _HiddenMarkovModel(p, e)

    transition_matrix = hmm.p
    states = hmm.states

    for index1, state1 in enumerate(states):
        for index2, state2 in enumerate(states):

            actual = hmm.transition_probability(state1, state2)
            expected = transition_matrix[index2, index1]

            assert _np_isclose(actual, expected)

    emission_matrix = hmm.e
    symbols = hmm.symbols

    for index1, state in enumerate(states):
        for index2, symbol in enumerate(symbols):

            actual = hmm.emission_probability(symbol, state)
            expected = emission_matrix[index1, index2]

            assert _np_isclose(actual, expected)


def test_properties(p, e, value):

    hmm = _HiddenMarkovModel(p, e)

    actual = hmm.is_ergodic
    expected = value[0]

    assert actual == expected

    actual = hmm.is_regular
    expected = value[1]

    assert actual == expected


# noinspection PyArgumentEqualDefault
def test_random(seed, n, k, p_zeros, p_mask, e_zeros, e_mask, value):

    hmm = _HiddenMarkovModel.random(n, k, None, p_zeros, p_mask, None, e_zeros, e_mask, seed)

    actual = hmm.p
    expected = _np_array(value[0])

    _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    if p_zeros > 0 and p_mask is None:

        actual = n**2 - _np_count_nonzero(hmm.p)
        expected = p_zeros

        assert actual == expected

    if p_mask is not None:

        indices = ~_np_isnan(_np_array(p_mask))

        actual = hmm.p[indices]
        expected = _np_array(value[0])[indices]

        _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    actual = hmm.e
    expected = _np_array(value[1])

    _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

    if e_zeros > 0 and e_mask is None:

        actual = (n * k) - _np_count_nonzero(hmm.e)
        expected = e_zeros

        assert actual == expected

    if e_mask is not None:

        indices = ~_np_isnan(_np_array(e_mask))

        actual = hmm.e[indices]
        expected = _np_array(value[1])[indices]

        _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


# noinspection DuplicatedCode, PyBroadException
def test_restrict(p, e, states, symbols, value):

    hmm = _HiddenMarkovModel(p, e)

    try:
        hmm_restricted = hmm.restrict(states, symbols)
    except Exception:
        hmm_restricted = None

    if hmm_restricted is None:

        actual = hmm_restricted
        expected = value

        assert actual == expected

    else:

        actual = hmm_restricted.p
        expected = _np_array(value[0])

        _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)

        actual = hmm_restricted.e
        expected = _np_array(value[1])

        _npt_assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_simulate(p, e, seed, steps, initial_state, final_state, final_symbol, output_indices, value):

    hmm = _HiddenMarkovModel(p, e)

    actual = hmm.simulate(steps, initial_state, final_state, final_symbol, output_indices, seed)
    expected = tuple(value)

    assert actual == expected

    if initial_state is not None:

        actual = actual[0][0]
        expected = initial_state

        assert actual == expected


# noinspection PyBroadException
def test_viterbi(p, e, symbols, initial_distribution, output_indices, value):

    hmm = _HiddenMarkovModel(p, e)

    try:
        actual = hmm.viterbi(symbols, initial_distribution, output_indices)
        actual = (round(actual[0], 8), actual[1])
    except ValueError:
        actual = None

    expected = value if value is None else tuple(value)

    assert actual == expected
