# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Standard

from os.path import (
    abspath as _osp_abspath,
    dirname as _osp_dirname,
    join as _osp_join
)

# Libraries

from networkx import (
    DiGraph as _nx_DiGraph,
    MultiDiGraph as _nx_MultiDiGraph
)

from numpy import (
    array as _np_array,
    ndarray as _np_ndarray
)

from numpy.random import (
    RandomState as _npr_RandomState
)

from pytest import (
    mark as _pt_mark,
    skip as _pt_skip
)

# Internal

from pydtmc import (
    HiddenMarkovModel as _HiddenMarkovModel,
    MarkovChain as _MarkovChain
)

from pydtmc.validation import (
    validate_boolean as _validate_boolean,
    validate_boundary_condition as _validate_boundary_condition,
    validate_dictionary as _validate_dictionary,
    validate_distribution as _validate_distribution,
    validate_dpi as _validate_dpi,
    validate_enumerator as _validate_enumerator,
    validate_file_path as _validate_file_path,
    validate_float as _validate_float,
    validate_graph as _validate_graph,
    validate_hidden_markov_model as _validate_hidden_markov_model,
    validate_hidden_markov_models as _validate_hidden_markov_models,
    validate_hmm_emission as _validate_hmm_emission,
    validate_integer as _validate_integer,
    validate_hyperparameter as _validate_hyperparameter,
    validate_interval as _validate_interval,
    validate_label as _validate_label,
    validate_labels_current as _validate_labels_current,
    validate_labels_input as _validate_labels_input,
    validate_markov_chain as _validate_markov_chain,
    validate_markov_chains as _validate_markov_chains,
    validate_mask as _validate_mask,
    validate_matrix as _validate_matrix,
    validate_model as _validate_model,
    validate_partitions as _validate_partitions,
    validate_random_distribution as _validate_random_distribution,
    validate_rewards as _validate_rewards,
    validate_sequence as _validate_sequence,
    validate_sequences as _validate_sequences,
    validate_status as _validate_status,
    validate_strings as _validate_strings,
    validate_time_points as _validate_time_points,
    validate_transition_function as _validate_transition_function,
    validate_transition_matrix as _validate_transition_matrix,
    validate_vector as _validate_vector,
)

from .utilities import (
    evaluate as _evaluate,
    string_to_function as _string_to_function
)


#############
# CONSTANTS #
#############

_base_directory = _osp_abspath(_osp_dirname(__file__))


#########
# TESTS #
#########

# noinspection PyBroadException
def test_validate_boolean(value, is_valid):

    try:
        result = _validate_boolean(value)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:
        result_check = isinstance(result, bool)
        assert result_check is True


# noinspection PyBroadException
def test_validate_boundary_condition(value, is_valid):

    try:
        result = _validate_boundary_condition(value)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:
        result_check = isinstance(result, (float, int, str))
        assert result_check is True


# noinspection PyBroadException
def test_validate_dictionary(dictionary_elements, key_tuple, is_valid):

    if dictionary_elements is None:
        d = None
    else:

        d = {}

        for dictionary_element in dictionary_elements:
            if key_tuple:
                d[tuple(dictionary_element[:-1])] = dictionary_element[-1]
            else:
                d[dictionary_element[0]] = dictionary_element[1]

    try:
        result = _validate_dictionary(d)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:
        result_check = isinstance(result, dict) and all(isinstance(k, tuple) and all(isinstance(x, str) for x in k) and isinstance(v, float) for k, v in result.items())
        assert result_check is True


# noinspection DuplicatedCode, PyBroadException
def test_validate_distribution(value, size, is_valid):

    if isinstance(value, list):
        for index, v in enumerate(value):
            value[index] = _np_array(v)

    try:
        result = _validate_distribution(value, size)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:
        result_check = isinstance(result, int) or (isinstance(result, list) and all(isinstance(v, _np_ndarray) for v in result))
        assert result_check is True


# noinspection PyBroadException
def test_validate_dpi(value, is_valid):

    try:
        result = _validate_dpi(value)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:
        result_check = isinstance(result, int)
        assert result_check is True


# noinspection PyBroadException
def test_validate_enumerator(value, possible_values, is_valid):

    try:
        result = _validate_enumerator(value, possible_values)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:
        result_check = isinstance(result, str)
        assert result_check is True


# noinspection PyBroadException
@_pt_mark.slow
def test_validate_file_path(value, accepted_extensions, write_permission, is_valid):

    if isinstance(value, str) and value.startswith('file_'):
        value = _osp_join(_base_directory, f'files/{value}')

    try:
        result = _validate_file_path(value, accepted_extensions, write_permission)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:
        result_check = result[0] == value
        assert result_check is True


# noinspection PyBroadException
def test_validate_float(value, lower_limit, upper_limit, is_valid):

    lower_limit = None if lower_limit is None else tuple(lower_limit)
    upper_limit = None if upper_limit is None else tuple(upper_limit)

    try:
        result = _validate_float(value, lower_limit, upper_limit)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:
        result_check = isinstance(result, float)
        assert result_check is True


# noinspection PyBroadException
def test_validate_graph(seed, multi, graph_nodes, layers, is_valid):

    if graph_nodes is None:
        graph = None
    else:

        graph = _nx_MultiDiGraph() if multi else _nx_DiGraph()

        if isinstance(graph_nodes, list) and all(isinstance(x, list) for x in graph_nodes):

            for index, nodes in enumerate(graph_nodes):
                graph.add_nodes_from(nodes, layer=index)

        else:

            graph.add_nodes_from(graph_nodes)

        nodes = graph.nodes
        size = len(nodes)**2

        rng = _npr_RandomState(seed)
        weights = [max(1e-8, r) for r in list(rng.random(size))]
        weights_offset = 0

        for node_i in nodes:
            for node_j in nodes:
                graph.add_edge(node_i, node_j, type='P', weight=weights[weights_offset])
                weights_offset += 1

    try:
        result = _validate_graph(graph, layers)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:
        result_check = isinstance(result, _nx_DiGraph)
        assert result_check is True


# noinspection PyBroadException
def test_validate_hidden_markov_model(value, is_valid):

    if isinstance(value, str):
        value, skip = _evaluate(value)
    else:
        skip = False

    if skip:
        _pt_skip('Pandas library could not be imported.')
    else:

        try:
            result = _validate_hidden_markov_model(value)
            result_is_valid = True
        except Exception:
            result = None
            result_is_valid = False

        actual = result_is_valid
        expected = is_valid

        assert actual == expected

        if result_is_valid:
            result_check = isinstance(result, _HiddenMarkovModel)
            assert result_check is True


# noinspection PyBroadException
def test_validate_hidden_markov_models(value, is_valid):

    if isinstance(value, str):
        value, skip = _evaluate(value)
    else:
        skip = False

    if skip:
        _pt_skip('Pandas library could not be imported.')
    else:

        try:
            result = _validate_hidden_markov_models(value)
            result_is_valid = True
        except Exception:
            result = None
            result_is_valid = False

        actual = result_is_valid
        expected = is_valid

        assert actual == expected

        if result_is_valid:
            result_check = isinstance(result, list) and all(isinstance(v, _HiddenMarkovModel) for v in result)
            assert result_check is True


# noinspection PyBroadException
def test_validate_hmm_emission(value, size, is_valid):

    try:
        result = _validate_hmm_emission(value, size)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:
        result_check = isinstance(result, _np_ndarray)
        assert result_check is True


# noinspection PyBroadException
def test_validate_hyperparameter(value, size, is_valid):

    try:
        result = _validate_hyperparameter(value, size)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:
        result_check = isinstance(result, _np_ndarray)
        assert result_check is True


# noinspection PyBroadException
def test_validate_integer(value, lower_limit, upper_limit, is_valid):

    lower_limit = None if lower_limit is None else tuple(lower_limit)
    upper_limit = None if upper_limit is None else tuple(upper_limit)

    try:
        result = _validate_integer(value, lower_limit, upper_limit)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:
        result_check = isinstance(result, int)
        assert result_check is True


# noinspection PyBroadException
def test_validate_interval(value, is_valid):

    value = tuple(value) if isinstance(value, list) else value

    try:
        result = _validate_interval(value)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:
        result_check = all(isinstance(v, float) for v in result)
        assert result_check is True


# noinspection PyBroadException
def test_validate_label(value, labels, is_valid):

    try:
        result = _validate_label(value, labels)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:
        assert isinstance(result, int) is True
        assert result == (labels.index(value) if isinstance(value, str) else labels.index(labels[value]))


# noinspection PyBroadException
def test_validate_labels_current(value, labels, subset, minimum_length, is_valid):

    try:
        result = _validate_labels_current(value, labels, subset, minimum_length)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:
        result_check = isinstance(result, list) and all(isinstance(v, int) for v in result)
        assert result_check is True


# noinspection PyBroadException
def test_validate_labels_input(value, size, is_valid):

    try:
        result = _validate_labels_input(value, size)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:
        result_check = isinstance(result, list) and all(isinstance(v, str) for v in result)
        assert result_check is True


# noinspection PyBroadException
def test_validate_markov_chain(value, is_valid):

    if isinstance(value, str):
        value, skip = _evaluate(value)
    else:
        skip = False

    if skip:
        _pt_skip('Pandas library could not be imported.')
    else:

        try:
            result = _validate_markov_chain(value)
            result_is_valid = True
        except Exception:
            result = None
            result_is_valid = False

        actual = result_is_valid
        expected = is_valid

        assert actual == expected

        if result_is_valid:
            result_check = isinstance(result, _MarkovChain)
            assert result_check is True


# noinspection PyBroadException
def test_validate_markov_chains(value, is_valid):

    if isinstance(value, str):
        value, skip = _evaluate(value)
    else:
        skip = False

    if skip:
        _pt_skip('Pandas library could not be imported.')
    else:

        try:
            result = _validate_markov_chains(value)
            result_is_valid = True
        except Exception:
            result = None
            result_is_valid = False

        actual = result_is_valid
        expected = is_valid

        assert actual == expected

        if result_is_valid:
            result_check = isinstance(result, list) and all(isinstance(v, _MarkovChain) for v in result)
            assert result_check is True


# noinspection PyBroadException
def test_validate_mask(value, rows, columns, is_valid):

    value = _np_array(value)

    try:
        result = _validate_mask(value, rows, columns)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:
        result_check = isinstance(result, _np_ndarray)
        assert result_check is True


# noinspection PyBroadException
def test_validate_matrix(value, rows, columns, is_valid):

    value = _np_array(value)

    try:
        result = _validate_matrix(value, rows, columns)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:
        result_check = isinstance(result, _np_ndarray)
        assert result_check is True


# noinspection PyBroadException
def test_validate_model(value, is_valid):

    if isinstance(value, str):
        value, skip = _evaluate(value)
    else:
        skip = False

    if skip:
        _pt_skip('Pandas library could not be imported.')
    else:

        try:
            result = _validate_model(value)
            result_is_valid = True
        except Exception:
            result = None
            result_is_valid = False

        actual = result_is_valid
        expected = is_valid

        assert actual == expected

        if result_is_valid:
            result_check = isinstance(result, (_HiddenMarkovModel, _MarkovChain))
            assert result_check is True


# noinspection DuplicatedCode, PyBroadException
def test_validate_partitions(value, labels, is_valid):

    try:
        result = _validate_partitions(value, labels)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:

        actual = isinstance(result, list) and all(isinstance(v, list) for v in result) and all(isinstance(s, int) for v in result for s in v)
        expected = True

        assert actual == expected


# noinspection PyBroadException
def test_validate_random_distribution(value, accepted_values, is_valid):

    try:
        result = _validate_random_distribution(value, _npr_RandomState(0), accepted_values)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:

        actual = callable(result)
        expected = True

        assert actual == expected


# noinspection PyBroadException
def test_validate_rewards(value, size, is_valid):

    try:
        result = _validate_rewards(value, size)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:

        actual = isinstance(result, _np_ndarray)
        expected = True

        assert actual == expected


# noinspection DuplicatedCode, PyBroadException
def test_validate_sequence(value, labels, is_valid):

    try:
        result = _validate_sequence(value, labels)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:
        result_check = isinstance(result, list) and all(isinstance(v, int) for v in result)
        assert result_check is True


# noinspection DuplicatedCode, PyBroadException
def test_validate_sequences(value, labels, flex, is_valid):

    try:
        result = _validate_sequences(value, labels, flex)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:
        result_check = isinstance(result, list) and all(isinstance(v, list) and all(isinstance(s, int) for s in v) for v in result)
        assert result_check is True


# noinspection PyBroadException
def test_validate_status(value, labels, is_valid):

    try:
        result = _validate_status(value, labels)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:

        actual = isinstance(result, _np_ndarray)
        expected = True

        assert actual == expected


# noinspection PyBroadException
def test_validate_strings(value, size, is_valid):

    try:
        result = _validate_strings(value, size)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:

        actual = isinstance(result, list) and all(isinstance(v, str) for v in result)
        expected = True

        assert actual == expected


# noinspection DuplicatedCode, PyBroadException
def test_validate_time_points(value, is_valid):

    try:
        result = _validate_time_points(value)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:

        actual = isinstance(result, int) or (isinstance(result, list) and all(isinstance(v, int) for v in result))
        expected = True

        assert actual == expected


# noinspection PyBroadException
def test_validate_transition_function(value, is_valid):

    if isinstance(value, str):
        if value.startswith('def'):
            value = _string_to_function(value)
        elif value.startswith('lambda'):
            value = eval(value)

    try:
        result = _validate_transition_function(value)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:

        actual = callable(result)
        expected = True

        assert actual == expected


# noinspection PyBroadException
def test_validate_transition_matrix(value, size, is_valid):

    try:
        result = _validate_transition_matrix(value, size)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:

        actual = isinstance(result, _np_ndarray)
        expected = True

        assert actual == expected


# noinspection PyBroadException
def test_validate_vector(value, vector_type, flex, size, is_valid):

    try:
        result = _validate_vector(value, vector_type, flex, size)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:

        actual = isinstance(result, _np_ndarray)
        expected = True

        assert actual == expected
