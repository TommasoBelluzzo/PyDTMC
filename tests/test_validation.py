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
    from_numpy_matrix as _nx_from_numpy_matrix,
    relabel_nodes as _nx_relabel_nodes
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
    validate_hmm_emission as _validate_hmm_emission,
    validate_hmm_sequence as _validate_hmm_sequence,
    validate_hmm_symbols as _validate_hmm_symbols,
    validate_integer as _validate_integer,
    validate_hyperparameter as _validate_hyperparameter,
    validate_interval as _validate_interval,
    validate_labels_input as _validate_labels_input,
    validate_markov_chain as _validate_markov_chain,
    validate_markov_chains as _validate_markov_chains,
    validate_mask as _validate_mask,
    validate_matrix as _validate_matrix,
    validate_object as _validate_object,
    validate_partitions as _validate_partitions,
    validate_random_distribution as _validate_random_distribution,
    validate_rewards as _validate_rewards,
    validate_state as _validate_state,
    validate_states as _validate_states,
    validate_status as _validate_status,
    validate_strings as _validate_strings,
    validate_time_points as _validate_time_points,
    validate_transition_function as _validate_transition_function,
    validate_transition_matrix as _validate_transition_matrix,
    validate_vector as _validate_vector,
    validate_walk as _validate_walk,
    validate_walks as _validate_walks
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
def test_validate_graph(graph_data, is_valid):

    if graph_data is None:
        g = None
    else:

        if isinstance(graph_data, list) and all(isinstance(x, list) for x in graph_data):

            g = _nx_from_numpy_matrix(_np_array(graph_data), create_using=_nx_DiGraph()) if len(graph_data) > 0 else _nx_DiGraph()
            g = _nx_relabel_nodes(g, dict(zip(range(len(g.nodes)), [str(i + 1) for i in range(len(g.nodes))])))

        else:

            g = _nx_DiGraph()

            for x in graph_data:
                g.add_node(x)

    try:
        result = _validate_graph(g)
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


# noinspection DuplicatedCode, PyBroadException
def test_validate_hmm_sequence(value, possible_states, possible_symbols, is_valid):

    try:
        result = _validate_hmm_sequence(value, possible_states, possible_symbols)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:
        result_check = isinstance(result, tuple) and all(isinstance(v, list) and all(isinstance(s, int) for s in v) for v in result)
        assert result_check is True


# noinspection PyBroadException
def test_validate_hmm_symbols(value, possible_symbols, allow_lists, is_valid):

    try:
        result = _validate_hmm_symbols(value, possible_symbols, allow_lists)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:

        if allow_lists:
            result_check = isinstance(result, list) and all(isinstance(v, list) for v in result) and all(isinstance(s, int) for v in result for s in v)
        else:
            result_check = isinstance(result, list) and all(isinstance(v, int) for v in result)

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

        actual = isinstance(result, list) and all(isinstance(v, str) for v in result)
        expected = True

        assert actual == expected


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

            actual = isinstance(result, _MarkovChain)
            expected = True

            assert actual == expected


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

            actual = isinstance(result, list) and all(isinstance(v, _MarkovChain) for v in result)
            expected = True

            assert actual == expected


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

        actual = isinstance(result, _np_ndarray)
        expected = True

        assert actual == expected


# noinspection PyBroadException
def test_validate_matrix(value, columns, rows, is_valid):

    value = _np_array(value)

    try:
        result = _validate_matrix(value, columns, rows)
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
def test_validate_object(value, is_valid):

    if isinstance(value, str):
        value, skip = _evaluate(value)
    else:
        skip = False

    if skip:
        _pt_skip('Pandas library could not be imported.')
    else:

        try:
            _validate_object(value)
            result_is_valid = True
        except Exception:
            result_is_valid = False

        actual = result_is_valid
        expected = is_valid

        assert actual == expected


# noinspection DuplicatedCode, PyBroadException
def test_validate_partitions(value, current_states, is_valid):

    try:
        result = _validate_partitions(value, current_states)
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


# noinspection PyBroadException
def test_validate_state(value, current_states, is_valid):

    try:
        result = _validate_state(value, current_states)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:

        actual = isinstance(result, int)
        expected = True

        assert actual == expected

        actual = result
        expected = current_states.index(value) if isinstance(value, str) else current_states.index(current_states[value])

        assert actual == expected


# noinspection PyBroadException
def test_validate_states(value, possible_states, subset, length_limit, is_valid):

    try:
        result = _validate_states(value, possible_states, subset, length_limit)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:

        actual = isinstance(result, list) and all(isinstance(v, int) for v in result)
        expected = True

        assert actual == expected


# noinspection PyBroadException
def test_validate_status(value, current_states, is_valid):

    try:
        result = _validate_status(value, current_states)
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


# noinspection PyBroadException
def test_validate_walk(value, possible_states, is_valid):

    try:
        result = _validate_walk(value, possible_states)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result_is_valid:

        actual = isinstance(result, list) and all(isinstance(v, int) for v in result)
        expected = True

        assert actual == expected


# noinspection DuplicatedCode, PyBroadException
def test_validate_walks(value, possible_states, is_valid):

    try:
        result = _validate_walks(value, possible_states)
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
