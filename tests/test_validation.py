# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Standard

from ast import (
    parse as _ast_parse
)

from re import (
    search as _re_search
)

from types import (
    CodeType as _CodeType,
    FunctionType as _FunctionType
)

# Libraries

import networkx as _nx
import numpy as _np
import pytest as _pt

# noinspection PyUnresolvedReferences
import scipy.sparse as _spsp  # noqa

try:
    import pandas as _pd
except ImportError:  # noqa
    _pd = None

# Internal

from pydtmc import (
    MarkovChain as _MarkovChain
)

# noinspection PyUnresolvedReferences
from pydtmc.base_class import (  # noqa
    BaseClass as _BaseClass
)

# noinspection PyProtectedMember
from pydtmc.validation import (
    _extract,
    _extract_as_numeric,
    validate_boolean as _validate_boolean,
    validate_boundary_condition as _validate_boundary_condition,
    validate_dictionary as _validate_dictionary,
    validate_distribution as _validate_distribution,
    validate_dpi as _validate_dpi,
    validate_enumerator as _validate_enumerator,
    validate_float as _validate_float,
    validate_graph as _validate_graph,
    validate_integer as _validate_integer,
    validate_hyperparameter as _validate_hyperparameter,
    validate_interval as _validate_interval,
    validate_markov_chain as _validate_markov_chain,
    validate_mask as _validate_mask,
    validate_matrix as _validate_matrix,
    validate_partitions as _validate_partitions,
    validate_rewards as _validate_rewards,
    validate_state as _validate_state,
    validate_state_names as _validate_state_names,
    validate_states as _validate_states,
    validate_status as _validate_status,
    validate_time_points as _validate_time_points,
    validate_transition_function as _validate_transition_function,
    validate_transition_matrix as _validate_transition_matrix,
    validate_vector as _validate_vector
)


#############
# FUNCTIONS #
#############

def _eval_replace(value):

    value = value.replace('np.', '_np.')
    value = value.replace('nx.', '_nx.')
    value = value.replace('pd.', '_pd.')
    value = value.replace('spsp.', '_spsp.')

    return value


def _string_to_function(source):

    ast_tree = _ast_parse(source)
    module_object = compile(ast_tree, '<ast>', 'exec')
    code_object = [c for c in module_object.co_consts if isinstance(c, _CodeType)][0]

    # noinspection PyArgumentList
    f = _FunctionType(code_object, {})

    return f


#########
# TESTS #
#########

def test_validate_extract(value, evaluate, is_valid):

    if value is not None and isinstance(value, str) and evaluate:
        value = eval(value)

    # noinspection PyBroadException
    try:
        result = _extract(value)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, list)
        expected = True

        assert actual == expected


def test_validate_extract_as_numeric(value, evaluate, is_valid):

    should_skip = False

    if value is not None and isinstance(value, str) and evaluate:

        if 'pd.' in value and _pd is None:
            should_skip = True
        else:
            value = _eval_replace(value)
            value = eval(value)

    if should_skip:
        _pt.skip('The test could not be performed because Pandas library could not be imported.')
    else:

        # noinspection PyBroadException
        try:
            result = _extract_as_numeric(value)
            result_is_valid = True
        except Exception:
            result = None
            result_is_valid = False

        actual = result_is_valid
        expected = is_valid

        assert actual == expected

        if result is not None:

            actual = isinstance(result, _np.ndarray)
            expected = True

            assert actual == expected


def test_validate_boolean(value, is_valid):

    # noinspection PyBroadException
    try:
        result = _validate_boolean(value)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, bool)
        expected = True

        assert actual == expected


def test_validate_boundary_condition(value, is_valid):

    # noinspection PyBroadException
    try:
        result = _validate_boundary_condition(value)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, (float, int, str))
        expected = True

        assert actual == expected


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

    # noinspection PyBroadException
    try:
        result = _validate_dictionary(d)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, dict)
        expected = True

        assert actual == expected


def test_validate_distribution(value, size, is_valid):

    if isinstance(value, list):
        for index, v in enumerate(value):
            value[index] = _np.asarray(v)

    # noinspection PyBroadException
    try:
        result = _validate_distribution(value, size)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, int) or (isinstance(result, list) and all(isinstance(v, _np.ndarray) for v in result))
        expected = True

        assert actual == expected


def test_validate_dpi(value, is_valid):

    # noinspection PyBroadException
    try:
        result = _validate_dpi(value)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, int)
        expected = True

        assert actual == expected


def test_validate_enumerator(value, possible_values, is_valid):

    # noinspection PyBroadException
    try:
        result = _validate_enumerator(value, possible_values)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, str)
        expected = True

        assert actual == expected


def test_validate_float(value, lower_limit, upper_limit, is_valid):

    lower_limit = None if lower_limit is None else tuple(lower_limit)
    upper_limit = None if upper_limit is None else tuple(upper_limit)

    # noinspection PyBroadException
    try:
        result = _validate_float(value, lower_limit, upper_limit)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, float)
        expected = True

        assert actual == expected


def test_validate_graph(graph_data, is_valid):

    if graph_data is None:
        g = None
    elif isinstance(graph_data, list) and all(isinstance(x, list) for x in graph_data):
        g = _nx.from_numpy_matrix(_np.array(graph_data), create_using=_nx.DiGraph()) if len(graph_data) > 0 else _nx.DiGraph()
        g = _nx.relabel_nodes(g, dict(zip(range(len(g.nodes)), [str(i + 1) for i in range(len(g.nodes))])))
    else:

        g = _nx.DiGraph()

        for x in graph_data:
            g.add_node(x)

    # noinspection PyBroadException
    try:
        result = _validate_graph(g)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, _nx.DiGraph)
        expected = True

        assert actual == expected

    if graph_data is None:
        g = None
    elif isinstance(graph_data, list) and all(isinstance(x, list) for x in graph_data):
        g = _nx.from_numpy_matrix(_np.array(graph_data), create_using=_nx.DiGraph()) if len(graph_data) > 0 else _nx.DiGraph()
        g = _nx.relabel_nodes(g, dict(zip(range(len(g.nodes)), [str(i + 1) for i in range(len(g.nodes))])))
    else:

        g = _nx.DiGraph()

        for x in graph_data:
            g.add_node(x)

    # noinspection PyBroadException
    try:
        result = _validate_graph(g)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, _nx.DiGraph)
        expected = True

        assert actual == expected


def test_validate_integer(value, lower_limit, upper_limit, is_valid):

    lower_limit = None if lower_limit is None else tuple(lower_limit)
    upper_limit = None if upper_limit is None else tuple(upper_limit)

    # noinspection PyBroadException
    try:
        result = _validate_integer(value, lower_limit, upper_limit)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, int)
        expected = True

        assert actual == expected


def test_validate_hyperparameter(value, size, is_valid):

    # noinspection PyBroadException
    try:
        result = _validate_hyperparameter(value, size)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, _np.ndarray)
        expected = True

        assert actual == expected


def test_validate_interval(value, is_valid):

    value = tuple(value) if isinstance(value, list) else value

    # noinspection PyBroadException
    try:
        result = _validate_interval(value)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = all(isinstance(v, float) for v in result)
        expected = True

        assert actual == expected


def test_validate_markov_chain(value, is_valid):

    should_skip = False

    if value is not None and isinstance(value, str):

        if 'pd.' in value and _pd is None:
            should_skip = True
        else:

            value = _eval_replace(value)

            if _re_search(r'^(?:BaseClass|MarkovChain)\(', value):
                value = '_' + value

            value = eval(value)

    if should_skip:
        _pt.skip('The test could not be performed because Pandas library could not be imported.')
    else:

        # noinspection PyBroadException
        try:
            result = _validate_markov_chain(value)
            result_is_valid = True
        except Exception:
            result = None
            result_is_valid = False

        actual = result_is_valid
        expected = is_valid

        assert actual == expected

        if result is not None:
            actual = isinstance(result, _MarkovChain)
            expected = True

            assert actual == expected


def test_validate_mask(value, size, is_valid):

    value = _np.asarray(value)

    # noinspection PyBroadException
    try:
        result = _validate_mask(value, size)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, _np.ndarray)
        expected = True

        assert actual == expected


def test_validate_matrix(value, is_valid):

    value = _np.asarray(value)

    # noinspection PyBroadException
    try:
        result = _validate_matrix(value)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, _np.ndarray)
        expected = True

        assert actual == expected


def test_validate_partitions(value, current_states, is_valid):

    # noinspection PyBroadException
    try:
        result = _validate_partitions(value, current_states)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, list) and all(isinstance(v, list) for v in result) and all(isinstance(s, int) for v in result for s in v)
        expected = True

        assert actual == expected


def test_validate_rewards(value, size, is_valid):

    # noinspection PyBroadException
    try:
        result = _validate_rewards(value, size)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, _np.ndarray)
        expected = True

        assert actual == expected


def test_validate_state(value, current_states, is_valid):

    # noinspection PyBroadException
    try:
        result = _validate_state(value, current_states)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, int)
        expected = True

        assert actual == expected

        actual = result
        expected = current_states.index(value) if isinstance(value, str) else current_states.index(current_states[value])

        assert actual == expected


def test_validate_state_names(value, size, is_valid):

    # noinspection PyBroadException
    try:
        result = _validate_state_names(value, size)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, list) and all(isinstance(v, str) for v in result)
        expected = True

        assert actual == expected


def test_validate_states(value, current_states, states_type, flex, is_valid):

    # noinspection PyBroadException
    try:
        result = _validate_states(value, current_states, states_type, flex)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, list) and all(isinstance(v, int) for v in result)
        expected = True

        assert actual == expected


def test_validate_status(value, current_states, is_valid):

    # noinspection PyBroadException
    try:
        result = _validate_status(value, current_states)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, _np.ndarray)
        expected = True

        assert actual == expected


def test_validate_time_points(value, is_valid):

    # noinspection PyBroadException
    try:
        result = _validate_time_points(value)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, int) or (isinstance(result, list) and all(isinstance(v, int) for v in result))
        expected = True

        assert actual == expected


def test_validate_transition_function(value, is_valid):

    if value is not None and isinstance(value, str):
        if value.startswith('def'):
            value = _string_to_function(value)
        elif value.startswith('lambda'):
            value = eval(value)

    # noinspection PyBroadException
    try:
        result = _validate_transition_function(value)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = callable(result)
        expected = True

        assert actual == expected


def test_validate_transition_matrix(value, is_valid):

    # noinspection PyBroadException
    try:
        result = _validate_transition_matrix(value)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, _np.ndarray)
        expected = True

        assert actual == expected


def test_validate_vector(value, vector_type, flex, size, is_valid):

    # noinspection PyBroadException
    try:
        result = _validate_vector(value, vector_type, flex, size)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, _np.ndarray)
        expected = True

        assert actual == expected
