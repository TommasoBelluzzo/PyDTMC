# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Standard

from ast import (
    parse
)

# noinspection PyPep8Naming
from re import (
    IGNORECASE as flag_ignorecase,
    search
)

from types import (
    CodeType,
    FunctionType
)

# Libraries

import networkx as nx
import numpy as np
# noinspection PyUnresolvedReferences
import scipy.sparse as spsp  # noqa

try:
    import pandas as pd
except ImportError:
    pd = None

from pytest import (
    skip
)

# Internal

from pydtmc import (
    MarkovChain
)

# noinspection PyUnresolvedReferences
from pydtmc.base_class import (  # noqa
    BaseClass
)

# noinspection PyProtectedMember
from pydtmc.validation import (
    _extract,
    _extract_as_numeric,
    validate_boolean,
    validate_boundary_condition,
    validate_dictionary,
    validate_distribution,
    validate_dpi,
    validate_enumerator,
    validate_float,
    validate_graph,
    validate_integer,
    validate_hyperparameter,
    validate_interval,
    validate_markov_chain,
    validate_mask,
    validate_matrix,
    validate_partitions,
    validate_rewards,
    validate_state,
    validate_state_names,
    validate_states,
    validate_status,
    validate_time_points,
    validate_transition_function,
    validate_transition_matrix,
    validate_vector
)


#############
# FUNCTIONS #
#############

def _string_to_function(source):

    ast_tree = parse(source)
    module_object = compile(ast_tree, '<ast>', 'exec')
    code_object = [c for c in module_object.co_consts if isinstance(c, CodeType)][0]

    # noinspection PyArgumentList
    f = FunctionType(code_object, {})

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

        if value.startswith('pd.') and pd is None:
            should_skip = True
        else:
            value = eval(value)

    if should_skip:
        skip('The test could not be performed because Pandas library could not be imported.')
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

            actual = isinstance(result, np.ndarray)
            expected = True

            assert actual == expected


def test_validate_boolean(value, is_valid):

    # noinspection PyBroadException
    try:
        result = validate_boolean(value)
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
        result = validate_boundary_condition(value)
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
        result = validate_dictionary(d)
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
            value[index] = np.asarray(v)

    # noinspection PyBroadException
    try:
        result = validate_distribution(value, size)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, int) or (isinstance(result, list) and all(isinstance(v, np.ndarray) for v in result))
        expected = True

        assert actual == expected


def test_validate_dpi(value, is_valid):

    # noinspection PyBroadException
    try:
        result = validate_dpi(value)
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
        result = validate_enumerator(value, possible_values)
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
        result = validate_float(value, lower_limit, upper_limit)
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
        g = nx.from_numpy_matrix(np.array(graph_data), create_using=nx.DiGraph()) if len(graph_data) > 0 else nx.DiGraph()
        g = nx.relabel_nodes(g, dict(zip(range(len(g.nodes)), [str(i + 1) for i in range(len(g.nodes))])))
    else:

        g = nx.DiGraph()

        for x in graph_data:
            g.add_node(x)

    # noinspection PyBroadException
    try:
        result = validate_graph(g)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, nx.DiGraph)
        expected = True

        assert actual == expected

    if graph_data is None:
        g = None
    elif isinstance(graph_data, list) and all(isinstance(x, list) for x in graph_data):
        g = nx.from_numpy_matrix(np.array(graph_data), create_using=nx.DiGraph()) if len(graph_data) > 0 else nx.DiGraph()
        g = nx.relabel_nodes(g, dict(zip(range(len(g.nodes)), [str(i + 1) for i in range(len(g.nodes))])))
    else:

        g = nx.DiGraph()

        for x in graph_data:
            g.add_node(x)

    # noinspection PyBroadException
    try:
        result = validate_graph(g)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, nx.DiGraph)
        expected = True

        assert actual == expected


def test_validate_integer(value, lower_limit, upper_limit, is_valid):

    lower_limit = None if lower_limit is None else tuple(lower_limit)
    upper_limit = None if upper_limit is None else tuple(upper_limit)

    # noinspection PyBroadException
    try:
        result = validate_integer(value, lower_limit, upper_limit)
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
        result = validate_hyperparameter(value, size)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, np.ndarray)
        expected = True

        assert actual == expected


def test_validate_interval(value, is_valid):

    value = tuple(value) if isinstance(value, list) else value

    # noinspection PyBroadException
    try:
        result = validate_interval(value)
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

    if value is not None and isinstance(value, str) and search(r'^[A-Z]+\([^)]*\)$', value, flags=flag_ignorecase):
        value = eval(value)

    # noinspection PyBroadException
    try:
        result = validate_markov_chain(value)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, MarkovChain)
        expected = True

        assert actual == expected


def test_validate_mask(value, size, is_valid):

    value = np.asarray(value)

    # noinspection PyBroadException
    try:
        result = validate_mask(value, size)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, np.ndarray)
        expected = True

        assert actual == expected


def test_validate_matrix(value, is_valid):

    value = np.asarray(value)

    # noinspection PyBroadException
    try:
        result = validate_matrix(value)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, np.ndarray)
        expected = True

        assert actual == expected


def test_validate_partitions(value, current_states, is_valid):

    # noinspection PyBroadException
    try:
        result = validate_partitions(value, current_states)
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
        result = validate_rewards(value, size)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, np.ndarray)
        expected = True

        assert actual == expected


def test_validate_state(value, current_states, is_valid):

    # noinspection PyBroadException
    try:
        result = validate_state(value, current_states)
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
        result = validate_state_names(value, size)
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
        result = validate_states(value, current_states, states_type, flex)
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
        result = validate_status(value, current_states)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, np.ndarray)
        expected = True

        assert actual == expected


def test_validate_time_points(value, is_valid):

    # noinspection PyBroadException
    try:
        result = validate_time_points(value)
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
        result = validate_transition_function(value)
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
        result = validate_transition_matrix(value)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, np.ndarray)
        expected = True

        assert actual == expected


def test_validate_vector(value, vector_type, flex, size, is_valid):

    # noinspection PyBroadException
    try:
        result = validate_vector(value, vector_type, flex, size)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, np.ndarray)
        expected = True

        assert actual == expected
