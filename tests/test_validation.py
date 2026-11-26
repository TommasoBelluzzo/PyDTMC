# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Standard

import os.path as _osp
import pathlib as _pl
import random as _rd

# Libraries

import networkx as _nx
import numpy as _np
import numpy.random as _npr
import pytest as _pt

# Internal

from pydtmc import (
    HiddenMarkovModel as _HiddenMarkovModel,
    MarkovChain as _MarkovChain
)

# noinspection PyProtectedMember
from pydtmc._validation import (
    validate_attention_component as _validate_attention_component,
    validate_boolean as _validate_boolean,
    validate_boundary_condition as _validate_boundary_condition,
    validate_dictionary_hmm as _validate_dictionary_hmm,
    validate_dictionary_mc as _validate_dictionary_mc,
    validate_dimensions as _validate_dimensions,
    validate_dpi as _validate_dpi,
    validate_emission_matrix as _validate_emission_matrix,
    validate_enumerator as _validate_enumerator,
    validate_file_path as _validate_file_path,
    validate_float as _validate_float,
    validate_graph_hmm as _validate_graph_hmm,
    validate_graph_mc as _validate_graph_mc,
    validate_hidden_markov_model as _validate_hidden_markov_model,
    validate_hidden_markov_models as _validate_hidden_markov_models,
    validate_integer as _validate_integer,
    validate_hyperparameter as _validate_hyperparameter,
    validate_inertial_weights as _validate_inertial_weights,
    validate_interval as _validate_interval,
    validate_label as _validate_label,
    validate_labels_current as _validate_labels_current,
    validate_labels_pair as _validate_labels_pair,
    validate_labels_input as _validate_labels_input,
    validate_lags as _validate_lags,
    validate_markov_chain as _validate_markov_chain,
    validate_markov_chains as _validate_markov_chains,
    validate_mask as _validate_mask,
    validate_matrix as _validate_matrix,
    validate_model as _validate_model,
    validate_models as _validate_models,
    validate_none as _validate_none,
    validate_observable as _validate_observable,
    validate_partition as _validate_partitions,
    validate_pattern_probabilities as _validate_pattern_probabilities,
    validate_random_distribution as _validate_random_distribution,
    validate_rewards as _validate_rewards,
    validate_sequence as _validate_sequence,
    validate_sequences as _validate_sequences,
    validate_status as _validate_status,
    validate_string as _validate_string,
    validate_strings as _validate_strings,
    validate_time_points as _validate_time_points,
    validate_transition_function as _validate_transition_function,
    validate_transition_matrix as _validate_transition_matrix,
    validate_vector_birth_death as _validate_vector_birth_death,
    validate_vector_reweighting as _validate_vector_reweighting,
    validate_vector_stochastic as _validate_vector_stochastic
)

from tests.utilities import (
    evaluate as _evaluate,
    string_to_function as _string_to_function
)


#############
# CONSTANTS #
#############

_BASE_DIRECTORY = _osp.abspath(_osp.dirname(__file__))


#########
# TESTS #
#########

def test_validate_attention_component(value, states, features, is_valid):

    if is_valid:
        result = _validate_attention_component(value, states, features)
        assert isinstance(result, _np.ndarray) and (result.ndim == 2)
    else:
        with _pt.raises(Exception):
            _validate_attention_component(value, states, features)


def test_validate_boolean(value, is_valid):

    if is_valid:
        result = _validate_boolean(value)
        assert isinstance(result, bool)
    else:
        with _pt.raises(Exception):
            _validate_boolean(value)


def test_validate_boundary_condition(value, is_valid):

    if is_valid:
        result = _validate_boundary_condition(value)
        assert isinstance(result, (float, int, str))
    else:
        with _pt.raises(Exception):
            _validate_boundary_condition(value)


def test_validate_dictionary_hmm(dictionary_elements, is_valid):

    if not isinstance(dictionary_elements, list):
        d = dictionary_elements
    else:

        d = {}

        for dictionary_element in dictionary_elements:
            if len(dictionary_element) == 2:
                d[dictionary_element[0]] = dictionary_element[1]
            else:
                d[tuple(dictionary_element[:-1])] = dictionary_element[-1]

    if is_valid:
        result = _validate_dictionary_hmm(d)
        assert isinstance(result, dict) and all(isinstance(k, tuple) and all(isinstance(x, str) for x in k) and isinstance(v, float) for k, v in result.items())
    else:
        with _pt.raises(Exception):
            _validate_dictionary_hmm(d)


def test_validate_dictionary_mc(dictionary_elements, is_valid):

    if not isinstance(dictionary_elements, list):
        d = dictionary_elements
    else:

        d = {}

        for dictionary_element in dictionary_elements:
            if len(dictionary_element) == 2:
                d[dictionary_element[0]] = dictionary_element[1]
            else:
                d[tuple(dictionary_element[:-1])] = dictionary_element[-1]

    if is_valid:
        result = _validate_dictionary_mc(d)
        assert isinstance(result, dict) and all(isinstance(k, tuple) and all(isinstance(x, str) for x in k) and isinstance(v, float) for k, v in result.items())
    else:
        with _pt.raises(Exception):
            _validate_dictionary_mc(d)


def test_validate_dimensions(value, is_valid):

    if is_valid:
        result = _validate_dimensions(value)
        assert isinstance(result, list) and all(isinstance(v, int) and (v >= 2) for v in result)
    else:
        with _pt.raises(Exception):
            _validate_dimensions(value)


def test_validate_dpi(value, is_valid):

    if is_valid:
        result = _validate_dpi(value)
        assert isinstance(result, int)
    else:
        with _pt.raises(Exception):
            _validate_dpi(value)


def test_validate_emission_matrix(value, rows, columns, is_valid):

    if is_valid:
        result = _validate_emission_matrix(value, rows, columns)
        assert isinstance(result, _np.ndarray)
    else:
        with _pt.raises(Exception):
            _validate_emission_matrix(value, rows, columns)


def test_validate_enumerator(value, possible_values, is_valid):

    if is_valid:
        result = _validate_enumerator(value, possible_values)
        assert isinstance(result, str)
    else:
        with _pt.raises(Exception):
            _validate_enumerator(value, possible_values)


@_pt.mark.slow
def test_validate_file_path(value, accepted_extensions, write_permission, is_valid):

    if isinstance(value, str) and value.startswith('file_'):

        value = _osp.join(_BASE_DIRECTORY, f'files{_osp.sep}{value}')

        if _rd.random() < 0.5:
            value = _pl.Path(value)

    if is_valid:
        _validate_file_path(value, accepted_extensions, write_permission)
    else:
        with _pt.raises(Exception):
            _validate_file_path(value, accepted_extensions, write_permission)


def test_validate_float(value, lower_limit, upper_limit, is_valid):

    lower_limit = None if lower_limit is None else tuple(lower_limit)
    upper_limit = None if upper_limit is None else tuple(upper_limit)

    if is_valid:
        result = _validate_float(value, lower_limit, upper_limit)
        assert isinstance(result, float)
    else:
        with _pt.raises(Exception):
            _validate_float(value, lower_limit, upper_limit)


def test_validate_graph_hmm(multi, graph_nodes, graph_edges, is_valid):

    if not isinstance(graph_nodes, list):
        graph = graph_nodes
    else:

        graph = _nx.MultiDiGraph() if multi else _nx.DiGraph()

        for index, nodes in enumerate(graph_nodes):
            graph.add_nodes_from(nodes, layer=index)

        for node_i, node_j, edge_type, edge_weight in graph_edges:
            graph.add_edge(node_i, node_j, type=edge_type, weight=edge_weight)

    if is_valid:
        result = _validate_graph_hmm(graph)
        assert isinstance(result, _nx.DiGraph)
    else:
        with _pt.raises(Exception):
            _validate_graph_hmm(graph)


def test_validate_graph_mc(multi, graph_nodes, graph_edges, is_valid):

    if not isinstance(graph_nodes, list):
        graph = graph_nodes
    else:

        graph = _nx.MultiDiGraph() if multi else _nx.DiGraph()

        graph.add_nodes_from(graph_nodes)

        for node_i, node_j, edge_weight in graph_edges:
            graph.add_edge(node_i, node_j, weight=edge_weight)

    if is_valid:
        result = _validate_graph_mc(graph)
        assert isinstance(result, _nx.DiGraph)
    else:
        with _pt.raises(Exception):
            _validate_graph_mc(graph)


def test_validate_hidden_markov_model(value, is_valid):

    if isinstance(value, str):
        value, skip = _evaluate(value)
    else:
        skip = False

    if skip:
        _pt.skip('Pandas library could not be imported.')
    else:
        if is_valid:
            result = _validate_hidden_markov_model(value)
            assert isinstance(result, _HiddenMarkovModel)
        else:
            with _pt.raises(Exception):
                _validate_hidden_markov_model(value)


def test_validate_hidden_markov_models(value, minimum_length, is_valid):

    if isinstance(value, str):
        value, skip = _evaluate(value)
    else:
        skip = False

    if skip:
        _pt.skip('Pandas library could not be imported.')
    else:
        if is_valid:
            result = _validate_hidden_markov_models(value, minimum_length)
            assert isinstance(result, list) and all(isinstance(v, _HiddenMarkovModel) for v in result)
        else:
            with _pt.raises(Exception):
                _validate_hidden_markov_models(value, minimum_length)


def test_validate_hyperparameter(value, size, is_valid):

    if is_valid:
        result = _validate_hyperparameter(value, size)
        assert isinstance(result, _np.ndarray)
    else:
        with _pt.raises(Exception):
            _validate_hyperparameter(value, size)


def test_validate_inertial_weights(value, size, is_valid):

    if is_valid:

        result = _validate_inertial_weights(value, size)
        assert isinstance(result, _np.ndarray) and (result.ndim == 1) and (result.size == size)

        if _np.isscalar(value):
            assert _np.all(result == float(value))

    else:
        with _pt.raises(Exception):
            _validate_inertial_weights(value, size)


def test_validate_integer(value, lower_limit, upper_limit, is_valid):

    lower_limit = None if lower_limit is None else tuple(lower_limit)
    upper_limit = None if upper_limit is None else tuple(upper_limit)

    if is_valid:
        result = _validate_integer(value, lower_limit, upper_limit)
        assert isinstance(result, int)
    else:
        with _pt.raises(Exception):
            _validate_integer(value, lower_limit, upper_limit)


def test_validate_interval(value, is_valid):

    value = tuple(value) if isinstance(value, list) else value

    if is_valid:
        result = _validate_interval(value)
        assert all(isinstance(v, float) for v in result)
    else:
        with _pt.raises(Exception):
            _validate_interval(value)


def test_validate_label(value, labels, is_valid):

    if is_valid:
        result = _validate_label(value, labels)
        assert isinstance(result, int)
        assert result == (labels.index(value) if isinstance(value, str) else labels.index(labels[value]))
    else:
        with _pt.raises(Exception):
            _validate_label(value, labels)


def test_validate_labels_current(value, labels, subset, minimum_length, is_valid):

    if is_valid:
        result = _validate_labels_current(value, labels, subset, minimum_length)
        assert isinstance(result, list) and all(isinstance(v, int) for v in result)
    else:
        with _pt.raises(Exception):
            _validate_labels_current(value, labels, subset, minimum_length)


def test_validate_labels_input(value, size, is_valid):

    if is_valid:
        result = _validate_labels_input(value, size)
        assert isinstance(result, list) and all(isinstance(v, str) for v in result)
    else:
        with _pt.raises(Exception):
            _validate_labels_input(value, size)


def test_validate_labels_pair(value, labels, is_valid):

    if is_valid:
        result = _validate_labels_pair(value, labels)
        assert isinstance(result, list) and all(isinstance(v, int) for v in result)
    else:
        with _pt.raises(Exception):
            _validate_labels_pair(value, labels)


def test_validate_lags(value, size, is_valid):

    if is_valid:
        result = _validate_lags(value, size)
        assert isinstance(result, list) and all(isinstance(v, int) for v in result) and (result == sorted(result))
    else:
        with _pt.raises(Exception):
            _validate_lags(value, size)


def test_validate_markov_chain(value, size, is_valid):

    if isinstance(value, str):
        value, skip = _evaluate(value)
    else:
        skip = False

    if skip:
        _pt.skip('Pandas library could not be imported.')
    else:
        if is_valid:
            result = _validate_markov_chain(value, size)
            assert isinstance(result, _MarkovChain)
        else:
            with _pt.raises(Exception):
                _validate_markov_chain(value, size)


def test_validate_markov_chains(value, minimum_length, size, is_valid):

    if isinstance(value, str):
        value, skip = _evaluate(value)
    else:
        skip = False

    if skip:
        _pt.skip('Pandas library could not be imported.')
    else:
        if is_valid:
            result = _validate_markov_chains(value, minimum_length, size)
            assert isinstance(result, list) and all(isinstance(v, _MarkovChain) for v in result)
        else:
            with _pt.raises(Exception):
                _validate_markov_chains(value, minimum_length, size)


def test_validate_mask(value, rows, columns, is_valid):

    value = _np.array(value)

    if is_valid:
        result = _validate_mask(value, rows, columns)
        assert isinstance(result, _np.ndarray)
    else:
        with _pt.raises(Exception):
            _validate_mask(value, rows, columns)


def test_validate_matrix(value, rows, columns, is_valid):

    value = _np.array(value)

    if is_valid:
        result = _validate_matrix(value, rows, columns)
        assert isinstance(result, _np.ndarray)
    else:
        with _pt.raises(Exception):
            _validate_matrix(value, rows, columns)


def test_validate_model(value, is_valid):

    if isinstance(value, str):
        value, skip = _evaluate(value)
    else:
        skip = False

    if skip:
        _pt.skip('Pandas library could not be imported.')
    else:
        if is_valid:
            result = _validate_model(value)
            assert isinstance(result, (_HiddenMarkovModel, _MarkovChain))
        else:
            with _pt.raises(Exception):
                _validate_model(value)


def test_validate_models(value, minimum_length, is_valid):

    if isinstance(value, str):
        value, skip = _evaluate(value)
    else:
        skip = False

    if skip:
        _pt.skip('Pandas library could not be imported.')
    else:
        if is_valid:
            result = _validate_models(value, minimum_length)
            assert isinstance(result, list) and all(isinstance(v, (_HiddenMarkovModel, _MarkovChain)) for v in result)
        else:
            with _pt.raises(Exception):
                _validate_models(value, minimum_length)


def test_validate_none(value, is_valid):

    if is_valid:
        actual = _validate_none(value)
        assert actual is None
    else:
        with _pt.raises(TypeError):
            _validate_none(value)


def test_validate_observable(value, size, is_valid):

    if is_valid:
        result = _validate_observable(value, size)
        assert isinstance(result, (str, _np.ndarray))
    else:
        with _pt.raises(Exception):
            _validate_observable(value, size)


def test_validate_partition(value, labels, is_valid):

    if is_valid:
        result = _validate_partitions(value, labels)
        assert isinstance(result, list) and all(isinstance(v, list) for v in result) and all(isinstance(s, int) for v in result for s in v)
    else:
        with _pt.raises(Exception):
            _validate_partitions(value, labels)


def test_validate_pattern_probabilities(value, is_valid):

    if is_valid:
        result = _validate_pattern_probabilities(value)
        assert isinstance(result, dict) and all(isinstance(k, str) and (len(k) == 1) and isinstance(v, float) for k, v in result.items())
        assert _np.isclose(sum(result.values()), 1.0)
    else:
        with _pt.raises(Exception):
            _validate_pattern_probabilities(value)


def test_validate_random_distribution(value, accepted_values, is_valid):

    if is_valid:
        result = _validate_random_distribution(value, _npr.RandomState(0), accepted_values)  # pylint: disable=no-member
        assert callable(result)
    else:
        with _pt.raises(Exception):
            _validate_random_distribution(value, _npr.RandomState(0), accepted_values)  # pylint: disable=no-member


def test_validate_rewards(value, size, is_valid):

    if is_valid:
        result = _validate_rewards(value, size)
        assert isinstance(result, _np.ndarray) and (result.ndim == 1) and (result.size == size)
    else:
        with _pt.raises(Exception):
            _validate_rewards(value, size)


def test_validate_sequence(value, labels, is_valid):

    if is_valid:
        result = _validate_sequence(value, labels)
        assert isinstance(result, list) and all(isinstance(v, int) for v in result)
    else:
        with _pt.raises(Exception):
            _validate_sequence(value, labels)


def test_validate_sequences(value, labels, flex, is_valid):

    if is_valid:
        result = _validate_sequences(value, labels, flex)
        assert isinstance(result, list) and all(isinstance(v, list) and all(isinstance(s, int) for s in v) for v in result)
    else:
        with _pt.raises(Exception):
            _validate_sequences(value, labels, flex)


def test_validate_status(value, labels, is_valid):

    if is_valid:
        result = _validate_status(value, labels, False)
        assert isinstance(result, _np.ndarray) and (result.ndim == 1) and (result.size == len(labels))
    else:
        with _pt.raises(Exception):
            _validate_status(value, labels, False)


def test_validate_string(value, excluded_chars, is_valid):

    if is_valid:
        result = _validate_string(value, excluded_chars)
        assert isinstance(result, str)
    else:
        with _pt.raises(Exception):
            _validate_string(value, excluded_chars)


def test_validate_strings(value, excluded_chars, size, is_valid):

    if is_valid:
        result = _validate_strings(value, excluded_chars, size)
        assert isinstance(result, list) and all(isinstance(v, str) for v in result)
    else:
        with _pt.raises(Exception):
            _validate_strings(value, excluded_chars, size)


def test_validate_time_points(value, is_valid):

    if is_valid:
        result = _validate_time_points(value)
        assert isinstance(result, int) or (isinstance(result, list) and all(isinstance(v, int) for v in result))
    else:
        with _pt.raises(Exception):
            _validate_time_points(value)


def test_validate_transition_function(value, is_valid):

    if isinstance(value, str):
        if value.startswith('def'):
            value = _string_to_function(value)
        elif value.startswith('lambda'):
            value = eval(value)

    if is_valid:
        result = _validate_transition_function(value)
        assert callable(result)
    else:
        with _pt.raises(Exception):
            _validate_transition_function(value)


def test_validate_transition_matrix(value, size, is_valid):

    if is_valid:
        result = _validate_transition_matrix(value, size)
        assert isinstance(result, _np.ndarray)
    else:
        with _pt.raises(Exception):
            _validate_transition_matrix(value, size)


def test_validate_vector_birth_death(value, vector_type, is_valid):

    if is_valid:
        result = _validate_vector_birth_death(value, vector_type)
        assert isinstance(result, _np.ndarray) and (result.ndim == 1)
    else:
        with _pt.raises(Exception):
            _validate_vector_birth_death(value, vector_type)


def test_validate_vector_stochastic(value, size, is_valid):

    if is_valid:
        result = _validate_vector_stochastic(value, size, False)
        assert isinstance(result, _np.ndarray) and (result.ndim == 1) and (result.size == size)
    else:
        with _pt.raises(Exception):
            _validate_vector_stochastic(value, size, False)


def test_validate_vector_reweighting(value, size, is_valid):

    if is_valid:
        result = _validate_vector_reweighting(value, size)
        assert isinstance(result, _np.ndarray) and (result.ndim == 1) and (result.size == size)
    else:
        with _pt.raises(Exception):
            _validate_vector_reweighting(value, size)
