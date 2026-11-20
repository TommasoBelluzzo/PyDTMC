# -*- coding: utf-8 -*-

__all__ = [
    'validate_boolean',
    'validate_boundary_condition',
    'validate_dictionary_hmm',
    'validate_dictionary_mc',
    'validate_dimensions',
    'validate_dpi',
    'validate_emission_matrix',
    'validate_enumerator',
    'validate_file_path',
    'validate_float',
    'validate_graph_hmm',
    'validate_graph_mc',
    'validate_hidden_markov_model',
    'validate_hidden_markov_models',
    'validate_hyperparameter',
    'validate_inertial_weights',
    'validate_integer',
    'validate_interval',
    'validate_label',
    'validate_labels_current',
    'validate_labels_pair',
    'validate_labels_input',
    'validate_markov_chain',
    'validate_markov_chains',
    'validate_mask',
    'validate_matrix',
    'validate_model',
    'validate_models',
    'validate_none',
    'validate_observable',
    'validate_partition',
    'validate_pattern_probabilities',
    'validate_random_distribution',
    'validate_rewards',
    'validate_sequence',
    'validate_sequences',
    'validate_status',
    'validate_string',
    'validate_strings',
    'validate_time_points',
    'validate_transition_function',
    'validate_transition_matrix',
    'validate_vector_birth_death',
    'validate_vector_stochastic'
]


###########
# IMPORTS #
###########

# Standard

import inspect as _ins
import itertools as _it
import os.path as _osp
import pathlib as _pl

# Libraries

import numpy as _np

# Internal

from . import _constants as _c
from . import _custom_types as _ct
from . import _utilities as _u


#############
# FUNCTIONS #
#############

def _validate_dictionary_keys(value: _ct.tany, keys_length: int) -> None:

    if (not _u.is_dictionary(value)) or (len(value) == 0):
        raise ValueError('The "@arg@" parameter must be a non-empty dictionary.')

    if not all(_u.is_tuple(k) and len(k) == keys_length and all(_u.is_string(x) for x in k) for k in value.keys()):
        raise ValueError(f'The "@arg@" parameter keys must be defined as tuples of {keys_length:d} non-empty strings.')


def _validate_dictionary_values(value: _ct.tany) -> _ct.tmdl_dict:

    values = value.values()

    if not all(_u.is_number(v) for v in values):
        raise ValueError('The "@arg@" parameter values must be float or integer numbers.')

    result = {}

    for k, v in value.items():

        v = float(v)

        if _np.isfinite(v) and (0.0 <= v <= 1.0):
            result[k] = v
        else:
            raise ValueError('The "@arg@" parameter values can contain only finite real numbers between 0.0 and 1.0.')

    return result


def _validate_graph(value: _ct.tany) -> _ct.tgraphs:

    result, _ = _u.is_graph(value)

    if not result:
        raise ValueError('The "@arg@" parameter must be a directed graph.')

    value = value.copy()

    nodes = list(value.nodes)

    if not all(_u.is_string(node) for node in nodes):  # pragma: no cover
        raise ValueError('The "@arg@" parameter must define node labels as non-empty strings.')

    edge_weights = list(value.edges(data='weight', default=0.0))

    if not all(_u.is_number(edge_weight[2]) and _np.isfinite(edge_weight[2]) and (edge_weight[2] > 0.0) for edge_weight in edge_weights):  # pragma: no cover
        raise ValueError('The "@arg@" parameter must define edge weight attributes as strictly positive finite numbers.')

    return value


def _validate_labels(value: _ct.tany, labels: _ct.tlist_str) -> _ct.tlist_int:

    if all(_u.is_integer(label) for label in value):
        value_type = 'integer'
    elif all(_u.is_string(label) for label in value):
        value_type = 'string'
    else:
        raise TypeError('The "@arg@" parameter contains invalid elements.')

    labels_length = len(labels)

    if value_type == 'integer':

        value = [int(label) for label in value]

        if any((label < 0) or (label >= labels_length) for label in value):
            raise ValueError(f'The "@arg@" parameter, when specified as a list of integers, must contain only values between 0 and {labels_length - 1:d}.')

    else:

        value = [labels.index(label) if label in labels else -1 for label in value]

        if any((label == -1) for label in value):
            raise ValueError(f'The "@arg@" parameter, when specified as a list of strings, must contain only values matching the following strings: {", ".join(labels)}.')

    return value


def _validate_limits(value: _ct.tnum, value_type: str, lower_limit: _ct.olim_num, upper_limit: _ct.olim_num) -> None:

    def _get_limit_text(glt_value_type, glt_limit_value):

        text = f'{glt_limit_value:d}' if glt_value_type == 'integer' else f'{glt_limit_value:f}'

        return text

    if lower_limit is not None:

        lower_limit_value, lower_limit_excluded = lower_limit

        if lower_limit_excluded:
            if value <= lower_limit_value:
                raise ValueError(f'The "@arg@" parameter must be greater than {_get_limit_text(value_type, lower_limit_value)}.')
        else:
            if value < lower_limit_value:
                raise ValueError(f'The "@arg@" parameter must be greater than or equal to {_get_limit_text(value_type, lower_limit_value)}.')

    if upper_limit is not None:

        upper_limit_value, upper_limit_excluded = upper_limit

        if upper_limit_excluded:
            if value >= upper_limit_value:
                raise ValueError(f'The "@arg@" parameter must be less than {_get_limit_text(value_type, upper_limit_value)}.')
        else:
            if value > upper_limit_value:
                raise ValueError(f'The "@arg@" parameter must be less than or equal to {_get_limit_text(value_type, upper_limit_value)}.')


def _validate_models(value: _ct.tany, validator: _ct.trfunc) -> _ct.tany:

    if not _u.is_list(value):
        raise ValueError('The "@arg@" parameter must be a list.')

    value_length = len(value)

    if value_length < 2:
        raise ValueError('The "@arg@" parameter must contain at least 2 elements.')

    for i in range(value_length):
        try:
            validator(value[i])
        except Exception as ex:
            raise ValueError('The "@arg@" parameter contains invalid elements.') from ex

    value = value.copy()

    return value


def _validate_probabilities(value: _ct.tarray, stochastic: bool) -> None:

    if not _np.all(_np.isfinite(value) & (value >= 0.0) & (value <= 1.0)):
        raise ValueError('The "@arg@" parameter must contain only finite real values between 0.0 and 1.0.')

    if not stochastic:
        return

    if value.ndim == 1:
        if not _np.isclose(_np.sum(value), 1.0, rtol=0.0, atol=_c.ETOL):
            raise ValueError('The "@arg@" parameter values must sum to 1.0.')
    else:
        if not _np.allclose(_np.sum(value, axis=1), 1.0, rtol=0.0, atol=_c.ETOL):
            raise ValueError('The "@arg@" parameter rows must sum to 1.0.')


def _validate_vector(value: _ct.tany) -> _ct.tarray:

    is_array = isinstance(value, _np.ndarray)

    try:
        value = _u.extract_numeric(value, float)
    except Exception as ex:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from ex

    if (value.size == 0) or ((value.ndim != 1) and not (is_array and (value.ndim == 2) and (value.shape[1] == 1))):
        raise ValueError('The "@arg@" parameter must be a valid vector.')

    value = _np.ravel(value)

    return value


def validate_boolean(value: _ct.tany) -> bool:

    if not _u.is_boolean(value):
        raise TypeError('The "@arg@" parameter must be a boolean value.')

    return value


def validate_boundary_condition(value: _ct.tany) -> _ct.tbcond:

    if _u.is_number(value):

        value = float(value)

        if (value < 0.0) or (value > 1.0):
            raise ValueError('The "@arg@" parameter, when specified as a number, must have a value between 0 and 1.')

        return value

    if _u.is_string(value):

        possible_values = ['absorbing', 'reflecting']

        if value not in possible_values:
            raise ValueError(f'The "@arg@" parameter, when specified as a string, must have one of the following values: {", ".join(possible_values)}.')

        return value

    raise TypeError('The "@arg@" parameter must be either a float representing the first probability of the semi-reflecting condition or a non-empty string representing the boundary condition type.')


def validate_dictionary_hmm(value: _ct.tany) -> _ct.thmm_dict:

    _validate_dictionary_keys(value, 3)

    attributes = ['P', 'E']
    labels_by_attribute = {attribute: set() for attribute in attributes}

    for k in value.keys():

        key_attribute = k[0]

        if key_attribute not in attributes:
            raise ValueError(f'The "@arg@" parameter keys must be defined as tuples of non-empty strings whose first item matches one of the following values: {", ".join(attributes)}.')

        labels_by_attribute['P'].add(k[1])
        labels_by_attribute[key_attribute].add(k[2])

    labels_first = labels_by_attribute['P']

    if len(labels_by_attribute['E']) == 0:
        raise ValueError('The "@arg@" parameter keys must define both transition and emission probabilities.')

    if len(list(labels_by_attribute['P'] & labels_by_attribute['E'])) > 0:
        raise ValueError('The "@arg@" parameter must define different hidden-state and emitted-symbol names.')

    for attribute, labels in labels_by_attribute.items():
        if not all((attribute,) + combination in value for combination in _it.product(labels_first, labels)):
            raise ValueError(f'The "@arg@" parameter keys must contain all the possible combinations of labels P and {attribute}.')

    result = _validate_dictionary_values(value)

    return result


def validate_dictionary_mc(value: _ct.tany) -> _ct.tmc_dict:

    _validate_dictionary_keys(value, 2)

    labels = set()

    for key in value.keys():
        labels.add(key[0])
        labels.add(key[1])

    if not all(combination in value for combination in _it.product(labels, repeat=2)):
        raise ValueError('The "@arg@" parameter keys must contain all the possible combinations of labels.')

    result = _validate_dictionary_values(value)

    return result


def validate_dimensions(value: _ct.tany) -> _ct.tlist_int:

    if not _u.is_list(value):
        raise ValueError('The "@arg@" parameter must be a list.')

    if len(value) < 2:
        raise ValueError('The "@arg@" parameter must contain at least 2 elements.')

    if not all(_u.is_integer(v) for v in value):
        raise TypeError('The "@arg@" parameter must contain only integers.')

    value = [int(v) for v in value]

    if any(v < 2 for v in value):
        raise ValueError('The "@arg@" parameter values must be greater than or equal to 2.')

    return value


def validate_dpi(value: _ct.tany) -> int:

    if not _u.is_integer(value):
        raise TypeError('The "@arg@" parameter must be an integer.')

    value = int(value)

    possible_values = (75, 100, 150, 200, 250, 300, 600)

    if value not in possible_values:
        possible_values = [str(possible_value) for possible_value in possible_values]
        raise ValueError(f'The "@arg@" parameter must have one of the following values: {", ".join(possible_values)}.')

    return value


def validate_emission_matrix(value: _ct.tany, rows: int, columns: _ct.oint = None) -> _ct.tarray:

    try:
        value = _u.extract_numeric(value, float)
    except Exception as ex:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from ex

    if columns is None:

        if (value.ndim != 2) or (value.shape[0] != rows) or (value.shape[1] < 2):
            raise ValueError(f'The "@arg@" parameter must be a 2d matrix with {rows:d} rows and at least 2 columns.')

    else:

        if (value.ndim != 2) or (value.shape[0] != rows) or (value.shape[1] != columns):
            raise ValueError(f'The "@arg@" parameter must be a 2d matrix with {rows:d} rows and {columns:d} columns.')

    _validate_probabilities(value, True)

    return value


def validate_enumerator(value: _ct.tany, possible_values: _ct.tlist_str) -> str:

    if not _u.is_string(value):
        raise TypeError('The "@arg@" parameter must be a non-empty string.')

    if value not in possible_values:
        raise ValueError(f'The "@arg@" parameter value must be one of the following: {", ".join(possible_values)}.')

    return value


def validate_file_path(value: _ct.tany, accepted_extensions: _ct.olist_str, write_permission: bool) -> _ct.tfile:

    if isinstance(value, _pl.Path):
        file_path = value
    elif _u.is_string(value):
        file_path = _pl.Path(value)
    else:
        raise TypeError('The "@arg@" parameter must be a non-empty string or a path object.')

    try:
        file_extension = _u.get_file_extension(file_path)
    except Exception as ex:  # pragma: no cover
        raise ValueError('The "@arg@" parameter defines an invalid file path.') from ex

    if (accepted_extensions is not None) and (len(accepted_extensions) > 0) and (file_extension not in accepted_extensions):
        raise ValueError(f'The "@arg@" parameter must have one of the following extensions: {", ".join(sorted(accepted_extensions)).replace(".", "")}.')

    if write_permission:

        if (not _osp.isdir(_osp.dirname(file_path))) and _osp.isabs(file_path):  # pragma: no cover
            raise ValueError('The "@arg@" parameter defines a non-existent parent directory.')

        try:

            with open(file_path, mode='a'):
                pass

        except Exception as ex:  # pragma: no cover
            raise ValueError('The "@arg@" parameter defines the path to an inaccessible file.') from ex

    else:

        if not _osp.isfile(file_path):
            raise ValueError('The "@arg@" parameter defines an invalid file path.')

        file_empty = False

        try:

            with open(file_path, mode='r') as file:

                file.seek(0)

                if not file.read(1):
                    file_empty = True

        except Exception as ex:  # pragma: no cover
            raise ValueError('The "@arg@" parameter defines the path to an inaccessible file.') from ex

        if file_empty:
            raise ValueError('The "@arg@" parameter defines the path to an empty file.')

    return file_path, file_extension


def validate_float(value: _ct.tany, lower_limit: _ct.olim_float = None, upper_limit: _ct.olim_float = None) -> float:

    if not _u.is_float(value):
        raise TypeError('The "@arg@" parameter must be a float.')

    value = float(value)

    if not _np.isfinite(value):
        raise ValueError('The "@arg@" parameter be a finite real value.')

    _validate_limits(value, 'float', lower_limit, upper_limit)

    return value


def validate_graph_hmm(value: _ct.tany) -> _ct.tgraphs:

    value = _validate_graph(value)

    layers = (0, 1)
    nodes = list(value.nodes(data='layer', default=-1))
    nodes_by_layer = {layer: [] for layer in layers}

    for node in nodes:

        node_layer = node[1]

        if not _u.is_integer(node_layer) or node_layer not in layers:  # pragma: no cover
            raise ValueError('The "@arg@" parameter must define node layer attributes as integers equal to 0 or 1.')

        nodes_by_layer[node_layer].append(node[0])

    if any(len(nodes) < 2 for nodes in nodes_by_layer.values()):
        raise ValueError('The "@arg@" parameter must define at least 2 nodes for each layer.')

    edges = list(value.edges(data='type', default=''))

    for edge in edges:

        edge_type = edge[2]

        if (not _u.is_string(edge_type)) or (edge_type not in ['E', 'P']):
            raise ValueError('The "@arg@" parameter must define edge type attributes as strings matching one of the following values: E, P.')

        origin_layer = value.nodes[edge[0]]['layer']
        target_layer = value.nodes[edge[1]]['layer']

        if (origin_layer != 1) or ((edge_type == 'P') and (target_layer != 1)) or ((edge_type == 'E') and (target_layer != 0)):
            raise ValueError('The "@arg@" parameter contains edges that are inconsistent with the node layers.')

    return value


def validate_graph_mc(value: _ct.tany) -> _ct.tgraphs:

    value = _validate_graph(value)

    if len(value.nodes) < 2:
        raise ValueError('The "@arg@" parameter must contain at least 2 nodes.')

    return value


def validate_hidden_markov_model(value: _ct.tany) -> _ct.thmm:

    if (value is None) or (f'{value.__module__}.{value.__class__.__name__}' != 'pydtmc.hidden_markov_model.HiddenMarkovModel'):
        raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    return value


def validate_hidden_markov_models(value: _ct.tany) -> _ct.tlist_hmm:

    value = _validate_models(value, validate_hidden_markov_model)

    return value


def validate_hyperparameter(value: _ct.tany, size: int) -> _ct.tarray:

    try:
        value = _u.extract_numeric(value, float)
    except Exception as ex:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from ex

    if (value.ndim != 2) or (value.shape[0] != size) or (value.shape[1] != size):
        raise ValueError(f'The "@arg@" parameter must be a 2d square matrix with size equal to {size:d}.')

    if not _np.all(_np.isfinite(value) & (value >= 1.0) & (value == _np.floor(value))):
        raise ValueError('The "@arg@" parameter must contain only integers greater than or equal to 1.')

    return value


def validate_inertial_weights(value: _ct.tany, size: int) -> _ct.tarray:

    if _u.is_number(value):
        value = _np.repeat(float(value), size)
    else:
        value = _validate_vector(value)

    if value.size != size:
        raise ValueError(f'The "@arg@" parameter length must be equal to {size:d}.')

    _validate_probabilities(value, False)

    return value


def validate_integer(value: _ct.tany, lower_limit: _ct.olim_int = None, upper_limit: _ct.olim_int = None) -> int:

    if not _u.is_integer(value):
        raise TypeError('The "@arg@" parameter must be an integer.')

    value = int(value)

    _validate_limits(value, 'integer', lower_limit, upper_limit)

    return value


def validate_interval(value: _ct.tany) -> _ct.tival:

    if not _u.is_tuple(value):
        raise TypeError('The "@arg@" parameter must be a tuple.')

    if len(value) != 2:
        raise ValueError('The "@arg@" parameter must contain 2 elements.')

    a, b = value

    if (not _u.is_number(a)) or (not _u.is_number(b)):
        raise ValueError('The "@arg@" parameter must contain only floats and integers.')

    a, b = float(a), float(b)

    if not (_np.isfinite(a) and _np.isfinite(b) and (a >= 0.0) and (b >= 0.0)):
        raise ValueError('The "@arg@" parameter must contain only finite real values greater than or equal to 0.0.')

    if a >= b:
        raise ValueError('The "@arg@" parameter must contain two distinct values, and the first value must be less than the second one.')

    return a, b


def validate_label(value: _ct.tany, labels: _ct.tlist_str) -> int:

    if _u.is_integer(value):

        label = int(value)
        limit = len(labels) - 1

        if label < 0 or label > limit:
            raise ValueError(f'The "@arg@" parameter, when specified as an integer, must have a value between 0 and {limit:d}.')

        return label

    if _u.is_string(value):

        if value not in labels:
            raise ValueError(f'The "@arg@" parameter, when specified as a string, must match one of the following strings: {", ".join(labels)}.')

        label = labels.index(value)

        return label

    raise TypeError('The "@arg@" parameter must be either an integer or a non-empty string.')


def validate_labels_current(value: _ct.tany, labels: _ct.tlist_str, subset: bool, minimum_length: _ct.oint = None) -> _ct.tlist_int:

    if _u.is_integer(value) or _u.is_string(value):

        value = validate_label(value, labels)

        if (minimum_length is not None) and (minimum_length > 1):  # pragma: no cover
            raise ValueError(f'The "@arg@" parameter must contain at least {minimum_length:d} elements.')

        value = [value]

        return value

    if _u.is_list(value):

        value = _validate_labels(value, labels)
        value_length = len(value)
        labels_length = len(labels)

        if len(set(value)) < value_length:
            raise ValueError('The "@arg@" parameter must contain only unique values.')

        if value_length == 0:
            raise ValueError('The "@arg@" parameter must contain at least an element.')

        maximum_length = labels_length - 1 if subset else labels_length

        if (minimum_length is None) or (minimum_length == 1):
            if value_length > maximum_length:
                raise ValueError(f'The "@arg@" parameter must contain no more than {maximum_length:d} elements.')
        else:

            if (value_length < minimum_length) or (value_length > maximum_length):

                if minimum_length == maximum_length:
                    length = {minimum_length, maximum_length}.pop()
                    raise ValueError(f'The "@arg@" parameter must contain a number of elements equal to {length:d}.')

                raise ValueError(f'The "@arg@" parameter must contain a number of elements between {minimum_length:d} and {maximum_length:d}.')  # pragma: no cover

        value = sorted(value)

        return value

    raise TypeError('The "@arg@" parameter must be either an integer, a non-empty string, a list of integers or a list of non-empty strings.')


def validate_labels_pair(value: _ct.tany, labels: _ct.tlist_str) -> _ct.tlist_int:

    if not _u.is_list(value):
        raise ValueError('The "@arg@" parameter must be a list.')

    if len(value) != 2:
        raise ValueError('The "@arg@" parameter must contain 2 elements.')

    value = _validate_labels(value, labels)

    if value[0] == value[1]:
        raise ValueError('The "@arg@" parameter must contain 2 distinct elements.')

    return value


def validate_labels_input(value: _ct.tany, size: _ct.oint = None) -> _ct.tlist_str:

    if not _u.is_list(value):
        raise ValueError('The "@arg@" parameter must be a list.')

    if not all(_u.is_string(label) for label in value):
        raise TypeError('The "@arg@" parameter must contain only non-empty strings.')

    labels_length = len(value)

    if labels_length < 2:
        raise ValueError('The "@arg@" parameter must contain at least 2 elements.')

    labels_unique_length = len(set(value))

    if labels_unique_length < labels_length:
        raise ValueError('The "@arg@" parameter must contain only unique values.')

    if size is not None and labels_length != size:
        raise ValueError(f'The "@arg@" parameter must contain a number of elements equal to {size:d}.')

    value = value.copy()

    return value


def validate_markov_chain(value: _ct.tany, size: _ct.oint = None) -> _ct.tmc:

    if (value is None) or (f'{value.__module__}.{value.__class__.__name__}' != 'pydtmc.markov_chain.MarkovChain'):
        raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    if (size is not None) and (value.size != size):
        raise ValueError(f'The "@arg@" parameter must be a Markov chain with size equal to {size:d}.')

    return value


def validate_markov_chains(value: _ct.tany) -> _ct.tlist_mc:

    value = _validate_models(value, validate_markov_chain)

    return value


def validate_mask(value: _ct.tany, rows: int, columns: int) -> _ct.tarray:

    try:
        value = _u.extract_numeric(value, float)
    except Exception as ex:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from ex

    if (value.ndim != 2) or (value.shape[0] != rows) or (value.shape[1] != columns):

        if rows == columns:
            size = {rows, columns}.pop()
            raise ValueError(f'The "@arg@" parameter must be a 2d square matrix with size equal to {size:d}.')

        raise ValueError(f'The "@arg@" parameter must be a 2d matrix with {rows:d} rows and {columns:d} columns.')

    if not _np.all(_np.isnan(value) | (_np.isfinite(value) & (value >= 0.0) & (value <= 1.0))):
        raise ValueError('The "@arg@" parameter can contain only NaNs and finite real values between 0.0 and 1.0.')

    if not _np.all(_np.nansum(value, axis=1, dtype=float) <= 1.0):
        raise ValueError('The "@arg@" parameter row sums must not exceed 1.')

    return value


def validate_matrix(value: _ct.tany, rows: _ct.oint = None, columns: _ct.oint = None) -> _ct.tarray:

    try:
        value = _u.extract_numeric(value, float)
    except Exception as ex:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from ex

    if (columns is None) and (rows is None):
        if (value.ndim != 2) or (value.shape[0] < 2) or (value.shape[0] != value.shape[1]):
            raise ValueError('The "@arg@" parameter must be a 2d square matrix with size greater than or equal to 2.')
    elif columns is not None and rows is None:
        if (value.ndim != 2) or (value.shape[0] < 2) or (value.shape[1] != columns):
            raise ValueError(f'The "@arg@" parameter must be a 2d matrix with at least 2 rows and {columns:d} columns.')
    elif columns is None and rows is not None:
        if (value.ndim != 2) or (value.shape[0] != rows) or (value.shape[1] < 2):
            raise ValueError(f'The "@arg@" parameter must be a 2d matrix with {rows:d} rows and at least 2 columns.')
    else:
        if (value.ndim != 2) or (value.shape[0] != rows) or (value.shape[1] != columns):
            raise ValueError(f'The "@arg@" parameter must be a 2d matrix with {rows:d} rows and {columns:d} columns.')

    if not _np.all(_np.isfinite(value) & (value >= 0.0)):
        raise ValueError('The "@arg@" parameter must contain only finite real values greater than or equal to 0.0.')

    return value


# noinspection PyBroadException
def validate_model(value: _ct.tany) -> _ct.tmdl:

    if value is None:
        raise TypeError('The "@arg@" parameter is null.')

    try:
        value_module = value.__module__
    except Exception:
        value_module = value.__class__.__module__

    if not value_module.startswith('pydtmc.'):
        raise TypeError('The "@arg@" parameter is wrongly typed.')

    value_bases = value.__class__.__bases__ or ()
    value_base = None if len(value_bases) == 0 else _u.get_full_name(value_bases[0])

    if (value_base is None) or (value_base != 'pydtmc._base_classes.Model'):
        raise TypeError('The "@arg@" parameter is wrongly typed.')

    return value


def validate_models(value: _ct.tany) -> _ct.tlist_mdl:

    value = _validate_models(value, validate_model)

    return value


def validate_none(value: _ct.tany) -> _ct.tany:

    if value is not None:
        raise TypeError('The "@arg@" parameter must be None.')

    return value


def validate_observable(value: _ct.tany, size: int) -> _ct.tobs:

    if _u.is_string(value):

        possible_values = ['entropy_production', 'state_occupation', 'transition_count', 'transition_current']
        value = validate_enumerator(value, possible_values)

    else:

        try:
            value = _u.extract_numeric(value, float)
        except Exception as ex:
            raise TypeError('The "@arg@" parameter is null or wrongly typed.') from ex

        if (value.ndim != 2) or (value.shape[0] != size) or (value.shape[1] != size):
            raise ValueError(f'The "@arg@" parameter must be a 2d square matrix with size equal to {size:d}.')

        if not _np.all(_np.isfinite(value)):
            raise ValueError('The "@arg@" parameter must contain only finite real values.')

    return value


def validate_partition(value: _ct.tany, labels: _ct.tlist_str) -> _ct.tlists_int:

    if not _u.is_list(value):
        raise ValueError('The "@arg@" parameter must be a list.')

    partitions_length = len(value)
    labels_length = len(labels)

    if (partitions_length < 2) or (partitions_length >= labels_length):

        if labels_length == 2:  # pragma: no cover
            raise ValueError('The "@arg@" parameter must contain a number of elements equal to 2.')

        raise ValueError(f'The "@arg@" parameter must contain a number of elements between 2 and {labels_length - 1:d}.')

    partitions_flat = []
    partitions_groups = []

    for partition in value:

        if not _u.is_list(partition):
            raise TypeError('The "@arg@" parameter must contain only lists.')

        partition_list = list(partition)

        partitions_flat.extend(partition_list)
        partitions_groups.append(len(partition_list))

    if all(_u.is_integer(state) for state in partitions_flat):

        partitions_flat = [int(state) for state in partitions_flat]

        if any(label < 0 or label >= labels_length for label in partitions_flat):
            raise ValueError(f'The "@arg@" parameter subelements, when specified as integers, must be values between 0 and {labels_length - 1:d}.')

    elif all(_u.is_string(partition_flat) for partition_flat in partitions_flat):

        partitions_flat = [labels.index(state) if state in labels else -1 for state in partitions_flat]

        if any(label == -1 for label in partitions_flat):
            raise ValueError(f'The "@arg@" parameter subelements, when specified as strings, must be only values matching the following strings: {", ".join(labels)}.')

    else:
        raise TypeError('The "@arg@" parameter must contain only lists of integers or lists of non-empty strings.')

    partitions_flat_length = len(partitions_flat)

    if len(set(partitions_flat)) < partitions_flat_length or partitions_flat_length != labels_length or partitions_flat != list(range(labels_length)):
        raise ValueError('The "@arg@" parameter subelements must be unique, include all the existing labels and follow a sequential order.')

    result = []
    offset = 0

    for partitions_group in partitions_groups:
        extension = offset + partitions_group
        result.append(partitions_flat[offset:extension])
        offset += partitions_group

    return result


def validate_pattern_probabilities(value: _ct.tany) -> _ct.tpprobs:

    if not _u.is_dictionary(value) or len(value) == 0:
        raise ValueError('The "@arg@" parameter must be a non-empty dictionary.')

    if not all(_u.is_string(s) and (len(s) == 1) for s in value):
        raise ValueError('The "@arg@" parameter keys must be single-character strings.')

    if not all(_u.is_number(v) and _np.isfinite(v) and (0.0 <= v <= 1.0) for v in value.values()):
        raise ValueError('The "@arg@" parameter values must be finite real numbers between 0.0 and 1.0.')

    if not _np.isclose(sum(value.values()), 1.0, rtol=0.0, atol=_c.ETOL):
        raise ValueError('The "@arg@" parameter values must sum to 1.0.')

    value = {k: float(v) for k, v in value.items()}

    return value


def validate_random_distribution(value: _ct.tany, rng: _ct.trand, accepted_values: _ct.tlist_str) -> _ct.trfunc:

    if value is None:
        raise TypeError('The "@arg@" parameter is null.')

    if callable(value):  # pragma: no cover

        if 'of numpy.random' not in repr(value):
            raise ValueError('The "@arg@" parameter, when specified as a callable function, must be defined in the "numpy.random" module.')

        value = value.__name__

        if value not in dir(rng):
            raise ValueError('The "@arg@" parameter, when specified as a callable function, must reference a valid "numpy.random" module object.')

        if len(accepted_values) > 0 and value not in accepted_values:
            raise ValueError(f'The "@arg@" parameter, when specified as a callable function, must reference one of the following "numpy.random" module objects: {", ".join(accepted_values)}.')

        value = getattr(rng, value)

        return value

    if _u.is_string(value):

        if value is None or value not in dir(rng):
            raise ValueError('The "@arg@" parameter, when specified as a string, must reference a valid "numpy.random" module object.')

        if len(accepted_values) > 0 and value not in accepted_values:
            raise ValueError(f'The "@arg@" parameter, when specified as a string, must reference one of the following "numpy.random" module objects: {", ".join(accepted_values)}.')

        value = getattr(rng, value)

        return value

    raise TypeError('The "@arg@" parameter must be either a callable function or the name of a callable function.')


def validate_rewards(value: _ct.tany, size: int) -> _ct.tarray:

    value = _validate_vector(value)

    if value.size != size:
        raise ValueError(f'The "@arg@" parameter length must be equal to {size:d}.')

    if not _np.all(_np.isfinite(value)):
        raise ValueError('The "@arg@" parameter must contain only finite real values.')

    return value


def validate_sequence(value: _ct.tany, labels: _ct.tlist_str) -> _ct.tlist_int:

    if not _u.is_list(value):
        raise ValueError('The "@arg@" parameter must be a list.')

    if len(value) < 2:
        raise ValueError('The "@arg@" parameter must contain at least 2 elements.')

    if all(_u.is_integer(label) for label in value):
        value_type = 'integer'
    elif all(_u.is_string(label) for label in value):
        value_type = 'string'
    else:
        raise TypeError('The "@arg@" parameter must be either a list of integers or a list of non-empty strings.')

    if value_type == 'integer':

        value = [int(label) for label in value]
        labels_length = len(labels)

        if any((label < 0) or (label >= labels_length) for label in value):
            raise ValueError(f'The "@arg@" parameter, when specified as a list of integers, must contain only values between 0 and {labels_length - 1:d}.')

    else:

        value = [labels.index(label) if label in labels else -1 for label in value]

        if any(label == -1 for label in value):
            raise ValueError(f'The "@arg@" parameter, when specified as a list of strings, must contain only values matching the following strings: {", ".join(labels)}.')

    return value


def validate_sequences(value: _ct.tany, labels: _ct.tlist_str, flex: bool) -> _ct.tlists_int:

    if not _u.is_list(value):
        raise ValueError('The "@arg@" parameter must be a list.')

    if len(value) < 2:
        raise ValueError('The "@arg@" parameter must contain at least 2 elements.')

    if all(_u.is_list(state) for state in value):
        value_type = 'list'
    elif flex and all(_u.is_integer(label) for label in value):
        value_type = 'integer'
    elif flex and all(_u.is_string(label) for label in value):
        value_type = 'string'
    else:

        if flex:
            raise TypeError('The "@arg@" parameter must be either a list of integers, a list of non-empty strings, a list of lists of integers or a list of lists of non-empty strings.')

        raise TypeError('The "@arg@" parameter must be either a list of lists of integers or a list of lists of non-empty strings.')

    if value_type in ['integer', 'string']:
        value = [value]

    value_base = value
    value = []

    for value_current in value_base:
        try:
            value.append(validate_sequence(value_current, labels))
        except Exception as ex:
            raise ValueError('The "@arg@" parameter contains invalid elements.') from ex

    return value


def validate_status(value: _ct.tany, labels: _ct.tlist_str, emit_uniform: bool) -> _ct.tarray:

    size = len(labels)

    if (value is None) and emit_uniform:
        return _np.full(size, 1.0 / size, dtype=float)

    if _u.is_integer(value) or _u.is_string(value):

        value = validate_label(value, labels)

        result = _np.zeros(size, dtype=float)
        result[value] = 1.0

        return result

    value = validate_vector_stochastic(value, size, False)

    return value


def validate_string(value: _ct.tany, excluded_chars: _ct.olist_str = None) -> str:

    if not _u.is_string(value):
        raise TypeError('The "@arg@" parameter must be a non-empty string.')

    if (excluded_chars is not None) and any(c in value for c in excluded_chars):
        raise ValueError(f'The "@arg@" parameter must not contain any of the following characters: {", ".join(excluded_chars)}.')

    return value


def validate_strings(value: _ct.tany, excluded_chars: _ct.olist_str = None, size: _ct.oint = None) -> _ct.tlist_str:

    if not _u.is_list(value):
        raise ValueError('The "@arg@" parameter must be a list.')

    if not all(_u.is_string(s) for s in value):
        raise TypeError('The "@arg@" parameter must contain only non-empty strings.')

    if (excluded_chars is not None) and any(c in s for s in value for c in excluded_chars):
        raise ValueError(f'The "@arg@" parameter must not contain strings with any of the following characters: {", ".join(excluded_chars)}.')

    value_length = len(value)

    if value_length == 0:
        raise ValueError('The "@arg@" parameter must contain at least an element.')

    if (size is not None) and (value_length != size):
        raise ValueError(f'The "@arg@" parameter must contain a number of elements equal to {size:d}.')

    value = value.copy()

    return value


def validate_time_points(value: _ct.tany) -> _ct.ttimes_in:

    if _u.is_integer(value):

        value = int(value)

        if value < 0:
            raise ValueError('The "@arg@" parameter, when specified as an integer, must be greater than or equal to 0.')

        return value

    if _u.is_list(value):

        if not all(_u.is_integer(time_point) for time_point in value):
            raise TypeError('The "@arg@" parameter must be either an integer or an array_like object of integers.')

        value = [int(time_point) for time_point in value]

        if any(time_point < 0 for time_point in value):
            raise ValueError('The "@arg@" parameter, when specified as a list of integers, must contain only values greater than or equal to 0.')

        time_points_length = len(value)

        if time_points_length < 1:
            raise ValueError('The "@arg@" parameter must contain at least an element.')

        time_points_unique = len(set(value))

        if time_points_unique < time_points_length:
            raise ValueError('The "@arg@" parameter must contain only unique values.')

        value = sorted(value)

        return value

    raise TypeError('The "@arg@" parameter must be either an integer or a list of integers.')


# noinspection PyBroadException
def validate_transition_function(value: _ct.tany) -> _ct.ttfunc:

    if value is None or _ins.isclass(value) or not callable(value):
        raise TypeError('The "@arg@" parameter must be a callable function or method.')

    sig = _ins.signature(value)

    if len(sig.parameters) != 4:
        raise ValueError('The "@arg@" parameter must accept 4 input arguments.')

    valid_parameters = ('x_index', 'x_value', 'y_index', 'y_value')

    if not all(parameter in valid_parameters for parameter in sig.parameters.keys()):
        raise ValueError(f'The "@arg@" parameter must define the following input arguments: {", ".join(valid_parameters)}.')

    try:
        result = value(1, 1.0, 1, 1.0)
    except Exception as ex:  # pragma: no cover
        raise ValueError('The "@arg@" parameter behavior is not compliant.') from ex

    if (not _u.is_number(result)) or (not _np.isfinite(result)):
        raise ValueError('The "@arg@" parameter behavior is not compliant.')

    return value


def validate_transition_matrix(value: _ct.tany, size: _ct.oint = None) -> _ct.tarray:

    try:
        value = _u.extract_numeric(value, float)
    except Exception as ex:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from ex

    if (value.ndim != 2) or (value.shape[0] != value.shape[1]) or (value.shape[0] < 2):
        raise ValueError('The "@arg@" parameter must be a 2d square matrix with size greater than or equal to 2.')

    if (size is not None) and (value.shape[0] != size):
        raise ValueError(f'The "@arg@" parameter must have a size equal to {size:d}.')

    _validate_probabilities(value, True)

    return value


def validate_vector_birth_death(value: _ct.tany, vector_type: str) -> _ct.tarray:

    value = _validate_vector(value)

    _validate_probabilities(value, False)

    if vector_type == 'annihilation':

        if not _np.isclose(value[0], 0.0, rtol=0.0, atol=_c.ETOL):
            raise ValueError('The "@arg@" parameter must contain a value equal to 0 in the first index.')

    else:

        if not _np.isclose(value[-1], 0.0, rtol=0.0, atol=_c.ETOL):
            raise ValueError('The "@arg@" parameter must contain a value equal to 0 in the last index.')

    return value


def validate_vector_stochastic(value: _ct.tany, size: int, emit_uniform: bool) -> _ct.tarray:

    if (value is None) and emit_uniform:
        return _np.full(size, 1.0 / size, dtype=float)

    value = _validate_vector(value)

    if value.size != size:
        raise ValueError(f'The "@arg@" parameter length must be equal to {size:d}.')

    _validate_probabilities(value, True)

    return value
