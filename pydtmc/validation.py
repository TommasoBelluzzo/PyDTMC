# -*- coding: utf-8 -*-

__all__ = [
    'validate_boolean',
    'validate_boundary_condition',
    'validate_dictionary',
    'validate_dpi',
    'validate_emission_matrix',
    'validate_enumerator',
    'validate_file_path',
    'validate_float',
    'validate_graph',
    'validate_hidden_markov_model',
    'validate_hidden_markov_models',
    'validate_hyperparameter',
    'validate_integer',
    'validate_interval',
    'validate_label',
    'validate_labels_current',
    'validate_labels_input',
    'validate_markov_chain',
    'validate_markov_chains',
    'validate_mask',
    'validate_matrix',
    'validate_model',
    'validate_models',
    'validate_partitions',
    'validate_random_distribution',
    'validate_rewards',
    'validate_sequence',
    'validate_sequences',
    'validate_status',
    'validate_strings',
    'validate_time_points',
    'validate_transition_function',
    'validate_transition_matrix',
    'validate_vector'
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

import networkx as _nx
import numpy as _np

# Internal

from .custom_types import (
    oint as _oint,
    olimit_float as _olimit_float,
    olimit_int as _olimit_int,
    olimit_scalar as _olimit_scalar,
    tlist_model as _tlist_model,
    olist_str as _olist_str,
    tany as _tany,
    tarray as _tarray,
    tbcond as _tbcond,
    oedge_attributes as _oedge_attributes,
    tfile as _tfile,
    tgraphs as _tgraphs,
    thmm as _thmm,
    tinterval as _tinterval,
    tlist_int as _tlist_int,
    tlist_str as _tlist_str,
    tlists_int as _tlists_int,
    tmc as _tmc,
    tmc_dict as _tmc_dict,
    tmodel as _tmodel,
    trand as _trand,
    trandfunc as _trandfunc,
    tscalar as _tscalar,
    ttfunc as _ttfunc,
    ttimes_in as _ttimes_in
)

from .utilities import (
    extract_numeric as _extract_numeric,
    get_file_extension as _get_file_extension,
    get_full_name as _get_full_name,
    is_boolean as _is_boolean,
    is_dictionary as _is_dictionary,
    is_float as _is_float,
    is_graph as _is_graph,
    is_integer as _is_integer,
    is_list as _is_list,
    is_number as _is_number,
    is_string as _is_string,
    is_tuple as _is_tuple
)


#############
# FUNCTIONS #
#############

def _validate_limits(value: _tscalar, value_type: str, lower_limit: _olimit_scalar, upper_limit: _olimit_scalar):

    def _get_limit_text(glt_value_type, glt_limit_value):

        text = f'{glt_limit_value:d}' if glt_value_type == 'integer' else f'{glt_limit_value:f}'

        return text

    if lower_limit is not None:

        lower_limit_value, lower_limit_included = lower_limit

        if lower_limit_included:
            if value <= lower_limit_value:
                raise ValueError(f'The "@arg@" parameter must be greater than {_get_limit_text(value_type, lower_limit_value)}.')
        else:
            if value < lower_limit_value:
                raise ValueError(f'The "@arg@" parameter must be greater than or equal to {_get_limit_text(value_type, lower_limit_value)}.')

    if upper_limit is not None:

        upper_limit_value, upper_limit_included = upper_limit

        if upper_limit_included:
            if value >= upper_limit_value:
                raise ValueError(f'The "@arg@" parameter must be less than {_get_limit_text(value_type, upper_limit_value)}.')
        else:
            if value > upper_limit_value:
                raise ValueError(f'The "@arg@" parameter must be less than or equal to {_get_limit_text(value_type, upper_limit_value)}.')


def validate_boolean(value: _tany) -> bool:

    if not _is_boolean(value):
        raise TypeError('The "@arg@" parameter must be a boolean value.')

    return value


def validate_boundary_condition(value: _tany) -> _tbcond:

    if _is_number(value):

        value = float(value)

        if (value < 0.0) or (value > 1.0):
            raise ValueError('The "@arg@" parameter, when specified as a number, must have a value between 0 and 1.')

        return value

    if _is_string(value):

        possible_values = ('absorbing', 'reflecting')

        if value not in possible_values:
            raise ValueError(f'The "@arg@" parameter, when specified as a string, must have one of the following values: {", ".join(possible_values)}.')

        return value

    raise TypeError('The "@arg@" parameter must be either a float representing the first probability of the semi-reflecting condition or a non-empty string representing the boundary condition type.')


# noinspection DuplicatedCode
def validate_dictionary(value: _tany, attributes: _olist_str = None) -> _tmc_dict:

    if not _is_dictionary(value) or len(value) == 0:
        raise ValueError('The "@arg@" parameter must be a non-empty dictionary.')

    keys_length = 2 if attributes is None else 3
    keys = value.keys()

    if not all(_is_tuple(k) and len(k) == keys_length and all(_is_string(x) for x in k) for k in keys):
        raise ValueError(f'The "@arg@" parameter keys must be defined as tuples of {keys_length:d} non-empty strings.')

    labels = set()

    if attributes is not None:

        attribute_first = attributes[0]
        labels_by_attribute = {attribute: set() for attribute in attributes}

        for k in keys:

            key_attribute = k[0]

            if key_attribute not in attributes:
                raise ValueError(f'The "@arg@" parameter keys must be defined as tuples of non-empty strings whose first item matches one of the following values: {", ".join(attributes)}.')

            labels_by_attribute[attribute_first].add(k[1])
            labels_by_attribute[key_attribute].add(k[2])

        labels_first = labels_by_attribute[attribute_first]

        for attribute, labels in labels_by_attribute.items():
            if not all((attribute,) + combination in value for combination in _it.product(labels_first, labels)):
                raise ValueError(f'The "@arg@" parameter keys must contain all the possible combinations of labels {attribute_first} and {attribute}.')

    else:

        for key in keys:
            labels.add(key[0])
            labels.add(key[1])

        if not all(combination in value for combination in _it.product(labels, repeat=2)):
            raise ValueError('The "@arg@" parameter keys must contain all the possible combinations of labels.')

    values = value.values()

    if not all(_is_number(v) for v in values):
        raise ValueError('The "@arg@" parameter values must be float or integer numbers.')

    result = {}

    for k, v in value.items():

        v = float(v)

        if _np.isfinite(v) and _np.isreal(v) and 0.0 <= v <= 1.0:
            result[k] = v
        else:
            raise ValueError('The "@arg@" parameter values can contain only finite real numbers between 0.0 and 1.0.')

    return result


def validate_dpi(value: _tany) -> int:

    if not _is_integer(value):
        raise TypeError('The "@arg@" parameter must be an integer.')

    value = int(value)

    possible_values = (75, 100, 150, 200, 250, 300, 600)

    if value not in possible_values:
        possible_values = [str(possible_value) for possible_value in possible_values]
        raise ValueError(f'The "@arg@" parameter must have one of the following values: {", ".join(possible_values)}.')

    return value


# noinspection DuplicatedCode
def validate_emission_matrix(value: _tany, size: int) -> _tarray:

    try:
        value = _extract_numeric(value, float)
    except Exception as ex:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from ex

    if value.ndim != 2 or value.shape[0] != size or value.shape[1] < 2:
        raise ValueError(f'The "@arg@" parameter must be a 2d matrix with at least 2 columns and {size:d} rows.')

    if not all(_np.isfinite(x) and _np.isreal(x) and 0.0 <= x <= 1.0 for _, x in _np.ndenumerate(value)):
        raise ValueError('The "@arg@" parameter must contain only finite real values between 0.0 and 1.0.')

    if not _np.allclose(_np.sum(value, axis=1), _np.ones(value.shape[0], dtype=float)):
        raise ValueError('The "@arg@" parameter rows must sum to 1.0.')

    return value


def validate_enumerator(value: _tany, possible_values: _tlist_str) -> str:

    if not _is_string(value):
        raise TypeError('The "@arg@" parameter must be a non-empty string.')

    if value not in possible_values:
        raise ValueError(f'The "@arg@" parameter value must be one of the following: {", ".join(possible_values)}.')

    return value


def validate_file_path(value: _tany, accepted_extensions: _olist_str, write_permission: bool) -> _tfile:

    if isinstance(value, _pl.Path):
        file_path = value
    elif _is_string(value):
        file_path = _pl.Path(value)
    else:
        raise TypeError('The "@arg@" parameter must be a non-empty string or a path object.')

    try:
        file_extension = _get_file_extension(file_path)
    except Exception as ex:  # pragma: no cover
        raise ValueError('The "@arg@" parameter defines an invalid file path.') from ex

    if accepted_extensions is not None and len(accepted_extensions) > 0 and file_extension not in accepted_extensions:
        raise ValueError(f'The "@arg@" parameter must have one of the following extensions: {", ".join(sorted(accepted_extensions)).replace(".", "")}.')

    if write_permission:

        if not _osp.isdir(_osp.dirname(file_path)) and _osp.isabs(file_path):
            raise ValueError('The "@arg@" parameter defines a non-existent parent directory.')

        try:

            with open(file_path, mode='w'):
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


def validate_float(value: _tany, lower_limit: _olimit_float = None, upper_limit: _olimit_float = None) -> float:

    if not _is_float(value):
        raise TypeError('The "@arg@" parameter must be a float.')

    value = float(value)

    if not _np.isfinite(value) or not _np.isreal(value):
        raise ValueError('The "@arg@" parameter be a finite real value.')

    _validate_limits(value, 'float', lower_limit, upper_limit)

    return value


def validate_graph(value: _tany, layers: _oint = None, edge_attributes: _oedge_attributes = None) -> _tgraphs:

    result, multi = _is_graph(value)

    if not result:
        raise ValueError('The "@arg@" parameter must be a directed graph.')

    if multi:
        value = _nx.DiGraph(value)

    if layers is not None:

        layers = tuple(range(layers))

        nodes = list(value.nodes(data='layer', default=-1))
        nodes_all = []
        nodes_by_layer = {layer: [] for layer in layers}

        for node in nodes:

            node_label = node[0]

            if not _is_string(node_label):  # pragma: no cover
                raise ValueError('The "@arg@" parameter must define node labels as non-empty strings.')

            node_layer = node[1]

            if not _is_integer(node_layer) or node_layer not in layers:  # pragma: no cover
                raise ValueError(f'The "@arg@" parameter must define node layer attributes as integers matching one of the following values: {", ".join(str(layer) for layer in layers)}.')

            nodes_all.append(node_label)
            nodes_by_layer[node_layer].append(node_label)

        if any(len(nodes) < 2 for nodes in nodes_by_layer.values()):
            raise ValueError('The "@arg@" parameter must define at least 2 nodes for each layer.')

    else:

        nodes = list(value.nodes)

        if len(nodes) < 2:
            raise ValueError('The "@arg@" parameter must contain at least 2 nodes.')

        if not all(_is_string(node) for node in nodes):  # pragma: no cover
            raise ValueError('The "@arg@" parameter must define node labels as non-empty strings.')

    edge_weights = list(value.edges(data='weight', default=0.0))

    if not all(_is_number(edge_weight[2]) and float(edge_weight[2]) > 0.0 for edge_weight in edge_weights):  # pragma: no cover
        raise ValueError('The "@arg@" parameter must define edge weight attributes as non-negative numbers.')

    if edge_attributes is not None:

        for edge_attribute, edge_attribute_values in edge_attributes:

            edges = list(value.edges(data=edge_attribute, default=''))

            if not all(_is_string(edge[2]) and edge[2] in edge_attribute_values for edge in edges):
                raise ValueError(f'The "@arg@" parameter must define edge {edge_attribute} attributes as strings matching one of the following values: {", ".join(edge_attribute_values)}.')

    return value


def validate_hidden_markov_model(value: _tany) -> _thmm:

    if value is None or (f'{value.__module__}.{value.__class__.__name__}' != 'pydtmc.hidden_markov_model.HiddenMarkovModel'):
        raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    return value


def validate_hidden_markov_models(value: _tany) -> _tmc:

    if not _is_list(value):
        raise ValueError('The "@arg@" parameter must be a list.')

    value_length = len(value)

    if value_length < 2:
        raise ValueError('The "@arg@" parameter must contain at least 2 elements.')

    for i in range(value_length):
        try:
            validate_hidden_markov_model(value[i])
        except Exception as ex:
            raise ValueError('The "@arg@" parameter contains invalid elements.') from ex

    return value


def validate_hyperparameter(value: _tany, size: int) -> _tarray:

    try:
        value = _extract_numeric(value, float)
    except Exception as ex:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from ex

    if value.ndim != 2 or value.shape[0] != size or value.shape[1] != size:
        raise ValueError(f'The "@arg@" parameter must be a 2d square matrix with size equal to {size:d}.')

    if not all(_np.isfinite(x) and _np.isreal(x) and _np.equal(_np.mod(x, 1.0), 0.0) and x >= 1.0 for _, x in _np.ndenumerate(value)):
        raise ValueError('The "@arg@" parameter must contain only integers greater than or equal to 1.')

    return value


def validate_integer(value: _tany, lower_limit: _olimit_int = None, upper_limit: _olimit_int = None) -> int:

    if not _is_integer(value):
        raise TypeError('The "@arg@" parameter must be an integer.')

    value = int(value)

    _validate_limits(value, 'integer', lower_limit, upper_limit)

    return value


def validate_interval(value: _tany) -> _tinterval:

    if not _is_tuple(value):
        raise TypeError('The "@arg@" parameter must be a tuple.')

    if len(value) != 2:
        raise ValueError('The "@arg@" parameter must contain 2 elements.')

    a, b = value

    if not _is_number(a) or not _is_number(b):
        raise ValueError('The "@arg@" parameter must contain only floats and integers.')

    a, b = float(a), float(b)

    if not all(_np.isfinite(x) and _np.isreal(x) and x >= 0.0 for x in [a, b]):
        raise ValueError('The "@arg@" parameter must contain only finite real values greater than or equal to 0.0.')

    if a >= b:
        raise ValueError('The "@arg@" parameter must contain two distinct values, and the first value must be less than the second one.')

    return a, b


def validate_label(value: _tany, labels: _tlist_str) -> int:

    if _is_integer(value):

        label = int(value)
        limit = len(labels) - 1

        if label < 0 or label > limit:
            raise ValueError(f'The "@arg@" parameter, when specified as an integer, must have a value between 0 and {limit:d}.')

        return label

    if _is_string(value):

        if value not in labels:
            raise ValueError(f'The "@arg@" parameter, when specified as a string, must match one of the following strings: {", ".join(labels)}.')

        label = labels.index(value)

        return label

    raise TypeError('The "@arg@" parameter must be either an integer or a non-empty string.')


def validate_labels_current(value: _tany, labels: _tlist_str, subset: bool, minimum_length: _oint = None) -> _tlist_int:

    if _is_integer(value):

        value = int(value)
        limit = len(labels) - 1

        if value < 0 or value > limit:
            raise ValueError(f'The "@arg@" parameter, when specified as an integer, must have a value between 0 and {limit:d}.')

        value = [value]

        return value

    if _is_list(value):

        if all(_is_integer(label) for label in value):
            value_type = 'integer'
        elif all(_is_string(label) for label in value):
            value_type = 'string'
        else:
            raise TypeError('The "@arg@" parameter contains invalid elements.')

        labels_length = len(labels)

        if value_type == 'integer':

            value = [int(label) for label in value]

            if any(label < 0 or label >= labels_length for label in value):
                raise ValueError(f'The "@arg@" parameter, when specified as a list of integers, must contain only values between 0 and {labels_length - 1:d}.')

        else:

            value = [labels.index(label) if label in labels else -1 for label in value]

            if any(label == -1 for label in value):
                raise ValueError(f'The "@arg@" parameter, when specified as a list of strings, must contain only values matching the following strings: {", ".join(labels)}.')

        value_length = len(value)

        if len(set(value)) < value_length:
            raise ValueError('The "@arg@" parameter must contain only unique values.')

        if value_length == 0:
            raise ValueError('The "@arg@" parameter must contain at least an element.')

        maximum_length = labels_length - 1 if subset else labels_length

        if minimum_length is None or minimum_length == 1:
            if value_length > maximum_length:
                raise ValueError(f'The "@arg@" parameter must contain no more than {maximum_length:d} elements.')
        else:

            if value_length < minimum_length or value_length > maximum_length:

                if minimum_length == maximum_length:
                    length = {minimum_length, maximum_length}.pop()
                    raise ValueError(f'The "@arg@" parameter must contain a number of elements equal to {length:d}.')

                raise ValueError(f'The "@arg@" parameter must contain a number of elements between {minimum_length:d} and {maximum_length:d}.')  # pragma: no cover

        value = sorted(value)

        return value

    if _is_string(value):

        if value not in labels:
            raise ValueError(f'The "@arg@" parameter, when specified as a string, must match one of the following strings: {", ".join(labels)}.')

        value = [labels.index(value)]

        return value

    raise TypeError('The "@arg@" parameter must be either an integer, a non-empty string, a list of integers or a list of non-empty strings.')


def validate_labels_input(value: _tany, size: _oint = None) -> _tlist_str:

    if not _is_list(value):
        raise ValueError('The "@arg@" parameter must be a list.')

    if not all(_is_string(label) for label in value):
        raise TypeError('The "@arg@" parameter must contain only non-empty strings.')

    labels_length = len(value)

    if labels_length < 2:
        raise ValueError('The "@arg@" parameter must contain at least 2 elements.')

    labels_unique_length = len(set(value))

    if labels_unique_length < labels_length:
        raise ValueError('The "@arg@" parameter must contain only unique values.')

    if size is not None and labels_length != size:
        raise ValueError(f'The "@arg@" parameter must contain a number of elements equal to {size:d}.')

    return value


def validate_markov_chain(value: _tany, size: _oint = None) -> _tmc:

    if value is None or (f'{value.__module__}.{value.__class__.__name__}' != 'pydtmc.markov_chain.MarkovChain'):
        raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    if size is not None and value.size != size:
        raise ValueError(f'The "@arg@" parameter must be a Markov chain with size equal to {size:d}.')

    return value


def validate_markov_chains(value: _tany) -> _tmc:

    if not _is_list(value):
        raise ValueError('The "@arg@" parameter must be a list.')

    value_length = len(value)

    if value_length < 2:
        raise ValueError('The "@arg@" parameter must contain at least 2 elements.')

    for i in range(value_length):
        try:
            validate_markov_chain(value[i])
        except Exception as ex:
            raise ValueError('The "@arg@" parameter contains invalid elements.') from ex

    return value


def validate_mask(value: _tany, rows: int, columns: int) -> _tarray:

    try:
        value = _extract_numeric(value, float)
    except Exception as ex:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from ex

    if value.ndim != 2 or value.shape[0] != rows or value.shape[1] != columns:

        if rows == columns:
            size = {rows, columns}.pop()
            raise ValueError(f'The "@arg@" parameter must be a 2d square matrix with size equal to {size:d}.')

        raise ValueError(f'The "@arg@" parameter must be a 2d matrix with {rows:d} rows and {columns:d} columns.')

    if not all(_np.isnan(x) or (_np.isfinite(x) and _np.isreal(x) and 0.0 <= x <= 1.0) for _, x in _np.ndenumerate(value)):
        raise ValueError('The "@arg@" parameter can contain only NaNs and finite real values between 0.0 and 1.0.')

    if _np.any(_np.nansum(value, axis=1, dtype=float) > 1.0):
        raise ValueError('The "@arg@" parameter row sums must not exceed 1.')

    return value


def validate_matrix(value: _tany, rows: _oint = None, columns: _oint = None) -> _tarray:

    try:
        value = _extract_numeric(value, float)
    except Exception as ex:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from ex

    if columns is None and rows is None:
        if value.ndim != 2 or value.shape[0] < 2 or value.shape[0] != value.shape[1]:
            raise ValueError('The "@arg@" parameter must be a 2d square matrix with size greater than or equal to 2.')
    elif columns is not None and rows is None:
        if value.ndim != 2 or value.shape[0] < 2 or value.shape[1] != columns:
            raise ValueError(f'The "@arg@" parameter must be a 2d matrix with at least 2 rows and {columns:d} columns.')
    elif columns is None and rows is not None:
        if value.ndim != 2 or value.shape[0] != rows or value.shape[1] < 2:
            raise ValueError(f'The "@arg@" parameter must be a 2d matrix with {rows:d} rows and at least 2 columns.')
    else:
        if value.ndim != 2 or value.shape[0] != rows or value.shape[1] != columns:
            raise ValueError(f'The "@arg@" parameter must be a 2d matrix with {rows:d} rows and {columns:d} columns.')

    if not all(_np.isfinite(x) and _np.isreal(x) and x >= 0.0 for _, x in _np.ndenumerate(value)):
        raise ValueError('The "@arg@" parameter must contain only finite real values greater than or equal to 0.0.')

    return value


# noinspection PyBroadException
def validate_model(value: _tany) -> _tmodel:

    if value is None:
        raise TypeError('The "@arg@" parameter is null.')

    try:
        value_module = value.__module__
    except Exception:
        value_module = value.__class__.__module__

    if not value_module.startswith('pydtmc.'):
        raise TypeError('The "@arg@" parameter is wrongly typed.')

    value_bases = value.__class__.__bases__ or ()
    value_base = None if len(value_bases) == 0 else _get_full_name(value_bases[0])

    if value_base is None or value_base != 'pydtmc.base_classes.Model':
        raise TypeError('The "@arg@" parameter is wrongly typed.')

    return value


def validate_models(value: _tany) -> _tlist_model:

    if not _is_list(value):
        raise ValueError('The "@arg@" parameter must be a list.')

    value_length = len(value)

    if value_length < 2:
        raise ValueError('The "@arg@" parameter must contain at least 2 elements.')

    for i in range(value_length):
        try:
            validate_model(value[i])
        except Exception as ex:
            raise ValueError('The "@arg@" parameter contains invalid elements.') from ex

    return value


def validate_partitions(value: _tany, labels: _tlist_str) -> _tlists_int:

    if not _is_list(value):
        raise ValueError('The "@arg@" parameter must be a list.')

    partitions_length = len(value)
    labels_length = len(labels)

    if partitions_length < 2 or partitions_length >= labels_length:

        if labels_length == 2:  # pragma: no cover
            raise ValueError('The "@arg@" parameter must contain a number of elements equal to 2.')

        raise ValueError(f'The "@arg@" parameter must contain a number of elements between 2 and {labels_length - 1:d}.')

    partitions_flat = []
    partitions_groups = []

    for partition in value:

        if not _is_list(partition):
            raise TypeError('The "@arg@" parameter must contain only lists.')

        partition_list = list(partition)

        partitions_flat.extend(partition_list)
        partitions_groups.append(len(partition_list))

    if all(_is_integer(state) for state in partitions_flat):

        partitions_flat = [int(state) for state in partitions_flat]

        if any(label < 0 or label >= labels_length for label in partitions_flat):
            raise ValueError(f'The "@arg@" parameter subelements, when specified as integers, must be values between 0 and {labels_length - 1:d}.')

    elif all(_is_string(partition_flat) for partition_flat in partitions_flat):

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


def validate_random_distribution(value: _tany, rng: _trand, accepted_values: _tlist_str) -> _trandfunc:

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

    if _is_string(value):

        if value is None or value not in dir(rng):
            raise ValueError('The "@arg@" parameter, when specified as a string, must reference a valid "numpy.random" module object.')

        if len(accepted_values) > 0 and value not in accepted_values:
            raise ValueError(f'The "@arg@" parameter, when specified as a string, must reference one of the following "numpy.random" module objects: {", ".join(accepted_values)}.')

        value = getattr(rng, value)

        return value

    raise TypeError('The "@arg@" parameter must be either a callable function or the name of a callable function.')


def validate_rewards(value: _tany, size: int) -> _tarray:

    try:
        value = _extract_numeric(value, float)
    except Exception as ex:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from ex

    if value.ndim > 2 or (value.ndim == 2 and value.shape[0] != 1) or (value.ndim == 1 and value.shape[0] == 0):
        raise ValueError('The "@arg@" parameter must be a valid vector.')

    value = _np.ravel(value)

    if value.size != size:
        raise ValueError(f'The "@arg@" parameter length must be equal to {size:d}.')

    if not all(_np.isfinite(x) and _np.isreal(x) for _, x in _np.ndenumerate(value)):
        raise ValueError('The "@arg@" parameter must contain only finite real values.')

    return value


def validate_sequence(value: _tany, labels: _tlist_str) -> _tlist_int:

    if not _is_list(value):
        raise ValueError('The "@arg@" parameter must be a list.')

    if len(value) < 2:
        raise ValueError('The "@arg@" parameter must contain at least 2 elements.')

    if all(_is_integer(label) for label in value):
        value_type = 'integer'
    elif all(_is_string(label) for label in value):
        value_type = 'string'
    else:
        raise TypeError('The "@arg@" parameter must be either a list of integers or a list of non-empty strings.')

    if value_type == 'integer':

        labels_length = len(labels)

        if any(label < 0 or label >= labels_length for label in value):
            raise ValueError(f'The "@arg@" parameter, when specified as a list of integers, must contain only values between 0 and {labels_length - 1:d}.')

    else:

        value = [labels.index(label) if label in labels else -1 for label in value]

        if any(label == -1 for label in value):
            raise ValueError(f'The "@arg@" parameter, when specified as a list of strings, must contain only values matching the following strings: {", ".join(labels)}.')

    return value


def validate_sequences(value: _tany, labels: _tlist_str, flex: bool) -> _tlists_int:

    if not _is_list(value):
        raise ValueError('The "@arg@" parameter must be a list.')

    if len(value) < 2:
        raise ValueError('The "@arg@" parameter must contain at least 2 elements.')

    if all(_is_list(state) for state in value):
        value_type = 'list'
    elif flex and all(_is_integer(label) for label in value):
        value_type = 'integer'
    elif flex and all(_is_string(label) for label in value):
        value_type = 'string'
    else:

        if flex:
            raise TypeError('The "@arg@" parameter must be either a list of integers, a list of non-empty strings, a list of lists of integers or a list of lists of non-empty strings.')

        raise TypeError('The "@arg@" parameter must be either a list of lists of integers or a list of lists of non-empty strings.')

    if value_type in ['integer', 'string']:
        value = [value]

    for index, value_current in enumerate(value):
        try:
            value[index] = validate_sequence(value_current, labels)
        except Exception as ex:
            raise ValueError('The "@arg@" parameter contains invalid elements.') from ex

    return value


def validate_status(value: _tany, labels: _tlist_str) -> _tarray:

    size = len(labels)

    if _is_integer(value):

        value = int(value)
        limit = size - 1

        if value < 0 or value > limit:
            raise ValueError(f'The "@arg@" parameter, when specified as an integer, must have a value between 0 and {limit:d}.')

        result = _np.zeros(size, dtype=float)
        result[value] = 1.0

        return result

    if _is_string(value):

        if value not in labels:
            raise ValueError(f'The "@arg@" parameter, when specified as a string, must match one of the following strings: {", ".join(labels)}.')

        value = labels.index(value)

        result = _np.zeros(size, dtype=float)
        result[value] = 1.0

        return result

    try:
        value = _extract_numeric(value, float)
    except Exception as ex:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from ex

    if value.ndim > 2 or (value.ndim == 2 and value.shape[0] != 1):
        raise ValueError('The "@arg@" parameter must be a valid vector.')

    value = _np.ravel(value)

    if value.size != size:
        raise ValueError(f'The "@arg@" parameter length must be equal to {size:d}.')

    if not all(_np.isfinite(x) and _np.isreal(x) and 0.0 <= x <= 1.0 for _, x in _np.ndenumerate(value)):
        raise ValueError('The "@arg@" parameter must contain only finite real values between 0.0 and 1.0.')

    if not _np.isclose(_np.sum(value), 1.0):
        raise ValueError('The "@arg@" parameter values must sum to 1.0.')

    return value


def validate_strings(value: _tany, size: _oint = None) -> _tlist_str:

    if not _is_list(value):
        raise ValueError('The "@arg@" parameter must be a list.')

    if not all(_is_string(s) for s in value):
        raise TypeError('The "@arg@" parameter must contain only non-empty strings.')

    value_length = len(value)

    if value_length == 0:
        raise ValueError('The "@arg@" parameter must contain at least an element.')

    if size is not None and value_length != size:
        raise ValueError(f'The "@arg@" parameter must contain a number of elements equal to {size:d}.')

    return value


def validate_time_points(value: _tany) -> _ttimes_in:

    if _is_integer(value):

        value = int(value)

        if value < 0:
            raise ValueError('The "@arg@" parameter, when specified as an integer, must be greater than or equal to 0.')

        return value

    if _is_list(value):

        if not all(_is_integer(time_point) for time_point in value):
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
def validate_transition_function(value: _tany) -> _ttfunc:

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

    if not _is_number(result):
        raise ValueError('The "@arg@" parameter behavior is not compliant.')

    result = float(result)

    if not _np.isfinite(result) or not _np.isreal(result):
        raise ValueError('The "@arg@" parameter behavior is not compliant.')

    return value


# noinspection DuplicatedCode
def validate_transition_matrix(value: _tany, size: _oint = None) -> _tarray:

    try:
        value = _extract_numeric(value, float)
    except Exception as ex:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from ex

    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] < 2:
        raise ValueError('The "@arg@" parameter must be a 2d square matrix with size greater than or equal to 2.')

    if size is not None and value.shape[0] != size:
        raise ValueError(f'The "@arg@" parameter must have a size equal to {size:d}.')

    if not all(_np.isfinite(x) and _np.isreal(x) and 0.0 <= x <= 1.0 for _, x in _np.ndenumerate(value)):
        raise ValueError('The "@arg@" parameter must contain only finite real values between 0.0 and 1.0.')

    if not _np.allclose(_np.sum(value, axis=1), _np.ones(value.shape[0], dtype=float)):
        raise ValueError('The "@arg@" parameter rows must sum to 1.0.')

    return value


def validate_vector(value: _tany, vector_type: str, flex: bool, size: _oint = None) -> _tarray:

    if flex and _is_number(value):

        if vector_type != 'unconstrained':
            raise ValueError('The "@arg@" parameter must be unconstrained.')

        if size is None:
            raise ValueError('The "@arg@" parameter must have a defined size.')

        value = _np.repeat(float(value), size)

    else:

        try:
            value = _extract_numeric(value, float)
        except Exception as ex:
            raise TypeError('The "@arg@" parameter is null or wrongly typed.') from ex

        if value.ndim > 2 or (value.ndim == 2 and value.shape[0] != 1) or (value.ndim == 1 and value.shape[0] == 0):
            raise ValueError('The "@arg@" parameter must be a valid vector.')

        value = _np.ravel(value)

        if size is not None and value.size != size:
            raise ValueError(f'The "@arg@" parameter length must be equal to {size:d}.')

    if not all(_np.isfinite(x) and _np.isreal(x) and 0.0 <= x <= 1.0 for _, x in _np.ndenumerate(value)):
        raise ValueError('The "@arg@" parameter must contain only finite real values between 0.0 and 1.0.')

    if vector_type == 'annihilation' and not _np.isclose(value[0], 0.0):
        raise ValueError('The "@arg@" parameter must contain a value equal to 0 in the first index.')

    if vector_type == 'creation' and not _np.isclose(value[-1], 0.0):
        raise ValueError('The "@arg@" parameter must contain a value equal to 0 in the last index.')

    if vector_type == 'stochastic' and not _np.isclose(_np.sum(value), 1.0):
        raise ValueError('The "@arg@" parameter values must sum to 1.0.')

    return value
