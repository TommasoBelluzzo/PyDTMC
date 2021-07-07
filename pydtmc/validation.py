# -*- coding: utf-8 -*-

__all__ = [
    'validate_boolean',
    'validate_boundary_condition',
    'validate_dictionary',
    'validate_distribution',
    'validate_dpi',
    'validate_enumerator',
    'validate_file_path',
    'validate_float',
    'validate_graph',
    'validate_hyperparameter',
    'validate_integer',
    'validate_interval',
    'validate_markov_chain',
    'validate_mask',
    'validate_matrix',
    'validate_partitions',
    'validate_rewards',
    'validate_state',
    'validate_state_names',
    'validate_states',
    'validate_status',
    'validate_time_points',
    'validate_transition_function',
    'validate_transition_matrix',
    'validate_vector'
]


###########
# IMPORTS #
###########

# Standard

from copy import (
    deepcopy
)

from inspect import (
    isclass,
    signature
)

from itertools import (
    product
)

from os.path import (
    isfile
)

from typing import (
    Iterable
)

# Libraries

import networkx as nx
import numpy as np
import scipy.sparse as spsp

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

# Internal

from .custom_types import (
    oint,
    olimit_float,
    olimit_int,
    tany,
    tarray,
    tbcond,
    tdists_flex,
    tgraphs,
    tinterval,
    tlist_any,
    tlist_int,
    tlist_str,
    tlists_int,
    tmc,
    tmc_dict,
    ttfunc,
    ttimes_in
)


#############
# FUNCTIONS #
#############

def _is_array(value: tany) -> bool:

    return value is not None and isinstance(value, np.ndarray)


def _is_bool(value: tany) -> bool:

    return value is not None and isinstance(value, bool)


def _is_dictionary(value: tany) -> bool:

    return value is not None and isinstance(value, dict)


def _is_float(value: tany) -> bool:

    return value is not None and isinstance(value, (float, np.floating))


def _is_graph(value: tany, multi: bool) -> bool:

    if multi:
        return value is not None and isinstance(value, nx.MultiDiGraph)

    return value is not None and isinstance(value, nx.DiGraph)


def _is_integer(value: tany) -> bool:

    return value is not None and isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _is_iterable(value: tany) -> bool:

    return value is not None and isinstance(value, Iterable) and not isinstance(value, (bytearray, bytes, str))


def _is_list(value: tany) -> bool:

    return value is not None and isinstance(value, list)


def _is_number(value: tany) -> bool:

    return _is_float(value) or _is_integer(value)


def _is_pandas(value: tany) -> bool:

    if pd is None:  # pragma: no cover
        return False

    return value is not None and isinstance(value, (pd.DataFrame, pd.Series))


def _is_spmatrix(value: tany) -> bool:

    return value is not None and isinstance(value, spsp.spmatrix)


def _is_string(value: tany) -> bool:

    return value is not None and isinstance(value, str) and len(value) > 0


def _is_tuple(value: tany) -> bool:

    return value is not None and isinstance(value, tuple)


def _extract(data: tany) -> tlist_any:

    if _is_list(data):
        result = deepcopy(data)
    elif _is_dictionary(data):
        result = list(data.values())
    elif _is_iterable(data):
        result = list(data)
    else:
        result = None

    if result is None:
        raise TypeError('The data type is not supported.')

    return result


def _extract_as_numeric(data: tany) -> tarray:

    if _is_list(data):
        result = np.array(data)
    elif _is_dictionary(data):
        result = np.array(list(data.values()))
    elif _is_array(data):
        result = np.copy(data)
    elif _is_spmatrix(data):
        result = np.array(data.todense())
    elif _is_pandas(data):
        result = data.to_numpy(copy=True)
    elif _is_iterable(data):
        result = np.array(list(data))
    else:
        result = None

    if result is None or not np.issubdtype(result.dtype, np.number):
        raise TypeError('The data type is not supported.')

    return result


def validate_boolean(value: tany) -> bool:

    if not _is_bool(value):
        raise TypeError('The "@arg@" parameter must be a boolean value.')

    return value


def validate_boundary_condition(value: tany) -> tbcond:

    if _is_number(value):

        value = float(value)

        if (value < 0.0) or (value > 1.0):
            raise ValueError('The "@arg@" parameter, when specified as a number, must have a value between 0 and 1.')

        return value

    if _is_string(value):

        possible_values = ['absorbing', 'reflecting']

        if value not in possible_values:
            raise ValueError(f'The "@arg@" parameter, when specified as a string, must have one of the following values: {", ".join(possible_values)}.')

        return value

    raise TypeError('The "@arg@" parameter must be either a float representing the first probability of the semi-reflecting condition or a non-empty string representing the boundary condition type.')


def validate_dictionary(value: tany) -> tmc_dict:

    if not _is_dictionary(value):
        raise ValueError('The "@arg@" parameter must be a dictionary.')

    d_keys = value.keys()

    if not all(_is_tuple(d_key) and len(d_key) == 2 and _is_string(d_key[0]) and _is_string(d_key[1]) for d_key in d_keys):
        raise ValueError('The "@arg@" parameter keys must be tuples containing two non-empty strings.')

    states = set()

    for d_key in d_keys:
        states.add(d_key[0])
        states.add(d_key[1])

    combinations = list(product(states, repeat=2))

    if len(value) != len(combinations) or not all(combination in value for combination in combinations):
        raise ValueError('The "@arg@" parameter keys must contain all the possible combinations of states.')

    d_values = value.values()

    if not all(_is_number(d_value) for d_value in d_values):
        raise ValueError('The "@arg@" parameter values must be float or integer numbers.')

    result = {}

    for d_key, d_value in value.items():

        d_value = float(d_value)

        if np.isfinite(d_value) and np.isreal(d_value) and 0.0 <= d_value <= 1.0:
            result[d_key] = d_value
        else:
            raise ValueError('The "@arg@" parameter values can contain only finite real numbers between 0 and 1.')

    return result


def validate_distribution(value: tany, size: int) -> tdists_flex:

    if _is_integer(value):

        value = int(value)

        if value <= 0:
            raise ValueError('The "@arg@" parameter, when specified as an integer, must be greater than or equal to 1.')

        return value

    if _is_list(value):

        value_len = len(value)

        if value_len <= 1:
            raise ValueError('The "@arg@" parameter, when specified as a list of vectors, must contain at least 2 elements.')

        for index, vector in enumerate(value):

            if not _is_array(vector) or not np.issubdtype(vector.dtype, np.number):
                raise TypeError('The "@arg@" parameter must contain only numeric vectors.')

            vector = vector.astype(float)
            value[index] = vector

            if vector.ndim != 1 or vector.size != size:
                raise ValueError('The "@arg@" parameter must contain only vectors of size {size:d}.')

            if not all(np.isfinite(x) and np.isreal(x) and 0.0 <= x <= 1.0 for x in np.nditer(vector)):
                raise ValueError('The "@arg@" parameter must contain only vectors consisting of finite real values between 0 and 1.')

            if not np.isclose(np.sum(vector), 1.0):
                raise ValueError('The "@arg@" parameter must contain only vectors consisting of values whose sum is 1.')

        return value

    raise TypeError('The "@arg@" parameter must be either an integer representing the number of redistributions to perform or a list of valid distributions.')


def validate_dpi(value: tany) -> int:

    if not _is_integer(value):
        raise TypeError('The "@arg@" parameter must be an integer.')

    value = int(value)

    possible_values = [75, 100, 150, 200, 300]

    if value not in possible_values:
        possible_values = [str(possible_value) for possible_value in possible_values]
        raise ValueError(f'The "@arg@" parameter must have one of the following values: {", ".join(possible_values)}.')

    return value


def validate_enumerator(value: tany, possible_values: tlist_str) -> str:

    if not all(_is_string(possible_value) for possible_value in possible_values):
        raise ValueError('The list of possible enumerator values must contain only non-empty strings.')

    if not _is_string(value):
        raise TypeError('The "@arg@" parameter must be a non-empty string.')

    if value not in possible_values:
        raise ValueError(f'The "@arg@" parameter value must be one of the following: {", ".join(possible_values)}.')

    return value


def validate_file_path(value: tany, write_permission: bool) -> str:  # pragma: no cover

    if not _is_string(value):
        raise TypeError('The "@arg@" parameter must be a non-empty string.')

    if len(value.strip()) == 0:
        raise ValueError('The "@arg@" parameter must not be a non-empty string.')

    if write_permission:

        try:

            with open(value, mode='w'):
                pass

        except Exception as e:
            raise ValueError('The "@arg@" parameter defines the path to an inaccessible file.') from e

    else:

        if not isfile(value):
            raise ValueError('The "@arg@" parameter defines an invalid file path.')

        file_empty = False

        try:

            with open(value, mode='r') as file:

                file.seek(0)

                if not file.read(1):
                    file_empty = True

        except Exception as e:
            raise ValueError('The "@arg@" parameter defines the path to an inaccessible file.') from e

        if file_empty:
            raise ValueError('The "@arg@" parameter defines the path to an empty file.')

    return value


def validate_float(value: tany, lower_limit: olimit_float = None, upper_limit: olimit_float = None) -> float:

    if not _is_float(value):
        raise TypeError('The "@arg@" parameter must be a float.')

    value = float(value)

    if not np.isfinite(value) or not np.isreal(value):
        raise ValueError('The "@arg@" parameter be a finite real value.')

    if lower_limit is not None:
        if lower_limit[1]:
            if value <= lower_limit[0]:
                raise ValueError(f'The "@arg@" parameter must be greater than {lower_limit[0]:f}.')
        else:
            if value < lower_limit[0]:
                raise ValueError(f'The "@arg@" parameter must be greater than or equal to {lower_limit[0]:f}.')

    if upper_limit is not None:
        if upper_limit[1]:
            if value >= upper_limit[0]:
                raise ValueError(f'The "@arg@" parameter must be less than {upper_limit[0]:f}.')
        else:
            if value > upper_limit[0]:
                raise ValueError(f'The "@arg@" parameter must be less than or equal to {upper_limit[0]:f}.')

    return value


def validate_graph(value: tany) -> tgraphs:

    non_multi = _is_graph(value, False)
    multi = _is_graph(value, True)

    if not non_multi and not multi:
        raise ValueError('The "@arg@" parameter must be a directed graph.')

    if multi:
        value = nx.DiGraph(value)

    nodes = list(value.nodes)
    nodes_length = len(nodes)

    if nodes_length < 2:
        raise ValueError('The "@arg@" parameter must contain a number of nodes greater than or equal to 2.')

    if not all(_is_string(node) for node in nodes):
        raise ValueError('The "@arg@" parameter must define node labels as non-empty strings.')

    edges = list(value.edges(data='weight', default=0.0))

    if not all(_is_number(edge[2]) and float(edge[2]) > 0.0 for edge in edges):
        raise ValueError('The "@arg@" parameter must define edge wright as non-negative numbers.')

    return value


def validate_hyperparameter(value: tany, size: int) -> tarray:

    try:
        value = _extract_as_numeric(value)
    except Exception as e:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from e

    value = value.astype(float)

    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] != size:
        raise ValueError(f'The "@arg@" parameter must be a 2d square matrix with size equal to {size:d}.')

    if not all(np.isfinite(x) and np.isreal(x) and np.equal(np.mod(x, 1.0), 0.0) and x >= 1.0 for x in np.nditer(value)):
        raise ValueError('The "@arg@" parameter must contain only integers greater than or equal to 1.')

    return value


def validate_integer(value: tany, lower_limit: olimit_int = None, upper_limit: olimit_int = None) -> int:

    if not _is_integer(value):
        raise TypeError('The "@arg@" parameter must be an integer.')

    value = int(value)

    if lower_limit is not None:
        if lower_limit[1]:
            if value <= lower_limit[0]:
                raise ValueError(f'The "@arg@" parameter must be greater than {lower_limit[0]:d}.')
        else:
            if value < lower_limit[0]:
                raise ValueError(f'The "@arg@" parameter must be greater than or equal to {lower_limit[0]:d}.')

    if upper_limit is not None:
        if upper_limit[1]:
            if value >= upper_limit[0]:
                raise ValueError(f'The "@arg@" parameter must be less than {upper_limit[0]:d}.')
        else:
            if value > upper_limit[0]:
                raise ValueError(f'The "@arg@" parameter must be less than or equal to {upper_limit[0]:d}.')

    return value


def validate_interval(value: tany) -> tinterval:

    if not _is_tuple(value):
        raise TypeError('The "@arg@" parameter must be a tuple.')

    if len(value) != 2:
        raise ValueError('The "@arg@" parameter must contain 2 elements.')

    a = value[0]
    b = value[1]

    if not _is_number(a) or not _is_number(b):
        raise ValueError('The "@arg@" parameter must contain only floats and integers.')

    a = float(a)
    b = float(b)

    if not all(np.isfinite(x) and np.isreal(x) and x >= 0.0 for x in [a, b]):
        raise ValueError('The "@arg@" parameter must contain only finite real values greater than or equal to 0.0.')

    if a >= b:
        raise ValueError('The "@arg@" parameter must contain two distinct values, and the first value must be less than the second one.')

    return a, b


def validate_markov_chain(value: tany) -> tmc:

    if value is None or (f'{value.__module__}.{value.__class__.__name__}' != 'pydtmc.markov_chain.MarkovChain'):
        raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    return value


def validate_mask(value: tany, size: int) -> tarray:

    try:
        value = _extract_as_numeric(value)
    except Exception as e:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from e

    value = value.astype(float)

    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] != size:
        raise ValueError(f'The "@arg@" parameter must be a 2d square matrix with size equal to {size:d}.')

    if not all(np.isnan(x) or (np.isfinite(x) and np.isreal(x) and 0.0 <= x <= 1.0) for x in np.nditer(value)):
        raise ValueError('The "@arg@" parameter can contain only NaNs and finite real values between 0 and 1.')

    if np.any(np.nansum(value, axis=1, dtype=float) > 1.0):
        raise ValueError('The "@arg@" parameter row sums must not exceed 1.')

    return value


def validate_matrix(value: tany) -> tarray:

    try:
        value = _extract_as_numeric(value)
    except Exception as e:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from e

    value = value.astype(float)

    if value.ndim != 2 or value.shape[0] < 2 or value.shape[0] != value.shape[1]:
        raise ValueError('The "@arg@" parameter must be a 2d square matrix with size greater than or equal to 2.')

    if not all(np.isfinite(x) and np.isreal(x) and x >= 0.0 for x in np.nditer(value)):
        raise ValueError('The "@arg@" parameter must contain only finite real values greater than or equal to 0.0.')

    return value


def validate_partitions(value: tany, current_states: tlist_str) -> tlists_int:

    if not _is_list(value):
        raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    partitions_length = len(value)
    current_states_length = len(current_states)

    if partitions_length < 2 or partitions_length >= current_states_length:
        raise ValueError(f'The "@arg@" parameter must contain a number of elements between 0 and the number of existing states minus one ({current_states_length - 1:d}).')

    partitions_flat = []
    partitions_groups = []

    for partition in value:

        if not _is_iterable(partition):
            raise TypeError('The "@arg@" parameter must contain only array_like objects.')

        partition_list = list(partition)

        partitions_flat.extend(partition_list)
        partitions_groups.append(len(partition_list))

    if all(_is_integer(state) for state in partitions_flat):

        partitions_flat = [int(state) for state in partitions_flat]

        if any(state < 0 or state >= current_states_length for state in partitions_flat):
            raise ValueError(f'The "@arg@" parameter subelements, when specified as integers, must be values between 0 and the number of existing states minus one ({current_states_length - 1:d}).')

    elif all(_is_string(partition_flat) for partition_flat in partitions_flat):

        partitions_flat = [current_states.index(state) if state in current_states else -1 for state in partitions_flat]

        if any(state == -1 for state in partitions_flat):
            raise ValueError(f'The "@arg@" parameter subelements, when specified as strings, must contain only values matching the names of the existing states ({", ".join(current_states)}).')

    else:
        raise TypeError('The "@arg@" parameter must contain only array_like objects of integers or array_like objects of non-empty strings.')

    partitions_flat_length = len(partitions_flat)

    if len(set(partitions_flat)) < partitions_flat_length or partitions_flat_length != current_states_length or partitions_flat != list(range(current_states_length)):
        raise ValueError('The "@arg@" parameter subelements must be unique, include all the existing states and follow a sequential order.')

    result = []
    offset = 0

    for partitions_group in partitions_groups:
        extension = offset + partitions_group
        result.append(partitions_flat[offset:extension])
        offset += partitions_group

    return result


def validate_rewards(value: tany, size: int) -> tarray:

    try:
        value = _extract_as_numeric(value)
    except Exception as e:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from e

    value = value.astype(float)

    if (value.ndim > 2) or (value.ndim == 2 and value.shape[0] != 1):
        raise ValueError('The "@arg@" parameter must be a valid vector.')

    value = np.ravel(value)

    if value.size != size:
        raise ValueError(f'The "@arg@" parameter length must be equal to the number of states ({size:d}).')

    if not all(np.isfinite(x) and np.isreal(x) for x in np.nditer(value)):
        raise ValueError('The "@arg@" parameter must contain only finite real values.')

    return value


def validate_state(value: tany, current_states: tlist_str) -> int:

    if _is_integer(value):

        state = int(value)
        limit = len(current_states) - 1

        if state < 0 or state > limit:
            raise ValueError(f'The "@arg@" parameter, when specified as an integer, must have a value between 0 and the number of existing states minus one ({limit:d}).')

        return state

    if _is_string(value):

        if value not in current_states:
            raise ValueError(f'The "@arg@" parameter, when specified as a string, must match the name of an existing state ({", ".join(current_states)}).')

        return current_states.index(value)

    raise TypeError('The "@arg@" parameter must be either an integer or a non-empty string.')


def validate_state_names(value: tany, size: oint = None) -> tlist_str:

    try:
        value = _extract(value)
    except Exception as e:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from e

    if not all(_is_string(state) for state in value):
        raise TypeError('The "@arg@" parameter must contain only non-empty strings.')

    states_length = len(value)

    if states_length < 2:
        raise ValueError('The "@arg@" parameter must contain at least two elements.')

    states_unique = len(set(value))

    if states_unique < states_length:
        raise ValueError('The "@arg@" parameter must contain only unique values.')

    if size is not None and states_length != size:
        raise ValueError(f'The "@arg@" parameter must contain a number of elements equal to {size:d}.')

    return value


def validate_states(value: tany, current_states: tlist_str, states_type: str, flex: bool) -> tlist_int:

    if flex:

        if _is_integer(value):

            if states_type == 'walk':
                raise ValueError('The "@arg@" parameter, when specified as an integer, cannot be associated to a walk.')

            value = int(value)
            limit = len(current_states) - 1

            if value < 0 or value > limit:
                raise ValueError(f'The "@arg@" parameter, when specified as an integer, must have a value between 0 and the number of existing states minus one ({limit:d}).')

            value = [value]

            return value

        if _is_string(value):

            if states_type == 'walk':
                raise ValueError('The "@arg@" parameter, when specified as a string, cannot be associated to a walk.')

            if value not in current_states:
                raise ValueError(f'The "@arg@" parameter, when specified as a string, must match the name of an existing state ({", ".join(current_states)}).')

            value = [current_states.index(value)]

            return value

    try:
        value = _extract(value)
    except Exception as e:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from e

    current_states_length = len(current_states)

    if all(_is_integer(state) for state in value):

        value = [int(state) for state in value]

        if any(state < 0 or state >= current_states_length for state in value):
            raise ValueError(f'The "@arg@" parameter, when specified as a list of integers, must contain only values between 0 and the number of existing states minus one ({current_states_length - 1:d}).')

    elif all(_is_string(state) for state in value):

        value = [current_states.index(state) if state in current_states else -1 for state in value]

        if any(state == -1 for state in value):
            raise ValueError(f'The "@arg@" parameter, when specified as a list of strings, must contain only values matching the names of the existing states ({", ".join(current_states)}).')

    else:

        if not flex:
            raise TypeError('The "@arg@" parameter must be either an array_like object of integers or an array_like object of non-empty strings.')

        raise TypeError('The "@arg@" parameter must be either an integer, a non-empty string, an array_like object of integers or an array_like object of non-empty strings.')

    states_length = len(value)

    if states_type != 'walk' and len(set(value)) < states_length:
        raise ValueError('The "@arg@" parameter must contain only unique values.')

    if states_type == 'regular':

        if states_length < 1 or states_length > current_states_length:
            raise ValueError(f'The "@arg@" parameter must contain a number of elements between 1 and the number of existing states ({current_states_length:d}).')

        value = sorted(value)

    elif states_type == 'subset':

        if states_length < 1 or states_length >= current_states_length:
            raise ValueError(f'The "@arg@" parameter must contain a number of elements between 1 and the number of existing states minus one ({current_states_length - 1:d}).')

        value = sorted(value)

    else:

        if states_length < 2:
            raise ValueError('The "@arg@" parameter must contain at least two elements.')

    return value


def validate_status(value: tany, current_states: tlist_str) -> tarray:

    size = len(current_states)

    if _is_integer(value):

        value = int(value)
        limit = size - 1

        if value < 0 or value > limit:
            raise ValueError(f'The "@arg@" parameter, when specified as an integer, must have a value between 0 and the number of existing states minus one ({limit:d}).')

        result = np.zeros(size, dtype=float)
        result[value] = 1.0

        return result

    if _is_string(value):

        if value not in current_states:
            raise ValueError(f'The "@arg@" parameter, when specified as a string, must match the name of an existing state ({", ".join(current_states)}).')

        value = current_states.index(value)

        result = np.zeros(size, dtype=float)
        result[value] = 1.0

        return result

    try:
        value = _extract_as_numeric(value)
    except Exception as e:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from e

    value = value.astype(float)

    if value.ndim > 2 or (value.ndim == 2 and value.shape[0] != 1):
        raise ValueError('The "@arg@" parameter must be a valid vector.')

    value = np.ravel(value)

    if value.size != size:
        raise ValueError(f'The "@arg@" parameter length must be equal to the number of states ({size:d}).')

    if not all(np.isfinite(x) and np.isreal(x) and 0.0 <= x <= 1.0 for x in np.nditer(value)):
        raise ValueError('The "@arg@" parameter must contain only finite real values between 0 and 1.')

    if not np.isclose(np.sum(value), 1.0):
        raise ValueError('The "@arg@" parameter values must sum to 1.')

    return value


def validate_time_points(value: tany) -> ttimes_in:

    if _is_integer(value):

        value = int(value)

        if value < 0:
            raise ValueError('The "@arg@" parameter, when specified as an integer, must be greater than or equal to 0.')

        return value

    try:
        value = _extract(value)
    except Exception as e:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from e

    if all(_is_integer(time_point) for time_point in value):

        value = [int(time_point) for time_point in value]

        if any(time_point < 0 for time_point in value):
            raise ValueError('The "@arg@" parameter, when specified as a list of integers, must contain only values greater than or equal to 0.')

    else:
        raise TypeError('The "@arg@" parameter must be either an integer or an array_like object of integers.')

    time_points_length = len(value)

    if time_points_length < 1:
        raise ValueError('The "@arg@" parameter must contain at least one element.')

    time_points_unique = len(set(value))

    if time_points_unique < time_points_length:
        raise ValueError('The "@arg@" parameter must contain only unique values.')

    value = sorted(value)

    return value


def validate_transition_function(value: tany) -> ttfunc:

    if value is None or isclass(value) or not callable(value):
        raise TypeError('The "@arg@" parameter must be a callable function or method.')

    sig = signature(value)

    if len(sig.parameters) != 4:
        raise ValueError('The "@arg@" parameter must accept 4 input arguments.')

    valid_parameters = ['x_index', 'x_value', 'y_index', 'y_value']

    if not all(parameter in valid_parameters for parameter in sig.parameters.keys()):
        raise ValueError(f'The "@arg@" parameter must define the following input arguments: {", ".join(valid_parameters)}.')

    # noinspection PyBroadException
    try:
        result = value(1, 1.0, 1, 1.0)
    except Exception as e:  # pragma: no cover
        raise ValueError('The "@arg@" parameter behavior is not compliant.') from e

    if not _is_number(result):
        raise ValueError('The "@arg@" parameter behavior is not compliant.')

    result = float(result)

    if not np.isfinite(result) or not np.isreal(result):
        raise ValueError('The "@arg@" parameter behavior is not compliant.')

    return value


def validate_transition_matrix(value: tany) -> tarray:

    try:
        value = _extract_as_numeric(value)
    except Exception as e:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from e

    value = value.astype(float)

    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] < 2:
        raise ValueError('The "@arg@" parameter must be a 2d square matrix with size greater than or equal to 2.')

    if not all(np.isfinite(x) and np.isreal(x) and 0.0 <= x <= 1.0 for x in np.nditer(value)):
        raise ValueError('The "@arg@" parameter must contain only finite real values between 0 and 1.')

    if not np.allclose(np.sum(value, axis=1), np.ones(value.shape[0], dtype=float)):
        raise ValueError('The "@arg@" parameter rows must sum to 1.')

    return value


def validate_vector(value: tany, vector_type: str, flex: bool, size: oint = None) -> tarray:

    if flex and _is_number(value):

        if vector_type != 'unconstrained':
            raise ValueError('The "@arg@" parameter must be unconstrained.')

        if size is None:
            raise ValueError('The "@arg@" parameter must have a defined size.')

        value = np.repeat(float(value), size)

    else:

        try:
            value = _extract_as_numeric(value)
        except Exception as e:
            raise TypeError('The "@arg@" parameter is null or wrongly typed.') from e

        value = value.astype(float)

        if value.ndim > 2 or (value.ndim == 2 and value.shape[0] != 1):
            raise ValueError('The "@arg@" parameter must be a valid vector.')

        value = np.ravel(value)

        if size is not None and (value.size != size):
            raise ValueError(f'The "@arg@" parameter length must be equal to the number of states ({size:d}).')

    if not all(np.isfinite(x) and np.isreal(x) and 0.0 <= x <= 1.0 for x in np.nditer(value)):
        raise ValueError('The "@arg@" parameter must contain only finite real values between 0 and 1.')

    if vector_type == 'annihilation' and not np.isclose(value[0], 0.0):
        raise ValueError('The "@arg@" parameter must contain a value equal to 0 in the first index.')

    if vector_type == 'creation' and not np.isclose(value[-1], 0.0):
        raise ValueError('The "@arg@" parameter must contain a value equal to 0 in the last index.')

    if vector_type == 'stochastic' and not np.isclose(np.sum(value), 1.0):
        raise ValueError('The "@arg@" parameter values must sum to 1.')

    return value
