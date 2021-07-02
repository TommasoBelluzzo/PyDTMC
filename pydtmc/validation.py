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

# Libraries

import networkx as nx
import numpy as np
import scipy.sparse as spsp

try:
    import pandas as pd
except ImportError:
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
    titerable,
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

def _extract(data: tany) -> tlist_any:

    result = None

    if isinstance(data, list):
        result = deepcopy(data)
    elif isinstance(data, dict):
        result = list(data.values())
    elif isinstance(data, titerable) and not isinstance(data, str):
        result = list(data)

    if result is None:
        raise TypeError('The data type is not supported.')

    return result


def _extract_as_numeric(data: tany) -> tarray:

    result = None

    if isinstance(data, list):
        result = np.array(data)
    elif isinstance(data, dict):
        result = np.array(list(data.values()))
    elif isinstance(data, np.ndarray):
        result = np.copy(data)
    elif isinstance(data, spsp.spmatrix):
        result = np.array(data.todense())
    elif pd is not None and isinstance(data, (pd.DataFrame, pd.Series)):
        result = data.to_numpy(copy=True)
    elif isinstance(data, titerable) and not isinstance(data, str):
        result = np.array(list(data))

    if result is None or not np.issubdtype(result.dtype, np.number):
        raise TypeError('The data type is not supported.')

    return result


def _is_float(value: tany) -> bool:

    return value is not None and isinstance(value, (float, np.floating))


def _is_integer(value: tany) -> bool:

    return value is not None and isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _is_number(value: tany) -> bool:

    return _is_float(value) or _is_integer(value)


def validate_boolean(value: tany) -> bool:

    if isinstance(value, bool):
        return value

    raise TypeError('The "@arg@" parameter must be a boolean value.')


def validate_boundary_condition(value: tany) -> tbcond:

    if _is_number(value):

        value = float(value)

        if (value < 0.0) or (value > 1.0):
            raise ValueError('The "@arg@" parameter, when specified as a number, must have a value between 0 and 1.')

        return value

    if isinstance(value, str):

        possible_values = ['absorbing', 'reflecting']

        if value not in possible_values:
            raise ValueError(f'The "@arg@" parameter, when specified as a string, must have one of the following values: {", ".join(possible_values)}.')

        return value

    raise TypeError('The "@arg@" parameter must be either a float representing the first probability of the semi-reflecting condition or a string representing the boundary condition type.')


def validate_dictionary(value: tany) -> tmc_dict:

    if value is None or not isinstance(value, dict):
        raise ValueError('The "@arg@" parameter must be a dictionary.')

    d_keys = value.keys()

    if not all(isinstance(d_key, tuple) and len(d_key) == 2 and isinstance(d_key[0], str) and len(d_key[0]) > 0 and isinstance(d_key[1], str) and len(d_key[1]) > 0 for d_key in d_keys):
        raise ValueError('The "@arg@" parameter keys must be tuples containing two valid string values.')

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

    if isinstance(value, list):

        value_len = len(value)

        if value_len <= 1:
            raise ValueError('The "@arg@" parameter, when specified as a list of vectors, must contain at least 2 elements.')

        for index, vector in enumerate(value):

            if not isinstance(vector, np.ndarray) or not np.issubdtype(vector.dtype, np.number):
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
        raise TypeError('The "@arg@" parameter must be an integer value.')

    value = int(value)

    possible_values = [75, 100, 150, 200, 300]

    if value not in possible_values:
        possible_values = [str(possible_value) for possible_value in possible_values]
        raise ValueError(f'The "@arg@" parameter must have one of the following values: {", ".join(possible_values)}.')

    return value


def validate_enumerator(value: tany, possible_values: tlist_str) -> str:

    if not all(isinstance(possible_value, str) and len(possible_value) > 0 for possible_value in possible_values):
        raise ValueError('The list of possible enumerator values must contain only non-empty strings.')

    if not isinstance(value, str):
        raise TypeError('The "@arg@" parameter must be a string value.')

    if value not in possible_values:
        raise ValueError(f'The "@arg@" parameter value must be one of the following: {", ".join(possible_values)}.')

    return value


def validate_file_path(value: tany, write_permission: bool) -> str:  # pragma: no cover

    if not isinstance(value, str):
        raise TypeError('The "@arg@" parameter must be a string value.')

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

    if value is None or not isinstance(value, (float, np.floating)):
        raise TypeError('The "@arg@" parameter must be a float value.')

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

    non_multi = value is not None and isinstance(value, nx.DiGraph)
    multi = value is not None and isinstance(value, nx.MultiDiGraph)

    if not non_multi and not multi:
        raise ValueError('The "@arg@" parameter must be a directed graph.')

    if multi:
        value = nx.DiGraph(value)

    nodes = list(value.nodes)
    nodes_length = len(nodes)

    if nodes_length < 2:
        raise ValueError('The "@arg@" parameter must contain a number of nodes greater than or equal to 2.')

    if not all(isinstance(node, str) and len(node) > 0 for node in nodes):
        raise ValueError('The "@arg@" parameter must define node labels as non-empty strings.')

    edges = list(value.edges(data='weight', default=0.0))

    if not all(_is_number(edge[2]) and float(edge[2]) > 0.0 for edge in edges):
        raise ValueError('The "@arg@" parameter must define edge wright as non-negative numbers.')

    return value


def validate_hyperparameter(hyperparameter: tany, size: int) -> tarray:

    try:
        hyperparameter = _extract_as_numeric(hyperparameter)
    except Exception as e:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from e

    hyperparameter = hyperparameter.astype(float)

    if hyperparameter.ndim != 2 or hyperparameter.shape[0] != hyperparameter.shape[1] or hyperparameter.shape[0] != size:
        raise ValueError(f'The "@arg@" parameter must be a 2d square matrix with size equal to {size:d}.')

    if not all(np.isfinite(x) and np.isreal(x) and np.equal(np.mod(x, 1.0), 0.0) and x >= 1.0 for x in np.nditer(hyperparameter)):
        raise ValueError('The "@arg@" parameter must contain only integer values greater than or equal to 1.')

    return hyperparameter


def validate_integer(value: tany, lower_limit: olimit_int = None, upper_limit: olimit_int = None) -> int:

    if value is None or not _is_integer(value):
        raise TypeError('The "@arg@" parameter must be an integer value.')

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

    if not isinstance(value, tuple):
        raise TypeError('The "@arg@" parameter must be a tuple.')

    if len(value) != 2:
        raise ValueError('The "@arg@" parameter must contain 2 elements.')

    a = value[0]
    b = value[1]

    if not _is_number(a) or not _is_number(b):
        raise ValueError('The "@arg@" parameter must contain only float and integer values.')

    a = float(a)
    b = float(b)

    if not all(np.isfinite(x) and np.isreal(x) and x >= 0.0 for x in [a, b]):
        raise ValueError('The "@arg@" parameter must contain only finite real values greater than or equal to 0.0.')

    if a >= b:
        raise ValueError('The "@arg@" parameter must contain two distinct values, and the first value must be less than the second one.')

    return a, b


def validate_markov_chain(mc: tany) -> tmc:

    if mc is None or (f'{mc.__module__}.{mc.__class__.__name__}' != 'pydtmc.markov_chain.MarkovChain'):
        raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    return mc


def validate_mask(mask: tany, size: int) -> tarray:

    try:
        mask = _extract_as_numeric(mask)
    except Exception as e:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from e

    mask = mask.astype(float)

    if mask.ndim != 2 or mask.shape[0] != mask.shape[1] or mask.shape[0] != size:
        raise ValueError(f'The "@arg@" parameter must be a 2d square matrix with size equal to {size:d}.')

    if not all(np.isnan(x) or (np.isfinite(x) and np.isreal(x) and 0.0 <= x <= 1.0) for x in np.nditer(mask)):
        raise ValueError('The "@arg@" parameter can contain only NaNs and finite real values between 0 and 1.')

    if np.any(np.nansum(mask, axis=1, dtype=float) > 1.0):
        raise ValueError('The "@arg@" parameter row sums must not exceed 1.')

    return mask


def validate_matrix(m: tany) -> tarray:

    try:
        m = _extract_as_numeric(m)
    except Exception as e:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from e

    m = m.astype(float)

    if m.ndim != 2 or m.shape[0] < 2 or m.shape[0] != m.shape[1]:
        raise ValueError('The "@arg@" parameter must be a 2d square matrix with size greater than or equal to 2.')

    if not all(np.isfinite(x) and np.isreal(x) and x >= 0.0 for x in np.nditer(m)):
        raise ValueError('The "@arg@" parameter must contain only finite real values greater than or equal to 0.0.')

    return m


def validate_partitions(value: tany, current_states: tlist_str) -> tlists_int:

    if value is None or not isinstance(value, list):
        raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    partitions_length = len(value)
    current_states_length = len(current_states)

    if partitions_length < 2 or partitions_length >= current_states_length:
        raise ValueError(f'The "@arg@" parameter must contain a number of elements between 0 and the number of existing states minus one ({current_states_length - 1:d}).')

    partitions_flat = []
    partitions_groups = []

    for partition in value:

        if not isinstance(partition, titerable):
            raise TypeError('The "@arg@" parameter must contain only array_like objects.')

        partition_list = list(partition)

        partitions_flat.extend(partition_list)
        partitions_groups.append(len(partition_list))

    if all(_is_integer(state) for state in partitions_flat):

        partitions_flat = [int(state) for state in partitions_flat]

        if any(state < 0 or state >= current_states_length for state in partitions_flat):
            raise ValueError(f'The "@arg@" parameter subelements, when specified as integers, must be values between 0 and the number of existing states minus one ({current_states_length - 1:d}).')

    elif all(isinstance(partition_flat, str) for partition_flat in partitions_flat):

        partitions_flat = [current_states.index(s) if s in current_states else -1 for s in partitions_flat]

        if any(s == -1 for s in partitions_flat):
            raise ValueError(f'The "@arg@" parameter subelements, when specified as strings, must contain only values matching the names of the existing states ({", ".join(current_states)}).')

    else:
        raise TypeError('The "@arg@" parameter must contain only array_like objects of integers or array_like objects of strings.')

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


def validate_rewards(rewards: tany, size: int) -> tarray:

    try:
        rewards = _extract_as_numeric(rewards)
    except Exception as e:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from e

    rewards = rewards.astype(float)

    if (rewards.ndim > 2) or (rewards.ndim == 2 and rewards.shape[0] != 1):
        raise ValueError('The "@arg@" parameter must be a valid vector.')

    rewards = np.ravel(rewards)

    if rewards.size != size:
        raise ValueError(f'The "@arg@" parameter length must be equal to the number of states ({size:d}).')

    if not all(np.isfinite(x) and np.isreal(x) for x in np.nditer(rewards)):
        raise ValueError('The "@arg@" parameter must contain only finite real values.')

    return rewards


def validate_state(value: tany, current_states: tlist_str) -> int:

    if _is_integer(value):

        state = int(value)
        limit = len(current_states) - 1

        if state < 0 or state > limit:
            raise ValueError(f'The "@arg@" parameter, when specified as an integer, must have a value between 0 and the number of existing states minus one ({limit:d}).')

        return state

    if isinstance(value, str):

        if value not in current_states:
            raise ValueError(f'The "@arg@" parameter, when specified as a string, must match the name of an existing state ({", ".join(current_states)}).')

        return current_states.index(value)

    raise TypeError('The "@arg@" parameter must be either an integer or a string.')


def validate_status(status: tany, current_states: tlist_str) -> tarray:

    size = len(current_states)

    if _is_integer(status):

        status = int(status)
        limit = size - 1

        if status < 0 or status > limit:
            raise ValueError(f'The "@arg@" parameter, when specified as an integer, must have a value between 0 and the number of existing states minus one ({limit:d}).')

        result = np.zeros(size, dtype=float)
        result[status] = 1.0

        return result

    if isinstance(status, str):

        if status not in current_states:
            raise ValueError(f'The "@arg@" parameter, when specified as a string, must match the name of an existing state ({", ".join(current_states)}).')

        status = current_states.index(status)

        result = np.zeros(size, dtype=float)
        result[status] = 1.0

        return result

    try:
        status = _extract_as_numeric(status)
    except Exception as e:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from e

    status = status.astype(float)

    if status.ndim > 2 or (status.ndim == 2 and status.shape[0] != 1):
        raise ValueError('The "@arg@" parameter must be a valid vector.')

    status = np.ravel(status)

    if status.size != size:
        raise ValueError(f'The "@arg@" parameter length must be equal to the number of states ({size:d}).')

    if not all(np.isfinite(x) and np.isreal(x) and 0.0 <= x <= 1.0 for x in np.nditer(status)):
        raise ValueError('The "@arg@" parameter must contain only finite real values between 0 and 1.')

    if not np.isclose(np.sum(status), 1.0):
        raise ValueError('The "@arg@" parameter values must sum to 1.')

    return status


def validate_state_names(states: tany, size: oint = None) -> tlist_str:

    try:
        states = _extract(states)
    except Exception as e:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from e

    if not all(state is not None and isinstance(state, str) and (len(state) > 0) for state in states):
        raise TypeError('The "@arg@" parameter must contain only valid string values.')

    states_length = len(states)

    if states_length < 2:
        raise ValueError('The "@arg@" parameter must contain at least two elements.')

    states_unique = len(set(states))

    if states_unique < states_length:
        raise ValueError('The "@arg@" parameter must contain only unique values.')

    if size is not None and states_length != size:
        raise ValueError(f'The "@arg@" parameter must contain a number of elements equal to {size:d}.')

    return states


def validate_states(states: tany, current_states: tlist_str, state_type: str, flex: bool) -> tlist_int:

    if flex:

        if _is_integer(states):

            states = int(states)
            limit = len(current_states) - 1

            if states < 0 or states > limit:
                raise ValueError(f'The "@arg@" parameter, when specified as an integer, must have a value between 0 and the number of existing states minus one ({limit:d}).')

            return [states]

        if state_type != 'walk' and isinstance(states, str):

            if states not in current_states:
                raise ValueError(f'The "@arg@" parameter, when specified as a string, must match the name of an existing state ({", ".join(current_states)}).')

            return [current_states.index(states)]

    try:
        states = _extract(states)
    except Exception as e:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from e

    current_states_length = len(current_states)

    if all(_is_integer(state) for state in states):

        states = [int(state) for state in states]

        if any(state < 0 or state >= current_states_length for state in states):
            raise ValueError(f'The "@arg@" parameter, when specified as a list of integers, must contain only values between 0 and the number of existing states minus one ({current_states_length - 1:d}).')

    elif all(isinstance(state, str) for state in states):

        states = [current_states.index(s) if s in current_states else -1 for s in states]

        if any(s == -1 for s in states):
            raise ValueError(f'The "@arg@" parameter, when specified as a list of strings, must contain only values matching the names of the existing states ({", ".join(current_states)}).')

    else:

        if not flex:
            raise TypeError('The "@arg@" parameter must be either an array_like object of integers or an array_like object of strings.')

        if state_type == 'walk':
            raise TypeError('The "@arg@" parameter must be either an integer, an array_like object of integers or an array_like object of strings.')

        raise TypeError('The "@arg@" parameter must be either an integer, a string, an array_like object of integers or an array_like object of strings.')

    states_length = len(states)

    if state_type != 'walk' and len(set(states)) < states_length:
        raise ValueError('The "@arg@" parameter must contain only unique values.')

    if state_type == 'regular':

        if states_length < 1 or states_length > current_states_length:
            raise ValueError(f'The "@arg@" parameter must contain a number of elements between 1 and the number of existing states ({current_states_length:d}).')

        states = sorted(states)

    elif state_type == 'subset':

        if states_length < 1 or states_length >= current_states_length:
            raise ValueError(f'The "@arg@" parameter must contain a number of elements between 1 and the number of existing states minus one ({current_states_length - 1:d}).')

        states = sorted(states)

    else:

        if states_length < 2:
            raise ValueError('The "@arg@" parameter must contain at least two elements.')

    return states


def validate_time_points(time_points: tany) -> ttimes_in:

    if _is_integer(time_points):

        time_points = int(time_points)

        if time_points < 0:
            raise ValueError('The "@arg@" parameter, when specified as an integer, must be greater than or equal to 0.')

        return time_points

    try:
        time_points = _extract(time_points)
    except Exception as e:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from e

    if all(_is_integer(time_point) for time_point in time_points):

        time_points = [int(time_point) for time_point in time_points]

        if any(time_point < 0 for time_point in time_points):
            raise ValueError('The "@arg@" parameter, when specified as a list of integers, must contain only values greater than or equal to 0.')

    else:
        raise TypeError('The "@arg@" parameter must be either an integer or an array_like object of integers.')

    time_points_length = len(time_points)

    if time_points_length < 1:
        raise ValueError('The "@arg@" parameter must contain at least one element.')

    time_points_unique = len(set(time_points))

    if time_points_unique < time_points_length:
        raise ValueError('The "@arg@" parameter must contain only unique values.')

    time_points = sorted(time_points)

    return time_points


def validate_transition_function(f: tany) -> ttfunc:

    if f is None or isclass(f) or not callable(f):
        raise TypeError('The "@arg@" parameter must be a callable function or method.')

    s = signature(f)

    if len(s.parameters) != 4:
        raise ValueError('The "@arg@" parameter must accept 4 input arguments.')

    valid_parameters = ['x_index', 'x_value', 'y_index', 'y_value']

    if not all(parameter in valid_parameters for parameter in s.parameters.keys()):
        raise ValueError(f'The "@arg@" parameter must define the following input arguments: {", ".join(valid_parameters)}.')

    # noinspection PyBroadException
    try:
        result = f(1, 1.0, 1, 1.0)
    except Exception as e:  # pragma: no cover
        raise ValueError('The "@arg@" parameter behavior is not compliant.') from e

    if not _is_number(result):
        raise ValueError('The "@arg@" parameter behavior is not compliant.')

    result = float(result)

    if not np.isfinite(result) or not np.isreal(result):
        raise ValueError('The "@arg@" parameter behavior is not compliant.')

    return f


def validate_transition_matrix(p: tany) -> tarray:

    try:
        p = _extract_as_numeric(p)
    except Exception as e:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from e

    p = p.astype(float)

    if p.ndim != 2 or p.shape[0] != p.shape[1] or p.shape[0] < 2:
        raise ValueError('The "@arg@" parameter must be a 2d square matrix with size greater than or equal to 2.')

    if not all(np.isfinite(x) and np.isreal(x) and 0.0 <= x <= 1.0 for x in np.nditer(p)):
        raise ValueError('The "@arg@" parameter must contain only finite real values between 0 and 1.')

    if not np.allclose(np.sum(p, axis=1), np.ones(p.shape[0], dtype=float)):
        raise ValueError('The "@arg@" parameter rows must sum to 1.')

    return p


def validate_vector(vector: tany, vector_type: str, flex: bool, size: oint = None) -> tarray:

    if flex and size is not None and _is_number(vector):
        vector = np.repeat(float(vector), size)
    else:

        try:
            vector = _extract_as_numeric(vector)
        except Exception as e:
            raise TypeError('The "@arg@" parameter is null or wrongly typed.') from e

    vector = vector.astype(float)

    if vector.ndim > 2 or (vector.ndim == 2 and vector.shape[0] != 1):
        raise ValueError('The "@arg@" parameter must be a valid vector.')

    vector = np.ravel(vector)

    if size is not None and (vector.size != size):
        raise ValueError(f'The "@arg@" parameter length must be equal to the number of states ({size:d}).')

    if not all(np.isfinite(x) and np.isreal(x) and 0.0 <= x <= 1.0 for x in np.nditer(vector)):
        raise ValueError('The "@arg@" parameter must contain only finite real values between 0 and 1.')

    if vector_type == 'annihilation' and not np.isclose(vector[0], 0.0):
        raise ValueError('The "@arg@" parameter must contain a value equal to 0 in the first index.')

    if vector_type == 'creation' and not np.isclose(vector[-1], 0.0):
        raise ValueError('The "@arg@" parameter must contain a value equal to 0 in the last index.')

    if vector_type == 'stochastic' and not np.isclose(np.sum(vector), 1.0):
        raise ValueError('The "@arg@" parameter values must sum to 1.')

    return vector
