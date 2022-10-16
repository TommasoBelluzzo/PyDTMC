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
    'validate_markov_chains',
    'validate_mask',
    'validate_matrix',
    'validate_partitions',
    'validate_random_distribution',
    'validate_rewards',
    'validate_state',
    'validate_state_names',
    'validate_states',
    'validate_status',
    'validate_strings',
    'validate_time_points',
    'validate_transition_function',
    'validate_transition_matrix',
    'validate_vector',
    'validate_walk',
    'validate_walks'
]


###########
# IMPORTS #
###########

# Standard

from inspect import (
    isclass as _ins_isclass,
    signature as _ins_signature
)

from itertools import (
    product as _it_product
)

from os.path import (
    isfile as _osp_isfile
)

# Libraries

from networkx import (
    DiGraph as _nx_DiGraph
)

from numpy import (
    allclose as _np_allclose,
    any as _np_any,
    equal as _np_equal,
    isclose as _np_isclose,
    isfinite as _np_isfinite,
    isnan as _np_isnan,
    isreal as _np_isreal,
    issubdtype as _np_issubdtype,
    mod as _np_mod,
    ndenumerate as _np_ndenumerate,
    nansum as _np_nansum,
    number as _np_number,
    ones as _np_ones,
    ravel as _np_ravel,
    repeat as _np_repeat,
    sum as _np_sum,
    zeros as _np_zeros
)

try:
    from pandas import (
        DataFrame as _pd_DataFrame,
        Series as _pd_Series
    )
    _pandas_found = True
except ImportError:  # pragma: no cover
    _pd_DataFrame = None
    _pd_Series = None
    _pandas_found = False

# Internal

from .custom_types import (
    oint as _oint,
    olimit_float as _olimit_float,
    olimit_int as _olimit_int,
    olimit_scalar as _olimit_scalar,
    olist_str as _olist_str,
    tany as _tany,
    tarray as _tarray,
    tbcond as _tbcond,
    tdists_flex as _tdists_flex,
    tfile as _tfile,
    tgraphs as _tgraphs,
    tinterval as _tinterval,
    tlist_int as _tlist_int,
    tlist_str as _tlist_str,
    tlists_int as _tlists_int,
    tmc as _tmc,
    tmc_dict as _tmc_dict,
    trand as _trand,
    trandfunc as _trandfunc,
    tscalar as _tscalar,
    ttfunc as _ttfunc,
    ttimes_in as _ttimes_in,
    tvalid_walk as _tvalid_walk,
    tvalid_walks as _tvalid_walks
)

from .utilities import (
    extract_data_generic as _extract_data_generic,
    extract_data_numeric as _extract_data_numeric,
    get_file_extension as _get_file_extension,
    is_array as _is_array,
    is_bool as _is_bool,
    is_dictionary as _is_dictionary,
    is_float as _is_float,
    is_graph as _is_graph,
    is_integer as _is_integer,
    is_iterable as _is_iterable,
    is_list as _is_list,
    is_number as _is_number,
    is_string as _is_string,
    is_tuple as _is_tuple
)


#############
# FUNCTIONS #
#############

def _extract_numeric_matrix(data: _tany, size: int) -> _tarray:

    try:
        result = _extract_data_numeric(data)
    except Exception as e:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from e

    result = result.astype(float)

    if result.ndim != 2 or result.shape[0] != result.shape[1] or result.shape[0] != size:
        raise ValueError(f'The "@arg@" parameter must be a 2d square matrix with size equal to {size:d}.')

    return result


def _extract_numeric_vector(data: _tany, size: _oint) -> _tarray:

    try:
        result = _extract_data_numeric(data)
    except Exception as e:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from e

    result = result.astype(float)

    if result.ndim > 2 or (result.ndim == 2 and result.shape[0] != 1) or (result.ndim == 1 and result.shape[0] == 0):
        raise ValueError('The "@arg@" parameter must be a valid vector.')

    result = _np_ravel(result)

    if size is not None and result.size != size:
        raise ValueError(f'The "@arg@" parameter length must be equal to the number of states ({size:d}).')

    return result


def _validate_limits(value: _tscalar, value_type: str, lower_limit: _olimit_scalar, upper_limit: _olimit_scalar):

    def _get_limit_text(gly_value_type, glt_limit_value):
        return f'{glt_limit_value:d}' if gly_value_type == 'integer' else f'{glt_limit_value:f}'

    if lower_limit is not None:

        lower_limit_value = lower_limit[0]
        lower_limit_included = lower_limit[1]

        if lower_limit_included and value <= lower_limit_value:
            raise ValueError(f'The "@arg@" parameter must be greater than {_get_limit_text(value_type, lower_limit_value)}.')

        if not lower_limit_included and value < lower_limit_value:
            raise ValueError(f'The "@arg@" parameter must be greater than or equal to {_get_limit_text(value_type, lower_limit_value)}.')

    if upper_limit is not None:

        upper_limit_value = upper_limit[0]
        upper_limit_included = upper_limit[1]

        if upper_limit_included and value >= upper_limit_value:
            raise ValueError(f'The "@arg@" parameter must be less than {_get_limit_text(value_type, upper_limit_value)}.')

        if not upper_limit_included and value > upper_limit_value:
            raise ValueError(f'The "@arg@" parameter must be less than or equal to {_get_limit_text(value_type, upper_limit_value)}.')


def validate_boolean(value: _tany) -> bool:

    if not _is_bool(value):
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


def validate_dictionary(value: _tany) -> _tmc_dict:

    if not _is_dictionary(value):
        raise ValueError('The "@arg@" parameter must be a dictionary.')

    d_keys = value.keys()

    if not all(_is_tuple(d_key) and len(d_key) == 2 and _is_string(d_key[0]) and _is_string(d_key[1]) for d_key in d_keys):
        raise ValueError('The "@arg@" parameter keys must be tuples containing two non-empty strings.')

    states = set()

    for d_key in d_keys:
        states.add(d_key[0])
        states.add(d_key[1])

    combinations = list(_it_product(states, repeat=2))

    if len(value) != len(combinations) or not all(combination in value for combination in combinations):
        raise ValueError('The "@arg@" parameter keys must contain all the possible combinations of states.')

    d_values = value.values()

    if not all(_is_number(d_value) for d_value in d_values):
        raise ValueError('The "@arg@" parameter values must be float or integer numbers.')

    result = {}

    for d_key, d_value in value.items():

        d_value = float(d_value)

        if _np_isfinite(d_value) and _np_isreal(d_value) and 0.0 <= d_value <= 1.0:
            result[d_key] = d_value
        else:
            raise ValueError('The "@arg@" parameter values can contain only finite real numbers between 0 and 1.')

    return result


def validate_distribution(value: _tany, size: int) -> _tdists_flex:

    if _is_integer(value):

        value = int(value)

        if value <= 0:
            raise ValueError('The "@arg@" parameter, when specified as an integer, must be greater than or equal to 1.')

        return value

    if _is_list(value):

        value_length = len(value)

        if value_length <= 1:
            raise ValueError('The "@arg@" parameter, when specified as a list of vectors, must contain at least 2 elements.')

        for index, vector in enumerate(value):

            if not _is_array(vector) or not _np_issubdtype(vector.dtype, _np_number):
                raise TypeError('The "@arg@" parameter must contain only numeric vectors.')

            vector = vector.astype(float)
            value[index] = vector

            if vector.ndim != 1 or vector.size != size:
                raise ValueError('The "@arg@" parameter must contain only vectors of size {size:d}.')

            if not all(_np_isfinite(x) and _np_isreal(x) and 0.0 <= x <= 1.0 for _, x in _np_ndenumerate(vector)):
                raise ValueError('The "@arg@" parameter must contain only vectors consisting of finite real values between 0 and 1.')

            if not _np_isclose(_np_sum(vector), 1.0):
                raise ValueError('The "@arg@" parameter must contain only vectors consisting of values whose sum is 1.')

        return value

    raise TypeError('The "@arg@" parameter must be either an integer representing the number of redistributions to perform or a list of valid distributions.')


def validate_dpi(value: _tany) -> int:

    if not _is_integer(value):
        raise TypeError('The "@arg@" parameter must be an integer.')

    value = int(value)

    possible_values = (75, 100, 150, 200, 300)

    if value not in possible_values:
        possible_values = [str(possible_value) for possible_value in possible_values]
        raise ValueError(f'The "@arg@" parameter must have one of the following values: {", ".join(possible_values)}.')

    return value


def validate_enumerator(value: _tany, possible_values: _tlist_str) -> str:

    if not all(_is_string(possible_value) for possible_value in possible_values):
        raise ValueError('The list of possible enumerator values must contain only non-empty strings.')

    if not _is_string(value):
        raise TypeError('The "@arg@" parameter must be a non-empty string.')

    if value not in possible_values:
        raise ValueError(f'The "@arg@" parameter value must be one of the following: {", ".join(possible_values)}.')

    return value


def validate_file_path(value: _tany, accepted_extensions: _olist_str, write_permission: bool) -> _tfile:

    if not _is_string(value):
        raise TypeError('The "@arg@" parameter must be a non-empty string.')

    if len(value.strip()) == 0:
        raise ValueError('The "@arg@" parameter must not be a non-empty string.')

    file_path = value

    if not _osp_isfile(file_path):
        raise ValueError('The "@arg@" parameter defines an invalid file path.')

    try:
        file_extension = _get_file_extension(file_path)
    except Exception as e:  # pragma: no cover
        raise ValueError('The "@arg@" parameter defines an invalid file path.') from e

    if accepted_extensions is not None and len(accepted_extensions) > 0 and file_extension not in accepted_extensions:
        raise ValueError(f'The "@arg@" parameter must have one of the following extensions: {", ".join(sorted(accepted_extensions)).replace(".", "")}.')

    if write_permission:

        try:

            with open(file_path, mode='w'):
                pass

        except Exception as e:  # pragma: no cover
            raise ValueError('The "@arg@" parameter defines the path to an inaccessible file.') from e

    else:

        file_empty = False

        try:

            with open(file_path, mode='r') as file:

                file.seek(0)

                if not file.read(1):
                    file_empty = True

        except Exception as e:  # pragma: no cover
            raise ValueError('The "@arg@" parameter defines the path to an inaccessible file.') from e

        if file_empty:
            raise ValueError('The "@arg@" parameter defines the path to an empty file.')

    value = (file_path, file_extension)

    return value


def validate_float(value: _tany, lower_limit: _olimit_float = None, upper_limit: _olimit_float = None) -> float:

    if not _is_float(value):
        raise TypeError('The "@arg@" parameter must be a float.')

    value = float(value)

    if not _np_isfinite(value) or not _np_isreal(value):
        raise ValueError('The "@arg@" parameter be a finite real value.')

    _validate_limits(value, 'float', lower_limit, upper_limit)

    return value


def validate_graph(value: _tany) -> _tgraphs:

    non_multi = _is_graph(value, False)
    multi = _is_graph(value, True)

    if not non_multi and not multi:
        raise ValueError('The "@arg@" parameter must be a directed graph.')

    if multi:
        value = _nx_DiGraph(value)

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


def validate_hyperparameter(value: _tany, size: int) -> _tarray:

    value = _extract_numeric_matrix(value, size)

    if not all(_np_isfinite(x) and _np_isreal(x) and _np_equal(_np_mod(x, 1.0), 0.0) and x >= 1.0 for _, x in _np_ndenumerate(value)):
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

    a = value[0]
    b = value[1]

    if not _is_number(a) or not _is_number(b):
        raise ValueError('The "@arg@" parameter must contain only floats and integers.')

    a = float(a)
    b = float(b)

    if not all(_np_isfinite(x) and _np_isreal(x) and x >= 0.0 for x in [a, b]):
        raise ValueError('The "@arg@" parameter must contain only finite real values greater than or equal to 0.0.')

    if a >= b:
        raise ValueError('The "@arg@" parameter must contain two distinct values, and the first value must be less than the second one.')

    return a, b


def validate_markov_chain(value: _tany) -> _tmc:

    if value is None or (f'{value.__module__}.{value.__class__.__name__}' != 'pydtmc.markov_chain.MarkovChain'):
        raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    return value


def validate_markov_chains(value: _tany) -> _tmc:

    if not _is_list(value):
        raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    value_length = len(value)

    if value_length < 2:
        raise ValueError('The "@arg@" parameter must contain at least two elements.')

    for i in range(value_length):
        try:
            value[i] = validate_markov_chain(value[i])
        except Exception as e:
            raise ValueError('The "@arg@" parameter contains invalid elements.') from e

    return value


def validate_mask(value: _tany, size: int) -> _tarray:

    value = _extract_numeric_matrix(value, size)

    if not all(_np_isnan(x) or (_np_isfinite(x) and _np_isreal(x) and 0.0 <= x <= 1.0) for _, x in _np_ndenumerate(value)):
        raise ValueError('The "@arg@" parameter can contain only NaNs and finite real values between 0 and 1.')

    if _np_any(_np_nansum(value, axis=1, dtype=float) > 1.0):
        raise ValueError('The "@arg@" parameter row sums must not exceed 1.')

    return value


def validate_matrix(value: _tany) -> _tarray:

    try:
        value = _extract_data_numeric(value)
    except Exception as e:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from e

    value = value.astype(float)

    if value.ndim != 2 or value.shape[0] < 2 or value.shape[0] != value.shape[1]:
        raise ValueError('The "@arg@" parameter must be a 2d square matrix with size greater than or equal to 2.')

    if not all(_np_isfinite(x) and _np_isreal(x) and x >= 0.0 for _, x in _np_ndenumerate(value)):
        raise ValueError('The "@arg@" parameter must contain only finite real values greater than or equal to 0.0.')

    return value


def validate_partitions(value: _tany, current_states: _tlist_str) -> _tlists_int:

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

    value = _extract_numeric_vector(value, size)

    if not all(_np_isfinite(x) and _np_isreal(x) for _, x in _np_ndenumerate(value)):
        raise ValueError('The "@arg@" parameter must contain only finite real values.')

    return value


def validate_state(value: _tany, current_states: _tlist_str) -> int:

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


def validate_state_names(value: _tany, size: _oint = None) -> _tlist_str:

    try:
        value = _extract_data_generic(value)
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


def validate_states(value: _tany, possible_states: _tlist_str, subset: bool) -> _tlist_int:

    if _is_integer(value):

        value = int(value)
        limit = len(possible_states) - 1

        if value < 0 or value > limit:
            raise ValueError(f'The "@arg@" parameter, when specified as an integer, must have a value between 0 and the number of existing states minus one ({limit:d}).')

        value = [value]

        return value

    if _is_string(value):

        if value not in possible_states:
            raise ValueError(f'The "@arg@" parameter, when specified as a string, must match the name of an existing state ({", ".join(possible_states)}).')

        value = [possible_states.index(value)]

        return value

    try:
        value = _extract_data_generic(value)
    except Exception as e:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from e

    if all(_is_integer(state) for state in value):
        value_type = 'integer'
    elif all(_is_string(state) for state in value):
        value_type = 'string'
    else:
        raise TypeError('The "@arg@" parameter must be either an integer, a non-empty string, an array_like object of integers or an array_like object of non-empty strings.')

    if value_type == 'integer':

        value = [int(state) for state in value]

        try:
            possible_states = validate_state_names(possible_states)
        except Exception as e:
            raise ValueError(
                'The "@arg@" parameter must be validated against a coherent list of possible states.') from e

        possible_states_length = len(possible_states)

        if any(state < 0 or state >= possible_states_length for state in value):
            raise ValueError(f'The "@arg@" parameter, when specified as a list of integers, must contain only values between 0 and the number of existing states minus one ({possible_states_length - 1:d}).')

    else:

        try:
            possible_states = validate_state_names(possible_states)
        except Exception as e:
            raise ValueError(
                'The "@arg@" parameter must be validated against a coherent list of possible states.') from e

        possible_states_length = len(possible_states)

        value = [possible_states.index(state) if state in possible_states else -1 for state in value]

        if any(state == -1 for state in value):
            raise ValueError(f'The "@arg@" parameter, when specified as a list of strings, must contain only values matching the names of the existing states ({", ".join(possible_states)}).')

    value_length = len(value)

    if len(set(value)) < value_length:
        raise ValueError('The "@arg@" parameter must contain only unique values.')

    if subset:
        if value_length < 1 or value_length >= possible_states_length:
            raise ValueError(f'The "@arg@" parameter must contain a number of elements between 1 and the number of existing states minus one ({possible_states_length - 1:d}).')
    else:
        if value_length < 1 or value_length > possible_states_length:
            raise ValueError(f'The "@arg@" parameter must contain a number of elements between 1 and the number of existing states ({possible_states_length:d}).')

    value = sorted(value)

    return value


def validate_status(value: _tany, current_states: _tlist_str) -> _tarray:

    size = len(current_states)

    if _is_integer(value):

        value = int(value)
        limit = size - 1

        if value < 0 or value > limit:
            raise ValueError(f'The "@arg@" parameter, when specified as an integer, must have a value between 0 and the number of existing states minus one ({limit:d}).')

        result = _np_zeros(size, dtype=float)
        result[value] = 1.0

        return result

    if _is_string(value):

        if value not in current_states:
            raise ValueError(f'The "@arg@" parameter, when specified as a string, must match the name of an existing state ({", ".join(current_states)}).')

        value = current_states.index(value)

        result = _np_zeros(size, dtype=float)
        result[value] = 1.0

        return result

    try:
        value = _extract_data_numeric(value)
    except Exception as e:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from e

    value = value.astype(float)

    if value.ndim > 2 or (value.ndim == 2 and value.shape[0] != 1):
        raise ValueError('The "@arg@" parameter must be a valid vector.')

    value = _np_ravel(value)

    if value.size != size:
        raise ValueError(f'The "@arg@" parameter length must be equal to the number of states ({size:d}).')

    if not all(_np_isfinite(x) and _np_isreal(x) and 0.0 <= x <= 1.0 for _, x in _np_ndenumerate(value)):
        raise ValueError('The "@arg@" parameter must contain only finite real values between 0 and 1.')

    if not _np_isclose(_np_sum(value), 1.0):
        raise ValueError('The "@arg@" parameter values must sum to 1.')

    return value


def validate_strings(value: _tany, size: _oint = None) -> _tlist_str:

    try:
        value = _extract_data_generic(value)
    except Exception as e:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from e

    if not all(_is_string(s) for s in value):
        raise TypeError('The "@arg@" parameter must contain only non-empty strings.')

    value_length = len(value)

    if value_length == 0:
        raise ValueError('The "@arg@" parameter must contain at least one element.')

    if size is not None and value_length != size:
        raise ValueError(f'The "@arg@" parameter must contain a number of elements equal to {size:d}.')

    return value


def validate_time_points(value: _tany) -> _ttimes_in:

    if _is_integer(value):

        value = int(value)

        if value < 0:
            raise ValueError('The "@arg@" parameter, when specified as an integer, must be greater than or equal to 0.')

        return value

    try:
        value = _extract_data_generic(value)
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


# noinspection PyBroadException
def validate_transition_function(value: _tany) -> _ttfunc:

    if value is None or _ins_isclass(value) or not callable(value):
        raise TypeError('The "@arg@" parameter must be a callable function or method.')

    sig = _ins_signature(value)

    if len(sig.parameters) != 4:
        raise ValueError('The "@arg@" parameter must accept 4 input arguments.')

    valid_parameters = ('x_index', 'x_value', 'y_index', 'y_value')

    if not all(parameter in valid_parameters for parameter in sig.parameters.keys()):
        raise ValueError(f'The "@arg@" parameter must define the following input arguments: {", ".join(valid_parameters)}.')

    try:
        result = value(1, 1.0, 1, 1.0)
    except Exception as e:  # pragma: no cover
        raise ValueError('The "@arg@" parameter behavior is not compliant.') from e

    if not _is_number(result):
        raise ValueError('The "@arg@" parameter behavior is not compliant.')

    result = float(result)

    if not _np_isfinite(result) or not _np_isreal(result):
        raise ValueError('The "@arg@" parameter behavior is not compliant.')

    return value


def validate_transition_matrix(value: _tany) -> _tarray:

    try:
        value = _extract_data_numeric(value)
    except Exception as e:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from e

    value = value.astype(float)

    if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] < 2:
        raise ValueError('The "@arg@" parameter must be a 2d square matrix with size greater than or equal to 2.')

    if not all(_np_isfinite(x) and _np_isreal(x) and 0.0 <= x <= 1.0 for _, x in _np_ndenumerate(value)):
        raise ValueError('The "@arg@" parameter must contain only finite real values between 0 and 1.')

    if not _np_allclose(_np_sum(value, axis=1), _np_ones(value.shape[0], dtype=float)):
        raise ValueError('The "@arg@" parameter rows must sum to 1.')

    return value


def validate_vector(value: _tany, vector_type: str, flex: bool, size: _oint = None) -> _tarray:

    if flex and _is_number(value):

        if vector_type != 'unconstrained':
            raise ValueError('The "@arg@" parameter must be unconstrained.')

        if size is None:
            raise ValueError('The "@arg@" parameter must have a defined size.')

        value = _np_repeat(float(value), size)

    else:

        value = _extract_numeric_vector(value, size)

    if not all(_np_isfinite(x) and _np_isreal(x) and 0.0 <= x <= 1.0 for _, x in _np_ndenumerate(value)):
        raise ValueError('The "@arg@" parameter must contain only finite real values between 0 and 1.')

    if vector_type == 'annihilation' and not _np_isclose(value[0], 0.0):
        raise ValueError('The "@arg@" parameter must contain a value equal to 0 in the first index.')

    if vector_type == 'creation' and not _np_isclose(value[-1], 0.0):
        raise ValueError('The "@arg@" parameter must contain a value equal to 0 in the last index.')

    if vector_type == 'stochastic' and not _np_isclose(_np_sum(value), 1.0):
        raise ValueError('The "@arg@" parameter values must sum to 1.')

    return value


def validate_walk(value: _tany, possible_states: _olist_str) -> _tvalid_walk:

    try:
        value = _extract_data_generic(value)
    except Exception as e:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from e

    if all(_is_integer(state) for state in value):
        value_type = 'integer'
    elif all(_is_string(state) for state in value):
        value_type = 'string'
    else:
        raise TypeError('The "@arg@" parameter must be either an array_like object of integers or an array_like object of non-empty strings.')

    if value_type == 'integer':

        value = [int(state) for state in value]

        if possible_states is None:

            possible_states = [str(i) for i in range(1, len(set(value)) + 1)]
            possible_states_length = len(possible_states)

            if possible_states_length < 2:
                raise ValueError('The "@arg@" parameter does not provide enough data to infer the possible states.')

        else:

            try:
                possible_states = validate_state_names(possible_states)
            except Exception as e:
                raise ValueError('The "@arg@" parameter must be validated against a valid list of possible states.') from e

            possible_states_length = len(possible_states)

        if any(state < 0 or state >= possible_states_length for state in value):
            raise ValueError(f'The "@arg@" parameter, when specified as a list of integers, must contain only values between 0 and the number of existing states minus one ({possible_states_length - 1:d}).')

    else:

        if possible_states is None:

            possible_states = sorted(set(value))

            if len(possible_states) < 2:
                raise ValueError('The "@arg@" parameter does not provide enough data to infer the possible states.')

        else:

            try:
                possible_states = validate_state_names(possible_states)
            except Exception as e:
                raise ValueError('The "@arg@" parameter must be validated against a valid list of possible states.') from e

        value = [possible_states.index(state) if state in possible_states else -1 for state in value]

        if any(state == -1 for state in value):
            raise ValueError(f'The "@arg@" parameter, when specified as a list of strings, must contain only values matching the names of the existing states ({", ".join(possible_states)}).')

    if len(value) < 2:
        raise ValueError('The "@arg@" parameter must contain at least two elements.')

    return value, possible_states


def validate_walks(value: _tany, possible_states: _tlist_str) -> _tvalid_walks:

    try:
        value = _extract_data_generic(value)
    except Exception as e:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.') from e

    value_length = len(value)

    if value_length < 2:
        raise ValueError('The "@arg@" parameter must contain at least 2 elements.')

    try:
        possible_states = validate_state_names(possible_states)
    except Exception as e:
        raise ValueError('The "@arg@" parameter must be validated against a valid list of possible states.') from e

    for i in range(value_length):
        try:
            value[i], _ = validate_walk(value[i], possible_states)
        except Exception as e:
            raise ValueError('The "@arg@" parameter contains invalid elements.') from e

    return value, possible_states
