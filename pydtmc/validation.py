# -*- coding: utf-8 -*-

__all__ = [
    'validate_boolean', 'validate_enumerator', 'validate_dpi', 'validate_integer',
    'validate_dictionary', 'validate_mask', 'validate_matrix', 'validate_transition_matrix', 'validate_transition_matrix_size',
    'validate_distribution', 'validate_hyperparameter', 'validate_rewards', 'validate_vector', 'validate_walk',
    'validate_state', 'validate_states', 'validate_state_names', 'validate_status'
]


###########
# IMPORTS #
###########


# Major

import numpy as _np

# Minor

from copy import (
    deepcopy as _deepcopy
)

from numbers import (
    Number as _Number
)

from typing import (
    Any as _Any,
    Dict as _Dict,
    Iterable as _Iterable,
    List as _List,
    Optional as _Optional,
    Tuple as _Tuple,
    Union as _Union
)

# Optional

try:
    import pandas as _pd
except ImportError:
    _pd = None

try:
    import scipy.sparse as _sps
except ImportError:
    _sps = None


#############
# FUNCTIONS #
#############


def extract_non_numeric(data: _Any) -> _List[_Any]:

    result = None

    if isinstance(data, _Dict):
        result = list(data.values())
    elif isinstance(data, _List):
        result = _deepcopy(data)
    elif isinstance(data, _Iterable) and not isinstance(data, str):
        result = list(data)

    if result is None or any(isinstance(x, _Number) for x in result):
        raise TypeError('The data type is not supported.')

    return result


def extract_numeric(data: _Any) -> _np.ndarray:

    result = None

    if isinstance(data, list):
        result = _np.array(data)
    elif isinstance(data, _np.ndarray):
        result = _np.copy(data)
    elif _pd and isinstance(data, (_pd.DataFrame, _pd.Series)):
        result = data.to_numpy(copy=True)
    elif _sps and isinstance(data, _sps.bsr.bsr_matrix):
        result = _np.array(data.todense())
    elif _sps and isinstance(data, _sps.coo.coo_matrix):
        result = _np.array(data.todense())
    elif _sps and isinstance(data, _sps.csc.csc_matrix):
        result = _np.array(data.todense())
    elif _sps and isinstance(data, _sps.csr.csr_matrix):
        result = _np.array(data.todense())
    elif _sps and isinstance(data, _sps.dia.dia_matrix):
        result = _np.array(data.todense())
    elif _sps and isinstance(data, _sps.dok.dok_matrix):
        result = _np.array(data.todense())
    elif _sps and isinstance(data, _sps.lil.lil_matrix):
        result = _np.array(data.todense())
    elif isinstance(data, _Dict):
        result = _np.array(list(data.values()))
    elif isinstance(data, _Iterable) and not isinstance(data, str):
        result = _np.array(list(data))

    if result is None or not _np.issubdtype(result.dtype, _np.number):
        raise TypeError('The data type is not supported.')

    return result


def validate_boolean(value: _Any) -> bool:

    if isinstance(value, bool):
        return value

    raise TypeError('The "@arg@" parameter must be a boolean value.')


def validate_distribution(distribution: _Any, size: int) -> _Union[int, _List[_np.ndarray]]:

    if isinstance(distribution, int):

        if distribution <= 0:
            raise ValueError('The "@arg@" parameter, when specified as an integer, must be greater than or equal to 1.')

        return distribution

    elif isinstance(distribution, list):

        distribution_len = len(distribution)

        if distribution_len == 0:
            raise ValueError('The "@arg@" parameter, when specified as a list of vectors, must be non-empty.')

        for i, vector in distribution:

            if not isinstance(vector, _np.ndarray) or not _np.issubdtype(vector.dtype, _np.number):
                raise TypeError('The "@arg@" parameter must contain only numeric vectors.')

            if distribution_len <= 1:
                raise ValueError('The "@arg@" parameter, when specified as a list of vectors, must contain at least 2 elements.')

            vector = vector.astype(float)
            distribution[i] = vector

            if (vector.ndim != 1) or (vector.size != size):
                raise ValueError('The "@arg@" parameter must contain only vectors of size {size:d}.')

            if not all(_np.isfinite(x) and (x >= 0.0) and (x <= 1.0) for x in _np.nditer(vector)):
                raise ValueError('The "@arg@" parameter must contain only vectors consisting of values between 0 and 1.')

            if not _np.isclose(_np.sum(vector), 1.0):
                raise ValueError('The "@arg@" parameter must contain only vectors consisting of values whose sum is 1.')

        return distribution

    else:
        raise TypeError('The "@arg@" parameter must be either an integer representing the number of redistributions to perform or a list of valid distributions.')


def validate_enumerator(value: _Any, possible_values: _List[str]) -> str:

    if not isinstance(value, str):
        raise TypeError('The "@arg@" parameter must be a string value.')

    if value not in possible_values:
        raise ValueError(f'The "@arg@" parameter value must be one of the following: {", ".join(possible_values)}.')

    return value


def validate_hyperparameter(hyperparameter: _Any, size: int) -> _np.ndarray:

    try:
        hyperparameter = extract_numeric(hyperparameter)
    except Exception:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    if not _np.issubdtype(hyperparameter.dtype, _np.number):
        raise TypeError('The "@arg@" parameter must contain only integer values.')

    hyperparameter = hyperparameter.astype(float)

    if (hyperparameter.ndim != 2) or (hyperparameter.shape[0] != hyperparameter.shape[1]):
        raise ValueError('The "@arg@" parameter must be a 2d square matrix.')

    if hyperparameter.shape[0] != size:
        raise ValueError(f'The "@arg@" parameter size must be equal to {size:d}.')

    if not all(_np.isfinite(x) and _np.isreal(x) and _np.equal(_np.mod(x, 1), 0) and (x >= 1.0) for x in _np.nditer(hyperparameter)):
        raise ValueError('The "@arg@" parameter must contain only integer values greater than or equal to 1.')

    return hyperparameter


def validate_dictionary(d: _Any) -> _Dict[_Tuple[str, str], float]:

    if not isinstance(d, dict):
        raise ValueError('The "@arg@" parameter must be a dictionary.')

    keys = d.keys()

    if not all(isinstance(key, tuple) and (len(key) == 2) and isinstance(key[0], str) and isinstance(key[1], str) for key in keys):
        raise ValueError('The "@arg@" parameter keys must be tuples containing 2 string values.')

    keys = sorted(list(set(sum(keys, ()))))

    if not all(key is not None and (len(key) > 0) for key in keys):
        raise TypeError('The "@arg@" parameter keys must contain only valid string values.')

    values = d.values()

    if not all(isinstance(value, (float, int)) for value in values):
        raise ValueError('The "@arg@" parameter values must be float or integer numbers.')

    values = [float(value) for value in values]

    if not all((value >= 0.0) and (value <= 1.0) for value in values):
        raise ValueError('The "@arg@" parameter values can contain only numbers between 0 and 1.')

    result = {}

    for key, value in d.items():
        result[key] = float(value)

    return result


def validate_dpi(value: _Any) -> int:

    if not isinstance(value, int):
        raise TypeError('The "@arg@" parameter must be an integer value.')

    if (value < 72) or (value > 300):
        raise ValueError('The "@arg@" parameter must have a value between 72 and 300.')

    return value


def validate_integer(value: _Any, lower_limit: _Tuple[int, bool] = None, upper_limit: _Tuple[int, bool] = None) -> int:

    if not isinstance(value, int):
        raise TypeError('The "@arg@" parameter must be an integer value.')

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


def validate_mask(mask: _Any, size: int) -> _np.ndarray:

    try:
        mask = extract_numeric(mask)
    except Exception:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    if not _np.issubdtype(mask.dtype, _np.number):
        raise TypeError('The "@arg@" parameter must contain only numeric values.')

    mask = mask.astype(float)

    if (mask.ndim != 2) or (mask.shape[0] != mask.shape[1]):
        raise ValueError('The "@arg@" parameter must be a 2d square matrix.')

    if mask.shape[0] != size:
        raise ValueError(f'The "@arg@" parameter size must be equal to {size:d}.')

    if not all(_np.isnan(x) or ((x >= 0.0) and (x <= 1.0)) for x in _np.nditer(mask)):
        raise ValueError('The "@arg@" parameter can contain only NaNs and values between 0 and 1.')

    if not _np.any(_np.nansum(mask, axis=1, dtype=float) > 1.0):
        raise ValueError('The "@arg@" parameter row sums must not exceed 1.')

    return mask


def validate_matrix(m: _Any) -> _np.ndarray:

    try:
        m = extract_numeric(m)
    except Exception:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    if not _np.issubdtype(m.dtype, _np.number):
        raise TypeError('The "@arg@" parameter must contain only numeric values.')

    m = m.astype(float)

    if (m.ndim != 2) or (m.shape[0] < 2) or (m.shape[0] != m.shape[1]):
        raise ValueError('The "@arg@" parameter must be a 2d square matrix with size greater than or equal to 2.')

    if not all(_np.isfinite(x) for x in _np.nditer(m)):
        raise ValueError('The "@arg@" parameter must contain only finite values.')

    return m


def validate_rewards(rewards: _Any, size: int) -> _np.ndarray:

    try:
        rewards = extract_numeric(rewards)
    except Exception:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    if not _np.issubdtype(rewards.dtype, _np.number):
        raise TypeError('The "@arg@" parameter must contain only numeric values.')

    rewards = rewards.astype(float)

    if (rewards.ndim < 1) or ((rewards.ndim == 2) and (rewards.shape[0] != 1)) or (rewards.ndim > 2):
        raise ValueError('The "@arg@" parameter must be a vector.')

    rewards = _np.ravel(rewards)

    if rewards.size != size:
        raise ValueError(f'The "@arg@" parameter length must be equal to the number of states ({size:d}).')

    if not all(_np.isfinite(x) and _np.isreal(x) for x in _np.nditer(rewards)):
        raise ValueError('The "@arg@" parameter must contain only real finite values.')

    return rewards


def validate_state(state: _Any, current_states: list) -> int:

    if isinstance(state, int):

        limit = len(current_states) - 1

        if (state < 0) or (state > limit):
            raise ValueError(f'The "@arg@" parameter, when specified as an integer, must have a value between 0 and the number of existing states minus one ({limit:d}).')

        return state

    if isinstance(state, str):

        if state not in current_states:
            raise ValueError(f'The "@arg@" parameter, when specified as a string, must match the name of an existing state ({", ".join(current_states)}).')

        return current_states.index(state)

    raise TypeError('The "@arg@" parameter must be either an integer or a string.')


def validate_status(status: _Any, current_states: list) -> _np.ndarray:

    size = len(current_states)

    if isinstance(status, int):

        limit = size - 1

        if (status < 0) or (status > limit):
            raise ValueError(f'The "@arg@" parameter, when specified as an integer, must have a value between 0 and the number of existing states minus one ({limit:d}).')

        result = _np.zeros(size, dtype=float)
        result[status] = 1.0

        return result

    if isinstance(status, str):

        if status not in current_states:
            raise ValueError(f'The "@arg@" parameter, when specified as a string, must match the name of an existing state ({", ".join(current_states)}).')

        status = current_states.index(status)

        result = _np.zeros(size, dtype=float)
        result[status] = 1.0

        return result

    try:
        status = extract_numeric(status)
    except Exception:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    if not _np.issubdtype(status.dtype, _np.number):
        raise TypeError('The "@arg@" parameter must contain only numeric values.')

    status = status.astype(float)

    if (status.ndim < 1) or ((status.ndim == 2) and (status.shape[0] != 1) and (status.shape[1] != 1)) or (status.ndim > 2):
        raise ValueError('The "@arg@" parameter must be a valid vector.')

    status = _np.ravel(status)

    if status.size != size:
        raise ValueError(f'The "@arg@" parameter length must be equal to the number of states ({size:d}).')

    if not all(_np.isfinite(x) and (x >= 0.0) and (x <= 1.0) for x in _np.nditer(status)):
        raise ValueError('The "@arg@" parameter must contain only values between 0 and 1.')

    if not _np.isclose(_np.sum(status), 1.0):
        raise ValueError('The "@arg@" parameter values must sum to 1.')

    return status


def validate_state_names(states: _Any, size: _Optional[int] = None) -> _List[str]:

    try:
        states = extract_non_numeric(states)
    except Exception:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    if not all(isinstance(s, str) for s in states):
        raise TypeError('The "@arg@" parameter must contain only string values.')

    if not all(s is not None and (len(s) > 0) for s in states):
        raise TypeError('The "@arg@" parameter must contain only valid string values.')

    states_length = len(states)
    states_unique = len(set(states))

    if states_unique < states_length:
        raise ValueError('The "@arg@" parameter must contain only unique values.')

    if size is not None and (states_length != size):
        raise ValueError(f'The "@arg@" parameter must contain a number of elements equal to {size:d}.')

    return states


def validate_states(states: _Any, current_states: _List[str], state_type: str, flex: bool) -> _List[int]:

    if flex:

        if isinstance(states, int):

            limit = len(current_states) - 1

            if (states < 0) or (states > limit):
                raise ValueError(f'The "@arg@" parameter, when specified as an integer, must have a value between 0 and the number of existing states minus one ({limit:d}).')

            return [states]

        if isinstance(states, str):

            if states not in current_states:
                raise ValueError(f'The "@arg@" parameter, when specified as a string, must match the name of an existing state ({", ".join(current_states)}).')

            return [current_states.index(states)]

    try:
        states = extract_non_numeric(states)
    except Exception:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    current_states_length = len(current_states)

    if all(isinstance(s, int) for s in states):

        if any((s < 0) or (s >= current_states_length) for s in states):
            raise ValueError(f'The "@arg@" parameter, when specified as a list of integers, must contain only values between 0 and the number of existing states minus one ({current_states_length - 1:d}).')

    elif all(isinstance(s, str) for s in states):

        states = [current_states.index(s) if s in current_states else -1 for s in states]

        if any(s == -1 for s in states):
            raise ValueError(f'The "@arg@" parameter, when specified as a list of strings, must contain only values matching the names of the existing states ({", ".join(current_states)}).')

    else:

        if flex:
            raise TypeError('The "@arg@" parameter must be either an integer, a string, an array_like object of integers or an array_like object of strings.')
        else:
            raise TypeError('The "@arg@" parameter must be either an array_like object of integers or an array_like object of strings.')

    states_length = len(states)

    if (state_type != 'walk') and (len(set(states)) < states_length):
        raise ValueError('The "@arg@" parameter must contain only unique values.')

    if state_type == 'regular':
        if (states_length < 1) or (states_length > current_states_length):
            raise ValueError(f'The "@arg@" parameter must contain a number of elements between 1 and the number of existing states ({current_states_length:d}).')
    elif state_type == 'subset':
        if (states_length < 1) or (states_length >= current_states_length):
            raise ValueError(f'The "@arg@" parameter must contain a number of elements between 1 and the number of existing states minus one ({current_states_length - 1:d}).')
    else:
        if states_length < 2:
            raise ValueError('The "@arg@" parameter must contain at least two elements.')

    return states


def validate_transition_matrix(p: _Any) -> _np.ndarray:

    try:
        p = extract_numeric(p)
    except Exception:
        raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    if not _np.issubdtype(p.dtype, _np.number):
        raise TypeError('The "@arg@" parameter must contain only numeric values.')

    p = p.astype(float)

    if (p.ndim != 2) or (p.shape[0] != p.shape[1]):
        raise ValueError('The "@arg@" parameter must be a 2d square matrix.')

    size = p.shape[0]

    if size < 2:
        raise ValueError('The "@arg@" parameter size must be greater than or equal to 2.')

    if not all(_np.isfinite(x) and (x >= 0.0) and (x <= 1.0) for x in _np.nditer(p)):
        raise ValueError('The "@arg@" parameter must contain only values between 0 and 1.')

    if not _np.allclose(_np.sum(p, axis=1), _np.ones(size)):
        raise ValueError('The "@arg@" parameter rows must sum to 1.')

    return p


def validate_transition_matrix_size(size: _Any) -> int:

    if not isinstance(size, int):
        raise TypeError('The "@arg@" parameter must be an integer value.')

    if size < 2:
        raise ValueError('The "@arg@" parameter must be greater than or equal to 2.')

    return size


def validate_vector(vector: _Any, vector_type: str, flex: bool, size: _Optional[int] = None) -> _np.ndarray:

    if flex and size is not None and isinstance(vector, int):
        vector = _np.repeat(float(vector), size)
    elif flex and size is not None and isinstance(vector, float):
        vector = _np.repeat(vector, size)
    else:
        try:
            vector = extract_numeric(vector)
        except Exception:
            raise TypeError('The "@arg@" parameter is null or wrongly typed.')

    if not _np.issubdtype(vector.dtype, _np.number):
        raise TypeError('The "@arg@" parameter must contain only numeric values.')

    vector = vector.astype(float)

    if (vector.ndim < 1) or ((vector.ndim == 2) and (vector.shape[0] != 1) and (vector.shape[1] != 1)) or (vector.ndim > 2):
        raise ValueError('The "@arg@" parameter must be a valid vector.')

    vector = _np.ravel(vector)

    if size is not None and (vector.size != size):
        raise ValueError(f'The "@arg@" parameter length must be equal to the number of states ({size:d}).')

    if not all(_np.isfinite(x) and (x >= 0.0) and (x <= 1.0) for x in _np.nditer(vector)):
        raise ValueError('The "@arg@" parameter must contain only values between 0 and 1.')

    if vector_type == 'annihilation':
        if not _np.isclose(vector[0], 0.0):
            raise ValueError('The "@arg@" parameter must contain a value equal to 0 in the first index.')
    elif vector_type == 'creation':
        if not _np.isclose(vector[-1], 0.0):
            raise ValueError('The "@arg@" parameter must contain a value equal to 0 in the last index.')
    elif vector_type == 'stochastic':
        if not _np.isclose(_np.sum(vector), 1.0):
            raise ValueError('The "@arg@" parameter values must sum to 1.')

    return vector


def validate_walk(walk: _Any, current_states: _List[str]) -> _Union[int, _List[int], _List[str]]:

    if isinstance(walk, int):

        if walk <= 0:
            raise ValueError('The "@arg@" parameter must be positive.')

        return walk

    elif isinstance(walk, list):

        current_states_length = len(current_states)

        if all(isinstance(s, int) for s in walk):

            if any((s < 0) or (s >= current_states_length) for s in walk):
                raise ValueError(f'The "@arg@" parameter, when specified as a list of integers, must contain only values between 0 and the number of existing states minus one ({current_states_length - 1:d}).')

        elif all(isinstance(s, str) for s in walk):

            walk = [current_states.index(s) if s in current_states else -1 for s in walk]

            if any(s == -1 for s in walk):
                raise ValueError(f'The "@arg@" parameter, when specified as a list of strings, must contain only values matching the names of the existing states ({", ".join(current_states)}).')

        else:
            raise TypeError('The "@arg@" parameter must be either an array_like object of integers representing the indices of existing states or an array_like object of strings matching the names of existing states.')

        states_length = len(walk)

        if states_length < 1:
            raise ValueError('The "@arg@" parameter must contain at least one element.')

        return walk

    else:
        raise TypeError('The "@arg@" parameter must be either an integer representing the number of walks to perform or a list of valid states.')
