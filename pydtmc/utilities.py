# -*- coding: utf-8 -*-

__all__ = [
    'create_rng',
    'generate_validation_error',
    'get_file_extension',
    'get_numpy_random_distributions',
    'extract_data_generic',
    'extract_data_numeric',
    'is_array',
    'is_bool',
    'is_dictionary',
    'is_float',
    'is_graph',
    'is_integer',
    'is_iterable',
    'is_list',
    'is_number',
    'is_pandas',
    'is_spmatrix',
    'is_string',
    'is_tuple'
]


###########
# IMPORTS #
###########

# Standard

from copy import (
    deepcopy as _cp_deepcopy
)

from pathlib import (
    Path as _pl_Path
)

from typing import (
    Iterable as _tp_Iterable
)

# Libraries

from networkx import (
    DiGraph as _nx_DiGraph,
    MultiDiGraph as _nx_MultiDiGraph
)

from numpy import (
    array as _np_array,
    copy as _np_copy,
    floating as _np_floating,
    integer as _np_integer,
    issubdtype as _np_issubdtype,
    ndarray as _np_ndarray,
    number as _np_number
)

from numpy.random import (
    RandomState as _npr_RandomState
)

# noinspection PyProtectedMember
from numpy.random.mtrand import (
    _rand as _nprm_rand
)

from scipy.sparse import (
    spmatrix as _spsp_spmatrix
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
    tany as _tany,
    tarray as _tarray,
    texception as _texception,
    tlist_any as _tlist_any,
    tlist_str as _tlist_str,
    trand as _trand
)

from .exceptions import (
    ValidationError as _ValidationError
)


#############
# FUNCTIONS #
#############

def create_rng(seed: _oint) -> _trand:

    if seed is None:
        return _nprm_rand

    if isinstance(seed, (int, _np_integer)):
        return _npr_RandomState(int(seed))

    raise TypeError('The specified seed is not a valid RNG initializer.')


def extract_data_generic(data: _tany) -> _tlist_any:

    if is_list(data):
        result = _cp_deepcopy(data)
    elif is_dictionary(data):
        result = list(data.values())
    elif is_iterable(data):
        result = list(data)
    else:
        result = None

    if result is None:
        raise TypeError('The data type is not supported.')

    return result


def extract_data_numeric(data: _tany) -> _tarray:

    if is_list(data):
        result = _np_array(data)
    elif is_dictionary(data):
        result = _np_array(list(data.values()))
    elif is_array(data):
        result = _np_copy(data)
    elif is_spmatrix(data):
        result = _np_array(data.todense())
    elif is_pandas(data):
        result = data.to_numpy(copy=True)
    elif is_iterable(data):
        result = _np_array(list(data))
    else:
        result = None

    if result is None or not _np_issubdtype(result.dtype, _np_number):
        raise TypeError('The data type is not supported.')

    return result


def generate_validation_error(e: _texception, trace: _tany) -> _ValidationError:

    arguments = ''.join(trace[0][4]).split('=', 1)[0].strip()

    if ',' in arguments:
        arguments = arguments[:arguments.index(',')]

    message = str(e).replace('@arg@', arguments)
    validation_error = _ValidationError(message)

    return validation_error


def get_file_extension(file_path: str) -> str:

    result = ''.join(_pl_Path(file_path).suffixes).lower()

    return result


def get_numpy_random_distributions() -> _tlist_str:

    try:
        from numpydoc.docscrape import NumpyDocString as _NumpyDocString  # noqa
    except ImportError:  # pragma: no cover
        return []

    excluded_funcs = ('dirichlet', 'multinomial', 'multivariate_normal')
    valid_summaries = ('DRAW RANDOM SAMPLES', 'DRAW SAMPLES', 'DRAWS SAMPLES')

    result = []

    for func_name in dir(_npr_RandomState):

        func = getattr(_npr_RandomState, func_name)

        if not callable(func) or func_name.startswith('_') or func_name.startswith('standard_') or func_name in excluded_funcs:
            continue

        doc = _NumpyDocString(func.__doc__)

        if 'Summary' not in doc:
            continue

        doc_summary = doc['Summary']

        if not isinstance(doc_summary, list) or (len(doc_summary) == 0) or not isinstance(doc_summary[0], str):
            continue

        doc_summary_first = doc_summary[0].upper()

        for valid_summary in valid_summaries:
            if doc_summary_first.startswith(valid_summary):
                result.append(func_name)
                break

    return result


def is_array(value: _tany) -> bool:

    return value is not None and isinstance(value, _np_ndarray)


def is_bool(value: _tany) -> bool:

    return value is not None and isinstance(value, bool)


def is_dictionary(value: _tany) -> bool:

    return value is not None and isinstance(value, dict)


def is_float(value: _tany) -> bool:

    return value is not None and isinstance(value, (float, _np_floating))


def is_graph(value: _tany, multi: bool) -> bool:

    if multi:
        return value is not None and isinstance(value, _nx_MultiDiGraph)

    return value is not None and isinstance(value, _nx_DiGraph)


def is_integer(value: _tany) -> bool:

    return value is not None and isinstance(value, (int, _np_integer)) and not isinstance(value, bool)


def is_iterable(value: _tany) -> bool:

    return value is not None and isinstance(value, _tp_Iterable) and not isinstance(value, (bytearray, bytes, str))


def is_list(value: _tany) -> bool:

    return value is not None and isinstance(value, list)


def is_number(value: _tany) -> bool:

    return is_float(value) or is_integer(value)


def is_pandas(value: _tany) -> bool:

    if not _pandas_found:  # pragma: no cover
        return False

    return value is not None and isinstance(value, (_pd_DataFrame, _pd_Series))


def is_spmatrix(value: _tany) -> bool:

    return value is not None and isinstance(value, _spsp_spmatrix)


def is_string(value: _tany) -> bool:

    return value is not None and isinstance(value, str) and len(value) > 0


def is_tuple(value: _tany) -> bool:

    return value is not None and isinstance(value, tuple)
