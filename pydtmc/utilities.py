# -*- coding: utf-8 -*-

__all__ = [
    'create_rng',
    'generate_validation_error',
    'get_file_extension',
    'get_numpy_random_distributions'
]


###########
# IMPORTS #
###########

# Standard

from pathlib import (
    Path as _pl_Path
)

# Libraries

from numpy import (
    integer as _np_integer
)

from numpy.random import (
    RandomState as _npr_RandomState
)

# noinspection PyProtectedMember
from numpy.random.mtrand import (
    _rand as _nprm_rand
)

# Internal

from .custom_types import (
    oint as _oint,
    tany as _tany,
    texception as _texception,
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


def generate_validation_error(e: _texception, trace: _tany) -> _ValidationError:

    arguments = ''.join(trace[0][4]).split('=', 1)[0].strip()
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
