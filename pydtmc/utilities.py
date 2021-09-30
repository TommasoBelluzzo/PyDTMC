# -*- coding: utf-8 -*-

__all__ = [
    'create_rng',
    'generate_validation_error',
    'get_file_extension'
]


###########
# IMPORTS #
###########

# Standard

from pathlib import (
    Path as _Path
)

# Libraries

import numpy as _np
import numpy.random as _npr
import numpy.random.mtrand as _nprm

# Internal

from .custom_types import (
    oint as _oint,
    tany as _tany,
    texception as _texception,
    trand as _trand
)

from .exceptions import (
    ValidationError as _ValidationError
)


#############
# FUNCTIONS #
#############

# noinspection PyProtectedMember
def create_rng(seed: _oint) -> _trand:

    if seed is None:
        return _nprm._rand

    if isinstance(seed, (int, _np.integer)):
        return _npr.RandomState(int(seed))

    raise TypeError('The specified seed is not a valid RNG initializer.')


def generate_validation_error(e: _texception, trace: _tany) -> _ValidationError:

    arguments = ''.join(trace[0][4]).split('=', 1)[0].strip()
    message = str(e).replace('@arg@', arguments)

    return _ValidationError(message)


def get_file_extension(file_path: str) -> str:

    return ''.join(_Path(file_path).suffixes).lower()
