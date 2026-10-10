# -*- coding: utf-8 -*-

__all__ = [
    'ETOL',
    'FTOL'
]


###########
# IMPORTS #
###########

# Libraries

import numpy as _np


#############
# VARIABLES #
#############

_EPS = _np.finfo(_np.float64).eps  # pylint: disable=no-member

ETOL = float(_np.sqrt(_EPS))
FTOL = 100.0 * float(_EPS)
