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

EPS = _np.finfo(_np.float64).eps  # pylint: disable=no-member

ETOL = _np.sqrt(EPS)
FTOL = 100.0 * EPS
