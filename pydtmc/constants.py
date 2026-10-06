# -*- coding: utf-8 -*-

__all__ = [
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

FTOL = 100.0 * _np.finfo(_np.float64).eps  # pylint: disable=no-member
