# -*- coding: utf-8 -*-

__all__ = [
    'onumeric', 'tnumeric'
]


###########
# IMPORTS #
###########


# Major

import numpy as _np

# Minor

from typing import (
    Iterable as _Iterable,
    Optional as _Optional,
    Union as _Union
)


#########
# TYPES #
#########


tnumeric = _Union[_Iterable, _np.ndarray]

# noinspection PyBroadException
try:
    import pandas as _pd
    tnumeric = _Union[tnumeric, _pd.DataFrame, _pd.Series]
except Exception:
    pass

# noinspection PyBroadException
try:
    import scipy.sparse as _sps
    tnumeric = _Union[tnumeric, _sps.bsr.bsr_matrix]
    tnumeric = _Union[tnumeric, _sps.coo.coo_matrix]
    tnumeric = _Union[tnumeric, _sps.csc.csc_matrix]
    tnumeric = _Union[tnumeric, _sps.csr.csr_matrix]
    tnumeric = _Union[tnumeric, _sps.dia.dia_matrix]
    tnumeric = _Union[tnumeric, _sps.dok.dok_matrix]
    tnumeric = _Union[tnumeric, _sps.lil.lil_matrix]
except Exception:
    pass

onumeric = _Optional[tnumeric]
