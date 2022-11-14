# -*- coding: utf-8 -*-

__all__ = [
    'evaluate',
    'string_to_function'
]


###########
# IMPORTS #
###########

# Standard

from ast import (
    parse as _ast_parse
)

from types import (
    CodeType as _tp_CodeType,
    FunctionType as _tp_FunctionType
)

# Libraries

# noinspection PyUnresolvedReferences
from networkx import (  # noqa
    DiGraph as _nx_DiGraph,
    MultiDiGraph as _nx_MultiDiGraph
)

# noinspection PyUnresolvedReferences
from numpy import (  # noqa
    array as _np_array
)

# noinspection PyUnresolvedReferences
from scipy.sparse import (  # noqa
    coo_matrix as _spsp_coo_matrix,
    csr_matrix as _spsp_csr_matrix
)

try:
    from pandas import (
        DataFrame as _pd_DataFrame,
        Series as _pd_Series
    )
    _pandas_found = True
except ImportError:  # pragma: no cover
    _pd_DataFrame, _pd_Series = None, None
    _pandas_found = False

# Internal

# noinspection PyUnresolvedReferences
from pydtmc import (  # noqa
    HiddenMarkovModel as _HiddenMarkovModel,
    MarkovChain as _MarkovChain,
    ValidationError as _ValidationError
)


#############
# FUNCTIONS #
#############

def evaluate(value):

    if 'pd.' in value and not _pandas_found:
        skip = True
    else:

        skip = False

        value = value.replace('np.', '_np_')
        value = value.replace('nx.', '_nx_')
        value = value.replace('pd.', '_pd_')
        value = value.replace('spsp.', '_spsp_')
        value = value.replace('HiddenMarkovModel', '_HiddenMarkovModel')
        value = value.replace('MarkovChain', '_MarkovChain')
        value = value.replace('ValidationError', '_ValidationError')
        value = eval(value)

    return value, skip


def string_to_function(source):

    ast_tree = _ast_parse(source)
    module_object = compile(ast_tree, '<ast>', 'exec')
    code_object = [c for c in module_object.co_consts if isinstance(c, _tp_CodeType)][0]

    func = _tp_FunctionType(code_object, {})

    return func
