# -*- coding: utf-8 -*-

__all__ = [
    'evaluate',
    'string_to_function'
]


###########
# IMPORTS #
###########

# Standard

import ast as _ast
import types as _tp

# Libraries

# noinspection PyUnresolvedReferences
import networkx as _nx  # noqa: F401

# noinspection PyUnresolvedReferences
import numpy as _np  # noqa: F401

# noinspection PyUnresolvedReferences
import scipy.sparse as _spsp  # noqa: F401

try:
    import pandas as _pd
    _pandas_found = True
except ImportError:  # pragma: no cover
    _pd = None
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

        value = value.replace('np.', '_np.')
        value = value.replace('nx.', '_nx.')
        value = value.replace('pd.', '_pd.')
        value = value.replace('spsp.', '_spsp.')
        value = value.replace('HiddenMarkovModel', '_HiddenMarkovModel')
        value = value.replace('MarkovChain', '_MarkovChain')
        value = value.replace('ValidationError', '_ValidationError')
        value = eval(value)

    return value, skip


def string_to_function(source):

    ast_tree = _ast.parse(source)
    module_object = compile(ast_tree, '<ast>', 'exec')
    code_object = [c for c in module_object.co_consts if isinstance(c, _tp.CodeType)][0]

    func = _tp.FunctionType(code_object, {})

    return func
