# -*- coding: utf-8 -*-

__all__ = [
    'assert_stochastic_matrix',
    'evaluate',
    'hasattr_deep',
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

import numpy as _np
import numpy.testing as _npt

# noinspection PyUnresolvedReferences
import scipy.sparse as _spsp  # noqa: F401

try:
    import pandas as _pd
    _pandas_found = True
except ImportError:  # pragma: no cover
    _pd = None
    _pandas_found = False

# Internal

# noinspection PyProtectedMember
from pydtmc._constants import (
    FTOL as _FTOL
)

# noinspection PyUnresolvedReferences
from pydtmc import (  # noqa
    HiddenMarkovModel as _HiddenMarkovModel,
    MarkovChain as _MarkovChain,
    ValidationError as _ValidationError
)


#############
# FUNCTIONS #
#############

def assert_stochastic_matrix(matrix):

    assert matrix.ndim == 2
    assert matrix.shape[0] >= 2
    assert matrix.shape[1] >= 2

    assert _np.all(_np.isfinite(matrix))
    assert _np.all((matrix >= 0.0) & (matrix <= 1.0))

    _npt.assert_allclose(_np.sum(matrix, axis=1), 1.0, rtol=0.0, atol=_FTOL)


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


def hasattr_deep(obj, *names):
    for name in names:

        if not hasattr(obj, name):
            return False

        obj = getattr(obj, name)

    return True


def string_to_function(source):

    ast_tree = _ast.parse(source)

    function_definitions = [node for node in ast_tree.body if isinstance(node, _ast.FunctionDef)]

    if len(function_definitions) != 1:
        raise ValueError('The source must contain exactly one function definition.')

    function_name = function_definitions[0].name
    module_object = compile(ast_tree, '<ast>', 'exec')

    code_objects = [
        code_object
        for code_object in module_object.co_consts
        if isinstance(code_object, _tp.CodeType) and code_object.co_name == function_name
    ]

    if len(code_objects) != 1:
        raise ValueError(f'Unable to compile function "{function_name}".')

    func = _tp.FunctionType(code_objects[0], {})

    return func
