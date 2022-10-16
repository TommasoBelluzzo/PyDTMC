# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Standard

from ast import (
    parse as _ast_parse
)

from os.path import (
    abspath as _osp_abspath,
    dirname as _osp_dirname
)

from types import (
    CodeType as _tp_CodeType,
    FunctionType as _tp_FunctionType
)

# Libraries

# noinspection PyUnresolvedReferences
from numpy import (  # noqa
    asarray as _np_asarray,
    ndarray as _np_ndarray
)

# noinspection DuplicatedCode
try:
    from pandas import (
        DataFrame as _pd_DataFrame,
        Series as _pd_Series
    )
    _pandas_found = True
except ImportError:  # noqa
    _pd_DataFrame = None
    _pd_Series = None
    _pandas_found = False

from pytest import (
    skip as _pt_skip
)

# noinspection PyUnresolvedReferences
from scipy.sparse import (  # noqa
    coo_matrix as _spsp_coo_matrix,
    csr_matrix as _spsp_csr_matrix
)

# Internal

from pydtmc.utilities import (
    extract_data_generic as _extract_data_generic,
    extract_data_numeric as _extract_data_numeric
)


#############
# CONSTANTS #
#############

_base_directory = _osp_abspath(_osp_dirname(__file__))


#############
# FUNCTIONS #
#############

def _eval_replace(value):

    value = value.replace('np.', '_np_')
    value = value.replace('nx.', '_nx_')
    value = value.replace('pd.', '_pd_')
    value = value.replace('spsp.', '_spsp_')

    return value


# noinspection PyArgumentList
def _string_to_function(source):

    ast_tree = _ast_parse(source)
    module_object = compile(ast_tree, '<ast>', 'exec')
    code_object = [c for c in module_object.co_consts if isinstance(c, _tp_CodeType)][0]

    f = _tp_FunctionType(code_object, {})

    return f


#########
# TESTS #
#########

# noinspection PyBroadException
def test_extract_data_generic(value, evaluate, is_valid):

    if value is not None and isinstance(value, str) and evaluate:
        value = eval(value)

    try:
        result = _extract_data_generic(value)
        result_is_valid = True
    except Exception:
        result = None
        result_is_valid = False

    actual = result_is_valid
    expected = is_valid

    assert actual == expected

    if result is not None:

        actual = isinstance(result, list)
        expected = True

        assert actual == expected


# noinspection PyBroadException
def test_extract_data_numeric(value, evaluate, is_valid):

    should_skip = False

    if value is not None and isinstance(value, str) and evaluate:

        if 'pd.' in value and not _pandas_found:
            should_skip = True
        else:
            value = _eval_replace(value)
            value = eval(value)

    if should_skip:
        _pt_skip('Pandas library could not be imported.')
    else:

        try:
            result = _extract_data_numeric(value)
            result_is_valid = True
        except Exception:
            result = None
            result_is_valid = False

        actual = result_is_valid
        expected = is_valid

        assert actual == expected

        if result is not None:

            actual = isinstance(result, _np_ndarray)
            expected = True

            assert actual == expected
