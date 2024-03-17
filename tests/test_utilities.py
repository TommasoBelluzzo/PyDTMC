# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Standard

from os.path import (
    abspath as _osp_abspath,
    dirname as _osp_dirname
)

# Libraries

import numpy as _np
import pytest as _pt

# Internal

from pydtmc.utilities import (
    extract_numeric as _extract_numeric
)

from tests.utilities import (
    evaluate as _evaluate
)


#############
# CONSTANTS #
#############

_base_directory = _osp_abspath(_osp_dirname(__file__))


#########
# TESTS #
#########

# noinspection PyBroadException
def test_extract_numeric(value, evaluate, is_valid):

    if evaluate and isinstance(value, str):
        value, skip = _evaluate(value)
    else:
        skip = False

    if skip:
        _pt.skip('Pandas library could not be imported.')
    else:

        try:
            result = _extract_numeric(value)
            result_is_valid = True
        except Exception:
            result = None
            result_is_valid = False

        actual = result_is_valid
        expected = is_valid

        assert actual == expected

        if result_is_valid:
            result_check = isinstance(result, _np.ndarray)
            assert result_check is True
