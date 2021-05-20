# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########


# Major

import numpy as np
import numpy.linalg as npl

# Minor

from collections import (
    namedtuple
)

from pydtmc import (
    MarkovChain
)

from pytest import (
    mark
)


##############
# TEST CASES #
##############


Case = namedtuple('Case', ['id', 'p', 'determinant', 'rank', 'is_absorbing', 'is_ergodic', 'is_reversible', 'is_symmetric', 'kemeny_constant', 'period', 'entropy_rate', 'mixing_rate', 'relaxation_rate'])

cases = [
    Case(
        '#1',
        [[0.0, 1.0], [1.0, 0.0]],
        -1.0, 2,
        False, False, True, True,
        None,
        2,
        None, None, None
    ),
    Case(
        '#2',
        [[0.05, 0.95], [0.8, 0.2]],
        -0.75, 2,
        False, True, True, False,
        None,
        1,
        0.36239685545027234, 3.476059496782207, 4.0
    ),
    Case(
        '#3',
        [[0.5, 0.5], [0.5, 0.5]],
        0.0, 1,
        False, True, True, True,
        None,
        1,
        0.6931471805599453, None, None
    ),
    Case(
        '#4',
        [[1.0, 0.0], [0.1, 0.9]],
        0.9, 2,
        True, False, False, False,
        10.000000000000002,
        1,
        None, None, None
    ),
    Case(
        '#5',
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        1.0, 3,
        True, False, False, True,
        None,
        1,
        None, None, None
    ),
    Case(
        '#6',
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        1.0, 3,
        False, False, False, False,
        None,
        3,
        None, None, None
    ),
    Case(
        '#7',
        [[0.0, 0.3, 0.7], [0.1, 0.5, 0.4], [0.1, 0.2, 0.7]],
        -0.029999999999999995, 3,
        False, True, False, False,
        None,
        1,
        0.8267342221751819, 0.8305835450825373, 1.4285714285714284
    ),
    Case(
        '#8',
        [[0.0, 1.0, 0.0], [0.3, 0.0, 0.7], [0.0, 1.0, 0.0]],
        0.0, 2,
        False, False, True, False,
        None,
        2,
        None, None, None
    ),
    Case(
        '#9',
        [[0.1, 0.1, 0.8], [0.3, 0.3, 0.4], [0.25, 0.5, 0.25]],
        0.05000000000000001, 3,
        False, True, False, False,
        None,
        1,
        0.963389503699457, 0.6676164013906685, 1.288007155526294
    ),
    Case(
        '#10',
        [[0.05, 0.9, 0.05], [0.5, 0.4, 0.1], [0.3, 0.2, 0.5]],
        -0.19, 3,
        False, True, False, False,
        None,
        1,
        0.7766627625167336, 1.293588988765789, 1.8573766183083227
    ),
    Case(
        '#11',
        [[0.1, 0.1, 0.8], [0.3, 0.3, 0.4], [0.25, 0.5, 0.25]],
        0.05000000000000001, 3,
        False, True, False, False,
        None,
        1,
        0.963389503699457, 0.6676164013906685, 1.288007155526294
    ),
    Case(
        '#12',
        [[0.78, 0.22, 0.0], [0.0, 0.1, 0.9], [0.45, 0.0, 0.55]],
        0.132, 3,
        False, True, False, False,
        None,
        1,
        0.5439208317104742, 0.9876770710214384, 1.5706429060964282
    ),
    Case(
        '#13',
        [[0.5, 0.5, 0.0], [0.5, 0.25, 0.25], [0, 1/3, 2/3]],
        -0.12500000000000003, 3,
        False, True, True, False,
        None,
        1,
        0.8037285736803539, 2.08276541630889, 2.622623436526918
    ),
    Case(
        '#14',
        [[0.5, 0.4, 0.1], [0.25, 0.25, 0.5], [0.05, 0.9, 0.05]],
        -0.19250000000000003, 3,
        False, True, False, False,
        None,
        1,
        0.83846856618281, 1.6726967362944687, 2.2222222222222223
    ),
    Case(
        '#15',
        [[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.1, 0.7, 0.2]],
        0.05000000000000001, 3,
        False, True, False, False,
        None,
        1,
        0.7077237190515739, 0.6676164013906681, 1.2880071555262935
    ),
    Case(
        '#16',
        [[0.0, 0.0, 0.5, 0.5], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        0.0, 3,
        False, False, False, False,
        None,
        3,
        None, None, None
    ),
    Case(
        '#17',
        [[0.7, 0.1, 0.1, 0.1], [0.85, 0.05, 0.05, 0.05], [0.2, 0.2, 0.5, 0.1], [0.3, 0.3, 0.3, 0.1]],
        0.002999999999999999, 4,
        False, True, False, False,
        None,
        1,
        0.981530111623928, 1.043229199083855, 1.6219122507466015
    ),
    Case(
        '#18',
        [[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0, 0.0]],
        0.0, 4,
        False, False, False, False,
        None,
        3,
        None, None, None
    ),
    Case(
        '#19',
        [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.25, 0.75, 0.0, 0.0, 0.0, 0.0], [0.2, 0.0, 0.4, 0.2, 0.2, 0.0], [0.0, 0.0, 0.0, 0.4, 0.6, 0.0], [0.0, 0.0, 0.0, 0.5, 0.0, 0.5], [0.0, 0.0, 0.0, 0.4, 0.0, 0.6]],
        0.005999999999999998, 6,
        False, False, False, False,
        None,
        1,
        None, None, None
    ),
    Case(
        '#20',
        [[0.2, 0.2, 0.2, 0.2, 0.0, 0.2], [0.05, 0.75, 0.0, 0.1, 0.0, 0.1], [0.4, 0.6, 0.0, 0.0, 0.0, 0.0], [0.1, 0.1, 0.2, 0.3, 0.0, 0.3], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.3, 0.0, 0.1, 0.1, 0.0, 0.5]],
        -0.0020000000000000005, 6,
        False, False, False, False,
        None,
        1,
        None, None, None
    )
]


########
# TEST #
########


@mark.parametrize(
    argnames=('p', 'is_absorbing'),
    argvalues=[(case.p, case.is_absorbing) for case in cases],
    ids=['test_absorption ' + case.id for case in cases]
)
def test_absorption(p, is_absorbing):

    mc = MarkovChain(p)

    actual = mc.is_absorbing
    expected = is_absorbing

    assert actual == expected


@mark.parametrize(
    argnames=('p', 'is_ergodic'),
    argvalues=[(case.p, case.is_ergodic) for case in cases],
    ids=['test_ergodicity ' + case.id for case in cases]
)
def test_ergodicity(p, is_ergodic):

    mc = MarkovChain(p)

    actual = mc.is_ergodic
    expected = is_ergodic

    assert actual == expected


@mark.parametrize(
    argnames='p',
    argvalues=[case.p for case in cases],
    ids=['test_irreducibility ' + case.id for case in cases]
)
def test_irreducibility(p):

    mc = MarkovChain(p)

    if mc.is_irreducible:

        actual = mc.states
        expected = mc.recurrent_states

        assert actual == expected

        actual = len(mc.communicating_classes)
        expected = 1

        assert actual == expected

        cf = mc.to_canonical_form()
        actual = cf.p
        expected = mc.p

        assert np.allclose(actual, expected)


@mark.parametrize(
    argnames=('p', 'kemeny_constant'),
    argvalues=[(case.p, case.kemeny_constant) for case in cases],
    ids=['test_kemeny_constant ' + case.id for case in cases]
)
def test_kemeny_constant(p, kemeny_constant):

    mc = MarkovChain(p)

    actual = mc.kemeny_constant
    expected = kemeny_constant

    assert actual == expected


@mark.parametrize(
    argnames=('p', 'determinant', 'rank'),
    argvalues=[(case.p, case.determinant, case.rank) for case in cases],
    ids=['test_matrix ' + case.id for case in cases]
)
def test_matrix(p, determinant, rank):

    mc = MarkovChain(p)

    actual = mc.determinant
    expected = determinant

    assert np.isclose(actual, expected)

    actual = mc.rank
    expected = rank

    assert actual == expected


@mark.parametrize(
    argnames=('p', 'period'),
    argvalues=[(case.p, case.period) for case in cases],
    ids=['test_periodicity ' + case.id for case in cases]
)
def test_periodicity(p, period):

    mc = MarkovChain(p)

    actual = mc.period
    expected = period

    assert actual == expected

    actual = mc.is_aperiodic
    expected = (period == 1)

    assert actual == expected


@mark.parametrize(
    argnames=('p', 'entropy_rate', 'mixing_rate', 'relaxation_rate'),
    argvalues=[(case.p, case.entropy_rate, case.mixing_rate, case.relaxation_rate) for case in cases],
    ids=['test_rates ' + case.id for case in cases]
)
def test_rates(p, entropy_rate, mixing_rate, relaxation_rate):

    mc = MarkovChain(p)

    actual = mc.entropy_rate
    expected = entropy_rate

    assert actual == expected

    actual = mc.mixing_rate
    expected = mixing_rate

    assert actual == expected

    actual = mc.relaxation_rate
    expected = relaxation_rate

    assert actual == expected


@mark.parametrize(
    argnames='p',
    argvalues=[case.p for case in cases],
    ids=['test_regularity ' + case.id for case in cases]
)
def test_regularity(p):

    mc = MarkovChain(p)

    if mc.is_regular:

        actual = mc.is_irreducible
        expected = True

        assert actual == expected

        values = np.sort(np.abs(npl.eigvals(mc.p)))
        actual = np.sum(np.logical_or(np.isclose(values, 1.0), values > 1.0))
        expected = 1

        assert actual == expected


@mark.parametrize(
    argnames=('p', 'is_reversible'),
    argvalues=[(case.p, case.is_reversible) for case in cases],
    ids=['test_reversibility ' + case.id for case in cases]
)
def test_reversibility(p, is_reversible):

    mc = MarkovChain(p)

    actual = mc.is_reversible
    expected = is_reversible

    assert actual == expected


@mark.parametrize(
    argnames=('p', 'is_symmetric'),
    argvalues=[(case.p, case.is_symmetric) for case in cases],
    ids=['test_symmetry ' + case.id for case in cases]
)
def test_symmetry(p, is_symmetric):

    mc = MarkovChain(p)

    actual = mc.is_symmetric
    expected = is_symmetric

    assert actual == expected
