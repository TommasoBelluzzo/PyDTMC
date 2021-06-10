# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Full

import numpy as np
import numpy.linalg as npl

# Partial

from collections import (
    namedtuple
)

from pydtmc import (
    MarkovChain
)

from pytest import (
    mark,
    skip
)


##############
# TEST CASES #
##############

Case = namedtuple('Case', [
    'id',
    'p',
    'determinant', 'rank',
    'is_absorbing', 'is_canonical', 'is_ergodic', 'is_reversible', 'is_symmetric',
    'fundamental_matrix', 'kemeny_constant',
    'period',
    'entropy_rate', 'entropy_rate_normalized', 'topological_entropy',
    'mixing_rate', 'relaxation_rate', 'spectral_gap', 'implied_timescales',
    'lumping_partitions',
    'accessibility_matrix',
    'adjacency_matrix',
    'communication_matrix'
])

cases = [
    Case(
        '#1',
        [[0.0, 1.0], [1.0, 0.0]],
        -1.0, 2,
        False, True, False, True, True,
        None, None,
        2,
        0.0, 0.0, 0.0,
        None, None, None, None,
        [],
        [[1, 1], [1, 1]],
        [[0, 1], [1, 0]],
        [[1, 1], [1, 1]]
    ),
    Case(
        '#2',
        [[0.05, 0.95], [0.8, 0.2]],
        -0.75, 2,
        False, True, True, True, False,
        None, None,
        1,
        0.36239685545027234, 0.5228281461918625, 0.6931471805599453,
        3.476059496782207, 4.0, 0.25, [np.inf, 3.4760595],
        [],
        [[1, 1], [1, 1]],
        [[1, 1], [1, 1]],
        [[1, 1], [1, 1]]
    ),
    Case(
        '#3',
        [[0.5, 0.5], [0.5, 0.5]],
        0.0, 1,
        False, True, True, True, True,
        None, None,
        1,
        0.6931471805599453, 1.0, 0.6931471805599453,
        None, None, None, [np.inf, 0.02722066],
        [],
        [[1, 1], [1, 1]],
        [[1, 1], [1, 1]],
        [[1, 1], [1, 1]]
    ),
    Case(
        '#4',
        [[1.0, 0.0], [0.1, 0.9]],
        0.9, 2,
        True, False, False, False, False,
        [[10.0]], 10.000000000000002,
        1,
        None, None, 0.0,
        None, None, None, None,
        [],
        [[1, 0], [1, 1]],
        [[1, 0], [1, 1]],
        [[1, 0], [0, 1]]
    ),
    Case(
        '#5',
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        1.0, 3,
        True, True, False, False, True,
        None, None,
        1,
        None, None, 0.0,
        None, None, None, None,
        [[[0], [1, 2]], [[0, 1], [2]]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    ),
    Case(
        '#6',
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        1.0, 3,
        False, True, False, False, False,
        None, None,
        3,
        0.0, 0.0, 8.881784197001248e-16,
        None, None, None, None,
        [],
        [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
        [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    ),
    Case(
        '#7',
        [[0.0, 0.3, 0.7], [0.1, 0.5, 0.4], [0.1, 0.2, 0.7]],
        -0.029999999999999995, 3,
        False, True, True, False, False,
        None, None,
        1,
        0.8267342221751819, 0.8225781143836239, 1.005052538742381,
        0.8305835450825373, 1.4285714285714284, 0.7000000000000001, [np.inf, 0.83058355, 0.43429448],
        [],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[0, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    ),
    Case(
        '#8',
        [[0.0, 1.0, 0.0], [0.3, 0.0, 0.7], [0.0, 1.0, 0.0]],
        0.0, 2,
        False, True, False, True, False,
        None, None,
        2,
        0.30543215102744675, 0.8812908992306926, 0.3465735902799727,
        None, None, None, None,
        [],
        [[1, 1, 0], [1, 1, 1], [0, 1, 1]],
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    ),
    Case(
        '#9',
        [[0.1, 0.1, 0.8], [0.3, 0.3, 0.4], [0.25, 0.5, 0.25]],
        0.05000000000000001, 3,
        False, True, True, False, False,
        None, None,
        1,
        0.963389503699457, 0.8769149167878063, 1.0986122886681098,
        0.6676164013906685, 1.288007155526294, 0.7763932022500208, [np.inf, 0.6676164, 0.6676164],
        [],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    ),
    Case(
        '#10',
        [[0.05, 0.9, 0.05], [0.5, 0.4, 0.1], [0.3, 0.2, 0.5]],
        -0.19, 3,
        False, True, True, False, False,
        None, None,
        1,
        0.7766627625167336, 0.7069489123030946, 1.0986122886681098,
        1.293588988765789, 1.8573766183083227, 0.5383937700856752, [np.inf, 1.29358899, 1.12652175],
        [],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    ),
    Case(
        '#11',
        [[0.1, 0.1, 0.8], [0.3, 0.3, 0.4], [0.25, 0.5, 0.25]],
        0.05000000000000001, 3,
        False, True, True, False, False,
        None, None,
        1,
        0.963389503699457, 0.8769149167878063, 1.0986122886681098,
        0.6676164013906685, 1.288007155526294, 0.7763932022500208, [np.inf, 0.6676164, 0.6676164],
        [],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    ),
    Case(
        '#12',
        [[0.78, 0.22, 0.0], [0.0, 0.1, 0.9], [0.45, 0.0, 0.55]],
        0.132, 3,
        False, True, True, False, False,
        None, None,
        1,
        0.5439208317104742, 0.7847118865449014, 0.6931471805599455,
        0.9876770710214384, 1.5706429060964282, 0.6366819575083007, [np.inf, 0.98767707, 0.98767707],
        [],
        [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
        [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    ),
    Case(
        '#13',
        [[0.5, 0.5, 0.0], [0.5, 0.25, 0.25], [0, 0.33, 0.67]],
        -0.12500000000000003, 3,
        False, True, True, True, False,
        None, None,
        1,
        0.8026275636153916, 0.9106553400693125, 0.8813735870195428,
        2.1005151596269247, 2.6400389096682244, 0.37878229610095815, [np.inf, 2.10051516, 0.6236872],
        [],
        [[1, 1, 0], [1, 1, 1], [0, 1, 1]],
        [[1, 1, 0], [1, 1, 1], [0, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    ),
    Case(
        '#14',
        [[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [0.1, 0.7, 0.2]],
        0.05000000000000001, 3,
        False, True, True, False, False,
        None, None,
        1,
        0.7077237190515739, 0.8947364036861599, 0.7909857206389213,
        0.6676164013906681, 1.2880071555262935, 0.7763932022500211, [np.inf, 0.6676164, 0.6676164],
        [],
        [[1, 1, 0], [0, 1, 1], [1, 1, 1]],
        [[0, 1, 0], [0, 1, 1], [1, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    ),
    Case(
        '#15',
        [[0.0, 0.0, 0.5, 0.5], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        0.0, 3,
        False, True, False, False, False,
        None, None,
        3,
        0.23104906018664842, 0.9999999999999983, 0.2310490601866488,
        None, None, None, None,
        [[[0], [1], [2, 3]]],
        [[1, 0, 1, 1], [1, 1, 0, 0], [0, 1, 1, 0], [0, 1, 0, 1]],
        [[0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    ),
    Case(
        '#16',
        [[0.7, 0.1, 0.1, 0.1], [0.85, 0.05, 0.05, 0.05], [0.2, 0.2, 0.5, 0.1], [0.3, 0.3, 0.3, 0.1]],
        0.002999999999999999, 4,
        False, True, True, False, False,
        None, None,
        1,
        0.981530111623928, 0.7080243122615159, 1.3862943611198906,
        1.043229199083855, 1.6219122507466015, 0.6165561666728137, [np.inf, 1.0432292, 0.41232175, 0.41232175],
        [],
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
        [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    ),
    Case(
        '#17',
        [[0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0, 0.0]],
        0.0, 4,
        False, True, False, False, False,
        None, None,
        3,
        0.23104906018664842, 1.0, 0.2310490601866474,
        None, None, None, None,
        [],
        [[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [1, 0, 1, 1, 0], [0, 0, 0, 1, 1], [0, 0, 1, 0, 1]],
        [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [1, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0]],
        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
    ),
    Case(
        '#18',
        [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.25, 0.75, 0.0, 0.0, 0.0, 0.0], [0.2, 0.0, 0.4, 0.2, 0.2, 0.0], [0.0, 0.0, 0.0, 0.4, 0.6, 0.0], [0.0, 0.0, 0.0, 0.5, 0.0, 0.5], [0.0, 0.0, 0.0, 0.4, 0.0, 0.6]],
        0.005999999999999998, 6,
        False, False, False, False, False,
        None, None,
        1,
        None, None, 0.6931471805599451,
        None, None, None, None,
        [[[0, 1], [2], [3, 4, 5]], [[0], [1], [2], [3, 4, 5]], [[0, 1], [2], [3], [4], [5]]],
        [[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 0, 1]],
        [[0, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [1, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 1, 0, 1], [0, 0, 0, 1, 0, 1]],
        [[1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]]
    ),
    Case(
        '#19',
        [[0.2, 0.2, 0.2, 0.2, 0.0, 0.2], [0.05, 0.75, 0.0, 0.1, 0.0, 0.1], [0.4, 0.6, 0.0, 0.0, 0.0, 0.0], [0.1, 0.1, 0.2, 0.3, 0.0, 0.3], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.3, 0.0, 0.1, 0.1, 0.0, 0.5]],
        -0.0020000000000000005, 6,
        False, True, False, False, False,
        None, None,
        1,
        None, None, 1.4186063950217234,
        None, None, None, None,
        [],
        [[1, 1, 1, 1, 0, 1], [1, 1, 0, 1, 0, 1], [1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 1], [0, 0, 0, 0, 1, 0], [1, 0, 1, 1, 0, 1]],
        [[1, 1, 1, 1, 0, 1], [1, 1, 0, 1, 0, 1], [1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 0, 1], [0, 0, 0, 0, 1, 0], [1, 0, 1, 1, 0, 1]],
        [[1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 0, 1], [0, 0, 0, 0, 1, 0], [1, 1, 1, 1, 0, 1]]
    ),
    Case(
        '#20',
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.6, 0.0, 0.0, 0.4, 0.0, 0.0], [0.0, 0.0, 0.6, 0.0, 0.4, 0.0], [0.0, 0.0, 0.0, 0.6, 0.0, 0.4], [0.0, 0.4, 0.0, 0.0, 0.6, 0.0]],
        0.05760000000000002, 6,
        True, False, False, False, False,
        [[1.54028436, 0.90047393, 0.47393365, 0.18957346], [1.3507109, 2.25118483, 1.18483412, 0.47393365], [1.06635071, 1.77725118, 2.25118483, 0.90047393], [0.63981043, 1.06635071, 1.3507109, 1.54028436]], 7.582938388625593,
        1,
        None, None, 0.4812118250596031,
        None, None, None, None,
        [[[0, 1], [2], [3], [4], [5]]],
        [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [1, 0, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 1], [0, 1, 0, 0, 1, 1]],
        [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [1, 0, 0, 1, 0, 0], [0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 1], [0, 1, 0, 0, 1, 0]],
        [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1]]
    )
]


#########
# TESTS #
#########

@mark.parametrize(
    argnames='p',
    argvalues=[case.p for case in cases],
    ids=['test_irreducibility ' + case.id for case in cases]
)
def test_irreducibility(p):

    mc = MarkovChain(p)

    if not mc.is_irreducible:
        skip('Markov chain is not irreducible.')
    else:

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
    argnames='p',
    argvalues=[case.p for case in cases],
    ids=['test_regularity ' + case.id for case in cases]
)
def test_regularity(p):

    mc = MarkovChain(p)

    if not mc.is_regular:
        skip('Markov chain is not regular.')
    else:

        actual = mc.is_irreducible
        expected = True

        assert actual == expected

        values = np.sort(np.abs(npl.eigvals(mc.p)))
        actual = np.sum(np.logical_or(np.isclose(values, 1.0), values > 1.0))
        expected = 1

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
    argnames=('p', 'is_absorbing', 'is_canonical', 'is_ergodic', 'is_reversible', 'is_symmetric'),
    argvalues=[(case.p, case.is_absorbing, case.is_canonical, case.is_ergodic, case.is_reversible, case.is_symmetric) for case in cases],
    ids=['test_attributes ' + case.id for case in cases]
)
def test_attributes(p, is_absorbing, is_canonical, is_ergodic, is_reversible, is_symmetric):

    mc = MarkovChain(p)

    actual = mc.is_absorbing
    expected = is_absorbing

    assert actual == expected

    actual = mc.is_canonical
    expected = is_canonical

    assert actual == expected

    actual = mc.is_ergodic
    expected = is_ergodic

    assert actual == expected

    actual = mc.is_reversible
    expected = is_reversible

    assert actual == expected

    actual = mc.is_symmetric
    expected = is_symmetric

    assert actual == expected


@mark.parametrize(
    argnames=('p', 'entropy_rate', 'entropy_rate_normalized', 'topological_entropy'),
    argvalues=[(case.p, case.entropy_rate, case.entropy_rate_normalized, case.topological_entropy) for case in cases],
    ids=['test_entropy ' + case.id for case in cases]
)
def test_entropy(p, entropy_rate, entropy_rate_normalized, topological_entropy):

    mc = MarkovChain(p)

    actual = mc.entropy_rate
    expected = entropy_rate

    if actual is not None and expected is not None:
        assert np.isclose(actual, expected)
    else:
        assert actual == expected

    actual = mc.entropy_rate_normalized
    expected = entropy_rate_normalized

    if actual is not None and expected is not None:
        assert np.isclose(actual, expected)
    else:
        assert actual == expected

    actual = mc.topological_entropy
    expected = topological_entropy

    assert np.isclose(actual, expected)


@mark.parametrize(
    argnames=('p', 'fundamental_matrix', 'kemeny_constant'),
    argvalues=[(case.p, case.fundamental_matrix, case.kemeny_constant) for case in cases],
    ids=['test_fundamental_matrix ' + case.id for case in cases]
)
def test_fundamental_matrix(p, fundamental_matrix, kemeny_constant):

    mc = MarkovChain(p)

    actual = mc.fundamental_matrix
    expected = fundamental_matrix

    if actual is not None and expected is not None:
        expected = np.asarray(expected)
        assert np.allclose(actual, expected)
    else:
        assert actual == expected

    actual = mc.kemeny_constant
    expected = kemeny_constant

    if actual is not None and expected is not None:
        assert np.isclose(actual, expected)
    else:
        assert actual == expected


@mark.parametrize(
    argnames=('p', 'lumping_partitions'),
    argvalues=[(case.p, case.lumping_partitions) for case in cases],
    ids=['test_lumping_partitions ' + case.id for case in cases]
)
def test_lumping_partitions(p, lumping_partitions):

    mc = MarkovChain(p)

    actual = mc.lumping_partitions
    expected = lumping_partitions

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
    expected = period == 1

    assert actual == expected


@mark.parametrize(
    argnames=('p', 'mixing_rate', 'relaxation_rate', 'spectral_gap', 'implied_timescales'),
    argvalues=[(case.p, case.mixing_rate, case.relaxation_rate, case.spectral_gap, case.implied_timescales) for case in cases],
    ids=['test_times ' + case.id for case in cases]
)
def test_times(p, mixing_rate, relaxation_rate, spectral_gap, implied_timescales):

    mc = MarkovChain(p)

    actual = mc.mixing_rate
    expected = mixing_rate

    if actual is not None and expected is not None:
        assert np.isclose(actual, expected)
    else:
        assert actual == expected

    actual = mc.relaxation_rate
    expected = relaxation_rate

    if actual is not None and expected is not None:
        assert np.isclose(actual, expected)
    else:
        assert actual == expected

    actual = mc.spectral_gap
    expected = spectral_gap

    if actual is not None and expected is not None:
        assert np.isclose(actual, expected)
    else:
        assert actual == expected

    actual = mc.implied_timescales
    expected = implied_timescales

    if actual is not None and expected is not None:
        expected = np.asarray(expected)
        assert np.allclose(actual, expected)
    else:
        assert actual == expected


@mark.parametrize(
    argnames=('p', 'accessibility_matrix', 'adjacency_matrix', 'communication_matrix'),
    argvalues=[(case.p, case.accessibility_matrix, case.adjacency_matrix, case.communication_matrix) for case in cases],
    ids=['test_binary_matrices ' + case.id for case in cases]
)
def test_binary_matrices(p, accessibility_matrix, adjacency_matrix, communication_matrix):

    mc = MarkovChain(p)

    actual = mc.accessibility_matrix
    expected = np.asarray(accessibility_matrix)

    assert np.array_equal(actual, expected)

    for i in range(mc.size):
        for j in range(mc.size):

            actual = mc.is_accessible(j, i)
            expected = mc.accessibility_matrix[i, j] != 0
            assert actual == expected

            actual = mc.are_communicating(i, j)
            expected = mc.accessibility_matrix[i, j] != 0 and mc.accessibility_matrix[j, i] != 0
            assert actual == expected

    actual = mc.adjacency_matrix
    expected = np.asarray(adjacency_matrix)

    assert np.array_equal(actual, expected)

    actual = mc.communication_matrix
    expected = np.asarray(communication_matrix)

    assert np.array_equal(actual, expected)
