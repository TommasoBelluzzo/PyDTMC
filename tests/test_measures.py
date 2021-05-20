# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########


# Major

import numpy as np

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


Case = namedtuple('Case', ['id', 'p', 'cp_data', 'hit_data', 'mfptb_data', 'mfptt_data', 'mean_absorption_times', 'mean_recurrence_times'])
SubcaseCp = namedtuple('SubcaseCp', ['states1', 'states2', 'value_backward', 'value_forward'])
SubcaseHit = namedtuple('SubcaseHit', ['targets', 'value_probabilities', 'value_times'])
SubcaseMfptb = namedtuple('SubcaseMfptb', ['origins', 'targets', 'value'])
SubcaseMfptt = namedtuple('SubcaseMfptt', ['targets', 'value'])

cases = [
    Case(
        '#1',
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [
            SubcaseCp([0], [1], None, None),
            SubcaseCp([1], [0, 2], None, None)
        ],
        [
            SubcaseHit([0], [1.0, 0.0, 0.0], [0.0, np.inf, np.inf]),
            SubcaseHit([1, 2], [0.0, 1.0, 1.0], [np.inf, 0.0, 0.0])
        ],
        [
            SubcaseMfptb([0], [1], None),
            SubcaseMfptb([1, 2], [0], None)
        ],
        [
            SubcaseMfptt(None, None),
            SubcaseMfptt(0, None)
        ],
        None,
        None
    ),
    Case(
        '#2',
        [[0.6, 0.3, 0.1], [0.2, 0.3, 0.5], [0.4, 0.1, 0.5]],
        [
            SubcaseCp([2], [0], [0.0, 0.19642857, 1.0], [1.0, 0.28571429, 0.0]),
            SubcaseCp([2], [1], [0.73333333, 0.0, 1.0], [0.75, 1.0, 0.0])
        ],
        [
            SubcaseHit([0, 1], [1.0, 1.0, 1.0], [0.0, 0.0, 2.0]),
            SubcaseHit([0, 2], [1.0, 1.0, 1.0], [0.0, 1.42857143, 0.0])
        ],
        [
            SubcaseMfptb([0], [1], 3.75),
            SubcaseMfptb([0, 1], [2], 3.91304347826087)
        ],
        [
            SubcaseMfptt(None, [[0.0, 3.75, 4.54545455], [3.33333333, 0.0, 2.72727273], [2.66666667, 5.0, 0.0]]),
            SubcaseMfptt(2, [4.54545455, 2.72727273, 0.0])
        ],
        None,
        [2.26666667, 4.25, 3.09090909]
    ),
    Case(
        '#3',
        [[0.5, 0.25, 0.25], [0.5, 0.0, 0.5], [0.25, 0.25, 0.50]],
        [
            SubcaseCp([1], [2], [0.5, 1.0, 0.0], [0.5, 0.0, 1.0]),
            SubcaseCp([1, 2], [0], [0.0, 1.0, 1.0], [1.0, 0.0, 0.0])
        ],
        [
            SubcaseHit([1], [1.0, 1.0, 1.0], [4.0, 0.0, 4.0]),
            SubcaseHit([0, 1, 2], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
        ],
        [
            SubcaseMfptb([0], [1], 4.0),
            SubcaseMfptb([1], [0, 2], 1.0)
        ],
        [
            SubcaseMfptt(None, [[0.0, 4.0, 3.33333333], [2.66666667, 0.0, 2.66666667], [3.33333333, 4.0, 0.0]]),
            SubcaseMfptt(1, [4.0, 0.0, 4.0])
        ],
        None,
        [2.5, 5.0, 2.5]
    ),
    Case(
        '#4',
        [[1.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.5, 0.0], [0.0, 0.5, 0.0, 0.5], [0.0, 0.0, 0.0, 1.0]],
        [
            SubcaseCp([0], [2, 3], None, None),
            SubcaseCp([3], [1, 2], None, None)
        ],
        [
            SubcaseHit([0], [1.0, 0.66666667, 0.33333333, 0.0], [0.0, np.inf, np.inf, np.inf]),
            SubcaseHit([0, 1], [1.0, 1.0, 0.5, 0.0], [0.0, 0.0, np.inf, np.inf])
        ],
        [
            SubcaseMfptb([0], [1], None),
            SubcaseMfptb([1], [0, 2], None)
        ],
        [
            SubcaseMfptt(None, None),
            SubcaseMfptt(1, None)
        ],
        [2.0, 2.0],
        None
    ),
    Case(
        '#5',
        [[0.7, 0.1, 0.1, 0.1], [0.85, 0.05, 0.05, 0.05], [0.2, 0.2, 0.5, 0.1], [0.3, 0.3, 0.3, 0.1]],
        [
            SubcaseCp([0], [2], [1.0, 0.64285714, 0.0, 0.75], [0.0, 0.07142857, 1.0, 0.35714286]),
            SubcaseCp([1], [3], [0.74358974, 1.0, 0.53846154, 0.0], [0.46153846, 0.0, 0.38461538, 1.0])
        ],
        [
            SubcaseHit([0], [1.0, 1.0, 1.0, 1.0], [0.0, 1.34920635, 3.05555556, 2.57936508]),
            SubcaseHit([0, 1], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 2.38095238, 1.9047619])
        ],
        [
            SubcaseMfptb([0], [1], 7.1428571428571415),
            SubcaseMfptb([1], [0, 2], 1.130952380952381)
        ],
        [
            SubcaseMfptt(None, [[0.0, 7.14285714, 8.83333333, 10.58333333], [1.34920635, 0.0, 9.33333333, 11.08333333], [3.05555556, 5.95238095, 0.0, 10.66666667], [2.57936508, 5.47619048, 7.16666667, 0.0]]),
            SubcaseMfptt(1, [7.14285714, 0.0, 5.95238095, 5.47619048])
        ],
        None,
        [1.6984127, 7.64285714, 5.35, 10.7]
    ),
    Case(
        '#6',
        [[0.0, 0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.2, 0.8], [0.0, 0.0, 0.0, 0.4, 0.6],  [1.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0, 0.5]],
        [
            SubcaseCp([0], [3, 4], [1.0, 1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0]),
            SubcaseCp([4], [2], [0.77777778, 0.77777778, 0.0, 0.25925926, 1.0], [0.55555556, 0.11111111, 1.0, 0.55555556, 0.0])
        ],
        [
            SubcaseHit([0, 4], [1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 1.2, 1.4, 1.0, 0.0]),
            SubcaseHit([3, 4], [1.0, 1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0, 0.0, 0.0])
        ],
        [
            SubcaseMfptb([0], [1], 4.6),
            SubcaseMfptb([1], [0, 2], 2.8000000000000003)
        ],
        [
            SubcaseMfptt(None, [[0.0, 4.6, 4.8, 11.33333333, 3.28571429], [2.8, 0.0, 7.6, 11.66666667, 1.85714286], [2.6, 7.2, 0.0, 9.0, 2.71428571], [1.0, 5.6, 5.8, 0.0, 4.28571429], [2.0, 6.6, 6.8, 13.33333333, 0.0]]),
            SubcaseMfptt(1, [4.6, 0.0, 7.2, 5.6, 6.6])
        ],
        None,
        [3.7, 7.4, 7.4, 12.33333333, 2.64285714]
    )
]


########
# TEST #
########


@mark.parametrize(
    argnames=('p', 'states1', 'states2', 'value_backward', 'value_forward'),
    argvalues=[(case.p, subcase.states1, subcase.states2, subcase.value_backward, subcase.value_forward) for case in cases for subcase in case.cp_data],
    ids=['test_committor_probabilities ' + case.id + '-' + str(index + 1) for case in cases for (index, subcase) in enumerate(case.cp_data)]
)
def test_committor_probabilities(p, states1, states2, value_backward, value_forward):

    mc = MarkovChain(p)

    actual = mc.committor_probabilities('backward', states1, states2)

    if mc.is_ergodic:

        expected = np.asarray(value_backward)
        assert np.allclose(actual, expected)

    else:

        expected = value_backward
        assert actual == expected

    actual = mc.committor_probabilities('forward', states1, states2)

    if mc.is_ergodic:

        expected = np.asarray(value_forward)
        assert np.allclose(actual, expected)

    else:

        expected = value_forward
        assert actual == expected


@mark.parametrize(
    argnames=('p', 'targets', 'value'),
    argvalues=[(case.p, subcase.targets, subcase.value_probabilities) for case in cases for subcase in case.hit_data],
    ids=['test_hitting_probabilities ' + case.id + '-' + str(index + 1) for case in cases for (index, subcase) in enumerate(case.hit_data)]
)
def test_hitting_probabilities(p, targets, value):

    mc = MarkovChain(p)

    actual = mc.hitting_probabilities(targets)
    expected = np.asarray(value)

    assert np.allclose(actual, expected)

    if mc.is_irreducible:

        expected = np.ones(mc.size, dtype=float)
        assert np.allclose(actual, expected)


@mark.parametrize(
    argnames=('p', 'targets', 'value'),
    argvalues=[(case.p, subcase.targets, subcase.value_times) for case in cases for subcase in case.hit_data],
    ids=['test_hitting_times ' + case.id + '-' + str(index + 1) for case in cases for (index, subcase) in enumerate(case.hit_data)]
)
def test_hitting_times(p, targets, value):

    mc = MarkovChain(p)

    actual = mc.hitting_times(targets)
    expected = np.asarray(value)

    assert np.allclose(actual, expected)


@mark.parametrize(
    argnames=('p', 'origins', 'targets', 'value'),
    argvalues=[(case.p, subcase.origins, subcase.targets, subcase.value) for case in cases for subcase in case.mfptb_data],
    ids=['test_mean_first_passage_times_between ' + case.id + '-' + str(index + 1) for case in cases for (index, subcase) in enumerate(case.mfptb_data)]
)
def test_mean_first_passage_times_between(p, origins, targets, value):

    mc = MarkovChain(p)

    actual = mc.mean_first_passage_times_between(origins, targets)
    expected = np.asarray(value)

    assert actual == expected


@mark.parametrize(
    argnames=('p', 'targets', 'value'),
    argvalues=[(case.p, subcase.targets, subcase.value) for case in cases for subcase in case.mfptt_data],
    ids=['test_mean_first_passage_times_to ' + case.id + '-' + str(index + 1) for case in cases for (index, subcase) in enumerate(case.mfptt_data)]
)
def test_mean_first_passage_times_to(p, targets, value):

    mc = MarkovChain(p)

    actual = mc.mean_first_passage_times_to(targets)
    expected = np.asarray(value)

    if actual is not None and expected is not None:
        assert np.allclose(actual, expected)
    else:
        assert actual == expected


@mark.parametrize(
    argnames=('p', 'mean_absorption_times'),
    argvalues=[(case.p, case.mean_absorption_times) for case in cases],
    ids=['test_mean_absorption_times ' + case.id for case in cases]
)
def test_mean_absorption_times(p, mean_absorption_times):

    mc = MarkovChain(p)

    if mc.is_absorbing and len(mc.transient_states) > 0:

        actual = mc.mean_absorption_times
        expected = np.asarray(mean_absorption_times)

        if actual is not None and expected is not None:
            assert np.allclose(actual, expected)
        else:
            assert actual == expected

        actual = mc.mean_absorption_times.size
        expected = len(mc.absorbing_states)

        assert actual == expected

    else:

        actual = mc.mean_absorption_times
        expected = mean_absorption_times

        assert actual == expected


@mark.parametrize(
    argnames=('p', 'mean_recurrence_times'),
    argvalues=[(case.p, case.mean_recurrence_times) for case in cases],
    ids=['test_mean_recurrence_times ' + case.id for case in cases]
)
def test_mean_recurrence_times(p, mean_recurrence_times):

    mc = MarkovChain(p)

    if mc.is_ergodic:

        actual = mc.mean_recurrence_times
        expected = np.asarray(mean_recurrence_times)

        if actual is not None and expected is not None:
            assert np.allclose(actual, expected)
        else:
            assert actual == expected

        actual = np.nan_to_num(mc.mean_recurrence_times**-1.0)
        expected = np.dot(actual, mc.p)

        assert np.allclose(actual, expected)


@mark.parametrize(
    argnames=('p', 'mfptt_targets'),
    argvalues=[(case.p, subcase.targets) for case in cases for subcase in case.mfptt_data if subcase.targets is None],
    ids=['test_mpft_and_recurrence_relation ' + case.id for case in cases]
)
def test_mpft_and_recurrence_relation(p, mfptt_targets):

    mc = MarkovChain(p)

    mfpt = mc.mean_first_passage_times_to(mfptt_targets)

    if mfpt is not None:

        actual = mfpt
        expected = np.dot(mc.p, mfpt) + np.ones((mc.size, mc.size), dtype=float) - np.diag(mc.mean_recurrence_times)

        assert np.allclose(actual, expected)
