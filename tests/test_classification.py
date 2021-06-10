# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Full

import numpy as np

# Partial

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

identity_maximum_size = 20

Case = namedtuple('Case', [
    'id',
    'p',
    'recurrent_classes',
    'transient_classes',
    'communicating_classes',
    'cyclic_classes',
    'absorbing_states'
])

cases = [
    Case(
        '#1',
        [[1.0, 0.0], [0.0, 1.0]],
        [['1'], ['2']], [],
        [['1'], ['2']], [],
        ['1', '2']
    ),
    Case(
        '#2',
        [[0.05, 0.95], [0.8, 0.2]],
        [['1', '2']], [],
        [['1', '2']], [['1', '2']],
        []
    ),
    Case(
        '#3',
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [['1'], ['2'], ['3']], [],
        [['1'], ['2'], ['3']], [],
        ['1', '2', '3']
    ),
    Case(
        '#4',
        [[0.4, 0.3, 0.3], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5]],
        [['2']], [['1', '3']],
        [['1', '3'], ['2']], [],
        ['2']
    ),
    Case(
        '#5',
        [[0.05, 0.85, 0.1], [0.1, 0.8, 0.1], [0.3, 0.4, 0.3]],
        [['1', '2', '3']], [],
        [['1', '2', '3']], [['1', '2', '3']],
        []
    ),
    Case(
        '#6',
        [[0.0, 0.0, 0.5, 0.5], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        [['1', '2', '3', '4']], [],
        [['1', '2', '3', '4']], [['3', '4'], ['1'], ['2']],
        []
    ),
    Case(
        '#7',
        [[0.1, 0.9, 0.0, 0.0], [0.2, 0.0, 0.8, 0.0], [0.3, 0.0, 0.0, 0.7], [0.4, 0.0, 0.0, 0.6]],
        [['1', '2', '3', '4']], [],
        [['1', '2', '3', '4']], [['1', '2', '3', '4']],
        []
    ),
    Case(
        '#8',
        [[0.5, 0.5, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]],
        [['1', '2'], ['3']], [['4'], ['5']],
        [['1', '2'], ['3'], ['4'], ['5']], [],
        ['3']
    ),
    Case(
        '#9',
        [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.25, 0.75, 0.0, 0.0, 0.0, 0.0], [0.2, 0.0, 0.4, 0.2, 0.2, 0.0], [0.0, 0.0, 0.0, 0.4, 0.6, 0.0], [0.0, 0.0, 0.0, 0.5, 0.0, 0.5], [0.0, 0.0, 0.0, 0.4, 0.0, 0.6]],
        [['4', '5', '6'], ['1', '2']], [['3']],
        [['4', '5', '6'], ['1', '2'], ['3']], [],
        []
    ),
    Case(
        '#10',
        [[0.0, 0.4, 0.0, 0.6, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.5, 0.5, 0.0]],
        [['4', '5', '6'], ['2', '3']], [['1']],
        [['4', '5', '6'], ['2', '3'], ['1']], [],
        []
    )
]


#########
# TESTS #
#########

@mark.parametrize(
    argnames=('p', 'recurrent_classes'),
    argvalues=[(case.p, case.recurrent_classes) for case in cases],
    ids=['test_classes_recurrent ' + case.id for case in cases]
)
def test_classes_recurrent(p, recurrent_classes):

    mc = MarkovChain(p)

    actual = mc.recurrent_classes
    expected = recurrent_classes

    assert actual == expected

    actual = sum([len(i) for i in mc.recurrent_classes])
    expected = len(set([state for recurrent_class in recurrent_classes for state in recurrent_class]))

    assert actual == expected


@mark.parametrize(
    argnames=('p', 'transient_classes'),
    argvalues=[(case.p, case.transient_classes) for case in cases],
    ids=['test_classes_transient ' + case.id for case in cases]
)
def test_classes_transient(p, transient_classes):

    mc = MarkovChain(p)

    actual = mc.transient_classes
    expected = transient_classes

    assert actual == expected

    actual = sum([len(i) for i in mc.transient_classes])
    expected = len(set([state for transient_class in transient_classes for state in transient_class]))

    assert actual == expected


@mark.parametrize(
    argnames=('p', 'communicating_classes'),
    argvalues=[(case.p, case.communicating_classes) for case in cases],
    ids=['test_classes_communicating ' + case.id for case in cases]
)
def test_classes_communicating(p, communicating_classes):

    mc = MarkovChain(p)

    actual = mc.communicating_classes
    expected = communicating_classes

    assert actual == expected

    if len(communicating_classes) > 1:
        assert np.array_equal(mc.communication_matrix, np.transpose(mc.communication_matrix))


@mark.parametrize(
    argnames='maximum_size',
    argvalues=[identity_maximum_size],
    ids=['classes_communicating_identity']
)
def test_classes_communicating_identity(maximum_size):

    for size in range(2, maximum_size + 1):

        mc = MarkovChain.identity(size)

        actual = [state for states in mc.communicating_classes for state in states]
        expected = mc.states

        assert actual == expected


@mark.parametrize(
    argnames=('p', 'cyclic_classes'),
    argvalues=[(case.p, case.cyclic_classes) for case in cases],
    ids=['test_classes_cyclic ' + case.id for case in cases]
)
def test_classes_cyclic(p, cyclic_classes):

    mc = MarkovChain(p)

    actual = mc.cyclic_classes
    expected = cyclic_classes

    assert actual == expected


@mark.parametrize(
    argnames=('p', 'recurrent_classes', 'transient_classes'),
    argvalues=[(case.p, case.recurrent_classes, case.transient_classes) for case in cases],
    ids=['test_states_space ' + case.id for case in cases]
)
def test_states_space(p, recurrent_classes, transient_classes):

    mc = MarkovChain(p)

    actual = mc.recurrent_states
    expected = sorted([state for recurrent_class in recurrent_classes for state in recurrent_class])

    assert actual == expected

    actual = mc.transient_states
    expected = sorted([state for transient_class in transient_classes for state in transient_class])

    assert actual == expected

    actual = sorted(mc.recurrent_states + mc.transient_states)
    expected = mc.states

    assert actual == expected

    if len(mc.recurrent_states) > 0:
        for state in mc.recurrent_states:
            assert mc.is_recurrent_state(state) is True

    if len(mc.transient_states) > 0:
        for state in mc.transient_states:
            assert mc.is_transient_state(state) is True


@mark.parametrize(
    argnames=('p', 'recurrent_classes', 'absorbing_states'),
    argvalues=[(case.p, case.recurrent_classes, case.absorbing_states) for case in cases],
    ids=['test_states_absorbing ' + case.id for case in cases]
)
def test_states_absorbing(p, recurrent_classes, absorbing_states):

    mc = MarkovChain(p)

    actual = mc.absorbing_states
    expected = absorbing_states

    assert actual == expected

    actual = sum([1 if len(recurrent_class) == 1 and recurrent_class[0] in actual else 0 for recurrent_class in recurrent_classes])
    expected = len(absorbing_states)

    assert actual == expected

    if len(mc.absorbing_states) > 0:
        for state in mc.absorbing_states:
            assert mc.is_absorbing_state(state) is True


@mark.parametrize(
    argnames='p',
    argvalues=[case.p for case in cases],
    ids=['test_states_cyclic ' + case.id for case in cases]
)
def test_states_cyclic(p):

    mc = MarkovChain(p)

    if len(mc.cyclic_states) > 0:
        for state in mc.cyclic_states:
            assert mc.is_cyclic_state(state) is True


@mark.parametrize(
    argnames='maximum_size',
    argvalues=[identity_maximum_size],
    ids=['test_states_recurrent_identity']
)
def test_states_recurrent_identity(maximum_size):

    for size in range(2, maximum_size + 1):

        mc = MarkovChain.identity(size)

        actual = mc.recurrent_states
        expected = mc.states

        assert actual == expected
