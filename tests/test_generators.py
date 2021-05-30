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

random_seed = 7331

CaseInstance = namedtuple('CaseInstance', [
    'id',
    'p',
    'bounded_data',
    'lazy_data',
    'lump_data',
    'sub_data'
])

SubcaseBounded = namedtuple('SubcaseBounded', ['boundary_condition', 'value'])
SubcaseLazy = namedtuple('SubcaseLazy', ['inertial_weights', 'value'])
SubcaseLump = namedtuple('SubcaseLump', ['partitions', 'value'])
SubcaseSub = namedtuple('SubcaseSub', ['states', 'value'])

cases_instance = [
    CaseInstance(
        '#1',
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [
            SubcaseBounded(0.5, [[0.5, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.5, 0.5]]),
            SubcaseBounded('absorbing', [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            SubcaseBounded('reflecting', [[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        ],
        [
            SubcaseLazy(0.5, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            SubcaseLazy([0.3, 0.01, 0.3], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        ],
        [
            SubcaseLump([[0, 1], [2]], [[1.0, 0.0], [0.0, 1.0]]),
            SubcaseLump([[0], [1, 2]], [[1.0, 0.0], [0.0, 1.0]])
        ],
        [
            SubcaseSub(0, None),
            SubcaseSub([1, 2], [[1.0, 0.0], [0.0, 1.0]])
        ]
    ),
    CaseInstance(
        '#2',
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        [
            SubcaseBounded(0.75, [[0.25, 0.75, 0.0], [0.0, 0.0, 1.0], [0.0, 0.25, 0.75]]),
            SubcaseBounded('absorbing', [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            SubcaseBounded('reflecting', [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        ],
        [
            SubcaseLazy(0.5, [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]]),
            SubcaseLazy([0.0, 0.75, 1.0], [[0.0, 1.0, 0.0], [0.0, 0.75, 0.25], [0.0, 0.0, 1.0]])
        ],
        [
            SubcaseLump([[0, 1], [2]], None),
            SubcaseLump([[0], [1, 2]], None)
        ],
        [
            SubcaseSub(1, [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
            SubcaseSub([0, 2], [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
        ]
    ),

    CaseInstance(
        '#3',
        [[0.2, 0.7, 0.1], [0.1, 0.6, 0.3], [0.2, 0.7, 0.1]],
        [
            SubcaseBounded(0.2, [[0.8, 0.2, 0.0], [0.1, 0.6, 0.3], [0.0, 0.8, 0.2]]),
            SubcaseBounded('absorbing', [[1.0, 0.0, 0.0], [0.1, 0.6, 0.3], [0.0, 0.0, 1.0]]),
            SubcaseBounded('reflecting', [[0.0, 1.0, 0.0], [0.1, 0.6, 0.3], [0.0, 1.0, 0.0]])
        ],
        [
            SubcaseLazy(0.0, [[0.2, 0.7, 0.1], [0.1, 0.6, 0.3], [0.2, 0.7, 0.1]]),
            SubcaseLazy(1.0, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        ],
        [
            SubcaseLump([[0, 1], [2]], None),
            SubcaseLump([[0], [1, 2]], None)
        ],
        [
            SubcaseSub(2, [[0.2, 0.7, 0.1], [0.1, 0.6, 0.3], [0.2, 0.7, 0.1]]),
            SubcaseSub([0, 1], [[0.2, 0.7, 0.1], [0.1, 0.6, 0.3], [0.2, 0.7, 0.1]])
        ]
    ),
    CaseInstance(
        '#4',
        [[0.5, 0.5, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0], [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5]],
        [
            SubcaseBounded(0.95, [[0.05, 0.95, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0], [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.05, 0.95]]),
            SubcaseBounded('absorbing', [[1.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0], [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.0, 1.0]]),
            SubcaseBounded('reflecting', [[0.0, 1.0, 0.0, 0.0], [0.0, 0.5, 0.5, 0.0], [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 1.0, 0.0]])
        ],
        [
            SubcaseLazy([0.2, 0.3, 0.95, 0.95], [[0.6, 0.4, 0.0, 0.0], [0.0, 0.65, 0.35, 0.0], [0.0, 0.0, 0.975, 0.025], [0.0, 0.0, 0.025, 0.975]]),
            SubcaseLazy([0.4, 1.0, 0.0, 0.3], [[0.7, 0.3, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.35, 0.65]])
        ],
        [
            SubcaseLump([[0, 1], [2, 3]], None),
            SubcaseLump([[0], [1, 2, 3]], [[0.5, 0.5], [0.0, 1.0]])
        ],
        [
            SubcaseSub(1, [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.0, 0.5, 0.5]]),
            SubcaseSub(3, [[0.5, 0.5], [0.5, 0.5]])
        ]
    ),
    CaseInstance(
        '#5',
        [[0.1, 0.3, 0.2, 0.3, 0.1], [0.4, 0.0, 0.0, 0.6, 0.0], [0.1, 0.5, 0.2, 0.1, 0.1], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]],
        [
            SubcaseBounded(0.45, [[0.55, 0.45, 0.0, 0.0, 0.0], [0.4, 0.0, 0.0, 0.6, 0.0], [0.1, 0.5, 0.2, 0.1, 0.1], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.55, 0.45]]),
            SubcaseBounded('absorbing', [[1.0, 0.0, 0.0, 0.0, 0.0], [0.4, 0.0, 0.0, 0.6, 0.0], [0.1, 0.5, 0.2, 0.1, 0.1], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]]),
            SubcaseBounded('reflecting', [[0.0, 1.0, 0.0, 0.0, 0.0], [0.4, 0.0, 0.0, 0.6, 0.0], [0.1, 0.5, 0.2, 0.1, 0.1], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0]])
        ],
        [
            SubcaseLazy(0.33, [[0.397, 0.201, 0.134, 0.201, 0.067], [0.268, 0.33, 0.0, 0.402, 0.0], [0.067, 0.335, 0.464, 0.067, 0.067], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]]),
            SubcaseLazy(0.9, [[0.91, 0.03, 0.02, 0.03, 0.01], [0.04, 0.9, 0.0, 0.06, 0.0], [0.01, 0.05, 0.92, 0.01, 0.01], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]])
        ],
        [
            SubcaseLump([[0, 1], [2, 3, 4]], None),
            SubcaseLump([[0], [1], [2], [3, 4]], [[0.1, 0.3, 0.2, 0.4], [0.4, 0.0, 0.0, 0.6], [0.1, 0.5, 0.2, 0.2], [0.0, 0.0, 0.0, 1.0]])
        ],
        [
            SubcaseSub(2, [[0.1, 0.3, 0.2, 0.3, 0.1], [0.4, 0.0, 0.0, 0.6, 0.0], [0.1, 0.5, 0.2, 0.1, 0.1], [0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0]]),
            SubcaseSub(4, None)
        ]
    )
]

CaseStatic = namedtuple('CaseStatic', [
    'birth_death_data',
    'gamblers_ruin_data',
    'random_data',
    'urn_model_data'
])

SubcaseBirthDeath = namedtuple('SubcaseBirthDeath', ['p', 'q', 'value'])
SubcaseGamblersRuin = namedtuple('SubcaseGamblersRuin', ['size', 'w', 'value'])
SubcaseRandom = namedtuple('SubcaseRandom', ['size', 'zeros', 'mask', 'value'])
SubcaseUrnModel = namedtuple('SubcaseUrnModel', ['n', 'model', 'value'])

cases_static = CaseStatic(
    [
        SubcaseBirthDeath([1.0, 0.0], [0.0, 1.0], [[0.0, 1.0], [1.0, 0.0]]),
        SubcaseBirthDeath([0.3, 0.7, 0.0], [0.0, 0.3, 0.3], [[0.7, 0.3, 0.0], [0.3, 0.0, 0.7], [0.0, 0.3, 0.7]]),
        SubcaseBirthDeath([0.0, 0.5, 0.0], [0.0, 0.5, 0.0], [[1.0, 0.0, 0.0], [0.5, 0.0, 0.5], [0.0, 0.0, 1.0]]),
        SubcaseBirthDeath([1.0, 0.3, 0.6, 0.0], [0.0, 0.7, 0.4, 1.0], [[0.0, 1.0, 0.0, 0.0], [0.7, 0.0, 0.3, 0.0], [0.0, 0.4, 0.0, 0.6], [0.0, 0.0, 1.0, 0.0]]),
        SubcaseBirthDeath([0.2, 0.2, 0.7, 0.0], [0.0, 0.8, 0.2, 0.4], [[0.8, 0.2, 0.0, 0.0], [0.8, 0.0, 0.2, 0.0], [0.0, 0.2, 0.1, 0.7], [0.0, 0.0, 0.4, 0.6]])
    ],
    [
        SubcaseGamblersRuin(3, 0.1, [[1.0, 0.0, 0.0], [0.9, 0.0, 0.1], [0.0, 0.0, 1.0]]),
        SubcaseGamblersRuin(3, 0.4, [[1.0, 0.0, 0.0], [0.6, 0.0, 0.4], [0.0, 0.0, 1.0]]),
        SubcaseGamblersRuin(3, 0.8, [[1.0, 0.0, 0.0], [0.2, 0.0, 0.8], [0.0, 0.0, 1.0]]),
        SubcaseGamblersRuin(4, 0.3, [[1.0, 0.0, 0.0, 0.0], [0.7, 0.0, 0.3, 0.0], [0.0, 0.7, 0.0, 0.3], [0.0, 0.0, 0.0, 1.0]]),
        SubcaseGamblersRuin(5, 0.3, [[1.0, 0.0, 0.0, 0.0, 0.0], [0.7, 0.0, 0.3, 0.0, 0.0], [0.0, 0.7, 0.0, 0.3, 0.0], [0.0, 0.0, 0.7, 0.0, 0.3], [0.0, 0.0, 0.0, 0.0, 1.0]])
    ],
    [
        SubcaseRandom(2, 1, None, [[1.0, 0.0], [0.17920243, 0.82079757]]),
        SubcaseRandom(2, 0, [[0.5, np.nan], [np.nan, np.nan]], [[0.5, 0.5], [0.47714488, 0.52285512]]),
        SubcaseRandom(2, 0, [[np.nan, 1.0], [1.0, np.nan]], [[0.0, 1.0], [1.0, 0.0]]),
        SubcaseRandom(3, 3, None, [[0.0, 0.0, 1.0], [0.41251495, 0.58748505, 0.0], [0.37580972, 0.22746209, 0.39672819]]),
        SubcaseRandom(3, 6, None, [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        SubcaseRandom(3, 0, [[1.0, 0.0, 0.0], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]], [[1.0, 0.0, 0.0], [0.14331205, 0.40220784, 0.45448012], [0.6003005, 0.10145724, 0.29824226]]),
        SubcaseRandom(3, 2, [[np.nan, np.nan, np.nan], [0.05, np.nan, 0.05], [0.3, np.nan, np.nan]], [[0.0, 0.58544476, 0.41455524], [0.05, 0.9, 0.05], [0.3, 0.0, 0.7]]),
        SubcaseRandom(4, 0, None, [[0.23012099, 0.6032467, 0.00998088, 0.15665143], [0.19814307, 0.39753383, 0.04223579, 0.36208731], [0.40389448, 0.51418304, 0.07435139, 0.00757109], [0.15594156, 0.41004987, 0.19492674, 0.23908183]]),
        SubcaseRandom(4, 3, [[np.nan, 1.0, np.nan, np.nan], [np.nan, np.nan, 0.2, 0.1], [np.nan, np.nan, np.nan, np.nan], [0.5, np.nan, 0.4, np.nan]], [[0.0, 1.0, 0.0, 0.0], [0.7, 0.0, 0.2, 0.1], [0.34470209, 0.0, 0.51437746, 0.14092046], [0.5, 0.0, 0.4, 0.1]]),
        SubcaseRandom(5, 3, None, [[0.12216488, 0.27473981, 0.30369964, 0.28944879, 0.00994688], [0.34207151, 0.0, 0.0, 0.43017028, 0.22775821], [0.42579337, 0.05766587, 0.1253785, 0.2437706, 0.14739167], [0.22169328, 0.05810026, 0.25874351, 0.23953959, 0.22192335], [0.26783291, 0.01749013, 0.63296795, 0.0, 0.08170901]])
    ],
    [
        SubcaseUrnModel(1, 'bernoulli-laplace', [[0.0, 1.0, 0.0], [0.25, 0.5, 0.25], [0.0, 1.0, 0.0]]),
        SubcaseUrnModel(1, 'ehrenfest', [[0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]]),
        SubcaseUrnModel(2, 'bernoulli-laplace', [[0.0, 1.0, 0.0, 0.0, 0.0], [0.0625, 0.375, 0.5625, 0.0, 0.0], [0.0, 0.25, 0.5, 0.25, 0.0], [0.0, 0.0, 0.5625, 0.375, 0.0625], [0.0, 0.0, 0.0, 1.0, 0.0]]),
        SubcaseUrnModel(2, 'ehrenfest', [[0.0, 1.0, 0.0, 0.0, 0.0], [0.25, 0.0, 0.75, 0.0, 0.0], [0.0, 0.5, 0.0, 0.5, 0.0], [0.0, 0.0, 0.75, 0.0, 0.25], [0.0, 0.0, 0.0, 1.0, 0.0]]),
        SubcaseUrnModel(3, 'ehrenfest', [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.16666667, 0.0, 0.83333333, 0.0, 0.0, 0.0, 0.0], [0.0, 0.33333333, 0.0, 0.66666667, 0.0, 0.0, 0.0], [0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.66666667, 0.0, 0.33333333, 0.0], [0.0, 0.0, 0.0, 0.0, 0.83333333, 0.0, 0.16666667], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    ]
)


########
# TEST #
########


@mark.parametrize(
    argnames=('p', 'boundary_condition', 'value'),
    argvalues=[(case.p, subcase.boundary_condition, subcase.value) for case in cases_instance for subcase in case.bounded_data],
    ids=['test_bounded ' + case.id + '-' + str(index + 1) for case in cases_instance for (index, subcase) in enumerate(case.bounded_data)]
)
def test_bounded(p, boundary_condition, value):

    mc = MarkovChain(p)
    mc_bounded = mc.to_bounded_chain(boundary_condition)

    actual = mc_bounded.p
    expected = np.asarray(value)

    assert np.allclose(actual, expected)


@mark.parametrize(
    argnames=('p', 'inertial_weights', 'value'),
    argvalues=[(case.p, subcase.inertial_weights, subcase.value) for case in cases_instance for subcase in case.lazy_data],
    ids=['test_lazy ' + case.id + '-' + str(index + 1) for case in cases_instance for (index, subcase) in enumerate(case.lazy_data)]
)
def test_lazy(p, inertial_weights, value):

    mc = MarkovChain(p)
    mc_lazy = mc.to_lazy_chain(inertial_weights)

    actual = mc_lazy.p
    expected = np.asarray(value)

    assert np.allclose(actual, expected)


@mark.parametrize(
    argnames=('p', 'partitions', 'value'),
    argvalues=[(case.p, subcase.partitions, subcase.value) for case in cases_instance for subcase in case.lump_data],
    ids=['test_lump ' + case.id + '-' + str(index + 1) for case in cases_instance for (index, subcase) in enumerate(case.lump_data)]
)
def test_lump(p, partitions, value):

    mc = MarkovChain(p)

    if value is not None:

        mc_lazy = mc.lump(partitions)

        actual = mc_lazy.p
        expected = np.asarray(value)

        assert np.allclose(actual, expected)


@mark.parametrize(
    argnames=('p', 'states', 'value'),
    argvalues=[(case.p, subcase.states, subcase.value) for case in cases_instance for subcase in case.sub_data],
    ids=['test_sub ' + case.id + '-' + str(index + 1) for case in cases_instance for (index, subcase) in enumerate(case.sub_data)]
)
def test_sub(p, states, value):

    mc = MarkovChain(p)

    exception = False

    try:
        mc_sub = mc.to_subchain(states)
    except ValueError:
        mc_sub = None
        exception = True

    if value is None:

        assert exception is True

    else:

        actual = mc_sub.p
        expected = np.asarray(value)

        assert np.allclose(actual, expected)


@mark.parametrize(
    argnames=('p', 'q', 'value'),
    argvalues=[(case.p, case.q, case.value) for case in cases_static.birth_death_data],
    ids=['test_birth_death #' + str(index + 1) for (index, _) in enumerate(cases_static.birth_death_data)]
)
def test_birth_death(p, q, value):

    mc = MarkovChain.birth_death(p, q)

    actual = mc.p
    expected = np.asarray(value)

    assert np.allclose(actual, expected)


@mark.parametrize(
    argnames=('size', 'w', 'value'),
    argvalues=[(case.size, case.w, case.value) for case in cases_static.gamblers_ruin_data],
    ids=['test_gamblers_ruin #' + str(index + 1) for (index, _) in enumerate(cases_static.gamblers_ruin_data)]
)
def test_gamblers_ruin(size, w, value):

    mc = MarkovChain.gamblers_ruin(size, w)

    actual = mc.p
    expected = np.asarray(value)

    assert np.allclose(actual, expected)


@mark.parametrize(
    argnames=('size', 'zeros', 'mask', 'seed', 'value'),
    argvalues=[(case.size, case.zeros, case.mask, random_seed, case.value) for case in cases_static.random_data],
    ids=['test_random #' + str(index + 1) for (index, _) in enumerate(cases_static.random_data)]
)
def test_random(size, zeros, mask, seed, value):

    mc = MarkovChain.random(size, None, zeros, mask, seed)

    actual = mc.p
    expected = np.asarray(value)

    assert np.allclose(actual, expected)

    if zeros > 0 and mask is None:

        actual = size**2 - np.count_nonzero(mc.p)
        expected = zeros

        assert actual == expected

    if mask is not None:

        indices = ~np.isnan(np.asarray(mask))

        actual = mc.p[indices]
        expected = np.asarray(value)[indices]

        assert np.allclose(actual, expected)


@mark.parametrize(
    argnames=('n', 'model', 'value'),
    argvalues=[(case.n, case.model, case.value) for case in cases_static.urn_model_data],
    ids=['test_urn_model #' + str(index + 1) for (index, _) in enumerate(cases_static.urn_model_data)]
)
def test_urn_model(n, model, value):

    mc = MarkovChain.urn_model(n, model)

    actual = mc.p
    expected = np.asarray(value)

    assert np.allclose(actual, expected)
