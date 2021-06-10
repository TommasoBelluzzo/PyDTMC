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
    mark,
    skip
)


##############
# TEST CASES #
##############

Case = namedtuple('Case', [
    'id',
    'p',
    'committor_data',
    'first_passage_data',
    'hitting_data',
    'mfptb_data',
    'mfptt_data',
    'mixing_data',
    'sensitivity_data',
    'absorption_probabilities',
    'mean_absorption_times',
    'mean_number_visits',
    'mean_recurrence_times'
])

SubcaseCommittor = namedtuple('SubcaseCommittor', ['states1', 'states2', 'value_backward', 'value_forward'])
SubcaseFirstPassage = namedtuple('SubcaseFirstPassage', ['steps', 'initial_state', 'first_passage_states_probabilities', 'first_passage_states_reward', 'rewards', 'value_probabilities', 'value_rewards'])
SubcaseHitting = namedtuple('SubcaseHitting', ['targets', 'value_probabilities', 'value_times'])
SubcaseMfptb = namedtuple('SubcaseMfptb', ['origins', 'targets', 'value'])
SubcaseMfptt = namedtuple('SubcaseMfptt', ['targets', 'value'])
SubcaseMixing = namedtuple('SubcaseMixing', ['initial_distribution', 'jump', 'cutoff_type', 'value'])
SubcaseSensitivity = namedtuple('SubcaseSensitivity', ['state', 'value'])

cases = [
    Case(
        '#1',
        [[0.0, 1.0], [1.0, 0.0]],
        [
            SubcaseCommittor(0, 1, None, None),
            SubcaseCommittor(1, 0, None, None)
        ],
        [
            SubcaseFirstPassage(2, 0, None, None, None, [[0.0, 1.0], [1.0, 0.0]], None),
            SubcaseFirstPassage(2, 1, 0, None, None, [1.0, 0.0], None)
        ],
        [
            SubcaseHitting(None, [1.0, 1.0], [0.0, 0.0]),
            SubcaseHitting(1, [1.0, 1.0], [1.0, 0.0])
        ],
        [
            SubcaseMfptb(0, 1, None),
            SubcaseMfptb(1, 0, None)
        ],
        [
            SubcaseMfptt(None, None),
            SubcaseMfptt(0, None)
        ],
        [
            SubcaseMixing(None, 1, 'natural', None),
            SubcaseMixing([0.2, 0.8], 3, 'natural', None)
        ],
        [
            SubcaseSensitivity(0, [[0.125, -0.125], [0.125, -0.125]]),
            SubcaseSensitivity(1, [[-0.125, 0.125], [-0.125, 0.125]])
        ],
        None,
        None,
        [[np.Inf, np.Inf], [np.Inf, np.Inf]],
        None
    ),
    Case(
        '#2',
        [[0.3, 0.7], [0.25, 0.75]],
        [
            SubcaseCommittor(0, 1, [1.0, 0.0], [0.0, 1.0]),
            SubcaseCommittor(1, 0, [0.0, 1.0], [1.0, 0.0])
        ],
        [
            SubcaseFirstPassage(3, 0, 1, None, None, [0.7, 0.21, 0.063], None),
            SubcaseFirstPassage(6, 1, None, None, None, [[0.25, 0.75], [0.1875, 0.175], [0.140625, 0.0525], [0.10546875, 0.01575], [0.07910156, 0.004725], [0.05932617, 0.0014175]], None)
        ],
        [
            SubcaseHitting(0, [1.0, 1.0], [0.0, 4.0]),
            SubcaseHitting(1, [1.0, 1.0], [1.42857143, 0.0])
        ],
        [
            SubcaseMfptb(0, 1, 1.4285714285714286),
            SubcaseMfptb(1, 0, 4.0)
        ],
        [
            SubcaseMfptt(None, [[0.0, 1.42857143], [4.0, 0.0]]),
            SubcaseMfptt(1, [1.42857143, 0.0])
        ],
        [
            SubcaseMixing(None, 8, 'natural', 8),
            SubcaseMixing([0.5, 0.5], 1, 'natural', 1)
        ],
        [
            SubcaseSensitivity(0, [[0.20411139, -0.07289692], [0.57151188, -0.20411139]]),
            SubcaseSensitivity(1, [[-0.20411139, 0.07289692], [-0.57151188, 0.20411139]])
        ],
        None,
        None,
        [[np.Inf, np.Inf], [np.Inf, np.Inf]],
        [3.8, 1.35714286]
    ),
    Case(
        '#3',
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [
            SubcaseCommittor(0, 1, None, None),
            SubcaseCommittor(1, [0, 2], None, None)
        ],
        [
            SubcaseFirstPassage(5, 0, None, 1, [7.3, 10.0, 0.3], [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], 36.5),
            SubcaseFirstPassage(7, 0, 2, 1, [4.7, 1.0, 3.9], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 32.9)
        ],
        [
            SubcaseHitting(None, [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]),
            SubcaseHitting([1, 2], [0.0, 1.0, 1.0], [np.inf, 0.0, 0.0])
        ],
        [
            SubcaseMfptb(0, 1, None),
            SubcaseMfptb([1, 2], 0, None)
        ],
        [
            SubcaseMfptt(None, None),
            SubcaseMfptt(0, None)
        ],
        [
            SubcaseMixing(None, 5, 'natural', None),
            SubcaseMixing([0.3, 0.2, 0.5], 1, 'natural', None)
        ],
        [
            SubcaseSensitivity(0, None),
            SubcaseSensitivity(2, None)
        ],
        None,
        None,
        [[np.inf, 0.0, 0.0], [0.0, np.inf, 0.0], [0.0, 0.0, np.inf]],
        None
    ),
    Case(
        '#4',
        [[0.6, 0.3, 0.1], [0.2, 0.3, 0.5], [0.4, 0.1, 0.5]],
        [
            SubcaseCommittor(2, 0, [0.0, 0.19642857, 1.0], [1.0, 0.28571429, 0.0]),
            SubcaseCommittor(2, 1, [0.73333333, 0.0, 1.0], [0.75, 1.0, 0.0])
        ],
        [
            SubcaseFirstPassage(3, 1, None, 0, [0.0, 0.5, 0.0], [[0.2, 0.3, 0.5], [0.26, 0.11, 0.17], [0.188, 0.123, 0.093]], 0.261),
            SubcaseFirstPassage(7, 1, 2, 2, [2.1, 2.1, 2.1], [0.5, 0.17, 0.093, 0.0633, 0.04581, 0.033633, 0.0247725], 3.2178961500000005)
        ],
        [
            SubcaseHitting([0, 1], [1.0, 1.0, 1.0], [0.0, 0.0, 2.0]),
            SubcaseHitting([0, 2], [1.0, 1.0, 1.0], [0.0, 1.42857143, 0.0])
        ],
        [
            SubcaseMfptb(0, 1, 3.75),
            SubcaseMfptb([0, 1], 2, 3.91304347826087)
        ],
        [
            SubcaseMfptt(None, [[0.0, 3.75, 4.54545455], [3.33333333, 0.0, 2.72727273], [2.66666667, 5.0, 0.0]]),
            SubcaseMfptt(2, [4.54545455, 2.72727273, 0.0])
        ],
        [
            SubcaseMixing(None, 3, 'natural', 3),
            SubcaseMixing([0.05, 0.9, 0.05], 1, 'traditional', 2)
        ],
        [
            SubcaseSensitivity(0, [[0.32057806, -0.32821087, -0.19845308], [0.17097496, -0.1750458, -0.10584164], [0.23509058, -0.24068797, -0.14553226]]),
            SubcaseSensitivity(1, [[-0.04961327, 0.33966009, -0.17937106], [-0.02646041, 0.18115205, -0.09566456], [-0.03638307, 0.24908406, -0.13153877]])
        ],
        None,
        None,
        [[np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf]],
        [2.26666667, 4.25, 3.09090909]
    ),
    Case(
        '#5',
        [[0.5, 0.25, 0.25], [0.5, 0.0, 0.5], [0.25, 0.25, 0.50]],
        [
            SubcaseCommittor(1, 2, [0.5, 1.0, 0.0], [0.5, 0.0, 1.0]),
            SubcaseCommittor([1, 2], 0, [0.0, 1.0, 1.0], [1.0, 0.0, 0.0])
        ],
        [
            SubcaseFirstPassage(1, 0, 1, 1, [0.4, 6.1, 9.7], [0.25], 2.625),
            SubcaseFirstPassage(4, 0, None, 1, [0.4, 6.1, 9.7], [[0.5, 0.25, 0.25], [0.1875, 0.1875, 0.25], [0.09375, 0.140625, 0.15625], [0.0703125, 0.10546875, 0.109375]], 8.8125)
        ],
        [
            SubcaseHitting(1, [1.0, 1.0, 1.0], [4.0, 0.0, 4.0]),
            SubcaseHitting([0, 1, 2], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
        ],
        [
            SubcaseMfptb(0, 1, 4.0),
            SubcaseMfptb(1, [0, 2], 1.0)
        ],
        [
            SubcaseMfptt(None, [[0.0, 4.0, 3.33333333], [2.66666667, 0.0, 2.66666667], [3.33333333, 4.0, 0.0]]),
            SubcaseMfptt([1, 2], [2.0, 0.0, 0.0])
        ],
        [
            SubcaseMixing(None, 1, 'natural', 1),
            SubcaseMixing([0.3, 0.6, 0.1], 5, 'natural', 10)
        ],
        [
            SubcaseSensitivity(1, [[-0.064, 0.256, -0.064], [-0.032, 0.128, -0.032], [-0.064, 0.256, -0.064]]),
            SubcaseSensitivity(2, [[-0.23466667, -0.128, 0.29866667], [-0.11733333, -0.064, 0.14933333], [-0.23466667, -0.128, 0.29866667]])
        ],
        None,
        None,
        [[np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf]],
        [2.5, 5.0, 2.5]
    ),
    Case(
        '#6',
        [[1.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.5, 0.0], [0.0, 0.5, 0.0, 0.5], [0.0, 0.0, 0.0, 1.0]],
        [
            SubcaseCommittor(0, [2, 3], None, None),
            SubcaseCommittor(3, [1, 2], None, None)
        ],
        [
            SubcaseFirstPassage(2, 1, None, [0, 3], [0.4, 6.1, 9.7, 9.7], [[0.5, 0.0, 0.5, 0.0], [0.0, 0.25, 0.0, 0.25]], 6.375),
            SubcaseFirstPassage(6, 2, [0, 1], [0, 1], [5.2, 0.2, 0.0, 1.5], [0.5, 0.25, 0.0, 0.0625, 0.0, 0.015625], 4.5)
        ],
        [
            SubcaseHitting(None, [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]),
            SubcaseHitting([0, 1], [1.0, 1.0, 0.5, 0.0], [0.0, 0.0, np.inf, np.inf])
        ],
        [
            SubcaseMfptb(0, 1, None),
            SubcaseMfptb(1, [0, 2], None)
        ],
        [
            SubcaseMfptt(None, None),
            SubcaseMfptt(1, None)
        ],
        [
            SubcaseMixing(None, 4, 'natural', None),
            SubcaseMixing([0.1, 0.6, 0.1, 0.2], 2, 'traditional', None)
        ],
        [
            SubcaseSensitivity(2, None),
            SubcaseSensitivity(3, None)
        ],
        [[0.66666667, 0.33333333], [0.33333333, 0.66666667]],
        [2.0, 2.0],
        [[np.inf, 0.0, 0.0, 0.0], [np.inf, 0.33333333, 0.66666667, np.inf], [np.inf, 0.66666667, 0.33333333, np.inf], [0.0, 0.0, 0.0, np.inf]],
        None
    ),
    Case(
        '#7',
        [[0.7, 0.1, 0.1, 0.1], [0.85, 0.05, 0.05, 0.05], [0.2, 0.2, 0.5, 0.1], [0.3, 0.3, 0.3, 0.1]],
        [
            SubcaseCommittor(0, 2, [1.0, 0.64285714, 0.0, 0.75], [0.0, 0.07142857, 1.0, 0.35714286]),
            SubcaseCommittor(1, 3, [0.74358974, 1.0, 0.53846154, 0.0], [0.46153846, 0.0, 0.38461538, 1.0])
        ],
        [
            SubcaseFirstPassage(2, 0, [1, 2], [1, 2], [7.1, 3.5, 8.1, 3.5], [0.2, 0.225], 9.291999999999998),
            SubcaseFirstPassage(4, 3, None, [1, 2], [7.1, 3.5, 8.1, 3.5], [[0.3, 0.3, 0.3, 0.1], [0.345, 0.12, 0.075, 0.075], [0.14475, 0.093, 0.06975, 0.07725], [0.0845625, 0.0759, 0.0637875, 0.0699375]], 6.727039999999999)
        ],
        [
            SubcaseHitting(0, [1.0, 1.0, 1.0, 1.0], [0.0, 1.34920635, 3.05555556, 2.57936508]),
            SubcaseHitting([0, 1], [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 2.38095238, 1.9047619])
        ],
        [
            SubcaseMfptb(0, 1, 7.1428571428571415),
            SubcaseMfptb(1, [0, 2], 1.130952380952381)
        ],
        [
            SubcaseMfptt(None, [[0.0, 7.14285714, 8.83333333, 10.58333333], [1.34920635, 0.0, 9.33333333, 11.08333333], [3.05555556, 5.95238095, 0.0, 10.66666667], [2.57936508, 5.47619048, 7.16666667, 0.0]]),
            SubcaseMfptt([1, 3], [4.61538462, 0.0, 3.84615385, 0.0])
        ],
        [
            SubcaseMixing(None, 1, 'natural', 1),
            SubcaseMixing([0.4, 0.3, 0.0, 0.3], 1, 'traditional', 1)
        ],
        [
            SubcaseSensitivity(0, [[0.3427594, -0.12496704, -0.71650342, -0.5514235], [0.07616875, -0.02777045, -0.15922298, -0.12253856], [0.10881251, -0.03967208, -0.2274614, -0.17505508], [0.05440625, -0.01983604, -0.1137307, -0.08752754]]),
            SubcaseSensitivity(2, [[-0.19165042, -0.24667706, 0.78048689, -0.00822828], [-0.04258898, -0.05481712, 0.17344153, -0.00182851], [-0.0608414, -0.07831018, 0.24777362, -0.00261215], [-0.0304207, -0.03915509, 0.12388681, -0.00130608]])
        ],
        None,
        None,
        [[np.inf, np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf, np.inf]],
        [1.6984127, 7.64285714, 5.35, 10.7]
    ),
    Case(
        '#8',
        [[0.0, 0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.2, 0.8], [0.0, 0.0, 0.0, 0.4, 0.6],  [1.0, 0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.0, 0.5]],
        [
            SubcaseCommittor(0, [3, 4], [1.0, 1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0]),
            SubcaseCommittor(4, 2, [0.77777778, 0.77777778, 0.0, 0.25925926, 1.0], [0.55555556, 0.11111111, 1.0, 0.55555556, 0.0])
        ],
        [
            SubcaseFirstPassage(4, 1, None, 3, [0.0, 0.0, 4.2, 0.0, 0.0], [[0.0, 0.0, 0.0, 0.2, 0.8], [0.6, 0.0, 0.0, 0.0, 0.0], [0.2, 0.3, 0.3, 0.0, 0.0], [0.1, 0.1, 0.1, 0.12, 0.14]], 1.2600000000000002),
            SubcaseFirstPassage(4, 3, [0, 1, 2], [0, 1, 2], [0.9, 2.4, 1.2, 1.2, 1.2], [1.0, 1.0, 0.0, 0.0], 0.0)
        ],
        [
            SubcaseHitting([0, 4], [1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 1.2, 1.4, 1.0, 0.0]),
            SubcaseHitting([3, 4], [1.0, 1.0, 1.0, 1.0, 1.0], [2.0, 1.0, 1.0, 0.0, 0.0])
        ],
        [
            SubcaseMfptb(0, 1, 4.6),
            SubcaseMfptb(1, [0, 2], 2.8000000000000003)
        ],
        [
            SubcaseMfptt(None, [[0.0, 4.6, 4.8, 11.33333333, 3.28571429], [2.8, 0.0, 7.6, 11.66666667, 1.85714286], [2.6, 7.2, 0.0, 9.0, 2.71428571], [1.0, 5.6, 5.8, 0.0, 4.28571429], [2.0, 6.6, 6.8, 13.33333333, 0.0]]),
            SubcaseMfptt(1, [4.6, 0.0, 7.2, 5.6, 6.6])
        ],
        [
            SubcaseMixing(None, 2, 'natural', 2),
            SubcaseMixing([0.1, 0.4, 0.1, 0.2, 0.2], 4, 'natural', 16)
        ],
        [
            SubcaseSensitivity(1, [[0.02072928, 0.18873512, -0.07423055, -0.01579373, -0.05231674], [0.01036464, 0.09436756, -0.03711527, -0.00789687, -0.02615837], [0.01036464, 0.09436756, -0.03711527, -0.00789687, -0.02615837], [0.00621878, 0.05662054, -0.02226916, -0.00473812, -0.01569502], [0.02902099, 0.26422917, -0.10392277, -0.02211123, -0.07324344]]),
            SubcaseSensitivity(3, [[-0.00947624, -0.01678084, 0.04165597, 0.23888022, -0.05330385], [-0.00473812, -0.00839042, 0.02082799, 0.11944011, -0.02665193], [-0.00473812, -0.00839042, 0.02082799, 0.11944011, -0.02665193], [-0.00284287, -0.00503425, 0.01249679, 0.07166407, -0.01599116], [-0.01326674, -0.02349318, 0.05831836, 0.33443231, -0.07462539]])
        ],
        None,
        None,
        [[np.inf, np.inf, np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf]],
        [3.7, 7.4, 7.4, 12.33333333, 2.64285714]
    ),
    Case(
        '#9',
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.6, 0.0, 0.0, 0.4, 0.0, 0.0], [0.0, 0.0, 0.6, 0.0, 0.4, 0.0], [0.0, 0.0, 0.0, 0.6, 0.0, 0.4], [0.0, 0.4, 0.0, 0.0, 0.6, 0.0]],
        [
            SubcaseCommittor(0, 1, None, None),
            SubcaseCommittor(3, [2, 4], None, None)
        ],
        [
            SubcaseFirstPassage(2, 2, 0, 0, [4.0, 0.6, 0.8, 3.8, 2.2, 10.1], [0.6, 0.0], 2.064),
            SubcaseFirstPassage(6, 3, [2, 5], [2, 5], [4.0, 0.6, 0.8, 3.8, 2.2, 10.1], [0.6, 0.16, 0.144, 0.0768, 0.06912, 0.036864], 2.3252992)
        ],
        [
            SubcaseHitting(1, [0.0, 1.0, 0.07582938, 0.18957346, 0.36018957, 0.61611374], [np.inf, 0.0, np.inf, np.inf, np.inf, np.inf]),
            SubcaseHitting(3, [0.0, 0.0, 0.4, 1.0, 0.78947368, 0.47368421], [np.inf, np.inf, np.inf, 0.0, np.inf, np.inf])
        ],
        [
            SubcaseMfptb(0, 1, None),
            SubcaseMfptb(3, [2, 5], None)
        ],
        [
            SubcaseMfptt(None, None),
            SubcaseMfptt(1, None)
        ],
        [
            SubcaseMixing(None, 3, 'natural', None),
            SubcaseMixing([0.05, 0.05, 0.1, 0.2, 0.2, 0.4], 1, 'natural', None)
        ],
        [
            SubcaseSensitivity(4, None),
            SubcaseSensitivity(5, None)
        ],
        [[0.92417062, 0.81042654, 0.63981043, 0.38388626], [0.07582938, 0.18957346, 0.36018957, 0.61611374]],
        [3.1042654, 5.26066351, 5.99526066, 4.5971564],
        [[np.inf, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, np.inf, 0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, 0.5402844, 0.9004739, 0.4739336, 0.1895735], [np.inf, np.inf, 1.3507109, 1.25118483, 1.18483412, 0.47393365], [np.inf, np.inf, 1.06635071, 1.77725118, 1.25118483, 0.90047393], [np.inf, np.inf, 0.63981043, 1.06635071, 1.3507109, 0.54028436]],
        None
    )
]


#########
# TESTS #
#########

@mark.parametrize(
    argnames=('p', 'absorption_probabilities'),
    argvalues=[(case.p, case.absorption_probabilities) for case in cases],
    ids=['test_absorption_probabilities ' + case.id for case in cases]
)
def test_absorption_probabilities(p, absorption_probabilities):

    mc = MarkovChain(p)

    actual = mc.absorption_probabilities()
    expected = absorption_probabilities

    if actual is not None and expected is not None:
        expected = np.asarray(expected)
        assert np.allclose(actual, expected)
    else:
        assert actual == expected


@mark.parametrize(
    argnames=('p', 'states1', 'states2', 'value_backward', 'value_forward'),
    argvalues=[(case.p, subcase.states1, subcase.states2, subcase.value_backward, subcase.value_forward) for case in cases for subcase in case.committor_data],
    ids=['test_committor_probabilities ' + case.id + '-' + str(index + 1) for case in cases for (index, subcase) in enumerate(case.committor_data)]
)
def test_committor_probabilities(p, states1, states2, value_backward, value_forward):

    mc = MarkovChain(p)

    actual = mc.committor_probabilities('backward', states1, states2)
    expected = value_backward

    if actual is not None and expected is not None:
        expected = np.asarray(expected)
        assert np.allclose(actual, expected)
    else:
        assert actual == expected

    actual = mc.committor_probabilities('forward', states1, states2)
    expected = value_forward

    if actual is not None and expected is not None:
        expected = np.asarray(expected)
        assert np.allclose(actual, expected)
    else:
        assert actual == expected


@mark.parametrize(
    argnames=('p', 'steps', 'initial_state', 'first_passage_states', 'value'),
    argvalues=[(case.p, subcase.steps, subcase.initial_state, subcase.first_passage_states_probabilities, subcase.value_probabilities) for case in cases for subcase in case.first_passage_data],
    ids=['test_first_passage_probabilities ' + case.id + '-' + str(index + 1) for case in cases for (index, subcase) in enumerate(case.first_passage_data)]
)
def test_first_passage_probabilities(p, steps, initial_state, first_passage_states, value):

    mc = MarkovChain(p)

    actual = mc.first_passage_probabilities(steps, initial_state, first_passage_states)
    expected = np.asarray(value)

    if first_passage_states is not None:
        assert actual.size == steps

    assert np.allclose(actual, expected)


@mark.parametrize(
    argnames=('p', 'steps', 'initial_state', 'first_passage_states', 'rewards', 'value'),
    argvalues=[(case.p, subcase.steps, subcase.initial_state, subcase.first_passage_states_reward, subcase.rewards, subcase.value_rewards) for case in cases for subcase in case.first_passage_data],
    ids=['test_first_passage_reward ' + case.id + '-' + str(index + 1) for case in cases for (index, subcase) in enumerate(case.first_passage_data)]
)
def test_first_passage_reward(p, steps, initial_state, first_passage_states, rewards, value):

    mc = MarkovChain(p)

    if mc.size <= 2:
        skip('Markov chain size is less than or equal to 2.')
    else:

        actual = mc.first_passage_reward(steps, initial_state, first_passage_states, rewards)
        expected = value

        assert np.isclose(actual, expected)


@mark.parametrize(
    argnames=('p', 'targets', 'value'),
    argvalues=[(case.p, subcase.targets, subcase.value_probabilities) for case in cases for subcase in case.hitting_data],
    ids=['test_hitting_probabilities ' + case.id + '-' + str(index + 1) for case in cases for (index, subcase) in enumerate(case.hitting_data)]
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
    argvalues=[(case.p, subcase.targets, subcase.value_times) for case in cases for subcase in case.hitting_data],
    ids=['test_hitting_times ' + case.id + '-' + str(index + 1) for case in cases for (index, subcase) in enumerate(case.hitting_data)]
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
    expected = value

    if actual is not None and expected is not None:
        assert np.isclose(actual, expected)
    else:
        assert actual == expected


@mark.parametrize(
    argnames=('p', 'targets', 'value'),
    argvalues=[(case.p, subcase.targets, subcase.value) for case in cases for subcase in case.mfptt_data],
    ids=['test_mean_first_passage_times_to ' + case.id + '-' + str(index + 1) for case in cases for (index, subcase) in enumerate(case.mfptt_data)]
)
def test_mean_first_passage_times_to(p, targets, value):

    mc = MarkovChain(p)

    actual = mc.mean_first_passage_times_to(targets)
    expected = value

    if actual is not None and expected is not None:
        expected = np.asarray(expected)
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

    actual = mc.mean_absorption_times()
    expected = mean_absorption_times

    if actual is not None and expected is not None:
        expected = np.asarray(expected)
        assert np.allclose(actual, expected)
    else:
        assert actual == expected

    if mc.is_absorbing and len(mc.transient_states) > 0:

        actual = actual.size
        expected = mc.size - len(mc.absorbing_states)

        assert actual == expected


@mark.parametrize(
    argnames=('p', 'mean_number_visits'),
    argvalues=[(case.p, case.mean_number_visits) for case in cases],
    ids=['test_mean_number_visits ' + case.id for case in cases]
)
def test_mean_number_visits(p, mean_number_visits):

    mc = MarkovChain(p)

    actual = mc.mean_number_visits()
    expected = np.asarray(mean_number_visits)

    assert np.allclose(actual, expected)


@mark.parametrize(
    argnames=('p', 'mean_recurrence_times'),
    argvalues=[(case.p, case.mean_recurrence_times) for case in cases],
    ids=['test_mean_recurrence_times ' + case.id for case in cases]
)
def test_mean_recurrence_times(p, mean_recurrence_times):

    mc = MarkovChain(p)

    actual = mc.mean_recurrence_times()
    expected = mean_recurrence_times

    if actual is not None and expected is not None:
        expected = np.asarray(expected)
        assert np.allclose(actual, expected)
    else:
        assert actual == expected

    if mc.is_ergodic:

        actual = np.nan_to_num(actual**-1.0)
        expected = np.dot(actual, mc.p)

        assert np.allclose(actual, expected)


@mark.parametrize(
    argnames=('p', 'initial_distribution', 'jump', 'cutoff_type', 'value'),
    argvalues=[(case.p, subcase.initial_distribution, subcase.jump, subcase.cutoff_type, subcase.value) for case in cases for subcase in case.mixing_data],
    ids=['test_mixing_time ' + case.id + '-' + str(index + 1) for case in cases for (index, subcase) in enumerate(case.mixing_data)]
)
def test_mixing_time(p, initial_distribution, jump, cutoff_type, value):

    mc = MarkovChain(p)

    actual = mc.mixing_time(initial_distribution, jump, cutoff_type)
    expected = value

    assert actual == expected


@mark.parametrize(
    argnames=('p', 'state', 'value'),
    argvalues=[(case.p, subcase.state, subcase.value) for case in cases for subcase in case.sensitivity_data],
    ids=['test_sensitivity ' + case.id + '-' + str(index + 1) for case in cases for (index, subcase) in enumerate(case.sensitivity_data)]
)
def test_sensitivity(p, state, value):

    mc = MarkovChain(p)

    actual = mc.sensitivity(state)
    expected = value

    if actual is not None and expected is not None:
        expected = np.asarray(expected)
        assert np.allclose(actual, expected)
    else:
        assert actual == expected


@mark.parametrize(
    argnames=('p', 'mfptt_targets'),
    argvalues=[(case.p, subcase.targets) for case in cases for subcase in case.mfptt_data if subcase.targets is None],
    ids=['test_mpft_and_recurrence_relation ' + case.id for case in cases]
)
def test_mpft_and_recurrence_relation(p, mfptt_targets):

    mc = MarkovChain(p)

    if not mc.is_ergodic:
        skip('Markov chain is not ergodic.')
    else:

        mfpt = mc.mean_first_passage_times_to(mfptt_targets)
        mrt = mc.mean_recurrence_times()

        actual = mfpt
        expected = np.dot(mc.p, mfpt) + np.ones((mc.size, mc.size), dtype=float) - np.diag(mrt)

        assert np.allclose(actual, expected)
