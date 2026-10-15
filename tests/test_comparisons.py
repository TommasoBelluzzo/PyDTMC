# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Libraries

import numpy as _np

# Internal

from pydtmc import (
    MarkovChain as _MarkovChain
)

from pydtmc.comparisons import (
    calculate_stationary_distance as _calculate_stationary_distance,
    calculate_transition_distance as _calculate_transition_distance,
    calculate_transition_divergence as _calculate_transition_divergence
)


#########
# TESTS #
#########

def test_calculate_stationary_distance(p1, p2, value):

    model1 = _MarkovChain(p1)
    model2 = _MarkovChain(p2)

    actual = _calculate_stationary_distance(model1, model2)
    expected = value

    assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_calculate_transition_distance(p1, p2, value_f, value_h, value_m, value_s, value_tv):

    model1 = _MarkovChain(p1)
    model2 = _MarkovChain(p2)

    distances_dictionary = {
        'frobenius': value_f,
        'hellinger': value_h,
        'maximum': value_m,
        'spectral': value_s,
        'total_variation': value_tv
    }

    for distance_type, distance_value in distances_dictionary.items():

        actual = _calculate_transition_distance(distance_type, model1, model2)
        expected = distance_value

        assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)

        actual = _calculate_transition_distance(distance_type, model2, model1)
        expected = distance_value

        assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_calculate_transition_divergence(p1, p2, value_js, value_kl):

    model1 = _MarkovChain(p1)
    model2 = _MarkovChain(p2)

    actual = _calculate_transition_divergence('jensen_shannon', model1, model2)
    expected = value_js

    assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)

    actual = _calculate_transition_divergence('jensen_shannon', model2, model1)
    expected = value_js

    assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)

    actual = _calculate_transition_divergence('kullback_leibler', model1, model2)
    expected = value_kl

    assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)
