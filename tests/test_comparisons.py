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

from pydtmc.exceptions import (
    ValidationError as _ValidationError
)

from pydtmc.comparisons import (
    calculate_first_passage_distance as _calculate_first_passage_distance,
    calculate_stationary_distance as _calculate_stationary_distance,
    calculate_stationary_divergence as _calculate_stationary_divergence,
    calculate_transition_distance as _calculate_transition_distance,
    calculate_transition_divergence as _calculate_transition_divergence
)


#########
# TESTS #
#########

def test_calculate_first_passage_distance(p1, p2, value_f, value_m):

    model1 = _MarkovChain(p1)
    model2 = _MarkovChain(p2)

    distances_dictionary = {
        'f': value_f,
        'm': value_m
    }

    for distance_type, distance_value in distances_dictionary.items():

        actual = _calculate_first_passage_distance(distance_type, model1, model2)
        expected = distance_value

        assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)

        actual = _calculate_first_passage_distance(distance_type, model2, model1)
        expected = distance_value

        assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_calculate_stationary_distance(p1, p2, value_h, value_tv):

    model1 = _MarkovChain(p1)
    model2 = _MarkovChain(p2)

    distances_dictionary = {
        'h': value_h,
        'tv': value_tv
    }

    for distance_type, distance_value in distances_dictionary.items():

        actual = _calculate_stationary_distance(distance_type, model1, model2)
        expected = distance_value

        assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)

        actual = _calculate_stationary_distance(distance_type, model2, model1)
        expected = distance_value

        assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_calculate_stationary_divergence(p1, p2, value_js, value_kl):

    model1 = _MarkovChain(p1)
    model2 = _MarkovChain(p2)

    actual = _calculate_stationary_divergence('js', model1, model2)
    expected = value_js

    assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)

    actual = _calculate_stationary_divergence('js', model2, model1)
    expected = value_js

    assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)

    actual = _calculate_stationary_divergence('kl', model1, model2)
    expected = float(value_kl)

    assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_calculate_transition_distance(p1, p2, value_f, value_h, value_m, value_s, value_tv):

    model1 = _MarkovChain(p1)
    model2 = _MarkovChain(p2)

    distances_dictionary = {
        'f': value_f,
        'h': value_h,
        'm': value_m,
        's': value_s,
        'tv': value_tv
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

    actual = _calculate_transition_divergence('js', model1, model2)
    expected = value_js

    assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)

    actual = _calculate_transition_divergence('js', model2, model1)
    expected = value_js

    assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)

    actual = _calculate_transition_divergence('kl', model1, model2)
    expected = float(value_kl)

    assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_calculate_transition_distance_steps(p1, p2, steps, value_f, value_h, value_m, value_s, value_tv):

    model1 = _MarkovChain(p1)
    model2 = _MarkovChain(p2)

    distances_dictionary = {
        'f': value_f,
        'h': value_h,
        'm': value_m,
        's': value_s,
        'tv': value_tv
    }

    for distance_type, distance_value in distances_dictionary.items():

        actual = _calculate_transition_distance(distance_type, model1, model2, steps)
        expected = distance_value

        assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)

        actual = _calculate_transition_distance(distance_type, model2, model1, steps)
        expected = distance_value

        assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_calculate_transition_divergence_steps(p1, p2, steps, value_js, value_kl):

    model1 = _MarkovChain(p1)
    model2 = _MarkovChain(p2)

    actual = _calculate_transition_divergence('js', model1, model2, steps)
    expected = value_js

    assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)

    actual = _calculate_transition_divergence('js', model2, model1, steps)
    expected = value_js

    assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)

    actual = _calculate_transition_divergence('kl', model1, model2, steps)
    expected = float(value_kl)

    assert _np.isclose(actual, expected, rtol=1e-5, atol=1e-8)


def test_comparisons_invalid_models():

    model1 = _MarkovChain([[0.9, 0.1], [0.2, 0.8]])
    model2 = _MarkovChain([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5]])
    model3 = _MarkovChain([[1.0, 0.0], [0.0, 1.0]])

    comparisons = [
        (_calculate_first_passage_distance, 'f'),
        (_calculate_stationary_distance, 'h'),
        (_calculate_stationary_divergence, 'js'),
        (_calculate_transition_distance, 'f'),
        (_calculate_transition_divergence, 'js')
    ]

    for comparison, comparison_type in comparisons:

        with _np.testing.assert_raises(ValueError):
            comparison(comparison_type, model1, model2)

    with _np.testing.assert_raises(ValueError):
        _calculate_first_passage_distance('f', model1, model3)

    with _np.testing.assert_raises(ValueError):
        _calculate_stationary_distance('h', model1, model3)

    with _np.testing.assert_raises(ValueError):
        _calculate_stationary_divergence('js', model1, model3)


def test_comparisons_invalid_arguments():

    model1 = _MarkovChain([[0.9, 0.1], [0.2, 0.8]])
    model2 = _MarkovChain([[0.7, 0.3], [0.4, 0.6]])

    comparisons = [
        (_calculate_first_passage_distance, 'js'),
        (_calculate_stationary_distance, 'js'),
        (_calculate_stationary_divergence, 'rer'),
        (_calculate_transition_distance, 'js'),
        (_calculate_transition_divergence, 'rer')
    ]

    for comparison, comparison_type in comparisons:

        with _np.testing.assert_raises(_ValidationError):
            comparison(comparison_type, model1, model2)

    for steps in (0, -1, True):

        with _np.testing.assert_raises(_ValidationError):
            _calculate_transition_distance('f', model1, model2, steps)

        with _np.testing.assert_raises(_ValidationError):
            _calculate_transition_divergence('js', model1, model2, steps)
