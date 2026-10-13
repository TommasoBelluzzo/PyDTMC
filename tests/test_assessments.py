# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Standard

import math as _mt

# Internal

from pydtmc.assessments import (
    assess_change_point as _assess_change_point,
    assess_homogeneity as _assess_homogeneity,
    assess_independence as _assess_independence,
    assess_markov_property as _assess_markov_property,
    assess_order as _assess_order,
    assess_stationarity as _assess_stationarity,
    assess_theoretical_compatibility as _assess_theoretical_compatibility,
    assess_time_homogeneity as _assess_time_homogeneity
)

from pydtmc.markov_chain import (
    MarkovChain as _MarkovChain
)


#########
# TESTS #
#########

def _extract_data_change_point(result, value):

    actual_change_point, actual_p_value, actual_statistic = result
    actual_p_value = 'NaN' if _mt.isnan(actual_p_value) else round(actual_p_value, 8)
    actual_statistic = 'NaN' if _mt.isnan(actual_statistic) else round(actual_statistic, 8)

    expected_first_element, expected_p_value, expected_statistic = tuple(value)
    expected_p_value = 'NaN' if _mt.isnan(expected_p_value) else round(expected_p_value, 8)
    expected_statistic = 'NaN' if _mt.isnan(expected_statistic) else round(expected_statistic, 8)

    actual = (actual_change_point, actual_p_value, actual_statistic)
    expected = (expected_first_element, expected_p_value, expected_statistic)

    return actual, expected


def _extract_data_order(result, value):

    actual_order, actual_scores = result
    actual_scores = {int(k): round(v, 8) for k, v in actual_scores.items()}

    expected_order, expected_scores = tuple(value)
    expected_scores = {int(k): v for k, v in expected_scores.items()}

    actual = (actual_order, actual_scores)
    expected = (expected_order, expected_scores)

    return actual, expected


def _extract_data_statistic(result, value):

    actual_rejection, actual_p_value, stats = result
    actual_p_value = 'NaN' if _mt.isnan(actual_p_value) else round(actual_p_value, 8)
    actual_statistic = 'NaN' if _mt.isnan(stats['statistic']) else round(stats['statistic'], 8)
    actual_dof = 'NaN' if _mt.isnan(stats['dof']) else stats['dof']

    expected_rejection, expected_p_value, expected_statistic, expected_dof = tuple(value)
    expected_p_value = 'NaN' if _mt.isnan(expected_p_value) else expected_p_value
    expected_statistic = 'NaN' if _mt.isnan(expected_statistic) else expected_statistic
    expected_dof = 'NaN' if _mt.isnan(expected_dof) else expected_dof

    actual = (actual_rejection, actual_p_value, actual_statistic, actual_dof)
    expected = (expected_rejection, expected_p_value, expected_statistic, expected_dof)

    return actual, expected


def test_assess_change_point(possible_states, sequence, significance, value):

    result = _assess_change_point(possible_states, sequence, significance, simulations=100, seed=7331)
    actual, expected = _extract_data_change_point(result, value)

    assert actual == expected


def test_assess_homogeneity(possible_states, sequences, significance, value):

    result = _assess_homogeneity(possible_states, sequences, significance)
    actual, expected = _extract_data_statistic(result, value)

    assert actual == expected


def test_assess_independence(possible_states, sequence, method, significance, value):

    if method == 'simulation':
        result = _assess_independence(possible_states, sequence, method, significance, simulations=100, seed=7331)
    else:
        result = _assess_independence(possible_states, sequence, method, significance)

    actual, expected = _extract_data_statistic(result, value)

    assert actual == expected


def test_assess_markov_property(possible_states, sequence, method, significance, value):

    if method == 'simulation':
        result = _assess_markov_property(possible_states, sequence, method, significance, simulations=100, seed=7331)
    else:
        result = _assess_markov_property(possible_states, sequence, method, significance)

    actual, expected = _extract_data_statistic(result, value)

    assert actual == expected


def test_assess_order(possible_states, sequence, maximum_order, criterion, value):

    result = _assess_order(possible_states, sequence, maximum_order, criterion)
    actual, expected = _extract_data_order(result, value)

    assert actual == expected


def test_assess_stationarity(p, states, sequences, significance, value):

    mc = _MarkovChain(p, states)

    result = _assess_stationarity(mc, sequences, significance)
    actual, expected = _extract_data_statistic(result, value)

    assert actual == expected


def test_assess_theoretical_compatibility(p, states, sequence, significance, value):

    mc = _MarkovChain(p, states)

    result = _assess_theoretical_compatibility(mc, sequence, significance)
    actual, expected = _extract_data_statistic(result, value)

    assert actual == expected


def test_assess_time_homogeneity(possible_states, sequence, blocks, significance, value):

    result = _assess_time_homogeneity(possible_states, sequence, blocks, significance)
    actual, expected = _extract_data_statistic(result, value)

    assert actual == expected
