# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Standard

from math import (
    isnan as _math_isnan
)

# Internal

from pydtmc.assessments import (
    assess_first_order as _assess_first_order,
    assess_homogeneity as _assess_homogeneity,
    assess_markov_property as _assess_markov_property,
    assess_stationarity as _assess_stationarity,
    assess_theoretical_compatibility as _assess_theoretical_compatibility
)

from pydtmc.markov_chain import (
    MarkovChain as _MarkovChain
)


#########
# TESTS #
#########

def _extract_data_chi2(result, value):

    actual_rejection, actual_p_value, stats = result
    actual_p_value = 'NaN' if _math_isnan(actual_p_value) else round(actual_p_value, 8)
    actual_chi2 = 'NaN' if _math_isnan(stats['chi2']) else round(stats['chi2'], 8)
    actual_dof = 'NaN' if _math_isnan(stats['dof']) else stats['dof']

    expected_rejection, expected_p_value, expected_chi2, expected_dof = tuple(value)
    expected_p_value = 'NaN' if _math_isnan(expected_p_value) else expected_p_value
    expected_chi2 = 'NaN' if _math_isnan(expected_chi2) else expected_chi2
    expected_dof = 'NaN' if _math_isnan(expected_dof) else expected_dof

    actual = (actual_rejection, actual_p_value, actual_chi2, actual_dof)
    expected = (expected_rejection, expected_p_value, expected_chi2, expected_dof)

    return actual, expected


def test_assess_first_order(walk, possible_states, significance, value):

    result = _assess_first_order(walk, possible_states, significance)
    actual, expected = _extract_data_chi2(result, value)

    assert actual == expected


def test_assess_homogeneity(walks, possible_states, significance, value):

    result = _assess_homogeneity(walks, possible_states, significance)
    actual, expected = _extract_data_chi2(result, value)

    assert actual == expected


def test_assess_markov_property(walk, possible_states, significance, value):

    result = _assess_markov_property(walk, possible_states, significance)
    actual, expected = _extract_data_chi2(result, value)

    assert actual == expected


def test_assess_stationarity(walk, possible_states, blocks, significance, value):

    result = _assess_stationarity(walk, possible_states, blocks, significance)
    actual, expected = _extract_data_chi2(result, value)

    assert actual == expected


def test_assess_theoretical_compatibility(p, walk, possible_states, significance, value):

    mc = _MarkovChain(p, possible_states)

    result = _assess_theoretical_compatibility(mc, walk, possible_states, significance)
    actual, expected = _extract_data_chi2(result, value)

    assert actual == expected
