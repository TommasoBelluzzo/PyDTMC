# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Standard

import math as _mt

# Internal

from pydtmc.assessments import (
    assess_homogeneity as _assess_homogeneity,
    assess_markov_property as _assess_markov_property,
    assess_theoretical_compatibility as _assess_theoretical_compatibility,
    assess_time_homogeneity as _assess_time_homogeneity
)

from pydtmc.markov_chain import (
    MarkovChain as _MarkovChain
)


#########
# TESTS #
#########

def _extract_data_chi2(result, value):

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


def test_assess_homogeneity(sequences, possible_states, significance, value):

    result = _assess_homogeneity(possible_states, sequences, significance)
    actual, expected = _extract_data_chi2(result, value)

    assert actual == expected


def test_assess_markov_property(sequence, possible_states, method, significance, value):

    if method == 'simulation':
        result = _assess_markov_property(possible_states, sequence, method, significance, simulations=100, seed=7331)
    else:
        result = _assess_markov_property(possible_states, sequence, method, significance)

    actual, expected = _extract_data_chi2(result, value)

    assert actual == expected


def test_assess_theoretical_compatibility(p, states, sequence, significance, value):

    mc = _MarkovChain(p, states)

    result = _assess_theoretical_compatibility(mc, sequence, significance)
    actual, expected = _extract_data_chi2(result, value)

    assert actual == expected


def test_assess_time_homogeneity(sequence, possible_states, blocks, significance, value):

    result = _assess_time_homogeneity(possible_states, sequence, blocks, significance)
    actual, expected = _extract_data_chi2(result, value)

    assert actual == expected
