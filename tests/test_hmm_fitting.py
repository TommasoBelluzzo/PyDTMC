# -*- coding: utf-8 -*-


###########
# IMPORTS #
###########

# Libraries

import numpy as _np
import numpy.testing as _npt

# Internal

from pydtmc import (
    HiddenMarkovModel as _HiddenMarkovModel
)

from tests.utilities import (
    assert_stochastic_matrix as _assert_stochastic_matrix
)


#########
# TESTS #
#########

# noinspection PyBroadException
def test_estimate(possible_states, possible_symbols, states, symbols, value):

    try:
        hmm = _HiddenMarkovModel.estimate(possible_states, possible_symbols, states, symbols)
    except Exception:
        hmm = None

    if value is None:
        assert hmm is None
    else:

        actual = hmm.p
        expected = _np.array(value[0])

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
        _assert_stochastic_matrix(actual)

        actual = hmm.e
        expected = _np.array(value[1])

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
        _assert_stochastic_matrix(actual)


# noinspection PyBroadException
def test_fit(fitting_type, possible_states, possible_symbols, p_guess, e_guess, symbols, initial_status, regularization, fitting_param, value):

    p_guess = _np.array(p_guess)
    e_guess = _np.array(e_guess)

    try:
        hmm_fit = _HiddenMarkovModel.fit(fitting_type, possible_states, possible_symbols, p_guess, e_guess, symbols, initial_status, regularization, fitting_param)
    except Exception:
        hmm_fit = None

    if hmm_fit is None:

        actual = hmm_fit
        expected = value

        assert actual == expected

    else:

        actual = hmm_fit.p
        expected = _np.array(value[0])

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
        _assert_stochastic_matrix(actual)

        actual = hmm_fit.e
        expected = _np.array(value[1])

        _npt.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
        _assert_stochastic_matrix(actual)


def test_fit_baum_welch_monotonicity():

    def _expectation(ex_initial_distribution, ex_p_current, ex_e_current):

        tc = _np.zeros_like(ex_p_current)
        ec = _np.zeros_like(ex_e_current)
        llt = 0.0

        for symbols in sequences:

            ll, forward, backward, scaling = _forward_backward(ex_initial_distribution, ex_p_current, ex_e_current, symbols)
            llt += ll

            for t, symbol in enumerate(symbols):

                posterior = forward[t] * backward[t]
                ec[:, symbol] += posterior

                if t > 0:
                    destination = ex_e_current[:, symbol] * backward[t]
                    tc += (forward[t - 1, :, _np.newaxis] * ex_p_current * destination[_np.newaxis, :]) / scaling[t]

        return llt, tc, ec

    def _forward_backward(fb_initial_distribution, fb_p, fb_e, fb_symbols):

        n = fb_p.shape[0]
        f = len(fb_symbols)

        forward = _np.zeros((f, n), dtype=float)
        backward = _np.ones((f, n), dtype=float)
        scaling = _np.ones(f, dtype=float)

        forward[0] = fb_initial_distribution * fb_e[:, fb_symbols[0]]
        scaling[0] = _np.sum(forward[0])

        assert scaling[0] > 0.0

        forward[0] /= scaling[0]

        for t, symbol in enumerate(fb_symbols[1:], start=1):

            forward[t] = _np.dot(forward[t - 1], fb_p) * fb_e[:, symbol]
            scaling[t] = _np.sum(forward[t])

            assert scaling[t] > 0.0

            forward[t] /= scaling[t]

        for t in reversed(range(f - 1)):
            backward[t] = _np.dot(fb_p, (fb_e[:, fb_symbols[t + 1]] * backward[t + 1]))
            backward[t] /= scaling[t + 1]

        log_likelihood = _np.sum(_np.log(scaling))

        return log_likelihood, forward, backward, scaling

    p = _np.array([
        [0.7, 0.3],
        [0.4, 0.6]
    ])

    e = _np.array([
        [0.6, 0.3, 0.1],
        [0.2, 0.3, 0.5]
    ])

    sequences = [
        [0, 1, 2, 1, 0, 2, 1],
        [2, 2, 1, 0, 1, 2, 0]
    ]

    initial_distribution = _np.array([0.5, 0.5])
    previous_likelihood = None

    for _ in range(20):

        likelihood, transition_counts, emission_counts = _expectation(initial_distribution, p, e)

        if previous_likelihood is not None:
            assert likelihood >= previous_likelihood - 1e-10

        previous_likelihood = likelihood

        transition_sums = _np.sum(transition_counts, axis=1, keepdims=True)
        emission_sums = _np.sum(emission_counts, axis=1, keepdims=True)

        assert _np.all(transition_sums > 0.0)
        assert _np.all(emission_sums > 0.0)

        p = transition_counts / transition_sums
        e = emission_counts / emission_sums

        _assert_stochastic_matrix(p)
        _assert_stochastic_matrix(e)
