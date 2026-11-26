# -*- coding: utf-8 -*-

__all__ = [
    'plot_comparison',
    'plot_convergence',
    'plot_decoding',
    'plot_flow',
    'plot_graph',
    'plot_matrix',
    'plot_posterior',
    'plot_redistributions',
    'plot_residuals',
    'plot_sequence',
    'plot_spectrum',
    'plot_trellis'
]


###########
# IMPORTS #
###########

# Standard

import inspect as _ins

# Internal

from . import _custom_types as _ct
from . import _decorators as _dc
from . import _rendering as _r
from . import _utilities as _u
from . import _validation as _v

from .exceptions import (
    ValidationError as _ValidationError
)

from .hidden_markov_model import (
    HiddenMarkovModel as _HiddenMarkovModel
)

from .markov_chain import (
    MarkovChain as _MarkovChain
)


#############
# FUNCTIONS #
#############

@_dc.interactive_plot
def plot_comparison(models: _ct.tlist_mdl, underlying_matrices: str = 'transition', names: _ct.olist_str = None, dpi: int = 100) -> _ct.tplot:

    """
    | The function plots a heatmap comparison of the transition or emission matrices of the given models.

    | **Notes:**

    - If `Matplotlib <https://matplotlib.org/>`_ is in `interactive mode <https://matplotlib.org/stable/users/interactive.html>`_, the plot is immediately displayed.

    :param models: a list containing at least two models to compare.
    :param underlying_matrices:
     - **emission** for comparing the emission matrices;
     - **transition** for comparing the transition matrices.
    :param names: the subplot titles (*if omitted, standard names are generated*).
    :param dpi: the resolution of the plot expressed in dots per inch.
    :raises ValidationError: if any input argument is not compliant or if emission matrices are requested for models that are not all hidden Markov models.
    """

    try:

        models = _v.validate_models(models, 2)
        underlying_matrices = _v.validate_enumerator(underlying_matrices, ['emission', 'transition'])
        names = _u.create_models_names(models) if names is None else _v.validate_strings(names, size=len(models))
        dpi = _v.validate_dpi(dpi)

    except Exception as ex:  # pragma: no cover
        raise _u.create_validation_error(ex, _ins.trace()) from None

    if (underlying_matrices == 'emission') and (not all(isinstance(model, _HiddenMarkovModel) for model in models)):  # pragma: no cover
        raise _ValidationError('In order to compare emission matrices, the list must contain only hidden Markov models.')

    return _r.plot_comparison(models, underlying_matrices, names, dpi)


@_dc.interactive_plot
def plot_convergence(model: _ct.tmdl, steps: int, initial_status: _ct.ostatus = None, dpi: int = 100, full_scale: bool = False) -> _ct.tplot:

    """
    | The function plots the total variation distance between the evolving state distribution and the stationary distribution of the given model.

    | **Notes:**

    - The underlying Markov chain must be **ergodic**.
    - For hidden Markov models, convergence refers to the underlying state-transition chain.
    - The total variation distance at each step is half the sum of the absolute differences between the current and stationary state probabilities.
    - If `Matplotlib <https://matplotlib.org/>`_ is in `interactive mode <https://matplotlib.org/stable/users/interactive.html>`_, the plot is immediately displayed.

    :param model: the model.
    :param steps: the number of steps to evaluate (*the initial distribution at step 0 is also included*).
    :param initial_status: the initial state or initial state distribution (*if omitted, the states are assumed to be uniformly distributed*).
    :param dpi: the resolution of the plot expressed in dots per inch.
    :param full_scale: whether to display the full total variation distance range from 0 to 1.
    :raises ValidationError: if any input argument is not compliant.
    """

    try:

        model = _v.validate_model(model)
        steps = _v.validate_integer(steps, lower_limit=(1, False))
        initial_status = None if initial_status is None else _v.validate_status(initial_status, model.states, False)
        dpi = _v.validate_dpi(dpi)
        full_scale = _v.validate_boolean(full_scale)

    except Exception as ex:  # pragma: no cover
        raise _u.create_validation_error(ex, _ins.trace()) from None

    mc = model if model.__class__.__name__ == 'MarkovChain' else _MarkovChain(model.p, model.states)

    if not mc.is_ergodic:  # pragma: no cover
        raise _ValidationError('The underlying Markov chain must be ergodic in order to plot convergence.')

    return _r.plot_convergence(mc, steps, initial_status, dpi, full_scale)


@_dc.interactive_plot
def plot_decoding(hmm: _ct.thmm, symbols: _ct.tseq, initial_status: _ct.ostatus = None, dpi: int = 100) -> _ct.tplot:

    """
    | The function plots an observed sequence of emitted symbols alongside the corresponding decoded sequences of hidden states.

    | **Notes:**

    - Observations for which the two decoding methods produce different hidden states are highlighted.
    - If `Matplotlib <https://matplotlib.org/>`_ is in `interactive mode <https://matplotlib.org/stable/users/interactive.html>`_, the plot is immediately displayed.

    :param hmm: the hidden Markov model.
    :param symbols: the observed sequence of emitted symbols.
    :param initial_status: the initial hidden state or initial hidden-state distribution (*if omitted, the hidden states are assumed to be uniformly distributed*).
    :param dpi: the resolution of the plot expressed in dots per inch.
    :raises ValidationError: if any input argument is not compliant.
    :raises ValueError: if the computation of the decoded sequences fails.
    """

    try:

        hmm = _v.validate_hidden_markov_model(hmm)
        symbols = _v.validate_sequence(symbols, hmm.symbols)
        initial_status = _v.validate_status(initial_status, hmm.states, True)
        dpi = _v.validate_dpi(dpi)

    except Exception as ex:  # pragma: no cover
        raise _u.create_validation_error(ex, _ins.trace()) from None

    return _r.plot_decoding(hmm, symbols, initial_status, dpi)


@_dc.interactive_plot
def plot_flow(model: _ct.tmdl, steps: int, interval: int, initial_status: _ct.ostatus = None, dpi: int = 100) -> _ct.tplot:

    """
    | The function plots an alluvial diagram representing the evolution of state probabilities and their redistribution between displayed time points.

    | **Notes:**

    - The diagram includes the initial distribution at step 0.
    - If `Matplotlib <https://matplotlib.org/>`_ is in `interactive mode <https://matplotlib.org/stable/users/interactive.html>`_, the plot is immediately displayed.

    :param model: the model.
    :param steps: the number of intervals between displayed time points.
    :param interval: the number of Markov-chain transitions between consecutive displayed time points.
    :param initial_status: the initial state or initial state distribution (*if omitted, the states are assumed to be uniformly distributed*).
    :param dpi: the resolution of the plot expressed in dots per inch.
    :raises ValidationError: if any input argument is not compliant.
    """

    try:

        model = _v.validate_model(model)
        steps = _v.validate_integer(steps, lower_limit=(1, False))
        interval = _v.validate_integer(interval, lower_limit=(1, False))
        initial_status = None if initial_status is None else _v.validate_status(initial_status, model.states, False)
        dpi = _v.validate_dpi(dpi)

    except Exception as ex:  # pragma: no cover
        raise _u.create_validation_error(ex, _ins.trace()) from None

    mc = model if model.__class__.__name__ == 'MarkovChain' else _MarkovChain(model.p, model.states)

    return _r.plot_flow(mc, steps, interval, initial_status, dpi)


@_dc.interactive_plot
def plot_graph(model: _ct.tmdl, nodes_color: bool = True, nodes_shape: bool = True, edges_label: bool = True, force_standard: bool = False, dpi: int = 100) -> _ct.tplot:

    """
    | The function plots the directed graph of the given model, representing state transitions and, for hidden Markov models, symbol emissions.

    | **Notes:**

    - For Markov chains, node colors identify communicating classes and node shapes distinguish recurrent from transient states.
    - For hidden Markov models, state nodes have distinct colors and elliptical shapes, while symbol nodes are gray and hexagonal.
    - `Graphviz <https://graphviz.org/>`_ and `pydot <https://pypi.org/project/pydot/>`_ are optional dependencies that enable an extended rendering mode.
    - Rendering quality may deteriorate for large graphs, particularly in standard mode.
    - If `Matplotlib <https://matplotlib.org/>`_ is in `interactive mode <https://matplotlib.org/stable/users/interactive.html>`_, the plot is immediately displayed.

    :param model: the model.
    :param nodes_color: whether to distinguish node types using colors.
    :param nodes_shape: whether to distinguish node types using shapes.
    :param edges_label: whether to display transition and emission probabilities on edges.
    :param force_standard: whether to force standard rendering even when extended rendering is available.
    :param dpi: the resolution of the plot expressed in dots per inch.
    :raises ValidationError: if any input argument is not compliant.
    """

    try:

        model = _v.validate_model(model)
        nodes_color = _v.validate_boolean(nodes_color)
        nodes_shape = _v.validate_boolean(nodes_shape)
        edges_label = _v.validate_boolean(edges_label)
        force_standard = _v.validate_boolean(force_standard)
        dpi = _v.validate_dpi(dpi)

    except Exception as ex:  # pragma: no cover
        raise _u.create_validation_error(ex, _ins.trace()) from None

    return _r.plot_graph(model, nodes_color, nodes_shape, edges_label, force_standard, dpi)


@_dc.interactive_plot
def plot_matrix(model: _ct.tmdl, underlying_matrix: str = 'transition', dpi: int = 100) -> _ct.tplot:

    """
    | The function plots a heatmap of the transition or emission matrix of the given model.

    | **Notes:**

    - Matrix rows represent origin states, while columns represent destination states or emitted symbols.
    - If `Matplotlib <https://matplotlib.org/>`_ is in `interactive mode <https://matplotlib.org/stable/users/interactive.html>`_, the plot is immediately displayed.

    :param model: the model.
    :param underlying_matrix:
     - **emission** for displaying the emission matrix;
     - **transition** for displaying the transition matrix.
    :param dpi: the resolution of the plot expressed in dots per inch.
    :raises ValidationError: if any input argument is not compliant or if an emission matrix is requested for a Markov chain.
    """

    try:

        model = _v.validate_model(model)
        underlying_matrix = _v.validate_enumerator(underlying_matrix, ['emission', 'transition'])
        dpi = _v.validate_dpi(dpi)

    except Exception as ex:  # pragma: no cover
        raise _u.create_validation_error(ex, _ins.trace()) from None

    if (underlying_matrix == 'emission') and (not isinstance(model, _HiddenMarkovModel)):  # pragma: no cover
        raise _ValidationError('Emission matrices are only available for hidden Markov models.')

    return _r.plot_matrix(model, underlying_matrix, dpi)


@_dc.interactive_plot
def plot_posterior(hmm: _ct.thmm, symbols: _ct.tseq, initial_status: _ct.ostatus = None, plot_type: str = 'projection', dpi: int = 100) -> _ct.tplot:

    """
    | The function plots the posterior probabilities of the hidden states given an observed sequence of emitted symbols.

    | **Notes:**

    - Filtering considers only the observations available up to the current step, whereas smoothing considers the entire observed sequence.
    - Both probability distributions are computed using the forward-backward algorithm.
    - If `Matplotlib <https://matplotlib.org/>`_ is in `interactive mode <https://matplotlib.org/stable/users/interactive.html>`_, the plot is immediately displayed.

    :param hmm: the hidden Markov model.
    :param symbols: the observed sequence of emitted symbols.
    :param initial_status: the initial hidden state or initial hidden-state distribution (*if omitted, the hidden states are assumed to be uniformly distributed*).
    :param plot_type:
     - **heatmap** for displaying the filtered and smoothed posterior probabilities across successive steps;
     - **projection** for displaying the evolution of the smoothed posterior probabilities of each hidden state.
    :param dpi: the resolution of the plot expressed in dots per inch.
    :raises ValidationError: if any input argument is not compliant.
    :raises ValueError: if the computation of the posterior probabilities fails.
    """

    try:

        hmm = _v.validate_hidden_markov_model(hmm)
        symbols = _v.validate_sequence(symbols, hmm.symbols)
        initial_status = _v.validate_status(initial_status, hmm.states, True)
        plot_type = _v.validate_enumerator(plot_type, ['heatmap', 'projection'])
        dpi = _v.validate_dpi(dpi)

    except Exception as ex:  # pragma: no cover
        raise _u.create_validation_error(ex, _ins.trace()) from None

    return _r.plot_posterior(plot_type, hmm, symbols, initial_status, dpi)


@_dc.interactive_plot
def plot_redistributions(model: _ct.tmdl, redistributions: int, initial_status: _ct.ostatus = None, plot_type: str = 'projection', dpi: int = 100) -> _ct.tplot:

    """
    | The function plots the evolution of the state probability distribution over successive steps of the given model.

    | **Notes:**

    - The plot includes the initial distribution at step 0.
    - If `Matplotlib <https://matplotlib.org/>`_ is in `interactive mode <https://matplotlib.org/stable/users/interactive.html>`_, the plot is immediately displayed.

    :param model: the model.
    :param redistributions: the number of redistributions to perform.
    :param initial_status: the initial state or initial state distribution (*if omitted, the states are assumed to be uniformly distributed*).
    :param plot_type:
     - **heatmap** for displaying state probabilities across successive steps;
     - **projection** for displaying the evolution of each state's probability as a line plot.
    :param dpi: the resolution of the plot expressed in dots per inch.
    :raises ValidationError: if any input argument is not compliant.
    """

    try:

        model = _v.validate_model(model)
        redistributions = _v.validate_integer(redistributions, lower_limit=(1, False))
        initial_status = None if initial_status is None else _v.validate_status(initial_status, model.states, False)
        plot_type = _v.validate_enumerator(plot_type, ['heatmap', 'projection'])
        dpi = _v.validate_dpi(dpi)

    except Exception as ex:  # pragma: no cover
        raise _u.create_validation_error(ex, _ins.trace()) from None

    mc = model if model.__class__.__name__ == 'MarkovChain' else _MarkovChain(model.p, model.states)

    return _r.plot_redistributions(plot_type, mc, redistributions, initial_status, dpi)


@_dc.interactive_plot
def plot_residuals(model: _ct.tmdl, sequence: _ct.tseq, seed: _ct.oint = None, dpi: int = 100) -> _ct.tplot:

    """
    | The function plots diagnostic diagrams of the randomized quantile residuals of an observed sequence.

    | **Notes:**

    - The plot includes the residual sequence, the normal quantile-quantile diagram, the empirical residual distribution and the residual autocorrelation function.
    - The observed sequence consists of states or emitted symbols for Markov chains and hidden Markov models, respectively.
    - For hidden Markov models, the initial hidden states are assumed to be uniformly distributed.
    - The autocorrelation, confidence bounds are approximate and pointwise.
    - If `Matplotlib <https://matplotlib.org/>`_ is in `interactive mode <https://matplotlib.org/stable/users/interactive.html>`_, the plot is immediately displayed.

    :param model: the model.
    :param sequence: the observed sequence.
    :param seed: a seed to be used as RNG initializer for reproducibility purposes.
    :param dpi: the resolution of the plot expressed in dots per inch.
    :raises ValidationError: if any input argument is not compliant.
    :raises ValueError: if the observed sequence has zero probability under the model.
    """

    try:

        model = _v.validate_model(model)
        sequence = _v.validate_sequence(sequence, model.states if model.__class__.__name__ == 'MarkovChain' else model.symbols)
        seed = None if seed is None else _v.validate_integer(seed, lower_limit=(0, False), upper_limit=(2**32 - 1, False))
        dpi = _v.validate_dpi(dpi)

    except Exception as ex:  # pragma: no cover
        raise _u.create_validation_error(ex, _ins.trace()) from None

    return _r.plot_residuals(model, sequence, seed, dpi)


@_dc.interactive_plot
def plot_sequence(model: _ct.tmdl, steps: int, initial_state: _ct.ostate = None, plot_type: str = 'histogram', seed: _ct.oint = None, dpi: int = 100) -> _ct.tplot:

    """
    | The function plots a random sequence of states for Markov chains, and of hidden states and emitted symbols for hidden Markov models.

    | **Notes:**

    - If `Matplotlib <https://matplotlib.org/>`_ is in `interactive mode <https://matplotlib.org/stable/users/interactive.html>`_, the plot is immediately displayed.

    :param model: the model.
    :param steps: the number of steps to simulate.
    :param initial_state: the initial state (*if omitted, it is chosen uniformly at random*).
    :param plot_type:
     - **heatmap** for displaying the relative frequencies of observed transitions between consecutive states or symbols;
     - **histogram** for displaying the empirical frequencies of states or symbols;
     - **matrix** for displaying the simulated states or symbols across successive steps.
    :param seed: a seed to be used as RNG initializer for reproducibility purposes.
    :param dpi: the resolution of the plot expressed in dots per inch.
    :raises ValidationError: if any input argument is not compliant.
    """

    try:

        model = _v.validate_model(model)
        steps = _v.validate_integer(steps, lower_limit=(2, False))
        initial_state = None if initial_state is None else _v.validate_label(initial_state, model.states)
        plot_type = _v.validate_enumerator(plot_type, ['heatmap', 'histogram', 'matrix'])
        seed = None if seed is None else _v.validate_integer(seed, lower_limit=(0, False), upper_limit=(2**32 - 1, False))
        dpi = _v.validate_dpi(dpi)

    except Exception as ex:  # pragma: no cover
        raise _u.create_validation_error(ex, _ins.trace()) from None

    return _r.plot_sequence(plot_type, model, steps, initial_state, seed, dpi)


@_dc.interactive_plot
def plot_spectrum(model: _ct.tmdl, dpi: int = 100) -> _ct.tplot:

    """
    | The function plots the eigenvalues and eigenvector components of the transition matrix of the given model.

    | **Notes:**

    - The eigenvalues are displayed on the complex plane.
    - The eigenvalues plot, for **irreducible** models, also illustrates the second-largest eigenvalue modulus and the corresponding spectral gap.
    - The eigenvector components heatmap displays the real components of the normalized right eigenvectors, ordered by decreasing eigenvalue modulus, with the eigenvalue 1 placed first.
    - If `Matplotlib <https://matplotlib.org/>`_ is in `interactive mode <https://matplotlib.org/stable/users/interactive.html>`_, the plot is immediately displayed.

    :param model: the model.
    :param dpi: the resolution of the plot expressed in dots per inch.
    :raises ValidationError: if any input argument is not compliant.
    """

    try:

        model = _v.validate_model(model)
        dpi = _v.validate_dpi(dpi)

    except Exception as ex:  # pragma: no cover
        raise _u.create_validation_error(ex, _ins.trace()) from None

    mc = model if model.__class__.__name__ == 'MarkovChain' else _MarkovChain(model.p, model.states)

    return _r.plot_spectrum(mc, dpi)


@_dc.interactive_plot
def plot_trellis(hmm: _ct.thmm, steps: int, initial_state: _ct.ostate = None, seed: _ct.oint = None, force_standard: bool = False, dpi: int = 100) -> _ct.tplot:

    """
    | The function plots the forward and backward trellis diagrams of a simulated observation sequence on the given hidden Markov model.

    | **Notes:**

    - Highlighted nodes and edges identify the most probable hidden-state sequence calculated using the Viterbi algorithm.
    - If `Matplotlib <https://matplotlib.org/>`_ is in `interactive mode <https://matplotlib.org/stable/users/interactive.html>`_, the plot is immediately displayed.
    - `Graphviz <https://graphviz.org/>`_ and `pydot <https://pypi.org/project/pydot/>`_ are optional dependencies that enable an extended rendering mode.
    - Rendering quality may deteriorate for large simulations.

    :param hmm: the hidden Markov model.
    :param steps: the number of steps to simulate.
    :param initial_state: the initial state of the simulation (*if omitted, it is chosen uniformly at random*).
    :param seed: a seed to be used as RNG initializer for reproducibility purposes.
    :param force_standard: whether to force standard rendering even when extended rendering is available.
    :param dpi: the resolution of the plot expressed in dots per inch.
    :raises ValidationError: if any input argument is not compliant.
    :raises ValueError: if the computation of the forward probabilities, the backward probabilities or the most probable hidden-state sequence fails.
    """

    try:

        hmm = _v.validate_hidden_markov_model(hmm)
        steps = _v.validate_integer(steps, lower_limit=(2, False))
        initial_state = None if initial_state is None else _v.validate_label(initial_state, hmm.states)
        seed = None if seed is None else _v.validate_integer(seed, lower_limit=(0, False), upper_limit=(2**32 - 1, False))
        force_standard = _v.validate_boolean(force_standard)
        dpi = _v.validate_dpi(dpi)

    except Exception as ex:  # pragma: no cover
        raise _u.create_validation_error(ex, _ins.trace()) from None

    return _r.plot_trellis(hmm, steps, initial_state, seed, force_standard, dpi)
