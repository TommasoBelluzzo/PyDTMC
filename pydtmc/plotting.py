# -*- coding: utf-8 -*-

__all__ = [
    'plot_comparison',
    'plot_convergence',
    'plot_eigenvalues',
    'plot_flow',
    'plot_graph',
    'plot_matrix',
    'plot_redistributions',
    'plot_sequence',
    'plot_trellis'
]


###########
# IMPORTS #
###########

# Standard

import copy as _cp
import inspect as _ins
import io as _io
import math as _mt
import subprocess as _sub

# Libraries

import matplotlib.colorbar as _mplcb
import matplotlib.colors as _mplcr
import matplotlib.image as _mpli
import matplotlib.patches as _mplpc
import matplotlib.pyplot as _mplp
import matplotlib.ticker as _mplt
import networkx as _nx
import numpy as _np
import numpy.linalg as _npl

try:
    import pydot as _pyd
    _pydot_found = True
except ImportError:  # pragma: no cover
    _pyd = None
    _pydot_found = False

# Internal

from . import _plotting as _pl
from . import _custom_types as _ct
from . import _decorators as _dc
from . import _numerical as _n
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
# CONSTANTS #
#############

_color_black = '#000000'
_color_gray = '#E0E0E0'
_color_white = '#FFFFFF'
_colors = ('#80B1D3', '#FFED6F', '#B3DE69', '#BEBADA', '#FDB462', '#8DD3C7', '#FB8072', '#FCCDE5', '#E5C494')

_default_color_edge = _color_black
_default_color_node = _color_white
_default_color_path = _colors[0]
_default_color_symbol = _color_gray
_default_node_size = 600


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

        models = _v.validate_models(models)
        underlying_matrices = _v.validate_enumerator(underlying_matrices, ['emission', 'transition'])
        names = _u.create_models_names(models) if names is None else _v.validate_strings(names, size=len(models))
        dpi = _v.validate_dpi(dpi)

    except Exception as ex:  # pragma: no cover
        raise _u.create_validation_error(ex, _ins.trace()) from None

    if (underlying_matrices == 'emission') and (not all(isinstance(model, _HiddenMarkovModel) for model in models)):  # pragma: no cover
        raise _ValidationError('In order to compare emission matrices, the list must contain only hidden Markov models.')

    space = len(models)
    rows = int(_mt.sqrt(space))
    columns = int(_mt.ceil(space / float(rows)))

    figure, axes = _mplp.subplots(nrows=rows, ncols=columns, constrained_layout=True, dpi=dpi)
    axes = list(axes.flat)
    ax_is = None

    color_map = _mplcr.LinearSegmentedColormap.from_list('ColorMap', [_color_white, _colors[0]], 20)

    for ax, model, name in zip(axes, models, names):

        matrix = model.e if underlying_matrices == 'emission' else model.p

        ax_is = ax.imshow(matrix, aspect='auto', cmap=color_map, interpolation='none', vmin=0.0, vmax=1.0)
        ax.set_title(name, fontsize=9.0, fontweight='normal', pad=1)

        ax.set_xticks([])
        ax.set_xticks([], minor=True)
        ax.set_yticks([])
        ax.set_yticks([], minor=True)

    for ax in axes[len(models):]:
        ax.set_visible(False)

    color_map_ax, color_map_ax_kwargs = _mplcb.make_axes(axes[:len(models)], drawedges=True, orientation='horizontal', ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    figure.colorbar(ax_is, cax=color_map_ax, **color_map_ax_kwargs)
    color_map_ax.set_xticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])

    figure.suptitle('Comparison Plot', fontsize=15.0, fontweight='bold')

    return figure, axes


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

    if model.__class__.__name__ == 'MarkovChain':
        mc = model
    else:
        mc = _MarkovChain(model.p, model.states)

    if not mc.is_ergodic:  # pragma: no cover
        raise _ValidationError('The underlying Markov chain must be ergodic in order to plot convergence.')

    stationary_distribution = mc.pi[0]

    distributions = mc.propagate(steps, initial_status=initial_status, output_last=False)
    distributions = _np.asarray(distributions, dtype=float)

    distances = 0.5 * _np.sum(_np.abs(distributions - stationary_distribution), axis=1)

    figure, ax = _mplp.subplots(dpi=dpi)

    ax.plot(_np.arange(distances.size), distances, color=_colors[0], linewidth=1.75)

    ax.set_xlim(0, distances.size - 1)

    ax.set_ylabel('Total Variation Distance', fontsize=13.0)
    if full_scale:
        ax.set_ylim(0.0, 1.0)
    else:
        distance_max = _np.max(distances)
        ax.set_ylim(0.0, min(1.0, distance_max * 1.1) if distance_max > 0.0 else 1.0)

    ax.grid()

    ax.set_title('Convergence Plot', fontsize=15.0, fontweight='bold')

    return figure, ax


@_dc.interactive_plot
def plot_eigenvalues(model: _ct.tmdl, dpi: int = 100) -> _ct.tplot:

    """
    | The function plots the eigenvalues of the transition matrix of the given model on the complex plane.

    | **Notes:**

    - For **irreducible** models, the plot also illustrates the second-largest eigenvalue modulus and the corresponding spectral gap.
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

    if model.__class__.__name__ == 'MarkovChain':
        mc = model
    else:
        mc = _MarkovChain(model.p, model.states)

    figure, ax = _mplp.subplots(dpi=dpi)

    handles, labels = [], []

    theta = _np.linspace(0.0, 2.0 * _np.pi, 200)

    evalues = _npl.eigvals(model.p).astype(complex)
    evalues_final = _np.copy(evalues)

    argmin_index = _np.argmin(_np.abs(evalues_final - 1.0))

    if _np.isclose(evalues_final[argmin_index], 1.0):
        evalues_final[argmin_index] = 1.0 + 0.0j
    else:  # pragma: no cover
        evalues_final = _np.append(evalues_final, 1.0 + 0.0j)

    evalues_final = _np.unique(evalues_final)

    x_unit_circle = _np.cos(theta)
    y_unit_circle = _np.sin(theta)

    if mc.is_irreducible:

        mu = _n.slem(model.p)

        if not _np.isclose(mu, 0.0):

            x_slem_circle = mu * x_unit_circle
            y_slem_circle = mu * y_unit_circle

            if not _np.isclose(mu, 1.0):

                cs = _np.linspace(-1.1, 1.1, 201)
                x_spectral_gap, y_spectral_gap = _np.meshgrid(cs, cs)
                z_spectral_gap = x_spectral_gap**2.0 + y_spectral_gap**2.0

                h = ax.contourf(x_spectral_gap, y_spectral_gap, z_spectral_gap, alpha=0.2, colors='r', levels=[mu**2.0, 1.0])
                handles.append(_mplp.Rectangle((0.0, 0.0), 1.0, 1.0, fc=h.get_facecolor()[0]))
                labels.append('Spectral Gap')

            ax.plot(x_slem_circle, y_slem_circle, color='red', linestyle='--', linewidth=1.5)

    ax.plot(x_unit_circle, y_unit_circle, color='red', linestyle='-', linewidth=3.0)

    h, = ax.plot(_np.real(evalues_final), _np.imag(evalues_final), color='blue', linestyle='None', marker='*', markersize=12.5)
    handles.append(h)
    labels.append('Eigenvalues')

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')

    formatter = _mplt.FormatStrFormatter('%g')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xticks(_np.linspace(-1.0, 1.0, 9), minor=False)
    ax.set_yticks(_np.linspace(-1.0, 1.0, 9), minor=False)

    ax.grid(which='major')

    ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=len(handles))
    ax.set_title('Eigenvalues Plot', fontsize=15.0, fontweight='bold')

    _mplp.subplots_adjust(bottom=0.2)

    return figure, ax


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

    if model.__class__.__name__ == 'MarkovChain':
        mc = model
    else:
        mc = _MarkovChain(model.p, model.states)

    p = mc.p

    indices = list(range(0, (steps * interval) + 1, interval))
    distributions = mc.propagate(indices[-1], initial_status=initial_status, output_last=False)
    distributions = _np.transpose(_np.stack([distribution for index, distribution in enumerate(distributions) if index in indices]))

    d = distributions * 0.9
    bb, bt = _pl._get_boundaries(d)
    bm = (bb + bt) / 2.0
    c = _pl._get_colors(d)
    lh, ll = _pl._get_legend(mc, c)

    polygons_bars = _pl._get_polygons_bars(d, bb, bt, c)
    polygons_flows = _pl._get_polygons_flows(p, d, bb, c, interval)

    figure, ax = _mplp.subplots(dpi=dpi)

    for e in polygons_flows:
        ax.add_patch(e)

    for e in polygons_bars:
        ax.add_patch(e)

    for ai in range(distributions.shape[0]):
        for aj in range(distributions.shape[1]):

            dv = distributions[ai, aj]

            if dv > 0.05:
                ax.text(aj, float(bm[ai, aj]), f'{dv:.3f}', horizontalalignment='center', verticalalignment='center')

    _pl._xticks_steps(ax, steps + 1, interval)

    ax.set_ylim(0.0, 1.0)
    ax.invert_yaxis()

    ax.legend(lh, ll, bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=len(lh))
    ax.set_title('Flow Plot', fontsize=15.0, fontweight='bold')

    _mplp.subplots_adjust(bottom=0.2)

    return figure, ax


# noinspection PyBroadException
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

    extended_graph = not force_standard and _pydot_found

    if extended_graph:
        try:
            _sub.call(['dot', '-V'], stdout=_sub.PIPE, stderr=_sub.PIPE)
        except Exception:  # pragma: no cover
            extended_graph = False

    model_mc = model.__class__.__name__ == 'MarkovChain'

    if extended_graph:
        func = _pl._plot_mc_extended if model_mc else _pl._plot_hmm_extended
    else:
        func = _pl._plot_mc_standard if model_mc else _pl._plot_hmm_standard

    figure, ax = func(model, nodes_color, nodes_shape, edges_label, dpi)
    figure.suptitle('Graph Plot', fontsize=15.0, fontweight='bold')

    return figure, ax


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

    if underlying_matrix == 'emission' and not isinstance(model, _HiddenMarkovModel):  # pragma: no cover
        raise _ValidationError('An emission matrix can only be plotted for a hidden Markov model.')

    matrix = model.e if underlying_matrix == 'emission' else model.p
    labels_x = model.symbols if underlying_matrix == 'emission' else model.states

    figure, ax = _mplp.subplots(dpi=dpi)

    color_map = _mplcr.LinearSegmentedColormap.from_list('ColorMap', [_color_white, _colors[0]], 20)
    ax_is = ax.imshow(matrix, aspect='auto', cmap=color_map, interpolation='none', vmin=0.0, vmax=1.0)

    _pl._xticks_labels(ax, len(labels_x), 'Symbols' if underlying_matrix == 'emission' else 'States', labels_x, True)
    _pl._yticks_labels(ax, len(model.states), 'States', model.states)

    ax.grid(which='minor', color='k')

    cb = figure.colorbar(ax_is, drawedges=True, orientation='horizontal', ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    cb.ax.set_xticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])

    ax.set_title('Matrix Plot (' + underlying_matrix.capitalize() + ')', fontsize=15.0, fontweight='bold')

    return figure, ax


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

    if model.__class__.__name__ == 'MarkovChain':
        mc = model
    else:
        mc = _MarkovChain(model.p, model.states)

    distributions = mc.propagate(redistributions, initial_status=initial_status, output_last=False)
    distributions_length = 1 if isinstance(distributions, _np.ndarray) else len(distributions)
    distributions = _np.array([distributions]) if isinstance(distributions, _np.ndarray) else _np.array(distributions)

    figure, ax = _mplp.subplots(dpi=dpi)

    if plot_type == 'heatmap':

        color_map = _mplcr.LinearSegmentedColormap.from_list('ColorMap', [_color_white, _colors[0]], 20)
        ax_is = ax.imshow(_np.transpose(distributions), aspect='auto', cmap=color_map, interpolation='none', vmin=0.0, vmax=1.0)

        _pl._xticks_steps(ax, distributions_length)
        _pl._yticks_labels(ax, mc.size, None, mc.states)

        ax.grid(which='minor', color='k')

        cb = figure.colorbar(ax_is, drawedges=True, orientation='horizontal', ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
        cb.ax.set_xticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])

        ax.set_title('Redistributions Plot (Heatmap)', fontsize=15.0, fontweight='bold')

    else:

        ax.set_prop_cycle('color', _colors)

        for i in range(mc.size):

            overlaps = [j for j in range(mc.size) if _np.allclose(distributions[:, i], distributions[:, j], rtol=1e-12, atol=1e-14)]
            overlap_index = overlaps.index(i)
            marker = 'o' if distributions_length == 2 else None
            linestyle = '-'

            if len(overlaps) > 1:
                linestyle = ('-', '--', ':', '-.')[overlap_index % 4]
                if overlap_index > 0:
                    marker = ('o', 's', '^', 'D')[((overlap_index - 1) // 4) % 4]

            ax.plot(
                _np.arange(0.0, distributions_length, 1.0), distributions[:, i],
                label=mc.states[i], linewidth=1.75, linestyle=linestyle, marker=marker,
                markevery=max(1, distributions_length // 8), markersize=4.0
            )

        if _np.allclose(distributions[0, :], _np.ones(mc.size, dtype=float) / mc.size):
            ax.plot(0.0, distributions[0, 0], color=_color_black, label="Start", marker='o', markeredgecolor=_color_black, markerfacecolor=_color_black)
            legend_size = mc.size + 1
        else:  # pragma: no cover
            legend_size = mc.size

        _pl._xticks_steps(ax, distributions_length)
        _pl._yticks_frequency(ax, -0.05, 1.05)

        ax.grid()

        ax.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=legend_size)
        ax.set_title('Redistributions Plot (Projection)', fontsize=15.0, fontweight='bold')

        _mplp.subplots_adjust(bottom=0.2)

    return figure, ax


@_dc.interactive_plot
def plot_sequence(model: _ct.tmdl, steps: int, initial_state: _ct.ostate = None, plot_type: str = 'histogram', seed: _ct.oint = None, dpi: int = 100) -> _ct.tplot:

    """
    | The function plots a random sequence of states and, for hidden Markov models, emitted symbols.

    | **Notes:**

    - If `Matplotlib <https://matplotlib.org/>`_ is in `interactive mode <https://matplotlib.org/stable/users/interactive.html>`_, the plot is immediately displayed.

    :param model: the model.
    :param steps: the number of steps to simulate.
    :param initial_state: the initial state of the sequence (*if omitted, it is chosen uniformly at random*).
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
        dpi = _v.validate_dpi(dpi)

    except Exception as ex:  # pragma: no cover
        raise _u.create_validation_error(ex, _ins.trace()) from None

    model_mc = model.__class__.__name__ == 'MarkovChain'
    model_sequence = model.simulate(steps, initial_state=initial_state, output_indices=True, seed=seed)

    if model_mc:
        walk_data = (
            steps + 1,
            [(model.n, 'States', model.states, model_sequence)]
        )
    else:
        walk_data = (
            steps + 1,
            [(model.n, 'States', model.states, model_sequence[0]),
             (model.k, 'Symbols', model.symbols, model_sequence[1])]
        )

    if plot_type == 'heatmap':
        func = _pl._plot_heatmap
    elif plot_type == 'histogram':
        func = _pl._plot_histogram
    else:
        func = _pl._plot_matrix

    figure, ax = func(walk_data, dpi)

    return figure, ax


# noinspection PyBroadException
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
        force_standard = _v.validate_boolean(force_standard)
        dpi = _v.validate_dpi(dpi)

    except Exception as ex:  # pragma: no cover
        raise _u.create_validation_error(ex, _ins.trace()) from None

    if initial_state is None:
        initial_distribution = _np.full(hmm.n, 1.0 / hmm.n, dtype=float)
    else:
        initial_distribution = _np.zeros(hmm.n, dtype=float)
        initial_distribution[initial_state] = 1.0

    _, symbols = hmm.simulate(steps, initial_state=initial_state, output_indices=True, seed=seed)

    decoding = hmm.decode(symbols, initial_status=initial_distribution)

    if decoding is None:  # pragma: no cover
        raise ValueError('The computation of backward and forward probabilities failed.')

    _, _, _, backward, forward, _ = decoding

    prediction = hmm.predict('mle', symbols, initial_status=initial_distribution, output_indices=True)

    if prediction is None:  # pragma: no cover
        raise ValueError('The computation of the most probable states path failed.')

    _, states_path = prediction

    extended_graph = not force_standard and _pydot_found

    if extended_graph:
        try:
            _sub.call(['dot', '-V'], stdout=_sub.PIPE, stderr=_sub.PIPE)
        except Exception:  # pragma: no cover
            extended_graph = False

    if extended_graph:
        figure, ax = _pl._plot_extended(hmm, initial_distribution, symbols, backward, forward, states_path, dpi)
    else:
        figure, ax = _pl._plot_standard(hmm, initial_distribution, symbols, backward, forward, states_path, dpi)

    return figure, ax
