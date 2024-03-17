# -*- coding: utf-8 -*-

__all__ = [
    'plot_comparison',
    'plot_eigenvalues',
    'plot_flow',
    'plot_graph',
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
import scipy.interpolate as _spip

try:
    import pydot as _pyd
    _pydot_found = True
except ImportError:  # pragma: no cover
    _pyd = None
    _pydot_found = False

# Internal

from .custom_types import (
    oint as _oint,
    olist_str as _olist_str,
    oplot as _oplot,
    ostate as _ostate,
    ostatus as _ostatus,
    thmm as _thmm,
    tlist_model as _tlist_model,
    tmodel as _tmodel
)

from .exceptions import (
    ValidationError as _ValidationError
)

from .hidden_markov_model import (
    HiddenMarkovModel as _HiddenMarkovModel
)

from .markov_chain import (
    MarkovChain as _MarkovChain
)

from .utilities import (
    create_validation_error as _create_validation_error,
    create_models_names as _create_models_names
)

from .validation import (
    validate_boolean as _validate_boolean,
    validate_dpi as _validate_dpi,
    validate_enumerator as _validate_enumerator,
    validate_hidden_markov_model as _validate_hidden_markov_model,
    validate_integer as _validate_integer,
    validate_label as _validate_label,
    validate_model as _validate_model,
    validate_models as _validate_models,
    validate_status as _validate_status,
    validate_strings as _validate_strings
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


def _decode_image(g, dpi):

    buffer = _io.BytesIO()
    buffer.write(g.create(format='png'))
    buffer.seek(0)

    img = _mpli.imread(buffer)

    img_x = img.shape[0] / dpi
    img_xi = img_x * 1.1
    img_xo = ((img_xi - img_x) / 2.0) * dpi

    img_y = img.shape[1] / dpi
    img_yi = img_y * 1.1
    img_yo = ((img_yi - img_y) / 2.0) * dpi

    return img, img_x, img_xo, img_y, img_yo


def _xticks_labels(ax, size, labels_name, labels, minor_major):

    if labels_name is not None:
        ax.set_xlabel(labels_name, fontsize=13.0)

    if minor_major:
        ax.set_xticks(_np.arange(0.0, size, 1.0), minor=False)
        ax.set_xticks(_np.arange(-0.5, size, 1.0), minor=True)
    else:
        ax.set_xticks(_np.arange(0.0, size, 1.0))

    ax.set_xticklabels(labels)


def _xticks_steps(ax, length):

    ax.set_xlabel('Steps', fontsize=13.0)
    ax.set_xticks(_np.arange(0.0, length + 1.0, 1.0 if length <= 11 else 10.0), minor=False)
    ax.set_xticks(_np.arange(-0.5, length, 1.0), minor=True)
    ax.set_xticklabels(_np.arange(0, length + 1, 1 if length <= 11 else 10))
    ax.set_xlim(-0.5, length - 0.5)


def _yticks_frequency(ax, bottom, top):

    ax.set_ylabel('Frequency', fontsize=13.0)
    ax.set_yticks(_np.linspace(0.0, 1.0, 11))
    ax.set_ylim(bottom, top)


def _yticks_labels(ax, size, labels_name, labels):

    if labels_name is not None:
        ax.set_ylabel(labels_name, fontsize=13.0)

    ax.set_yticks(_np.arange(0.0, size, 1.0), minor=False)
    ax.set_yticks(_np.arange(-0.5, size, 1.0), minor=True)
    ax.set_yticklabels(labels)


def plot_comparison(models: _tlist_model, underlying_matrices: str = 'transition', names: _olist_str = None, dpi: int = 100) -> _oplot:

    """
    The function plots the underlying matrices of the given models in the form of a heatmap.

    | **Notes:**

    * If `Matplotlib <https://matplotlib.org/>`_ is in `interactive mode <https://matplotlib.org/stable/users/interactive.html>`_, the plot is immediately displayed and the function does not return the plot handles.

    :param models: the models.
    :param underlying_matrices:
     - **emission** for comparing the emission matrices;
     - **transition** for comparing the transition matrices.
    :param names: the name of each model subplot (*if omitted, a standard name is given to each subplot*).
    :param dpi: the resolution of the plot expressed in dots per inch.
    :raises ValidationError: if any input argument is not compliant.
    """

    try:

        models = _validate_models(models)
        underlying_matrices = _validate_enumerator(underlying_matrices, ['emission', 'transition'])
        names = _create_models_names(models) if names is None else _validate_strings(names, len(models))
        dpi = _validate_dpi(dpi)

    except Exception as ex:  # pragma: no cover
        raise _create_validation_error(ex, _ins.trace()) from None

    if underlying_matrices == 'emission' and not all(isinstance(model, _HiddenMarkovModel) for model in models):  # pragma: no cover
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

    color_map_ax, color_map_ax_kwargs = _mplcb.make_axes(axes, drawedges=True, orientation='horizontal', ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    figure.colorbar(ax_is, cax=color_map_ax, **color_map_ax_kwargs)
    color_map_ax.set_xticklabels([0.0, 0.25, 0.5, 0.75, 1.0])

    figure.suptitle('Comparison Plot', fontsize=15.0, fontweight='bold')

    if _mplp.isinteractive():  # pragma: no cover
        _mplp.show(block=False)
        return None

    return figure, axes


def plot_eigenvalues(model: _tmodel, dpi: int = 100) -> _oplot:

    """
    The function plots the eigenvalues of the transition matrix of the given model on the complex plane.

    | **Notes:**

    * If `Matplotlib <https://matplotlib.org/>`_ is in `interactive mode <https://matplotlib.org/stable/users/interactive.html>`_, the plot is immediately displayed and the function does not return the plot handles.

    :param model: the model.
    :param dpi: the resolution of the plot expressed in dots per inch.
    :raises ValidationError: if any input argument is not compliant.
    """

    try:

        model = _validate_model(model)
        dpi = _validate_dpi(dpi)

    except Exception as ex:  # pragma: no cover
        raise _create_validation_error(ex, _ins.trace()) from None

    if model.__class__.__name__ == 'MarkovChain':
        mc = model
    else:
        mc = _MarkovChain(model.p, model.states)

    figure, ax = _mplp.subplots(dpi=dpi)

    handles, labels = [], []

    theta = _np.linspace(0.0, 2.0 * _np.pi, 200)

    evalues = _npl.eigvals(model.p).astype(complex)
    evalues_final = _np.unique(_np.append(evalues, _np.array([1.0]).astype(complex)))

    x_unit_circle = _np.cos(theta)
    y_unit_circle = _np.sin(theta)

    if mc.is_ergodic:

        values_abs = _np.sort(_np.abs(evalues))
        values_ct1 = _np.isclose(values_abs, 1.0)

        if not _np.all(values_ct1):

            mu = values_abs[~values_ct1][-1]

            if not _np.isclose(mu, 0.0):

                x_slem_circle = mu * x_unit_circle
                y_slem_circle = mu * y_unit_circle

                cs = _np.linspace(-1.1, 1.1, 201)
                x_spectral_gap, y_spectral_gap = _np.meshgrid(cs, cs)
                z_spectral_gap = x_spectral_gap**2.0 + y_spectral_gap**2.0

                h = ax.contourf(x_spectral_gap, y_spectral_gap, z_spectral_gap, alpha=0.2, colors='r', levels=[mu**2.0, 1.0])
                handles.append(_mplp.Rectangle((0.0, 0.0), 1.0, 1.0, fc=h.collections[0].get_facecolor()[0]))
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

    if _mplp.isinteractive():  # pragma: no cover
        _mplp.show(block=False)
        return None

    return figure, ax


def plot_flow(model: _tmodel, steps: int, interval: int, initial_status: _ostatus = None, palette: str = 'viridis', dpi: int = 100) -> _oplot:

    """
    The function produces an alluvial diagram of the given model.

    | **Notes:**

    * If `Matplotlib <https://matplotlib.org/>`_ is in `interactive mode <https://matplotlib.org/stable/users/interactive.html>`_, the plot is immediately displayed and the function does not return the plot handles.

    :param model: the model.
    :param steps: the number of steps.
    :param interval: the interval between each step.
    :param initial_status: the initial state or the initial distribution of the states (*if omitted, the states are assumed to be uniformly distributed*).
    :param palette: the palette of the plot.
    :param dpi: the resolution of the plot expressed in dots per inch.
    :raises ValidationError: if any input argument is not compliant.
    """

    def _get_boundaries(gb_d):

        i, j = gb_d.shape
        k = 0.1 / (i - 1.0)

        b = _np.zeros((i, j), dtype=float)
        t = _np.zeros((i, j), dtype=float)

        for o in range(j):
            dj = d[:, o]
            b[:, o] = _np.cumsum(dj + k) - dj - k
            t[:, o] = _np.cumsum(dj + k) - k

        b = _np.clip(b, 0.0, 1.0)
        t = _np.clip(t, 0.0, 1.0)

        return b, t

    def _get_colors(gc_pn, gc_d):

        i = gc_d.shape[0]

        cm = _np.array(_mplp.get_cmap(gc_pn).colors)

        ipf = _spip.interp1d(_np.linspace(0.0, 1.0, cm.shape[0]), cm, kind='linear', axis=0)
        ipv = ipf(_np.linspace(0.0, 1.0, 3 + ((i - 1) * 10)))

        cm = ipv[1:-1:10, :]

        return cm

    def _get_curves(gc_n, gc_x1, gc_y1, gc_x2, gc_y2):

        tx = _np.reshape(_np.linspace(gc_x1, gc_x2, 15), (1, -1))
        cx = _np.tile(_np.transpose(tx), (1, gc_n))

        ty = (1.0 - _np.cos(_np.reshape(_np.linspace(0.0, _np.pi, 15), (1, -1)))) / 2.0
        cy = _np.tile(gc_y1, (15, 1)) + (_np.tile(gc_y2 - gc_y1, (15, 1)) * _np.tile(_np.transpose(ty), (1, gc_n)))

        return cx, cy

    def _get_legend(gl_mc, gl_c):

        handles = []
        labels = gl_mc.states

        for i, label in enumerate(labels):
            handles.append(_mplpc.Patch(color=gl_c[i, :], label=label))

        return handles, labels

    def _get_polygons_bars(gpb_d, gpb_bb, gpb_bt, gpb_c):

        i, j = gpb_d.shape
        w = j / 40.0

        polygons = []

        for oj in range(j):

            xm = oj - w
            xp = oj + w

            for oi in range(i):

                yb = gpb_bb[oi, oj]
                yt = gpb_bt[oi, oj]

                x = [xm, xp, xp, xm]
                y = [yb, yb, yt, yt]

                polygons.append(_mplpc.Polygon(list(zip(x, y)), edgecolor=None, facecolor=gpb_c[oi, :], alpha=0.8))

        return polygons

    def _get_polygons_flows(gpf_p, gpf_d, gpf_bb, gpf_c):

        i, j = gpf_d.shape
        w = j / 40.0

        polygons = []

        for oj in range(j - 1):

            q = _npl.matrix_power(gpf_p, indices[oj + 1] - indices[oj])
            bj = _np.copy(gpf_bb[:, oj + 1])

            x_lo = oj + w
            x_hi = oj - w + 1.0

            for oi in range(i):

                dij = gpf_d[oi, oj]
                bij = bb[oi, oj]

                qi = q[oi, :]
                qis = _np.cumsum(qi)

                tl = ((qis - qi) * dij) + bij
                bl = (qis * dij) + bij
                tr = _np.copy(bj)
                br = tr + bl - tl

                bj += bl - tl

                [bottom_x, bottom_y] = _get_curves(i, x_lo, bl, x_hi, br)
                [top_x, top_y] = _get_curves(i, x_hi, tr, x_lo, tl)
                x = _np.concatenate([bottom_x, top_x], axis=0)
                y = _np.concatenate([bottom_y, top_y], axis=0)

                for z in range(x.shape[1]):
                    polygons.append(_mplpc.Polygon(list(zip(x[:, z], y[:, z])), edgecolor=None, facecolor=gpf_c[oi, :], alpha=0.3))

        return polygons

    try:

        model = _validate_model(model)
        steps = _validate_integer(steps, lower_limit=(1, False))
        interval = _validate_integer(interval, lower_limit=(1, False))
        initial_status = None if initial_status is None else _validate_status(initial_status, model.states)
        palette = _validate_enumerator(palette, list(_mplp.colormaps))
        dpi = _validate_dpi(dpi)

    except Exception as ex:  # pragma: no cover
        raise _create_validation_error(ex, _ins.trace()) from None

    if model.__class__.__name__ == 'MarkovChain':
        mc = model
    else:
        mc = _MarkovChain(model.p, model.states)

    p = mc.p

    indices = list(range(0, (steps * interval) + 1, interval))
    distributions = mc.redistribute(indices[-1], initial_status=initial_status, output_last=False)
    distributions = _np.transpose(_np.stack([distribution for index, distribution in enumerate(distributions) if index in indices]))

    d = distributions * 0.9
    bb, bt = _get_boundaries(d)
    bm = (bb + bt) / 2.0
    c = _get_colors(palette, d)
    lh, ll = _get_legend(mc, c)

    polygons_bars = _get_polygons_bars(d, bb, bt, c)
    polygons_flows = _get_polygons_flows(p, d, bb, c)

    figure, ax = _mplp.subplots(dpi=dpi)

    for e in polygons_flows:
        ax.add_patch(e)

    for e in polygons_bars:
        ax.add_patch(e)

    for ai in range(distributions.shape[0]):
        for aj in range(distributions.shape[1]):

            dv = distributions[ai, aj]

            if dv > 0.05:
                ax.text(aj, bm[ai, aj], f'{dv:.3f}', horizontalalignment='center', verticalalignment='center')

    _xticks_steps(ax, steps)

    ax.set_ylim(0.0, 1.0)
    ax.invert_yaxis()

    ax.legend(lh, ll, bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=len(lh))
    ax.set_title('Flow Plot', fontsize=15.0, fontweight='bold')

    _mplp.subplots_adjust(bottom=0.2)

    if _mplp.isinteractive():  # pragma: no cover
        _mplp.show(block=False)
        return None

    return figure, ax


# noinspection PyBroadException
def plot_graph(model: _tmodel, nodes_color: bool = True, nodes_shape: bool = True, edges_label: bool = True, force_standard: bool = False, dpi: int = 100) -> _oplot:

    """
    The function plots the directed graph of the given model.

    | **Notes:**

    * If `Matplotlib <https://matplotlib.org/>`_ is in `interactive mode <https://matplotlib.org/stable/users/interactive.html>`_, the plot is immediately displayed and the function does not return the plot handles.
    * `Graphviz <https://graphviz.org/>`_ and `pydot <https://pypi.org/project/pydot/>`_ are not required, but they provide access to extended mode with improved rendering and additional features.
    * The rendering, especially in standard mode or for big graphs, is not granted to be high-quality.
    * For Markov chains, the color of nodes is based on communicating classes; for hidden Markov models, every state node has a different color and symbol nodes are gray.
    * For Markov chains, recurrent nodes have an elliptical shape and transient nodes have a rectangular shape; for hidden Markov models, state nodes have an elliptical shape and symbol nodes have a hexagonal shape.

    :param model: the model.
    :param nodes_color: a boolean indicating whether to use a different color for every type of node.
    :param nodes_shape: a boolean indicating whether to use a different shape for every type of node.
    :param edges_label: a boolean indicating whether to display the probability of every edge as text.
    :param force_standard: a boolean indicating whether to use standard mode even if extended mode is available.
    :param dpi: the resolution of the plot expressed in dots per inch.
    :raises ValidationError: if any input argument is not compliant.
    """

    def _calculate_magnitude(*cm_elements):

        magnitudes = []

        for element in cm_elements:
            element_minimum = _np.min(element).item()
            element_magnitude = 0 if element_minimum == 0.0 else int(-_mt.floor(_mt.log10(abs(element_minimum))))
            magnitudes.append(element_magnitude)

        magnitude = max(1, min(max(magnitudes), 4))

        return magnitude

    def _draw_edge_labels_curved(delc_ax, delc_positions, delc_edge_labels):

        for (n1, n2), (rad, label) in delc_edge_labels.items():

            (x1, y1) = delc_positions[n1]
            (x2, y2) = delc_positions[n2]
            p1 = delc_ax.transData.transform(_np.array(delc_positions[n1]))
            p2 = delc_ax.transData.transform(_np.array(delc_positions[n2]))

            linear_mid = (0.5 * p1) + (0.5 * p2)
            cp_mid = linear_mid + (rad * _np.dot(_np.array([(0, 1), (-1, 0)]), p2 - p1))
            cp1 = (0.5 * p1) + (0.5 * cp_mid)
            cp2 = (0.5 * p2) + (0.5 * cp_mid)
            bezier_mid = (0.5 * cp1) + (0.5 * cp2)

            (x, y) = delc_ax.transData.inverted().transform(bezier_mid)
            xy = _np.array((x, y))

            angle = (_np.arctan2(y2 - y1, x2 - x1) / (2.0 * _np.pi)) * 360.0

            if angle > 90.0:
                angle -= 180.0

            if angle < -90.0:
                angle += 180.0

            rotation = delc_ax.transData.transform_angles(_np.array((angle,)), xy.reshape((1, 2)))[0]
            transform = delc_ax.transData
            bbox = {
                'boxstyle': 'round',
                'ec': (1.0, 1.0, 1.0),
                'fc': (1.0, 1.0, 1.0)
            }

            delc_ax.text(
                x, y,
                label, color='k', size=10, family='sans-serif', weight='normal',
                horizontalalignment='center', verticalalignment='center',
                bbox=bbox, clip_on=True, rotation=rotation, transform=transform, zorder=1
            )

        delc_ax.tick_params(
            axis='both', which='both',
            bottom=False, left=False,
            labelbottom=False, labelleft=False
        )

    def _node_colors(nc_count):

        colors = _cp.deepcopy(_colors)
        colors_limit = len(colors) - 1
        colors_offset = 0

        colors_list = []

        while nc_count > 0:

            colors_list.append(colors[colors_offset])
            colors_offset += 1

            if colors_offset > colors_limit:  # pragma: no cover
                colors_offset = 0

            nc_count -= 1

        return colors_list

    # noinspection DuplicatedCode
    def _plot_hmm_extended(phe_hmm, phe_nodes_color, phe_nodes_type, phe_edges_label, phe_dpi):

        magnitude = _calculate_magnitude(phe_hmm.p, phe_hmm.e)

        node_colors = _node_colors(phe_hmm.n) if phe_nodes_color else []
        edge_colors = _cp.deepcopy(node_colors) if phe_nodes_color else []

        g = _pyd.Dot(graph_type='digraph')

        g_sub1 = _pyd.Subgraph()
        g.add_subgraph(g_sub1)

        g_sub2 = _pyd.Subgraph(rank='same')
        g.add_subgraph(g_sub2)

        for i in range(phe_hmm.n):

            state = phe_hmm.states[i]

            node_attributes = {}

            if phe_nodes_color:
                node_attributes['style'] = 'filled'
                node_attributes['fillcolor'] = node_colors[i]

            if phe_nodes_type:
                node_attributes['shape'] = 'ellipse'

            g_sub1.add_node(_pyd.Node(state, **node_attributes))

        for symbol in phe_hmm.symbols:

            node_attributes = {}

            if phe_nodes_color:
                node_attributes['style'] = 'filled'
                node_attributes['fillcolor'] = _default_color_symbol

            if phe_nodes_type:
                node_attributes['shape'] = 'hexagon'

            g_sub2.add_node(_pyd.Node(symbol, **node_attributes))

        for i in range(phe_hmm.n):

            state_i = phe_hmm.states[i]

            for j in range(phe_hmm.n):

                tp = phe_hmm.p[i, j]

                if tp > 0.0:

                    state_j = phe_hmm.states[j]

                    edge_attributes = {
                        'style': 'filled',
                        'color': _default_color_edge
                    }

                    if phe_edges_label:
                        edge_attributes['label'] = f' {round(tp, magnitude):.{magnitude}f} '
                        edge_attributes['fontsize'] = 9

                    g.add_edge(_pyd.Edge(state_i, state_j, **edge_attributes))

            for j in range(phe_hmm.k):

                ep = phe_hmm.e[i, j]

                if ep > 0.0:

                    symbol = phe_hmm.symbols[j]

                    edge_attributes = {
                        'style': 'dashed',
                        'color': edge_colors[i] if phe_nodes_color else _default_color_edge
                    }

                    if phe_edges_label:
                        edge_attributes['label'] = f' {round(ep, magnitude):.{magnitude}f} '
                        edge_attributes['fontsize'] = 9

                    g.add_edge(_pyd.Edge(state_i, symbol, **edge_attributes))

        img, img_x, img_xo, img_y, img_yo = _decode_image(g, phe_dpi)

        f = _mplp.figure(figsize=(img_y * 1.1, img_x * 1.1), dpi=phe_dpi)
        f.figimage(img, yo=img_yo, xo=img_xo)

        a = f.gca()
        a.axis('off')

        return f, a

    def _plot_hmm_standard(phs_hmm, phs_nodes_color, phs_nodes_shape, phe_edges_label, phs_dpi):

        g = phs_hmm.to_graph()
        positions = _nx.multipartite_layout(g, align='horizontal', subset_key='layer')
        magnitude = _calculate_magnitude(phs_hmm.p, phs_hmm.e)

        node_colors = _node_colors(phs_hmm.n) if phs_nodes_color else []
        edge_colors = _cp.deepcopy(node_colors) if phs_nodes_color else []

        mpi = _mplp.isinteractive()
        _mplp.interactive(False)

        f, a = _mplp.subplots(dpi=phs_dpi)

        for i, node in enumerate(g.nodes):

            if phs_nodes_color:
                if node in phs_hmm.states:
                    node_color = node_colors[i]
                else:
                    node_color = _default_color_symbol
            else:
                node_color = _default_color_node

            if phs_nodes_shape:
                if node in phs_hmm.states:
                    node_shape = 'o'
                else:
                    node_shape = 'H'
            else:
                node_shape = 'o'

            _nx.draw_networkx_nodes(g, positions, ax=a, nodelist=[node], node_color=node_color, node_shape=node_shape, node_size=_default_node_size, edgecolors=_color_black)

        _nx.draw_networkx_labels(g, positions, ax=a)

        edge_labels_curved, edge_labels_straight_state, edge_labels_straight_symbol = {}, {}, {}

        for i in range(phs_hmm.n):

            state_i = phs_hmm.states[i]

            for j in range(phs_hmm.n):

                tp = phs_hmm.p[i, j]

                if tp > 0.0:

                    state_j = phs_hmm.states[j]

                    edge = (state_i, state_j)
                    edge_color = _default_color_edge

                    if i != j and reversed(edge) in g.edges:

                        edge_length = abs(i - j)
                        edge_rad = 0.15 if edge_length == 1 else edge_length * 0.25
                        edge_connection = f'arc3, rad={edge_rad:f}'

                        if phe_edges_label:
                            edge_labels_curved[edge] = (edge_rad, f' {round(tp, magnitude):.{magnitude}f} ')

                    else:

                        edge_connection = 'arc3'

                        if phe_edges_label:
                            edge_labels_straight_state[edge] = f' {round(tp, magnitude):.{magnitude}f} '

                    _nx.draw_networkx_edges(g, positions, ax=a, edgelist=[edge], edge_color=edge_color, arrows=True, connectionstyle=edge_connection)

            for j in range(phs_hmm.k):

                ep = phs_hmm.e[i, j]

                if ep > 0.0:

                    symbol_j = phs_hmm.symbols[j]

                    edge = (state_i, symbol_j)
                    edge_color = edge_colors[i] if phs_nodes_color else _default_color_edge

                    if phe_edges_label:
                        edge_labels_straight_symbol[edge] = f' {round(ep, magnitude):.{magnitude}f} '

                    _nx.draw_networkx_edges(g, positions, ax=a, edgelist=[edge], edge_color=edge_color, arrows=True, style='dashed')

        if len(edge_labels_straight_state) > 0:
            _nx.draw_networkx_edge_labels(g, positions, ax=a, edge_labels=edge_labels_straight_state)

        if len(edge_labels_straight_symbol) > 0:
            _nx.draw_networkx_edge_labels(g, positions, ax=a, edge_labels=edge_labels_straight_symbol, label_pos=0.25)

        if len(edge_labels_curved) > 0:
            _draw_edge_labels_curved(a, positions, edge_labels_curved)

        _mplp.interactive(mpi)

        return f, a

    # noinspection DuplicatedCode
    def _plot_mc_extended(pme_mc, pme_nodes_color, pme_nodes_shape, phe_edges_label, pme_dpi):

        magnitude = _calculate_magnitude(pme_mc.p)

        node_colors = _node_colors(len(pme_mc.communicating_classes)) if pme_nodes_color else []

        g = _pyd.Dot(graph_type='digraph')

        for i in range(pme_mc.size):

            state_i = pme_mc.states[i]

            node_attributes = {}

            if pme_nodes_color:
                for index, cc in enumerate(pme_mc.communicating_classes):
                    if state_i in cc:
                        node_attributes['style'] = 'filled'
                        node_attributes['fillcolor'] = node_colors[index]
                        break

            if pme_nodes_shape:
                if state_i in pme_mc.transient_states:  # pragma: no cover
                    node_attributes['shape'] = 'box'
                else:
                    node_attributes['shape'] = 'ellipse'

            g.add_node(_pyd.Node(state_i, **node_attributes))

            for j in range(pme_mc.size):

                tp = pme_mc.p[i, j]

                if tp > 0.0:

                    state_j = pme_mc.states[j]

                    edge_attributes = {
                        'style': 'filled',
                        'color': _default_color_edge
                    }

                    if phe_edges_label:
                        edge_attributes['label'] = f' {round(tp, magnitude):.{magnitude}f} '
                        edge_attributes['fontsize'] = 9

                    g.add_edge(_pyd.Edge(state_i, state_j, **edge_attributes))

        img, img_x, img_xo, img_y, img_yo = _decode_image(g, pme_dpi)

        f = _mplp.figure(figsize=(img_y * 1.1, img_x * 1.1), dpi=pme_dpi)
        f.figimage(img, yo=img_yo, xo=img_xo)

        a = f.gca()
        a.axis('off')

        return f, a

    def _plot_mc_standard(pms_mc, pms_nodes_color, pms_nodes_shape, phe_edges_label, pms_dpi):

        g = pms_mc.to_graph()
        positions = _nx.spring_layout(g)
        magnitude = _calculate_magnitude(pms_mc.p)

        node_colors = _node_colors(len(pms_mc.communicating_classes)) if pms_nodes_color else []

        mpi = _mplp.isinteractive()
        _mplp.interactive(False)

        f, a = _mplp.subplots(dpi=pms_dpi)

        for node in g.nodes:

            node_color = _default_color_node

            if pms_nodes_color:
                for index, cc in enumerate(pms_mc.communicating_classes):
                    if node in cc:
                        node_color = node_colors[index]
                        break

            if pms_nodes_shape:
                if node in pms_mc.transient_states:  # pragma: no cover
                    node_shape = 's'
                else:
                    node_shape = 'o'
            else:
                node_shape = 'o'

            _nx.draw_networkx_nodes(g, positions, ax=a, nodelist=[node], node_color=node_color, node_shape=node_shape, node_size=_default_node_size, edgecolors=_color_black)

        _nx.draw_networkx_labels(g, positions, ax=a)

        edge_labels_curved, edge_labels_straight = {}, {}

        for i in range(pms_mc.size):

            state_i = pms_mc.states[i]

            for j in range(pms_mc.size):

                tp = pms_mc.p[i, j]

                if tp > 0.0:

                    state_j = pms_mc.states[j]

                    edge = (state_i, state_j)
                    edge_color = _default_color_edge

                    if i != j and reversed(edge) in g.edges:

                        edge_connection = 'arc3, rad=0.1'

                        if phe_edges_label:
                            edge_labels_curved[edge] = (0.1, f' {round(tp, magnitude):.{magnitude}f} ')

                    else:

                        edge_connection = 'arc3'

                        if phe_edges_label:
                            edge_labels_straight[edge] = f' {round(tp, magnitude):.{magnitude}f} '

                    _nx.draw_networkx_edges(g, positions, ax=a, edgelist=[edge], edge_color=edge_color, arrows=True, connectionstyle=edge_connection)

        if len(edge_labels_straight) > 0:
            _nx.draw_networkx_edge_labels(g, positions, ax=a, edge_labels=edge_labels_straight)

        if len(edge_labels_curved) > 0:
            _draw_edge_labels_curved(a, positions, edge_labels_curved)

        _mplp.interactive(mpi)

        return f, a

    try:

        model = _validate_model(model)
        nodes_color = _validate_boolean(nodes_color)
        nodes_shape = _validate_boolean(nodes_shape)
        edges_label = _validate_boolean(edges_label)
        force_standard = _validate_boolean(force_standard)
        dpi = _validate_dpi(dpi)

    except Exception as ex:  # pragma: no cover
        raise _create_validation_error(ex, _ins.trace()) from None

    extended_graph = not force_standard and _pydot_found

    if extended_graph:
        try:
            _sub.call(['dot', '-V'], stdout=_sub.PIPE, stderr=_sub.PIPE)
        except Exception:  # pragma: no cover
            extended_graph = False

    model_mc = model.__class__.__name__ == 'MarkovChain'

    if extended_graph:
        func = _plot_mc_extended if model_mc else _plot_hmm_extended
    else:
        func = _plot_mc_standard if model_mc else _plot_hmm_standard

    figure, ax = func(model, nodes_color, nodes_shape, edges_label, dpi)

    if _mplp.isinteractive():  # pragma: no cover
        _mplp.show(block=False)
        return None

    return figure, ax


# noinspection DuplicatedCode
def plot_redistributions(model: _tmodel, redistributions: int, initial_status: _ostatus = None, plot_type: str = 'projection', dpi: int = 100) -> _oplot:

    """
    The function plots a redistribution of states on the given model.

    | **Notes:**

    * If `Matplotlib <https://matplotlib.org/>`_ is in `interactive mode <https://matplotlib.org/stable/users/interactive.html>`_, the plot is immediately displayed and the function does not return the plot handles.

    :param model: the model to be converted into a graph.
    :param redistributions: the number of redistributions to perform.
    :param initial_status: the initial state or the initial distribution of the states (*if omitted, the states are assumed to be uniformly distributed*).
    :param plot_type:
     - **heatmap** for displaying a heatmap plot;
     - **projection** for displaying a projection plot.
    :param dpi: the resolution of the plot expressed in dots per inch.
    :raises ValidationError: if any input argument is not compliant.
    :raises ValueError: if the "distributions" parameter represents a sequence of redistributions and the "initial_status" parameter does not match its first element.
    """

    try:

        model = _validate_model(model)
        redistributions = _validate_integer(redistributions, lower_limit=(1, False))
        initial_status = None if initial_status is None else _validate_status(initial_status, model.states)
        plot_type = _validate_enumerator(plot_type, ['heatmap', 'projection'])
        dpi = _validate_dpi(dpi)

    except Exception as ex:  # pragma: no cover
        raise _create_validation_error(ex, _ins.trace()) from None

    if model.__class__.__name__ == 'MarkovChain':
        mc = model
    else:
        mc = _MarkovChain(model.p, model.states)

    distributions = mc.redistribute(redistributions, initial_status=initial_status, output_last=False)

    if initial_status is not None and not _np.array_equal(distributions[0], initial_status):  # pragma: no cover
        raise ValueError('The "initial_status" parameter, if specified when the "distributions" parameter represents a sequence of redistributions, must match the first element.')

    distributions_length = 1 if isinstance(distributions, _np.ndarray) else len(distributions)
    distributions = _np.array([distributions]) if isinstance(distributions, _np.ndarray) else _np.array(distributions)

    figure, ax = _mplp.subplots(dpi=dpi)

    if plot_type == 'heatmap':

        color_map = _mplcr.LinearSegmentedColormap.from_list('ColorMap', [_color_white, _colors[0]], 20)
        ax_is = ax.imshow(_np.transpose(distributions), aspect='auto', cmap=color_map, interpolation='none', vmin=0.0, vmax=1.0)

        _xticks_steps(ax, distributions_length)
        _yticks_labels(ax, mc.size, None, mc.states)

        ax.grid(which='minor', color='k')

        cb = figure.colorbar(ax_is, drawedges=True, orientation='horizontal', ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
        cb.ax.set_xticklabels([0.0, 0.25, 0.5, 0.75, 1.0])

        ax.set_title('Redistributions Plot (Heatmap)', fontsize=15.0, fontweight='bold')

    else:

        ax.set_prop_cycle('color', _colors)

        if distributions_length == 2:
            for i in range(mc.size):
                ax.plot(_np.arange(0.0, distributions_length, 1.0), distributions[:, i], label=mc.states[i], marker='o')
        else:
            for i in range(mc.size):
                ax.plot(_np.arange(0.0, distributions_length, 1.0), distributions[:, i], label=mc.states[i])

        if _np.allclose(distributions[0, :], _np.ones(mc.size, dtype=float) / mc.size):
            ax.plot(0.0, distributions[0, 0], color=_color_black, label="Start", marker='o', markeredgecolor=_color_black, markerfacecolor=_color_black)
            legend_size = mc.size + 1
        else:  # pragma: no cover
            legend_size = mc.size

        _xticks_steps(ax, distributions_length)
        _yticks_frequency(ax, -0.05, 1.05)

        ax.grid()

        ax.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=legend_size)
        ax.set_title('Redistributions Plot (Projection)', fontsize=15.0, fontweight='bold')

        _mplp.subplots_adjust(bottom=0.2)

    if _mplp.isinteractive():  # pragma: no cover
        _mplp.show(block=False)
        return None

    return figure, ax


# noinspection DuplicatedCode
def plot_sequence(model: _tmodel, steps: int, initial_state: _ostate = None, plot_type: str = 'histogram', seed: _oint = None, dpi: int = 100) -> _oplot:

    """
    The function plots a random walk on the given model.

    | **Notes:**

    * If `Matplotlib <https://matplotlib.org/>`_ is in `interactive mode <https://matplotlib.org/stable/users/interactive.html>`_, the plot is immediately displayed and the function does not return the plot handles.

    :param model: the model.
    :param steps: the number of steps.
    :param initial_state: the initial state of the random walk (*if omitted, it is chosen uniformly at random*).
    :param plot_type:
     - **heatmap** for displaying heatmap-like plots;
     - **histogram** for displaying a histogram plots;
     - **matrix** for displaying matrix plots.
    :param seed: a seed to be used as RNG initializer for reproducibility purposes.
    :param dpi: the resolution of the plot expressed in dots per inch.
    :raises ValidationError: if any input argument is not compliant.
    """

    # noinspection DuplicatedCode
    def _plot_heatmap(phm_walk_data, phm_dpi):

        walk_steps, walks = phm_walk_data
        plots_count = len(walks)

        mpi = _mplp.isinteractive()
        _mplp.interactive(False)

        f, a = _mplp.subplots(nrows=plots_count, constrained_layout=True, dpi=phm_dpi)
        a = [a] if plots_count == 1 else list(a.flat)

        color_map = _mplcr.LinearSegmentedColormap.from_list('ColorMap', [_color_white, _colors[0]], 20)
        is_axes = []

        for a_current, (size, labels_name, labels, sequence) in zip(a, walks):

            sequence_matrix = _np.zeros((size, size), dtype=float)

            for i in range(1, walk_steps):
                sequence_matrix[sequence[i - 1], sequence[i]] += 1.0

            sequence_matrix /= _np.sum(sequence_matrix)

            a_current_is = a_current.imshow(sequence_matrix, aspect='auto', cmap=color_map, interpolation='none', vmin=0.0, vmax=1.0)
            is_axes.append(a_current_is)

            _xticks_labels(a_current, size, labels_name, labels, True)
            _yticks_labels(a_current, size, labels_name, labels)

            a_current.grid(which='minor', color='k')

        color_map_ax, color_map_ax_kwargs = _mplcb.make_axes(a, drawedges=True, orientation='vertical', ticks=[0.0, 0.25, 0.5, 0.75, 1.0])

        for is_ax in is_axes:
            f.colorbar(is_ax, cax=color_map_ax, **color_map_ax_kwargs)

        f.suptitle('Sequence Plot (Heatmap)', fontsize=15.0, fontweight='bold')

        _mplp.interactive(mpi)

        return f, a

    # noinspection DuplicatedCode
    def _plot_histogram(ph_walk_data, ph_dpi):

        walk_steps, walks = ph_walk_data
        plots_count = len(walks)

        mpi = _mplp.isinteractive()
        _mplp.interactive(False)

        f, a = _mplp.subplots(nrows=plots_count, tight_layout=True, dpi=ph_dpi)
        a = [a] if plots_count == 1 else list(a.flat)

        for a_current, (size, labels_name, labels, sequence) in zip(a, walks):

            sequence_histogram = _np.zeros((size, walk_steps), dtype=float)

            for index, label in enumerate(sequence):
                sequence_histogram[label, index] = 1.0

            sequence_histogram = _np.sum(sequence_histogram, axis=1) / _np.sum(sequence_histogram)

            a_current.bar(_np.arange(0.0, size, 1.0), sequence_histogram, edgecolor=_color_black, facecolor=_colors[0])

            _xticks_labels(a_current, size, labels_name, labels, False)
            _yticks_frequency(a_current, 0.0, 1.0)

        f.suptitle('Sequence Plot (Histogram)', fontsize=15.0, fontweight='bold')

        _mplp.interactive(mpi)

        return f, a

    # noinspection DuplicatedCode
    def _plot_matrix(pm_walk_data, pm_dpi):

        walk_steps, walks = pm_walk_data
        plots_count = len(walks)

        mpi = _mplp.isinteractive()
        _mplp.interactive(False)

        f, a = _mplp.subplots(nrows=plots_count, tight_layout=True, dpi=pm_dpi)
        a = [a] if plots_count == 1 else list(a.flat)

        color_map = _mplcr.LinearSegmentedColormap.from_list('ColorMap', [_color_white, _colors[0]], 2)

        for a_current, (size, labels_name, labels, sequence) in zip(a, walks):

            sequence_matrix = _np.zeros((size, walk_steps), dtype=float)

            for index, state in enumerate(sequence):
                sequence_matrix[state, index] = 1.0

            a_current.imshow(sequence_matrix, aspect='auto', cmap=color_map, interpolation='none', vmin=0.0, vmax=1.0)

            _xticks_steps(a_current, walk_steps)
            _yticks_labels(a_current, size, labels_name, labels)

            a_current.grid(which='minor', color='k')

        f.suptitle('Sequence Plot (Matrix)', fontsize=15.0, fontweight='bold')

        _mplp.interactive(mpi)

        return f, a

    try:

        model = _validate_model(model)
        steps = _validate_integer(steps, lower_limit=(2, False))
        initial_state = None if initial_state is None else _validate_label(initial_state, model.states)
        plot_type = _validate_enumerator(plot_type, ['heatmap', 'histogram', 'matrix'])
        dpi = _validate_dpi(dpi)

    except Exception as ex:  # pragma: no cover
        raise _create_validation_error(ex, _ins.trace()) from None

    model_mc = model.__class__.__name__ == 'MarkovChain'
    model_sequence = model.simulate(steps, initial_state=initial_state, output_indices=True, seed=seed)

    if model_mc:
        walk_data = (
            steps + 1,
            [
                (model.n, 'States', model.states, model_sequence)
            ]
        )
    else:
        walk_data = (
            steps + 1,
            [
                (model.n, 'States', model.states, model_sequence[0]),
                (model.k, 'Symbols', model.symbols, model_sequence[1])
            ]
        )

    if plot_type == 'heatmap':
        func = _plot_heatmap
    elif plot_type == 'histogram':
        func = _plot_histogram
    else:
        func = _plot_matrix

    figure, ax = func(walk_data, dpi)

    if _mplp.isinteractive():  # pragma: no cover
        _mplp.show(block=False)
        return None

    return figure, ax


# noinspection PyBroadException
def plot_trellis(hmm: _thmm, steps: int, initial_state: _ostate = None, seed: _oint = None, force_standard: bool = False, dpi: int = 100) -> _oplot:

    """
    The function plots the trellis diagrams of a random walk on the given hidden Markov model.

    | **Notes:**

    * If `Matplotlib <https://matplotlib.org/>`_ is in `interactive mode <https://matplotlib.org/stable/users/interactive.html>`_, the plot is immediately displayed and the function does not return the plot handles.
    * `Graphviz <https://graphviz.org/>`_ and `pydot <https://pypi.org/project/pydot/>`_ are not required, but they provide access to extended mode with improved rendering and additional features.
    * The rendering of large simulations is not granted to be high-quality.
    * Red nodes and edges belong to the most probable states path calculated using the Viterbi algorithm.

    :param hmm: the hidden Markov model.
    :param steps: the number of steps.
    :param initial_state: the initial state of the random walk (*if omitted, it is chosen uniformly at random*).
    :param seed: a seed to be used as RNG initializer for reproducibility purposes.
    :param force_standard: a boolean indicating whether to use standard mode even if extended mode is available.
    :param dpi: the resolution of the plot expressed in dots per inch.
    :raises ValidationError: if any input argument is not compliant.
    :raises ValueError: if the computation of backward and forward probabilities fails or if the computation of the most probable states path fails.
    """

    def _generate_trellis_extended(gte_hmm, gte_initial_distribution, gte_symbols, gte_forward, gte_matrix, gte_states_path):

        n, f = gte_hmm.n, len(gte_symbols)

        g = _pyd.Dot(graph_type='digraph', compound='true', margin='0')

        sub = _pyd.Subgraph('cluster_0', color='transparent')

        for row in range(n):

            sub_label = gte_hmm.states[row]
            sub.add_node(_pyd.Node(f'state{row}', color='transparent', label=sub_label, shape='plaintext'))

            if row > 0:
                sub.add_edge(_pyd.Edge(f'state{row - 1}', f'state{row}', style='invis'))

        g.add_subgraph(sub)

        for col in range(f):

            path_current = gte_states_path[col]

            sub_label = "<T<FONT POINT-SIZE='8'><SUB>0</SUB></FONT>>" if col == 0 else gte_hmm.symbols[gte_symbols[col]]
            sub = _pyd.Subgraph(f'cluster_{col + 1}', color='transparent', label=sub_label)

            for row in range(n):

                node_index = (row * f) + col
                node_color = _default_color_path if path_current == row else _default_color_node
                sub.add_node(_pyd.Node(f'node{node_index}', fillcolor=node_color, label=f'{round(gte_matrix[row, col], 2):.2f}', style='filled'))

                if row > 0:
                    sub.add_edge(_pyd.Edge(f'node{((row - 1) * f) + col}', f'node{node_index}', style='invis'))

            g.add_subgraph(sub)

        for row in range(n):

            row_offset = row * f

            for col in range(f - 1):

                if col == 0 and not gte_initial_distribution[col] > 0.0:
                    continue

                for row_next in range(n):

                    if hmm.p[row][row_next] > 0.0:

                        if gte_forward:
                            edge_from = f'node{row_offset + col}'
                            edge_to = f'node{(row_next * f) + col + 1}'
                        else:
                            edge_from = f'node{(row_next * f) + col + 1}'
                            edge_to = f'node{row_offset + col}'

                        edge_color = _default_color_path if gte_states_path[col] == row and gte_states_path[col + 1] == row_next else _color_black

                        g.add_edge(_pyd.Edge(edge_from, edge_to, color=edge_color, constraint='false'))

        return g

    def _generate_trellis_standard(gts_hmm, gts_initial_distribution, gts_symbols, gts_forward, gts_matrix, gts_states_path):

        n, f = gts_hmm.n, len(gts_symbols)

        trellis = _nx.DiGraph()
        node_colors, node_edges, node_labels, node_positions = [], [], {}, {}

        node_index = 0

        for row in range(n):

            row_offset = float(n - row)

            for col in range(f):

                trellis.add_node(node_index)

                if gts_states_path[col] == row:
                    node_colors.append(_default_color_path)
                else:
                    node_colors.append(_default_color_node)

                node_edges.append(_color_black)
                node_labels[node_index] = f'{round(gts_matrix[row, col], 2):.2f}'
                node_positions[node_index] = (col + 1.0, row_offset)

                node_index += 1

        edge_colors = []

        for row in range(n):

            row_offset = row * f

            for col in range(f - 1):

                if col == 0 and not gts_initial_distribution[col] > 0.0:
                    continue

                for row_next in range(n):

                    if hmm.p[row][row_next] > 0.0:

                        if gts_forward:
                            trellis.add_edge(row_offset + col, (row_next * f) + col + 1)
                            on_path = gts_states_path[col] == row and gts_states_path[col + 1] == row_next
                        else:
                            trellis.add_edge((row_next * f) + col + 1, row_offset + col)
                            on_path = gts_states_path[col] == row_next and gts_states_path[col + 1] == row

                        if on_path:
                            edge_colors.append(_default_color_path)
                        else:
                            edge_colors.append(_color_black)

        for row in range(n):

            trellis.add_node(node_index)

            node_colors.append('none')
            node_edges.append('none')
            node_labels[node_index] = hmm.states[row]
            node_positions[node_index] = (0.6, float(n - row))

            node_index += 1

        headers = ["$\mathregular{T_0}$"] + [hmm.symbols[symbol] for symbol in gts_symbols[1:]]

        for col, header in enumerate(headers):

            trellis.add_node(node_index)

            node_colors.append('none')
            node_edges.append('none')
            node_labels[node_index] = header
            node_positions[node_index] = (col + 1.0, n + 0.35)

            node_index += 1

        return trellis, node_colors, node_edges, node_labels, node_positions, edge_colors

    def _plot_extended(ps_hmm, ps_initial_distribution, ps_symbols, ps_backward, ps_forward, ps_states_path, ps_dpi):

        mpi = _mplp.isinteractive()
        _mplp.interactive(False)

        f, a = _mplp.subplots(nrows=2, tight_layout=True, dpi=ps_dpi)
        a = list(a.flat)

        trellis = _generate_trellis_extended(ps_hmm, ps_initial_distribution, ps_symbols, False, ps_backward, ps_states_path)
        img, _, _, _, _ = _decode_image(trellis, ps_dpi)

        ax_current = a[0]
        ax_current.imshow(img)
        ax_current.axis('off')
        ax_current.set_title('Backward Trellis', fontsize=15.0, fontweight='normal', pad=1)

        trellis = _generate_trellis_extended(ps_hmm, ps_initial_distribution, ps_symbols, True, ps_forward, ps_states_path)
        img, _, _, _, _ = _decode_image(trellis, ps_dpi)

        ax_current = a[1]
        ax_current.imshow(img)
        ax_current.axis('off')
        ax_current.set_title('Forward Trellis', fontsize=15.0, fontweight='normal', pad=1)

        _mplp.interactive(mpi)

        return f, a

    def _plot_standard(ps_hmm, ps_initial_distribution, ps_symbols, ps_backward, ps_forward, ps_states_path, ps_dpi):

        y_top = ps_hmm.n + 0.5

        mpi = _mplp.isinteractive()
        _mplp.interactive(False)

        f, a = _mplp.subplots(nrows=2, tight_layout=True, dpi=ps_dpi)
        a = list(a.flat)

        trellis, node_colors, node_edges, node_labels, node_positions, edge_colors = _generate_trellis_standard(ps_hmm, ps_initial_distribution, ps_symbols, False, ps_backward, ps_states_path)

        ax_current = a[0]
        _nx.draw_networkx(trellis, node_positions, ax=ax_current, edgecolors=node_edges, edge_color=edge_colors, font_size=9, labels=node_labels, node_color=node_colors, node_size=_default_node_size)
        ax_current.set_ylim(0.5, y_top)
        ax_current.axis('off')
        ax_current.set_title('Backward Trellis', fontsize=15.0, fontweight='normal', pad=1)

        trellis, node_colors, node_edges, node_labels, node_positions, edge_colors = _generate_trellis_standard(ps_hmm, ps_initial_distribution, ps_symbols, True, ps_forward, ps_states_path)

        ax_current = a[1]
        _nx.draw_networkx(trellis, node_positions, ax=ax_current, edgecolors=node_edges, edge_color=edge_colors, font_size=9, labels=node_labels, node_color=node_colors, node_size=_default_node_size)
        ax_current.set_ylim(0.5, y_top)
        ax_current.axis('off')
        ax_current.set_title('Forward Trellis', fontsize=15.0, fontweight='normal', pad=1)

        _mplp.interactive(mpi)

        return f, a

    try:

        hmm = _validate_hidden_markov_model(hmm)
        steps = _validate_integer(steps, lower_limit=(2, False))
        initial_state = None if initial_state is None else _validate_label(initial_state, hmm.states)
        force_standard = _validate_boolean(force_standard)
        dpi = _validate_dpi(dpi)

    except Exception as ex:  # pragma: no cover
        raise _create_validation_error(ex, _ins.trace()) from None

    if initial_state is None:
        initial_distribution = _np.full(hmm.n, 1.0 / hmm.n, dtype=float)
    else:
        initial_distribution = _np.zeros(hmm.n, dtype=float)
        initial_distribution[initial_state] = 1.0

    _, symbols = hmm.simulate(steps, initial_state=initial_state, output_indices=True, seed=seed)

    decoding = hmm.decode(symbols, initial_status=initial_distribution)

    if decoding is None:  # pragma: no cover
        raise ValueError('The computation of backward and forward probabilities failed.')

    _, _, backward, forward, _ = decoding

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
        figure, ax = _plot_extended(hmm, initial_distribution, symbols, backward, forward, states_path, dpi)
    else:
        figure, ax = _plot_standard(hmm, initial_distribution, symbols, backward, forward, states_path, dpi)

    if _mplp.isinteractive():  # pragma: no cover
        _mplp.show(block=False)
        return None

    return figure, ax
