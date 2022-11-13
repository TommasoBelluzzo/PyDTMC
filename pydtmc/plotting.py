# -*- coding: utf-8 -*-

__all__ = [
    'plot_comparison',
    'plot_eigenvalues',
    'plot_graph',
    'plot_redistributions',
    'plot_sequence'
]


###########
# IMPORTS #
###########

# Standard

from copy import (
    deepcopy as _cp_deepcopy
)

from inspect import (
    trace as _ins_trace
)

from io import (
    BytesIO as _io_BytesIO
)

from math import (
    ceil as _math_ceil,
    floor as _math_floor,
    log10 as _math_log10,
    sqrt as _math_sqrt
)

# noinspection PyPep8Naming
from subprocess import (
    call as _sp_call,
    PIPE as _sp_pipe
)

# Libraries

from matplotlib.colorbar import (
    make_axes as _mplcb_make_axes
)

from matplotlib.colors import (
    LinearSegmentedColormap as _mplcr_LinearSegmentedColormap
)

from matplotlib.image import (
    imread as _mpli_imread
)

from matplotlib.pyplot import (
    interactive as _mplp_interactive,
    isinteractive as _mplp_isinteractive,
    figure as _mplp_figure,
    Rectangle as _mplp_Rectangle,
    show as _mplp_show,
    subplots as _mplp_subplots,
    subplots_adjust as _mplp_subplots_adjust
)

from matplotlib.ticker import (
    FormatStrFormatter as _mplt_FormatStrFormatter
)

from networkx import (
    draw_networkx_edge_labels as _nx_draw_networkx_edge_labels,
    draw_networkx_edges as _nx_draw_networkx_edges,
    draw_networkx_labels as _nx_draw_networkx_labels,
    draw_networkx_nodes as _nx_draw_networkx_nodes,
    multipartite_layout as _nx_multipartite_layout,
    spring_layout as _nx_spring_layout
)

from numpy import (
    abs as _np_abs,
    all as _np_all,
    allclose as _np_allclose,
    append as _np_append,
    arange as _np_arange,
    arctan2 as _np_arctan2,
    array as _np_array,
    array_equal as _np_array_equal,
    dot as _np_dot,
    cos as _np_cos,
    imag as _np_imag,
    integer as _np_integer,
    isclose as _np_isclose,
    linspace as _np_linspace,
    meshgrid as _np_meshgrid,
    min as _np_min,
    ndarray as _np_ndarray,
    ones as _np_ones,
    pi as _np_pi,
    real as _np_real,
    sin as _np_sin,
    sort as _np_sort,
    sum as _np_sum,
    transpose as _np_transpose,
    unique as _np_unique,
    zeros as _np_zeros
)

from numpy.linalg import (
    eigvals as _npl_eigvals
)

try:
    from pydot import (
        Dot as _pyd_Dot,
        Edge as _pyd_Edge,
        Node as _pyd_Node,
        Subgraph as _pyd_Subgraph
    )
    _pydot_found = True
except ImportError:  # pragma: no cover
    _pyd_Dot, _pyd_Edge, _pyd_Node, _pyd_Subgraph = None, None, None, None
    _pydot_found = False

# Internal

from .custom_types import (
    oint as _oint,
    olist_str as _olist_str,
    oplot as _oplot,
    ostate as _ostate,
    ostatus as _ostatus,
    tdists_flex as _tdists_flex,
    tlist_mc as _tlist_mc,
    tmc as _tmc,
    tobject as _tobject,
    tsequence_flex as _tsequence_flex
)

from .utilities import (
    create_validation_error as _create_validation_error
)

from .validation import (
    validate_boolean as _validate_boolean,
    validate_distribution as _validate_distribution,
    validate_dpi as _validate_dpi,
    validate_enumerator as _validate_enumerator,
    validate_integer as _validate_integer,
    validate_markov_chain as _validate_markov_chain,
    validate_markov_chains as _validate_markov_chains,
    validate_object as _validate_object,
    validate_sequence as _validate_sequence,
    validate_state as _validate_state,
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
_default_color_symbol = _color_gray
_node_size = 500


#############
# FUNCTIONS #
#############

def _xticks_states(ax, mc: _tmc, label: bool, minor_major: bool):

    if label:
        ax.set_xlabel('States', fontsize=13.0)

    if minor_major:
        ax.set_xticks(_np_arange(0.0, mc.size, 1.0), minor=False)
        ax.set_xticks(_np_arange(-0.5, mc.size, 1.0), minor=True)
    else:
        ax.set_xticks(_np_arange(0.0, mc.size, 1.0))

    ax.set_xticklabels(mc.states)


def _xticks_steps(ax, length: int):

    ax.set_xlabel('Steps', fontsize=13.0)
    ax.set_xticks(_np_arange(0.0, length + 1.0, 1.0 if length <= 11 else 10.0), minor=False)
    ax.set_xticks(_np_arange(-0.5, length, 1.0), minor=True)
    ax.set_xticklabels(_np_arange(0, length + 1, 1 if length <= 11 else 10))
    ax.set_xlim(-0.5, length - 0.5)


def _yticks_frequency(ax, bottom: float, top: float):

    ax.set_ylabel('Frequency', fontsize=13.0)
    ax.set_yticks(_np_linspace(0.0, 1.0, 11))
    ax.set_ylim(bottom, top)


def _yticks_states(ax, mc: _tmc, label: bool):

    if label:
        ax.set_ylabel('States', fontsize=13.0)

    ax.set_yticks(_np_arange(0.0, mc.size, 1.0), minor=False)
    ax.set_yticks(_np_arange(-0.5, mc.size, 1.0), minor=True)
    ax.set_yticklabels(mc.states)


def plot_comparison(mcs: _tlist_mc, mcs_names: _olist_str = None, constrained_layout: bool = False, dark_colormap: bool = False, dpi: int = 100) -> _oplot:

    """
    The function plots the transition matrix of every Markov chain in the form of a heatmap.

    | **Notes:**

    * If `Matplotlib <https://matplotlib.org/>`_ is in `interactive mode <https://matplotlib.org/stable/users/interactive.html>`_, the plot is immediately displayed and the function does not return the plot handles.

    :param mcs: the Markov chains.
    :param mcs_names: the name of each Markov chain subplot (*if omitted, a standard name is given to each subplot*).
    :param constrained_layout: a boolean indicating whether to use a constrained layout.
    :param dark_colormap: a boolean indicating whether to use a dark colormap instead of the default one.
    :param dpi: the resolution of the plot expressed in dots per inch.
    :raises ValidationError: if any input argument is not compliant.
    """

    try:

        mcs = _validate_markov_chains(mcs)
        mcs_names = [f'MC{index + 1} Size={mc.size:d}' for index, mc in enumerate(mcs)] if mcs_names is None else _validate_strings(mcs_names, len(mcs))
        constrained_layout = _validate_boolean(constrained_layout)
        dark_colormap = _validate_boolean(dark_colormap)
        dpi = _validate_dpi(dpi)

    except Exception as ex:  # pragma: no cover
        raise _create_validation_error(ex, _ins_trace()) from None

    n = len(mcs)
    rows = int(_math_sqrt(n))
    columns = int(_math_ceil(n / float(rows)))

    figure, axes = _mplp_subplots(rows, columns, constrained_layout=constrained_layout, dpi=dpi)
    axes = list(axes.flat)
    ax_is = None

    if dark_colormap:

        for ax, mc, mc_name in zip(axes, mcs, mcs_names):
            ax_is = ax.imshow(mc.p, aspect='auto', cmap='hot', vmin=0.0, vmax=1.0)
            ax.set_title(mc_name, fontsize=9.0, fontweight='normal', pad=1)
            ax.set_axis_off()

    else:

        color_map = _mplcr_LinearSegmentedColormap.from_list('ColorMap', [_color_white, _colors[0]], 20)

        for ax, mc, mc_name in zip(axes, mcs, mcs_names):
            ax_is = ax.imshow(mc.p, aspect='auto', cmap=color_map, interpolation='none', vmin=0.0, vmax=1.0)
            ax.set_title(mc_name, fontsize=9.0, fontweight='normal', pad=1)
            ax.set_xticks([])
            ax.set_xticks([], minor=True)
            ax.set_yticks([])
            ax.set_yticks([], minor=True)

    figure.suptitle('Comparison Plot', fontsize=15.0, fontweight='bold')

    color_map_ax, color_map_ax_kwargs = _mplcb_make_axes(axes, drawedges=True, orientation='horizontal', ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    figure.colorbar(ax_is, cax=color_map_ax, **color_map_ax_kwargs)
    color_map_ax.set_xticklabels([0.0, 0.25, 0.5, 0.75, 1.0])

    if _mplp_isinteractive():  # pragma: no cover
        _mplp_show(block=False)
        return None

    return figure, axes


def plot_eigenvalues(mc: _tmc, dpi: int = 100) -> _oplot:

    """
    The function plots the eigenvalues of the Markov chain on the complex plane.

    | **Notes:**

    * If `Matplotlib <https://matplotlib.org/>`_ is in `interactive mode <https://matplotlib.org/stable/users/interactive.html>`_, the plot is immediately displayed and the function does not return the plot handles.

    :param mc: the Markov chain.
    :param dpi: the resolution of the plot expressed in dots per inch.
    :raises ValidationError: if any input argument is not compliant.
    """

    try:

        mc = _validate_markov_chain(mc)
        dpi = _validate_dpi(dpi)

    except Exception as ex:  # pragma: no cover
        raise _create_validation_error(ex, _ins_trace()) from None

    figure, ax = _mplp_subplots(dpi=dpi)

    handles, labels = [], []

    theta = _np_linspace(0.0, 2.0 * _np_pi, 200)

    evalues = _npl_eigvals(mc.p).astype(complex)
    evalues_final = _np_unique(_np_append(evalues, _np_array([1.0]).astype(complex)))

    x_unit_circle = _np_cos(theta)
    y_unit_circle = _np_sin(theta)

    if mc.is_ergodic:

        values_abs = _np_sort(_np_abs(evalues))
        values_ct1 = _np_isclose(values_abs, 1.0)

        if not _np_all(values_ct1):

            mu = values_abs[~values_ct1][-1]

            if not _np_isclose(mu, 0.0):

                x_slem_circle = mu * x_unit_circle
                y_slem_circle = mu * y_unit_circle

                cs = _np_linspace(-1.1, 1.1, 201)
                x_spectral_gap, y_spectral_gap = _np_meshgrid(cs, cs)
                z_spectral_gap = x_spectral_gap**2.0 + y_spectral_gap**2.0

                h = ax.contourf(x_spectral_gap, y_spectral_gap, z_spectral_gap, alpha=0.2, colors='r', levels=[mu**2.0, 1.0])
                handles.append(_mplp_Rectangle((0.0, 0.0), 1.0, 1.0, fc=h.collections[0].get_facecolor()[0]))
                labels.append('Spectral Gap')

                ax.plot(x_slem_circle, y_slem_circle, color='red', linestyle='--', linewidth=1.5)

    ax.plot(x_unit_circle, y_unit_circle, color='red', linestyle='-', linewidth=3.0)

    h, = ax.plot(_np_real(evalues_final), _np_imag(evalues_final), color='blue', linestyle='None', marker='*', markersize=12.5)
    handles.append(h)
    labels.append('Eigenvalues')

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')

    formatter = _mplt_FormatStrFormatter('%g')
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.set_xticks(_np_linspace(-1.0, 1.0, 9))
    ax.set_yticks(_np_linspace(-1.0, 1.0, 9))

    ax.grid(which='major')

    ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=len(handles))
    ax.set_title('Eigenvalues Plot', fontsize=15.0, fontweight='bold')

    _mplp_subplots_adjust(bottom=0.2)

    if _mplp_isinteractive():  # pragma: no cover
        _mplp_show(block=False)
        return None

    return figure, ax


# noinspection PyBroadException
def plot_graph(obj: _tobject, nodes_color: bool = True, nodes_shape: bool = True, edges_label: bool = True, force_standard: bool = False, dpi: int = 100) -> _oplot:

    """
    The function plots the directed graph of the given object, which can be either a Markov chain or a hidden Markov model.

    | **Notes:**

    * If `Matplotlib <https://matplotlib.org/>`_ is in `interactive mode <https://matplotlib.org/stable/users/interactive.html>`_, the plot is immediately displayed and the function does not return the plot handles.
    * `Graphviz <https://graphviz.org/>`_ and `pydot <https://pypi.org/project/pydot/>`_ are not required, but they provide access to extended mode with improved rendering and additional features.
    * The rendering, especially in standard mode or for big graphs, is not granted to be high-quality.
    * For Markov chains, the color of nodes is based on communicating classes; for hidden Markov models, every state node has a different color and symbol nodes are gray.
    * For Markov chains, recurrent nodes have an elliptical shape and transient nodes have a rectangular shape; for hidden Markov models, state nodes have an elliptical shape and symbol nodes have a hexagonal shape.

    :param obj: the object to be converted into a graph.
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
            element_minimum = _np_min(element).item()
            element_magnitude = 0 if element_minimum == 0.0 else int(-_math_floor(_math_log10(abs(element_minimum))))
            magnitudes.append(element_magnitude)

        magnitude = max(1, min(max(magnitudes), 4))

        return magnitude

    def _decode_image(di_g, di_dpi):

        buffer = _io_BytesIO()
        buffer.write(di_g.create(format='png'))
        buffer.seek(0)

        img = _mpli_imread(buffer)
        img_x = img.shape[0] / di_dpi
        img_xi = img_x * 1.1
        img_xo = ((img_xi - img_x) / 2.0) * di_dpi
        img_y = img.shape[1] / di_dpi
        img_yi = img_y * 1.1
        img_yo = ((img_yi - img_y) / 2.0) * di_dpi

        return img, img_x, img_xo, img_y, img_yo

    def _draw_edge_labels_curved(delc_ax, delc_positions, delc_edge_labels):

        for (n1, n2), (rad, label) in delc_edge_labels.items():

            (x1, y1) = delc_positions[n1]
            (x2, y2) = delc_positions[n2]
            p1 = delc_ax.transData.transform(_np_array(delc_positions[n1]))
            p2 = delc_ax.transData.transform(_np_array(delc_positions[n2]))

            linear_mid = (0.5 * p1) + (0.5 * p2)
            cp_mid = linear_mid + (rad * _np_dot(_np_array([(0, 1), (-1, 0)]), p2 - p1))
            cp1 = (0.5 * p1) + (0.5 * cp_mid)
            cp2 = (0.5 * p2) + (0.5 * cp_mid)
            bezier_mid = (0.5 * cp1) + (0.5 * cp2)

            (x, y) = delc_ax.transData.inverted().transform(bezier_mid)
            xy = _np_array((x, y))

            angle = (_np_arctan2(y2 - y1, x2 - x1) / (2.0 * _np_pi)) * 360.0

            if angle > 90.0:
                angle -= 180.0

            if angle < -90.0:
                angle += 180.0

            rotation = delc_ax.transData.transform_angles(_np_array((angle,)), xy.reshape((1, 2)))[0]
            transform = delc_ax.transData
            bbox = dict(boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0))

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

        colors = _cp_deepcopy(_colors)
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
        edge_colors = _cp_deepcopy(node_colors) if phe_nodes_color else []

        g = _pyd_Dot(graph_type='digraph')

        g_sub1 = _pyd_Subgraph(rank='same')
        g.add_subgraph(g_sub1)

        g_sub2 = _pyd_Subgraph(rank='same')
        g.add_subgraph(g_sub2)

        for i in range(phe_hmm.n):

            state = phe_hmm.states[i]

            node_attributes = {}

            if phe_nodes_color:
                node_attributes['style'] = 'filled'
                node_attributes['fillcolor'] = node_colors[i]

            if phe_nodes_type:
                node_attributes['shape'] = 'ellipse'

            g_sub1.add_node(_pyd_Node(state, **node_attributes))

        for symbol in phe_hmm.symbols:

            node_attributes = {}

            if phe_nodes_color:
                node_attributes['style'] = 'filled'
                node_attributes['fillcolor'] = _default_color_symbol

            if phe_nodes_type:
                node_attributes['shape'] = 'hexagon'

            g_sub2.add_node(_pyd_Node(symbol, **node_attributes))

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

                    if i == j:
                        edge_attributes['headport'] = 'n'
                        edge_attributes['tailport'] = 'n'

                    g.add_edge(_pyd_Edge(state_i, state_j, **edge_attributes))

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

                    g.add_edge(_pyd_Edge(state_i, symbol, **edge_attributes))

        img, img_x, img_xo, img_y, img_yo = _decode_image(g, phe_dpi)

        f = _mplp_figure(figsize=(img_y * 1.1, img_x * 1.1), dpi=phe_dpi)
        f.figimage(img, yo=img_yo, xo=img_xo)

        a = f.gca()
        a.axis('off')

        return f, a

    def _plot_hmm_standard(phs_hmm, phs_nodes_color, phs_nodes_shape, phe_edges_label, phs_dpi):

        g = phs_hmm.to_graph()
        positions = _nx_multipartite_layout(g, align='horizontal', subset_key='layer')
        magnitude = _calculate_magnitude(phs_hmm.p, phs_hmm.e)

        node_colors = _node_colors(phs_hmm.n) if phs_nodes_color else []
        edge_colors = _cp_deepcopy(node_colors) if phs_nodes_color else []

        mpi = _mplp_isinteractive()
        _mplp_interactive(False)

        f, a = _mplp_subplots(dpi=phs_dpi)

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

            _nx_draw_networkx_nodes(g, positions, ax=a, nodelist=[node], node_color=node_color, node_shape=node_shape, node_size=_node_size, edgecolors='k')

        _nx_draw_networkx_labels(g, positions, ax=a)

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

                    _nx_draw_networkx_edges(g, positions, ax=a, edgelist=[edge], edge_color=edge_color, arrows=True, connectionstyle=edge_connection)

            for j in range(phs_hmm.k):

                ep = phs_hmm.e[i, j]

                if ep > 0.0:

                    symbol_j = phs_hmm.symbols[j]

                    edge = (state_i, symbol_j)
                    edge_color = edge_colors[i] if phs_nodes_color else _default_color_edge

                    if phe_edges_label:
                        edge_labels_straight_symbol[edge] = f' {round(ep, magnitude):.{magnitude}f} '

                    _nx_draw_networkx_edges(g, positions, ax=a, edgelist=[edge], edge_color=edge_color, arrows=True, style='dashed')

        if len(edge_labels_straight_state) > 0:
            _nx_draw_networkx_edge_labels(g, positions, ax=a, edge_labels=edge_labels_straight_state)

        if len(edge_labels_straight_symbol) > 0:
            _nx_draw_networkx_edge_labels(g, positions, ax=a, edge_labels=edge_labels_straight_symbol, label_pos=0.25)

        if len(edge_labels_curved) > 0:
            _draw_edge_labels_curved(a, positions, edge_labels_curved)

        _mplp_interactive(mpi)

        return f, a

    # noinspection DuplicatedCode
    def _plot_mc_extended(pme_mc, pme_nodes_color, pme_nodes_shape, phe_edges_label, pme_dpi):

        magnitude = _calculate_magnitude(pme_mc.p)

        node_colors = _node_colors(len(pme_mc.communicating_classes)) if pme_nodes_color else []

        g = _pyd_Dot(graph_type='digraph')

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

            g.add_node(_pyd_Node(state_i, **node_attributes))

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

                    g.add_edge(_pyd_Edge(state_i, state_j, **edge_attributes))

        img, img_x, img_xo, img_y, img_yo = _decode_image(g, pme_dpi)

        f = _mplp_figure(figsize=(img_y * 1.1, img_x * 1.1), dpi=pme_dpi)
        f.figimage(img, yo=img_yo, xo=img_xo)

        a = f.gca()
        a.axis('off')

        return f, a

    def _plot_mc_standard(pms_mc, pms_nodes_color, pms_nodes_shape, phe_edges_label, pms_dpi):

        g = pms_mc.to_graph()
        positions = _nx_spring_layout(g)
        magnitude = _calculate_magnitude(pms_mc.p)

        node_colors = _node_colors(len(pms_mc.communicating_classes)) if pms_nodes_color else []

        mpi = _mplp_isinteractive()
        _mplp_interactive(False)

        f, a = _mplp_subplots(dpi=pms_dpi)

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

            _nx_draw_networkx_nodes(g, positions, ax=a, nodelist=[node], node_color=node_color, node_shape=node_shape, node_size=_node_size, edgecolors='k')

        _nx_draw_networkx_labels(g, positions, ax=a)

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

                    _nx_draw_networkx_edges(g, positions, ax=a, edgelist=[edge], edge_color=edge_color, arrows=True, connectionstyle=edge_connection)

        if len(edge_labels_straight) > 0:
            _nx_draw_networkx_edge_labels(g, positions, ax=a, edge_labels=edge_labels_straight)

        if len(edge_labels_curved) > 0:
            _draw_edge_labels_curved(a, positions, edge_labels_curved)

        _mplp_interactive(mpi)

        return f, a

    try:

        obj = _validate_object(obj)
        nodes_color = _validate_boolean(nodes_color)
        nodes_shape = _validate_boolean(nodes_shape)
        edges_label = _validate_boolean(edges_label)
        force_standard = _validate_boolean(force_standard)
        dpi = _validate_dpi(dpi)

    except Exception as ex:  # pragma: no cover
        raise _create_validation_error(ex, _ins_trace()) from None

    extended_graph = not force_standard and _pydot_found

    if extended_graph:
        try:
            _sp_call(['dot', '-V'], stdout=_sp_pipe, stderr=_sp_pipe)
        except Exception:  # pragma: no cover
            extended_graph = False

    obj_mc = obj.__class__.__name__ == 'MarkovChain'

    if extended_graph:
        func = _plot_mc_extended if obj_mc else _plot_hmm_extended
    else:
        func = _plot_mc_standard if obj_mc else _plot_hmm_standard

    figure, ax = func(obj, nodes_color, nodes_shape, edges_label, dpi)

    if _mplp_isinteractive():  # pragma: no cover
        _mplp_show(block=False)
        return None

    return figure, ax


# noinspection DuplicatedCode
def plot_redistributions(mc: _tmc, distributions: _tdists_flex, initial_status: _ostatus = None, plot_type: str = 'projection', dpi: int = 100) -> _oplot:

    """
    The function plots a redistribution of states on the given Markov chain.

    | **Notes:**

    * If `Matplotlib <https://matplotlib.org/>`_ is in `interactive mode <https://matplotlib.org/stable/users/interactive.html>`_, the plot is immediately displayed and the function does not return the plot handles.

    :param mc: the Markov chain.
    :param distributions: a sequence of redistributions or the number of redistributions to perform.
    :param initial_status: the initial state or the initial distribution of the states (*if omitted, the states are assumed to be uniformly distributed*).
    :param plot_type:
     - **heatmap** for displaying a heatmap plot;
     - **projection** for displaying a projection plot.
    :param dpi: the resolution of the plot expressed in dots per inch.
    :raises ValidationError: if any input argument is not compliant.
    :raises ValueError: if the "distributions" parameter represents a sequence of redistributions and the "initial_status" parameter does not match its first element.
    """

    try:

        mc = _validate_markov_chain(mc)
        distributions = _validate_distribution(distributions, mc.size)
        initial_status = None if initial_status is None else _validate_status(initial_status, mc.states)
        plot_type = _validate_enumerator(plot_type, ['heatmap', 'projection'])
        dpi = _validate_dpi(dpi)

    except Exception as ex:  # pragma: no cover
        raise _create_validation_error(ex, _ins_trace()) from None

    if isinstance(distributions, int):
        distributions = mc.redistribute(distributions, initial_status=initial_status, output_last=False)

    if initial_status is not None and not _np_array_equal(distributions[0], initial_status):  # pragma: no cover
        raise ValueError('The "initial_status" parameter, if specified when the "distributions" parameter represents a sequence of redistributions, must match the first element.')

    distributions_length = 1 if isinstance(distributions, _np_ndarray) else len(distributions)
    distributions = _np_array([distributions]) if isinstance(distributions, _np_ndarray) else _np_array(distributions)

    figure, ax = _mplp_subplots(dpi=dpi)

    if plot_type == 'heatmap':

        color_map = _mplcr_LinearSegmentedColormap.from_list('ColorMap', [_color_white, _colors[0]], 20)
        ax_is = ax.imshow(_np_transpose(distributions), aspect='auto', cmap=color_map, interpolation='none', vmin=0.0, vmax=1.0)

        _xticks_steps(ax, distributions_length)
        _yticks_states(ax, mc, False)

        ax.grid(which='minor', color='k')

        cb = figure.colorbar(ax_is, drawedges=True, orientation='horizontal', ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
        cb.ax.set_xticklabels([0.0, 0.25, 0.5, 0.75, 1.0])

        ax.set_title('Redistributions Plot (Heatmap)', fontsize=15.0, fontweight='bold')

    else:

        ax.set_prop_cycle('color', _colors)

        if distributions_length == 2:
            for i in range(mc.size):
                ax.plot(_np_arange(0.0, distributions_length, 1.0), distributions[:, i], label=mc.states[i], marker='o')
        else:
            for i in range(mc.size):
                ax.plot(_np_arange(0.0, distributions_length, 1.0), distributions[:, i], label=mc.states[i])

        if _np_allclose(distributions[0, :], _np_ones(mc.size, dtype=float) / mc.size):
            ax.plot(0.0, distributions[0, 0], color=_color_black, label="Start", marker='o', markeredgecolor=_color_black, markerfacecolor=_color_black)
            legend_size = mc.size + 1
        else:  # pragma: no cover
            legend_size = mc.size

        _xticks_steps(ax, distributions_length)
        _yticks_frequency(ax, -0.05, 1.05)

        ax.grid()

        ax.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=legend_size)
        ax.set_title('Redistributions Plot (Projection)', fontsize=15.0, fontweight='bold')

        _mplp_subplots_adjust(bottom=0.2)

    if _mplp_isinteractive():  # pragma: no cover
        _mplp_show(block=False)
        return None

    return figure, ax


# noinspection DuplicatedCode
def plot_sequence(mc: _tmc, sequence: _tsequence_flex, initial_state: _ostate = None, plot_type: str = 'histogram', seed: _oint = None, dpi: int = 100) -> _oplot:

    """
    The function plots a random walk on the given Markov chain.

    | **Notes:**

    * If `Matplotlib <https://matplotlib.org/>`_ is in `interactive mode <https://matplotlib.org/stable/users/interactive.html>`_, the plot is immediately displayed and the function does not return the plot handles.

    :param mc: the Markov chain.
    :param sequence: a sequence of states or the number of simulations to perform.
    :param initial_state: the initial state of the sequence (*if omitted, it is chosen uniformly at random*).
    :param plot_type:
     - **histogram** for displaying an histogram plot;
     - **matrix** for displaying a matrix plot;
     - **transitions** for displaying a transitions plot.
    :param seed: a seed to be used as RNG initializer for reproducibility purposes.
    :param dpi: the resolution of the plot expressed in dots per inch.
    :raises ValidationError: if any input argument is not compliant.
    :raises ValueError: if the "sequence" parameter represents a sequence of states and the "initial_state" parameter does not match its first element.
    """

    try:

        mc = _validate_markov_chain(mc)

        if isinstance(sequence, (int, _np_integer)):
            sequence = _validate_integer(sequence, lower_limit=(2, False))
        else:
            sequence = _validate_sequence(sequence, mc.states)

        if initial_state is not None:
            initial_state = _validate_state(initial_state, mc.states)

        plot_type = _validate_enumerator(plot_type, ['histogram', 'matrix', 'transitions'])
        dpi = _validate_dpi(dpi)

    except Exception as ex:  # pragma: no cover
        raise _create_validation_error(ex, _ins_trace()) from None

    if isinstance(sequence, int):
        sequence = mc.simulate(sequence, initial_state=initial_state, output_indices=True, seed=seed)

    if initial_state is not None and sequence[0] != initial_state:  # pragma: no cover
        raise ValueError('The "initial_state" parameter, if specified when the "sequence" parameter represents a sequence of states, must match the first element.')

    sequence_length = len(sequence)

    figure, ax = _mplp_subplots(dpi=dpi)

    if plot_type == 'histogram':

        sequence_histogram = _np_zeros((mc.size, sequence_length), dtype=float)

        for index, state in enumerate(sequence):
            sequence_histogram[state, index] = 1.0

        sequence_histogram = _np_sum(sequence_histogram, axis=1) / _np_sum(sequence_histogram)

        ax.bar(_np_arange(0.0, mc.size, 1.0), sequence_histogram, edgecolor=_color_black, facecolor=_colors[0])

        _xticks_states(ax, mc, True, False)
        _yticks_frequency(ax, 0.0, 1.0)

        ax.set_title('Sequence Plot (Histogram)', fontsize=15.0, fontweight='bold')

    elif plot_type == 'matrix':

        sequence_matrix = _np_zeros((mc.size, sequence_length), dtype=float)

        for index, state in enumerate(sequence):
            sequence_matrix[state, index] = 1.0

        color_map = _mplcr_LinearSegmentedColormap.from_list('ColorMap', [_color_white, _colors[0]], 2)
        ax.imshow(sequence_matrix, aspect='auto', cmap=color_map, interpolation='none', vmin=0.0, vmax=1.0)

        _xticks_steps(ax, sequence_length)
        _yticks_states(ax, mc, True)

        ax.grid(which='minor', color='k')

        ax.set_title('Sequence Plot (Matrix)', fontsize=15.0, fontweight='bold')

    else:

        sequence_matrix = _np_zeros((mc.size, mc.size), dtype=float)

        for i in range(1, sequence_length):
            sequence_matrix[sequence[i - 1], sequence[i]] += 1.0

        sequence_matrix /= _np_sum(sequence_matrix)

        color_map = _mplcr_LinearSegmentedColormap.from_list('ColorMap', [_color_white, _colors[0]], 20)
        ax_is = ax.imshow(sequence_matrix, aspect='auto', cmap=color_map, interpolation='none', vmin=0.0, vmax=1.0)

        _xticks_states(ax, mc, False, True)
        _yticks_states(ax, mc, False)

        ax.grid(which='minor', color='k')

        cb = figure.colorbar(ax_is, drawedges=True, orientation='horizontal', ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
        cb.ax.set_xticklabels([0.0, 0.25, 0.5, 0.75, 1.0])

        ax.set_title('Sequence Plot (Transitions)', fontsize=15.0, fontweight='bold')

    if _mplp_isinteractive():  # pragma: no cover
        _mplp_show(block=False)
        return None

    return figure, ax
