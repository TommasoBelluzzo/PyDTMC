# -*- coding: utf-8 -*-

from __future__ import annotations

__all__ = [
    'plot_comparison',
    'plot_convergence',
    'plot_decoding',
    'plot_flow',
    'plot_graph',
    'plot_implied_timescales',
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

import io as _io
import math as _mt
import subprocess as _sub

# Libraries

import matplotlib.colors as _mplcr
import matplotlib.image as _mpli
import matplotlib.patches as _mplpc
import matplotlib.pyplot as _mplp
import matplotlib.ticker as _mplt
import matplotlib.transforms as _mpltr
import networkx as _nx
import numpy as _np
import numpy.linalg as _npl
import scipy.special as _sps

try:
    import pydot as _pyd
    _pydot_found = True
except ImportError:  # pragma: no cover
    _pyd = None
    _pydot_found = False

# Internal

from . import _custom_types as _ct
from . import _numerical as _n

from .markov_chain import (
    MarkovChain as _MarkovChain
)


#############
# CONSTANTS #
#############

_COLOR_BLACK = '#000000'
_COLOR_GRAY = '#E0E0E0'
_COLOR_WHITE = '#FFFFFF'
_COLORS = ('#80B1D3', '#FFED6F', '#B3DE69', '#BEBADA', '#FDB462', '#8DD3C7', '#FB8072', '#FCCDE5', '#E5C494')

_DEFAULT_COLOR_DATA = _COLORS[0]
_DEFAULT_COLOR_REFERENCE = _COLORS[4]
_DEFAULT_COLOR_NEGATIVE = _COLORS[6]
_DEFAULT_COLOR_POSITIVE = _COLORS[0]
_DEFAULT_COLOR_EDGE = _COLOR_BLACK
_DEFAULT_COLOR_NODE = _COLOR_WHITE
_DEFAULT_COLOR_PATH = _DEFAULT_COLOR_DATA
_DEFAULT_COLOR_SYMBOL = _COLOR_GRAY
_DEFAULT_COLOR_GRID = _mplp.rcParams['grid.color']

_COLORBAR_ASPECT = 20
_COLORBAR_HEIGHT = 0.18
_COLORBAR_TICKS_PROBABILITY = (0.0, 0.25, 0.5, 0.75, 1.0)
_COLORBAR_TICKS_SIGNED = (-1.0, -0.5, 0.0, 0.5, 1.0)

_FIGURE_WIDTH, _FIGURE_HEIGHT = tuple(_mplp.rcParams['figure.figsize'])

_FONT_SIZE_AXES_LABEL = 12.0
_FONT_SIZE_AXES_TITLE = 13.0
_FONT_SIZE_FIGURE_TITLE = 15.0
_FONT_SIZE_GRAPH_LABEL = 10.0
_FONT_SIZE_EDGE_LABEL = 9.0
_FONT_SIZE_LEGEND = _mplp.rcParams['legend.fontsize']

_FONT_WEIGHT_AXES_LABEL = 'normal'
_FONT_WEIGHT_AXES_TITLE = 'normal'
_FONT_WEIGHT_FIGURE_TITLE = 'bold'

_GRAPHVIZ_OVERSAMPLING = 2

_GRID_ALPHA = 0.5
_GRID_LINE_WIDTH = 0.5

_LEGEND_BOTTOM = 0.02
_LEGEND_MARGIN = 0.2

_LINE_WIDTH = 1.75
_LINE_WIDTH_REFERENCE = 1.50
_LINE_WIDTH_EMPHASIZED = 3.00

_MARKER_SIZE = 4.0
_MARKER_SCATTER_SIZE = 12.0

_MAXIMUM_TICKS = 11

_NODE_SIZE = 600

_PAD_AXES_TITLE = 6.0
_PAD_LABEL = 4.0
_PAD_LABEL_SIDE = 10.0


#############
# FUNCTIONS #
#############

# noinspection PyTypeChecker
def _decode_graphviz_image(graph: _pyd.Dot, dpi: int, oversampling: int = 1) -> _ct.tarray:

    graph.set('dpi', str(dpi * oversampling))

    try:

        with _io.BytesIO(graph.create(format='png')) as buffer:
            img = _mpli.imread(buffer)

    finally:

        graph.set('dpi', str(dpi))

    if oversampling > 1:

        height, width, channels = img.shape

        padding_height = (-height) % oversampling
        padding_width = (-width) % oversampling

        img = _np.pad(img, ((0, padding_height), (0, padding_width), (0, 0)), mode='edge')

        height, width = img.shape[:2]

        img = img.reshape((
            height // oversampling, oversampling,
            width // oversampling, oversampling,
            channels
        )).mean(axis=(1, 3))

    return img


def _draw_colorbar(figure: _mplp.Figure, image: _ct.tany, axes: _ct.tany, orientation: str, signed: bool = False) -> None:

    locator = None

    def _locate_colorbar(ax, renderer):  # pragma: no cover

        bounds = locator(ax, renderer) if locator is not None else ax.get_position(original=True)

        if bounds.width > 0.0:

            width = bounds.width * figure.get_figwidth()
            height = _COLORBAR_HEIGHT * _mt.sqrt(figure.get_figheight() / _FIGURE_HEIGHT)

            aspect = min(1.0 / _COLORBAR_ASPECT, height / width)
            ax.set_box_aspect(aspect)

        return bounds

    ticks = _COLORBAR_TICKS_SIGNED if signed else _COLORBAR_TICKS_PROBABILITY
    labels = [f'{value:.2f}'.rstrip('0').rstrip('.') if value % 1.0 else f'{value:.1f}' for value in ticks]

    fraction = 0.05 if orientation == 'vertical' else 0.06
    pad = 0.05

    cb = figure.colorbar(image, ax=axes, orientation=orientation, ticks=ticks, drawedges=True, fraction=fraction, pad=pad, aspect=_COLORBAR_ASPECT)
    cb.set_ticklabels(labels)

    if orientation == 'horizontal':
        locator = cb.ax.get_axes_locator()
        cb.ax.set_axes_locator(_locate_colorbar)


def _draw_edge_labels_curved(ax: _mplp.Axes, positions: dict, edge_labels: dict) -> None:

    for (n1, n2), (rad, label) in edge_labels.items():

        (x1, y1) = positions[n1]
        (x2, y2) = positions[n2]
        p1 = ax.transData.transform(_np.array(positions[n1]))
        p2 = ax.transData.transform(_np.array(positions[n2]))

        linear_mid = (0.5 * p1) + (0.5 * p2)

        cp_mid = linear_mid + (rad * _np.dot(_np.array([(0, 1), (-1, 0)]), p2 - p1))
        cp1 = (0.5 * p1) + (0.5 * cp_mid)
        cp2 = (0.5 * p2) + (0.5 * cp_mid)
        bezier_mid = (0.5 * cp1) + (0.5 * cp2)

        (x, y) = ax.transData.inverted().transform(bezier_mid)
        xy = _np.array((x, y))

        angle = (_np.arctan2(y2 - y1, x2 - x1) / (2.0 * _np.pi)) * 360.0

        if angle > 90.0:
            angle -= 180.0

        if angle < -90.0:
            angle += 180.0

        rotation = ax.transData.transform_angles(_np.array((angle,)), xy.reshape((1, 2)))[0]
        transform = ax.transData

        bbox = {
            'boxstyle': 'round',
            'ec': _COLOR_WHITE,
            'fc': _COLOR_WHITE
        }

        ax.text(
            x, y, label,
            color=_DEFAULT_COLOR_EDGE, family='sans-serif', size=_FONT_SIZE_EDGE_LABEL, weight=_FONT_WEIGHT_AXES_LABEL,
            horizontalalignment='center', verticalalignment='center',
            rotation=rotation, bbox=bbox, clip_on=True, transform=transform, zorder=1
        )

    ax.tick_params(
        axis='both', which='both',
        bottom=False, left=False,
        labelbottom=False, labelleft=False
    )


def _draw_graphviz_image(graph: _pyd.Dot, ax: _mplp.Axes, dpi: int, initial_dpi: _ct.oint = None, callbacks: _ct.olist_cal = None, oversampling: int = 1) -> None:

    def _place_image(target_ax, target_artist):

        current_bounds = target_ax.get_window_extent()
        image_height, image_width = target_artist.get_array().shape[:2]

        target_artist.ox = int(_mt.floor(current_bounds.x0 + (current_bounds.width - image_width) / 2.0))
        target_artist.oy = int(_mt.floor(current_bounds.y0 + (current_bounds.height - image_height) / 2.0))

        return current_bounds

    def _refresh_image(target_graph, target_ax, target_artist):  # pragma: no cover

        nonlocal render_dpi, refreshing, displayed_bounds

        if refreshing:
            return False

        current_bounds = target_ax.get_window_extent()
        image_height, image_width = target_artist.get_array().shape[:2]

        if current_bounds.width <= 0.0 or current_bounds.height <= 0.0:
            return False

        target_dpi = max(1, int(_mt.floor(render_dpi * min(current_bounds.width / image_width, current_bounds.height / image_height))))
        current_dimensions = (round(current_bounds.width), round(current_bounds.height))

        if displayed_bounds == current_dimensions:
            return False

        if abs(target_dpi - render_dpi) < 2:
            _place_image(target_ax, target_artist)
            displayed_bounds = current_dimensions
            return False

        refreshing = True

        # noinspection PyBroadException
        try:
            new_image = _decode_graphviz_image(target_graph, target_dpi, oversampling)
        except Exception:  # pragma: no cover
            target_graph.set('dpi', str(render_dpi))
            return False
        finally:
            refreshing = False

        render_dpi = target_dpi
        target_artist.set_data(new_image)
        _place_image(target_ax, target_artist)
        displayed_bounds = current_dimensions

        return True

    def _on_draw(event):  # pragma: no cover

        if event.canvas is not canvas:
            return

        _place_image(ax, artist)

        if not resize_pending:
            current_bounds = ax.get_window_extent()
            if displayed_bounds != (round(current_bounds.width), round(current_bounds.height)):
                resize_timer.stop()
                resize_timer.start()

    def _on_resize(event):  # pragma: no cover

        nonlocal resize_pending

        if event.canvas is not canvas:
            return

        resize_pending = True
        resize_timer.stop()
        resize_timer.start()

    def _on_resize_complete():  # pragma: no cover

        nonlocal resize_pending

        resize_pending = False

        if _refresh_image(graph, ax, artist):
            canvas.draw_idle()

    def _register_callbacks():

        canvas.mpl_connect('draw_event', _on_draw)
        canvas.mpl_connect('resize_event', _on_resize)

    figure = ax.figure
    canvas = figure.canvas
    initial_bounds = ax.get_window_extent()

    render_dpi = max(1, int(dpi)) if initial_dpi is None else max(1, int(initial_dpi))
    image = _decode_graphviz_image(graph, render_dpi, oversampling)

    if (initial_bounds.width > 0.0) and (initial_bounds.height > 0.0):

        initial_scale = min(initial_bounds.width / image.shape[1], initial_bounds.height / image.shape[0])
        initial_target_dpi = max(1, int(_mt.floor(render_dpi * initial_scale)))

        if initial_target_dpi != render_dpi:
            image = _decode_graphviz_image(graph, initial_target_dpi, oversampling)
            render_dpi = initial_target_dpi

    ax.axis('off')

    artist = figure.figimage(image, origin='upper', resize=False)
    refreshing, resize_pending = False, False

    positioned_bounds = _place_image(ax, artist)
    displayed_bounds = (round(positioned_bounds.width), round(positioned_bounds.height))

    resize_timer = canvas.new_timer(interval=120)
    resize_timer.single_shot = True
    resize_timer.add_callback(_on_resize_complete)

    if callbacks is None:
        _register_callbacks()
    else:
        callbacks.append(_register_callbacks)


def _draw_legend_bottom(figure: _mplp.Figure, handles: list, labels: _ct.tlist_str, tight_layout: bool = False) -> float:

    columns = len(handles)

    if columns == 0:  # pragma: no cover
        return _LEGEND_MARGIN

    while True:

        indices = [index for column in range(columns) for index in range(column, len(handles), columns)]
        handles_ordered = [handles[index] for index in indices]
        labels_ordered = [labels[index] for index in indices]

        legend = figure.legend(
            handles_ordered, labels_ordered,
            ncol=columns,
            loc='lower center', bbox_to_anchor=(0.5, _LEGEND_BOTTOM),
            fontsize=_FONT_SIZE_LEGEND
        )

        figure.canvas.draw()
        bounds = legend.get_window_extent()

        if (bounds.width <= (figure.bbox.width * 0.96)) or (columns == 1):
            break

        legend.remove()
        columns -= 1

    width, height = figure.get_size_inches()
    width = max(width, bounds.width / (figure.dpi * 0.96))
    height = max(height, (bounds.height / figure.dpi) / 0.35)
    figure.set_size_inches(width, height)

    if tight_layout:
        margin = _LEGEND_BOTTOM + (bounds.height / figure.dpi + _PAD_LABEL / 72.0) / height
    else:
        margin = max(_LEGEND_MARGIN, _LEGEND_BOTTOM + bounds.height / (height * figure.dpi) + 0.12)

    return margin


def _draw_matrix_grid(ax: _mplp.Axes, length: _ct.oint = None, horizontal: bool = True) -> None:

    if horizontal:
        ax.grid(axis='y', which='minor', color=_COLOR_BLACK, linewidth=_GRID_LINE_WIDTH, alpha=1.0)

    if (length is None) or (length <= _MAXIMUM_TICKS):
        ax.grid(axis='x', which='minor', color=_COLOR_BLACK, linewidth=_GRID_LINE_WIDTH, alpha=1.0)
    else:
        ax.tick_params(axis='x', which='minor', bottom=False)


def _fit_graph_bounds(ax: _mplp.Axes) -> None:

    def _on_draw(event):  # pragma: no cover

        if event.canvas is not figure.canvas:
            return

        axes = event.canvas.figure.axes

        if len(axes) != 1:
            return

        if _update_graph_bounds(axes[0], event.renderer):
            event.canvas.draw_idle()

    figure = ax.figure

    for _ in range(5):

        figure.canvas.draw()

        # noinspection PyUnresolvedReferences
        if not _update_graph_bounds(ax, figure.canvas.get_renderer()):
            break

    figure.canvas.mpl_connect('draw_event', _on_draw)


def _generate_hmm_extended(hmm: _ct.thmm, nodes_color: bool, nodes_shape: bool, edges_label: bool) -> _pyd.Dot:

    magnitude = _n.calculate_magnitude(hmm.p, hmm.e)
    node_colors = _get_colors(hmm.n) if nodes_color else []
    edge_colors = list(node_colors) if nodes_color else []

    g = _pyd.Dot(graph_type='digraph')
    g.set_node_defaults(fontsize=_FONT_SIZE_GRAPH_LABEL, fontname='Arial')
    g.set_edge_defaults(fontsize=_FONT_SIZE_EDGE_LABEL, fontname='Arial')

    g_sub1 = _pyd.Subgraph()
    g.add_subgraph(g_sub1)

    g_sub2 = _pyd.Subgraph(rank='same')
    g.add_subgraph(g_sub2)

    for i in range(hmm.n):

        state = hmm.states[i]

        node_attributes = {}

        if nodes_color:
            node_attributes['style'] = 'filled'
            node_attributes['fillcolor'] = node_colors[i]

        if nodes_shape:
            node_attributes['shape'] = 'ellipse'

        g_sub1.add_node(_pyd.Node(f'ST{i}', label=_get_graphviz_node_label(state), **node_attributes))

    for j, symbol in enumerate(hmm.symbols):

        node_attributes = {}

        if nodes_color:
            node_attributes['style'] = 'filled'
            node_attributes['fillcolor'] = _DEFAULT_COLOR_SYMBOL

        if nodes_shape:
            node_attributes['shape'] = 'hexagon'

        g_sub2.add_node(_pyd.Node(f'SY{j}', label=_get_graphviz_node_label(symbol), **node_attributes))

    for i in range(hmm.n):

        for j in range(hmm.n):

            tp = hmm.p[i, j]

            if tp > 0.0:

                edge_attributes = {
                    'style': 'filled',
                    'color': _DEFAULT_COLOR_EDGE
                }

                if edges_label:
                    edge_attributes['label'] = _get_graphviz_edge_label(tp, magnitude)
                    edge_attributes['fontsize'] = _FONT_SIZE_EDGE_LABEL

                g.add_edge(_pyd.Edge(f'ST{i}', f'ST{j}', **edge_attributes))

        for j in range(hmm.k):

            ep = hmm.e[i, j]

            if ep > 0.0:

                edge_attributes = {
                    'style': 'dashed',
                    'color': edge_colors[i] if nodes_color else _DEFAULT_COLOR_EDGE
                }

                if edges_label:
                    edge_attributes['label'] = _get_graphviz_edge_label(ep, magnitude)
                    edge_attributes['fontsize'] = _FONT_SIZE_EDGE_LABEL

                g.add_edge(_pyd.Edge(f'ST{i}', f'SY{j}', **edge_attributes))

    return g


def _generate_hmm_standard(hmm: _ct.thmm, nodes_color: bool, nodes_shape: bool, edges_label: bool) -> _ct.ttup_any:

    magnitude = _n.calculate_magnitude(hmm.p, hmm.e)
    node_colors = _get_colors(hmm.n) if nodes_color else []
    edge_colors = list(node_colors) if nodes_color else []

    g = hmm.to_graph()
    positions = _nx.multipartite_layout(g, align='horizontal', subset_key='layer')

    node_data = []

    for node in g.nodes:

        if nodes_color:
            if node in hmm.states:
                node_color = node_colors[hmm.states.index(node)]
            else:
                node_color = _DEFAULT_COLOR_SYMBOL
        else:
            node_color = _DEFAULT_COLOR_NODE

        if nodes_shape:
            if node in hmm.states:
                node_shape = 'o'
            else:
                node_shape = 'H'
        else:
            node_shape = 'o'

        node_data.append((node, node_color, node_shape))

    edge_data = []
    edge_labels_curved, edge_labels_straight_state, edge_labels_straight_symbol = {}, {}, {}

    for i in range(hmm.n):

        state_i = hmm.states[i]

        for j in range(hmm.n):

            tp = hmm.p[i, j]

            if tp > 0.0:

                state_j = hmm.states[j]

                edge = (state_i, state_j)
                edge_color = _DEFAULT_COLOR_EDGE

                if (i != j) and ((edge[1], edge[0]) in g.edges):

                    edge_length = abs(i - j)
                    edge_rad = 0.15 if edge_length == 1 else edge_length * 0.25
                    edge_connection = f'arc3, rad={edge_rad:f}'

                    if edges_label:
                        edge_labels_curved[edge] = (edge_rad, f' {round(tp, magnitude):.{magnitude}f} ')

                else:

                    edge_connection = 'arc3'

                    if edges_label:
                        edge_labels_straight_state[edge] = f' {round(tp, magnitude):.{magnitude}f} '

                edge_data.append((edge, edge_color, {'arrows': True, 'connectionstyle': edge_connection}))

        for j in range(hmm.k):

            ep = hmm.e[i, j]

            if ep > 0.0:

                symbol_j = hmm.symbols[j]

                edge = (state_i, symbol_j)
                edge_color = edge_colors[i] if nodes_color else _DEFAULT_COLOR_EDGE

                if edges_label:
                    edge_labels_straight_symbol[edge] = f' {round(ep, magnitude):.{magnitude}f} '

                edge_data.append((edge, edge_color, {'style': 'dashed'}))

    return g, positions, node_data, edge_data, edge_labels_straight_state, edge_labels_straight_symbol, edge_labels_curved


def _generate_mc_extended(mc: _ct.tmc, nodes_color: bool, nodes_shape: bool, edges_label: bool) -> _pyd.Dot:

    magnitude = _n.calculate_magnitude(mc.p)
    node_colors = _get_colors(len(mc.communicating_classes)) if nodes_color else []

    g = _pyd.Dot(graph_type='digraph')
    g.set_node_defaults(fontsize=_FONT_SIZE_GRAPH_LABEL, fontname='Arial')
    g.set_edge_defaults(fontsize=_FONT_SIZE_EDGE_LABEL, fontname='Arial')

    for i in range(mc.size):

        state_i = mc.states[i]

        node_attributes = {}

        if nodes_color:
            for index, cc in enumerate(mc.communicating_classes):
                if state_i in cc:
                    node_attributes['style'] = 'filled'
                    node_attributes['fillcolor'] = node_colors[index]
                    break

        if nodes_shape:
            if state_i in mc.transient_states:  # pragma: no cover
                node_attributes['shape'] = 'box'
            else:
                node_attributes['shape'] = 'ellipse'

        g.add_node(_pyd.Node(f'S{i}', label=_get_graphviz_node_label(state_i), **node_attributes))

        for j in range(mc.size):

            tp = mc.p[i, j]

            if tp > 0.0:

                edge_attributes = {
                    'style': 'filled',
                    'color': _DEFAULT_COLOR_EDGE
                }

                if edges_label:
                    edge_attributes['label'] = _get_graphviz_edge_label(tp, magnitude)
                    edge_attributes['fontsize'] = _FONT_SIZE_EDGE_LABEL

                g.add_edge(_pyd.Edge(f'S{i}', f'S{j}', **edge_attributes))

    return g


def _generate_mc_standard(mc: _ct.tmc, nodes_color: bool, nodes_shape: bool, edges_label: bool) -> _ct.ttup_any:

    magnitude = _n.calculate_magnitude(mc.p)
    node_colors = _get_colors(len(mc.communicating_classes)) if nodes_color else []

    g = mc.to_graph()
    positions = _nx.spring_layout(g, seed=0)

    node_data = []

    for node in g.nodes:

        node_color = _DEFAULT_COLOR_NODE

        if nodes_color:
            for index, cc in enumerate(mc.communicating_classes):
                if node in cc:
                    node_color = node_colors[index]
                    break

        if nodes_shape:
            if node in mc.transient_states:  # pragma: no cover
                node_shape = 's'
            else:
                node_shape = 'o'
        else:
            node_shape = 'o'

        node_data.append((node, node_color, node_shape))

    edge_data = []
    edge_labels_curved, edge_labels_straight = {}, {}

    for i in range(mc.size):

        state_i = mc.states[i]

        for j in range(mc.size):

            tp = mc.p[i, j]

            if tp > 0.0:

                state_j = mc.states[j]

                edge = (state_i, state_j)
                edge_color = _DEFAULT_COLOR_EDGE

                if (i != j) and ((edge[1], edge[0]) in g.edges):

                    edge_connection = 'arc3, rad=0.1'

                    if edges_label:
                        edge_labels_curved[edge] = (0.1, f' {round(tp, magnitude):.{magnitude}f} ')

                else:

                    edge_connection = 'arc3'

                    if edges_label:
                        edge_labels_straight[edge] = f' {round(tp, magnitude):.{magnitude}f} '

                edge_data.append((edge, edge_color, {'arrows': True, 'connectionstyle': edge_connection}))

    return g, positions, node_data, edge_data, edge_labels_straight, {}, edge_labels_curved


def _generate_trellis_extended(
    hmm: _ct.thmm, initial_distribution: _ct.tarray, symbols: _ct.tlist_int,
    forward: bool, matrix: _ct.tarray,
    prediction_states: _ct.tlist_int
) -> _pyd.Dot:

    n, f = hmm.n, len(symbols)

    g = _pyd.Dot(
        graph_type='digraph', layout='neato',
        overlap='true', splines='line', outputorder='edgesfirst', margin='0',
        fontname='sans-serif'
    )

    g.set_node_defaults(fontsize=_FONT_SIZE_GRAPH_LABEL, fontname='Arial')

    for row in range(n):
        g.add_node(_pyd.Node(
            f'state{row}',
            label=_get_graphviz_node_label(hmm.states[row]), shape='plaintext',
            pos=f'0.4,{n - row}!'
        ))

    for col in range(f):

        path_current = prediction_states[col]
        col_position = (col + 1) * 1.1

        header_label = "<T<FONT POINT-SIZE='8'><SUB>0</SUB></FONT>>" if col == 0 else _get_graphviz_node_label(hmm.symbols[symbols[col]])

        g.add_node(_pyd.Node(
            f'header{col}',
            label=header_label, shape='plaintext',
            pos=f'{col_position},{n + 0.6}!'
        ))

        for row in range(n):

            node_index = (row * f) + col
            node_color = _DEFAULT_COLOR_PATH if path_current == row else _DEFAULT_COLOR_NODE

            g.add_node(_pyd.Node(
                f'node{node_index}',
                fillcolor=node_color, label=f'{float(matrix[row, col]):.2f}', style='filled',
                pos=f'{col_position},{n - row}!'
            ))

    for row in range(n):

        row_offset = row * f

        for col in range(f - 1):

            if (col == 0) and (not initial_distribution[row] > 0.0):
                continue

            for row_next in range(n):

                if hmm.p[row][row_next] > 0.0:

                    if forward:
                        edge_from = f'node{row_offset + col}'
                        edge_to = f'node{(row_next * f) + col + 1}'
                    else:
                        edge_from = f'node{(row_next * f) + col + 1}'
                        edge_to = f'node{row_offset + col}'

                    edge_color = _DEFAULT_COLOR_PATH if prediction_states[col] == row and prediction_states[
                        col + 1] == row_next else _DEFAULT_COLOR_EDGE

                    g.add_edge(_pyd.Edge(edge_from, edge_to, color=edge_color))

    return g


def _generate_trellis_standard(
    hmm: _ct.thmm, initial_distribution: _ct.tarray, symbols: _ct.tlist_int,
    forward: bool, matrix: _ct.tarray,
    prediction_states: _ct.tlist_int
) -> _ct.ttup_any:

    n, f = hmm.n, len(symbols)

    trellis = _nx.DiGraph()
    node_colors, node_edges, node_labels, node_positions = [], [], {}, {}

    node_index = 0

    for row in range(n):

        row_offset = float(n - row)

        for col in range(f):

            trellis.add_node(node_index)

            if prediction_states[col] == row:
                node_colors.append(_DEFAULT_COLOR_PATH)
            else:
                node_colors.append(_DEFAULT_COLOR_NODE)

            node_edges.append(_DEFAULT_COLOR_EDGE)
            node_labels[node_index] = f'{float(matrix[row, col]):.2f}'
            node_positions[node_index] = (col + 1.0, row_offset)

            node_index += 1

    for row in range(n):

        row_offset = row * f

        for col in range(f - 1):

            if (col == 0) and (not initial_distribution[row] > 0.0):
                continue

            for row_next in range(n):

                if hmm.p[row][row_next] > 0.0:

                    if forward:
                        edge = (row_offset + col, (row_next * f) + col + 1)
                    else:
                        edge = ((row_next * f) + col + 1, row_offset + col)

                    if (prediction_states[col] == row) and (prediction_states[col + 1] == row_next):
                        edge_color = _DEFAULT_COLOR_PATH
                    else:
                        edge_color = _DEFAULT_COLOR_EDGE

                    trellis.add_edge(*edge, color=edge_color)

    edge_colors = [trellis.edges[edge]['color'] for edge in trellis.edges()]

    return trellis, node_colors, node_edges, node_labels, node_positions, edge_colors


def _get_boundaries(d: _ct.tarray) -> _ct.tpair_array:

    n, k = d.shape
    w = 0.1 / (n - 1.0) if n > 1 else 0.0

    bottom = _np.zeros((n, k), dtype=float)
    top = _np.zeros((n, k), dtype=float)

    for i in range(k):
        d_i = d[:, i]
        bottom[:, i] = _np.cumsum(d_i + w) - d_i - w
        top[:, i] = _np.cumsum(d_i + w) - w

    bottom = _np.clip(bottom, 0.0, 1.0)
    top = _np.clip(top, 0.0, 1.0)

    return bottom, top


def _get_colormap(signed: bool = False, binary: bool = False) -> _mplcr.LinearSegmentedColormap:

    if signed:
        colors, count = [_DEFAULT_COLOR_NEGATIVE, _COLOR_WHITE, _DEFAULT_COLOR_POSITIVE], 21
    else:
        colors, count = [_COLOR_WHITE, _DEFAULT_COLOR_DATA], 2 if binary else 20

    return _mplcr.LinearSegmentedColormap.from_list('ColorMap', colors, count)


def _get_colors(count: int) -> _ct.tlist_str:

    colors = []
    colors_count = len(_COLORS)

    for index in range(count):

        base_color = _COLORS[index % colors_count]
        cycle = index // colors_count

        if cycle == 0:
            color = base_color
        else:  # pragma: no cover

            base_color_rgb = _np.array(_mplcr.to_rgb(base_color))
            variation = min(0.12 * ((cycle + 1) // 2), 0.36)

            if (cycle % 2) == 1:
                color_rgb = base_color_rgb + ((1.0 - base_color_rgb) * variation)
            else:
                color_rgb = base_color_rgb * (1.0 - variation)

            color = _mplcr.to_hex(color_rgb).upper()

        colors.append(color)

    return colors


def _get_curves(n: int, x1: float, y1: _ct.tarray, x2: float, y2: _ct.tarray) -> _ct.tpair_array:

    tx = _np.reshape(_np.linspace(x1, x2, 15), (1, -1))
    cx = _np.tile(_np.transpose(tx), (1, n))

    ty = (1.0 - _np.cos(_np.reshape(_np.linspace(0.0, _np.pi, 15), (1, -1)))) / 2.0
    cy = _np.tile(y1, (15, 1)) + (_np.tile(y2 - y1, (15, 1)) * _np.tile(_np.transpose(ty), (1, n)))

    return cx, cy


def _get_graphviz_edge_label(probability: float, magnitude: int) -> str:

    label = f'{round(probability, magnitude):.{magnitude}f}'

    return (
        f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="1" BGCOLOR="{_COLOR_WHITE}">'
        f'<TR><TD>{label}</TD></TR>'
        f'</TABLE>>'
    )


def _get_graphviz_node_label(label: str) -> str:

    label = label.replace('\\', '\\\\').replace('"', '\\"')
    label = label.replace('\r\n', '\n').replace('\r', '\n').replace('\n', '\\n')

    return f'"{label}"'


def _get_line_styles(matrix: _ct.tarray) -> _ct.tlist_any:

    count, length = matrix.shape
    styles = []

    for i in range(count):

        overlaps = [j for j in range(count) if _np.allclose(matrix[i, :], matrix[j, :], rtol=1e-12, atol=1e-14)]
        overlap_index = overlaps.index(i)

        marker = 'o' if length == 2 else None
        line_style = '-'

        if len(overlaps) > 1:  # pragma: no cover
            line_style = ('-', '--', ':', '-.')[overlap_index % 4]
            if overlap_index > 0:
                marker = ('o', 's', '^', 'D')[((overlap_index - 1) // 4) % 4]

        styles.append((line_style, marker))

    return styles


def _get_polygons_bars(d: _ct.tarray, bb: _ct.tarray, bt: _ct.tarray, colors: _ct.tlist_str) -> _ct.tlist_any:

    i, j = d.shape
    w = min(j / 40.0, 0.45)

    polygons = []

    for oj in range(j):

        xm = oj - w
        xp = oj + w

        for oi in range(i):

            yb = bb[oi, oj]
            yt = bt[oi, oj]

            x = [xm, xp, xp, xm]
            y = [yb, yb, yt, yt]

            polygons.append(_mplpc.Polygon(list(zip(x, y)), facecolor=colors[oi], edgecolor=None, alpha=0.8))

    return polygons


def _get_polygons_flows(p: _ct.tarray, d: _ct.tarray, bb: _ct.tarray, colors: _ct.tlist_str, interval: int) -> _ct.tlist_any:

    i, j = d.shape
    w = min(j / 40.0, 0.45)
    q = _npl.matrix_power(p, interval)

    polygons = []

    for oj in range(j - 1):

        bj = _np.copy(bb[:, oj + 1])

        x_lo = oj + w
        x_hi = oj - w + 1.0

        for oi in range(i):

            dij = d[oi, oj]
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
                polygons.append(_mplpc.Polygon(list(zip(x[:, z], y[:, z])), facecolor=colors[oi], edgecolor=None, alpha=0.3))

    return polygons


def _get_ticks_steps(length: int, interval: int = 1, offset: int = 0, cell_boundaries: bool = False, cell_centered: bool = True) -> tuple:

    tick_step = 1

    if length > _MAXIMUM_TICKS:

        target_step = (length - 1) / (_MAXIMUM_TICKS - 1)
        magnitude = 10.0 ** _mt.floor(_mt.log10(target_step))

        for multiplier in (1, 2, 5, 10):

            tick_step = int(multiplier * magnitude)

            if tick_step >= target_step:
                break

    tick_start = int(_mt.ceil(offset / float(tick_step))) * tick_step
    ticks = _np.arange(tick_start, length + offset, tick_step)
    ticks_minor = _np.arange(offset - 0.5, length + offset, 1.0) if cell_boundaries and length <= _MAXIMUM_TICKS else []

    if cell_centered or (length == 1):
        limits = (offset - 0.5, length + offset - 0.5)
    else:
        limits = (offset, length + offset - 1)

    return ticks, ticks_minor, ticks * interval, limits


def _is_graphviz_available(layout: _ct.ostr = None) -> bool:

    if not _pydot_found:  # pragma: no cover
        return False

    if layout is None:

        try:
            result = _sub.run(['dot', '-V'], stdout=_sub.PIPE, stderr=_sub.PIPE, check=False)
        except (OSError, _sub.SubprocessError):  # pragma: no cover
            return False

    else:

        try:
            result = _sub.run(['dot', f'-K{layout}', '-Tpng'], input=b'digraph { A -> B }', stdout=_sub.PIPE, stderr=_sub.PIPE, check=False)
        except (OSError, _sub.SubprocessError):  # pragma: no cover
            return False

    return result.returncode == 0


def _update_graph_bounds(ax: _mplp.Axes, renderer: _ct.tany) -> bool:

    bounds = []

    for artist in [*ax.patches, *ax.texts, *ax.collections]:

        if not artist.get_visible():  # pragma: no cover
            continue

        extent = artist.get_window_extent(renderer)
        values = _np.array(extent.extents, dtype=float)

        if _np.all(_np.isfinite(values)) and (extent.width > 0.0) and (extent.height > 0.0):
            bounds.append(values)

    if len(bounds) == 0:  # pragma: no cover
        return False

    bounds = _np.array(bounds, dtype=float)

    padding = 8.0
    x_min = _np.min(bounds[:, 0]) - padding
    y_min = _np.min(bounds[:, 1]) - padding
    x_max = _np.max(bounds[:, 2]) + padding
    y_max = _np.max(bounds[:, 3]) + padding

    axes_bounds = ax.get_window_extent(renderer)

    if (x_min >= axes_bounds.x0) and (y_min >= axes_bounds.y0) and (x_max <= axes_bounds.x1) and (y_max <= axes_bounds.y1):
        return False

    data_bounds = ax.transData.inverted().transform([[x_min, y_min], [x_max, y_max]])
    data_x_min, data_y_min = data_bounds[0]
    data_x_max, data_y_max = data_bounds[1]
    axis_x_min, axis_x_max = ax.get_xlim()
    axis_y_min, axis_y_max = ax.get_ylim()

    ax.set_xlim(min(axis_x_min, data_x_min), max(axis_x_max, data_x_max))
    ax.set_ylim(min(axis_y_min, data_y_min), max(axis_y_max, data_y_max))

    return True


def _xticks_labels(ax: _mplp.Axes, labels_name: _ct.ostr, labels: _ct.tlist_str, cell_boundaries: bool = False) -> None:

    size = len(labels)

    ax.set_xticks(_np.arange(0.0, size, 1.0), minor=False)
    ax.set_xticks(_np.arange(-0.5, size, 1.0) if cell_boundaries else [], minor=True)
    ax.set_xticklabels(labels)

    if labels_name is not None:
        ax.set_xlabel(labels_name, fontsize=_FONT_SIZE_AXES_LABEL, fontweight=_FONT_WEIGHT_AXES_LABEL, labelpad=_PAD_LABEL)


def _xticks_steps(ax: _mplp.Axes, ticks_data: tuple, labels_name: _ct.ostr = 'Steps') -> None:

    ticks, ticks_minor, tick_labels, limits = ticks_data

    ax.set_xlim(*limits)
    ax.set_xticks(ticks, minor=False)
    ax.set_xticks(ticks_minor, minor=True)
    ax.set_xticklabels(tick_labels)

    if labels_name is not None:
        ax.set_xlabel(labels_name, fontsize=_FONT_SIZE_AXES_LABEL, fontweight=_FONT_WEIGHT_AXES_LABEL, labelpad=_PAD_LABEL)


def _yticks_frequency(ax: _mplp.Axes, bottom: float, top: float, labels_name: _ct.ostr = 'Frequency') -> None:

    ax.set_ylim(bottom, top)
    ax.set_yticks(_np.linspace(0.0, 1.0, 11))

    if labels_name is not None:
        ax.set_ylabel(labels_name, fontsize=_FONT_SIZE_AXES_LABEL, fontweight=_FONT_WEIGHT_AXES_LABEL, labelpad=_PAD_LABEL)


def _yticks_labels(ax: _mplp.Axes, labels_name: _ct.ostr, labels: _ct.tlist_str) -> None:

    size = len(labels)

    ax.set_yticks(_np.arange(0.0, size, 1.0), minor=False)
    ax.set_yticks(_np.arange(-0.5, size, 1.0), minor=True)
    ax.set_yticklabels(labels)

    if labels_name is not None:
        ax.set_ylabel(labels_name, fontsize=_FONT_SIZE_AXES_LABEL, fontweight=_FONT_WEIGHT_AXES_LABEL, labelpad=_PAD_LABEL)


def plot_comparison(models: _ct.tlist_mdl, underlying_matrices: str, names: _ct.tlist_str, dpi: int) -> _ct.tplot:

    matrices = [model.e if underlying_matrices == 'emission' else model.p for model in models]

    space = len(models)
    rows = int(_mt.sqrt(space))
    columns = int(_mt.ceil(space / float(rows)))
    color_map = _get_colormap()

    figure, axes = _mplp.subplots(nrows=rows, ncols=columns, squeeze=False, figsize=(_FIGURE_WIDTH, _FIGURE_HEIGHT), constrained_layout=True, dpi=dpi)
    axes = list(axes.flat)
    ax_is = None

    for ax, matrix, name in zip(axes, matrices, names):

        matrix_rows, matrix_columns = matrix.shape
        detailed = max(matrix_rows, matrix_columns) <= _MAXIMUM_TICKS

        ticks_x = _np.arange(-0.5, matrix_columns, 1.0) if detailed else []
        ticks_y = _np.arange(-0.5, matrix_rows, 1.0) if detailed else []

        ax_is = ax.imshow(matrix, cmap=color_map, vmin=0.0, vmax=1.0, aspect='auto', interpolation='none')

        ax.set_xticks([], minor=False)
        ax.set_xticks(ticks_x, minor=True)
        ax.set_yticks([], minor=False)
        ax.set_yticks(ticks_y, minor=True)

        ax.tick_params(axis='both', which='both', bottom=False, left=False)

        if detailed:
            _draw_matrix_grid(ax)

        ax.set_title(name, fontsize=_FONT_SIZE_AXES_TITLE, fontweight=_FONT_WEIGHT_AXES_TITLE, pad=_PAD_AXES_TITLE)

    for ax in axes[space:]:
        ax.set_visible(False)

    _draw_colorbar(figure, ax_is, axes[:space], 'horizontal')

    figure.suptitle('Comparison Plot', fontsize=_FONT_SIZE_FIGURE_TITLE, fontweight=_FONT_WEIGHT_FIGURE_TITLE)

    return figure, axes


def plot_convergence(mc: _ct.tmc, steps: int, initial_status: _ct.oarray, dpi: int, full_scale: bool) -> _ct.tplot:

    stationary_distribution = mc.pi[0]
    distributions = mc.propagate(steps, initial_status=initial_status, output_last=False)
    distributions = _np.asarray(distributions, dtype=float)
    distances = 0.5 * _np.sum(_np.abs(distributions - stationary_distribution), axis=1)

    size = distances.size
    x = _np.arange(size)
    ticks_data = _get_ticks_steps(size, cell_centered=False)
    distance_max = _np.max(distances)
    y_max = 1.0 if full_scale or distance_max == 0.0 else min(1.0, distance_max * 1.1)

    figure, ax = _mplp.subplots(figsize=(_FIGURE_WIDTH, _FIGURE_HEIGHT), dpi=dpi)

    ax.plot(x, distances, color=_DEFAULT_COLOR_DATA, linewidth=_LINE_WIDTH)

    _xticks_steps(ax, ticks_data)
    ax.set_ylim(0.0, y_max)
    ax.set_ylabel('Total Variation Distance', fontsize=_FONT_SIZE_AXES_LABEL, fontweight=_FONT_WEIGHT_AXES_LABEL, labelpad=_PAD_LABEL)

    ax.grid(which='major', color=_DEFAULT_COLOR_GRID, linewidth=_GRID_LINE_WIDTH, alpha=_GRID_ALPHA)

    figure.suptitle('Convergence Plot', fontsize=_FONT_SIZE_FIGURE_TITLE, fontweight=_FONT_WEIGHT_FIGURE_TITLE)

    return figure, ax


def plot_decoding(hmm: _ct.thmm, symbols: _ct.tlist_int, initial_distribution: _ct.tarray, dpi: int) -> _ct.tplot:

    decoding = hmm.decode(symbols, initial_status=initial_distribution)
    prediction = hmm.predict('mle', symbols, initial_status=initial_distribution, output_indices=True)

    if (decoding is None) or (prediction is None):  # pragma: no cover
        raise ValueError('The computation of the decoded sequences failed.')

    _, posterior, _, _, _, _ = decoding
    _, viterbi = prediction

    viterbi = _np.asarray(viterbi, dtype=int)
    map_states = _np.argmax(posterior, axis=0)
    disagreement = (viterbi != map_states).astype(int)

    size = len(symbols)
    detailed = size <= _MAXIMUM_TICKS
    ticks_data = _get_ticks_steps(size, cell_boundaries=True)

    colors_states = _get_colors(hmm.n)
    colors_symbols = _mplp.get_cmap('Greys')(_np.linspace(0.25, 0.75, hmm.k))

    color_map_states = _mplcr.ListedColormap(colors_states)
    color_map_symbols = _mplcr.ListedColormap(colors_symbols)
    color_map_disagreement = _mplcr.ListedColormap([_COLOR_WHITE, _DEFAULT_COLOR_NEGATIVE])

    sequences = [
        ('Observed', symbols, hmm.symbols, color_map_symbols),
        ('Viterbi', viterbi, hmm.states, color_map_states),
        ('MAP', map_states, hmm.states, color_map_states),
        ('Disagreement', disagreement, None, color_map_disagreement)
    ]

    panels = []

    for name, sequence, labels, color_map in sequences:

        matrix = _np.asarray(sequence, dtype=int).reshape(1, -1)
        ann = []

        if detailed and (labels is not None):
            for i, value in enumerate(sequence):
                color = color_map(int(value))
                luminance = (0.299 * color[0]) + (0.587 * color[1]) + (0.114 * color[2])
                text_color = _COLOR_BLACK if luminance > 0.5 else _COLOR_WHITE
                ann.append((i, labels[value], text_color))

        panels.append((name, matrix, color_map, ann))

    legend_handlers_symbols, legend_handlers_states = [], []

    if not detailed:
        legend_handlers_symbols = [_mplpc.Patch(facecolor=color_map_symbols(i), label=label) for i, label in enumerate(hmm.symbols)]
        legend_handlers_states = [_mplpc.Patch(facecolor=color_map_states(i), label=label) for i, label in enumerate(hmm.states)]

    figure, axes = _mplp.subplots(nrows=4, sharex=True, figsize=(_FIGURE_WIDTH * 1.5, _FIGURE_HEIGHT), tight_layout=True, dpi=dpi)
    axes = list(axes.flat)

    for ax, (name, matrix, color_map, ann) in zip(axes, panels):

        ax.imshow(matrix, cmap=color_map, vmin=-0.5, vmax=color_map.N - 0.5, aspect='auto', interpolation='none')

        for i, label, color in ann:
            ax.text(i, 0.0, label, color=color, horizontalalignment='center', verticalalignment='center')

        _xticks_steps(ax, ticks_data, 'Observations' if ax is axes[-1] else None)
        ax.set_yticks([], minor=False)
        ax.set_yticks([], minor=True)
        ax.set_ylabel(name, fontsize=_FONT_SIZE_AXES_LABEL, fontweight=_FONT_WEIGHT_AXES_LABEL, horizontalalignment='right', verticalalignment='center', rotation=0, labelpad=_PAD_LABEL_SIDE)
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

        if ax is axes[-1]:
            ax.tick_params(axis='x', which='major', bottom=True, labelbottom=True)

        if detailed:
            _draw_matrix_grid(ax, size, horizontal=False)

    legend_margin = None

    if not detailed:
        legend_handlers = legend_handlers_symbols + legend_handlers_states
        legend_labels = hmm.symbols + hmm.states
        legend_margin = _draw_legend_bottom(figure, legend_handlers, legend_labels, tight_layout=True)

    figure.suptitle('Decoding Plot', fontsize=_FONT_SIZE_FIGURE_TITLE, fontweight=_FONT_WEIGHT_FIGURE_TITLE)

    if legend_margin is not None:
        figure.get_layout_engine().set(rect=(0.0, legend_margin, 1.0, 1.0))

    figure.align_ylabels(axes)

    return figure, axes


def plot_flow(mc: _ct.tmc, steps: int, interval: int, initial_status: _ct.oarray, dpi: int) -> _ct.tplot:

    distributions = mc.propagate(steps * interval, initial_status=initial_status, output_last=False)
    distributions = _np.transpose(_np.asarray(distributions, dtype=float)[::interval, :])

    p = mc.p
    d = distributions * 0.9

    boundaries_bottom, boundaries_top = _get_boundaries(d)
    boundaries_middle = (boundaries_bottom + boundaries_top) / 2.0

    colors = _get_colors(mc.size)
    ticks_data = _get_ticks_steps(steps + 1, interval)
    legend_labels = mc.states
    legend_handlers = [_mplpc.Patch(color=colors[i], label=label) for i, label in enumerate(legend_labels)]

    polygons_bars = _get_polygons_bars(d, boundaries_bottom, boundaries_top, colors)
    polygons_flows = _get_polygons_flows(p, d, boundaries_bottom, colors, interval)
    ann = []

    for ai in range(distributions.shape[0]):
        for aj in range(distributions.shape[1]):
            dv = distributions[ai, aj]
            if dv > 0.05:
                ann.append((aj, float(boundaries_middle[ai, aj]), f'{dv:.3f}'))

    figure, ax = _mplp.subplots(figsize=(_FIGURE_WIDTH, _FIGURE_HEIGHT), dpi=dpi)

    for e in polygons_flows:
        ax.add_patch(e)

    for e in polygons_bars:
        ax.add_patch(e)

    for x, y, label in ann:
        ax.text(x, y, label, horizontalalignment='center', verticalalignment='center')

    _xticks_steps(ax, ticks_data)
    ax.set_ylim(0.0, 1.0)
    ax.invert_yaxis()

    legend_margin = _draw_legend_bottom(figure, legend_handlers, legend_labels)

    figure.suptitle('Flow Plot', fontsize=_FONT_SIZE_FIGURE_TITLE, fontweight=_FONT_WEIGHT_FIGURE_TITLE)

    figure.subplots_adjust(bottom=legend_margin)

    return figure, ax


def plot_graph(model: _ct.tmdl, nodes_color: bool, nodes_shape: bool, edges_label: bool, force_standard: bool, dpi: int) -> _ct.tplot:

    model_mc = model.__class__.__name__ == 'MarkovChain'
    layout = 'neato' if model_mc and model.is_irreducible else None
    extended_graph = (not force_standard) and _is_graphviz_available(layout)

    if extended_graph:

        if model_mc:
            graph = _generate_mc_extended(model, nodes_color, nodes_shape, edges_label)
        else:
            graph = _generate_hmm_extended(model, nodes_color, nodes_shape, edges_label)

        if layout is not None:
            graph.set('layout', layout)

        if layout == 'neato':
            graph.set('overlap', 'false')
            graph.set('splines', 'true')
            graph.set('sep', '+8')
        else:
            graph.set('nodesep', '0.50')
            graph.set('ranksep', '0.75')

        preview = _decode_graphviz_image(graph, dpi)
        img_height = preview.shape[0] / dpi
        img_width = preview.shape[1] / dpi

        figure, ax = _mplp.subplots(figsize=(max(_FIGURE_WIDTH, img_width * 1.5), max(_FIGURE_HEIGHT, img_height * 1.5)), dpi=dpi)

        figure.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.90)

        ax.axis('off')

        figure.suptitle('Graph Plot', fontsize=_FONT_SIZE_FIGURE_TITLE, fontweight=_FONT_WEIGHT_FIGURE_TITLE)

        figure.canvas.draw()

        bounds = ax.get_window_extent()
        initial_dpi = max(1, int(_mt.floor(dpi * min(bounds.width / preview.shape[1], bounds.height / preview.shape[0]))))
        _draw_graphviz_image(graph, ax, dpi, initial_dpi=initial_dpi, oversampling=_GRAPHVIZ_OVERSAMPLING)

        figure.canvas.draw()

    else:

        if model_mc:
            graph_data = _generate_mc_standard(model, nodes_color, nodes_shape, edges_label)
        else:
            graph_data = _generate_hmm_standard(model, nodes_color, nodes_shape, edges_label)

        graph, positions, node_data, edge_data, edge_labels_straight, edge_labels_symbol, edge_labels_curved = graph_data

        figure, ax = _mplp.subplots(figsize=(_FIGURE_WIDTH, _FIGURE_HEIGHT), dpi=dpi)

        for node, node_color, node_shape in node_data:
            _nx.draw_networkx_nodes(
                graph, positions, ax=ax,
                nodelist=[node], node_color=node_color, edgecolors=_DEFAULT_COLOR_EDGE,
                node_shape=node_shape, node_size=_NODE_SIZE
            )

        _nx.draw_networkx_labels(graph, positions, ax=ax, font_size=_FONT_SIZE_GRAPH_LABEL, font_weight=_FONT_WEIGHT_AXES_LABEL)

        for edge, edge_color, edge_attributes in edge_data:
            _nx.draw_networkx_edges(
                graph, positions, ax=ax,
                edgelist=[edge], edge_color=edge_color, **edge_attributes
            )

        if len(edge_labels_straight) > 0:
            _nx.draw_networkx_edge_labels(
                graph, positions, ax=ax,
                edge_labels=edge_labels_straight,
                font_size=_FONT_SIZE_EDGE_LABEL
            )

        if len(edge_labels_symbol) > 0:
            _nx.draw_networkx_edge_labels(
                graph, positions, ax=ax,
                edge_labels=edge_labels_symbol, label_pos=0.25,
                font_size=_FONT_SIZE_EDGE_LABEL
            )

        if len(edge_labels_curved) > 0:
            _draw_edge_labels_curved(ax, positions, edge_labels_curved)

        ax.set_aspect('equal')
        ax.axis('off')

        figure.suptitle('Graph Plot', fontsize=_FONT_SIZE_FIGURE_TITLE, fontweight=_FONT_WEIGHT_FIGURE_TITLE)

        _fit_graph_bounds(ax)

    return figure, ax


def plot_implied_timescales(lags: _ct.tlist_int, fitting_type: str, fitting_param: _ct.tany, possible_states: _ct.tlist_str, states: _ct.tlist_int, dpi: int) -> _ct.tplot:

    timescales = _np.full((len(lags), len(possible_states) - 1), _np.nan, dtype=float)

    for i, lag in enumerate(lags):

        try:
            mc = _MarkovChain.fit_sequence(fitting_type, possible_states, states[::lag], fitting_param)
        except ValueError:  # pragma: no cover

            if fitting_type != 'rmle':
                raise

            continue

        values = mc.implied_timescales

        if values is not None:
            timescales[i, :] = values[1:] * lag

    timescales[~_np.isfinite(timescales) | (timescales <= 0.0)] = _np.nan
    finite_timescales = timescales[_np.isfinite(timescales)]

    if finite_timescales.size == 0:  # pragma: no cover
        raise ValueError('The implied timescales cannot be computed for the supplied sequence and lags.')

    x = _np.asarray(lags, dtype=int)
    ticks_data = _get_ticks_steps(lags[-1] - lags[0] + 1, offset=lags[0], cell_centered=False)
    reference = _np.asarray(ticks_data[3], dtype=float)

    y_min = min(reference[0], _np.min(finite_timescales)) / 1.1
    y_max = max(reference[-1], _np.max(finite_timescales)) * 1.1

    colors = _get_colors(timescales.shape[1])

    figure, ax = _mplp.subplots(figsize=(_FIGURE_WIDTH, _FIGURE_HEIGHT), dpi=dpi)

    ax.fill_between(reference, y_min, reference, color=_COLOR_GRAY, alpha=0.5)
    ax.plot(reference, reference, color=_DEFAULT_COLOR_REFERENCE, linestyle='--', linewidth=_LINE_WIDTH_REFERENCE)

    for i, color in enumerate(colors):
        if _np.any(_np.isfinite(timescales[:, i])):
            ax.plot(
                x, timescales[:, i],
                color=color, linewidth=_LINE_WIDTH,
                marker='o', markersize=_MARKER_SIZE
            )

    _xticks_steps(ax, ticks_data, 'Lag')

    ax.set_yscale('log')
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_locator(_mplt.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(_mplt.FormatStrFormatter('%g'))
    ax.yaxis.set_minor_locator(_mplt.NullLocator())

    ax.grid(which='major', color=_DEFAULT_COLOR_GRID, linewidth=_GRID_LINE_WIDTH, alpha=_GRID_ALPHA)

    figure.suptitle('Implied Timescales Plot', fontsize=_FONT_SIZE_FIGURE_TITLE, fontweight=_FONT_WEIGHT_FIGURE_TITLE)

    return figure, ax


def plot_matrix(model: _ct.tmdl, underlying_matrix: str, dpi: int) -> _ct.tplot:

    matrix = model.e if underlying_matrix == 'emission' else model.p
    labels_x = model.symbols if underlying_matrix == 'emission' else model.states
    labels_y = model.states
    labels_name = 'Symbols' if underlying_matrix == 'emission' else 'States'
    color_map = _get_colormap()

    figure, ax = _mplp.subplots(figsize=(_FIGURE_WIDTH, _FIGURE_HEIGHT), constrained_layout=True, dpi=dpi)

    ax_is = ax.imshow(matrix, cmap=color_map, vmin=0.0, vmax=1.0, aspect='auto', interpolation='none')

    _xticks_labels(ax, labels_name, labels_x, True)
    _yticks_labels(ax, 'States', labels_y)

    _draw_matrix_grid(ax)

    _draw_colorbar(figure, ax_is, ax, 'horizontal')

    figure.suptitle(f'Matrix Plot ({underlying_matrix.capitalize()})', fontsize=_FONT_SIZE_FIGURE_TITLE, fontweight=_FONT_WEIGHT_FIGURE_TITLE)

    return figure, ax


def plot_posterior(plot_type: str, hmm: _ct.thmm, symbols: _ct.tlist_int, initial_distribution: _ct.tarray, dpi: int) -> _ct.tplot:

    decoding = hmm.decode(symbols, initial_status=initial_distribution)

    if decoding is None:  # pragma: no cover
        raise ValueError('The computation of the posterior probabilities failed.')

    _, posterior, _, _, forward, _ = decoding

    size = len(symbols)
    labels = hmm.states

    legend_margin = None

    if plot_type == 'heatmap':

        ticks_data = _get_ticks_steps(size, cell_boundaries=True)
        color_map = _get_colormap()
        panels = [('Filtering', forward), ('Smoothing', posterior)]

        figure, axes = _mplp.subplots(nrows=2, figsize=(_FIGURE_WIDTH, _FIGURE_HEIGHT), constrained_layout=True, dpi=dpi)
        axes = list(axes.flat)
        is_axes = []

        for ax, (name, matrix) in zip(axes, panels):

            ax_is = ax.imshow(matrix, cmap=color_map, vmin=0.0, vmax=1.0, aspect='auto', interpolation='none')
            is_axes.append(ax_is)

            _xticks_steps(ax, ticks_data)
            _yticks_labels(ax, 'States', labels)

            _draw_matrix_grid(ax, size)

            ax.set_title(name, fontsize=_FONT_SIZE_AXES_TITLE, fontweight=_FONT_WEIGHT_AXES_TITLE, pad=_PAD_AXES_TITLE)

        _draw_colorbar(figure, is_axes[0], axes, 'vertical')

    else:

        x = _np.arange(size)
        ticks_data = _get_ticks_steps(size, cell_centered=False)
        colors = _get_colors(hmm.n)
        styles = _get_line_styles(posterior)
        marker_interval = max(1, size // 8)

        figure, ax = _mplp.subplots(figsize=(_FIGURE_WIDTH, _FIGURE_HEIGHT), dpi=dpi)

        for i in range(hmm.n):
            ax.fill_between(x, 0.0, posterior[i, :], color=colors[i], alpha=0.35)

        for i, (line_style, marker) in enumerate(styles):
            ax.plot(
                x, posterior[i, :],
                color=colors[i], linestyle=line_style, linewidth=_LINE_WIDTH,
                marker=marker, markersize=_MARKER_SIZE, markevery=marker_interval,
                label=labels[i]
            )

        _xticks_steps(ax, ticks_data)
        _yticks_frequency(ax, 0.0, 1.0, 'Probability')

        ax.grid(which='major', color=_DEFAULT_COLOR_GRID, linewidth=_GRID_LINE_WIDTH, alpha=_GRID_ALPHA)

        legend_margin = _draw_legend_bottom(figure, *ax.get_legend_handles_labels())

        axes = ax

    figure.suptitle(f'Posterior Plot ({plot_type.capitalize()})', fontsize=_FONT_SIZE_FIGURE_TITLE, fontweight=_FONT_WEIGHT_FIGURE_TITLE)

    if legend_margin is not None:
        figure.subplots_adjust(bottom=legend_margin)

    return figure, axes


def plot_redistributions(plot_type: str, mc: _ct.tmc, redistributions: int, initial_status: _ct.ostatus, dpi: int) -> _ct.tplot:

    distributions = mc.propagate(redistributions, initial_status=initial_status, output_last=False)
    distributions = _np.asarray(distributions, dtype=float)
    distributions_length = distributions.shape[0]
    labels = mc.states

    legend_margin = None

    if plot_type == 'heatmap':

        matrix = _np.transpose(distributions)
        ticks_data = _get_ticks_steps(distributions_length, cell_boundaries=True)
        color_map = _get_colormap()

        figure, ax = _mplp.subplots(figsize=(_FIGURE_WIDTH, _FIGURE_HEIGHT), constrained_layout=True, dpi=dpi)

        ax_is = ax.imshow(matrix, cmap=color_map, vmin=0.0, vmax=1.0, aspect='auto', interpolation='none')

        _xticks_steps(ax, ticks_data)
        _yticks_labels(ax, None, labels)

        _draw_matrix_grid(ax, distributions_length)

        _draw_colorbar(figure, ax_is, ax, 'horizontal')

    else:

        x = _np.arange(0.0, distributions_length, 1.0)
        ticks_data = _get_ticks_steps(distributions_length)
        colors = _get_colors(mc.size)
        styles = _get_line_styles(_np.transpose(distributions))
        marker_interval = max(1, distributions_length // 8)
        uniform_start = _np.allclose(distributions[0, :], _np.ones(mc.size, dtype=float) / mc.size)

        figure, ax = _mplp.subplots(figsize=(_FIGURE_WIDTH, _FIGURE_HEIGHT), dpi=dpi)

        for i, (line_style, marker) in enumerate(styles):
            ax.plot(
                x, distributions[:, i],
                color=colors[i], linestyle=line_style, linewidth=_LINE_WIDTH,
                marker=marker, markersize=_MARKER_SIZE, markevery=marker_interval,
                label=labels[i]
            )

        if uniform_start:
            ax.plot(0.0, distributions[0, 0], color=_COLOR_BLACK, marker='o', markersize=_MARKER_SIZE, markerfacecolor=_COLOR_BLACK, markeredgecolor=_COLOR_BLACK, label='Start')

        _xticks_steps(ax, ticks_data)
        _yticks_frequency(ax, -0.05, 1.05)

        ax.grid(which='major', color=_DEFAULT_COLOR_GRID, linewidth=_GRID_LINE_WIDTH, alpha=_GRID_ALPHA)

        legend_margin = _draw_legend_bottom(figure, *ax.get_legend_handles_labels())

    figure.suptitle(f'Redistributions Plot ({plot_type.capitalize()})', fontsize=_FONT_SIZE_FIGURE_TITLE, fontweight=_FONT_WEIGHT_FIGURE_TITLE)

    if legend_margin is not None:
        figure.subplots_adjust(bottom=legend_margin)

    return figure, ax


def plot_residuals(model: _ct.tmdl, sequence: _ct.tlist_int, seed: _ct.oint, dpi: int) -> _ct.tplot:

    residuals = model.residuals(sequence, seed=seed)
    residuals_sorted = _np.sort(residuals)
    size = residuals.size

    quantiles = _sps.ndtri((_np.arange(1, size + 1) - 0.5) / size)  # pylint: disable=no-member
    quantiles_limits = [min(quantiles[0], residuals_sorted[0]), max(quantiles[-1], residuals_sorted[-1])]

    lower = min(-4.0, _np.min(residuals))
    upper = max(4.0, _np.max(residuals))
    x_distribution = _np.linspace(lower, upper, 200)
    y_distribution = _np.exp(-0.5 * x_distribution ** 2.0) / _np.sqrt(2.0 * _np.pi)
    histogram_bins = _np.histogram_bin_edges(residuals, bins='auto')

    maximum_lag = min(20, size - 1)
    centered = residuals - _np.mean(residuals)
    denominator = _np.dot(centered, centered)
    autocorrelation_available = (maximum_lag > 0) and (denominator > 0.0)

    if autocorrelation_available:
        lags = _np.arange(1, maximum_lag + 1)
        correlations = _np.array([_np.dot(centered[:-lag], centered[lag:]) / denominator for lag in lags])
        threshold = _sps.ndtri(0.975) / _np.sqrt(size)  # pylint: disable=no-member
        limit = min(1.0, max(_np.max(_np.abs(correlations)), threshold) * 1.1)
        ticks_lags = _get_ticks_steps(maximum_lag, offset=1)
    else:  # pragma: no cover
        lags, correlations = _np.array([]), _np.array([])
        threshold, limit, ticks_lags = None, None, None

    offset = 1 if model.__class__.__name__ == 'MarkovChain' else 0
    x_residuals = _np.arange(offset, size + offset)
    ticks_residuals = _get_ticks_steps(size, offset=offset)

    figure, axes = _mplp.subplots(nrows=2, ncols=2, figsize=(_FIGURE_WIDTH * 1.5, _FIGURE_HEIGHT * 1.5), tight_layout=True, dpi=dpi)
    axes = list(axes.flat)
    ax_residuals, ax_quantiles, ax_distribution, ax_autocorrelation = axes

    ax_residuals.scatter(x_residuals, residuals, color=_DEFAULT_COLOR_DATA, s=_MARKER_SCATTER_SIZE)
    ax_residuals.axhline(0.0, color=_DEFAULT_COLOR_REFERENCE, linestyle='--', linewidth=_LINE_WIDTH_REFERENCE)

    _xticks_steps(ax_residuals, ticks_residuals, None)

    ax_residuals.grid(which='major', color=_DEFAULT_COLOR_GRID, linewidth=_GRID_LINE_WIDTH, alpha=_GRID_ALPHA)

    ax_residuals.set_title('Residuals', fontsize=_FONT_SIZE_AXES_TITLE, fontweight=_FONT_WEIGHT_AXES_TITLE, pad=_PAD_AXES_TITLE)

    ax_quantiles.scatter(quantiles, residuals_sorted, color=_DEFAULT_COLOR_DATA, s=_MARKER_SCATTER_SIZE)
    ax_quantiles.plot(quantiles_limits, quantiles_limits, color=_DEFAULT_COLOR_REFERENCE, linestyle='--', linewidth=_LINE_WIDTH_REFERENCE)

    ax_quantiles.grid(which='major', color=_DEFAULT_COLOR_GRID, linewidth=_GRID_LINE_WIDTH, alpha=_GRID_ALPHA)

    ax_quantiles.set_title('Q-Q Plot', fontsize=_FONT_SIZE_AXES_TITLE, fontweight=_FONT_WEIGHT_AXES_TITLE, pad=_PAD_AXES_TITLE)

    ax_distribution.hist(residuals, bins=histogram_bins, density=True, facecolor=_DEFAULT_COLOR_DATA, edgecolor=_DEFAULT_COLOR_EDGE)
    ax_distribution.plot(x_distribution, y_distribution, color=_DEFAULT_COLOR_REFERENCE, linewidth=_LINE_WIDTH_REFERENCE)

    ax_distribution.grid(which='major', color=_DEFAULT_COLOR_GRID, linewidth=_GRID_LINE_WIDTH, alpha=_GRID_ALPHA)

    ax_distribution.set_title('Distribution', fontsize=_FONT_SIZE_AXES_TITLE, fontweight=_FONT_WEIGHT_AXES_TITLE, pad=_PAD_AXES_TITLE)

    if autocorrelation_available:

        ax_autocorrelation.axhspan(-threshold, threshold, color=_DEFAULT_COLOR_REFERENCE, alpha=0.2)
        ax_autocorrelation.vlines(lags, 0.0, correlations, color=_DEFAULT_COLOR_DATA, linewidth=_LINE_WIDTH)
        ax_autocorrelation.scatter(lags, correlations, color=_DEFAULT_COLOR_DATA, s=_MARKER_SCATTER_SIZE)
        ax_autocorrelation.axhline(0.0, color=_DEFAULT_COLOR_REFERENCE, linestyle='--', linewidth=_LINE_WIDTH_REFERENCE)

        _xticks_steps(ax_autocorrelation, ticks_lags, None)
        ax_autocorrelation.set_ylim(-limit, limit)

        ax_autocorrelation.grid(which='major', color=_DEFAULT_COLOR_GRID, linewidth=_GRID_LINE_WIDTH, alpha=_GRID_ALPHA)

    else:  # pragma: no cover
        ax_autocorrelation.set_xticks([])
        ax_autocorrelation.set_yticks([])

    ax_autocorrelation.set_title('Autocorrelation', fontsize=_FONT_SIZE_AXES_TITLE, fontweight=_FONT_WEIGHT_AXES_TITLE, pad=_PAD_AXES_TITLE)

    figure.suptitle('Residuals Plot', fontsize=_FONT_SIZE_FIGURE_TITLE, fontweight=_FONT_WEIGHT_FIGURE_TITLE)

    return figure, axes


def plot_sequence(plot_type: str, model: _ct.tmdl, steps: int, initial_state: _ct.ostate, seed: _ct.oint, dpi: int) -> _ct.tplot:

    simulation = model.simulate(steps, initial_state=initial_state, output_indices=True, seed=seed)

    walks = []

    if model.__class__.__name__ == 'MarkovChain':
        walks.append((model.n, 'States', model.states, simulation))
    else:
        walks.append((model.n, 'States', model.states, simulation[0]))
        walks.append((model.k, 'Symbols', model.symbols, simulation[1]))

    plots_count = len(walks)
    panels = []

    if plot_type == 'heatmap':

        for size, labels_name, labels, sequence in walks:

            sequence_matrix = _np.zeros((size, size), dtype=float)

            for i in range(1, steps + 1):
                sequence_matrix[sequence[i - 1], sequence[i]] += 1.0

            sequence_matrix /= _np.sum(sequence_matrix)
            panels.append((labels_name, labels, sequence_matrix))

        color_map = _get_colormap()

        figure, axes = _mplp.subplots(nrows=plots_count, squeeze=False, figsize=(_FIGURE_WIDTH, _FIGURE_HEIGHT), constrained_layout=True, dpi=dpi)
        axes = list(axes.flat)
        is_axes = []

        for ax, (labels_name, labels, matrix) in zip(axes, panels):

            ax_is = ax.imshow(matrix, cmap=color_map, vmin=0.0, vmax=1.0, aspect='auto', interpolation='none')
            is_axes.append(ax_is)

            _xticks_labels(ax, labels_name, labels, True)
            _yticks_labels(ax, labels_name, labels)

            _draw_matrix_grid(ax)

        _draw_colorbar(figure, is_axes[0], axes, 'vertical')

    elif plot_type == 'histogram':

        for size, labels_name, labels, sequence in walks:

            sequence_histogram = _np.bincount(sequence, minlength=size).astype(float)
            sequence_histogram /= _np.sum(sequence_histogram)
            panels.append((labels_name, labels, _np.arange(0.0, size, 1.0), sequence_histogram))

        figure, axes = _mplp.subplots(nrows=plots_count, squeeze=False, figsize=(_FIGURE_WIDTH, _FIGURE_HEIGHT), tight_layout=True, dpi=dpi)
        axes = list(axes.flat)

        for ax, (labels_name, labels, x, histogram) in zip(axes, panels):

            ax.bar(x, histogram, facecolor=_DEFAULT_COLOR_DATA, edgecolor=_DEFAULT_COLOR_EDGE)

            _xticks_labels(ax, labels_name, labels)
            _yticks_frequency(ax, 0.0, 1.0)

    else:

        for size, labels_name, labels, sequence in walks:

            sequence_matrix = _np.zeros((size, steps + 1), dtype=float)

            for index, state in enumerate(sequence):
                sequence_matrix[state, index] = 1.0

            panels.append((labels_name, labels, sequence_matrix))

        ticks_data = _get_ticks_steps(steps + 1, cell_boundaries=True)
        color_map = _get_colormap(binary=True)

        figure, axes = _mplp.subplots(nrows=plots_count, squeeze=False, figsize=(_FIGURE_WIDTH, _FIGURE_HEIGHT), tight_layout=True, dpi=dpi)
        axes = list(axes.flat)

        for ax, (labels_name, labels, matrix) in zip(axes, panels):

            ax.imshow(matrix, cmap=color_map, vmin=0.0, vmax=1.0, aspect='auto', interpolation='none')

            _xticks_steps(ax, ticks_data)
            _yticks_labels(ax, labels_name, labels)

            _draw_matrix_grid(ax, steps + 1)

    figure.suptitle(f'Sequence Plot ({plot_type.capitalize()})', fontsize=_FONT_SIZE_FIGURE_TITLE, fontweight=_FONT_WEIGHT_FIGURE_TITLE)

    return figure, axes


def plot_spectrum(mc: _ct.tmc, dpi: int) -> _ct.tplot:

    evalues, evectors = _npl.eig(mc.p)
    evalues = evalues.astype(complex)
    evectors = evectors.astype(complex)

    evalues_final = _np.copy(evalues)
    argmin_index = _np.argmin(_np.abs(evalues_final - 1.0))

    if _np.isclose(evalues_final[argmin_index], 1.0):
        evalues_final[argmin_index] = 1.0 + 0.0j
    else:  # pragma: no cover
        evalues_final = _np.append(evalues_final, 1.0 + 0.0j)

    evalues_final = _np.unique(evalues_final)

    evectors_indices = _np.argsort(-_np.abs(evalues), kind='stable')
    evectors_indices = evectors_indices[evectors_indices != argmin_index]
    evectors_indices = _np.insert(evectors_indices, 0, argmin_index)

    evectors = evectors[:, evectors_indices]

    for i in range(evectors.shape[1]):

        evector = evectors[:, i]
        index = _np.argmax(_np.abs(evector))
        phase = _np.angle(evector[index])

        evectors[:, i] *= _np.exp(-1.0j * phase)

    evectors_components = _np.real(evectors)
    mu = _n.slem(mc.p) if mc.is_irreducible else None

    theta = _np.linspace(0.0, 2.0 * _np.pi, 200)
    x_unit_circle = _np.cos(theta)
    y_unit_circle = _np.sin(theta)
    spectral_gap = (mu is not None) and (not _np.isclose(mu, 1.0))
    slem_circle = (mu is not None) and (not _np.isclose(mu, 0.0))

    if spectral_gap:
        cs = _np.linspace(-1.1, 1.1, 201)
        x_spectral_gap, y_spectral_gap = _np.meshgrid(cs, cs)
        z_spectral_gap = x_spectral_gap ** 2.0 + y_spectral_gap ** 2.0
        levels = [mu ** 2.0, 1.0]
    else:
        x_spectral_gap, y_spectral_gap, z_spectral_gap, levels = None, None, None, None

    if slem_circle:
        x_slem_circle = mu * x_unit_circle
        y_slem_circle = mu * y_unit_circle
    else:
        x_slem_circle, y_slem_circle = None, None

    x_evalues, y_evalues = _np.real(evalues_final), _np.imag(evalues_final)
    ticks = _np.linspace(-1.0, 1.0, 9)
    labels_x = [str(i + 1) for i in range(mc.size)]
    labels_y = mc.states
    color_map = _get_colormap(signed=True)

    figure, axes = _mplp.subplots(ncols=2, figsize=(_FIGURE_WIDTH * 1.5, _FIGURE_HEIGHT), dpi=dpi)
    axes = list(axes.flat)
    ax_evalues, ax_evectors = axes
    handles, labels = [], []

    if spectral_gap:
        ax_evalues.contourf(x_spectral_gap, y_spectral_gap, z_spectral_gap, levels=levels, colors=_DEFAULT_COLOR_REFERENCE, alpha=0.2)
        handles.append(_mplpc.Patch(facecolor=_DEFAULT_COLOR_REFERENCE, alpha=0.2))
        labels.append('Spectral Gap')

    if slem_circle:
        ax_evalues.plot(x_slem_circle, y_slem_circle, color=_DEFAULT_COLOR_REFERENCE, linestyle='--', linewidth=_LINE_WIDTH_REFERENCE)

    ax_evalues.plot(x_unit_circle, y_unit_circle, color=_DEFAULT_COLOR_REFERENCE, linestyle='-', linewidth=_LINE_WIDTH_EMPHASIZED)

    h, = ax_evalues.plot(x_evalues, y_evalues, color=_DEFAULT_COLOR_DATA, linestyle='None', marker='*', markersize=12.5)
    handles.append(h)
    labels.append('Eigenvalues')

    ax_evalues.set_xlim(-1.1, 1.1)
    ax_evalues.set_xticks(ticks, minor=False)
    ax_evalues.xaxis.set_major_formatter(_mplt.FormatStrFormatter('%g'))
    ax_evalues.set_ylim(-1.1, 1.1)
    ax_evalues.set_yticks(ticks, minor=False)
    ax_evalues.yaxis.set_major_formatter(_mplt.FormatStrFormatter('%g'))
    ax_evalues.set_aspect('equal')

    ax_evalues.grid(which='major', color=_DEFAULT_COLOR_GRID, linewidth=_GRID_LINE_WIDTH, alpha=_GRID_ALPHA)

    ax_evalues.legend(handles[::-1], labels[::-1], ncol=len(handles), loc='upper center', bbox_to_anchor=(0.5, -0.1), bbox_transform=ax_evalues.transAxes, fontsize=_FONT_SIZE_LEGEND)

    ax_evalues.set_title('Eigenvalues', fontsize=_FONT_SIZE_AXES_TITLE, fontweight=_FONT_WEIGHT_AXES_TITLE, pad=_PAD_AXES_TITLE)

    ax_is = ax_evectors.imshow(evectors_components, cmap=color_map, vmin=-1.0, vmax=1.0, aspect='equal', interpolation='none')

    _xticks_labels(ax_evectors, None, labels_x, True)
    _yticks_labels(ax_evectors, None, labels_y)

    _draw_matrix_grid(ax_evectors)

    _draw_colorbar(figure, ax_is, ax_evectors, 'vertical', signed=True)

    ax_evectors.set_title('Eigenvectors By State', fontsize=_FONT_SIZE_AXES_TITLE, fontweight=_FONT_WEIGHT_AXES_TITLE, pad=_PAD_AXES_TITLE)

    figure.suptitle('Spectrum Plot', fontsize=_FONT_SIZE_FIGURE_TITLE, fontweight=_FONT_WEIGHT_FIGURE_TITLE)

    figure.subplots_adjust(bottom=_LEGEND_MARGIN)

    return figure, axes


def plot_trellis(hmm: _ct.thmm, steps: int, initial_state: _ct.ostate, seed: _ct.oint, force_standard: bool, dpi: int) -> _ct.tplot:

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

    _, prediction_states = prediction

    extended_graph = (not force_standard) and _is_graphviz_available('neato')
    titles = ('Backward Trellis', 'Forward Trellis')

    n, f = hmm.n, len(symbols)
    horizontal = f <= hmm.n * 1.25
    rows, columns = (1, 2) if horizontal else (2, 1)

    if extended_graph:

        trellises = [
            _generate_trellis_extended(hmm, initial_distribution, symbols, False, backward, prediction_states),
            _generate_trellis_extended(hmm, initial_distribution, symbols, True, forward, prediction_states)
        ]

        figure_width = max(_FIGURE_WIDTH, min(_FIGURE_WIDTH * 3.0, max(_FIGURE_WIDTH * 1.25, f * 1.1)))
        figure_height = max(_FIGURE_HEIGHT, min(_FIGURE_HEIGHT * 2.5, (n * 0.9) + 1.0))

        if horizontal:
            figure_width = min(_FIGURE_WIDTH * 3.0, figure_width * 2.0)

        figure, axes = _mplp.subplots(nrows=rows, ncols=columns, figsize=(figure_width, figure_height), tight_layout=True, dpi=dpi)
        axes = list(axes.flat)

        for ax, title in zip(axes, titles):
            ax.axis('off')
            ax.set_title(title, fontsize=_FONT_SIZE_AXES_TITLE, fontweight=_FONT_WEIGHT_AXES_TITLE, pad=_PAD_AXES_TITLE)

        figure.suptitle('Trellis Plot', fontsize=_FONT_SIZE_FIGURE_TITLE, fontweight=_FONT_WEIGHT_FIGURE_TITLE)

        figure.canvas.draw()

        callbacks = []

        for ax, trellis in zip(axes, trellises):
            _draw_graphviz_image(trellis, ax, dpi, callbacks=callbacks)

        figure.canvas.draw()

        for register_callback in callbacks:
            register_callback()

    else:

        trellises = [
            _generate_trellis_standard(hmm, initial_distribution, symbols, False, backward, prediction_states),
            _generate_trellis_standard(hmm, initial_distribution, symbols, True, forward, prediction_states)
        ]

        headers = [r"$\mathregular{T_0}$"] + [hmm.symbols[symbol] for symbol in symbols[1:]]
        labels = hmm.states

        label_offset = (_mt.sqrt(_NODE_SIZE) / 2.0) + _PAD_LABEL
        title_offset = label_offset + _FONT_SIZE_GRAPH_LABEL + _PAD_AXES_TITLE

        x_limits = (0.5, f + 0.5)
        y_limits = (0.0, n + 0.5)

        figure_width = max(_FIGURE_WIDTH, min(_FIGURE_WIDTH * 3.0, f * 0.65))
        figure_height = max(_FIGURE_HEIGHT, min(_FIGURE_HEIGHT * 2.5, (n * 0.9) + 1.0))

        if horizontal:
            figure_width = min(_FIGURE_WIDTH * 3.0, figure_width * 2.0)

        figure, axes = _mplp.subplots(nrows=rows, ncols=columns, figsize=(figure_width, figure_height), constrained_layout=True, dpi=dpi)
        axes = list(axes.flat)

        for ax, title, (trellis, node_colors, node_edges, node_labels, node_positions, edge_colors) in zip(axes, titles, trellises):

            _nx.draw_networkx(
                trellis, node_positions, ax=ax,
                node_color=node_colors, edgecolors=node_edges, edge_color=edge_colors,
                node_size=_NODE_SIZE,
                labels=node_labels, font_size=_FONT_SIZE_GRAPH_LABEL, font_weight=_FONT_WEIGHT_AXES_LABEL
            )

            for col, header in enumerate(headers):
                ax.annotate(
                    header, (col + 1.0, n),
                    xytext=(0.0, label_offset), textcoords='offset points',
                    fontsize=_FONT_SIZE_GRAPH_LABEL, fontweight=_FONT_WEIGHT_AXES_LABEL,
                    horizontalalignment='center', verticalalignment='bottom',
                    annotation_clip=False
                )

            for row, label in enumerate(labels):
                ax.annotate(
                    label, (1.0, n - row),
                    xytext=(-label_offset, 0.0), textcoords='offset points',
                    fontsize=_FONT_SIZE_GRAPH_LABEL, fontweight=_FONT_WEIGHT_AXES_LABEL,
                    horizontalalignment='right', verticalalignment='center',
                    annotation_clip=False
                )

            ax.set_xlim(*x_limits)
            ax.set_ylim(*y_limits)
            ax.axis('off')

            for collection in ax.collections:
                collection.set_clip_on(False)

            ax.set_title(
                title, y=float(n),
                fontsize=_FONT_SIZE_AXES_TITLE, fontweight=_FONT_WEIGHT_AXES_TITLE,
                verticalalignment='bottom', pad=0.0
            )

            title_transform = _mpltr.offset_copy(ax.get_yaxis_transform(), fig=figure, y=title_offset, units='points')
            ax.title.set_transform(title_transform)

        figure.suptitle('Trellis Plot', fontsize=_FONT_SIZE_FIGURE_TITLE, fontweight=_FONT_WEIGHT_FIGURE_TITLE)

        figure.get_layout_engine().set(h_pad=_PAD_AXES_TITLE / 72.0)

    return figure, axes
