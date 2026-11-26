# -*- coding: utf-8 -*-

from __future__ import annotations

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
_DEFAULT_COLOR_PATH = _COLORS[0]
_DEFAULT_COLOR_SYMBOL = _COLOR_GRAY

_DEFAULT_FONT_SIZE_AXES_LABEL = 12.0
_DEFAULT_FONT_SIZE_AXES_TITLE = 13.0
_DEFAULT_FONT_SIZE_FIGURE_TITLE = 15.0

_DEFAULT_FONT_WEIGHT_AXES_LABEL = 'normal'
_DEFAULT_FONT_WEIGHT_AXES_TITLE = 'normal'
_DEFAULT_FONT_WEIGHT_FIGURE_TITLE = 'bold'

_DEFAULT_LINE_WIDTH = 1.75
_DEFAULT_LINE_WIDTH_REFERENCE = 1.50
_DEFAULT_LINE_WIDTH_EMPHASIZED = 3.00

_DEFAULT_GRID_ALPHA = 0.5
_DEFAULT_GRID_LINE_WIDTH = 0.5

_DEFAULT_MARKER_SIZE = 4.0
_DEFAULT_SCATTER_SIZE = 12.0

_DEFAULT_NODE_SIZE = 600

_MAXIMUM_TICKS = 11


#############
# FUNCTIONS #
#############

# noinspection PyTypeChecker
def _decode_graphviz_image(graph: _pyd.Dot, dpi: int) -> tuple:

    with _io.BytesIO(graph.create(format='png')) as buffer:
        img = _mpli.imread(buffer)

    img_x = img.shape[0] / dpi
    img_xi = img_x * 1.1
    img_xo = ((img_xi - img_x) / 2.0) * dpi

    img_y = img.shape[1] / dpi
    img_yi = img_y * 1.1
    img_yo = ((img_yi - img_y) / 2.0) * dpi

    return img, img_x, img_xo, img_y, img_yo


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
            'ec': (1.0, 1.0, 1.0),
            'fc': (1.0, 1.0, 1.0)
        }

        ax.text(
            x, y, label,
            color='k', family='sans-serif', size=10, weight='normal',
            horizontalalignment='center', verticalalignment='center',
            bbox=bbox, clip_on=True, rotation=rotation, transform=transform, zorder=1
        )

    ax.tick_params(
        axis='both', which='both',
        bottom=False, left=False,
        labelbottom=False, labelleft=False
    )


def _draw_graphviz_image(graph: _pyd.Dot, ax: _mplp.Axes, dpi: int, initial_dpi: _ct.oint = None, callbacks: _ct.olist_cal = None) -> _ct.tarray:

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
        target_graph.set('dpi', str(target_dpi))

        # noinspection PyBroadException
        try:
            new_image, _, _, _, _ = _decode_graphviz_image(target_graph, target_dpi)
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
    graph.set('dpi', str(render_dpi))

    image, _, _, _, _ = _decode_graphviz_image(graph, render_dpi)

    if (initial_bounds.width > 0.0) and (initial_bounds.height > 0.0):

        initial_scale = min(initial_bounds.width / image.shape[1], initial_bounds.height / image.shape[0])
        initial_target_dpi = max(1, int(_mt.floor(render_dpi * initial_scale * 0.98)))

        if initial_target_dpi != render_dpi:
            graph.set('dpi', str(initial_target_dpi))
            image, _, _, _, _ = _decode_graphviz_image(graph, initial_target_dpi)
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

    return image


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

        g_sub1.add_node(_pyd.Node(state, **node_attributes))

    for symbol in hmm.symbols:

        node_attributes = {}

        if nodes_color:
            node_attributes['style'] = 'filled'
            node_attributes['fillcolor'] = _DEFAULT_COLOR_SYMBOL

        if nodes_shape:
            node_attributes['shape'] = 'hexagon'

        g_sub2.add_node(_pyd.Node(symbol, **node_attributes))

    for i in range(hmm.n):

        state_i = hmm.states[i]

        for j in range(hmm.n):

            tp = hmm.p[i, j]

            if tp > 0.0:

                state_j = hmm.states[j]

                edge_attributes = {
                    'style': 'filled',
                    'color': _DEFAULT_COLOR_EDGE
                }

                if edges_label:
                    edge_attributes['label'] = f' {round(tp, magnitude):.{magnitude}f} '
                    edge_attributes['fontsize'] = 9

                g.add_edge(_pyd.Edge(state_i, state_j, **edge_attributes))

        for j in range(hmm.k):

            ep = hmm.e[i, j]

            if ep > 0.0:

                symbol = hmm.symbols[j]

                edge_attributes = {
                    'style': 'dashed',
                    'color': edge_colors[i] if nodes_color else _DEFAULT_COLOR_EDGE
                }

                if edges_label:
                    edge_attributes['label'] = f' {round(ep, magnitude):.{magnitude}f} '
                    edge_attributes['fontsize'] = 9

                g.add_edge(_pyd.Edge(state_i, symbol, **edge_attributes))

    return g


def _generate_hmm_standard(hmm: _ct.thmm, nodes_color: bool, nodes_shape: bool, edges_label: bool) -> tuple:

    magnitude = _n.calculate_magnitude(hmm.p, hmm.e)
    node_colors = _get_colors(hmm.n) if nodes_color else []
    edge_colors = list(node_colors) if nodes_color else []

    g = hmm.to_graph()
    positions = _nx.multipartite_layout(g, align='horizontal', subset_key='layer')

    node_data = []

    for i, node in enumerate(g.nodes):

        if nodes_color:
            if node in hmm.states:
                node_color = node_colors[i]
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

        g.add_node(_pyd.Node(state_i, **node_attributes))

        for j in range(mc.size):

            tp = mc.p[i, j]

            if tp > 0.0:

                state_j = mc.states[j]

                edge_attributes = {
                    'style': 'filled',
                    'color': _DEFAULT_COLOR_EDGE
                }

                if edges_label:
                    edge_attributes['label'] = f' {round(tp, magnitude):.{magnitude}f} '
                    edge_attributes['fontsize'] = 9

                g.add_edge(_pyd.Edge(state_i, state_j, **edge_attributes))

    return g


def _generate_mc_standard(mc: _ct.tmc, nodes_color: bool, nodes_shape: bool, edges_label: bool) -> tuple:

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
    hmm: _ct.thmm, initial_distribution: _ct.tarray, symbols: _ct.tarray,
    forward: bool, matrix: _ct.tarray,
    prediction_states: _ct.tlist_int
) -> _pyd.Dot:

    n, f = hmm.n, len(symbols)

    g = _pyd.Dot(
        graph_type='digraph', layout='neato',
        overlap='true', splines='line', outputorder='edgesfirst', margin='0',
        fontname='Arial'
    )

    for row in range(n):
        g.add_node(_pyd.Node(
            f'state{row}',
            label=hmm.states[row], shape='plaintext',
            pos=f'0.4,{n - row}!'
        ))

    for col in range(f):

        path_current = prediction_states[col]
        col_position = (col + 1) * 1.1

        header_label = "<T<FONT POINT-SIZE='8'><SUB>0</SUB></FONT>>" if col == 0 else hmm.symbols[symbols[col]]

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
                        col + 1] == row_next else _COLOR_BLACK

                    g.add_edge(_pyd.Edge(edge_from, edge_to, color=edge_color))

    return g


def _generate_trellis_standard(
    hmm: _ct.thmm, initial_distribution: _ct.tarray, symbols: _ct.tarray,
    forward: bool, matrix: _ct.tarray,
    prediction_states: _ct.tlist_int
) -> tuple:

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

            node_edges.append(_COLOR_BLACK)
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
                        edge_color = _COLOR_BLACK

                    trellis.add_edge(*edge, color=edge_color)

    for row in range(n):
        trellis.add_node(node_index)

        node_colors.append('none')
        node_edges.append('none')
        node_labels[node_index] = hmm.states[row]
        node_positions[node_index] = (0.6, float(n - row))

        node_index += 1

    headers = [r"$\mathregular{T_0}$"] + [hmm.symbols[symbol] for symbol in symbols[1:]]

    for col, header in enumerate(headers):
        trellis.add_node(node_index)

        node_colors.append('none')
        node_edges.append('none')
        node_labels[node_index] = header
        node_positions[node_index] = (col + 1.0, n + 0.35)

        node_index += 1

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


def _get_polygons_bars(d: _ct.tarray, bb: _ct.tarray, bt: _ct.tarray, colors: _ct.tlist_str) -> _ct.tlist_any:

    i, j = d.shape
    w = j / 40.0

    polygons = []

    for oj in range(j):

        xm = oj - w
        xp = oj + w

        for oi in range(i):

            yb = bb[oi, oj]
            yt = bt[oi, oj]

            x = [xm, xp, xp, xm]
            y = [yb, yb, yt, yt]

            polygons.append(_mplpc.Polygon(list(zip(x, y)), edgecolor=None, facecolor=colors[oi], alpha=0.8))

    return polygons


def _get_polygons_flows(p: _ct.tarray, d: _ct.tarray, bb: _ct.tarray, colors: _ct.tlist_str, interval: int) -> _ct.tlist_any:

    i, j = d.shape
    w = j / 40.0
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
                polygons.append(_mplpc.Polygon(list(zip(x[:, z], y[:, z])), edgecolor=None, facecolor=colors[oi], alpha=0.3))

    return polygons


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


def _xticks_labels(ax: _mplp.Axes, size: int, labels_name: _ct.ostr, labels: _ct.tlist_str, minor_major: bool) -> None:

    if labels_name is not None:
        ax.set_xlabel(labels_name, fontsize=13.0)

    if minor_major:
        ax.set_xticks(_np.arange(0.0, size, 1.0), minor=False)
        ax.set_xticks(_np.arange(-0.5, size, 1.0), minor=True)
    else:
        ax.set_xticks(_np.arange(0.0, size, 1.0))

    ax.set_xticklabels(labels)


def _xticks_steps(ax: _mplp.Axes, length: int, interval: int = 1) -> None:

    ax.set_xlabel('Steps', fontsize=13.0)

    tick_step = 1

    if length > _MAXIMUM_TICKS:

        target_step = (length - 1) / (_MAXIMUM_TICKS - 1)
        magnitude = 10.0 ** _mt.floor(_mt.log10(target_step))

        for multiplier in (1, 2, 5, 10):

            tick_step = int(multiplier * magnitude)

            if tick_step >= target_step:
                break

    ticks = _np.arange(0, length, tick_step)

    ax.set_xticks(ticks, minor=False)
    ax.set_xticks(_np.arange(-0.5, length, 1.0), minor=True)
    ax.set_xticklabels(ticks * interval)
    ax.set_xlim(-0.5, length - 0.5)


def _yticks_frequency(ax: _mplp.Axes, bottom: float, top: float) -> None:

    ax.set_ylabel('Frequency', fontsize=13.0)
    ax.set_yticks(_np.linspace(0.0, 1.0, 11))
    ax.set_ylim(bottom, top)


def _yticks_labels(ax: _mplp.Axes, size: int, labels_name: _ct.ostr, labels: _ct.tlist_str) -> None:

    if labels_name is not None:
        ax.set_ylabel(labels_name, fontsize=13.0)

    ax.set_yticks(_np.arange(0.0, size, 1.0), minor=False)
    ax.set_yticks(_np.arange(-0.5, size, 1.0), minor=True)
    ax.set_yticklabels(labels)


def plot_comparison(models: _ct.tlist_mdl, underlying_matrices: str, names: _ct.tlist_str, dpi: int) -> _ct.tplot:

    space = len(models)
    rows = int(_mt.sqrt(space))
    columns = int(_mt.ceil(space / float(rows)))

    figure, axes = _mplp.subplots(nrows=rows, ncols=columns, constrained_layout=True, dpi=dpi)
    axes = list(axes.flat)
    ax_is = None

    color_map = _mplcr.LinearSegmentedColormap.from_list('ColorMap', [_COLOR_WHITE, _COLORS[0]], 20)

    for ax, model, name in zip(axes, models, names):

        matrix = model.e if underlying_matrices == 'emission' else model.p

        ax_is = ax.imshow(matrix, aspect='auto', cmap=color_map, interpolation='none', vmin=0.0, vmax=1.0)
        ax.set_title(name, fontsize=12.0, fontweight='normal', pad=1)

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


def plot_convergence(mc: _ct.tmc, steps: int, initial_status: _ct.oarray, dpi: int, full_scale: bool) -> _ct.tplot:

    stationary_distribution = mc.pi[0]
    distributions = mc.propagate(steps, initial_status=initial_status, output_last=False)
    distributions = _np.asarray(distributions, dtype=float)
    distances = 0.5 * _np.sum(_np.abs(distributions - stationary_distribution), axis=1)

    figure_width, figure_height = _mplp.rcParams['figure.figsize']
    figure, ax = _mplp.subplots(figsize=(figure_width, figure_height), dpi=dpi)

    ax.plot(_np.arange(distances.size), distances, color=_COLORS[0], linewidth=1.75)

    _xticks_steps(ax, distances.size)
    ax.set_ylabel('Total Variation Distance', fontsize=13.0)

    if full_scale:
        ax.set_ylim(0.0, 1.0)
    else:
        distance_max = _np.max(distances)
        ax.set_ylim(0.0, min(1.0, distance_max * 1.1) if distance_max > 0.0 else 1.0)

    ax.grid()

    figure.suptitle('Convergence Plot', fontsize=15.0, fontweight='bold')

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

    colors_states = _get_colors(hmm.n)
    colors_symbols = _mplp.get_cmap('Greys')(_np.linspace(0.25, 0.75, hmm.k))

    color_map_states = _mplcr.ListedColormap(colors_states)
    color_map_symbols = _mplcr.ListedColormap(colors_symbols)
    color_map_disagreement = _mplcr.ListedColormap([_COLOR_WHITE, _COLORS[6]])

    figure_width, figure_height = _mplp.rcParams['figure.figsize']
    figure, axes = _mplp.subplots(figsize=(figure_width * 1.5, figure_height), nrows=4, tight_layout=True, sharex=True, dpi=dpi)
    axes = list(axes.flat)

    sequences = [
        ('Observed', symbols, hmm.symbols, color_map_symbols),
        ('Viterbi', viterbi, hmm.states, color_map_states),
        ('MAP', map_states, hmm.states, color_map_states),
        ('Disagreement', disagreement, None, color_map_disagreement)
    ]

    for ax, (name, sequence, labels, color_map) in zip(axes, sequences):

        matrix = _np.asarray(sequence, dtype=int).reshape(1, -1)
        colors_count = color_map.N

        ax.imshow(matrix, aspect='auto', cmap=color_map, interpolation='none', vmin=-0.5, vmax=colors_count - 0.5)

        ax.set_ylabel(name, fontsize=12.0, rotation=0, labelpad=10.0, horizontalalignment='right', verticalalignment='center')
        ax.set_yticks([])

        if size <= _MAXIMUM_TICKS:

            ax.grid(which='minor', color='k', linewidth=0.5)

            if labels is not None:

                for i, value in enumerate(sequence):

                    color = color_map(value)
                    luminance = (0.299 * color[0]) + (0.587 * color[1]) + (0.114 * color[2])
                    text_color = _COLOR_BLACK if luminance > 0.5 else _COLOR_WHITE

                    ax.text(i, 0.0, labels[value], color=text_color, horizontalalignment='center', verticalalignment='center')

        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    _xticks_steps(axes[-1], size)
    axes[-1].tick_params(axis='x', which='major', bottom=True, labelbottom=True)
    axes[-1].set_xlabel('Observations', fontsize=13.0)

    if size > _MAXIMUM_TICKS:
        legend_handlers_symbols = [_mplpc.Patch(facecolor=color_map_symbols(i), label=label) for i, label in enumerate(hmm.symbols)]
        legend_handlers_states = [_mplpc.Patch(facecolor=color_map_states(i), label=label) for i, label in enumerate(hmm.states)]
        axes[0].legend(handles=legend_handlers_symbols, loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=int(_mt.ceil(hmm.k / 4.0)))
        axes[1].legend(handles=legend_handlers_states, loc='center left', bbox_to_anchor=(1.02, 0.5), ncol=int(_mt.ceil(hmm.n / 4.0)))

    figure.align_ylabels(axes)
    figure.suptitle('Decoding Plot', fontsize=15.0, fontweight='bold')

    return figure, axes


def plot_flow(mc: _ct.tmc, steps: int, interval: int, initial_status: _ct.oarray, dpi: int) -> _ct.tplot:

    distributions = mc.propagate(steps * interval, initial_status=initial_status, output_last=False)
    distributions = _np.transpose(_np.asarray(distributions, dtype=float)[::interval, :])

    p = mc.p
    d = distributions * 0.9

    boundaries_bottom, boundaries_top = _get_boundaries(d)
    boundaries_middle = (boundaries_bottom + boundaries_top) / 2.0

    colors = _get_colors(mc.size)

    legend_handlers = []
    legend_labels = mc.states

    for i, label in enumerate(legend_labels):
        legend_handlers.append(_mplpc.Patch(color=colors[i], label=label))

    polygons_bars = _get_polygons_bars(d, boundaries_bottom, boundaries_top, colors)
    polygons_flows = _get_polygons_flows(p, d, boundaries_bottom, colors, interval)

    figure_width, figure_height = _mplp.rcParams['figure.figsize']
    figure, ax = _mplp.subplots(figsize=(figure_width, figure_height), dpi=dpi)

    for e in polygons_flows:
        ax.add_patch(e)

    for e in polygons_bars:
        ax.add_patch(e)

    for ai in range(distributions.shape[0]):
        for aj in range(distributions.shape[1]):

            dv = distributions[ai, aj]

            if dv > 0.05:
                ax.text(aj, float(boundaries_middle[ai, aj]), f'{dv:.3f}', horizontalalignment='center', verticalalignment='center')

    _xticks_steps(ax, steps + 1, interval)

    ax.set_ylim(0.0, 1.0)
    ax.invert_yaxis()

    figure.legend(legend_handlers, legend_labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=len(legend_handlers))
    figure.suptitle('Flow Plot', fontsize=15.0, fontweight='bold')
    figure.subplots_adjust(bottom=0.2)

    return figure, ax


def plot_graph(model: _ct.tmdl, nodes_color: bool, nodes_shape: bool, edges_label: bool, force_standard: bool, dpi: int) -> _ct.tplot:

    model_mc = model.__class__.__name__ == 'MarkovChain'
    extended_graph = (not force_standard) and _is_graphviz_available()

    if extended_graph:

        if model_mc:
            graph = _generate_mc_extended(model, nodes_color, nodes_shape, edges_label)
        else:
            graph = _generate_hmm_extended(model, nodes_color, nodes_shape, edges_label)

        graph.set('dpi', str(dpi))

        preview, img_height, _, img_width, _ = _decode_graphviz_image(graph, dpi)

        figure, ax = _mplp.subplots(figsize=(img_width * 1.5, img_height * 1.5), dpi=dpi)
        figure.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.90)

        ax.axis('off')
        figure.canvas.draw()

        bounds = ax.get_window_extent()
        initial_dpi = max(1,
                          int(_mt.floor(dpi * min(bounds.width / preview.shape[1], bounds.height / preview.shape[0]))))
        _draw_graphviz_image(graph, ax, dpi, initial_dpi=initial_dpi)

    else:

        if model_mc:
            graph_data = _generate_mc_standard(model, nodes_color, nodes_shape, edges_label)
        else:
            graph_data = _generate_hmm_standard(model, nodes_color, nodes_shape, edges_label)

        graph, positions, node_data, edge_data, edge_labels_straight, edge_labels_symbol, edge_labels_curved = graph_data

        figure_width, figure_height = _mplp.rcParams['figure.figsize']
        figure, ax = _mplp.subplots(figsize=(figure_width, figure_height), dpi=dpi)

        for node, node_color, node_shape in node_data:
            _nx.draw_networkx_nodes(
                graph, positions, ax=ax,
                nodelist=[node], node_color=node_color, node_shape=node_shape, node_size=_DEFAULT_NODE_SIZE,
                edgecolors=_COLOR_BLACK
            )

        _nx.draw_networkx_labels(graph, positions, ax=ax)

        for edge, edge_color, edge_attributes in edge_data:
            _nx.draw_networkx_edges(
                graph, positions, ax=ax,
                edgelist=[edge], edge_color=edge_color, **edge_attributes
            )

        if len(edge_labels_straight) > 0:
            _nx.draw_networkx_edge_labels(
                graph, positions, ax=ax,
                edge_labels=edge_labels_straight
            )

        if len(edge_labels_symbol) > 0:
            _nx.draw_networkx_edge_labels(
                graph, positions, ax=ax,
                edge_labels=edge_labels_symbol,
                label_pos=0.25
            )

        if len(edge_labels_curved) > 0:
            _draw_edge_labels_curved(ax, positions, edge_labels_curved)

        ax.set_aspect('equal')
        _fit_graph_bounds(ax)

    figure.suptitle('Graph Plot', fontsize=15.0, fontweight='bold')

    return figure, ax


def plot_matrix(model: _ct.tmdl, underlying_matrix: str, dpi: int) -> _ct.tplot:

    matrix = model.e if underlying_matrix == 'emission' else model.p
    labels_x = model.symbols if underlying_matrix == 'emission' else model.states

    figure_width, figure_height = _mplp.rcParams['figure.figsize']
    figure, ax = _mplp.subplots(figsize=(figure_width, figure_height), dpi=dpi)

    color_map = _mplcr.LinearSegmentedColormap.from_list('ColorMap', [_COLOR_WHITE, _COLORS[0]], 20)
    ax_is = ax.imshow(matrix, aspect='auto', cmap=color_map, interpolation='none', vmin=0.0, vmax=1.0)

    _xticks_labels(ax, len(labels_x), 'Symbols' if underlying_matrix == 'emission' else 'States', labels_x, True)
    _yticks_labels(ax, len(model.states), 'States', model.states)

    ax.grid(which='minor', color='k')

    cb = figure.colorbar(ax_is, drawedges=True, orientation='horizontal', ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
    cb.ax.set_xticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])

    figure.suptitle('Matrix Plot (' + underlying_matrix.capitalize() + ')', fontsize=15.0, fontweight='bold')

    return figure, ax


def plot_posterior(plot_type: str, hmm: _ct.thmm, symbols: _ct.tlist_int, initial_distribution: _ct.tarray, dpi: int) -> _ct.tplot:

    decoding = hmm.decode(symbols, initial_status=initial_distribution)

    if decoding is None:  # pragma: no cover
        raise ValueError('The computation of the posterior probabilities failed.')

    _, posterior, _, _, forward, _ = decoding

    size = len(symbols)
    figure_width, figure_height = _mplp.rcParams['figure.figsize']

    if plot_type == 'heatmap':

        figure, axes = _mplp.subplots(figsize=(figure_width, figure_height), nrows=2, constrained_layout=True, dpi=dpi)
        axes = list(axes.flat)

        color_map = _mplcr.LinearSegmentedColormap.from_list('ColorMap', [_COLOR_WHITE, _COLORS[0]], 20)
        is_axes = []

        for ax, (name, matrix) in zip(axes, [('Filtering', forward), ('Smoothing', posterior)]):

            ax_is = ax.imshow(matrix, aspect='auto', cmap=color_map, interpolation='none', vmin=0.0, vmax=1.0)
            is_axes.append(ax_is)

            _xticks_steps(ax, size)
            _yticks_labels(ax, hmm.n, 'States', hmm.states)

            if size <= _MAXIMUM_TICKS:
                ax.grid(which='minor', color='k')
            else:
                ax.tick_params(axis='x', which='minor', bottom=False)
                ax.grid(axis='y', which='minor', color='k')

            ax.set_title(name, fontsize=12.0, fontweight='normal')

        color_map_ax, color_map_ax_kwargs = _mplcb.make_axes(axes, drawedges=True, orientation='vertical', ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
        figure.colorbar(is_axes[0], cax=color_map_ax, **color_map_ax_kwargs)

    else:

        figure, ax = _mplp.subplots(figsize=(figure_width, figure_height), dpi=dpi)

        colors = _get_colors(hmm.n)
        x = _np.arange(size)

        for i in range(hmm.n):

            overlaps = [j for j in range(hmm.n) if _np.allclose(posterior[i, :], posterior[j, :], rtol=1e-12, atol=1e-14)]
            overlap_index = overlaps.index(i)

            marker = 'o' if size == 2 else None
            line_style = '-'

            if len(overlaps) > 1:  # pragma: no cover
                line_style = ('-', '--', ':', '-.')[overlap_index % 4]
                if overlap_index > 0:
                    marker = ('o', 's', '^', 'D')[((overlap_index - 1) // 4) % 4]

            ax.fill_between(x, 0.0, posterior[i, :], color=colors[i], alpha=0.35)

            ax.plot(
                x, posterior[i, :],
                color=colors[i], label=hmm.states[i],
                linewidth=1.75, linestyle=line_style,
                marker=marker, markersize=4.0, markevery=max(1, size // 8)
            )

        _xticks_steps(ax, size)
        _yticks_frequency(ax, 0.0, 1.0)
        ax.set_ylabel('Probability', fontsize=13.0)

        ax.grid()

        figure.legend(*ax.get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=hmm.n)
        figure.subplots_adjust(bottom=0.2)

        axes = ax

    figure.suptitle(f'Posterior Plot ({plot_type.capitalize()})', fontsize=15.0, fontweight='bold')

    return figure, axes


def plot_redistributions(plot_type: str, mc: _ct.tmc, redistributions: int, initial_status: _ct.ostatus, dpi: int) -> _ct.tplot:

    distributions = mc.propagate(redistributions, initial_status=initial_status, output_last=False)
    distributions = _np.asarray(distributions, dtype=float)
    distributions_length = distributions.shape[0]

    figure_width, figure_height = _mplp.rcParams['figure.figsize']
    figure, ax = _mplp.subplots(figsize=(figure_width, figure_height), dpi=dpi)

    if plot_type == 'heatmap':

        color_map = _mplcr.LinearSegmentedColormap.from_list('ColorMap', [_COLOR_WHITE, _COLORS[0]], 20)
        ax_is = ax.imshow(_np.transpose(distributions), aspect='auto', cmap=color_map, interpolation='none', vmin=0.0, vmax=1.0)

        _xticks_steps(ax, distributions_length)
        _yticks_labels(ax, mc.size, None, mc.states)

        ax.grid(which='minor', color='k')

        cb = figure.colorbar(ax_is, drawedges=True, orientation='horizontal', ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
        cb.ax.set_xticklabels(['0.0', '0.25', '0.5', '0.75', '1.0'])

        figure.suptitle(f'Redistributions Plot ({plot_type.capitalize()})', fontsize=15.0, fontweight='bold')

    else:

        ax.set_prop_cycle('color', _get_colors(mc.size))

        for i in range(mc.size):

            overlaps = [j for j in range(mc.size) if _np.allclose(distributions[:, i], distributions[:, j], rtol=1e-12, atol=1e-14)]
            overlap_index = overlaps.index(i)

            marker = 'o' if distributions_length == 2 else None
            line_style = '-'

            if len(overlaps) > 1:  # pragma: no cover
                line_style = ('-', '--', ':', '-.')[overlap_index % 4]
                if overlap_index > 0:
                    marker = ('o', 's', '^', 'D')[((overlap_index - 1) // 4) % 4]

            ax.plot(
                _np.arange(0.0, distributions_length, 1.0), distributions[:, i],
                label=mc.states[i],
                linewidth=1.75, linestyle=line_style,
                marker=marker, markersize=4.0, markevery=max(1, distributions_length // 8)
            )

        if _np.allclose(distributions[0, :], _np.ones(mc.size, dtype=float) / mc.size):
            ax.plot(0.0, distributions[0, 0], color=_COLOR_BLACK, label="Start", marker='o', markeredgecolor=_COLOR_BLACK, markerfacecolor=_COLOR_BLACK)
            legend_size = mc.size + 1
        else:  # pragma: no cover
            legend_size = mc.size

        _xticks_steps(ax, distributions_length)
        _yticks_frequency(ax, -0.05, 1.05)

        ax.grid()

        figure.legend(*ax.get_legend_handles_labels(), loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=legend_size)
        figure.suptitle(f'Redistributions Plot ({plot_type.capitalize()})', fontsize=15.0, fontweight='bold')
        figure.subplots_adjust(bottom=0.2)

    return figure, ax


def plot_residuals(model: _ct.tmdl, sequence: _ct.tlist_int, seed: _ct.oint, dpi: int) -> _ct.tplot:

    residuals = model.residuals(sequence, seed=seed)
    residuals_sorted = _np.sort(residuals)
    size = residuals.size

    quantiles = _sps.ndtri((_np.arange(1, size + 1) - 0.5) / size)  # pylint: disable=no-member

    figure_width, figure_height = _mplp.rcParams['figure.figsize']
    figure, axes = _mplp.subplots(figsize=(figure_width * 1.5, figure_height * 1.5), nrows=2, ncols=2, tight_layout=True, dpi=dpi)
    axes = list(axes.flat)
    ax_residuals, ax_quantiles, ax_distribution, ax_autocorrelation = axes

    offset = 1 if model.__class__.__name__ == 'MarkovChain' else 0

    ax_residuals.scatter(_np.arange(offset, size + offset), residuals, color=_COLORS[0], s=12.0)
    ax_residuals.axhline(0.0, color=_COLORS[4], linestyle='--')
    ax_residuals.set_xlim(offset - 0.5, size + offset - 0.5)
    ax_residuals.xaxis.set_major_locator(_mplt.MaxNLocator(nbins=_MAXIMUM_TICKS - 1, integer=True, min_n_ticks=1))
    ax_residuals.grid(alpha=0.5)
    ax_residuals.set_title('Residuals', fontsize=12.0, fontweight='normal')

    lower = min(quantiles[0], residuals_sorted[0])
    upper = max(quantiles[-1], residuals_sorted[-1])

    ax_quantiles.scatter(quantiles, residuals_sorted, color=_COLORS[0], s=12.0)
    ax_quantiles.plot([lower, upper], [lower, upper], color=_COLORS[4], linestyle='--')
    ax_quantiles.grid(alpha=0.5)
    ax_quantiles.set_title('Q-Q Plot', fontsize=12.0, fontweight='normal')

    lower = min(-4.0, _np.min(residuals))
    upper = max(4.0, _np.max(residuals))
    x = _np.linspace(lower, upper, 200)
    y = _np.exp(-0.5 * x ** 2.0) / _np.sqrt(2.0 * _np.pi)

    ax_distribution.hist(residuals, bins='auto', density=True, edgecolor=_COLOR_BLACK, facecolor=_COLORS[0])
    ax_distribution.plot(x, y, color=_COLORS[4], linewidth=1.75)
    ax_distribution.grid(alpha=0.5)
    ax_distribution.set_title('Distribution', fontsize=12.0, fontweight='normal')

    maximum_lag = min(20, size - 1)
    centered = residuals - _np.mean(residuals)
    denominator = _np.dot(centered, centered)

    if (maximum_lag > 0) and (denominator > 0.0):

        lags = _np.arange(1, maximum_lag + 1)
        correlations = _np.array([_np.dot(centered[:-lag], centered[lag:]) / denominator for lag in lags])
        threshold = _sps.ndtri(0.975) / _np.sqrt(size)  # pylint: disable=no-member
        limit = min(1.0, max(_np.max(_np.abs(correlations)), threshold) * 1.1)

        ax_autocorrelation.axhspan(-threshold, threshold, color=_COLORS[0], alpha=0.2)
        ax_autocorrelation.vlines(lags, 0.0, correlations, color=_COLORS[0], linewidth=1.75)
        ax_autocorrelation.scatter(lags, correlations, color=_COLORS[0], s=12.0)
        ax_autocorrelation.set_xlim(0.5, maximum_lag + 0.5)
        ax_autocorrelation.set_ylim(-limit, limit)
        ax_autocorrelation.xaxis.set_major_locator(_mplt.MaxNLocator(nbins=_MAXIMUM_TICKS - 1, integer=True, min_n_ticks=1))

    else:
        ax_autocorrelation.text(0.5, 0.5, 'Autocorrelation Unavailable', horizontalalignment='center', verticalalignment='center', transform=ax_autocorrelation.transAxes)
        ax_autocorrelation.set_ylim(-1.0, 1.0)
        ax_autocorrelation.set_xticks([])

    ax_autocorrelation.axhline(0.0, color=_COLORS[4], linestyle='--')
    ax_autocorrelation.grid(alpha=0.5)
    ax_autocorrelation.set_title('Autocorrelation', fontsize=12.0, fontweight='normal')

    figure.suptitle('Residuals Plot', fontsize=15.0, fontweight='bold')

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

    if plot_type == 'heatmap':

        figure, axes = _mplp.subplots(nrows=plots_count, constrained_layout=True, dpi=dpi)
        axes = [axes] if plots_count == 1 else list(axes.flat)

        color_map = _mplcr.LinearSegmentedColormap.from_list('ColorMap', [_COLOR_WHITE, _COLORS[0]], 20)
        is_axes = []

        for a_current, (size, labels_name, labels, sequence) in zip(axes, walks):

            sequence_matrix = _np.zeros((size, size), dtype=float)

            for i in range(1, steps + 1):
                sequence_matrix[sequence[i - 1], sequence[i]] += 1.0

            sequence_matrix /= _np.sum(sequence_matrix)

            a_current_is = a_current.imshow(sequence_matrix, aspect='auto', cmap=color_map, interpolation='none', vmin=0.0, vmax=1.0)
            is_axes.append(a_current_is)

            _xticks_labels(a_current, size, labels_name, labels, True)
            _yticks_labels(a_current, size, labels_name, labels)

            a_current.grid(which='minor', color='k')

        color_map_ax, color_map_ax_kwargs = _mplcb.make_axes(axes, drawedges=True, orientation='vertical', ticks=[0.0, 0.25, 0.5, 0.75, 1.0])
        figure.colorbar(is_axes[0], cax=color_map_ax, **color_map_ax_kwargs)

    elif plot_type == 'histogram':

        figure, axes = _mplp.subplots(nrows=plots_count, tight_layout=True, dpi=dpi)
        axes = [axes] if plots_count == 1 else list(axes.flat)

        for a_current, (size, labels_name, labels, sequence) in zip(axes, walks):

            sequence_histogram = _np.bincount(sequence, minlength=size).astype(float)
            sequence_histogram /= _np.sum(sequence_histogram)

            a_current.bar(_np.arange(0.0, size, 1.0), sequence_histogram, edgecolor=_COLOR_BLACK, facecolor=_COLORS[0])

            _xticks_labels(a_current, size, labels_name, labels, False)
            _yticks_frequency(a_current, 0.0, 1.0)

    else:

        figure, axes = _mplp.subplots(nrows=plots_count, tight_layout=True, dpi=dpi)
        axes = [axes] if plots_count == 1 else list(axes.flat)

        color_map = _mplcr.LinearSegmentedColormap.from_list('ColorMap', [_COLOR_WHITE, _COLORS[0]], 2)

        for a_current, (size, labels_name, labels, sequence) in zip(axes, walks):

            sequence_matrix = _np.zeros((size, steps + 1), dtype=float)

            for index, state in enumerate(sequence):
                sequence_matrix[state, index] = 1.0

            a_current.imshow(sequence_matrix, aspect='auto', cmap=color_map, interpolation='none', vmin=0.0, vmax=1.0)

            _xticks_steps(a_current, steps + 1)
            _yticks_labels(a_current, size, labels_name, labels)

            a_current.grid(which='minor', color='k')

    figure.suptitle(f'Sequence Plot ({plot_type.capitalize()})', fontsize=15.0, fontweight='bold')

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

    theta = _np.linspace(0.0, 2.0 * _np.pi, 200)
    x_unit_circle = _np.cos(theta)
    y_unit_circle = _np.sin(theta)

    figure_width, figure_height = _mplp.rcParams['figure.figsize']
    figure, axes = _mplp.subplots(ncols=2, figsize=(figure_width * 1.5, figure_height), dpi=dpi)
    axes = list(axes.flat)
    ax_evalues, ax_evectors = axes

    handles, labels = [], []

    if mc.is_irreducible:

        mu = _n.slem(mc.p)

        if not _np.isclose(mu, 1.0):

            cs = _np.linspace(-1.1, 1.1, 201)
            x_spectral_gap, y_spectral_gap = _np.meshgrid(cs, cs)
            z_spectral_gap = x_spectral_gap ** 2.0 + y_spectral_gap ** 2.0

            h = ax_evalues.contourf(x_spectral_gap, y_spectral_gap, z_spectral_gap, alpha=0.2, colors=_COLORS[4], levels=[mu ** 2.0, 1.0])
            handles.append(_mplp.Rectangle((0.0, 0.0), 1.0, 1.0, fc=h.get_facecolor()[0]))
            labels.append('Spectral Gap')

        if not _np.isclose(mu, 0.0):

            x_slem_circle = mu * x_unit_circle
            y_slem_circle = mu * y_unit_circle

            ax_evalues.plot(x_slem_circle, y_slem_circle, color=_COLORS[4], linestyle='--', linewidth=1.5)

    ax_evalues.plot(x_unit_circle, y_unit_circle, color=_COLORS[4], linestyle='-', linewidth=3.0)

    h, = ax_evalues.plot(_np.real(evalues_final), _np.imag(evalues_final), color=_COLORS[0], linestyle='None', marker='*', markersize=12.5)
    handles.append(h)
    labels.append('Eigenvalues')

    ax_evalues.set_xlim(-1.1, 1.1)
    ax_evalues.set_ylim(-1.1, 1.1)
    ax_evalues.set_aspect('equal')
    ax_evalues.xaxis.set_major_formatter(_mplt.FormatStrFormatter('%g'))
    ax_evalues.yaxis.set_major_formatter(_mplt.FormatStrFormatter('%g'))
    ax_evalues.set_xticks(_np.linspace(-1.0, 1.0, 9), minor=False)
    ax_evalues.set_yticks(_np.linspace(-1.0, 1.0, 9), minor=False)
    ax_evalues.grid(which='major')
    ax_evalues.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(handles), bbox_transform=ax_evalues.transAxes)
    ax_evalues.set_title('Eigenvalues', fontsize=12.0, fontweight='normal')

    color_map = _mplcr.LinearSegmentedColormap.from_list('ColorMap', [_COLORS[6], _COLOR_WHITE, _COLORS[0]], 21)
    ax_is = ax_evectors.imshow(evectors_components, aspect='equal', cmap=color_map, interpolation='none', vmin=-1.0, vmax=1.0)
    _xticks_labels(ax_evectors, mc.size, None, [str(i + 1) for i in range(mc.size)], True)
    _yticks_labels(ax_evectors, mc.size, 'States', mc.states)
    ax_evectors.grid(which='minor', color='k')
    cb = figure.colorbar(ax_is, ax=ax_evectors, drawedges=True, orientation='vertical', ticks=[-1.0, -0.5, 0.0, 0.5, 1.0])
    cb.ax.set_yticklabels(['-1.0', '-0.5', '0.0', '0.5', '1.0'])
    ax_evectors.set_title('Eigenvectors', fontsize=12.0, fontweight='normal')

    figure.suptitle('Spectrum Plot', fontsize=15.0, fontweight='bold')
    figure.subplots_adjust(bottom=0.2)

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

    if extended_graph:

        figure_width = min(18.0, max(8.0, len(symbols) * 1.1))
        figure, axes = _mplp.subplots(figsize=(figure_width, 7.5), nrows=2, tight_layout=True, dpi=dpi)
        axes = list(axes.flat)

        axes[0].axis('off')
        axes[0].set_title('Backward Trellis', fontsize=15.0, fontweight='normal', pad=1)
        axes[1].axis('off')
        axes[1].set_title('Forward Trellis', fontsize=15.0, fontweight='normal', pad=1)

        figure.canvas.draw()

        callbacks = []

        trellis = _generate_trellis_extended(hmm, initial_distribution, symbols, False, backward, prediction_states)
        _draw_graphviz_image(trellis, axes[0], dpi, callbacks=callbacks)

        trellis = _generate_trellis_extended(hmm, initial_distribution, symbols, True, forward, prediction_states)
        _draw_graphviz_image(trellis, axes[1], dpi, callbacks=callbacks)

        figure.canvas.draw()

        for register_callback in callbacks:
            register_callback()

    else:

        y_top = hmm.n + 0.5

        figure, axes = _mplp.subplots(nrows=2, tight_layout=True, dpi=dpi)
        axes = list(axes.flat)

        trellis, node_colors, node_edges, node_labels, node_positions, edge_colors = _generate_trellis_standard(
            hmm, initial_distribution, symbols,
            False, backward,
            prediction_states
        )

        ax_current = axes[0]

        _nx.draw_networkx(
            trellis, node_positions, ax=ax_current,
            edgecolors=node_edges, edge_color=edge_colors,
            node_color=node_colors, node_size=_DEFAULT_NODE_SIZE,
            labels=node_labels, font_size=9
        )

        ax_current.set_ylim(0.5, y_top)
        ax_current.axis('off')
        ax_current.set_title('Backward Trellis', fontsize=15.0, fontweight='normal', pad=1)

        trellis, node_colors, node_edges, node_labels, node_positions, edge_colors = _generate_trellis_standard(
            hmm, initial_distribution, symbols,
            True, forward,
            prediction_states
        )

        ax_current = axes[1]

        _nx.draw_networkx(
            trellis, node_positions, ax=ax_current,
            edgecolors=node_edges, edge_color=edge_colors,
            node_color=node_colors, node_size=_DEFAULT_NODE_SIZE,
            labels=node_labels, font_size=9
        )

        ax_current.set_ylim(0.5, y_top)
        ax_current.axis('off')
        ax_current.set_title('Forward Trellis', fontsize=15.0, fontweight='normal', pad=1)

    return figure, axes
